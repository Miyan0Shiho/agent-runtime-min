"""The in-process provider: a child agent driven by the same loop as its parent.

Running the child in this process is the simplest provider and the one that
proves the seam works. A container-backed provider would implement the same two
methods.

The isolation is what matters here, and each part of it is deliberate:

* **A fresh session.** The child's events go to its own log, so the parent's
  transcript never contains work it did not do. That is also what makes the
  child's run replayable on its own.
* **A restricted tool registry.** The child gets a layered view, so a subagent
  cannot recursively spawn subagents without a limit, and cannot reach tools the
  parent withheld.
* **Its own sandbox policy and approval service.** A delegated task must not be
  able to widen its own permissions, and an unattended child must not block on a
  prompt nobody will answer.

The child is driven to completion and then disposed. Nothing about it survives
the call except its session log -- which is the point: the caller gets a result,
not a second agent to manage.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..approval.service import ApprovalService
from ..kernel.errors import describe
from ..session.session import Session
from ..tools.registry import ToolRegistry
from .contracts import (
    STOP_ABORTED,
    SubagentError,
    STOP_COMPLETED,
    STOP_ERROR,
    DelegationRequest,
    DelegationResult,
)
from .runtime import (
    RunningSubagent,
    consumed_work,
    final_assistant_output,
    turn_stop_reason,
)

log = logging.getLogger("arm.subagent.inprocess")

#: Capabilities an in-process child can honour. It can do everything the runtime
#: asks of a provider, because it builds a real agent rather than approximating
#: one.
IN_PROCESS_CAPABILITIES = frozenset(
    {"depth_limit", "agent_options", "output_schema", "tool_filter", "persona"}
)


@dataclass
class ChildFactory:
    """Builds a child agent. Injected so the provider does not import the loop.

    The provider needs "an agent that can be driven"; the loop is one way to
    build that and a test double is another. Keeping the construction behind a
    callable is what lets this module be tested without a model.
    """

    build: Callable[
        [Session, ToolRegistry, DelegationRequest], Any
    ]


class InProcessProvider:
    """Runs a child agent in this process."""

    name = "in-process"
    supported = IN_PROCESS_CAPABILITIES

    def __init__(
        self,
        *,
        factory: ChildFactory,
        parent_registry: ToolRegistry,
        approval: ApprovalService | None = None,
        sandbox: Any = None,
        delegation_tool_factory: Callable[[Any, int], Any] | None = None,
    ) -> None:
        self.factory = factory
        self.parent_registry = parent_registry
        self.approval = approval
        self.sandbox = sandbox
        #: Builds the child's own ``subagent`` tool at the child's depth. Without
        #: it a child inherits the parent's tool, whose depth is baked in at 0,
        #: and the depth limit never triggers.
        self.delegation_tool_factory = delegation_tool_factory

    async def start(
        self, request: DelegationRequest, *, run_id: str, parent: Session
    ) -> RunningSubagent:
        """Create the child, drive it, and expose its outcome as a future.

        The child's session is created before anything runs, so a failure while
        building it is reported as a start failure rather than as a run that
        produced no result.
        """
        child_session = Session.create(f"subagent-{run_id[:8]}")
        # The link back to the parent is recorded on the child, so the lineage is
        # inspectable without the runtime keeping a side table.
        child_session.header["parent_session"] = parent.session_id
        child_session.header["delegated"] = True

        # A layered view: the child sees the parent's tools minus anything the
        # request excludes, and the runtime re-registers the subagent tool with a
        # depth-aware definition.
        registry = self._child_registry(request, child_session=child_session)

        try:
            child = self.factory.build(child_session, registry, request)
        except Exception as exc:  # noqa: BLE001 - a build failure is a start failure
            # A SubagentError, not a bare RuntimeError: every other refusal in
            # this module is typed, and a caller that catches SubagentError to
            # handle delegation failures would otherwise miss this one.
            raise SubagentError(
                f"could not build the child agent: {exc}",
                code="SUBAGENT_BUILD_FAILED",
            ) from exc

        loop = asyncio.get_running_loop()
        result: asyncio.Future[DelegationResult] = loop.create_future()
        start_seq = child_session.seq

        async def drive() -> None:
            try:
                await self._drive(
                    child, child_session, request, result, start_seq, run_id
                )
            except asyncio.CancelledError:
                if not result.done():
                    result.set_result(
                        DelegationResult(
                            run_id=run_id,
                            provider=self.name,
                            stop_reason=STOP_ABORTED,
                            error="the subagent was cancelled",
                            child_session=child_session.session_id,
                            events=len(child_session),
                        )
                    )
                raise
            except Exception as exc:  # noqa: BLE001 - a crash is a stop reason
                if not result.done():
                    result.set_result(
                        DelegationResult(
                            run_id=run_id,
                            provider=self.name,
                            stop_reason=STOP_ERROR,
                            error=describe(exc),
                            child_session=child_session.session_id,
                            events=len(child_session),
                        )
                    )

        task = asyncio.ensure_future(drive())

        async def dispose() -> None:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

        return RunningSubagent(
            run_id=run_id,
            provider=self.name,
            child_session=child_session,
            result=result,
            dispose=dispose,
            runtime=child,
        )

    def _child_registry(
        self, request: DelegationRequest, *, child_session: Any = None
    ) -> ToolRegistry:
        """A registry for the child, restricted to the request.

        The restriction is applied to a **separate registry object**, never to the
        parent's.

        This is the difference between a child that sees fewer tools and a parent
        that has lost them. ``ToolRegistry.restrict`` pushes a layer onto the
        registry it is called on, so restricting the parent in place would strip
        the parent's own tools for the rest of the session -- and the damage would
        be invisible until something the parent needed stopped working.

        A separate registry is also what keeps the child's capability set frozen
        at the moment of delegation: a tool registered on the parent afterwards is
        not visible to a child that already started.
        """
        if not request.tools:
            # No filter means "inherit everything the parent has now". The child
            # gets its own registry anyway, so a later parent registration does
            # not leak into a running child.
            child = ToolRegistry()
            child.adopt_from(self.parent_registry)
            self._install_child_delegation(child, request, child_session)
            return child

        child = ToolRegistry()
        child.adopt_from(self.parent_registry)
        self._install_child_delegation(child, request, child_session)
        try:
            child.restrict(
                allow=list(request.tools), layer_name="subagent"
            )
        except Exception as exc:  # noqa: BLE001 - a bad filter is a delegation error
            raise SubagentError(
                f"could not restrict the child's tools: {exc}",
                code="SUBAGENT_TOOL_FILTER_FAILED",
            ) from exc
        return child

    def _install_child_delegation(
        self,
        child_registry: ToolRegistry,
        request: DelegationRequest,
        child_session: Any,
    ) -> None:
        """Give the child its *own* delegation tool, at the child's depth.

        This is the difference between a depth limit and the appearance of one.

        The parent's ``subagent`` tool is a closure over ``depth=0`` and
        ``parent_session=<the parent>``. Copying the registry copies that closure,
        so every child's tool would report depth 0, ``assert_max_depth(0)`` would
        pass forever, and a chain would run until the concurrency limit stopped
        it -- which is a resource bound, not a depth bound.

        Re-registering replaces the copy with one that knows the child's real
        depth and its own session, so the limit is enforced by the same check that
        was always there, now receiving a number that actually increments.
        """
        if child_session is None:
            # Without a session there is no child to give a tool to. This only
            # happens in tests that drive the registry directly.
            return

        if self.delegation_tool_factory is None:
            # No factory means the child would inherit the parent's tool, whose
            # depth is baked in at 0 -- and the depth limit would never trigger,
            # silently. Removing the capability is the safe default: a child that
            # cannot delegate is bounded, whereas a child that can delegate
            # without a working limit is a fork bomb.
            #
            # A runtime that wants nested delegation must supply the factory, and
            # supplying it is what makes the limit real rather than apparent.
            child_registry.unregister("subagent")
            return

        child_depth = request.child_depth()
        # A depth at or beyond the limit gets no delegation tool at all. Removing
        # the capability is stronger than refusing it: a tool the model cannot
        # see cannot be argued with.
        if child_depth >= request.max_depth:
            child_registry.unregister("subagent")
            return

        child_registry.replace(
            self.delegation_tool_factory(child_session, child_depth)
        )

    async def _drive(
        self,
        child: Any,
        child_session: Session,
        request: DelegationRequest,
        result: "asyncio.Future[DelegationResult]",
        start_seq: int,
        run_id: str,
    ) -> None:
        """Run the child to completion and read its outcome from its own log.

        ``run_id`` is the runtime's identifier for this run, and it is what the
        result reports. Reporting the child's session id instead would hand the
        model an id that ``cancel`` does not accept -- so a model that decided to
        stop a runaway subagent could not.
        """
        prompt = request.prompt
        if request.persona:
            # The persona goes with the task rather than into the child's system
            # prompt: it describes *this delegation*, and a system prompt is
            # recorded once and reused across turns.
            prompt = f"{request.persona}\n\n{prompt}"

        child.submit(prompt)
        await child.run_until_idle()

        stop = turn_stop_reason(child_session, since=start_seq)
        output = final_assistant_output(child_session, since=start_seq)
        turns = consumed_work(child_session, since=start_seq)

        # A completion with no answer is an error, not an empty success.
        #
        # The caller asked a question and got nothing back; reporting
        # ``completed`` would hand it an empty string it cannot distinguish from
        # a real (if terse) answer. ``turns`` is recorded rather than used as the
        # test, because a turn *does* run in this case -- it simply produces no
        # assistant message, which is precisely what makes it worth reporting.
        if stop == STOP_COMPLETED and not output:
            stop = STOP_ERROR

        structured: Mapping[str, Any] | None = None
        if request.output_schema is not None and stop == STOP_COMPLETED:
            structured, error = _validate_structured(output, request.output_schema)
            if error is not None:
                # A completion without a valid structured value is an error: the
                # caller asked for a shape and did not get one, and reporting
                # success would hand back something it cannot use.
                stop = STOP_ERROR
                if not result.done():
                    result.set_result(
                        DelegationResult(
                            run_id=run_id,
                            provider=self.name,
                            stop_reason=stop,
                            output=output,
                            error=error,
                            child_session=child_session.session_id,
                            events=len(child_session),
                        )
                    )
                return

        if not result.done():
            result.set_result(
                DelegationResult(
                    run_id=run_id,
                    provider=self.name,
                    stop_reason=stop,
                    output=output,
                    structured=structured,
                    child_session=child_session.session_id,
                    events=len(child_session),
                )
            )


def _validate_structured(
    output: str, schema: Mapping[str, Any]
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Parse and validate a structured answer.

    The answer must be JSON in the output text. Anything else is reported as an
    error rather than coerced: guessing at a malformed answer would produce a
    value the caller believes was validated.
    """
    import json

    text = output.strip()
    # Tolerate a fenced block, which models produce routinely.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"the subagent's answer is not valid JSON: {exc}"

    if not isinstance(parsed, dict):
        return None, (
            f"the subagent's answer must be a JSON object, got "
            f"{type(parsed).__name__}"
        )

    # A shallow structural check: required keys present, no unexpected ones when
    # the schema forbids them. A full validator is the schema layer's job; this
    # catches the common case of a child answering the wrong shape entirely.
    required = schema.get("required") or []
    missing = [name for name in required if name not in parsed]
    if missing:
        return None, f"the subagent's answer is missing required fields: {missing}"

    properties = schema.get("properties") or {}
    if properties and schema.get("additionalProperties") is False:
        extra = [name for name in parsed if name not in properties]
        if extra:
            return None, (
                f"the subagent's answer has fields the schema does not allow: {extra}"
            )

    return parsed, None

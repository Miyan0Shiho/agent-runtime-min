"""The subagent runtime: the single publication boundary for delegation.

Every delegation goes through :meth:`SubagentRuntime.start`, and everything that
can refuse is checked **before** a provider is asked to create anything:

1. the provider exists;
2. the provider supports what the request requires;
3. the depth limit is respected;
4. the output schema is a well-formed object schema;
5. the descriptor is JSON-lossless.

The order is fixed, so a caller gets the *first* problem rather than an arbitrary
one -- and, more importantly, so a refusal leaves nothing behind. A delegation
that failed after a child session had been created would leave a session nobody
owns, and no code path would ever clean it up.

Two properties the isolation provides, both of which the parent depends on:

* **The child gets its own session.** A shared session would mean the child's
  turns appear in the parent's transcript, and the parent's context would fill
  with work it did not do.
* **The child gets a restricted tool view.** A subagent that could call
  ``subagent`` without a depth limit is a fork bomb; one that could write outside
  its sandbox could damage the work it was delegated to help with.

Results are read from the child's own log rather than from a return value, which
is the same rule the rest of the runtime follows: what happened is what was
recorded.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol

from ..kernel.errors import describe
from ..session.session import Session
from ..tools.registry import ToolRegistry
from .contracts import (
    DEFAULT_MAX_DEPTH,
    STOP_ABORTED,
    STOP_ERROR,
    DelegationRequest,
    DelegationResult,
    NoProviderError,
    SubagentError,
    assert_capabilities,
    assert_max_depth,
    map_stop_reason,
)

log = logging.getLogger("arm.subagent")


class SubagentProvider(Protocol):
    """Creates and runs one child agent.

    Declared as a protocol so an in-process implementation, a container-backed
    one and a test double are all the same shape to the runtime. The runtime
    never learns how a child is actually run.
    """

    name: str
    supported: frozenset[str]

    async def start(
        self, request: DelegationRequest, *, run_id: str, parent: Session
    ) -> "RunningSubagent": ...


@dataclass
class RunningSubagent:
    """A started child, plus how to reach its outcome and to stop it.

    ``dispose`` is separate from ``result`` because they fail independently: a
    child whose teardown fails still produced a result, and a child whose result
    channel broke still needs releasing.
    """

    run_id: str
    provider: str
    child_session: Session
    result: "asyncio.Future[DelegationResult]"
    dispose: Callable[[], Awaitable[None]]
    #: The child's own agent loop, when the provider runs one in this process.
    runtime: Any = None


def validate_output_schema(schema: Mapping[str, Any] | None) -> None:
    """Refuse an output schema that is not a well-formed object schema.

    The child's answer is validated against it, so a malformed schema would fail
    at the *end* of a run -- after all the work had been done. Checking it up
    front turns a late failure into an immediate refusal.
    """
    if schema is None:
        return
    if not isinstance(schema, Mapping):
        raise SubagentError(
            f"an output schema must be an object, got {type(schema).__name__}"
        )
    if schema.get("type") not in (None, "object"):
        raise SubagentError(
            f"an output schema must describe an object, got type "
            f"{schema.get('type')!r}"
        )
    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        raise SubagentError("an output schema's 'properties' must be an object")


@dataclass
class SubagentRuntime:
    """Starts, tracks and settles delegated runs.

    Holds the provider registry and the run bookkeeping. It does not know how a
    child is executed, which is what lets the same runtime drive an in-process
    child and a container-backed one.
    """

    registry: ToolRegistry
    max_depth: int = DEFAULT_MAX_DEPTH
    #: Cap on concurrently running children, across all parents.
    max_active: int = 8

    _providers: dict[str, SubagentProvider] = field(default_factory=dict)
    _running: dict[str, RunningSubagent] = field(default_factory=dict)
    #: Slots held by children that are still running.
    _slots: int = 0

    # -- providers -------------------------------------------------------

    def register_provider(self, provider: SubagentProvider) -> Callable[[], None]:
        """Register a provider. Returns a disposer, so registration is reversible."""
        if provider.name in self._providers:
            raise SubagentError(
                f"a provider named {provider.name!r} is already registered",
                code="DUPLICATE_PROVIDER",
            )
        self._providers[provider.name] = provider

        def _dispose() -> None:
            if self._providers.get(provider.name) is provider:
                del self._providers[provider.name]

        return _dispose

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    @property
    def active(self) -> int:
        return len(self._running)

    # -- the single publication boundary ---------------------------------

    async def start(
        self, request: DelegationRequest, *, parent: Session
    ) -> RunningSubagent:
        """Start a delegated run.

        The five prechecks run in a fixed order and complete before the provider
        is consulted, so a refusal creates nothing. That property is the reason
        this method exists as a single boundary rather than as checks scattered
        through the provider.
        """
        # 1. The provider exists.
        provider = self._providers.get(request.provider)
        if provider is None:
            raise NoProviderError(request.provider, self.providers)

        # 2. It supports what the request requires.
        assert_capabilities(request.provider, provider.supported, request)

        # 3. The depth limit is respected.
        assert_max_depth(request.depth, limit=self.max_depth)

        # 4. The output schema is well-formed.
        validate_output_schema(request.output_schema)

        # 5. The descriptor survives a JSON round-trip.
        #    Done last because it is the most expensive check, and doing it
        #    before the cheap ones would waste work on requests that fail anyway.
        self._assert_lossless(request)

        if self._slots >= self.max_active:
            raise SubagentError(
                f"{self._slots} subagents are already running (limit "
                f"{self.max_active}); wait for one to finish",
                code="ACTIVATION_LIMIT_REACHED",
            )

        run_id = str(uuid.uuid4())
        self._slots += 1

        try:
            running = await provider.start(request, run_id=run_id, parent=parent)
        except BaseException:
            # The slot must be released on *any* failure, including cancellation:
            # a leaked slot permanently reduces capacity, and nothing would
            # report it.
            self._slots -= 1
            raise

        self._running[run_id] = running
        # The result channel is owned by the runtime, so a caller that never
        # awaits it still releases the slot.
        running.result.add_done_callback(lambda _f: self._release(run_id))
        return running

    def _assert_lossless(self, request: DelegationRequest) -> None:
        """Refuse a request whose descriptor cannot be stored faithfully.

        The descriptor is written to the parent's log, so anything that does not
        survive a JSON round-trip would make the log disagree with what was
        actually delegated.
        """
        from ..kernel.freeze import assert_json_lossless

        descriptor: dict[str, Any] = {
            "prompt": request.prompt,
            "provider": request.provider,
            "depth": request.depth,
            "max_depth": request.max_depth,
        }
        if request.label:
            descriptor["label"] = request.label
        if request.tools:
            descriptor["tools"] = list(request.tools)
        if request.persona:
            descriptor["persona"] = request.persona
        if request.model:
            descriptor["model"] = request.model
        if request.output_schema is not None:
            descriptor["output_schema"] = dict(request.output_schema)

        try:
            assert_json_lossless(descriptor, what="subagent descriptor")
        except Exception as exc:  # noqa: BLE001 - normalized to a subagent error
            raise SubagentError(
                f"the delegation descriptor must be losslessly serializable: "
                f"{describe(exc)}",
                code="BAD_DESCRIPTOR",
            ) from exc

    def _release(self, run_id: str) -> None:
        if self._running.pop(run_id, None) is not None:
            self._slots = max(0, self._slots - 1)

    # -- running ---------------------------------------------------------

    async def run(
        self, request: DelegationRequest, *, parent: Session, timeout: float | None = None
    ) -> DelegationResult:
        """Start a run and wait for it.

        A convenience over :meth:`start` for the common case. It always disposes
        the child, and a disposal failure never replaces a result that was
        obtained -- the result is the answer, and losing it to a teardown problem
        would be the worse outcome.
        """
        running = await self.start(request, parent=parent)
        try:
            if timeout is None:
                return await running.result
            return await asyncio.wait_for(running.result, timeout=timeout)
        except asyncio.TimeoutError:
            return DelegationResult(
                run_id=running.run_id,
                provider=running.provider,
                stop_reason=STOP_ABORTED,
                error=f"the subagent did not finish within {timeout}s",
                child_session=running.child_session.session_id,
            )
        finally:
            try:
                await running.dispose()
            except Exception as exc:  # noqa: BLE001
                log.warning("disposing subagent %s failed: %s", running.run_id, exc)

    def get(self, run_id: str) -> RunningSubagent | None:
        return self._running.get(run_id)

    async def cancel(self, run_id: str) -> bool:
        """Stop a running child. Returns whether there was one to stop."""
        running = self._running.get(run_id)
        if running is None:
            return False
        if not running.result.done():
            running.result.cancel()
        try:
            await running.dispose()
        except Exception as exc:  # noqa: BLE001
            log.warning("disposing cancelled subagent %s failed: %s", run_id, exc)
        return True

    async def shutdown(self) -> list[str]:
        """Stop every running child, reporting any that would not settle."""
        orphans: list[str] = []
        for run_id in list(self._running):
            try:
                await self.cancel(run_id)
            except Exception:  # noqa: BLE001
                orphans.append(run_id)
        return orphans


# ---------------------------------------------------------------------------
# Reading a child's outcome from its log
# ---------------------------------------------------------------------------


def final_assistant_output(session: Session, *, since: int = 0) -> str:
    """The child's last assistant message, read from its own log.

    Read from the log rather than captured from a return value, for the same
    reason the rest of the runtime does: what happened is what was recorded, and
    a captured value could disagree with the transcript.
    """
    for event in reversed(session.events()):
        if event.seq < since:
            break
        if event.type == "assistant/message":
            message = event.data.get("message") or {}
            content = message.get("content") or ()
            text = "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, Mapping)
            )
            if text:
                return text
    return ""


def turn_stop_reason(session: Session, *, since: int = 0) -> str:
    """The child's last turn termination reason, mapped to the vocabulary."""
    for event in reversed(session.events()):
        if event.seq < since:
            break
        if event.type == "turn/end":
            return map_stop_reason(str(event.data.get("reason", "error")))
    return STOP_ERROR


def consumed_work(session: Session, *, since: int = 0) -> int:
    """How many turns the child actually ran.

    Used to distinguish "the child did nothing" from "the child worked and then
    stopped", which matters when deciding whether an empty output is a refusal
    or a failure.
    """
    return sum(
        1
        for event in session
        if event.seq >= since and event.type == "turn/end"
    )

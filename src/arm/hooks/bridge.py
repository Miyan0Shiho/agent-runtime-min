"""The bridge: hook decisions onto the runtime's existing interception points.

This is the piece that makes hooks a *bridge* rather than a second system. The
runtime already has waterfall points; this module adapts a subprocess's decision
onto the type each point already expects. The kernel never learns hooks exist.

The mapping is not arbitrary, and one part of it is a genuine limitation that is
recorded rather than hidden:

* **pre-tool**: a hook's ``deny`` becomes the pipeline's deny, and ``ask`` becomes
  its ask. This is the point where a hook can actually prevent something.
* **post-tool**: a hook's ``deny`` becomes a block, with its feedback shown to the
  model. The tool already ran, so "deny" here means "the result is unacceptable",
  not "do not run it".
* **pre-step**: a hook's ``deny`` rejects the user's message before the step.
* **turn-stopping**: a hook's ``stop`` asks the run to continue rather than stop.

**The limitation**: the dialects define an ``updatedInput`` field that rewrites a
tool's arguments. It is parsed and **not applied**, and that is deliberate --
applying it would let a hook change what a tool does after the arguments were
frozen, which breaks the pipeline's central guarantee that arguments are frozen
once. The field is reported as a diagnostic so a configuration relying on it is
visibly unsupported rather than silently ignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from .protocol import (
    POINT_AGENT_CREATED,
    DECISION_ASK,
    DECISION_DENY,
    POINT_POST_TOOL,
    POINT_PRE_STEP,
    POINT_PRE_TOOL,
    POINT_TURN_STOPPING,
    Decision,
)
from .runner import HookRunner

log = logging.getLogger("arm.hooks.bridge")


@dataclass
class HookBridge:
    """Adapts hook decisions to the runtime's pipeline vocabulary."""

    runner: HookRunner
    #: Whether hooks are enabled at all. A disabled bridge must be a real
    #: short-circuit rather than a runner with no handlers, so the cost of an
    #: empty configuration is zero.
    enabled: bool = True

    async def pre_tool(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> Decision:
        """Decide whether a tool call may proceed.

        This is the point where a hook can prevent something, so a ``deny`` here
        is the strongest thing a hook can do.
        """
        if not self.enabled:
            return Decision()
        return await self.runner.fire(
            POINT_PRE_TOOL,
            tool_name,
            {
                "point": POINT_PRE_TOOL,
                "tool": tool_name,
                "arguments": dict(arguments),
            },
        )

    def last_denier(self) -> str:
        """Which hook produced the most recent deny, or an empty string.

        Read from the outcomes the runner already keeps. Returned as a plain
        identifier rather than the outcome object, so a caller records a string
        and does not hold a reference into the runner.
        """
        for outcome in reversed(self.runner.outcomes):
            if outcome.decision.kind == DECISION_DENY:
                # `HookHandler` derives an id when one is not supplied, so this
                # is non-empty for any handler that reached the runner -- a
                # hand-built one included. It was empty only while the loader was
                # the sole source of ids.
                return outcome.handler_id
        return ""

    async def post_tool(
        self, tool_name: str, arguments: Mapping[str, Any], result: str,
        *, is_error: bool,
    ) -> Decision:
        """Decide whether a tool's result is acceptable.

        The tool has already run, so a ``deny`` here means "the result is
        unacceptable" and is translated into a block rather than a refusal to
        execute.
        """
        if not self.enabled:
            return Decision()
        return await self.runner.fire(
            POINT_POST_TOOL,
            tool_name,
            {
                "point": POINT_POST_TOOL,
                "tool": tool_name,
                "arguments": dict(arguments),
                "result": result,
                "is_error": is_error,
            },
        )

    async def agent_created(self, session_id: str) -> tuple[str, ...]:
        """Fire the observing hooks for a new session.

        This is `SessionStart` in the Claude Code dialect -- the hook point people
        reach for first, and the one that was mapped, loaded, and never fired. A
        user who installed one got silence, which is indistinguishable from a hook
        that ran and decided nothing.

        The point is **observing**: it runs the hooks and returns any context they
        contributed, but a decision it makes cannot change what happens, because
        there is nothing yet to change. That is stated rather than pretended --
        `fire_parallel` reports decisions instead of applying them.
        """
        if not self.enabled:
            return ()
        try:
            _decision, outcomes = await self.runner.fire_parallel(
                POINT_AGENT_CREATED,
                session_id,
                {"point": POINT_AGENT_CREATED, "session": session_id},
            )
        except Exception as exc:  # noqa: BLE001 - a hook fault is not a start fault
            log.warning("the session-start hook failed: %s", exc)
            return ()

        context: list[str] = []
        for outcome in outcomes:
            context.extend(outcome.decision.context)
        return tuple(context)

    async def pre_step(self, text: str) -> Decision:
        """Decide whether a user's message may be processed."""
        if not self.enabled:
            return Decision()
        return await self.runner.fire(
            POINT_PRE_STEP, "", {"point": POINT_PRE_STEP, "text": text}
        )

    async def pre_step_decision(self, text: str) -> Any:
        """Adapt a hook's pre-step decision to the loop's ``StepDecision``.

        This is `UserPromptSubmit`. A hook that denies here rejects the user's
        message before any model call happens -- the point of the hook is to
        refuse work, so the refusal has to short-circuit rather than be recorded.

        The adapter lives here rather than in the loop so the loop stays ignorant
        of hooks: it asks a narrow question ("may this step run?") and this module
        answers it in the vocabulary the loop already uses.
        """
        from ..agent.loop import StepDecision

        if not self.enabled:
            return StepDecision.enter()

        decision = await self.runner.fire(
            POINT_PRE_STEP, "", {"point": POINT_PRE_STEP, "text": text}
        )
        if decision.kind == DECISION_DENY:
            return StepDecision.reject(decision.reason or "a hook rejected this message")
        return StepDecision.enter()

    async def post_tool_decision(
        self, tool_name: str, arguments: Mapping[str, Any], result: str,
        *, is_error: bool,
    ) -> Decision:
        """Run the post-tool hooks.

        This is `PostToolUse`. The tool has already run, so a denial here means
        "the result is unacceptable" rather than "do not run it" -- the loop
        turns it into a block with the hook's feedback, not a refusal to execute.
        """
        return await self.post_tool(
            tool_name, arguments, result, is_error=is_error
        )

    async def turn_stopping(self, reason: str, steps: int) -> Decision:
        """Ask whether the turn should stop.

        A ``stop`` decision here is the dialects' "keep going": the hook has
        decided the turn is not actually finished.
        """
        if not self.enabled:
            return Decision()
        return await self.runner.fire(
            POINT_TURN_STOPPING,
            "",
            {"point": POINT_TURN_STOPPING, "reason": reason, "steps": steps},
        )

    def unsupported_fields(self, payload: Mapping[str, Any]) -> list[str]:
        """Which dialect features were requested but cannot be honoured.

        Reported rather than silently ignored: a configuration that relies on
        ``updatedInput`` believes it is rewriting a tool's arguments, and a
        silent no-op would leave it believing that forever.
        """
        notes: list[str] = []
        if "updatedInput" in payload:
            notes.append(
                "updatedInput is not supported: tool arguments are frozen once "
                "they are recorded, and rewriting them afterwards would break "
                "that guarantee"
            )
        return notes


def pipeline_hook(bridge: HookBridge):
    """Build a pre-execute hook for the tool pipeline.

    Returned as a closure so the pipeline does not import this module -- the
    dependency runs one way, from the bridge to the pipeline, which is what keeps
    hooks removable.
    """

    async def pre_execute(call: Any, next_: Any) -> Any:
        decision = await bridge.pre_tool(
            str(getattr(call, "name", "")), dict(getattr(call, "arguments", {}) or {})
        )
        if decision.kind == DECISION_DENY:
            # Short-circuit: do not call next_, which is what a waterfall means.
            return _deny(
                decision.reason or "a hook denied this tool call",
                # Which hook said so. Found by running a denying hook for real:
                # the runtime knew the identity (`claude-code:PreToolUse:1`) and
                # the log recorded only `code: DENIED`, so with more than one
                # hook configured a reader could not tell which one blocked the
                # call. The same reasoning as M3.10 -- an identity that exists
                # should persist, and it is not what the model reads.
                bridge.last_denier(),
            )
        if decision.kind == DECISION_ASK:
            return _ask(decision.reason)
        return await next_()

    return pre_execute


def _deny(reason: str, handler_id: str = "") -> Any:
    from ..tools.pipeline import Decision as PipelineDecision

    return PipelineDecision.deny(reason, hook=handler_id)


def _ask(reason: str) -> Any:
    from ..tools.pipeline import Decision as PipelineDecision

    return PipelineDecision.ask(reason)

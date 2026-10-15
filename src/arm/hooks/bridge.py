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

    async def pre_step(self, text: str) -> Decision:
        """Decide whether a user's message may be processed."""
        if not self.enabled:
            return Decision()
        return await self.runner.fire(
            POINT_PRE_STEP, "", {"point": POINT_PRE_STEP, "text": text}
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
            return _deny(decision.reason or "a hook denied this tool call")
        if decision.kind == DECISION_ASK:
            return _ask(decision.reason)
        return await next_()

    return pre_execute


def _deny(reason: str) -> Any:
    from ..tools.pipeline import Decision as PipelineDecision

    return PipelineDecision.deny(reason)


def _ask(reason: str) -> Any:
    from ..tools.pipeline import Decision as PipelineDecision

    return PipelineDecision.ask(reason)

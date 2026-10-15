"""The ``subagent`` tool: delegating a self-contained task.

One tool. It takes a task and returns the child's answer, with the child's
session id so the caller can inspect what happened.

Three choices shape it:

**The prompt is required and is the whole task.** A delegation with a vague
prompt produces a vague result, and the child has none of the parent's context --
it cannot ask a clarifying question, so the prompt has to stand alone. The tool
description says so, because the failure mode is silent: the child does something
reasonable that is not what was wanted.

**The result carries the stop reason.** A caller must be able to tell "the child
finished" from "the child ran out of tokens" or "the child was cancelled", and
those need different handling. Returning only the text would make a truncated
answer look complete.

**Delegation is bounded.** The depth is threaded through, so a child cannot spawn
an unbounded chain. The limit is enforced by the runtime, not by this tool --
the tool only reports what it was given.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .contracts import (
    DEFAULT_MAX_DEPTH,
    STOP_COMPLETED,
    DelegationRequest,
    SubagentError,
)
from .runtime import SubagentRuntime


class SubagentParams(BaseModel):
    prompt: str = Field(
        description=(
            "The complete task for the subagent. It has none of this "
            "conversation's context and cannot ask questions, so state everything "
            "it needs: what to do, where, and what a finished result looks like."
        )
    )
    label: str = Field(
        default="",
        description="A short name for this delegation, for the record.",
    )
    tools: list[str] = Field(
        default_factory=list,
        description=(
            "Restrict the subagent to these tools. Empty means it inherits the "
            "current tool set."
        ),
    )
    persona: str = Field(
        default="",
        description=(
            "Extra instructions for this delegation only, prepended to the task. "
            "Use it for how to work, not what to do."
        ),
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "A JSON Schema the subagent's final answer must satisfy, e.g. "
            '{"type":"object","properties":{"count":{"type":"integer"}},'
            '"required":["count"]}. The subagent must answer with JSON matching '
            "it, and a non-conforming answer is reported as a failure."
        ),
    )
    model: str = Field(
        default="",
        description="A model override for the subagent, if the provider supports it.",
    )


class SubagentTool:
    """The delegation tool, bound to a runtime and a parent session."""

    def __init__(
        self,
        runtime: SubagentRuntime,
        *,
        parent_session: Any,
        depth: int = 0,
        provider: str = "in-process",
        timeout: float | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        self.runtime = runtime
        self.parent_session = parent_session
        self.depth = depth
        self.provider = provider
        self.timeout = timeout
        #: The limit this delegation is subject to, passed through so the
        #: runtime's own limit is the one enforced rather than the default.
        self.max_depth = max_depth

    def register(self) -> RegisteredTool:
        return RegisteredTool(
            definition=ToolDefinition(
                name="subagent",
                description=(
                    "Delegate a self-contained task to a fresh agent with its own "
                    "context, and get back its answer. Use it for work that would "
                    "otherwise fill this conversation with intermediate detail, or "
                    "for independent work that can run on its own.\n"
                    "The subagent cannot see this conversation, so the prompt must "
                    "be complete on its own."
                ),
                parameters=SubagentParams.model_json_schema(),
            ),
            body=self.run,
            validator=SubagentParams,
        )

    async def run(self, arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
        """Run a delegation and render its result.

        Every failure path returns a result: an unknown provider, an unsupported
        capability and a depth limit are ordinary outcomes the model can read and
        adapt to, not exceptions that should interrupt the step.
        """
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return ToolResult.fail("a subagent needs a task")

        schema = arguments.get("output_schema") or None
        request = DelegationRequest(
            prompt=prompt,
            provider=self.provider,
            label=str(arguments.get("label", "")),
            depth=self.depth,
            max_depth=self.max_depth,
            tools=tuple(str(t) for t in (arguments.get("tools") or ())),
            persona=str(arguments.get("persona", "")),
            output_schema=dict(schema) if isinstance(schema, dict) and schema else None,
            model=str(arguments.get("model", "")),
        )

        try:
            result = await self.runtime.run(
                request, parent=self.parent_session, timeout=self.timeout
            )
        except SubagentError as exc:
            # The refusal already explains itself; surfacing the code lets the
            # model route on the kind of problem rather than parsing the message.
            return ToolResult.fail(str(exc), code=exc.code)

        rendered = _render(result)
        if result.ok:
            return ToolResult.ok(
                rendered,
                value={
                    "run_id": result.run_id,
                    "stop_reason": result.stop_reason,
                    "child_session": result.child_session,
                    "structured": dict(result.structured) if result.structured else None,
                },
            )
        return ToolResult.fail(rendered, code=f"SUBAGENT_{result.stop_reason.upper()}")


def _render(result: Any) -> str:
    """Render a delegation result for the model.

    A non-completion carries its reason explicitly. Returning the partial text
    alone would let a truncated answer read as a complete one -- and the model
    would build on it.
    """
    if result.stop_reason == STOP_COMPLETED:
        body = result.output or "(the subagent returned no output)"
        if result.structured is not None:
            import json

            return (
                f"{body}\n\nValidated result:\n"
                f"{json.dumps(dict(result.structured), ensure_ascii=False, indent=2)}"
            )
        return body

    reason = {
        "max-tokens": "the subagent ran out of output tokens",
        "aborted": "the subagent was stopped before finishing",
        "refusal": "the subagent declined the task",
        "error": "the subagent failed",
    }.get(result.stop_reason, f"the subagent stopped: {result.stop_reason}")

    parts = [f"Subagent did not complete: {reason}."]
    if result.error:
        parts.append(result.error)
    if result.output:
        parts.append(f"Partial output:\n{result.output}")
    if result.child_session:
        parts.append(f"Its session was {result.child_session}.")
    return "\n\n".join(parts)

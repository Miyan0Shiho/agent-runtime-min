"""The goal tools: ``create_goal``, ``update_goal``, ``get_goal``.

Three tools, and the interesting design is in what ``update_goal`` refuses.

**The revision is mandatory.** Every update must carry the revision the caller
believes it is editing. This is not ceremony: a model that has been thinking for
several steps may be holding a stale view, and a human may have changed the goal
in between. Requiring the revision turns that race into an error the model can
act on -- "read it again" -- rather than a silent overwrite of a decision it never
saw.

**``blocked`` needs corroboration.** A model reporting a blocker once is not
evidence of a wall; models report blockers optimistically and then find a way
through. The store refuses until the same condition has persisted for several
rounds, and the tool surfaces that refusal as a result the model can read.

**Resuming is how a goal comes back to life.** A goal restored from a log is
always disarmed, so the only route to automatic continuation is an explicit
``resume``. The tool's description says this in the terms a model needs: when a
human asks to continue, in any wording, resume the goal.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .store import (
    DEFAULT_MAX_ROUNDS,
    PHASE_ACTIVE,
    GoalError,
    GoalRef,
    GoalStore,
)


class CreateGoalParams(BaseModel):
    objective: str = Field(
        description=(
            "The objective, stated concretely. Infer it from the user's request "
            "even when they did not use the word 'goal'."
        )
    )
    max_rounds: int = Field(
        default=DEFAULT_MAX_ROUNDS,
        description="Maximum automatic continuation rounds before the goal stops.",
    )


class UpdateGoalParams(BaseModel):
    action: str = Field(
        description=(
            "One of: 'edit', 'pause', 'resume', 'complete', 'blocked', 'clear'."
        )
    )
    revision: int = Field(
        description=(
            "The revision you are editing. Read the goal first with get_goal; a "
            "mismatch means your view is stale and the update is refused."
        )
    )
    objective: str = Field(default="", description="New objective, for 'edit'.")
    max_rounds: int = Field(default=0, description="New round ceiling, for 'edit'.")
    blocked_reason: str = Field(
        default="",
        description=(
            "For 'blocked': the concrete condition that stops progress. Required, "
            "and it must be the same condition that has persisted across rounds."
        ),
    )


class GetGoalParams(BaseModel):
    """No arguments: reading the goal is always safe."""


class GoalTools:
    """The goal tool set, bound to one session's store."""

    def __init__(self, store: GoalStore) -> None:
        self.store = store

    def tools(self) -> list[RegisteredTool]:
        return [self._create(), self._get(), self._update()]

    # -- create ----------------------------------------------------------

    def _create(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            try:
                goal = self.store.create(
                    str(arguments.get("objective", "")),
                    max_rounds=int(arguments.get("max_rounds", DEFAULT_MAX_ROUNDS)),
                )
            except GoalError as exc:
                return ToolResult.fail(str(exc), code=exc.code)
            return ToolResult.ok(
                f"Created goal {goal.id} at revision {goal.revision}.\n"
                f"Objective: {goal.objective}\n"
                f"Round ceiling: {goal.snapshot.max_rounds}",
                value={"id": goal.id, "revision": goal.revision},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="create_goal",
                description=(
                    "Start working toward a long-running objective that should "
                    "survive across turns. Use it when the user describes "
                    "multi-step work they want pursued to completion, even if "
                    "they did not say 'goal'."
                ),
                parameters=CreateGoalParams.model_json_schema(),
            ),
            body=body,
            validator=CreateGoalParams,
        )

    # -- read ------------------------------------------------------------

    def _get(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            goal = self.store.current()
            if goal is None:
                return ToolResult.ok("There is no goal for this session.")

            lines = [
                f"id: {goal.id}",
                f"revision: {goal.revision}",
                f"objective: {goal.objective}",
                f"phase: {goal.phase}",
                f"rounds: {goal.rounds_started}/{goal.snapshot.max_rounds}",
                f"activation: {goal.activation}",
            ]
            if goal.snapshot.blocked_reason:
                lines.append(
                    f"blocked ({goal.snapshot.blocked_code}): "
                    f"{goal.snapshot.blocked_reason}"
                )
            if not goal.armed and goal.phase == PHASE_ACTIVE:
                lines.append(
                    "note: this goal is not armed, so it will not continue on its "
                    "own. Call update_goal with action 'resume' to re-arm it."
                )
            return ToolResult.ok(
                "\n".join(lines),
                value={
                    "id": goal.id,
                    "revision": goal.revision,
                    "phase": goal.phase,
                    "rounds_started": goal.rounds_started,
                },
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="get_goal",
                description=(
                    "Read the current goal, including its revision. Always read "
                    "before updating, because an update needs the current revision."
                ),
                parameters={"type": "object", "properties": {}},
                concurrency_safe=lambda _args: True,
            ),
            body=body,
        )

    # -- update ----------------------------------------------------------

    def _update(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            action = str(arguments.get("action", "")).strip()
            revision = arguments.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool):
                return ToolResult.fail(
                    "update_goal requires the integer 'revision' you are editing; "
                    "call get_goal first"
                )

            current = self.store.current()
            if current is None:
                return ToolResult.fail("there is no goal to update", code="GOAL_MISSING")

            ref = GoalRef(id=current.id, revision=revision)

            try:
                if action == "edit":
                    objective = str(arguments.get("objective", "")).strip()
                    max_rounds = int(arguments.get("max_rounds", 0))
                    goal = self.store.edit(
                        expected=ref,
                        objective=objective or None,
                        max_rounds=max_rounds or None,
                    )
                elif action == "pause":
                    goal = self.store.pause(expected=ref)
                elif action == "resume":
                    goal = self.store.resume(expected=ref)
                elif action == "complete":
                    goal = self.store.complete(expected=ref)
                elif action == "blocked":
                    goal = self.store.block(
                        str(arguments.get("blocked_reason", "")),
                        expected=ref,
                    )
                elif action == "clear":
                    self.store.clear(expected=ref)
                    return ToolResult.ok(
                        "Cleared the goal. The objective is abandoned, not "
                        "completed.",
                        value={"cleared": True},
                    )
                else:
                    return ToolResult.fail(
                        f"unknown action {action!r}; expected one of: edit, pause, "
                        "resume, complete, blocked, clear"
                    )
            except GoalError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            return ToolResult.ok(
                f"Goal is now {goal.phase} at revision {goal.revision} "
                f"(rounds {goal.rounds_started}/{goal.snapshot.max_rounds}).",
                value={
                    "id": goal.id,
                    "revision": goal.revision,
                    "phase": goal.phase,
                },
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="update_goal",
                description=(
                    "Change the current goal. Actions: 'edit' (objective or round "
                    "ceiling), 'pause', 'resume', 'complete', 'blocked', 'clear'.\n"
                    "Call 'resume' when a human asks to continue or resume an "
                    "active goal, in any wording: a goal restored from a previous "
                    "session is not armed and will not continue by itself.\n"
                    "Mark 'complete' only when the objective is actually achieved. "
                    "Mark 'blocked' only after the same blocking condition has "
                    "persisted for several rounds, and state that condition in "
                    "blocked_reason."
                ),
                parameters=UpdateGoalParams.model_json_schema(),
            ),
            body=body,
            validator=UpdateGoalParams,
        )

"""Plan mode: a mode that changes the prompt, and nothing else.

The single most important property of this module is what it does **not** do:

    plan mode restricts no tools.

That is deliberate, and it is the design's central claim. A mode that silently
removed capabilities would make "plan mode" mean "a different, weaker agent" --
and every tool-related bug report would have to consider whether plan mode was
active. Instead, plan mode contributes a prompt section that asks the agent to
plan before acting. Enforcement, when it is wanted, comes from the sandbox and
approval layers, which are configured separately and are honest about being
restrictions.

The consequence is that entering and leaving plan mode **must not change the
request's tool catalogue**. If it did, a tool call would fail for a reason that
has nothing to do with the tool, and the model would have no way to tell.

Two pieces of state, kept in different places on purpose:

* **``active`` lives in the log.** It is a whole-value replacement event, so the
  last one written wins and a resumed session knows its mode without any
  bookkeeping. Recovery, fork and replay all get it for free.
* **``pending`` lives in the process.** A user's choice cannot be written until a
  step is accepted -- every event belongs to a turn, and writing one outside a
  turn would produce a log the recovery logic cannot interpret. So the choice
  waits in memory for the next accepted step.

That split is why a mode change made while the agent is idle takes effect on the
next turn rather than immediately, and why a crash between the choice and the
next step loses the choice rather than corrupting the log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..kernel.errors import ArmError
from ..session.events import PLAN_MODE, REQUEST_HEADER
from ..session.session import Session

log = logging.getLogger("arm.plan")

#: The prompt section contributed while plan mode is active.
#:
#: Written as an instruction about *process*, not about permissions. It never
#: says a tool is unavailable, because none is.
PLAN_POLICY = """# Plan mode

You are in plan mode. Before making changes, investigate and propose a plan.

- Read and search freely. Understanding the problem is the work right now.
- Do not modify files or run commands that change state until the plan is
  approved. This is a request about sequencing, not a restriction on what you
  are able to do -- nothing will stop you, so the discipline is yours to keep.
- Write the plan as a document starting with a single `# ` heading, so it can be
  reviewed as a plan rather than as prose.
- Make the plan specific: name the files, the functions and the order. A plan
  that says "refactor the module" has not decided anything.

When the plan is ready, call `exit_plan_mode` to submit it for approval."""


class PlanError(ArmError):
    """A plan-mode operation could not be completed."""


@dataclass(frozen=True, slots=True)
class PlanState:
    """The committed state, read from the log."""

    active: bool = False
    #: The seq of the event that set the current value, for auditing.
    since: int | None = None


@dataclass
class PlanMode:
    """Tracks plan mode for one session.

    Reads its committed state from the log; holds an uncommitted choice in
    memory until a step is accepted.
    """

    session: Session
    #: A choice that cannot be written yet, because no step has been accepted.
    _pending: bool | None = None
    _pending_narrate: bool = True

    # -- committed state -------------------------------------------------

    def state(self) -> PlanState:
        """The committed state: the last ``plan/mode`` event, or inactive.

        Read by scanning forward and keeping the last value rather than by
        finding the first. The event is a whole-value replacement, so the most
        recent one is the current one.
        """
        active = False
        since: int | None = None
        for event in self.session:
            if event.type == PLAN_MODE:
                active = bool(event.data.get("active", False))
                since = event.seq
        return PlanState(active=active, since=since)

    @property
    def active(self) -> bool:
        """The effective state: a pending choice wins over the committed one.

        Callers that ask "should the prompt say plan mode?" want the answer that
        will be true for the step about to run, not the one that was true when
        the last step ran.
        """
        if self._pending is not None:
            return self._pending
        return self.state().active

    @property
    def pending(self) -> bool:
        """Whether a choice is waiting for an accepted step."""
        return self._pending is not None and self._pending != self.state().active

    # -- changing it -----------------------------------------------------

    def set(self, active: bool, *, narrate: bool = True) -> str:
        """Request a mode change.

        Returns one of ``committed``, ``queued`` or ``noop`` -- the same
        vocabulary the caller uses to tell the user what happened.

        The choice is committed immediately when a step is in flight, because
        then it belongs to that step. Otherwise it is queued, because an event
        written outside a turn would produce a log the recovery logic cannot
        read.
        """
        current = self.state().active
        if active == current and self._pending is None:
            return "noop"

        if self._can_commit():
            self._commit(active)
            return "committed"

        self._pending = active
        self._pending_narrate = narrate
        return "queued"

    def cancel_pending(self) -> bool:
        """Drop a queued choice. Returns whether there was one."""
        if self._pending is None:
            return False
        self._pending = None
        return True

    def _can_commit(self) -> bool:
        """Whether an event can be written right now.

        An event must belong to a turn. The session's own log tells us whether
        one is open: a ``turn/start`` with no matching ``turn/end`` means a step
        is in flight and a mode event would land inside it, which is where it
        belongs.
        """
        open_turn = False
        for event in self.session:
            if event.type == "turn/start":
                open_turn = True
            elif event.type == "turn/end":
                open_turn = False
        return open_turn

    def _commit(self, active: bool) -> None:
        self.session.append(
            PLAN_MODE,
            {"active": active, "at_turn": self._current_turn()},
        )
        self._pending = None

    def _current_turn(self) -> int | None:
        turn: int | None = None
        for event in self.session:
            if event.type == "turn/start":
                turn = int(event.data.get("turn", 0))
            elif event.type == "turn/end":
                turn = None
        return turn

    def commit_pending(self) -> bool:
        """Write a queued choice, if a step has since been accepted.

        Called from the pre-step hook, *after* the step is accepted -- so a
        rejected step does not consume the user's choice, and the choice is not
        lost either.
        """
        if self._pending is None:
            return False
        if not self._can_commit():
            return False
        active = self._pending
        self._commit(active)
        return True

    # -- what the model sees ---------------------------------------------

    def prompt_section(self) -> str:
        """The prompt contribution. Empty when inactive.

        Returns an empty string rather than omitting the segment, so that the
        assembler can treat "no plan policy" and "plan policy" uniformly.
        """
        return PLAN_POLICY if self.active else ""

    def notification(self) -> str | None:
        """A one-line notice that the mode changed, or ``None``.

        Compares the *effective* mode against what the last request header
        recorded. The effective mode is the right side of the comparison because
        the notice describes the request about to be made; the header is the
        right baseline because it records what the model was last told.

        Only emitted when the two differ. Without that check, every step would
        re-announce the current mode, which is pure noise -- and noise in the
        transcript is what makes a model stop reading carefully.
        """
        effective = self.active
        last_header_active = self._active_at_last_header()

        # A header with no recorded mode cannot say what the model was told. A
        # redundant notice costs one line; a missing one leaves the model acting
        # on a mode that changed, so the ambiguity resolves toward notifying.
        if last_header_active is None:
            return self._notice_for(effective) if effective else None

        if last_header_active == effective:
            return None
        return self._notice_for(effective)

    @staticmethod
    def _notice_for(active: bool) -> str:
        if active:
            return "Plan mode is active: propose a plan before changing anything."
        return "Plan mode is off: you may make changes again."

    def _active_at_last_header(self) -> bool | None:
        """What the most recent request header recorded, or ``None``.

        The header is the record of what the model was last told, so it is the
        right baseline for "has this changed since?".
        """
        for event in reversed(self.session.events()):
            if event.type == REQUEST_HEADER:
                if "plan_active" in event.data:
                    return bool(event.data["plan_active"])
                return None
        return None

    def header_fields(self) -> dict[str, Any]:
        """The fields to record on the next request header.

        Recording the mode on the header is what makes the notification
        idempotent: the next step can tell whether the model already knows.
        """
        return {"plan_active": self.active}


def plan_tools(plan: PlanMode, *, approve: Callable[[str], Any] | None = None):
    """The ``exit_plan_mode`` tool.

    Registered whether or not plan mode is active. That is the point: the tool
    catalogue must not change with the mode, so the tool is always present and
    refuses when there is nothing to exit.
    """
    from pydantic import BaseModel, Field

    from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult

    class ExitParams(BaseModel):
        plan: str = Field(
            description=(
                "The plan, as Markdown, starting with a single '# ' heading."
            )
        )

    async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
        text = str(arguments.get("plan", "")).strip()

        if not plan.state().active:
            return ToolResult.fail(
                "plan mode is not active; there is no plan to submit"
            )

        # A plan that does not start with a heading cannot be reviewed as a plan
        # -- it reads as prose, and the whole point of submitting one is that it
        # is a document someone can approve.
        if not text.startswith("# ") or len(text) < 4:
            return ToolResult.fail(
                "the plan must start with a single '# ' heading naming what it "
                "proposes, so it can be reviewed as a plan"
            )

        if approve is None:
            # Failing closed: a mode change the user cannot approve would leave
            # the session stuck in a mode nobody chose.
            return ToolResult.fail(
                "plan mode cannot be exited: no approval channel is available. "
                "The plan was not submitted."
            )

        try:
            granted = approve(text)
            if hasattr(granted, "__await__"):
                granted = await granted
        except Exception as exc:  # noqa: BLE001 - a broken channel denies
            return ToolResult.fail(f"the approval channel failed: {exc}")

        if not granted:
            return ToolResult.ok(
                "The plan was not approved. Stay in plan mode and revise it "
                "based on the feedback.",
                value={"approved": False},
            )

        # Narrated as silent: the approval already told the user what happened,
        # so a second notice would be redundant.
        plan.set(False, narrate=False)
        plan.commit_pending()
        return ToolResult.ok(
            "Plan approved. You may now make the changes it describes.",
            value={"approved": True},
            concludes_turn=True,
        )

    return RegisteredTool(
        definition=ToolDefinition(
            name="exit_plan_mode",
            description=(
                "Submit the current plan for approval and leave plan mode. Only "
                "call this when plan mode is active and the plan is complete."
            ),
            parameters=ExitParams.model_json_schema(),
        ),
        body=body,
        validator=ExitParams,
    )

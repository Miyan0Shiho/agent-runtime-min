"""Persistent goals: a long-running objective the session keeps working toward.

The central design decision is a **split between two kinds of state**, and it is
worth stating plainly because everything else follows from it:

* **``phase``** answers *"what has happened to this goal?"* -- active, paused,
  blocked, complete. It is durable, written to the log, and survives a restart.
* **``activation``** answers *"may this process continue the goal on its own?"*
  It is **process-local and never written to the log**.

Why the split matters: a goal that was active when the process died must not
silently resume when the process comes back. An agent that keeps working toward
an objective nobody is watching is exactly the failure mode this prevents. So a
goal is always ``disarmed`` on load, and only a human asking to resume re-arms
it. That is a structural guarantee rather than a discipline -- there is no code
path that can arm a goal implicitly, because the field has no representation in
the log to restore it from.

The second decision is **compare-and-set revisions**. Every change carries the
revision the caller believed it was editing, and a mismatch is an error rather
than a last-write-wins. This is what lets a model that is running concurrently
with a human notice that its view is stale, instead of overwriting a decision it
never saw.

The third is that **``rounds_started`` is counted, not stored**. A stored counter
in a snapshot would be a second source of truth that can drift from the events
that actually happened; counting them cannot drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ..kernel.errors import ArmError
from ..session.events import GOAL_CHANGED, GOAL_ROUND
from ..session.session import Session

log = logging.getLogger("arm.goal")

#: The goal's durable lifecycle phase.
PHASE_ACTIVE = "active"
PHASE_PAUSED = "paused"
PHASE_BLOCKED = "blocked"
PHASE_COMPLETE = "complete"

PHASES = frozenset({PHASE_ACTIVE, PHASE_PAUSED, PHASE_BLOCKED, PHASE_COMPLETE})

#: Whether the live process may continue the goal by itself. Never persisted.
ACTIVATION_ARMED = "armed"
ACTIVATION_DISARMED = "disarmed"

#: Default ceiling on automatic continuation rounds. A goal with no ceiling is a
#: loop that runs until someone notices.
DEFAULT_MAX_ROUNDS = 256

#: How many consecutive rounds must report the same blocker before ``blocked`` is
#: accepted. One bad round is a bad round; three in a row is a wall.
DEFAULT_BLOCKED_AFTER = 3

#: Machine-readable reasons a goal stopped making progress. Each is a durable,
#: routable code rather than prose, so a UI and a model can both act on it.
BLOCK_ROUND_LIMIT = "round-limit"
BLOCK_QUEUE_FAILED = "queue-failed"
BLOCK_PROMPT_REJECTED = "prompt-rejected"
BLOCK_MODEL_REPORTED = "model-reported"


class GoalError(ArmError):
    """A goal operation was refused."""

    def __init__(self, message: str, *, code: str = "GOAL_INVALID") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class GoalRef:
    """Identity plus revision: the compare-and-set token.

    Both fields together, not the id alone. An id identifies *which* goal; the
    revision identifies *which version of it* the caller was looking at.
    """

    id: str
    revision: int

    def to_record(self) -> dict[str, Any]:
        return {"id": self.id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class GoalSnapshot:
    """The durable part of a goal: exactly what is written to the log."""

    id: str
    revision: int
    objective: str
    phase: str = PHASE_ACTIVE
    blocked_reason: str = ""
    blocked_code: str = ""
    max_rounds: int = DEFAULT_MAX_ROUNDS
    created_at: str = ""
    updated_at: str = ""

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "revision": self.revision,
            "objective": self.objective,
            "phase": self.phase,
            "max_rounds": self.max_rounds,
        }
        if self.blocked_reason:
            record["blocked_reason"] = self.blocked_reason
        if self.blocked_code:
            record["blocked_code"] = self.blocked_code
        if self.created_at:
            record["created_at"] = self.created_at
        if self.updated_at:
            record["updated_at"] = self.updated_at
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "GoalSnapshot":
        return cls(
            id=str(record.get("id", "")),
            revision=int(record.get("revision", 1)),
            objective=str(record.get("objective", "")),
            phase=str(record.get("phase", PHASE_ACTIVE)),
            blocked_reason=str(record.get("blocked_reason", "")),
            blocked_code=str(record.get("blocked_code", "")),
            max_rounds=int(record.get("max_rounds", DEFAULT_MAX_ROUNDS)),
            created_at=str(record.get("created_at", "")),
            updated_at=str(record.get("updated_at", "")),
        )

    @property
    def ref(self) -> GoalRef:
        return GoalRef(id=self.id, revision=self.revision)


@dataclass(frozen=True, slots=True)
class GoalView:
    """The full picture: durable snapshot plus process-local facts."""

    snapshot: GoalSnapshot
    rounds_started: int = 0
    activation: str = ACTIVATION_DISARMED

    @property
    def id(self) -> str:
        return self.snapshot.id

    @property
    def revision(self) -> int:
        return self.snapshot.revision

    @property
    def ref(self) -> GoalRef:
        """The compare-and-set token for this view.

        Exposed here as well as on the snapshot so that a caller holding a view
        does not have to reach through it to build the token for its next write.
        """
        return GoalRef(id=self.snapshot.id, revision=self.snapshot.revision)

    @property
    def phase(self) -> str:
        return self.snapshot.phase

    @property
    def objective(self) -> str:
        return self.snapshot.objective

    @property
    def armed(self) -> bool:
        return self.activation == ACTIVATION_ARMED

    @property
    def may_continue(self) -> bool:
        """Whether an automatic round may start right now.

        Both conditions are required, and the process-local one is checked
        second on purpose: it is the one that cannot be restored from the log.
        """
        return (
            self.snapshot.phase == PHASE_ACTIVE
            and self.armed
            and self.rounds_started < self.snapshot.max_rounds
        )

    @property
    def exhausted(self) -> bool:
        return self.rounds_started >= self.snapshot.max_rounds


class GoalStore:
    """Reads and writes goals for one session.

    Every durable change is an event. Nothing is cached in a way that could
    disagree with the log: the current goal is folded from the events each time,
    which is cheap because goals change rarely and correct by construction.
    """

    def __init__(self, session: Session, *, blocked_after: int = DEFAULT_BLOCKED_AFTER) -> None:
        if blocked_after < 1:
            raise GoalError("blocked_after must be a positive integer")
        self.session = session
        self.blocked_after = blocked_after
        #: Process-local activation, keyed by goal id. Absent means disarmed.
        self._activation: dict[str, str] = {}
        #: Consecutive rounds that reported the same blocker, for the guardrail.
        self._consecutive_blocks: dict[str, int] = {}

    # -- reading ---------------------------------------------------------

    def current(self) -> GoalView | None:
        """The goal the session is working on, or ``None``.

        Folded from the events: the last ``goal/changed`` wins, because each
        carries a whole snapshot and the revisions are monotonic.
        """
        snapshot: GoalSnapshot | None = None
        rounds: dict[str, int] = {}

        for event in self.session:
            if event.type == GOAL_CHANGED:
                record = event.data.get("goal")
                if isinstance(record, Mapping):
                    snapshot = GoalSnapshot.from_record(record)
            elif event.type == GOAL_ROUND:
                goal_id = str(event.data.get("goal_id", ""))
                rounds[goal_id] = rounds.get(goal_id, 0) + 1

        if snapshot is None:
            return None

        return GoalView(
            snapshot=snapshot,
            rounds_started=rounds.get(snapshot.id, 0),
            # Absent from the map means disarmed, which is what makes a restart
            # safe: there is no way for the log to restore an armed state.
            activation=self._activation.get(snapshot.id, ACTIVATION_DISARMED),
        )

    def require_current(self) -> GoalView:
        goal = self.current()
        if goal is None:
            raise GoalError("there is no goal", code="GOAL_MISSING")
        return goal

    def history(self) -> tuple[GoalSnapshot, ...]:
        """Every snapshot ever written, oldest first."""
        out: list[GoalSnapshot] = []
        for event in self.session:
            if event.type == GOAL_CHANGED:
                record = event.data.get("goal")
                if isinstance(record, Mapping):
                    out.append(GoalSnapshot.from_record(record))
        return tuple(out)

    # -- writing ---------------------------------------------------------

    def create(
        self,
        objective: str,
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        replace_complete: bool = True,
    ) -> GoalView:
        """Start a goal.

        Refuses to overwrite a goal that is still running: silently replacing an
        objective would discard the work already done toward it, and the caller
        almost certainly meant to ``edit``.
        """
        objective = objective.strip()
        if not objective:
            raise GoalError("a goal needs an objective", code="GOAL_EMPTY")
        if max_rounds < 1:
            raise GoalError(
                "max_rounds must be a positive integer", code="GOAL_INVALID"
            )

        existing = self.current()
        if existing is not None and existing.phase != PHASE_COMPLETE:
            raise GoalError(
                f"a goal is already {existing.phase} ({existing.id!r}); "
                "edit it, or complete it first",
                code="GOAL_EXISTS",
            )
        if existing is not None and not replace_complete:
            raise GoalError(
                "a completed goal already exists; pass replace_complete to start "
                "a new one",
                code="GOAL_EXISTS",
            )

        from ..session.envelope import utc_now_iso

        now = utc_now_iso()
        snapshot = GoalSnapshot(
            id=_new_goal_id(objective),
            revision=1,
            objective=objective,
            phase=PHASE_ACTIVE,
            max_rounds=max_rounds,
            created_at=now,
            updated_at=now,
        )
        self._write(snapshot, action="create")
        # Creation arms the goal, because a human just asked for it in this
        # process. This is the only place activation is set without an explicit
        # resume.
        self._activation[snapshot.id] = ACTIVATION_ARMED
        return self.require_current()

    def edit(
        self,
        *,
        expected: GoalRef,
        objective: str | None = None,
        max_rounds: int | None = None,
    ) -> GoalView:
        """Change a goal's objective or its round ceiling.

        The revision must match. A mismatch means the caller's view is stale, and
        applying a stale edit would overwrite a decision made in between.
        """
        goal = self.require_current()
        self._assert_revision(goal, expected)

        if objective is None and max_rounds is None:
            raise GoalError("an edit must change something", code="GOAL_EMPTY_EDIT")

        new_objective = goal.objective if objective is None else objective.strip()
        if not new_objective:
            raise GoalError("the objective must not be empty", code="GOAL_EMPTY")
        new_max = goal.snapshot.max_rounds if max_rounds is None else max_rounds
        if new_max < 1:
            raise GoalError("max_rounds must be a positive integer", code="GOAL_INVALID")

        from ..session.envelope import utc_now_iso

        snapshot = GoalSnapshot(
            id=goal.id,
            revision=goal.revision + 1,
            objective=new_objective,
            phase=goal.phase,
            blocked_reason=goal.snapshot.blocked_reason,
            blocked_code=goal.snapshot.blocked_code,
            max_rounds=new_max,
            created_at=goal.snapshot.created_at,
            updated_at=utc_now_iso(),
        )
        self._write(snapshot, action="edit")
        return self.require_current()

    def pause(self, *, expected: GoalRef | None = None) -> GoalView:
        """Stop automatic continuation. The objective is kept."""
        goal = self.require_current()
        if expected is not None:
            self._assert_revision(goal, expected)
        if goal.phase not in (PHASE_ACTIVE, PHASE_BLOCKED):
            raise GoalError(
                f"cannot pause a goal that is {goal.phase}", code="GOAL_INVALID_TRANSITION"
            )
        return self._transition(goal, PHASE_PAUSED, action="pause")

    def resume(self, *, expected: GoalRef | None = None) -> GoalView:
        """Re-arm a goal and let this process continue it.

        Two refusals matter:

        * a goal that is already active *and armed* is a duplicate call, not a
          state change;
        * a goal that has spent its rounds cannot be resumed without a new
          ceiling, because resuming it would immediately hit the limit again.
        """
        goal = self.require_current()
        if expected is not None:
            self._assert_revision(goal, expected)
        if goal.phase == PHASE_COMPLETE:
            raise GoalError(
                "a completed goal cannot be resumed", code="GOAL_INVALID_TRANSITION"
            )
        if goal.phase == PHASE_ACTIVE and goal.armed:
            raise GoalError(
                "the goal is already active and armed", code="GOAL_INVALID_TRANSITION"
            )
        if goal.exhausted:
            raise GoalError(
                f"the goal has used all {goal.snapshot.max_rounds} rounds; raise "
                "max_rounds before resuming",
                code="GOAL_ROUND_LIMIT",
            )

        self._transition(goal, PHASE_ACTIVE, action="resume", clear_block=True)
        self._activation[goal.id] = ACTIVATION_ARMED
        self._consecutive_blocks.pop(goal.id, None)
        return self.require_current()

    def complete(self, *, expected: GoalRef | None = None) -> GoalView:
        """Mark the objective achieved."""
        goal = self.require_current()
        if expected is not None:
            self._assert_revision(goal, expected)
        if goal.phase == PHASE_COMPLETE:
            raise GoalError("the goal is already complete", code="GOAL_INVALID_TRANSITION")
        self._transition(goal, PHASE_COMPLETE, action="complete")
        self._activation[goal.id] = ACTIVATION_DISARMED
        # Re-read: the returned view must reflect the activation this call just
        # set, not the one that was in effect when the transition started.
        return self.require_current()

    def block(
        self,
        reason: str,
        *,
        code: str = BLOCK_MODEL_REPORTED,
        expected: GoalRef | None = None,
        force: bool = False,
    ) -> GoalView:
        """Record that the goal is stuck.

        The guardrail: a model reporting the same blocker once is not evidence of
        a wall, so ``blocked`` is refused until the same condition has persisted
        for several rounds. ``force`` exists for the runtime's own guardrails --
        a round limit and a rejected prompt are facts, not claims, and do not
        need corroboration.
        """
        goal = self.require_current()
        if expected is not None:
            self._assert_revision(goal, expected)

        reason = reason.strip()
        if not reason:
            raise GoalError("a blocked goal needs a reason", code="GOAL_BLOCKED_NO_REASON")

        if not force:
            streak = self._consecutive_blocks.get(goal.id, 0) + 1
            self._consecutive_blocks[goal.id] = streak
            if streak < self.blocked_after:
                raise GoalError(
                    f"the same blocker has been reported {streak} time(s); "
                    f"{self.blocked_after} consecutive rounds are required before "
                    "a goal may be marked blocked",
                    code="GOAL_BLOCKED_TOO_EARLY",
                )
        else:
            self._consecutive_blocks[goal.id] = self.blocked_after

        from ..session.envelope import utc_now_iso

        snapshot = GoalSnapshot(
            id=goal.id,
            revision=goal.revision + 1,
            objective=goal.objective,
            phase=PHASE_BLOCKED,
            blocked_reason=reason,
            blocked_code=code,
            max_rounds=goal.snapshot.max_rounds,
            created_at=goal.snapshot.created_at,
            updated_at=utc_now_iso(),
        )
        self._write(snapshot, action="block", code=code)
        self._activation[goal.id] = ACTIVATION_DISARMED
        return self.require_current()

    def clear(self, *, expected: GoalRef | None = None) -> None:
        """Abandon the goal.

        Written as a completed goal with a tombstone objective rather than as a
        deletion: the log has no deletion, and a reader should be able to see
        that a goal once existed and was abandoned.
        """
        goal = self.current()
        if goal is None:
            raise GoalError("there is no goal to clear", code="GOAL_MISSING")
        if expected is not None:
            self._assert_revision(goal, expected)

        from ..session.envelope import utc_now_iso

        snapshot = GoalSnapshot(
            id=goal.id,
            revision=goal.revision + 1,
            objective=goal.objective,
            phase=PHASE_COMPLETE,
            blocked_reason="cleared",
            blocked_code="cleared",
            max_rounds=goal.snapshot.max_rounds,
            created_at=goal.snapshot.created_at,
            updated_at=utc_now_iso(),
        )
        self._write(snapshot, action="clear")
        self._activation[goal.id] = ACTIVATION_DISARMED

    # -- rounds ----------------------------------------------------------

    def start_round(self) -> int:
        """Record that a round is starting, and return its number.

        Counting events rather than storing a counter is what keeps
        ``rounds_started`` honest: it cannot disagree with the rounds that
        actually began.
        """
        goal = self.require_current()
        if not goal.may_continue:
            raise GoalError(
                f"the goal may not continue: phase={goal.phase}, "
                f"activation={goal.activation}, rounds={goal.rounds_started}/"
                f"{goal.snapshot.max_rounds}",
                code="GOAL_NOT_RUNNABLE",
            )

        number = goal.rounds_started + 1
        self.session.append(
            GOAL_ROUND,
            {
                "goal_id": goal.id,
                "revision": goal.revision,
                "round": number,
            },
        )
        return number

    def disarm(self, reason: str = "") -> None:
        """Stop automatic continuation for this process, without changing phase.

        Used when a round ends in a way that makes continuing unsafe -- a
        truncated response, for instance. The phase is untouched because the goal
        is still active; only this process stops driving it.
        """
        goal = self.current()
        if goal is None:
            return
        self._activation[goal.id] = ACTIVATION_DISARMED
        if reason:
            log.info("goal %s disarmed: %s", goal.id, reason)

    def arm(self) -> None:
        """Allow this process to continue the goal. Does not change the phase."""
        goal = self.current()
        if goal is None:
            return
        self._activation[goal.id] = ACTIVATION_ARMED

    # -- helpers ---------------------------------------------------------

    def _transition(
        self,
        goal: GoalView,
        phase: str,
        *,
        action: str,
        clear_block: bool = False,
    ) -> GoalView:
        from ..session.envelope import utc_now_iso

        snapshot = GoalSnapshot(
            id=goal.id,
            revision=goal.revision + 1,
            objective=goal.objective,
            phase=phase,
            blocked_reason="" if clear_block else goal.snapshot.blocked_reason,
            blocked_code="" if clear_block else goal.snapshot.blocked_code,
            max_rounds=goal.snapshot.max_rounds,
            created_at=goal.snapshot.created_at,
            updated_at=utc_now_iso(),
        )
        self._write(snapshot, action=action)
        return self.require_current()

    def _write(self, snapshot: GoalSnapshot, *, action: str, **extra: Any) -> None:
        self.session.append(
            GOAL_CHANGED,
            {"action": action, "goal": snapshot.to_record(), **extra},
        )

    @staticmethod
    def _assert_revision(goal: GoalView, expected: GoalRef) -> None:
        if expected.id != goal.id:
            raise GoalError(
                f"this is goal {goal.id!r}, not {expected.id!r}",
                code="GOAL_STALE",
            )
        if expected.revision != goal.revision:
            raise GoalError(
                f"the goal has moved on: you are editing revision "
                f"{expected.revision} but the current revision is {goal.revision}. "
                "Read the goal again and retry.",
                code="GOAL_STALE",
            )


def _new_goal_id(objective: str) -> str:
    """A short, stable identifier derived from the objective.

    Derived from the text plus a random suffix rather than from a counter, so two
    sessions that create goals independently cannot collide, and so the id is
    recognisable in a log without a lookup.
    """
    import hashlib
    import secrets

    digest = hashlib.sha256(objective.encode("utf-8")).hexdigest()[:8]
    return f"goal-{digest}-{secrets.token_hex(2)}"

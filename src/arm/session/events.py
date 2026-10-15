"""The event vocabulary: 19 durable events, grouped by scope.

Event names are the one thing that costs a data migration to change, so this
list is deliberately small and deliberately explicit. Names that are *reserved*
for later phases appear in :data:`RESERVED_PREFIXES` and are refused today, so
that adding subagents or teams later cannot collide with a name someone already
wrote into a log.

Three kinds of event exist here, and the distinction matters:

* **surface events** -- they participate in the model-visible message sequence
  and must declare how (``append`` or ``replace``).
* **log-only events** -- durable facts that never appear on the wire (turn
  boundaries, request headers, compaction bookkeeping).
* **projection events** -- surface-eligible, but their content comes from a
  registered projection rather than from the event body.

The surface/log split is what makes "append-only, rewrite-by-shadowing"
possible at all.
"""

from __future__ import annotations

from typing import Final, Literal

#: Bumped when the on-disk shape changes incompatibly. One version, no chain.
FORMAT_VERSION: Final[int] = 1

# ---------------------------------------------------------------------------
# Event names
# ---------------------------------------------------------------------------

SESSION_START: Final = "session/start"
SESSION_END_SEED: Final = "session/end-seed"

TURN_START: Final = "turn/start"
TURN_END: Final = "turn/end"

STEP_START: Final = "step/start"
STEP_END: Final = "step/end"

REQUEST_HEADER: Final = "request/header"
REQUEST_CONTEXT: Final = "request/context"

SYSTEM_MESSAGE: Final = "system/message"
USER_MESSAGE: Final = "user/message"
DEVELOPER_MESSAGE: Final = "developer/message"

ASSISTANT_MESSAGE: Final = "assistant/message"
ASSISTANT_ATTEMPT: Final = "assistant/attempt"

TOOL_CALL: Final = "tool/call"
TOOL_RESULT: Final = "tool/result"

COMPACTION_START: Final = "compaction/start"
COMPACTION_SUMMARY: Final = "compaction/summary"
COMPACTION_PRUNE: Final = "compaction/prune"
COMPACTION_END: Final = "compaction/end"

ERROR_TURN_FAILURE: Final = "error/turn-failure"

#: Plan mode is a whole-value replacement: the last one written wins. There is
#: no separate "exit" event, because an exit is just an activation with
#: ``active: false`` -- two event names for one state would be two things to keep
#: in sync.
PLAN_MODE: Final = "plan/mode"

#: Goal lifecycle. Each event carries the full snapshot plus a revision, so the
#: log is self-describing: a reader needs no prior state to interpret the latest
#: one.
GOAL_CHANGED: Final = "goal/changed"
#: A goal round was started, so ``roundsStarted`` can be counted from the log
#: rather than stored -- a counter in a snapshot would drift from the events.
GOAL_ROUND: Final = "goal/round"

#: The todo list, written as a whole-table replacement. It is log-only: the model
#: wrote it, so re-injecting it into the context would feed the model its own
#: draft back as though it were an instruction.
TODO_WRITE: Final = "todo/write"

#: Team events. All of them are written to the *lead's* session log, which is the
#: single authoritative store for a team: membership, the mailbox and the task
#: board are all projections of it. That is what makes a team resumable without a
#: second database, and what keeps the team's history replayable alongside the
#: conversation that produced it.
TEAM_MEMBER: Final = "team/member"
TEAM_MESSAGE_QUEUED: Final = "team/message/queued"
TEAM_MESSAGE_DELIVERED: Final = "team/message/delivered"
TEAM_TASK: Final = "team/task"

#: The 19 durable events this build understands.
KNOWN_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        SESSION_START,
        SESSION_END_SEED,
        TURN_START,
        TURN_END,
        STEP_START,
        STEP_END,
        REQUEST_HEADER,
        REQUEST_CONTEXT,
        SYSTEM_MESSAGE,
        USER_MESSAGE,
        DEVELOPER_MESSAGE,
        ASSISTANT_MESSAGE,
        ASSISTANT_ATTEMPT,
        TOOL_CALL,
        TOOL_RESULT,
        COMPACTION_START,
        COMPACTION_SUMMARY,
        COMPACTION_PRUNE,
        COMPACTION_END,
        ERROR_TURN_FAILURE,
        PLAN_MODE,
        GOAL_CHANGED,
        GOAL_ROUND,
        TODO_WRITE,
        TEAM_MEMBER,
        TEAM_MESSAGE_QUEUED,
        TEAM_MESSAGE_DELIVERED,
        TEAM_TASK,
    }
)

# ---------------------------------------------------------------------------
# Surface eligibility
# ---------------------------------------------------------------------------

#: Events that carry a position in the model-visible message sequence.
#:
#: ``tool/call`` is deliberately absent: a call is durable bookkeeping that is
#: paired with its result. Only the *result* occupies a surface position, which
#: is what keeps pairing trivially checkable.
SURFACE_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        SYSTEM_MESSAGE,
        USER_MESSAGE,
        DEVELOPER_MESSAGE,
        ASSISTANT_MESSAGE,
        TOOL_RESULT,
    }
)

#: Surface events whose content must come from a registered projection.
#: Without a definition these are refused rather than dropped -- silently
#: losing content would change what the model sees without saying so.
PROJECTION_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        SYSTEM_MESSAGE,
        USER_MESSAGE,
        DEVELOPER_MESSAGE,
        ASSISTANT_MESSAGE,
        TOOL_RESULT,
    }
)

#: Names reserved for later phases. Refused now so that no log written today can
#: contain a name we later give a different meaning.
RESERVED_PREFIXES: Final[tuple[str, ...]] = (
    "subagent/",
    "feedback/",
    "hook/",
    "telemetry/",
    "image/",
    "session-log-",
    "skill/",
    "mcp/",
    "schedule/",
    "job/",
)

#: Message roles that occupy a surface position.
MessageRole = Literal["system", "user", "developer", "assistant", "tool"]


def is_known(event_type: str) -> bool:
    return event_type in KNOWN_EVENT_TYPES


def is_surface_eligible(event_type: str) -> bool:
    return event_type in SURFACE_EVENT_TYPES


def needs_projection(event_type: str) -> bool:
    return event_type in PROJECTION_EVENT_TYPES


def is_reserved(event_type: str) -> bool:
    return event_type.startswith(RESERVED_PREFIXES)


def validate_event_type(event_type: str) -> None:
    """Reject names this build must not accept.

    A reserved name is refused with a message that says *why*: it means a newer
    build wrote this log, and pretending to understand it would be worse than
    refusing.
    """
    if is_known(event_type):
        return
    if is_reserved(event_type):
        raise ValueError(
            f"event type {event_type!r} is reserved for a later format generation; "
            "this build cannot represent it"
        )


# ---------------------------------------------------------------------------
# Turn termination reasons
# ---------------------------------------------------------------------------

#: Why a turn ended. Every exit path writes exactly one of these.
TURN_COMPLETED: Final = "completed"
TURN_BLOCKED: Final = "blocked"
TURN_ERROR: Final = "error"
TURN_ABORTED: Final = "aborted"
#: The model hit its output limit mid-answer. A distinct reason because the
#: answer is *incomplete*: reporting ``completed`` would let a truncated reply be
#: treated as a finished one, and a caller would build on half a result.
TURN_MAX_TOKENS: Final = "max-tokens"

TURN_END_REASONS: Final[frozenset[str]] = frozenset(
    {TURN_COMPLETED, TURN_BLOCKED, TURN_ERROR, TURN_ABORTED, TURN_MAX_TOKENS}
)

#: Typed cancellation causes. Only these reach the log, so a cause can never
#: smuggle a non-serializable object into a durable event.
CANCEL_USER: Final = "user"
CANCEL_DISPOSED: Final = "disposed"
CANCEL_TIMEOUT: Final = "timeout"
CANCEL_SHUTDOWN: Final = "shutdown"

CANCEL_CAUSES: Final[frozenset[str]] = frozenset(
    {CANCEL_USER, CANCEL_DISPOSED, CANCEL_TIMEOUT, CANCEL_SHUTDOWN}
)

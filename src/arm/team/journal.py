"""The team journal: one log, three projections, one writer.

A team's entire state -- who is on it, what has been said to whom, and what work
is outstanding -- lives in the **lead's session log**. There is no second
database. That choice is what makes a team resumable: recovery is folding the
same events the conversation already recorded, and the team's history stays
replayable next to the messages that produced it.

Three projections are folded from the same events:

* **roster** -- membership and lifecycle phase;
* **mailbox** -- queued and delivered messages, with delivery tracked per
  recipient so a message is never delivered twice;
* **task board** -- a DAG with compare-and-set revisions.

Two rules hold across all three, and both exist because the alternative is a
state that cannot be recovered:

**Every write is an append, and every write goes through one journal.** A
projection that wrote directly would be a second source of truth, and the two
would diverge the first time one of them failed.

**Delivery is recorded by the recipient, not the sender.** A sender that
believed it delivered a message that the recipient never recorded would lose it
silently; the confirmation has to come from the side that actually holds it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..kernel.errors import ArmError
from ..session.events import (
    TEAM_MEMBER,
    TEAM_MESSAGE_DELIVERED,
    TEAM_MESSAGE_QUEUED,
    TEAM_TASK,
)
from ..session.session import Session

log = logging.getLogger("arm.team")

#: Member lifecycle. ``provisioning`` exists because creating a teammate is not
#: instantaneous: the member is recorded before it exists, so a crash leaves
#: evidence that reconciliation can act on rather than a member that appears
#: active but has no agent behind it.
MEMBER_PROVISIONING = "provisioning"
MEMBER_ACTIVE = "active"
MEMBER_FAILED = "failed"
MEMBER_STOPPED = "stopped"

MEMBER_PHASES = frozenset(
    {MEMBER_PROVISIONING, MEMBER_ACTIVE, MEMBER_FAILED, MEMBER_STOPPED}
)

#: The name reserved for the lead. A member cannot take it, because the lead is
#: the one who owns the log.
LEAD_NAME = "lead"

#: Task lifecycle.
TASK_PENDING = "pending"
TASK_IN_PROGRESS = "in_progress"
TASK_COMPLETED = "completed"
TASK_DELETED = "deleted"

TASK_PHASES = frozenset(
    {TASK_PENDING, TASK_IN_PROGRESS, TASK_COMPLETED, TASK_DELETED}
)

#: Limits, each validated as a positive integer at construction. A limit of zero
#: or a negative one would silently mean "nothing is allowed" or "anything goes",
#: and neither is what a caller writing `max_members=0` intended.
DEFAULT_MAX_MEMBERS = 16
DEFAULT_MAX_TASKS = 256
DEFAULT_MAX_PENDING_PER_MEMBER = 64
DEFAULT_MAX_MESSAGE_BYTES = 65_536

#: Task field ceilings, matching the mailbox's byte budget in spirit: a task is
#: read by a teammate with no more context than the text it carries.
MAX_SUBJECT_CHARS = 200
MAX_DESCRIPTION_CHARS = 16_384

#: Member names are kebab-case, like skill names and for the same reason: the
#: name is what a teammate types, so its shape is enforced rather than guessed.
import re

MEMBER_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class TeamError(ArmError):
    """A team operation was refused."""

    def __init__(self, message: str, *, code: str = "TEAM_INVALID") -> None:
        super().__init__(message)
        self.code = code


def _positive(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TeamError(f"{name} must be a positive integer, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamMember:
    """One teammate."""

    name: str
    phase: str = MEMBER_PROVISIONING
    provider: str = "in-process"
    label: str = ""
    child_session: str = ""
    error: str = ""

    @property
    def active(self) -> bool:
        return self.phase == MEMBER_ACTIVE

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "phase": self.phase,
            "provider": self.provider,
        }
        if self.label:
            record["label"] = self.label
        if self.child_session:
            record["child_session"] = self.child_session
        if self.error:
            record["error"] = self.error
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TeamMember":
        return cls(
            name=str(record.get("name", "")),
            phase=str(record.get("phase", MEMBER_PROVISIONING)),
            provider=str(record.get("provider", "in-process")),
            label=str(record.get("label", "")),
            child_session=str(record.get("child_session", "")),
            error=str(record.get("error", "")),
        )


def validate_member_name(name: str) -> None:
    """Refuse a name that is unusable or reserved."""
    if not name:
        raise TeamError("a member needs a name", code="TEAM_INVALID_MEMBER_NAME")
    if name == LEAD_NAME:
        raise TeamError(
            f"{LEAD_NAME!r} is reserved for the session that owns the team",
            code="TEAM_INVALID_MEMBER_NAME",
        )
    if len(name) > 64:
        raise TeamError(
            f"member name {name!r} is longer than 64 characters",
            code="TEAM_INVALID_MEMBER_NAME",
        )
    if not MEMBER_NAME.match(name):
        raise TeamError(
            f"member name {name!r} is not kebab-case (lowercase letters, digits "
            "and single hyphens)",
            code="TEAM_INVALID_MEMBER_NAME",
        )


# ---------------------------------------------------------------------------
# Mailbox
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamMessage:
    """One message, from one member to another."""

    message_id: str
    sender: str
    recipient: str
    body: str
    delivered: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "body": self.body,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TeamMessage":
        return cls(
            message_id=str(record.get("message_id", "")),
            sender=str(record.get("sender", "")),
            recipient=str(record.get("recipient", "")),
            body=str(record.get("body", "")),
        )


# ---------------------------------------------------------------------------
# Task board
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamTask:
    """One task on the board.

    ``revision`` is a compare-and-set token, not a version history. A caller must
    present the revision it read, so two members editing the same task cannot
    silently overwrite each other -- the second one is told its view is stale.
    """

    id: str
    revision: int
    subject: str
    description: str = ""
    phase: str = TASK_PENDING
    owner: str = ""
    blocked_by: tuple[str, ...] = ()
    write_scopes: tuple[str, ...] = ()
    created_by: str = ""

    @property
    def deleted(self) -> bool:
        return self.phase == TASK_DELETED

    @property
    def completed(self) -> bool:
        return self.phase == TASK_COMPLETED

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "revision": self.revision,
            "subject": self.subject,
            "phase": self.phase,
        }
        if self.description:
            record["description"] = self.description
        if self.owner:
            record["owner"] = self.owner
        if self.blocked_by:
            record["blocked_by"] = list(self.blocked_by)
        if self.write_scopes:
            record["write_scopes"] = list(self.write_scopes)
        if self.created_by:
            record["created_by"] = self.created_by
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TeamTask":
        return cls(
            id=str(record.get("id", "")),
            revision=int(record.get("revision", 1)),
            subject=str(record.get("subject", "")),
            description=str(record.get("description", "")),
            phase=str(record.get("phase", TASK_PENDING)),
            owner=str(record.get("owner", "")),
            blocked_by=tuple(str(b) for b in (record.get("blocked_by") or ())),
            write_scopes=tuple(str(s) for s in (record.get("write_scopes") or ())),
            created_by=str(record.get("created_by", "")),
        )


def assert_task_graph(
    candidate: TeamTask, existing: Mapping[str, TeamTask]
) -> None:
    """Validate a task's dependencies before it is written.

    Four checks, each of which catches a way a board becomes unusable:

    * a task cannot block itself;
    * it cannot list the same blocker twice;
    * a blocker must exist and not be deleted -- a dependency on something that
      is gone can never be satisfied, so the task would be permanently stuck;
    * the graph must stay acyclic. A cycle is not merely bad style: every task in
      it becomes permanently unclaimable, and nothing in the system would report
      why.

    The cycle check runs over the *candidate* graph rather than the stored one,
    so a change that would introduce a cycle is refused before it is written.
    """
    if candidate.id in candidate.blocked_by:
        raise TeamError(
            f"task {candidate.id!r} cannot block itself",
            code="TEAM_INVALID_ARGUMENT",
        )

    seen: set[str] = set()
    for blocker in candidate.blocked_by:
        if blocker in seen:
            raise TeamError(
                f"task {candidate.id!r} lists {blocker!r} as a blocker twice",
                code="TEAM_INVALID_ARGUMENT",
            )
        seen.add(blocker)
        target = existing.get(blocker)
        if target is None:
            raise TeamError(
                f"task {candidate.id!r} depends on {blocker!r}, which does not exist",
                code="TEAM_TASK_NOT_FOUND",
            )
        if target.deleted:
            raise TeamError(
                f"task {candidate.id!r} depends on {blocker!r}, which is deleted",
                code="TEAM_TASK_NOT_FOUND",
            )

    # Depth-first cycle detection over the candidate graph.
    graph: dict[str, tuple[str, ...]] = {
        task.id: task.blocked_by for task in existing.values() if not task.deleted
    }
    graph[candidate.id] = candidate.blocked_by

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, path: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle = " -> ".join((*path, node))
            raise TeamError(
                f"this change would create a dependency cycle: {cycle}",
                code="TEAM_TASK_DEPENDENCY_CYCLE",
            )
        visiting.add(node)
        for next_node in graph.get(node, ()):
            visit(next_node, (*path, node))
        visiting.discard(node)
        visited.add(node)

    visit(candidate.id, ())


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------


@dataclass
class TeamJournal:
    """Reads and writes a team's state in one session's log.

    Every operation folds the current state from the log, validates against it,
    and appends. There is no in-memory cache to drift: a team changes rarely
    enough that folding is cheap, and correctness-by-construction is worth more
    than the microseconds.
    """

    session: Session
    root_id: str = ""
    max_members: int = DEFAULT_MAX_MEMBERS
    max_tasks: int = DEFAULT_MAX_TASKS
    max_pending_per_member: int = DEFAULT_MAX_PENDING_PER_MEMBER
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES

    def __post_init__(self) -> None:
        _positive(self.max_members, "max_members")
        _positive(self.max_tasks, "max_tasks")
        _positive(self.max_pending_per_member, "max_pending_per_member")
        _positive(self.max_message_bytes, "max_message_bytes")
        if not self.root_id:
            self.root_id = self.session.session_id

    # -- writing ---------------------------------------------------------

    def _append(self, event_type: str, data: Mapping[str, Any]) -> None:
        self.session.append(event_type, {"root": self.root_id, **data})

    # -- roster ----------------------------------------------------------

    def _mine(self, event: Any) -> bool:
        """Whether an event belongs to this journal's root.

        A session log may carry more than one team -- a second journal over the
        same session is a legitimate way to run two independent efforts -- and
        without this filter each would see the other's members, messages and
        tasks. The failure is silent: the boards would simply be merged.
        """
        root = event.data.get("root", "")
        return not root or root == self.root_id

    def members(self) -> dict[str, TeamMember]:
        """The roster, folded from the log. The last record for a name wins."""
        roster: dict[str, TeamMember] = {}
        for event in self.session:
            if event.type != TEAM_MEMBER or not self._mine(event):
                continue
            record = event.data.get("member")
            if isinstance(record, Mapping):
                member = TeamMember.from_record(record)
                roster[member.name] = member
        return roster

    def active_members(self) -> dict[str, TeamMember]:
        return {n: m for n, m in self.members().items() if m.active}

    def add_member(self, member: TeamMember) -> None:
        """Record a member in the ``provisioning`` phase.

        Written *before* the member's agent exists. A crash between the two
        leaves a member that reconciliation can find and resolve; the reverse
        order would leave an agent that nothing knows about.
        """
        validate_member_name(member.name)
        roster = self.members()
        if member.name in roster and roster[member.name].phase != MEMBER_FAILED:
            raise TeamError(
                f"a member named {member.name!r} already exists",
                code="TEAM_MEMBER_NAME_TAKEN",
            )
        active = sum(1 for m in roster.values() if m.phase != MEMBER_FAILED)
        if active >= self.max_members:
            raise TeamError(
                f"the team already has {active} members (limit {self.max_members})",
                code="TEAM_MEMBER_LIMIT",
            )
        self._append(TEAM_MEMBER, {"member": member.to_record()})

    def set_member_phase(
        self, name: str, phase: str, *, error: str = "", child_session: str = ""
    ) -> TeamMember:
        """Move a member to a new phase."""
        if phase not in MEMBER_PHASES:
            raise TeamError(f"unknown member phase {phase!r}")
        roster = self.members()
        current = roster.get(name)
        if current is None:
            raise TeamError(f"no member named {name!r}", code="TEAM_MEMBER_NOT_FOUND")
        updated = TeamMember(
            name=current.name,
            phase=phase,
            provider=current.provider,
            label=current.label,
            child_session=child_session or current.child_session,
            error=error,
        )
        self._append(TEAM_MEMBER, {"member": updated.to_record()})
        return updated

    # -- mailbox ---------------------------------------------------------

    def messages(self) -> tuple[TeamMessage, ...]:
        """Every message, with delivery folded in.

        Delivery is a separate event because it is recorded by the *recipient*:
        a sender that believed it delivered a message the recipient never
        recorded would lose it silently.
        """
        queued: list[TeamMessage] = []
        delivered: set[str] = set()
        for event in self.session:
            if not self._mine(event):
                continue
            if event.type == TEAM_MESSAGE_QUEUED:
                record = event.data.get("message")
                if isinstance(record, Mapping):
                    queued.append(TeamMessage.from_record(record))
            elif event.type == TEAM_MESSAGE_DELIVERED:
                delivered.add(str(event.data.get("message_id", "")))
        return tuple(
            TeamMessage(
                message_id=m.message_id,
                sender=m.sender,
                recipient=m.recipient,
                body=m.body,
                delivered=m.message_id in delivered,
            )
            for m in queued
        )

    def pending_for(self, recipient: str) -> tuple[TeamMessage, ...]:
        return tuple(
            m for m in self.messages() if m.recipient == recipient and not m.delivered
        )

    def send(self, message: TeamMessage) -> None:
        """Queue a message, validating the mailbox's limits first."""
        if not message.body:
            raise TeamError("a message needs a body", code="TEAM_INVALID_ARGUMENT")
        if message.sender == message.recipient:
            raise TeamError(
                f"{message.sender!r} cannot send a message to itself",
                code="TEAM_SELF_MESSAGE",
            )

        roster = self.members()
        recipient = roster.get(message.recipient)
        if recipient is None or not recipient.active:
            raise TeamError(
                f"no active member named {message.recipient!r}",
                code="TEAM_MEMBER_NOT_FOUND",
            )

        # The size is measured in *bytes*, because the limit is a transport
        # limit and a multi-byte character is not one byte.
        size = len(message.body.encode("utf-8"))
        if size > self.max_message_bytes:
            raise TeamError(
                f"the message is {size} bytes, over the limit of "
                f"{self.max_message_bytes}",
                code="TEAM_MESSAGE_TOO_LARGE",
            )

        pending = len(self.pending_for(message.recipient))
        if pending >= self.max_pending_per_member:
            raise TeamError(
                f"{message.recipient!r} already has {pending} undelivered messages "
                f"(limit {self.max_pending_per_member})",
                code="TEAM_MAILBOX_FULL",
            )

        existing = {m.message_id for m in self.messages()}
        if message.message_id in existing:
            raise TeamError(
                f"message {message.message_id!r} was already queued",
                code="TEAM_DUPLICATE_MESSAGE",
            )

        self._append(TEAM_MESSAGE_QUEUED, {"message": message.to_record()})

    def confirm_delivery(self, message_id: str, recipient: str) -> bool:
        """Record that the recipient holds a message.

        Called only after the recipient's own log confirms it, so the record
        reflects a fact rather than an intention. Idempotent: confirming twice
        is not an error, because a retry after a crash is expected.
        """
        for message in self.messages():
            if message.message_id != message_id:
                continue
            if message.recipient != recipient:
                raise TeamError(
                    f"message {message_id!r} was addressed to "
                    f"{message.recipient!r}, not {recipient!r}",
                    code="TEAM_INVALID_ARGUMENT",
                )
            if message.delivered:
                return False
            self._append(
                TEAM_MESSAGE_DELIVERED,
                {"message_id": message_id, "recipient": recipient},
            )
            return True
        # A message error, not a task error: the code is what a caller routes on,
        # and a caller told "task not found" would look in the wrong place.
        raise TeamError(
            f"no message {message_id!r} was ever queued",
            code="TEAM_MESSAGE_NOT_FOUND",
        )

    # -- task board ------------------------------------------------------

    def tasks(self) -> dict[str, TeamTask]:
        """The board, folded from the log. The last record for an id wins."""
        board: dict[str, TeamTask] = {}
        for event in self.session:
            if event.type != TEAM_TASK or not self._mine(event):
                continue
            record = event.data.get("task")
            if isinstance(record, Mapping):
                task = TeamTask.from_record(record)
                board[task.id] = task
        return board

    def live_tasks(self) -> dict[str, TeamTask]:
        return {i: t for i, t in self.tasks().items() if not t.deleted}

    def create_task(
        self, *, subject: str, description: str = "", created_by: str = "",
        blocked_by: Sequence[str] = (), write_scopes: Sequence[str] = (),
    ) -> TeamTask:
        """Add a task to the board."""
        subject = subject.strip()
        if not subject:
            raise TeamError("a task needs a subject", code="TEAM_INVALID_ARGUMENT")
        if len(subject) > MAX_SUBJECT_CHARS:
            raise TeamError(
                f"the subject is {len(subject)} characters, over the limit of "
                f"{MAX_SUBJECT_CHARS}",
                code="TEAM_INVALID_ARGUMENT",
            )
        if len(description) > MAX_DESCRIPTION_CHARS:
            raise TeamError(
                f"the description is {len(description)} characters, over the "
                f"limit of {MAX_DESCRIPTION_CHARS}",
                code="TEAM_INVALID_ARGUMENT",
            )

        board = self.tasks()
        live = {i: t for i, t in board.items() if not t.deleted}
        if len(live) >= self.max_tasks:
            raise TeamError(
                f"the board already has {len(live)} tasks (limit {self.max_tasks})",
                code="TEAM_TASK_LIMIT",
            )

        # Ids are allocated by scanning for the first free one among *live*
        # tasks. Deleted tasks release their id, because a deleted task is gone
        # from the caller's point of view -- and refusing to reuse its number
        # would let a board of deleted tasks permanently refuse new work while
        # sitting well under its own limit.
        #
        # History is still unambiguous: a task id is always interpreted against
        # the board as it stands, and a deleted id is not resolvable at all.
        used = {
            int(task_id.split("-", 1)[1])
            for task_id in board
            if not board[task_id].deleted and task_id.startswith("task-")
            and task_id.split("-", 1)[1].isdigit()
        }
        next_number = 1
        while next_number in used:
            next_number += 1
            if next_number > self.max_tasks:
                raise TeamError(
                    "no task id is available; the board is at its limit",
                    code="TEAM_TASK_LIMIT",
                )
        task_id = f"task-{next_number}"

        candidate = TeamTask(
            id=task_id,
            revision=1,
            subject=subject,
            description=description,
            phase=TASK_PENDING,
            # ``blocked_by`` is passed through unchanged so ``assert_task_graph``
            # can see a duplicate and refuse it. Deduplicating here would make
            # that check unreachable -- the caller's mistake would be silently
            # corrected, and the validation would be dead code that looks live.
            #
            # ``write_scopes`` *is* deduplicated, because it is advisory metadata
            # with no validation claiming otherwise.
            blocked_by=tuple(str(b) for b in blocked_by),
            write_scopes=tuple(dict.fromkeys(str(s) for s in write_scopes)),
            created_by=created_by,
        )
        assert_task_graph(candidate, board)
        self._append(TEAM_TASK, {"task": candidate.to_record(), "action": "create"})
        return candidate

    def task_ready(self, task: TeamTask) -> bool:
        """Whether every blocker is complete.

        A blocker that is *deleted* makes the task permanently unready, which is
        why deletion is refused while dependents exist.
        """
        board = self.tasks()
        for blocker in task.blocked_by:
            target = board.get(blocker)
            if target is None or not target.completed:
                return False
        return True

    def update_task(
        self,
        task_id: str,
        *,
        expected_revision: int,
        actor: str,
        is_lead: bool = False,
        action: str,
        **changes: Any,
    ) -> TeamTask:
        """Change a task, with compare-and-set on its revision.

        The revision check is what turns a concurrent edit into an error the
        caller can act on -- "read it again" -- rather than a silent overwrite of
        a decision it never saw.
        """
        board = self.tasks()
        current = board.get(task_id)
        if current is None:
            raise TeamError(f"no task {task_id!r}", code="TEAM_TASK_NOT_FOUND")
        if current.deleted:
            raise TeamError(
                f"task {task_id!r} is deleted", code="TEAM_TASK_DELETED"
            )
        if expected_revision != current.revision:
            raise TeamError(
                f"task {task_id!r} has moved on: you are editing revision "
                f"{expected_revision} but the current revision is "
                f"{current.revision}. Read it again and retry.",
                code="TEAM_TASK_STALE_REVISION",
            )

        updated = self._apply_task_action(
            current, action=action, actor=actor, is_lead=is_lead, changes=changes
        )
        assert_task_graph(updated, board)
        self._append(
            TEAM_TASK,
            {"task": updated.to_record(), "action": action, "actor": actor},
        )
        return updated

    def _apply_task_action(
        self,
        current: TeamTask,
        *,
        action: str,
        actor: str,
        is_lead: bool,
        changes: Mapping[str, Any],
    ) -> TeamTask:
        """Compute the new task state for one action.

        Each action declares its own preconditions, and the authorization rule is
        stated once: a task may be changed by its owner or by the lead, and
        ``reassign`` is the lead's alone because it moves work between people.
        """
        def next_revision() -> int:
            return current.revision + 1

        def owned() -> bool:
            return is_lead or (bool(current.owner) and current.owner == actor)

        if action == "claim":
            if current.owner and current.owner != actor:
                raise TeamError(
                    f"task {current.id!r} is already claimed by {current.owner!r}",
                    code="TEAM_TASK_ALREADY_CLAIMED",
                )
            if current.phase != TASK_PENDING:
                raise TeamError(
                    f"task {current.id!r} is {current.phase}, not pending",
                    code="TEAM_INVALID_ARGUMENT",
                )
            if not self.task_ready(current):
                blockers = [
                    b
                    for b in current.blocked_by
                    if not (self.tasks().get(b) or TeamTask("", 0, "")).completed
                ]
                raise TeamError(
                    f"task {current.id!r} is blocked by {blockers}",
                    code="TEAM_TASK_BLOCKED",
                )
            return TeamTask(
                **{**_as_dict(current), "phase": TASK_IN_PROGRESS,
                   "owner": actor, "revision": next_revision()}
            )

        if action == "release":
            if not owned():
                raise TeamError(
                    f"only the owner of {current.id!r} or the lead may release it",
                    code="TEAM_TASK_UNAUTHORIZED",
                )
            if current.phase != TASK_IN_PROGRESS:
                raise TeamError(
                    f"task {current.id!r} is {current.phase}, not in progress",
                    code="TEAM_INVALID_ARGUMENT",
                )
            return TeamTask(
                **{**_as_dict(current), "phase": TASK_PENDING,
                   "owner": "", "revision": next_revision()}
            )

        if action == "complete":
            if not owned():
                raise TeamError(
                    f"only the owner of {current.id!r} or the lead may complete it",
                    code="TEAM_TASK_UNAUTHORIZED",
                )
            if current.phase != TASK_IN_PROGRESS:
                raise TeamError(
                    f"task {current.id!r} is {current.phase}, not in progress",
                    code="TEAM_INVALID_ARGUMENT",
                )
            return TeamTask(
                **{**_as_dict(current), "phase": TASK_COMPLETED,
                   "revision": next_revision()}
            )

        if action == "reopen":
            if not owned():
                raise TeamError(
                    f"only the owner of {current.id!r} or the lead may reopen it",
                    code="TEAM_TASK_UNAUTHORIZED",
                )
            if current.phase != TASK_COMPLETED:
                raise TeamError(
                    f"task {current.id!r} is {current.phase}, not completed",
                    code="TEAM_INVALID_ARGUMENT",
                )
            # A dependent that was claimed while this task was complete is now
            # working on a task whose blocker is no longer done. Reopening
            # therefore has to release them: leaving them in progress means the
            # board reports work as unblocked that the graph says is blocked, and
            # the two would disagree with no way to tell which is right.
            for other in self.live_tasks().values():
                if current.id not in other.blocked_by:
                    continue
                if other.phase != TASK_IN_PROGRESS:
                    continue
                released = TeamTask(
                    **{
                        **_as_dict(other),
                        "phase": TASK_PENDING,
                        "owner": "",
                        "revision": other.revision + 1,
                    }
                )
                self._append(
                    TEAM_TASK,
                    {
                        "task": released.to_record(),
                        "action": "reopen-unblock",
                        "actor": actor,
                    },
                )
            return TeamTask(
                **{**_as_dict(current), "phase": TASK_IN_PROGRESS,
                   "revision": next_revision()}
            )

        if action == "edit":
            subject = str(changes.get("subject", "")).strip()
            description = changes.get("description")
            if not subject and description is None:
                raise TeamError(
                    "an edit must change the subject or the description",
                    code="TEAM_INVALID_ARGUMENT",
                )
            if not owned():
                raise TeamError(
                    f"only the owner of {current.id!r} or the lead may edit it",
                    code="TEAM_TASK_UNAUTHORIZED",
                )
            if subject and len(subject) > MAX_SUBJECT_CHARS:
                raise TeamError("the subject is too long", code="TEAM_INVALID_ARGUMENT")
            if description is not None and len(str(description)) > MAX_DESCRIPTION_CHARS:
                raise TeamError(
                    "the description is too long", code="TEAM_INVALID_ARGUMENT"
                )
            return TeamTask(
                **{
                    **_as_dict(current),
                    "subject": subject or current.subject,
                    "description": (
                        current.description if description is None else str(description)
                    ),
                    "revision": next_revision(),
                }
            )

        if action == "set_dependencies":
            if "blocked_by" not in changes:
                raise TeamError(
                    "set_dependencies needs a blocked_by list",
                    code="TEAM_INVALID_ARGUMENT",
                )
            if not owned():
                raise TeamError(
                    f"only the owner of {current.id!r} or the lead may change its "
                    "dependencies",
                    code="TEAM_TASK_UNAUTHORIZED",
                )
            return TeamTask(
                **{
                    **_as_dict(current),
                    # Passed through so the duplicate check can see it.
                    "blocked_by": tuple(str(b) for b in changes["blocked_by"]),
                    "revision": next_revision(),
                }
            )

        if action == "reassign":
            if not is_lead:
                raise TeamError(
                    "only the lead may reassign a task",
                    code="TEAM_LEAD_REQUIRED",
                )
            if current.phase not in (TASK_PENDING, TASK_IN_PROGRESS):
                raise TeamError(
                    f"task {current.id!r} is {current.phase} and cannot be reassigned",
                    code="TEAM_INVALID_ARGUMENT",
                )
            target = str(changes.get("owner", "")).strip()
            if not target:
                # An empty owner returns the task to the pool.
                return TeamTask(
                    **{**_as_dict(current), "phase": TASK_PENDING, "owner": "",
                       "revision": next_revision()}
                )
            if target not in self.active_members():
                raise TeamError(
                    f"{target!r} is not an active member",
                    code="TEAM_MEMBER_NOT_FOUND",
                )
            if not self.task_ready(current):
                raise TeamError(
                    f"task {current.id!r} is still blocked", code="TEAM_TASK_BLOCKED"
                )
            return TeamTask(
                **{**_as_dict(current), "phase": TASK_IN_PROGRESS, "owner": target,
                   "revision": next_revision()}
            )

        if action == "delete":
            if not owned():
                raise TeamError(
                    f"only the owner of {current.id!r} or the lead may delete it",
                    code="TEAM_TASK_UNAUTHORIZED",
                )
            # A deleted task can never satisfy a dependency, so deleting one that
            # others wait on would strand them permanently.
            dependents = [
                t.id
                for t in self.live_tasks().values()
                if current.id in t.blocked_by
            ]
            if dependents:
                raise TeamError(
                    f"task {current.id!r} is a blocker for {dependents}; remove "
                    "those dependencies first",
                    code="TEAM_TASK_HAS_DEPENDENTS",
                )
            return TeamTask(
                **{**_as_dict(current), "phase": TASK_DELETED,
                   "revision": next_revision()}
            )

        raise TeamError(
            f"unknown task action {action!r}; expected one of: claim, release, "
            "complete, reopen, edit, set_dependencies, reassign, delete",
            code="TEAM_INVALID_ARGUMENT",
        )

    # -- readiness -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """A compact view, for the model and for a caller checking progress."""
        roster = self.members()
        board = self.live_tasks()
        return {
            "members": {
                "total": len(roster),
                "active": sum(1 for m in roster.values() if m.active),
                "provisioning": sum(
                    1 for m in roster.values() if m.phase == MEMBER_PROVISIONING
                ),
                "failed": sum(1 for m in roster.values() if m.phase == MEMBER_FAILED),
            },
            "tasks": {
                "total": len(board),
                "pending": sum(1 for t in board.values() if t.phase == TASK_PENDING),
                "in_progress": sum(
                    1 for t in board.values() if t.phase == TASK_IN_PROGRESS
                ),
                "completed": sum(
                    1 for t in board.values() if t.phase == TASK_COMPLETED
                ),
                "ready": sum(
                    1
                    for t in board.values()
                    if t.phase == TASK_PENDING and self.task_ready(t)
                ),
            },
            "mailbox": {
                "queued": sum(1 for m in self.messages() if not m.delivered),
                "delivered": sum(1 for m in self.messages() if m.delivered),
            },
        }


def _as_dict(task: TeamTask) -> dict[str, Any]:
    """The task's fields as a dict, for building a modified copy."""
    return {
        "id": task.id,
        "revision": task.revision,
        "subject": task.subject,
        "description": task.description,
        "phase": task.phase,
        "owner": task.owner,
        "blocked_by": task.blocked_by,
        "write_scopes": task.write_scopes,
        "created_by": task.created_by,
    }

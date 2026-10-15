"""Todos: a task list the model keeps for itself.

The data model is deliberately tiny -- ``content`` and ``status``, no id, no
priority, no ordering field -- because the list is **replaced as a whole** on
every write. Entries do not need stable identity if nothing ever refers to them
individually; adding ids would invite references that the whole-table replacement
would then invalidate.

The most consequential decision is what this module **does not do**:

    the todo list is never injected into the prompt.

That is not an oversight. The list is the model's own scratchpad. Feeding it back
into the context on every turn would spend tokens restating something the model
just wrote, and -- worse -- would present the model's own draft as though it were
an instruction from the user. What the model actually sees is what it already
has: the tool call it made, with the full list as an argument, and one line of
result. The full rendering is a *reader's* concern, which is why the projection
exists.

Two smaller decisions:

**``in_progress`` is limited to one by default.** Several tasks marked in-progress
at once is usually a sign that the list is being used as a wish list rather than
a plan. The limit is configurable rather than absolute, because parallel work is
sometimes genuinely parallel.

**A write is refused outside an agent session.** The list belongs to a session's
log; a caller with no session has nowhere to put it, and silently accepting the
write would lose it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..kernel.errors import ArmError
from ..session.events import TODO_WRITE, TURN_START
from ..session.session import Session

log = logging.getLogger("arm.todo")

#: The three states a task can be in.
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"

STATUSES = (STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED)


class TodoError(ArmError):
    """A todo write was refused."""

    def __init__(self, message: str, *, code: str = "TODO_INVALID") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class TodoItem:
    """One task. No identity, because the table is replaced wholesale."""

    content: str
    status: str = STATUS_PENDING

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise TodoError(
                f"unknown status {self.status!r}; expected one of {list(STATUSES)}",
                code="TODO_INVALID_STATUS",
            )

    def to_record(self) -> dict[str, str]:
        return {"content": self.content, "status": self.status}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TodoItem":
        return cls(
            content=str(record.get("content", "")),
            status=str(record.get("status", STATUS_PENDING)),
        )


@dataclass(frozen=True, slots=True)
class TodoSummary:
    """Counts by status, for the one line the model sees back."""

    pending: int = 0
    in_progress: int = 0
    completed: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.in_progress + self.completed

    def render(self) -> str:
        return (
            f"Updated todo list: {self.pending} pending, "
            f"{self.in_progress} in progress, {self.completed} completed."
        )


def validate(
    items: Sequence[TodoItem], *, allow_parallel_in_progress: bool = False
) -> list[TodoItem]:
    """Validate a proposed list, returning the normalized form.

    Three rules, each of which catches a real way a list goes wrong:

    * content must be non-empty after stripping -- an empty task is not a task;
    * content must be unique -- a duplicated entry makes "which one is done?"
      unanswerable, and the whole-table replacement means there is no id to
      disambiguate with;
    * at most one task may be in progress unless parallel work is enabled.
    """
    normalized: list[TodoItem] = []
    seen: set[str] = set()

    for index, item in enumerate(items):
        content = item.content.strip()
        if not content:
            raise TodoError(
                f"item {index} has empty content", code="TODO_EMPTY_CONTENT"
            )
        if content in seen:
            raise TodoError(
                f"duplicate item {content!r}; each task must be listed once",
                code="TODO_DUPLICATE",
            )
        seen.add(content)
        normalized.append(TodoItem(content=content, status=item.status))

    if not allow_parallel_in_progress:
        active = [i for i in normalized if i.status == STATUS_IN_PROGRESS]
        if len(active) > 1:
            raise TodoError(
                f"{len(active)} tasks are marked in progress; at most one may be "
                "unless parallel work is enabled. Mark the others pending.",
                code="TODO_PARALLEL_IN_PROGRESS",
            )

    return normalized


class TodoStore:
    """Reads and writes the todo list for one session.

    The list is folded from the log like every other durable state here, so it
    survives a restart for free and cannot disagree with what was written.
    """

    def __init__(
        self, session: Session, *, allow_parallel_in_progress: bool = False
    ) -> None:
        self.session = session
        self.allow_parallel_in_progress = allow_parallel_in_progress

    # -- reading ---------------------------------------------------------

    def current(self) -> list[TodoItem]:
        """The list as of the last write.

        Folded rather than stored: the last ``todo/write`` wins, because each
        carries the whole table.
        """
        latest: list[TodoItem] | None = None
        for event in self.session:
            if event.type == TODO_WRITE:
                raw = event.data.get("todos") or ()
                latest = [TodoItem.from_record(r) for r in raw]
        return list(latest or [])

    def summary(self) -> TodoSummary:
        """Counts by status, for the one-line result the model sees."""
        items = self.current()
        return TodoSummary(
            pending=sum(1 for i in items if i.status == STATUS_PENDING),
            in_progress=sum(1 for i in items if i.status == STATUS_IN_PROGRESS),
            completed=sum(1 for i in items if i.status == STATUS_COMPLETED),
        )

    def history(self) -> list[list[TodoItem]]:
        """Every list ever written, oldest first.

        Kept because a reader may want to see how a plan evolved -- and because
        the log has no deletion, so the history exists whether or not anything
        reads it.
        """
        out: list[list[TodoItem]] = []
        for event in self.session:
            if event.type == TODO_WRITE:
                raw = event.data.get("todos") or ()
                out.append([TodoItem.from_record(r) for r in raw])
        return out

    # -- writing ---------------------------------------------------------

    def write(self, items: Sequence[TodoItem]) -> TodoSummary:
        """Replace the list.

        The whole-table replacement is the entire write model: there is no add,
        no update, no delete. One operation means one code path means one place
        for a bug, and it makes the log entry self-describing -- a reader needs no
        prior state to know what the list is.
        """
        normalized = validate(
            items, allow_parallel_in_progress=self.allow_parallel_in_progress
        )
        self.session.append(
            TODO_WRITE, {"todos": [item.to_record() for item in normalized]}
        )
        return TodoSummary(
            pending=sum(1 for i in normalized if i.status == STATUS_PENDING),
            in_progress=sum(1 for i in normalized if i.status == STATUS_IN_PROGRESS),
            completed=sum(1 for i in normalized if i.status == STATUS_COMPLETED),
        )

    def clear(self) -> TodoSummary:
        """Empty the list. Written as an empty replacement, not as a deletion."""
        return self.write([])


def project(session: Session) -> list[TodoItem] | None:
    """The list as a *reader* should see it.

    Returns ``None`` when the list is stale -- that is, when a new turn has begun
    since the last write. A list from three turns ago describes work that is
    probably finished, and showing it as current would be a lie. The model's own
    view is unaffected; this is for a UI or an evaluation.
    """
    latest_seq: int | None = None
    latest: list[TodoItem] | None = None
    last_turn_start = -1

    for event in session:
        if event.type == TODO_WRITE:
            raw = event.data.get("todos") or ()
            latest = [TodoItem.from_record(r) for r in raw]
            latest_seq = event.seq
        elif event.type == TURN_START:
            last_turn_start = event.seq

    if latest is None:
        return None
    if latest_seq is not None and latest_seq < last_turn_start:
        # A new turn began after the last write, so the list is no longer current.
        return None
    return latest

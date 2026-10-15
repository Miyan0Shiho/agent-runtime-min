"""Schedule: reminders that fire even when no session is loaded.

This is the one durable store in the runtime that is **not** the session log, and
the reason is worth stating because it looks like an inconsistency.

Everything else lives in a session log because a session is always the thing
being worked on. A reminder is different: it must fire at a wall-clock time
whether or not its session is loaded, and it must survive a process that never
opened that session at all. Putting reminders in the session log would mean they
only exist once the session is loaded -- which is precisely the situation a
reminder is supposed to escape.

Three consequences follow, and each is a deliberate acceptance of a cost:

**The store is separate, so the two writes are not atomic.** Delivering a
reminder means persisting it into the session *and* advancing the schedule. A
crash between them can deliver the same reminder twice. That window is
documented rather than papered over, because the alternative -- pretending the
receipt proves execution -- would be a lie.

**A receipt means "the inbox has it", not "the model did it".** The runtime
cannot know whether a delivered reminder was acted on. It says only what it
knows: the message was persisted. Any stronger claim would be unverifiable.

**Only the most recent missed occurrence is delivered.** After a long downtime, a
one-minute reminder does not produce a thousand messages. It produces one, for
the latest occurrence, and then continues from the next scheduled time. Losing
the intermediate occurrences is the accepted cost of not having a backlog queue.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence

from ..kernel.errors import ArmError
from ..session.envelope import utc_now_iso

log = logging.getLogger("arm.schedule")

#: A task is either still firing or finished. ``failed`` exists so a reminder
#: whose session could not be resolved is visible rather than silently dropped.
STATUS_ACTIVE = "active"
STATUS_INACTIVE = "inactive"
STATUS_FAILED = "failed"

STATUSES = frozenset({STATUS_ACTIVE, STATUS_INACTIVE, STATUS_FAILED})

#: The kinds of recurrence.
KIND_AFTER = "after"
KIND_AT = "at"
KIND_EVERY = "every"
KIND_DAILY = "daily"
KIND_WEEKLY = "weekly"
KIND_CRON = "cron"

KINDS = frozenset({KIND_AFTER, KIND_AT, KIND_EVERY, KIND_DAILY, KIND_WEEKLY, KIND_CRON})

#: The recurring kinds, which advance rather than completing.
RECURRING = frozenset({KIND_EVERY, KIND_DAILY, KIND_WEEKLY, KIND_CRON})

#: A title is required and is **never derived from the instruction**. A derived
#: title would be a summary of something the user wrote, and a bad summary is
#: worse than no summary: it misrepresents what the reminder is for.
MAX_TITLE_CHARS = 120

#: The longest interval a schedule may have. ``timedelta`` cannot represent more
#: than about 2.7 million years, and a value past that raises ``OverflowError``
#: when the next occurrence is computed -- so it is refused at creation instead.
MAX_INTERVAL_SECONDS = 366 * 24 * 3600 * 1000
_MAX_INTERVAL_SECONDS = MAX_INTERVAL_SECONDS

#: How much delivery history to keep. Both bounds apply, and the stricter one
#: wins -- a cap on count alone would let a burst of reminders evict years of
#: history, and a cap on age alone would let a burst keep unbounded records.
DEFAULT_HISTORY_DAYS = 30
DEFAULT_HISTORY_RECORDS = 200


class ScheduleError(ArmError):
    """A schedule operation was refused."""

    def __init__(self, message: str, *, code: str = "SCHEDULE_INVALID") -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Occurrence arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Occurrence:
    """When a task fired, and when it fires next.

    ``next_at`` is ``None`` for a one-shot. This pair -- not a list of missed
    times -- is what makes "deliver only the latest" expressible: the decision
    needs the most recent due time and the next one, and nothing in between.
    """

    fired_at: datetime
    next_at: datetime | None = None


def _parse_instant(value: str) -> datetime:
    """Parse an ISO-8601 instant, requiring a timezone.

    A naive timestamp is refused rather than assumed to be local: the assumption
    is invisible, and it is wrong for exactly the reminders that cross a
    daylight-saving boundary or a travel.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScheduleError(f"{value!r} is not an ISO-8601 instant: {exc}") from exc
    if parsed.tzinfo is None:
        raise ScheduleError(
            f"{value!r} has no timezone; an instant must say which zone it is in"
        )
    return parsed


def _zone_for(name: str) -> timezone | Any:
    """Resolve an IANA zone name, or the system zone when none is given.

    An unknown zone is refused rather than silently replaced with UTC: a
    schedule stored with a typo would otherwise fire at an hour nobody chose, and
    the mistake would never be visible.
    """
    if not name:
        # The local zone, whatever the machine is set to. `astimezone()` with no
        # argument does exactly this.
        return datetime.now().astimezone().tzinfo
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001 - a bad zone name is a config error
        raise ScheduleError(
            f"{name!r} is not a known timezone: {exc}", code="SCHEDULE_BAD_TIMEZONE"
        ) from exc


def _parse_local_time(value: str) -> tuple[int, int, int]:
    """Parse an ``HH:MM[:SS]`` local time."""
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ScheduleError(f"{value!r} is not a time of day (HH:MM[:SS])")
    try:
        hour, minute = int(parts[0]), int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ScheduleError(f"{value!r} is not a time of day: {exc}") from exc
    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise ScheduleError(f"{value!r} is not a valid time of day")
    return hour, minute, second


def next_occurrence(
    record: "ScheduleRecord", *, now: datetime, fired_at: datetime
) -> datetime | None:
    """The next time a task should fire, or ``None`` if it is finished.

    For a recurring task this skips directly to the first target **after** the
    decision moment. The skipped occurrences are not queued: a reminder that
    fires once per minute and was offline for a day delivers one message, not
    1,440.
    """
    if record.kind not in RECURRING:
        return None

    if record.kind == KIND_EVERY:
        seconds = record.every_seconds
        if seconds is None or seconds < 1:
            raise ScheduleError("an 'every' task needs a positive interval")
        # Advance to the first aligned target after now, preserving the phase
        # established at creation so the interval does not drift.
        #
        # The anchor is a stored *instant*, so it is parsed here rather than
        # trusted to be a datetime -- the record round-trips through JSON, and a
        # field that is a datetime in memory and a string on disk is exactly the
        # kind of mismatch that works until it is reloaded.
        if not record.anchor:
            return now + timedelta(seconds=seconds)
        anchor = _parse_instant(record.anchor)
        elapsed = (now - anchor).total_seconds()
        steps = int(elapsed // seconds) + 1
        return anchor + timedelta(seconds=steps * seconds)

    if record.kind in (KIND_DAILY, KIND_WEEKLY):
        # A time of day is only meaningful in a stated zone. Resolving "09:00" in
        # whatever zone `now` happens to carry means a user in Asia/Shanghai who
        # asks for 09:00 gets 17:00 -- the reminder fires, at the wrong time, and
        # nothing reports the discrepancy.
        hour, minute, second = _parse_local_time(record.at_time)
        zone = _zone_for(record.timezone)

        if record.kind == KIND_DAILY:
            local_now = now.astimezone(zone)
            candidate = local_now.replace(
                hour=hour, minute=minute, second=second, microsecond=0
            )
            if candidate <= local_now:
                candidate += timedelta(days=1)
            return candidate

        weekdays = set(record.weekdays)
        if not weekdays:
            raise ScheduleError("a 'weekly' task needs at least one weekday")
        local_now = now.astimezone(zone)
        for offset in range(0, 8):
            candidate = (local_now + timedelta(days=offset)).replace(
                hour=hour, minute=minute, second=second, microsecond=0
            )
            # ISO weekday: Monday is 1.
            if candidate.isoweekday() in weekdays and candidate > local_now:
                return candidate
        return None

    if record.kind == KIND_CRON:
        return _next_cron(record.cron, now)

    return None


def _next_cron(expression: str, now: datetime) -> datetime | None:
    """The next time a five-field cron expression matches.

    Searched minute by minute, bounded to four years. That is not the most
    efficient algorithm, but it is obviously correct, and a reminder is evaluated
    once per wake-up rather than in a loop.
    """
    fields = expression.split()
    if len(fields) != 5:
        raise ScheduleError(
            f"{expression!r} is not a five-field cron expression "
            "(minute hour day-of-month month day-of-week)"
        )

    def matches(field: str, value: int, low: int, high: int, *, dow: bool = False) -> bool:
        for part in field.split(","):
            step = 1
            if "/" in part:
                part, _, step_text = part.partition("/")
                try:
                    step = int(step_text)
                except ValueError:
                    raise ScheduleError(f"invalid step in {field!r}") from None
                if step < 1:
                    raise ScheduleError(f"invalid step in {field!r}")
            if part == "*":
                start, end = low, high
            elif "-" in part:
                left, _, right = part.partition("-")
                try:
                    start, end = int(left), int(right)
                except ValueError:
                    raise ScheduleError(f"invalid range in {field!r}") from None
            else:
                try:
                    start = end = int(part)
                except ValueError:
                    raise ScheduleError(f"invalid value in {field!r}") from None
            # Cron's day-of-week accepts both 0 and 7 for Sunday. Python's
            # isoweekday() never returns 0, so a literal 7 would never match --
            # silently, because a cron expression that never fires looks exactly
            # like one that has not fired yet.
            if dow and start == 7:
                start = 0
            if dow and end == 7:
                end = 6 if start == 0 else 7
            if start <= value <= end and (value - start) % step == 0:
                return True
        return False

    minute, hour, dom, month, dow = fields
    candidate = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    limit = candidate + timedelta(days=366 * 4)

    # Cron's day fields are *restricted* only when they are not ``*``, and when
    # both are restricted the day matches if **either** matches. That OR is the
    # part everyone gets wrong: treating it as AND means "the 1st, if it is a
    # Monday", when the expression means "the 1st, and also every Monday".
    dom_restricted = dom.strip() != "*"
    dow_restricted = dow.strip() != "*"

    while candidate < limit:
        # Cron numbers Sunday 0; Python numbers Monday 1. Sunday is the only day
        # where the two disagree in value, so it maps to 0 explicitly.
        python_dow = candidate.isoweekday() % 7

        day_matches = True
        if dom_restricted and dow_restricted:
            day_matches = matches(dom, candidate.day, 1, 31) or matches(
                dow, python_dow, 0, 6, dow=True
            )
        elif dom_restricted:
            day_matches = matches(dom, candidate.day, 1, 31)
        elif dow_restricted:
            day_matches = matches(dow, python_dow, 0, 6, dow=True)

        if (
            matches(minute, candidate.minute, 0, 59)
            and matches(hour, candidate.hour, 0, 23)
            and matches(month, candidate.month, 1, 12)
            and day_matches
        ):
            return candidate
        candidate += timedelta(minutes=1)

    raise ScheduleError(f"{expression!r} has no occurrence in the next four years")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScheduleRecord:
    """One scheduled task.

    The ``instruction`` is what the model will be told. The ``title`` is what a
    human reads in a list. They are separate because deriving one from the other
    produces a bad summary, and a bad summary misrepresents what the reminder is
    for.
    """

    id: str
    title: str
    instruction: str
    session_id: str
    kind: str
    #: For ``after``: the delay in seconds, resolved to ``at_instant`` on creation.
    after_seconds: int | None = None
    #: The absolute instant a one-shot fires at.
    at_instant: str = ""
    #: For ``every``: the interval.
    every_seconds: int | None = None
    #: For ``daily``/``weekly``: the local time of day.
    at_time: str = ""
    #: For ``weekly``: ISO weekdays.
    weekdays: tuple[int, ...] = ()
    #: For ``cron``: the expression.
    cron: str = ""
    #: The IANA zone a time-of-day is expressed in. Empty means the system zone.
    #:
    #: Stored on the record rather than read from the clock, because a schedule
    #: outlives a process and must mean the same thing after a restart in a
    #: different zone.
    timezone: str = ""
    #: The alignment anchor for ``every``, so the interval does not drift.
    anchor: str = ""
    status: str = STATUS_ACTIVE
    #: When this task next fires. Absent for a finished one-shot.
    scheduled_at: str = ""
    created_at: str = ""
    last_delivery: str = ""

    def __post_init__(self) -> None:
        title = self.title.strip()
        if not title:
            raise ScheduleError(
                "a scheduled task needs a title", code="SCHEDULE_NO_TITLE"
            )
        if len(title) > MAX_TITLE_CHARS:
            raise ScheduleError(
                f"the title is {len(title)} characters, over the limit of "
                f"{MAX_TITLE_CHARS}",
                code="SCHEDULE_TITLE_TOO_LONG",
            )
        if not self.instruction.strip():
            raise ScheduleError(
                "a scheduled task needs an instruction", code="SCHEDULE_NO_INSTRUCTION"
            )
        if self.kind not in KINDS:
            raise ScheduleError(f"unknown schedule kind {self.kind!r}")
        if self.status not in STATUSES:
            raise ScheduleError(f"unknown schedule status {self.status!r}")
        if self.kind == KIND_EVERY:
            seconds = self.every_seconds or 0
            if seconds < 1:
                raise ScheduleError(
                    "an 'every' task needs a positive interval",
                    code="SCHEDULE_INVALID",
                )
            # Bounded because the interval is added to a datetime, and a value
            # beyond ``timedelta``'s range raises OverflowError at that point --
            # a crash, not a refusal, and one that happens when the schedule is
            # first computed rather than when it was created.
            if seconds > _MAX_INTERVAL_SECONDS:
                raise ScheduleError(
                    f"the interval {seconds}s is longer than the maximum of "
                    f"{_MAX_INTERVAL_SECONDS}s",
                    code="SCHEDULE_INTERVAL_TOO_LARGE",
                )
        if self.kind in (KIND_DAILY, KIND_WEEKLY):
            _parse_local_time(self.at_time)
            # Resolved here so a bad zone name is refused when the schedule is
            # created, not when it first comes due -- hours later, by which point
            # the user believes it is set.
            _zone_for(self.timezone)
        if self.kind == KIND_WEEKLY:
            if not self.weekdays:
                raise ScheduleError(
                    "a 'weekly' task needs at least one weekday",
                    code="SCHEDULE_INVALID",
                )
            # ISO weekdays are 1-7. An out-of-range value would never match, so
            # the task would be accepted, reported as scheduled, and then fire
            # immediately and once -- which looks like a working reminder until
            # it never fires again.
            bad = [d for d in self.weekdays if not 1 <= d <= 7]
            if bad:
                raise ScheduleError(
                    f"weekdays must be ISO day numbers 1-7 (Monday is 1), got {bad}",
                    code="SCHEDULE_INVALID_WEEKDAY",
                )
        if self.kind == KIND_AT:
            if not self.at_instant:
                raise ScheduleError(
                    "an 'at' task needs an instant", code="SCHEDULE_INVALID"
                )
            # Parsed here so a naive timestamp is refused when the task is
            # created, not hours later when it comes due. The assumption a naive
            # timestamp implies is invisible, and it is wrong exactly for the
            # reminders that cross a daylight-saving boundary.
            _parse_instant(self.at_instant)
        if self.scheduled_at:
            _parse_instant(self.scheduled_at)
        if self.anchor:
            _parse_instant(self.anchor)

    @property
    def recurring(self) -> bool:
        return self.kind in RECURRING

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "instruction": self.instruction,
            "session_id": self.session_id,
            "kind": self.kind,
            "status": self.status,
        }
        for key in (
            "after_seconds",
            "at_instant",
            "every_seconds",
            "at_time",
            "cron",
            "timezone",
            "anchor",
            "scheduled_at",
            "created_at",
            "last_delivery",
        ):
            value = getattr(self, key)
            if value not in ("", None):
                record[key] = value
        if self.weekdays:
            record["weekdays"] = list(self.weekdays)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ScheduleRecord":
        return cls(
            id=str(record.get("id", "")),
            title=str(record.get("title", "")),
            instruction=str(record.get("instruction", "")),
            session_id=str(record.get("session_id", "")),
            kind=str(record.get("kind", "")),
            after_seconds=(
                int(record["after_seconds"]) if record.get("after_seconds") else None
            ),
            at_instant=str(record.get("at_instant", "")),
            every_seconds=(
                int(record["every_seconds"]) if record.get("every_seconds") else None
            ),
            at_time=str(record.get("at_time", "")),
            weekdays=tuple(int(d) for d in (record.get("weekdays") or ())),
            cron=str(record.get("cron", "")),
            timezone=str(record.get("timezone", "")),
            anchor=str(record.get("anchor", "")),
            status=str(record.get("status", STATUS_ACTIVE)),
            scheduled_at=str(record.get("scheduled_at", "")),
            created_at=str(record.get("created_at", "")),
            last_delivery=str(record.get("last_delivery", "")),
        )

    def due_at(self) -> datetime | None:
        """When this task is next due, as an instant."""
        if self.kind == KIND_AT:
            return _parse_instant(self.at_instant) if self.at_instant else None
        if self.scheduled_at:
            return _parse_instant(self.scheduled_at)
        return None


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """A record that a reminder was persisted into its session.

    It says exactly one thing: the message reached the session's inbox. It does
    **not** say the model read it, acted on it, or even woke up. Claiming more
    would be unverifiable, and an unverifiable claim in a durable record is worse
    than no record.
    """

    schedule_id: str
    session_id: str
    delivered_at: str
    occurrence_at: str
    #: The next fire time after this delivery, if the task recurs.
    next_scheduled_at: str = ""
    message_id: str = ""

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schedule_id": self.schedule_id,
            "session_id": self.session_id,
            "delivered_at": self.delivered_at,
            "occurrence_at": self.occurrence_at,
        }
        if self.next_scheduled_at:
            record["next_scheduled_at"] = self.next_scheduled_at
        if self.message_id:
            record["message_id"] = self.message_id
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DeliveryReceipt":
        return cls(
            schedule_id=str(record.get("schedule_id", "")),
            session_id=str(record.get("session_id", "")),
            delivered_at=str(record.get("delivered_at", "")),
            occurrence_at=str(record.get("occurrence_at", "")),
            next_scheduled_at=str(record.get("next_scheduled_at", "")),
            message_id=str(record.get("message_id", "")),
        )


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class ScheduleStore:
    """A durable set of scheduled tasks.

    Deliberately storage-agnostic: it holds records and hands them to a sink. The
    runtime supplies a JSON file sink; a test supplies a dict. The semantics --
    what is due, what is delivered, what recurs -- do not depend on where the
    bytes land.
    """

    def __init__(
        self,
        *,
        sink: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        source: Sequence[Mapping[str, Any]] | None = None,
        history_days: int = DEFAULT_HISTORY_DAYS,
        history_records: int = DEFAULT_HISTORY_RECORDS,
    ) -> None:
        if history_days < 1 or history_records < 1:
            raise ScheduleError("history limits must be positive integers")
        self.history_days = history_days
        self.history_records = history_records
        self._tasks: dict[str, ScheduleRecord] = {}
        self._history: list[DeliveryReceipt] = []
        self._sink = sink
        if source:
            for record in source:
                task = ScheduleRecord.from_record(record)
                self._tasks[task.id] = task

    # -- reading ---------------------------------------------------------

    def get(self, task_id: str) -> ScheduleRecord | None:
        return self._tasks.get(task_id)

    def all(self) -> tuple[ScheduleRecord, ...]:
        return tuple(self._tasks.values())

    def for_session(self, session_id: str) -> tuple[ScheduleRecord, ...]:
        return tuple(t for t in self._tasks.values() if t.session_id == session_id)

    def due(self, *, now: datetime | None = None) -> tuple[ScheduleRecord, ...]:
        """Active tasks whose time has arrived.

        A task with no ``scheduled_at`` and a recurring kind is due immediately:
        a schedule created with no explicit start should begin, not wait forever
        for a time that was never set.
        """
        moment = now or datetime.now(timezone.utc)
        out: list[ScheduleRecord] = []
        for task in self._tasks.values():
            if task.status != STATUS_ACTIVE:
                continue
            when = task.due_at()
            if when is None:
                if task.recurring:
                    out.append(task)
                continue
            if when <= moment:
                out.append(task)
        return tuple(out)

    def history(
        self, *, schedule_id: str = "", limit: int = 50
    ) -> tuple[DeliveryReceipt, ...]:
        """Receipts, newest first, in *append* order rather than by timestamp.

        Append order is the honest ordering: two reminders delivered in the same
        second have a definite sequence, and wall-clock sorting would make it
        ambiguous.
        """
        if limit <= 0:
            # ``[-0:]`` is the whole list, so a caller asking for zero would get
            # everything. Refusing the request is the honest answer.
            return ()
        receipts = [
            r for r in self._history if not schedule_id or r.schedule_id == schedule_id
        ]
        return tuple(reversed(receipts[-limit:]))

    # -- writing ---------------------------------------------------------

    def put(self, task: ScheduleRecord) -> None:
        """Insert or replace a task."""
        self._tasks[task.id] = task
        self._flush()

    def delete(self, task_id: str) -> bool:
        """Remove a task. Returns whether there was one."""
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        self._flush()
        return True

    def record_delivery(self, receipt: DeliveryReceipt) -> None:
        """Append a receipt and prune the history."""
        self._history.append(receipt)
        self._prune_history()
        self._flush()

    def _prune_history(self) -> None:
        """Enforce both history bounds; the stricter one wins.

        A cap on count alone would let a burst of reminders evict years of
        history; a cap on age alone would let a burst keep an unbounded number of
        records. Keeping both means neither failure mode is reachable.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.history_days)
        kept: list[DeliveryReceipt] = []
        for receipt in self._history:
            try:
                when = _parse_instant(receipt.delivered_at)
            except ScheduleError:
                continue
            if when >= cutoff:
                kept.append(receipt)
        # The newest record is always kept, even if it is older than the cutoff:
        # a schedule that fires rarely would otherwise lose the only evidence it
        # ever fired.
        if not kept and self._history:
            kept = [self._history[-1]]
        self._history = kept[-self.history_records :]

    def _flush(self) -> None:
        if self._sink is None:
            return
        try:
            self._sink([t.to_record() for t in self._tasks.values()])
        except Exception as exc:  # noqa: BLE001 - a sink failure must be visible
            log.error("could not persist the schedule: %s", exc)
            raise ScheduleError(
                f"the schedule could not be persisted: {exc}",
                code="SCHEDULE_PERSIST_FAILED",
            ) from exc


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------


@dataclass
class ScheduleRuntime:
    """Delivers due reminders.

    Holds **one** timer. Multiple timers would race on the same tasks, and the
    "only the latest occurrence" rule assumes a single decision point per
    wake-up.
    """

    store: ScheduleStore
    #: Resolves a session id to something that can accept a message. Injected so
    #: the runtime does not depend on how sessions are loaded.
    resolve: Callable[[str], Any]
    #: Persists the delivered message and returns whether it was acknowledged.
    #: A `False` means the reminder was not confirmed and must not be marked
    #: delivered.
    deliver: Callable[[str, str, str], bool]
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    _timer: Any = None
    _driving: bool = False

    def request_drive(self) -> None:
        """Schedule a drive, merging concurrent requests.

        Merged rather than queued: if two things ask to drive at once, one drive
        is enough, and running two would deliver the same reminder twice.
        """
        if self._driving:
            return
        self._driving = True
        try:
            self.drive()
        finally:
            self._driving = False

    def drive(self) -> list[DeliveryReceipt]:
        """Deliver every due reminder, once each.

        The order is deliberate: resolve the session, re-sample the time, then
        deliver and only then advance the task. A reminder that cannot be
        delivered stays due, so the next drive retries it -- losing it silently
        would be the one outcome a reminder must never produce.
        """
        # Retried first, so a reminder that failed on an earlier drive is
        # reconsidered on this one. A failed task is not due (its status is not
        # active), so without this it would never be looked at again.
        self.retry_failed_on_drive()

        now = self.clock()
        due = self.store.due(now=now)
        if not due:
            return []

        # Recurring reminders for one session are grouped into a single delivery:
        # three reminders that fired while the process was down are one wake-up,
        # not three.
        by_session: dict[str, list[ScheduleRecord]] = {}
        for task in due:
            by_session.setdefault(task.session_id, []).append(task)

        receipts: list[DeliveryReceipt] = []
        failed: list[str] = []

        for session_id, tasks in by_session.items():
            try:
                session = self.resolve(session_id)
            except Exception as exc:  # noqa: BLE001 - a failed resolve is a state
                log.warning("could not resolve session %s: %s", session_id, exc)
                for task in tasks:
                    # Recorded as failed rather than dropped: the task stays in
                    # the store, so a later drive or a restart retries it.
                    self.store.put(
                        _replace(task, status=STATUS_FAILED, last_delivery="")
                    )
                    failed.append(task.id)
                continue

            if session is None:
                for task in tasks:
                    self.store.put(
                        _replace(task, status=STATUS_FAILED, last_delivery="")
                    )
                    failed.append(task.id)
                continue

            # Re-sample after resolving: resolving may take time, and a task whose
            # time has not actually arrived must not be delivered early.
            moment = self.clock()
            still_due = [t for t in tasks if _is_due(t, moment)]
            not_yet = [t for t in tasks if t not in still_due]
            for task in not_yet:
                # Kept active with its original time: the obligation survives,
                # only this drive skipped it.
                self.store.put(_replace(task, status=STATUS_ACTIVE))

            if not still_due:
                continue

            body = _render_delivery(still_due, moment)
            message_id = f"schedule-{still_due[0].id}-{int(moment.timestamp())}"

            # The claim happens **before** the delivery, not after.
            #
            # A reminder is claimed by advancing its schedule, so a re-entrant
            # drive -- a delivery that itself triggers a drive, or two drives
            # racing on the same store -- sees a task that is no longer due and
            # cannot deliver it a second time. Advancing afterwards leaves a
            # window in which the same reminder is delivered twice, and a
            # duplicate reminder is exactly as bad as a missing one.
            #
            # The cost is the reverse window: a crash between claiming and
            # delivering loses that occurrence. That window is documented rather
            # than hidden, and it is the smaller of the two, because the claim is
            # a local write and the delivery is not.
            claimed: list[tuple[ScheduleRecord, Occurrence | None]] = []
            for task in still_due:
                occurrence = task.due_at() or moment
                nxt = next_occurrence(task, now=moment, fired_at=occurrence)
                if nxt is None:
                    self.store.put(
                        _replace(task, status=STATUS_INACTIVE, scheduled_at="")
                    )
                else:
                    self.store.put(
                        _replace(
                            task, status=STATUS_ACTIVE, scheduled_at=nxt.isoformat()
                        )
                    )
                claimed.append((task, None if nxt is None else None))

            try:
                acknowledged = self.deliver(session_id, body, message_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("delivery to %s failed: %s", session_id, exc)
                acknowledged = False

            if not acknowledged:
                # Not acknowledged means not delivered, so the claim is undone:
                # the reminder goes back to being due and is retried on the next
                # drive. Leaving it claimed would lose it, which is the one thing
                # a reminder must never do.
                for task, _ in claimed:
                    self.store.put(
                        _replace(
                            task,
                            status=STATUS_ACTIVE,
                            scheduled_at=task.scheduled_at or moment.isoformat(),
                        )
                    )
                    failed.append(task.id)
                continue

            for task, _ in claimed:
                occurrence = task.due_at() or moment
                nxt = next_occurrence(task, now=moment, fired_at=occurrence)
                receipt = DeliveryReceipt(
                    schedule_id=task.id,
                    session_id=session_id,
                    delivered_at=utc_now_iso(),
                    occurrence_at=occurrence.isoformat(),
                    next_scheduled_at=nxt.isoformat() if nxt else "",
                    message_id=message_id,
                )
                self.store.record_delivery(receipt)
                receipts.append(receipt)

        if failed:
            log.warning("reminders not delivered: %s", failed)
        return receipts

    def retry_failed_on_drive(self) -> int:
        """Reactivate failed reminders before each drive.

        A failure is almost always transient -- a session that was not loadable
        yet, a lock, a momentary error -- so a reminder that failed must be
        retried rather than abandoned. Without this, a reminder whose session was
        not open at its moment was marked failed and **never fired again**, which
        is precisely the outcome the whole design exists to prevent.

        Called at the start of every drive rather than exposed only as a method:
        a recovery path nothing invokes is not a recovery path.
        """
        return self.retry_failed()

    def retry_failed(self) -> int:
        """Return failed tasks to the active set.

        A failure is usually transient -- a session that was not loadable yet --
        so the tasks are retried rather than abandoned. Returns how many were
        reactivated.
        """
        count = 0
        for task in self.store.all():
            if task.status == STATUS_FAILED:
                self.store.put(_replace(task, status=STATUS_ACTIVE))
                count += 1
        return count


@dataclass
class ScheduleTimer:
    """A background timer that drives the scheduler while the process is idle.

    **Without this, a reminder only fires if the user happens to be mid-turn when
    its time arrives.** That defeats the entire purpose: a reminder set for three
    hours from now would sit undelivered until the next unrelated conversation,
    and the user -- who set it precisely because they expected to forget -- would
    have no way to tell that it never fired.

    The timer is deliberately simple. It polls rather than computing an exact
    wake-up for the next task, because:

    * the polling interval is bounded and short, so the worst-case lateness is
      known and small;
    * a timer that recomputes on every task change has to be cancelled and
      rescheduled on every write, and a missed reschedule silently stops all
      reminders -- the failure mode this class exists to prevent.

    The cost is one cheap comparison per interval, which is nothing next to the
    cost of a reminder that never arrives.
    """

    runtime: "ScheduleRuntime"
    interval: float = 30.0
    #: Called when a delivery happens, so a host can react (log, notify, wake).
    on_delivery: Callable[[list[DeliveryReceipt]], None] | None = None

    _task: Any = None
    _stopped: bool = False

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ScheduleError("the poll interval must be positive")

    def start(self) -> None:
        """Begin polling. Idempotent: starting twice does not double the timers."""
        if self._task is not None and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.ensure_future(self._loop())

    async def _loop(self) -> None:
        """Poll until stopped.

        A failure inside one drive must not kill the loop: a single broken
        reminder would otherwise stop every future reminder, and the user would
        have no way to notice.
        """
        while not self._stopped:
            try:
                receipts = self.runtime.drive()
            except Exception as exc:  # noqa: BLE001 - one bad drive is not fatal
                log.warning("a scheduled drive failed: %s", exc)
                receipts = []
            if receipts and self.on_delivery is not None:
                try:
                    self.on_delivery(receipts)
                except Exception as exc:  # noqa: BLE001
                    log.warning("the delivery callback failed: %s", exc)
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                return

    async def stop(self) -> None:
        """Stop polling and wait for the loop to finish."""
        self._stopped = True
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def seconds_until_next(self) -> float | None:
        """How long until the soonest task is due, or ``None`` if there is none.

        Exposed so a host can pick a smarter interval than a fixed poll when it
        wants to -- and so a test can assert the schedule is not empty.
        """
        now = self.runtime.clock()
        soonest: float | None = None
        for task in self.runtime.store.all():
            if task.status != STATUS_ACTIVE:
                continue
            when = task.due_at()
            if when is None:
                return 0.0
            remaining = (when - now).total_seconds()
            if soonest is None or remaining < soonest:
                soonest = remaining
        return soonest


def _is_due(task: ScheduleRecord, moment: datetime) -> bool:
    when = task.due_at()
    if when is None:
        return task.recurring
    return when <= moment


def _replace(task: ScheduleRecord, **changes: Any) -> ScheduleRecord:
    """A copy of a task with some fields changed."""
    fields = {
        "id": task.id,
        "title": task.title,
        "instruction": task.instruction,
        "session_id": task.session_id,
        "kind": task.kind,
        "after_seconds": task.after_seconds,
        "at_instant": task.at_instant,
        "every_seconds": task.every_seconds,
        "at_time": task.at_time,
        "weekdays": task.weekdays,
        "cron": task.cron,
        "anchor": task.anchor,
        "status": task.status,
        "scheduled_at": task.scheduled_at,
        "created_at": task.created_at,
        "last_delivery": task.last_delivery,
    }
    fields.update(changes)
    return ScheduleRecord(**fields)


def _render_delivery(tasks: Sequence[ScheduleRecord], moment: datetime) -> str:
    """Render one delivery covering every due reminder for a session.

    The text states what the runtime knows -- that a reminder was scheduled and
    has come due -- and does not claim the model must act. A reminder is a
    prompt, not an instruction the runtime can enforce.
    """
    lines: list[str] = []
    if len(tasks) == 1:
        task = tasks[0]
        lines.append(f"Reminder: {task.title}")
        lines.append("")
        lines.append(task.instruction)
    else:
        lines.append(f"{len(tasks)} reminders are due:")
        lines.append("")
        for task in tasks:
            lines.append(f"- {task.title}: {task.instruction}")
    return "\n".join(lines)

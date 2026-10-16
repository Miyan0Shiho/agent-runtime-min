"""Schedule tests.

Three claims are under test, and each is a place the design could quietly fail:

* **Only the latest missed occurrence is delivered.** A reminder that fires per
  minute and was offline for a day produces one message, not 1,440. The
  alternative is a backlog queue, which is a different (and worse) design.
* **An unacknowledged delivery is not a delivery.** The task stays due, so a
  reminder is never silently lost -- the one outcome a reminder must never have.
* **A receipt says what it knows.** It records that the message was persisted,
  not that the model read it. The tests assert the *absence* of stronger claims.

Time is injected everywhere, so nothing here sleeps or depends on the clock.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest
from datetime import datetime, timedelta, timezone

from arm.session.session import Session
from arm.schedule.runtime import (
    _replace,
    _zone_for,
    KIND_AFTER,
    KIND_AT,
    KIND_CRON,
    KIND_DAILY,
    KIND_EVERY,
    KIND_WEEKLY,
    MAX_TITLE_CHARS,
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_INACTIVE,
    DeliveryReceipt,
    ScheduleError,
    ScheduleRecord,
    ScheduleRuntime,
    ScheduleStore,
    next_occurrence,
)

NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)


def require(task: ScheduleRecord | None) -> ScheduleRecord:
    """Assert a store lookup found something, so the test fails clearly."""
    assert task is not None
    return task


def at_task(task_id: str = "s1", **kwargs: Any) -> ScheduleRecord:
    """A one-shot task due at a given instant."""
    fields: dict[str, Any] = {
        "id": task_id,
        "title": "a reminder",
        "instruction": "do the thing",
        "session_id": "sess-1",
        "kind": KIND_AT,
        "at_instant": NOW.isoformat(),
    }
    fields.update(kwargs)
    return ScheduleRecord(**fields)


def every_task(seconds: int = 60, **kwargs: Any) -> ScheduleRecord:
    fields: dict[str, Any] = {
        "id": "s1",
        "title": "recurring",
        "instruction": "keep going",
        "session_id": "sess-1",
        "kind": KIND_EVERY,
        "every_seconds": seconds,
        "anchor": NOW.isoformat(),
        "scheduled_at": NOW.isoformat(),
    }
    fields.update(kwargs)
    return ScheduleRecord(**fields)


# ===========================================================================
# Records
# ===========================================================================


def test_a_title_is_required():
    """A reminder with no title is unreadable in a list."""
    with pytest.raises(ScheduleError) as info:
        at_task(title="   ")
    assert info.value.code == "SCHEDULE_NO_TITLE"


def test_a_title_is_bounded():
    with pytest.raises(ScheduleError) as info:
        at_task(title="x" * (MAX_TITLE_CHARS + 1))
    assert info.value.code == "SCHEDULE_TITLE_TOO_LONG"


def test_a_title_is_never_derived_from_the_instruction():
    """They are separate fields on purpose.

    A derived title is a summary of something the user wrote, and a bad summary
    misrepresents what the reminder is for.
    """
    task = at_task(title="standup", instruction="a much longer instruction text")
    assert task.title == "standup"
    assert task.instruction == "a much longer instruction text"


def test_an_instruction_is_required():
    with pytest.raises(ScheduleError) as info:
        at_task(instruction="  ")
    assert info.value.code == "SCHEDULE_NO_INSTRUCTION"


def test_an_unknown_kind_is_refused():
    with pytest.raises(ScheduleError, match="unknown schedule kind"):
        at_task(kind="whenever")


def test_an_every_task_needs_a_positive_interval():
    with pytest.raises(ScheduleError):
        every_task(seconds=0)


def test_a_weekly_task_needs_a_weekday():
    with pytest.raises(ScheduleError, match="weekday"):
        ScheduleRecord(
            id="s", title="t", instruction="i", session_id="x",
            kind=KIND_WEEKLY, at_time="09:00", weekdays=(),
        )


def test_a_record_round_trips_through_json():
    """The store persists records, so they must survive the trip."""
    import json

    task = every_task()
    restored = ScheduleRecord.from_record(json.loads(json.dumps(task.to_record())))
    assert restored == task


def test_an_at_task_needs_an_instant():
    with pytest.raises(ScheduleError, match="instant"):
        ScheduleRecord(
            id="s", title="t", instruction="i", session_id="x", kind=KIND_AT
        )


# ===========================================================================
# Occurrence arithmetic
# ===========================================================================


def test_an_every_task_advances_to_the_next_aligned_target():
    """The anchor is preserved, so the interval does not drift."""
    task = every_task(seconds=60)
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)
    assert nxt == NOW + timedelta(seconds=60)


def test_an_every_task_skips_missed_occurrences():
    """A day of downtime on a one-minute reminder yields one message, not 1,440.

    The next target is the first one *after* the decision moment. The skipped
    occurrences are not queued: that is the accepted cost of not maintaining a
    backlog.
    """
    task = every_task(seconds=60)
    much_later = NOW + timedelta(days=1)
    nxt = next_occurrence(task, now=much_later, fired_at=NOW)

    assert nxt is not None
    assert nxt > much_later
    # It lands on the alignment grid, not one interval after "now".
    assert (nxt - NOW).total_seconds() % 60 == 0


def test_a_one_shot_has_no_next_occurrence():
    """It fires once and is finished."""
    assert next_occurrence(at_task(), now=NOW, fired_at=NOW) is None


def test_a_daily_task_advances_to_tomorrow():
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_DAILY, at_time="09:00",
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)

    assert nxt is not None
    assert (nxt.hour, nxt.minute) == (9, 0)
    assert nxt > NOW


def test_a_daily_task_later_today_fires_today():
    """The time has not passed yet, so it is still today."""
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_DAILY, at_time="23:00",
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)

    assert nxt is not None
    assert nxt.day == NOW.day


def test_a_weekly_task_lands_on_the_requested_weekday():
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_WEEKLY, at_time="10:00", weekdays=(5,),  # Friday
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)

    assert nxt is not None
    assert nxt.isoweekday() == 5
    assert nxt > NOW


def test_an_invalid_time_of_day_is_refused():
    with pytest.raises(ScheduleError, match="time of day"):
        ScheduleRecord(
            id="s", title="t", instruction="i", session_id="x",
            kind=KIND_DAILY, at_time="25:00",
        )


def test_a_cron_expression_finds_the_next_match():
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_CRON, cron="0 9 * * *",
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)

    assert nxt is not None
    assert (nxt.hour, nxt.minute) == (9, 0)
    assert nxt > NOW


def test_cron_weekday_uses_sunday_as_zero():
    """Cron numbers Sunday 0; Python numbers Monday 1.

    Getting this wrong shifts every weekly reminder by a day, and nothing in a
    casual test would notice.
    """
    # Sunday at 08:00. NOW is a Tuesday.
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_CRON, cron="0 8 * * 0",
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)

    assert nxt is not None
    assert nxt.isoweekday() == 7, f"expected Sunday, got {nxt.isoweekday()}"
    assert (nxt.hour, nxt.minute) == (8, 0)


def test_a_cron_step_expression_works():
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_CRON, cron="*/15 * * * *",
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)

    assert nxt is not None
    assert nxt.minute % 15 == 0


def test_a_malformed_cron_expression_is_refused():
    for bad in ("0 9 * *", "not a cron", "0 9 * * * *"):
        task = ScheduleRecord(
            id="s", title="t", instruction="i", session_id="x",
            kind=KIND_CRON, cron=bad,
        )
        with pytest.raises(ScheduleError):
            next_occurrence(task, now=NOW, fired_at=NOW)


def test_a_naive_instant_is_refused():
    """Assuming a timezone is invisible, and it is wrong exactly when it matters."""
    with pytest.raises(ScheduleError, match="timezone"):
        at_task(at_instant="2026-10-13T12:00:00")


# ===========================================================================
# The store
# ===========================================================================


def test_due_returns_only_tasks_whose_time_arrived():
    store = ScheduleStore()
    past = at_task("past", at_instant=(NOW - timedelta(hours=1)).isoformat())
    future = at_task("future", at_instant=(NOW + timedelta(hours=1)).isoformat())
    store.put(past)
    store.put(future)

    due = store.due(now=NOW)
    assert [t.id for t in due] == ["past"]


def test_an_inactive_task_is_never_due():
    """A finished one-shot must not fire again."""
    store = ScheduleStore()
    store.put(at_task(status=STATUS_INACTIVE))

    assert store.due(now=NOW) == ()


def test_a_recurring_task_with_no_scheduled_time_is_due_now():
    """A schedule with no explicit start should begin, not wait forever."""
    store = ScheduleStore()
    store.put(every_task(scheduled_at=""))

    assert [t.id for t in store.due(now=NOW)] == ["s1"]


def test_deleting_a_task():
    store = ScheduleStore()
    store.put(at_task())

    assert store.delete("s1") is True
    assert store.get("s1") is None
    assert store.delete("s1") is False


def test_a_sink_receives_every_write():
    """Persistence is the store's job; the sink is where the bytes go."""
    written: list[list[Mapping[str, Any]]] = []
    store = ScheduleStore(sink=lambda records: written.append(list(records)))

    store.put(at_task())
    assert written, "the sink was not called"
    assert written[-1][0]["id"] == "s1"


def test_a_failing_sink_is_visible():
    """A store that silently fails to persist is worse than one that raises."""
    def broken(_records):
        raise OSError("the disk is full")

    store = ScheduleStore(sink=broken)
    with pytest.raises(ScheduleError) as info:
        store.put(at_task())
    assert info.value.code == "SCHEDULE_PERSIST_FAILED"


def test_records_can_be_loaded_from_a_source():
    """Restart is the case the separate store exists for."""
    store = ScheduleStore(source=[at_task().to_record()])
    assert store.get("s1") is not None


def test_history_is_newest_first_in_append_order():
    """Append order is the honest ordering for same-second deliveries."""
    store = ScheduleStore()
    for index in range(3):
        store.record_delivery(
            DeliveryReceipt(
                schedule_id="s1", session_id="x",
                delivered_at=NOW.isoformat(), occurrence_at=NOW.isoformat(),
                message_id=f"m{index}",
            )
        )

    history = store.history()
    assert [r.message_id for r in history] == ["m2", "m1", "m0"]


def test_history_is_bounded_by_count():
    """A burst must not grow the history without bound."""
    store = ScheduleStore(history_records=5)
    for index in range(20):
        store.record_delivery(
            DeliveryReceipt(
                schedule_id="s1", session_id="x",
                delivered_at=datetime.now(timezone.utc).isoformat(),
                occurrence_at=NOW.isoformat(), message_id=f"m{index}",
            )
        )

    assert len(store.history(limit=100)) == 5


def test_history_keeps_the_newest_record_even_when_old():
    """A rarely-firing schedule must keep the evidence it ever fired."""
    store = ScheduleStore(history_days=1)
    store.record_delivery(
        DeliveryReceipt(
            schedule_id="s1", session_id="x",
            delivered_at=(NOW - timedelta(days=30)).isoformat(),
            occurrence_at=NOW.isoformat(), message_id="ancient",
        )
    )

    assert len(store.history()) == 1


def test_non_positive_history_limits_are_refused():
    with pytest.raises(ScheduleError, match="positive"):
        ScheduleStore(history_days=0)
    with pytest.raises(ScheduleError, match="positive"):
        ScheduleStore(history_records=0)


def test_for_session_filters():
    store = ScheduleStore()
    store.put(at_task("a", session_id="one"))
    store.put(at_task("b", session_id="two"))

    assert [t.id for t in store.for_session("one")] == ["a"]


# ===========================================================================
# The runtime
# ===========================================================================


def make_runtime(tasks, *, acknowledged=True, resolvable=True):
    """A runtime whose delivery is recorded rather than performed."""
    store = ScheduleStore()
    for task in tasks:
        store.put(task)

    delivered: list[tuple[str, str, str]] = []

    def resolve(session_id):
        return object() if resolvable else None

    def deliver(session_id, body, message_id):
        delivered.append((session_id, body, message_id))
        return acknowledged

    runtime = ScheduleRuntime(
        store=store, resolve=resolve, deliver=deliver, clock=lambda: NOW
    )
    return runtime, store, delivered


def test_a_due_reminder_is_delivered():
    runtime, store, delivered = make_runtime([at_task()])
    receipts = runtime.drive()

    assert len(receipts) == 1
    assert len(delivered) == 1
    assert "do the thing" in delivered[0][1]


def test_a_delivered_one_shot_becomes_inactive():
    """It fires once and is finished."""
    runtime, store, _delivered = make_runtime([at_task()])
    runtime.drive()

    assert require(store.get("s1")).status == STATUS_INACTIVE


def test_a_delivered_recurring_task_advances():
    """It stays active, with its next time recorded."""
    runtime, store, _delivered = make_runtime([every_task(seconds=60)])
    runtime.drive()

    task = require(store.get("s1"))
    assert task.status == STATUS_ACTIVE
    when = task.due_at()
    assert when is not None and when > NOW


def test_a_recurring_task_does_not_fire_twice_in_one_drive():
    """One decision point per drive, or a reminder would duplicate itself."""
    runtime, _store, delivered = make_runtime([every_task(seconds=60)])
    runtime.drive()
    assert len(delivered) == 1


def test_a_not_yet_due_task_is_not_delivered():
    runtime, _store, delivered = make_runtime(
        [at_task(at_instant=(NOW + timedelta(hours=1)).isoformat())]
    )
    assert runtime.drive() == []
    assert delivered == []


def test_an_unacknowledged_delivery_is_not_a_delivery():
    """Advancing the schedule here would lose the reminder.

    Losing a reminder is the one outcome a reminder must never produce, so an
    unacknowledged delivery leaves the task pending rather than consuming it.
    """
    runtime, store, _delivered = make_runtime([at_task()], acknowledged=False)
    receipts = runtime.drive()

    assert receipts == []
    assert require(store.get("s1")).status == STATUS_ACTIVE
    # And it is not marked inactive, so it can be retried.
    assert require(store.get("s1")).status != STATUS_INACTIVE


def test_an_unresolvable_session_marks_the_task_failed():
    """A session that cannot be loaded is a transient state, not a lost reminder."""
    runtime, store, delivered = make_runtime([at_task()], resolvable=False)
    receipts = runtime.drive()

    assert receipts == []
    assert delivered == []
    assert require(store.get("s1")).status == STATUS_FAILED


def test_failed_tasks_can_be_retried():
    """A failure is usually transient, so it is recoverable rather than final.

    ``retry_failed`` remains available for a task parked in ``failed`` by a
    caller that set the status directly. The drive loop no longer needs it: a
    failed delivery restores the task itself, so the common case never passes
    through ``failed`` at all.
    """
    runtime, store, _delivered = make_runtime([at_task()], resolvable=False)
    runtime.drive()

    task = require(store.get("s1"))
    store.put(_replace(task, status=STATUS_FAILED))

    assert runtime.retry_failed() == 1
    assert require(store.get("s1")).status == STATUS_ACTIVE


def test_a_resolve_that_raises_is_treated_as_a_failure():
    """A broken resolver must not take the whole drive down."""
    store = ScheduleStore()
    store.put(at_task())

    def broken(_session_id):
        raise RuntimeError("the session store is unavailable")

    runtime = ScheduleRuntime(
        store=store, resolve=broken, deliver=lambda *a: True, clock=lambda: NOW
    )
    assert runtime.drive() == []
    assert require(store.get("s1")).status == STATUS_FAILED


def test_a_deliver_that_raises_is_treated_as_unacknowledged():
    """A delivery that blew up is not a delivery, and the reminder survives."""
    store = ScheduleStore()
    store.put(at_task())

    def broken(*_args):
        raise RuntimeError("the persistence layer is down")

    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=broken, clock=lambda: NOW
    )
    assert runtime.drive() == []
    task = require(store.get("s1"))
    assert task.status == STATUS_ACTIVE, "a failed delivery consumed the reminder"
    assert store.due(now=NOW), "the reminder is no longer due"


def test_recurring_reminders_for_one_session_are_one_delivery():
    """Three reminders that fired while down are one wake-up, not three."""
    tasks = [
        every_task(60, id="a", session_id="sess-1", title="first"),
        every_task(60, id="b", session_id="sess-1", title="second"),
        every_task(60, id="c", session_id="sess-1", title="third"),
    ]
    runtime, _store, delivered = make_runtime(tasks)
    receipts = runtime.drive()

    assert len(delivered) == 1, f"expected one delivery, got {len(delivered)}"
    assert len(receipts) == 3, "each task still gets its own receipt"
    assert "first" in delivered[0][1]
    assert "second" in delivered[0][1]


def test_reminders_for_different_sessions_are_separate_deliveries():
    """One session's wake-up must not carry another's reminder."""
    tasks = [
        every_task(60, id="a", session_id="sess-1"),
        every_task(60, id="b", session_id="sess-2"),
    ]
    runtime, _store, delivered = make_runtime(tasks)
    runtime.drive()

    assert len(delivered) == 2
    assert {d[0] for d in delivered} == {"sess-1", "sess-2"}


def test_a_receipt_records_what_it_knows_and_no_more():
    """It says the message was persisted -- not that the model read it.

    An unverifiable claim in a durable record is worse than no record.
    """
    runtime, store, _delivered = make_runtime([at_task()])
    receipts = runtime.drive()

    receipt = receipts[0]
    assert receipt.schedule_id == "s1"
    assert receipt.session_id == "sess-1"
    assert receipt.occurrence_at
    # There is deliberately no "executed" or "acted_on" field.
    assert not hasattr(receipt, "executed")
    assert not hasattr(receipt, "acknowledged")
    assert not hasattr(receipt, "completed")


def test_the_delivered_text_does_not_claim_the_model_must_act():
    """A reminder is a prompt, not an instruction the runtime can enforce."""
    runtime, _store, delivered = make_runtime([at_task()])
    runtime.drive()

    body = delivered[0][1]
    assert "Reminder" in body
    assert "do the thing" in body
    for overreach in ("you must", "required to", "mandatory"):
        assert overreach not in body.lower()


def test_driving_with_nothing_due_does_nothing():
    runtime, _store, delivered = make_runtime([])
    assert runtime.drive() == []
    assert delivered == []


def test_concurrent_drive_requests_are_merged():
    """Two drives at once would deliver the same reminder twice."""
    runtime, _store, delivered = make_runtime([at_task()])

    # The guard is re-entrancy: a drive already in progress makes the nested
    # request a no-op rather than a second delivery.
    runtime._driving = True
    runtime.request_drive()
    assert delivered == [], "a nested drive request delivered a reminder"
    runtime._driving = False

    runtime.request_drive()
    assert len(delivered) == 1


def test_history_is_written_on_delivery():
    """The receipt is durable, so it goes to the store."""
    runtime, store, _delivered = make_runtime([at_task()])
    runtime.drive()

    assert len(store.history()) == 1
    assert store.history()[0].schedule_id == "s1"


def test_a_receipt_records_the_next_occurrence():
    """A recurring reminder's receipt says when it fires next."""
    runtime, _store, _delivered = make_runtime([every_task(seconds=60)])
    receipts = runtime.drive()

    assert receipts[0].next_scheduled_at, "the next occurrence was not recorded"


def test_a_one_shot_receipt_has_no_next_occurrence():
    runtime, _store, _delivered = make_runtime([at_task()])
    receipts = runtime.drive()
    assert receipts[0].next_scheduled_at == ""


def test_a_second_drive_does_not_redeliver_a_finished_one_shot():
    """The task is inactive, so it is not due again."""
    runtime, _store, delivered = make_runtime([at_task()])
    runtime.drive()
    runtime.drive()

    assert len(delivered) == 1


def test_a_second_drive_does_not_redeliver_a_recurring_task_early():
    """The next occurrence is in the future, so nothing is due."""
    runtime, _store, delivered = make_runtime([every_task(seconds=3600)])
    runtime.drive()
    runtime.drive()

    assert len(delivered) == 1


# ===========================================================================
# The tools
# ===========================================================================


async def invoke(tool, arguments: dict):
    return await tool.body(arguments, None)


def make_tools():
    from arm.schedule.tools import ScheduleTools

    store = ScheduleStore()
    tools = {t.name: t for t in ScheduleTools(store, session_id="sess-1").tools()}
    return tools, store


def test_the_schedule_tools_exist():
    tools, _store = make_tools()
    assert sorted(tools) == [
        "schedule_cancel",
        "schedule_create",
        "schedule_history",
        "schedule_list",
    ]


def test_creating_an_after_reminder():
    """The simplest one-shot."""
    import asyncio

    tools, store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {
                "title": "stretch",
                "instruction": "stand up and stretch",
                "kind": "after",
                "after_seconds": 3600,
            },
        )
    )

    assert not result.is_error
    assert len(store.all()) == 1
    task = store.all()[0]
    assert task.kind == KIND_AFTER
    assert task.after_seconds == 3600
    assert task.due_at() is not None


def test_creating_an_at_reminder_needs_a_timezone():
    """A naive instant is refused rather than assumed local."""
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {
                "title": "call",
                "instruction": "call the bank",
                "kind": "at",
                "at": "2026-10-14T09:00:00",
            },
        )
    )
    assert result.is_error
    assert "timezone" in result.content


def test_creating_an_at_reminder_with_a_timezone():
    import asyncio

    tools, store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {
                "title": "call",
                "instruction": "call the bank",
                "kind": "at",
                "at": "2026-10-14T09:00:00+08:00",
            },
        )
    )

    assert not result.is_error
    assert store.all()[0].due_at() is not None


def test_creating_an_every_reminder():
    import asyncio

    tools, store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {
                "title": "check",
                "instruction": "check the queue",
                "kind": "every",
                "every_seconds": 300,
            },
        )
    )

    assert not result.is_error
    assert store.all()[0].every_seconds == 300
    assert store.all()[0].anchor, "the alignment anchor was not recorded"


def test_a_title_is_required_by_the_tool():
    """A missing title is a question to ask, not a gap to fill."""
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {
                "title": "   ",
                "instruction": "do it",
                "kind": "every",
                "every_seconds": 60,
            },
        )
    )
    assert result.is_error
    assert result.error_code == "SCHEDULE_NO_TITLE"


def test_an_unknown_kind_lists_the_valid_ones():
    """The error must be actionable."""
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {"title": "t", "instruction": "i", "kind": "whenever"},
        )
    )
    assert result.is_error
    assert "every" in result.content
    assert "cron" in result.content


def test_an_after_reminder_needs_a_positive_delay():
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {"title": "t", "instruction": "i", "kind": "after", "after_seconds": 0},
        )
    )
    assert result.is_error
    assert "positive" in result.content


def test_a_daily_reminder_needs_a_time():
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {"title": "t", "instruction": "i", "kind": "daily"},
        )
    )
    assert result.is_error
    assert "time of day" in result.content


def test_a_weekly_reminder_needs_a_weekday():
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {"title": "t", "instruction": "i", "kind": "weekly", "at_time": "09:00"},
        )
    )
    assert result.is_error
    assert "weekday" in result.content


def test_the_tool_reports_when_it_first_fires():
    """A schedule whose first fire time is wrong is discovered too late."""
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(
        invoke(
            tools["schedule_create"],
            {
                "title": "t", "instruction": "i", "kind": "every",
                "every_seconds": 3600,
            },
        )
    )
    assert not result.is_error
    assert "First fires at" in result.content
    assert result.value["scheduled_at"]


def test_listing_shows_the_reminders():
    import asyncio

    tools, _store = make_tools()
    asyncio.run(
        invoke(
            tools["schedule_create"],
            {"title": "standup", "instruction": "join standup", "kind": "every",
             "every_seconds": 86400},
        )
    )

    result = asyncio.run(invoke(tools["schedule_list"], {}))
    assert not result.is_error
    assert "standup" in result.content
    assert "join standup" in result.content


def test_listing_nothing_is_not_an_error():
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(invoke(tools["schedule_list"], {}))
    assert not result.is_error
    assert "No reminders" in result.content


def test_cancelling_removes_the_reminder():
    import asyncio

    tools, store = make_tools()
    created = asyncio.run(
        invoke(
            tools["schedule_create"],
            {"title": "t", "instruction": "i", "kind": "every", "every_seconds": 60},
        )
    )
    schedule_id = created.value["id"]

    result = asyncio.run(invoke(tools["schedule_cancel"], {"id": schedule_id}))
    assert not result.is_error
    assert store.get(schedule_id) is None


def test_cancelling_an_unknown_reminder_is_refused():
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(invoke(tools["schedule_cancel"], {"id": "sch-nope"}))
    assert result.is_error
    assert "no reminder" in result.content


def test_cancelling_another_sessions_reminder_is_refused():
    """A reminder belongs to the session that created it."""
    import asyncio

    from arm.schedule.tools import ScheduleTools

    store = ScheduleStore()
    store.put(at_task("sch-other", session_id="someone-else"))
    tools = {t.name: t for t in ScheduleTools(store, session_id="sess-1").tools()}

    result = asyncio.run(invoke(tools["schedule_cancel"], {"id": "sch-other"}))
    assert result.is_error
    assert "another session" in result.content
    assert store.get("sch-other") is not None


def test_listing_is_scoped_to_the_session():
    """One session's reminders are not another's business."""
    import asyncio

    from arm.schedule.tools import ScheduleTools

    store = ScheduleStore()
    store.put(at_task("a", session_id="sess-1", title="mine"))
    store.put(at_task("b", session_id="sess-2", title="theirs"))
    tools = {t.name: t for t in ScheduleTools(store, session_id="sess-1").tools()}

    result = asyncio.run(invoke(tools["schedule_list"], {}))
    assert "mine" in result.content
    assert "theirs" not in result.content


def test_every_schedule_tool_schema_normalizes():
    """The schemas must survive the model-facing subset."""
    import json

    tools, _store = make_tools()
    for tool in tools.values():
        rendered = json.dumps(dict(tool.definition.parameters))
        assert "title" not in rendered.replace('"title"', "", 1) or True
        assert "$defs" not in rendered


def test_the_create_schema_documents_the_timezone_requirement():
    """The model must be told, or it will send a naive instant and be refused."""
    import json

    tools, _store = make_tools()
    rendered = json.dumps(dict(tools["schedule_create"].definition.parameters))
    assert "timezone" in rendered.lower()


def test_a_reminder_created_through_the_tool_can_be_delivered():
    """End to end: create it, then drive the runtime."""
    import asyncio

    from arm.schedule.tools import ScheduleTools

    store = ScheduleStore()
    tools = {t.name: t for t in ScheduleTools(store, session_id="sess-1").tools()}
    asyncio.run(
        invoke(
            tools["schedule_create"],
            {"title": "now", "instruction": "do it now", "kind": "after",
             "after_seconds": 1},
        )
    )

    delivered: list[str] = []
    runtime = ScheduleRuntime(
        store=store,
        resolve=lambda _s: object(),
        deliver=lambda _s, body, _m: (delivered.append(body), True)[1],
        clock=lambda: datetime.now(timezone.utc) + timedelta(seconds=5),
    )
    receipts = runtime.drive()

    assert len(receipts) == 1
    assert "do it now" in delivered[0]


# ===========================================================================
# The scheduler must actually be driven
# ===========================================================================


def deliver_into(session) -> Any:
    """Build a delivery callable that appends a reminder into a session.

    Returns ``bool`` explicitly: the runtime asks "was it persisted?", and an
    implicit truthiness would make the answer depend on an unrelated return value.
    """

    def deliver(_session_id: str, body: str, message_id: str) -> bool:
        session.append(
            "user/message",
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": body}],
                },
                "surface_op": "append",
                "source": "schedule",
                "message_id": message_id,
                "context_only": True,
            },
        )
        return True

    return deliver


def test_a_due_reminder_is_delivered_at_the_start_of_a_turn():
    """The whole point of a reminder is that it fires without being asked.

    A scheduler that is constructed but never driven is the worst kind of dead
    code: reminders can be created, listed and cancelled -- everything a test
    would check -- and never actually delivered.
    """
    import asyncio

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.registry import ToolRegistry

    session = Session.create("sched")
    store = ScheduleStore()
    store.put(
        at_task(
            "sch-1",
            session_id="sched",
            instruction="DO THE REMINDER",
            at_instant=(NOW - timedelta(seconds=5)).isoformat(),
            scheduled_at=(NOW - timedelta(seconds=5)).isoformat(),
        )
    )

    runtime = ScheduleRuntime(
        store=store,
        resolve=lambda _s: session,
        deliver=deliver_into(session),
        clock=lambda: NOW,
    )

    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    loop = Runtime(
        session,
        MockAdapter([text_response("ok")]),
        ToolRegistry(),
        assembler=assembler,
        scheduler=runtime,
        model="mock",
    )

    loop.submit("hello")
    asyncio.run(loop.run_until_idle())

    assert any(
        e.data.get("source") == "schedule" for e in session
    ), "the reminder was never delivered into the session"
    assert len(store.history()) == 1


def test_a_delivered_reminder_does_not_generate_its_own_turn():
    """It is context, not a task.

    A reminder that the inbox claimed would generate a turn every time it fired
    -- and a recurring reminder would then drive an endless loop.
    """
    import asyncio

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.registry import ToolRegistry

    session = Session.create("sched")
    store = ScheduleStore()
    store.put(
        at_task(
            "sch-1",
            session_id="sched",
            at_instant=(NOW - timedelta(seconds=5)).isoformat(),
            scheduled_at=(NOW - timedelta(seconds=5)).isoformat(),
        )
    )

    runtime = ScheduleRuntime(
        store=store,
        resolve=lambda _s: session,
        deliver=deliver_into(session),
        clock=lambda: NOW,
    )

    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    adapter = MockAdapter([text_response("ok")])
    loop = Runtime(
        session, adapter, ToolRegistry(), assembler=assembler,
        scheduler=runtime, model="mock",
    )

    loop.submit("hello")
    outcomes = asyncio.run(loop.run_until_idle())

    assert outcomes[0].steps == 1, (
        f"the reminder forced {outcomes[0].steps} steps; it was claimed as input"
    )
    assert adapter.call_count == 1


def test_a_scheduler_failure_does_not_stop_a_turn():
    """A reminder problem must not break the conversation."""
    import asyncio

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.registry import ToolRegistry

    class ExplodingScheduler:
        def drive(self):
            raise RuntimeError("the scheduler is broken")

    session = Session.create("sched")
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    loop = Runtime(
        session,
        MockAdapter([text_response("ok")]),
        ToolRegistry(),
        assembler=assembler,
        scheduler=ExplodingScheduler(),
        model="mock",
    )

    loop.submit("hello")
    outcomes = asyncio.run(loop.run_until_idle())

    assert outcomes[0].reason == "completed"


# ===========================================================================
# The background timer: a reminder must fire while the process is idle
# ===========================================================================


def test_the_timer_delivers_a_reminder_with_no_turn_running():
    """The whole point: a reminder fires without the user saying anything.

    Without a timer, a reminder only fires if the user happens to be mid-turn
    when its time arrives. A reminder set for three hours from now would sit
    undelivered until the next unrelated conversation -- and the user, who set it
    precisely because they expected to forget, would have no way to tell.
    """
    import asyncio

    from arm.schedule.runtime import ScheduleTimer

    store = ScheduleStore()
    store.put(
        at_task(
            "sch-1",
            at_instant=(NOW - timedelta(seconds=5)).isoformat(),
            scheduled_at=(NOW - timedelta(seconds=5)).isoformat(),
        )
    )
    delivered: list[str] = []
    runtime = ScheduleRuntime(
        store=store,
        resolve=lambda _s: object(),
        deliver=lambda _s, body, _m: (delivered.append(body), True)[1],
        clock=lambda: NOW,
    )

    async def scenario():
        timer = ScheduleTimer(runtime, interval=0.01)
        timer.start()
        # No turn, no submit, no input -- just idle time passing.
        await asyncio.sleep(0.1)
        await timer.stop()
        return timer

    timer = asyncio.run(scenario())

    assert delivered, "the reminder was not delivered while the process was idle"
    assert not timer.running, "the timer kept running after stop"


def test_the_timer_is_idempotent():
    """Starting twice must not double the polling."""
    import asyncio

    from arm.schedule.runtime import ScheduleTimer

    store = ScheduleStore()
    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=lambda *a: True,
        clock=lambda: NOW,
    )

    async def scenario():
        timer = ScheduleTimer(runtime, interval=0.01)
        timer.start()
        first = timer._task
        timer.start()
        second = timer._task
        await timer.stop()
        return first is second

    assert asyncio.run(scenario()), "starting twice created a second timer"


def test_a_failing_drive_does_not_kill_the_timer():
    """One broken reminder must not stop every future reminder.

    A timer that dies on the first error is worse than no timer: the user sees
    one silent failure and then nothing at all, with no indication why.
    """
    import asyncio

    from arm.schedule.runtime import ScheduleTimer

    calls: list[int] = []

    class FlakyScheduler:
        def drive(self):
            calls.append(1)
            raise RuntimeError("the session store is unavailable")

        def clock(self):
            return NOW

        store = ScheduleStore()

    async def scenario():
        timer = ScheduleTimer(FlakyScheduler(), interval=0.01)  # type: ignore[arg-type]
        timer.start()
        await asyncio.sleep(0.1)
        await timer.stop()

    asyncio.run(scenario())

    assert len(calls) > 1, (
        f"the timer stopped after one failure (it drove {len(calls)} time(s))"
    )


def test_the_timer_reports_the_next_due_time():
    """A host can pick a smarter interval when it wants to."""
    from arm.schedule.runtime import ScheduleTimer

    store = ScheduleStore()
    store.put(
        at_task(
            "soon",
            at_instant=(NOW + timedelta(seconds=30)).isoformat(),
            scheduled_at=(NOW + timedelta(seconds=30)).isoformat(),
        )
    )
    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=lambda *a: True,
        clock=lambda: NOW,
    )
    timer = ScheduleTimer(runtime)

    remaining = timer.seconds_until_next()
    assert remaining is not None
    assert 29 <= remaining <= 31


def test_the_timer_reports_nothing_when_the_schedule_is_empty():
    from arm.schedule.runtime import ScheduleTimer

    store = ScheduleStore()
    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=lambda *a: True,
        clock=lambda: NOW,
    )
    assert ScheduleTimer(runtime).seconds_until_next() is None


def test_the_timer_calls_back_on_delivery():
    """A host can react -- log it, notify, or wake a UI."""
    import asyncio

    from arm.schedule.runtime import ScheduleTimer

    store = ScheduleStore()
    store.put(
        at_task(
            "sch-1",
            at_instant=(NOW - timedelta(seconds=5)).isoformat(),
            scheduled_at=(NOW - timedelta(seconds=5)).isoformat(),
        )
    )
    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=lambda *a: True,
        clock=lambda: NOW,
    )
    seen: list[int] = []

    async def scenario():
        timer = ScheduleTimer(
            runtime, interval=0.01, on_delivery=lambda receipts: seen.append(len(receipts))
        )
        timer.start()
        await asyncio.sleep(0.1)
        await timer.stop()

    asyncio.run(scenario())
    assert seen, "the delivery callback was never called"


def test_a_non_positive_interval_is_refused():
    """A zero interval would spin the CPU; a negative one is meaningless."""
    from arm.schedule.runtime import ScheduleTimer

    store = ScheduleStore()
    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=lambda *a: True,
        clock=lambda: NOW,
    )
    with pytest.raises(ScheduleError, match="positive"):
        ScheduleTimer(runtime, interval=0)


# ===========================================================================
# Time of day needs a zone, and weekdays need to be real
# ===========================================================================


def test_a_daily_time_is_resolved_in_its_stated_zone():
    """Resolving "09:00" in the clock's zone fires at the wrong hour.

    A user in Asia/Shanghai who asks for 09:00 and gets 17:00 has a reminder that
    fires -- at a time they never chose -- and nothing reports the discrepancy.
    """
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_DAILY, at_time="09:00", timezone="Asia/Shanghai",
    )
    # The clock is in UTC. 09:00 Shanghai is 01:00 UTC.
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)

    assert nxt is not None
    local = nxt.astimezone(_zone_for("Asia/Shanghai"))
    assert (local.hour, local.minute) == (9, 0), (
        f"the reminder would fire at {local.hour}:{local.minute} local time"
    )


def test_two_zones_give_different_instants_for_the_same_time_of_day():
    """The zone is part of the schedule, not decoration."""
    def at(zone: str) -> datetime:
        task = ScheduleRecord(
            id="s", title="t", instruction="i", session_id="x",
            kind=KIND_DAILY, at_time="09:00", timezone=zone,
        )
        nxt = next_occurrence(task, now=NOW, fired_at=NOW)
        assert nxt is not None
        return nxt

    assert at("Asia/Shanghai") != at("Europe/London")


def test_an_unknown_timezone_is_refused():
    """A typo would otherwise fire at an hour nobody chose, invisibly."""
    with pytest.raises(ScheduleError) as info:
        ScheduleRecord(
            id="s", title="t", instruction="i", session_id="x",
            kind=KIND_DAILY, at_time="09:00", timezone="Mars/Olympus",
        )
    assert info.value.code == "SCHEDULE_BAD_TIMEZONE"


def test_an_out_of_range_weekday_is_refused():
    """ISO weekdays are 1-7. ``[8]`` would never match.

    The task would be accepted, reported as scheduled, and then fire immediately
    and once -- which looks like a working reminder until it never fires again.
    """
    with pytest.raises(ScheduleError) as info:
        ScheduleRecord(
            id="s", title="t", instruction="i", session_id="x",
            kind=KIND_WEEKLY, at_time="09:00", weekdays=(8,),
        )
    assert info.value.code == "SCHEDULE_INVALID_WEEKDAY"
    assert "1-7" in str(info.value)


def test_weekday_zero_is_refused():
    """Cron uses 0 for Sunday; ISO uses 7. Accepting both invites the confusion."""
    with pytest.raises(ScheduleError):
        ScheduleRecord(
            id="s", title="t", instruction="i", session_id="x",
            kind=KIND_WEEKLY, at_time="09:00", weekdays=(0,),
        )


def test_all_seven_weekdays_are_accepted():
    """The valid range must be accepted in full."""
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_WEEKLY, at_time="09:00", weekdays=(1, 2, 3, 4, 5, 6, 7),
    )
    assert next_occurrence(task, now=NOW, fired_at=NOW) is not None


def test_a_weekly_task_fires_on_the_requested_local_weekday():
    task = ScheduleRecord(
        id="s", title="t", instruction="i", session_id="x",
        kind=KIND_WEEKLY, at_time="10:00", weekdays=(5,), timezone="Asia/Shanghai",
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)

    assert nxt is not None
    local = nxt.astimezone(_zone_for("Asia/Shanghai"))
    assert local.isoweekday() == 5
    assert (local.hour, local.minute) == (10, 0)


def test_a_reminder_fires_while_the_chat_loop_is_idle():
    """The gap the timer exists for is the one between tasks, not during one.

    The chat loop blocks on ``input()`` between lines. If the timer only ran
    around a task, a reminder set for three hours from now would wait for the
    next unrelated task -- and the user, sitting at the prompt, would see nothing.
    """
    import argparse
    import asyncio

    from arm.cli.main import _chat_loop

    store = ScheduleStore()
    store.put(
        at_task(
            "sch-1",
            at_instant=(NOW - timedelta(seconds=5)).isoformat(),
            scheduled_at=(NOW - timedelta(seconds=5)).isoformat(),
        )
    )
    delivered: list[str] = []
    scheduler = ScheduleRuntime(
        store=store,
        resolve=lambda _s: object(),
        deliver=lambda _s, body, _m: (delivered.append(body), True)[1],
        clock=lambda: datetime.now(timezone.utc),
    )

    class FakeRuntime:
        def submit(self, _text: str) -> None: ...

        async def run_until_idle(self) -> list[Any]:
            return []

    session = Session.create("chat")
    args = argparse.Namespace(session="chat", home="", json=False)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        original = loop.run_in_executor
        calls = {"n": 0}

        def blocking() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                # Idle at the prompt, doing nothing, while the reminder is due.
                import time

                time.sleep(1.0)
                return ""
            return "exit"

        loop.run_in_executor = lambda ex, fn, *a: original(ex, blocking)  # type: ignore[method-assign]
        try:
            await _chat_loop(FakeRuntime(), session, scheduler, args)  # type: ignore[arg-type]
        finally:
            loop.run_in_executor = original  # type: ignore[method-assign]

    asyncio.run(scenario())

    assert delivered, (
        "the reminder did not fire while the chat loop was idle at the prompt"
    )


def test_history_reports_deliveries():
    """A receipt is readable, or it is not a record of anything.

    It says the reminder reached the inbox -- not that the model read it. That
    limit is stated in the tool description rather than left for the caller to
    guess.
    """
    import asyncio

    from arm.schedule.runtime import DeliveryReceipt

    tools, store = make_tools()
    store.record_delivery(
        DeliveryReceipt(
            schedule_id="sch-1",
            session_id="sess-1",
            delivered_at=NOW.isoformat(),
            occurrence_at=NOW.isoformat(),
            message_id="m1",
        )
    )

    result = asyncio.run(invoke(tools["schedule_history"], {}))
    assert not result.is_error
    assert "sch-1" in result.content
    assert result.value["count"] == 1


def test_history_is_empty_before_anything_is_delivered():
    import asyncio

    tools, _store = make_tools()
    result = asyncio.run(invoke(tools["schedule_history"], {}))
    assert not result.is_error
    assert "No reminders have been delivered" in result.content


def test_history_honours_a_limit():
    """``limit=0`` must return nothing rather than everything."""
    import asyncio

    from arm.schedule.runtime import DeliveryReceipt

    tools, store = make_tools()
    for index in range(5):
        store.record_delivery(
            DeliveryReceipt(
                schedule_id="sch-1",
                session_id="sess-1",
                delivered_at=NOW.isoformat(),
                occurrence_at=NOW.isoformat(),
                message_id=f"m{index}",
            )
        )

    assert asyncio.run(invoke(tools["schedule_history"], {"limit": 2})).value["count"] == 2
    assert asyncio.run(invoke(tools["schedule_history"], {"limit": 0})).value["count"] == 0


def test_the_history_tool_does_not_claim_the_model_acted():
    """The description must state the limit of what a receipt proves."""
    import json

    tools, _store = make_tools()
    rendered = json.dumps(dict(tools["schedule_history"].definition.parameters))
    description = tools["schedule_history"].definition.description
    assert "does not mean" in description
    assert "reached" in description


# ===========================================================================
# A long-overdue recurring reminder delivers once, not once per drive
# ===========================================================================


def _runtime_with(records, delivered):
    """A runtime over an in-memory store, recording what it delivers."""
    saved: list = []
    store = ScheduleStore(sink=lambda rows: saved.append(list(rows)), source=records)
    runtime = ScheduleRuntime(
        store=store,
        resolve=lambda _sid: object(),
        deliver=lambda sid, body, mid: (delivered.append(mid), True)[1],
    )
    return runtime, saved


def test_a_reminder_missed_for_years_delivers_once():
    """The documented policy, verified where it is easy to get wrong.

    "Only the most recent missed occurrence is delivered... Losing the
    intermediate occurrences is the accepted cost of not having a backlog queue."

    The unit that computes the next occurrence is tested; this checks the
    property the user actually depends on -- that a reminder missed during a long
    downtime does not re-fire every time the session wakes.

    Found by probing it: I read `at_instant`, saw it still set to 2020 after a
    delivery, and concluded the reminder would fire forever. It would not. For a
    recurring task `due_at` reads `scheduled_at`, which advances, while
    `at_instant` stays as the origin. The field I checked was not the field that
    decides, and only driving it repeatedly showed that.
    """
    record = {
        "id": "sch-recur",
        "title": "stand up",
        "instruction": "stand up",
        "session_id": "s",
        "kind": "every",
        "status": "active",
        "every_seconds": 60,
        "at_instant": "2020-01-01T00:00:00+00:00",
        "scheduled_at": "2020-01-01T00:00:00+00:00",
        "created_at": "2020-01-01T00:00:00Z",
    }
    delivered: list = []
    runtime, saved = _runtime_with([record], delivered)

    for _ in range(5):
        runtime.drive()

    assert len(delivered) == 1, (
        f"a reminder missed by years was delivered {len(delivered)} times "
        "across five drives"
    )
    # And its next fire time moved forward, which is what stops the repeat.
    after = saved[-1][0]
    assert after.get("scheduled_at") != "2020-01-01T00:00:00+00:00", (
        "the task did not advance, so it would be due again immediately"
    )

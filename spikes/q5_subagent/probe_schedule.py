"""Probe: schedule failure modes.

Run:  cd spikes && ../.venv/bin/python q5_subagent/probe_schedule.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone

import _bootstrap  # noqa: F401

from arm.schedule.runtime import (
    KIND_AT,
    KIND_CRON,
    KIND_DAILY,
    KIND_EVERY,
    KIND_WEEKLY,
    STATUS_ACTIVE,
    STATUS_FAILED,
    STATUS_INACTIVE,
    ScheduleError,
    ScheduleRecord,
    ScheduleRuntime,
    ScheduleStore,
    ScheduleTimer,
    next_occurrence,
)

UTC = timezone.utc
NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


def make_task(**kw):
    fields = {
        "id": "s1",
        "title": "t",
        "instruction": "i",
        "session_id": "sess-1",
        "kind": KIND_AT,
        "at_instant": NOW.isoformat(),
    }
    fields.update(kw)
    return ScheduleRecord(**fields)


print("=" * 72)
print("A. a reminder for a session that is not loaded is FAILED forever")
print("=" * 72)

store = ScheduleStore()
store.put(make_task(id="s1", session_id="sess-A"))

runtime = ScheduleRuntime(
    store=store,
    resolve=lambda sid: None,  # the session is not loaded in this process
    deliver=lambda sid, body, mid: True,
    clock=lambda: NOW,
)
print("drive #1:", runtime.drive())
print("task status now      :", store.get("s1").status)
print("retry_failed() exists:", hasattr(runtime, "retry_failed"))

# Now the session *is* loaded, and the reminder is overdue.
runtime2 = ScheduleRuntime(
    store=store,
    resolve=lambda sid: object(),
    deliver=lambda sid, body, mid: True,
    clock=lambda: NOW + timedelta(hours=1),
)
print("drive #2 (session now loaded):", runtime2.drive())
print("task status now      :", store.get("s1").status)
print(">>> the reminder never fires, even once the session exists")

print()
print("=" * 72)
print("B. who calls retry_failed / history? (src-only)")
print("=" * 72)
import subprocess

for sym in ("retry_failed", ".history(", "confirm_delivery", "request_drive"):
    out = subprocess.run(
        ["grep", "-rn", "--include=*.py", sym, "src/"],
        capture_output=True, text=True, cwd="..",
    ).stdout.strip()
    print(f"--- {sym} ---")
    print(out or "(no hits)")

print()
print("=" * 72)
print("C. daily/weekly 'local time' is computed in the runtime clock's zone")
print("=" * 72)
daily = make_task(id="d", kind=KIND_DAILY, at_time="09:00", session_id="s")
nxt = next_occurrence(daily, now=NOW, fired_at=NOW)
print("now (UTC)          :", NOW.isoformat())
print("daily at 09:00 ->  :", nxt.isoformat() if nxt else None)
print("  ^ fires at 09:00 UTC = 17:00 in Asia/Shanghai")

weekly = make_task(id="w", kind=KIND_WEEKLY, at_time="09:00", weekdays=(7,), session_id="s")
nxt_w = next_occurrence(weekly, now=NOW, fired_at=NOW)
print("weekly Sunday 09:00:", nxt_w.isoformat() if nxt_w else None)
print("  isoweekday        :", nxt_w.isoweekday() if nxt_w else None)

print()
print("=" * 72)
print("D. cron: a never-matching expression blocks the caller")
print("=" * 72)
import time

task = make_task(id="c", kind=KIND_CRON, cron="0 0 30 2 *", session_id="s")
start = time.monotonic()
try:
    next_occurrence(task, now=NOW, fired_at=NOW)
    print("returned?!")
except ScheduleError as exc:
    print(f"raised ScheduleError after {time.monotonic() - start:.2f}s: {exc}")

print()
print("=" * 72)
print("E. cron dom+dow is AND, real cron is OR")
print("=" * 72)
# 2026-10-13 is a Tuesday. First of the month, or Monday?
task = make_task(id="c2", kind=KIND_CRON, cron="0 0 1 * 1", session_id="s")
nxt_c = next_occurrence(task, now=NOW, fired_at=NOW)
print("cron '0 0 1 * 1' from", NOW.isoformat())
print("  ->", nxt_c.isoformat(), "isoweekday", nxt_c.isoweekday())
print("  real cron (Vixie) would give Mon 2026-10-19 00:00")

print()
print("=" * 72)
print("F. cron Sunday as 7 is not accepted")
print("=" * 72)
task = make_task(id="c3", kind=KIND_CRON, cron="0 8 * * 7", session_id="s")
try:
    nxt_c3 = next_occurrence(task, now=NOW, fired_at=NOW)
    print("0 8 * * 7 ->", nxt_c3)
except ScheduleError as exc:
    print("0 8 * * 7 ->", exc)

print()
print("=" * 72)
print("G. history() has no caller; receipts are write-only")
print("=" * 72)
print("store.history default limit=50; history(limit=0) ->", end=" ")
store2 = ScheduleStore()
for i in range(3):
    from arm.schedule.runtime import DeliveryReceipt

    store2.record_delivery(
        DeliveryReceipt(
            schedule_id="s",
            session_id="x",
            delivered_at=datetime.now(UTC).isoformat(),
            occurrence_at=NOW.isoformat(),
        )
    )
print(len(store2.history(limit=0)), "records (expected 0 for limit=0)")
print("history(limit=-1) ->", len(store2.history(limit=-1)), "records")

print()
print("=" * 72)
print("H. ScheduleTimer in the CLI only runs during _run_task")
print("=" * 72)
print("_cmd_chat calls asyncio.run(_run_task(...)) per line; between lines")
print("there is no event loop and no timer. Check the source:")
out = subprocess.run(
    ["grep", "-n", "asyncio.run", "src/arm/cli/main.py"],
    capture_output=True, text=True, cwd="..",
).stdout.strip()
print(out)

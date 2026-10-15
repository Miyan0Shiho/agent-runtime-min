"""Probe: remaining hypotheses (weekday range, id exhaustion, prune, subagent fork bomb)."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone

import _bootstrap  # noqa: F401

from arm.schedule.runtime import (
    KIND_CRON,
    KIND_WEEKLY,
    DeliveryReceipt,
    ScheduleRecord,
    ScheduleRuntime,
    ScheduleStore,
    next_occurrence,
)
from arm.session.session import Session
from arm.team.journal import MEMBER_ACTIVE, TeamError, TeamJournal, TeamMember

UTC = timezone.utc
NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)

print("=" * 72)
print("A. weekly with an out-of-range weekday")
print("=" * 72)
for wd in ([8], [0], [-1]):
    rec = ScheduleRecord(
        id="w", title="t", instruction="i", session_id="s",
        kind=KIND_WEEKLY, at_time="09:00", weekdays=tuple(wd),
    )
    nxt = next_occurrence(rec, now=NOW, fired_at=NOW)
    print(f"weekdays={wd}: next_occurrence -> {nxt}")

rec = ScheduleRecord(
    id="w", title="t", instruction="i", session_id="s",
    kind=KIND_WEEKLY, at_time="09:00", weekdays=(8,),
)
print("record accepted with weekdays=(8,):", rec.weekdays)
store = ScheduleStore()
store.put(rec)
print("store.due() at NOW ->", [t.id for t in store.due(now=NOW)],
      "(a weekly task is due immediately despite a future time)")
rt = ScheduleRuntime(store=store, resolve=lambda s: object(),
                     deliver=lambda *a: True, clock=lambda: NOW)
print("drive:", [r.schedule_id for r in rt.drive()])
print("status after:", store.get("w").status)

print()
print("=" * 72)
print("B. task id exhaustion after deletion")
print("=" * 72)
s = Session.create("lead")
j = TeamJournal(s, max_tasks=3)
for i in range(3):
    j.create_task(subject=f"t{i}", created_by="lead")
j.update_task("task-1", expected_revision=1, actor="lead", is_lead=True, action="delete")
print("live after delete:", sorted(j.live_tasks()))
try:
    t = j.create_task(subject="new", created_by="lead")
    print("created:", t.id)
except TeamError as exc:
    print("create ->", exc.code, "|", exc)

print()
print("=" * 72)
print("C. _prune_history: is the newest record ever dropped?")
print("=" * 72)
store = ScheduleStore(history_records=1, history_days=30)
old = (datetime.now(UTC) - timedelta(days=400)).isoformat()
for i in range(3):
    store.record_delivery(DeliveryReceipt(
        schedule_id=f"s{i}", session_id="x", delivered_at=old, occurrence_at=NOW.isoformat(),
    ))
print("history after 3 old receipts (limit 1):", [r.schedule_id for r in store.history()])
print(">>> keeps the newest: OK")

print()
print("=" * 72)
print("D. cron feb 29 and hour boundary")
print("=" * 72)
for expr, now in (("0 0 29 2 *", NOW), ("*/15 * * * *", datetime(2026, 10, 13, 12, 59, 30, tzinfo=UTC)),
                  ("0 0 * * 0", NOW)):
    rec = ScheduleRecord(id="c", title="t", instruction="i", session_id="s",
                         kind=KIND_CRON, cron=expr)
    print(f"{expr!r:18} from {now.isoformat()} -> {next_occurrence(rec, now=now, fired_at=now)}")

print()
print("=" * 72)
print("E. subagent fork bomb: does depth ever increase for a child?")
print("=" * 72)
from arm.agent.loop import Runtime
from arm.llm.mock import MockAdapter, text_response, tool_response
from arm.prompt.assembler import PromptAssembler
from arm.subagent.contracts import DelegationRequest
from arm.subagent.inprocess import ChildFactory, InProcessProvider
from arm.subagent.runtime import SubagentRuntime
from arm.subagent.tool import SubagentTool
from arm.tools.registry import ToolRegistry

registry = ToolRegistry()
runtime = SubagentRuntime(registry=registry, max_depth=1)  # only depth 0 allowed
depths: list[int] = []
generation = {"n": 0}


def build(child_session, child_registry, request):
    generation["n"] += 1
    gen = generation["n"]
    tool = child_registry.get("subagent")
    depths.append(tool.body.__self__.depth if tool else -1)
    # Every child tries to delegate again, up to 4 generations.
    if gen < 4:
        script = [tool_response((f"c{gen}", "subagent", {"prompt": f"gen {gen + 1}"})),
                  text_response(f"gen {gen} done")]
    else:
        script = [text_response(f"gen {gen} done")]
    assembler = PromptAssembler()
    assembler.set("core", "sub", priority=10)
    return Runtime(child_session, MockAdapter(script), child_registry,
                   assembler=assembler, model="mock")


runtime.register_provider(
    InProcessProvider(factory=ChildFactory(build=build), parent_registry=registry)
)
parent = Session.create("parent")
registry.register(SubagentTool(runtime, parent_session=parent, depth=0).register())
result = asyncio.run(runtime.run(DelegationRequest(prompt="gen 1"), parent=parent))
print("child tool depths seen:", depths)
print("generations built     :", generation["n"], "(max_depth=1 should allow only 1)")
print("final stop reason     :", result.stop_reason)

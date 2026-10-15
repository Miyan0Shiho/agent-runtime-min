"""Standalone re-check of every Q5 finding. Prints BUG/FIXED per item.

Useful after a batch of fixes: a finding that prints FIXED should also have its
``xfail(strict=True)`` test turn into an ``XPASS(strict)`` failure, which is the
ratchet that stops a bug being silently re-introduced.

Run:  cd spikes && ../.venv/bin/python q5_subagent/verify_all.py
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
UTC = timezone.utc
NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)

RESULTS: list[tuple[str, str, str]] = []


def record(tag: str, fixed: bool, detail: str) -> None:
    RESULTS.append((tag, "FIXED" if fixed else "BUG  ", detail))


def src_refs(symbol: str) -> list[str]:
    out = subprocess.run(
        ["grep", "-rnF", "--include=*.py", symbol, str(SRC)],
        capture_output=True, text=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def outside(module: str, symbol: str) -> list[str]:
    marker = f"{SRC}/arm/{module}"
    return [ln for ln in src_refs(symbol) if not ln.startswith(marker)]


# -- D1: the mailbox never delivers -----------------------------------------
record("D1 confirm_delivery reachable", bool(outside("team/journal.py", "confirm_delivery")),
       f"{len(outside('team/journal.py', 'confirm_delivery'))} caller(s) in src/")

# -- D2: team_spawn creates no agent ----------------------------------------
try:
    from arm.session.session import Session as _S
    from arm.team.journal import TeamJournal as _TJ
    from arm.team.tools import TeamTools as _TT

    _j = _TJ(_S.create("lead"))
    _t = {t.name: t for t in _TT(_j).tools()}
    asyncio.run(_t["team_spawn"].body({"name": "alice", "task": "x"}, None))
    _cs = _j.members()["alice"].child_session
    record("D2 team_spawn creates an agent", bool(_cs), f"child_session={_cs!r}")
except Exception as exc:  # noqa: BLE001
    record("D2 team_spawn creates an agent", False, f"error: {exc}")

# -- D3: capabilities unreachable through the tool ---------------------------
try:
    from arm.subagent.contracts import CAPABILITY_ORDER
    from arm.subagent.tool import SubagentParams

    unreachable = sorted(set(CAPABILITY_ORDER) - {"tool_filter"})
    record("D3 capabilities reachable", not unreachable,
           f"unreachable: {unreachable} (params={sorted(SubagentParams.model_fields)})")
except Exception as exc:  # noqa: BLE001
    record("D3 capabilities reachable", False, f"error: {exc}")

# -- D4: get/cancel/shutdown reachable ---------------------------------------
refs = {s: len(outside("subagent/runtime.py", s)) for s in ("runtime.get(", "shutdown()")}
cancel = [
    ln for ln in outside("subagent/runtime.py", ".cancel(")
    if "self.cancel" not in ln
]
record("D4 get/cancel/shutdown reachable", all(refs.values()) and bool(cancel),
       f"{refs} cancel={len(cancel)}")

# -- D5: team tools not env-gated --------------------------------------------
cli_main = (SRC / "arm/cli/main.py").read_text(encoding="utf-8")
record("D5 team tools registered by default", "ARM_TEAM" not in cli_main,
       "ARM_TEAM gate " + ("present" if "ARM_TEAM" in cli_main else "absent"))

# -- D6: history() read somewhere --------------------------------------------
record("D6 history() reachable", bool(outside("schedule/runtime.py", ".history(")),
       f"{len(outside('schedule/runtime.py', '.history('))} caller(s)")

# -- D7: retry_failed reachable + a failed reminder recovers ------------------
try:
    from arm.schedule.runtime import (
        KIND_AT, STATUS_ACTIVE, ScheduleRecord, ScheduleRuntime, ScheduleStore,
    )

    store = ScheduleStore()
    store.put(ScheduleRecord(id="s1", title="t", instruction="i", session_id="A",
                             kind=KIND_AT, at_instant=NOW.isoformat()))
    ScheduleRuntime(store=store, resolve=lambda _s: None, deliver=lambda *_: True,
                    clock=lambda: NOW).drive()
    later = ScheduleRuntime(store=store, resolve=lambda _s: object(),
                            deliver=lambda *_: True,
                            clock=lambda: NOW + timedelta(hours=1))
    ok = bool(later.drive()) and store.get("s1").status == STATUS_ACTIVE
    record("D7 failed reminder retried", ok,
           f"retry_failed callers={len(outside('schedule/runtime.py', 'retry_failed'))}, "
           f"second drive delivered={ok}")
except Exception as exc:  # noqa: BLE001
    record("D7 failed reminder retried", False, f"error: {exc}")

# -- S1: nested delegation is depth-limited ----------------------------------
try:
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response, tool_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.subagent.contracts import DelegationRequest
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.subagent.runtime import SubagentRuntime
    from arm.subagent.tool import SubagentTool
    from arm.tools.registry import ToolRegistry

    reg = ToolRegistry()
    rt = SubagentRuntime(registry=reg, max_depth=1)
    depths: list[int] = []

    def build(cs, cr, req):
        tool = cr.get("subagent")
        depths.append(tool.body.__self__.depth if tool else -1)
        a = PromptAssembler()
        a.set("core", "s", priority=10)
        return Runtime(cs, MockAdapter([
            tool_response(("c", "subagent", {"prompt": "deeper"})), text_response("done")
        ]), cr, assembler=a, model="mock")

    rt.register_provider(InProcessProvider(factory=ChildFactory(build=build),
                                           parent_registry=reg))
    parent = Session.create("parent")
    reg.register(SubagentTool(rt, parent_session=parent, depth=0).register())
    asyncio.run(rt.run(DelegationRequest(prompt="go"), parent=parent))
    record("S1 nested delegation bounded", depths[:1] == [1],
           f"generations={len(depths)} depths={depths}")
except Exception as exc:  # noqa: BLE001
    record("S1 nested delegation bounded", False, f"error: {exc}")

# -- S2: result.run_id identifies the run ------------------------------------
try:
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.subagent.contracts import DelegationRequest
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.subagent.runtime import SubagentRuntime
    from arm.tools.registry import ToolRegistry

    reg = ToolRegistry()
    rt = SubagentRuntime(registry=reg)

    def build(cs, cr, req):
        a = PromptAssembler()
        a.set("core", "s", priority=10)
        return Runtime(cs, MockAdapter([text_response("ok")]), cr, assembler=a, model="mock")

    rt.register_provider(InProcessProvider(factory=ChildFactory(build=build),
                                           parent_registry=reg))

    async def one():
        running = await rt.start(DelegationRequest(prompt="x"), parent=Session.create("p"))
        res = await running.result
        await running.dispose()
        return running.run_id, res.run_id

    tracked, reported = asyncio.run(one())
    record("S2 result.run_id == run_id", tracked == reported,
           f"tracked={tracked!r} reported={reported!r}")
except Exception as exc:  # noqa: BLE001
    record("S2 result.run_id == run_id", False, f"error: {exc}")

# -- S3: a never-awaited run releases its slot -------------------------------
try:
    from arm.session.session import Session
    from arm.subagent.contracts import DelegationRequest
    from arm.subagent.runtime import RunningSubagent, SubagentRuntime
    from arm.tools.registry import ToolRegistry

    class Stuck:
        name = "stuck"
        supported = frozenset({"depth_limit"})

        async def start(self, request, *, run_id, parent):
            loop = asyncio.get_running_loop()
            fut = loop.create_future()

            async def dispose():
                pass

            return RunningSubagent(run_id=run_id, provider=self.name,
                                   child_session=Session.create("c"), result=fut,
                                   dispose=dispose)

    rt = SubagentRuntime(registry=ToolRegistry(), max_active=1)
    rt.register_provider(Stuck())  # type: ignore[arg-type]

    async def scenario():
        await rt.start(DelegationRequest(prompt="x", provider="stuck"),
                       parent=Session.create("p"))
        return rt._slots

    slots = asyncio.run(scenario())
    record("S3 unawaited run releases slot", slots == 0, f"slots held = {slots}")
except Exception as exc:  # noqa: BLE001
    record("S3 unawaited run releases slot", False, f"error: {exc}")

# -- T1: reopening a blocker re-blocks dependents ----------------------------
try:
    from arm.session.session import Session
    from arm.team.journal import MEMBER_ACTIVE, TeamJournal, TeamMember

    j = TeamJournal(Session.create("lead"))
    j.add_member(TeamMember(name="alice"))
    j.set_member_phase("alice", MEMBER_ACTIVE)
    a = j.create_task(subject="a", created_by="alice")
    b = j.create_task(subject="b", blocked_by=[a.id], created_by="alice")
    j.update_task(a.id, expected_revision=1, actor="alice", action="claim")
    j.update_task(a.id, expected_revision=2, actor="alice", action="complete")
    j.update_task(b.id, expected_revision=1, actor="alice", action="claim")
    j.update_task(a.id, expected_revision=3, actor="alice", action="reopen")
    dep = j.tasks()[b.id]
    record("T1 reopen re-blocks dependents", dep.phase != "in_progress",
           f"dependent phase={dep.phase}, task_ready={j.task_ready(dep)}")
except Exception as exc:  # noqa: BLE001
    record("T1 reopen re-blocks dependents", False, f"error: {exc}")

# -- T2: a deleted task frees a slot -----------------------------------------
try:
    from arm.session.session import Session
    from arm.team.journal import TeamError, TeamJournal

    j = TeamJournal(Session.create("lead"), max_tasks=3)
    for i in range(3):
        j.create_task(subject=f"t{i}", created_by="lead")
    j.update_task("task-1", expected_revision=1, actor="lead", is_lead=True, action="delete")
    try:
        new_id = j.create_task(subject="new", created_by="lead").id
        record("T2 delete frees an id", True, f"created {new_id}")
    except TeamError as exc:
        record("T2 delete frees an id", False, f"refused: {exc}")
except Exception as exc:  # noqa: BLE001
    record("T2 delete frees an id", False, f"error: {exc}")

# -- T3: root_id filters -----------------------------------------------------
try:
    from arm.session.session import Session
    from arm.team.journal import MEMBER_ACTIVE, TeamJournal, TeamMember

    s = Session.create("shared")
    a = TeamJournal(s, root_id="team-a")
    b = TeamJournal(s, root_id="team-b")
    for j, name in ((a, "alice"), (b, "bob")):
        j.add_member(TeamMember(name=name))
        j.set_member_phase(name, MEMBER_ACTIVE)
    record("T3 root_id filters", sorted(a.members()) == ["alice"],
           f"team-a sees {sorted(a.members())}")
except Exception as exc:  # noqa: BLE001
    record("T3 root_id filters", False, f"error: {exc}")

# -- T4: the tool can clear a description ------------------------------------
try:
    from arm.session.session import Session
    from arm.team.journal import TeamJournal
    from arm.team.tools import TeamTools

    j = TeamJournal(Session.create("lead"))
    task = j.create_task(subject="s", description="original", created_by="lead")
    tools = {t.name: t for t in TeamTools(j).tools()}
    asyncio.run(tools["team_task_update"].body(
        {"id": task.id, "revision": 1, "action": "edit", "subject": "new",
         "description": ""}, None))
    desc = j.tasks()[task.id].description
    record("T4 tool can clear description", desc == "", f"description={desc!r}")
except Exception as exc:  # noqa: BLE001
    record("T4 tool can clear description", False, f"error: {exc}")

# -- H1: weekly weekdays validated -------------------------------------------
try:
    from arm.schedule.runtime import KIND_WEEKLY, ScheduleError, ScheduleRecord

    try:
        ScheduleRecord(id="w", title="t", instruction="i", session_id="s",
                       kind=KIND_WEEKLY, at_time="09:00", weekdays=(8,))
        record("H1 weekdays validated", False, "weekdays=(8,) accepted")
    except ScheduleError as exc:
        record("H1 weekdays validated", True, f"refused: {exc}")
except Exception as exc:  # noqa: BLE001
    record("H1 weekdays validated", False, f"error: {exc}")

# -- H2: local time is the user's local time ---------------------------------
try:
    from arm.schedule.runtime import KIND_DAILY, ScheduleRecord, next_occurrence

    shanghai = timezone(timedelta(hours=8))
    task = ScheduleRecord(id="d", title="t", instruction="i", session_id="s",
                          kind=KIND_DAILY, at_time="09:00")
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)
    hour = nxt.astimezone(shanghai).hour if nxt else None
    record("H2 local time honoured", hour == 9,
           f"09:00 daily fires at {hour}:00 in the user's zone")
except Exception as exc:  # noqa: BLE001
    record("H2 local time honoured", False, f"error: {exc}")

# -- H3: cron dom/dow ORed, Sunday 7 accepted --------------------------------
try:
    from arm.schedule.runtime import KIND_CRON, ScheduleRecord, next_occurrence

    task = ScheduleRecord(id="c", title="t", instruction="i", session_id="s",
                          kind=KIND_CRON, cron="0 0 1 * 1")
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)
    dom_dow_ok = nxt is not None and nxt.date() == datetime(2026, 10, 19, tzinfo=UTC).date()
    task7 = ScheduleRecord(id="c", title="t", instruction="i", session_id="s",
                           kind=KIND_CRON, cron="0 8 * * 7")
    try:
        sunday_ok = next_occurrence(task7, now=NOW, fired_at=NOW) is not None
    except Exception:  # noqa: BLE001
        sunday_ok = False
    record("H3 cron dom/dow OR + Sunday 7", dom_dow_ok and sunday_ok,
           f"dom/dow -> {nxt}; '0 8 * * 7' accepted={sunday_ok}")
except Exception as exc:  # noqa: BLE001
    record("H3 cron dom/dow OR + Sunday 7", False, f"error: {exc}")

# -- H4: a recurring task with no scheduled_at is not due --------------------
try:
    from arm.schedule.runtime import (
        KIND_CRON, ScheduleRecord, ScheduleRuntime, ScheduleStore,
    )

    store = ScheduleStore()
    store.put(ScheduleRecord(id="c", title="t", instruction="i", session_id="s",
                             kind=KIND_CRON, cron="0 9 * * *"))
    receipts = ScheduleRuntime(store=store, resolve=lambda _s: object(),
                               deliver=lambda *_: True, clock=lambda: NOW).drive()
    record("H4 no premature recurring fire", not receipts,
           f"{len(receipts)} receipt(s) for a 09:00 cron driven at 12:00")
except Exception as exc:  # noqa: BLE001
    record("H4 no premature recurring fire", False, f"error: {exc}")

# -- H5: the timer covers the idle chat loop ---------------------------------
chat = cli_main[cli_main.index("def _cmd_chat"):cli_main.index("def _cmd_replay")]
record("H5 timer covers idle chat", "ScheduleTimer" in chat,
       "ScheduleTimer " + ("in" if "ScheduleTimer" in chat else "not in") + " _cmd_chat")

# -- H6: history honours a zero limit ----------------------------------------
try:
    from arm.schedule.runtime import DeliveryReceipt, ScheduleStore

    store = ScheduleStore()
    for i in range(3):
        store.record_delivery(DeliveryReceipt(
            schedule_id=f"s{i}", session_id="x",
            delivered_at=NOW.isoformat(), occurrence_at=NOW.isoformat()))
    zero = len(store.history(limit=0))
    record("H6 history honours limit=0", zero == 0, f"limit=0 returned {zero}")
except Exception as exc:  # noqa: BLE001
    record("H6 history honours limit=0", False, f"error: {exc}")

# -- H7: an invalid weekday is refused by the tool ---------------------------
try:
    from arm.schedule.runtime import ScheduleStore
    from arm.schedule.tools import ScheduleTools

    store = ScheduleStore()
    tools = {t.name: t for t in ScheduleTools(store, session_id="s").tools()}
    result = asyncio.run(tools["schedule_create"].body(
        {"title": "t", "instruction": "i", "kind": "weekly",
         "at_time": "09:00", "weekdays": [8]}, None))
    if result.is_error:
        record("H7 invalid weekday refused", True, "refused at creation")
    else:
        scheduled = store.all()[0].scheduled_at
        record("H7 invalid weekday refused", bool(scheduled),
               f"accepted; scheduled_at={scheduled!r} (fires immediately)")
except Exception as exc:  # noqa: BLE001
    record("H7 invalid weekday refused", False, f"error: {exc}")

# -- H8: no duplicate delivery on re-entrant drive ---------------------------
try:
    from arm.schedule.runtime import (
        KIND_AT, ScheduleRecord, ScheduleRuntime, ScheduleStore,
    )

    store = ScheduleStore()
    store.put(ScheduleRecord(id="s1", title="t", instruction="i", session_id="sess",
                             kind=KIND_AT, at_instant=NOW.isoformat()))
    deliveries: list[str] = []
    holder: dict = {}

    def deliver(session_id, body, message_id):
        deliveries.append(message_id)
        if holder.get("rt") is not None and len(deliveries) == 1:
            holder["rt"].drive()
        return True

    rt = ScheduleRuntime(store=store, resolve=lambda _s: object(), deliver=deliver,
                         clock=lambda: NOW)
    holder["rt"] = rt
    rt.drive()
    record("H8 no duplicate delivery", len(deliveries) == 1,
           f"{len(deliveries)} delivery/deliveries: {deliveries}")
except Exception as exc:  # noqa: BLE001
    record("H8 no duplicate delivery", False, f"error: {exc}")

# -- E1: a build failure is a SubagentError ----------------------------------
try:
    from arm.session.session import Session
    from arm.subagent.contracts import SubagentError
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.subagent.runtime import SubagentRuntime
    from arm.subagent.tool import SubagentTool
    from arm.tools.registry import ToolRegistry

    reg = ToolRegistry()
    rt = SubagentRuntime(registry=reg)

    def broken(*_a):
        raise ValueError("nope")

    rt.register_provider(InProcessProvider(factory=ChildFactory(build=broken),
                                           parent_registry=reg))
    tool = SubagentTool(rt, parent_session=Session.create("p"))
    try:
        asyncio.run(tool.run({"prompt": "x"}, None))
        record("E1 build failure is SubagentError", False, "no exception at all")
    except SubagentError:
        record("E1 build failure is SubagentError", True, "raised SubagentError")
    except Exception as exc:  # noqa: BLE001
        record("E1 build failure is SubagentError", False,
               f"raised {type(exc).__name__}: {exc}")
except Exception as exc:  # noqa: BLE001
    record("E1 build failure is SubagentError", False, f"error: {exc}")

# -- E2: a huge interval is a result, not an OverflowError --------------------
try:
    from arm.schedule.runtime import ScheduleStore
    from arm.schedule.tools import ScheduleTools

    store = ScheduleStore()
    tools = {t.name: t for t in ScheduleTools(store, session_id="s").tools()}
    try:
        result = asyncio.run(tools["schedule_create"].body(
            {"title": "t", "instruction": "i", "kind": "after",
             "after_seconds": 10**18}, None))
        record("E2 huge interval is a result", result.is_error,
               f"returned a result (is_error={result.is_error})")
    except Exception as exc:  # noqa: BLE001
        record("E2 huge interval is a result", False,
               f"raised {type(exc).__name__}: {exc}")
except Exception as exc:  # noqa: BLE001
    record("E2 huge interval is a result", False, f"error: {exc}")

# -- E3: max-tokens is a producible turn reason ------------------------------
try:
    from arm.session.events import TURN_END_REASONS

    record("E3 max-tokens producible", "max-tokens" in TURN_END_REASONS,
           f"TURN_END_REASONS={sorted(TURN_END_REASONS)}")
except Exception as exc:  # noqa: BLE001
    record("E3 max-tokens producible", False, f"error: {exc}")

# -- E4: a mailbox error carries a mailbox code ------------------------------
try:
    from arm.session.session import Session
    from arm.team.journal import TeamError, TeamJournal

    try:
        TeamJournal(Session.create("lead")).confirm_delivery("nope", "alice")
        record("E4 mailbox error code", False, "no exception")
    except TeamError as exc:
        record("E4 mailbox error code", "TASK" not in exc.code, f"code={exc.code!r}")
except Exception as exc:  # noqa: BLE001
    record("E4 mailbox error code", False, f"error: {exc}")


def main() -> int:
    width = max(len(tag) for tag, _, _ in RESULTS)
    bugs = 0
    for tag, state, detail in RESULTS:
        if state.strip() == "BUG":
            bugs += 1
        print(f"{state}  {tag:<{width}}  {detail}")
    print()
    print(f"{len(RESULTS) - bugs} of {len(RESULTS)} fixed; {bugs} still broken")
    return 0


if __name__ == "__main__":
    sys.exit(main())

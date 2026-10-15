"""FAILING tests: dead code, unwired features, and integration defects.

Targets: ``arm.subagent.*``, ``arm.team.*``, ``arm.schedule.*``.

Every test here is ``xfail(strict=True)``: it fails while the defect is present
and turns into ``XPASS(strict)`` — a hard failure — the moment the fix lands, so
a bug cannot be silently re-introduced and a stale marker cannot be left behind.

Nothing here modifies ``src/``, ``tests/``, ``docs/``, ``README.md`` or
``REQUIREMENTS.md``.

Run:  cd spikes && ../.venv/bin/python -m pytest q5_subagent -q
"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import _bootstrap  # noqa: F401

from arm.session.session import Session
from arm.subagent.contracts import (
    CAPABILITY_ORDER,
    STOP_COMPLETED,
    DelegationRequest,
)
from arm.subagent.runtime import RunningSubagent, SubagentRuntime
from arm.subagent.tool import SubagentParams, SubagentTool
from arm.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
UTC = timezone.utc
NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


def src_calls(symbol: str) -> list[str]:
    """Every ``src/`` line mentioning ``symbol``, as ``path:line: text``.

    ``-F`` is load-bearing: the symbols here contain ``.`` and ``(``, and a
    regex search for ``.history(`` matches ``def history(`` (the ``.`` matches
    the space), which would report a dead symbol as live.
    """
    out = subprocess.run(
        ["grep", "-rnF", "--include=*.py", symbol, str(SRC)],
        capture_output=True,
        text=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def callers_outside(module: str, symbol: str) -> list[str]:
    """References to ``symbol`` from any ``src/`` file other than ``module``."""
    marker = f"{SRC}/arm/{module}"
    return [ln for ln in src_calls(symbol) if not ln.startswith(marker)]


# ===========================================================================
# D1 — the mailbox never delivers: confirm_delivery has no caller
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="D1: TeamJournal.confirm_delivery has no caller in src/, so no message "
    "is ever marked delivered",
)
def test_d1_the_mailbox_can_actually_deliver_a_message():
    """A queued message must become delivered through a real code path.

    ``TeamJournal.confirm_delivery`` exists and is tested, but nothing in ``src/``
    calls it: not ``TeamTools``, not the CLI, not any agent. The mailbox is
    therefore a write-only queue in the running system — ``team_send`` says "it
    will be delivered when they next run", and no run ever delivers it.
    """
    from arm.team.journal import MEMBER_ACTIVE, TeamJournal, TeamMember, TeamMessage
    from arm.team.tools import TeamTools

    session = Session.create("lead")
    journal = TeamJournal(session)
    journal.add_member(TeamMember(name="alice"))
    journal.set_member_phase("alice", MEMBER_ACTIVE)

    tools = {tool.name: tool for tool in TeamTools(journal).tools()}
    asyncio.run(tools["team_send"].body({"name": "alice", "message": "review this"}, None))

    pending = journal.pending_for("alice")
    assert len(pending) == 1, "the message was not queued"

    # The only way a message becomes delivered is confirm_delivery, which nothing
    # in src/ calls. Drive every registered tool and check whether any of them
    # confirms the delivery.
    for name, tool in tools.items():
        asyncio.run(tool.body({"name": "alice", "id": pending[0].message_id}, None))

    assert journal.messages()[0].delivered, (
        "no code path marks a queued message delivered; confirm_delivery is dead code"
    )


@pytest.mark.xfail(
    strict=True,
    reason="D1: confirm_delivery is unreachable — dead code",
)
def test_d1_confirm_delivery_is_reachable_from_src():
    callers = callers_outside("team/journal.py", "confirm_delivery")
    assert callers, (
        "confirm_delivery is defined and tested but never called from src/:\n"
        + "\n".join(src_calls("confirm_delivery"))
    )


# ===========================================================================
# D2 — team_spawn records a member but never creates an agent
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="D2: team_spawn records a roster row and returns 'now active' without "
    "creating any agent or child session",
)
def test_d2_spawning_a_teammate_creates_something_that_can_run():
    """A teammate the lead cannot reach is a roster row, not a teammate.

    ``TeamMember.child_session`` exists and is persisted, and the subagent layer
    is the machinery that would fill it. ``TeamTools`` never imports or calls
    ``SubagentRuntime``: ``team_spawn`` writes a name and reports "now active",
    and the ``task`` it was given is echoed back into the message text and
    dropped.
    """
    from arm.team.journal import MEMBER_ACTIVE, TeamJournal, TeamMember
    from arm.team.tools import TeamTools

    session = Session.create("lead")
    journal = TeamJournal(session)
    tools = {tool.name: tool for tool in TeamTools(journal).tools()}

    result = asyncio.run(
        tools["team_spawn"].body({"name": "alice", "task": "review the parser"}, None)
    )
    assert not result.is_error

    member = journal.members()["alice"]
    assert member.child_session, (
        "team_spawn reported the teammate active but no child session exists: "
        "the teammate has no agent"
    )


# ===========================================================================
# D3 — the subagent tool cannot request output_schema / persona / model
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="D3: four of the five capabilities (agent_options, output_schema, "
    "depth_limit, persona) cannot be requested through any model-facing surface",
)
def test_d3_the_tool_can_request_every_capability_the_runtime_asserts():
    """Four of the five capabilities cannot be requested through any surface.

    ``CAPABILITY_ORDER`` names five capabilities and ``InProcessProvider``
    advertises all five. ``SubagentTool.run`` builds a ``DelegationRequest`` with
    ``prompt``, ``provider``, ``label``, ``depth``, ``max_depth`` and ``tools`` —
    never ``output_schema``, ``persona`` or ``model`` — and always passes
    ``max_depth=DEFAULT_MAX_DEPTH``. ``DelegationRequest.requires`` then reports
    ``False`` for four of the five, so ``assert_capabilities`` can never raise
    through the tool and the capabilities are dead configuration.

    Only ``tool_filter`` is reachable, because ``tools`` is the one field the tool
    exposes.
    """
    reachable = {
        "tool_filter",  # SubagentParams.tools
    }
    unreachable = sorted(set(CAPABILITY_ORDER) - reachable)
    assert not unreachable, (
        f"capabilities {unreachable} cannot be requested through SubagentParams "
        f"({sorted(SubagentParams.model_fields)}) or through the fixed arguments "
        "SubagentTool.run supplies"
    )


# ===========================================================================
# D4 — SubagentRuntime.get / cancel / shutdown are unreachable
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="D4: SubagentRuntime.get/cancel/shutdown have no caller in src/ — "
    "a run cannot be inspected or stopped",
)
def test_d4_a_running_subagent_can_be_cancelled_from_a_real_path():
    """The cancel surface exists but nothing can reach it.

    ``get``, ``cancel`` and ``shutdown`` are public API with tests, and no code
    outside ``SubagentRuntime`` calls any of them. ``SubagentTool`` has no cancel
    argument and no run id to cancel with, so a delegation that hangs can only be
    waited out (or timed out); ``shutdown`` is never called at process exit.
    """
    for symbol in ("runtime.get(", ".cancel(", "shutdown()"):
        assert callers_outside("subagent/runtime.py", symbol), (
            f"{symbol} has no caller outside subagent/runtime.py"
        )


# ===========================================================================
# D5 — the team journal is unreachable unless ARM_TEAM is set
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="D5: TeamJournal/TeamTools are gated behind the ARM_TEAM env var, so "
    "the feature is off in every default run",
)
def test_d5_the_team_tools_are_registered_without_an_env_var():
    """A feature behind an undocumented env var is a feature nobody has.

    ``build_runtime`` registers the five team tools only when ``ARM_TEAM`` is
    set. Nothing documents the variable, no test sets it, and the CLI never
    prints it — so in the shipped path the team module is unreachable.
    """
    text = (SRC / "arm/cli/main.py").read_text(encoding="utf-8")
    assert "ARM_TEAM" not in text, (
        "the team tools are registered only when ARM_TEAM is set, which no "
        "documented path sets"
    )


# ===========================================================================
# D6 — ScheduleStore.history is write-only; receipts are never read
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="D6: ScheduleStore.history has no caller in src/ — delivery receipts "
    "are persisted and never read",
)
def test_d6_delivery_history_is_readable_from_a_real_path():
    """The store keeps receipts and nothing ever asks for them.

    ``record_delivery`` appends and ``_prune_history`` bounds, both of which are
    tested. ``history()`` is the only reader and no ``src/`` file calls it, so
    the entire retention policy protects a list nobody reads — and no tool or CLI
    command shows a user what fired.
    """
    assert callers_outside("schedule/runtime.py", ".history("), (
        "ScheduleStore.history has no caller in src/:\n" + "\n".join(src_calls(".history("))
    )


@pytest.mark.xfail(
    strict=True,
    reason="D7: ScheduleRuntime.retry_failed has no caller in src/ — a reminder "
    "whose session was not loaded is failed forever",
)
def test_d7_a_failed_reminder_is_retried_when_its_session_loads():
    """``failed`` is a terminal state in practice.

    ``drive()`` marks a task ``failed`` when its session cannot be resolved, and
    ``retry_failed()`` exists to return it to the active set. Nothing calls
    ``retry_failed``: not the loop, not the CLI, not a tool. A reminder whose
    session was not open at the moment it came due therefore never fires again —
    the exact outcome the module docstring says must never happen ("losing a
    reminder silently would be the one outcome a reminder must never produce").
    """
    from arm.schedule.runtime import (
        KIND_AT,
        STATUS_ACTIVE,
        ScheduleRecord,
        ScheduleRuntime,
        ScheduleStore,
    )

    store = ScheduleStore()
    store.put(
        ScheduleRecord(
            id="s1", title="t", instruction="i", session_id="sess-A",
            kind=KIND_AT, at_instant=NOW.isoformat(),
        )
    )

    # The reminder comes due while its session is not loaded.
    first = ScheduleRuntime(
        store=store, resolve=lambda _sid: None, deliver=lambda *_: True, clock=lambda: NOW
    )
    first.drive()

    # Later, the session *is* loaded. The reminder is still overdue.
    later = ScheduleRuntime(
        store=store,
        resolve=lambda _sid: object(),
        deliver=lambda *_: True,
        clock=lambda: NOW + timedelta(hours=1),
    )
    receipts = later.drive()
    assert receipts, (
        "a reminder that failed while its session was unloaded never fires again; "
        "retry_failed is dead code"
    )
    assert store.get("s1").status == STATUS_ACTIVE


# ===========================================================================
# S1 — a nested delegation is never depth-limited (fork bomb)
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="S1: the child inherits the parent's subagent tool with depth=0 and "
    "parent_session=parent, so max_depth never binds — the chain runs to "
    "max_active generations",
)
def test_s1_a_child_cannot_spawn_past_the_depth_limit():
    """``max_depth`` is enforced against a depth that never increases.

    ``SubagentTool`` is registered once, on the parent's registry, with
    ``depth=0`` and ``parent_session=<parent>``. ``InProcessProvider`` copies that
    registry into the child, so the child's ``subagent`` tool still reports
    ``depth=0`` and still delegates *from the parent's session*. Every generation
    therefore passes ``assert_max_depth(0)``.

    The consequence is that ``max_depth`` is inert: with ``max_depth=1`` (the
    parent may delegate, the child may not) the chain still runs to
    ``max_active`` generations. The only thing stopping it is the concurrency
    limit — which is not a depth limit, and which each generation consumes, so a
    deep chain also starves every sibling delegation.

    ``test_a_depth_violation_is_refused_...`` in ``tests/test_subagent.py``
    calls ``runtime.start`` directly with ``depth=2``; no test drives a real
    nested delegation, which is why this is not caught.
    """
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response, tool_response
    from arm.prompt.assembler import PromptAssembler
    from arm.subagent.inprocess import ChildFactory, InProcessProvider

    registry = ToolRegistry()
    # max_depth=1: the parent (depth 0) may delegate, the child may not.
    # max_active=8 is the runtime default, so it is not what should bind.
    runtime = SubagentRuntime(registry=registry, max_depth=1, max_active=8)

    depths: list[int] = []

    def build(child_session, child_registry, request):
        tool = child_registry.get("subagent")
        depths.append(tool.body.__self__.depth if tool is not None else -1)
        # Every child immediately delegates again, forever.
        script = [
            tool_response(("c1", "subagent", {"prompt": "go deeper"})),
            text_response("done"),
        ]
        assembler = PromptAssembler()
        assembler.set("core", "sub", priority=10)
        return Runtime(
            child_session, MockAdapter(script), child_registry,
            assembler=assembler, model="mock",
        )

    runtime.register_provider(
        InProcessProvider(factory=ChildFactory(build=build), parent_registry=registry)
    )
    parent = Session.create("parent")
    registry.register(SubagentTool(runtime, parent_session=parent, depth=0).register())

    asyncio.run(runtime.run(DelegationRequest(prompt="go"), parent=parent))

    assert depths, "no child was built at all"
    assert depths[0] == 1, (
        f"the first child's subagent tool reports depth {depths[0]}, not 1: the "
        "child inherits the parent's depth-0 tool, so assert_max_depth(0) passes "
        "at every generation"
    )
    assert len(depths) == 1, (
        f"max_depth=1 allowed {len(depths)} generations (depths {depths}): the "
        "chain is bounded by max_active, not by max_depth"
    )


# ===========================================================================
# S2 — the result's run_id is not the runtime's run_id
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="S2: DelegationResult.run_id is derived from the child session, not the "
    "run_id the runtime tracks, so get()/cancel() cannot be used with it",
)
def test_s2_the_run_id_in_the_result_identifies_the_run():
    """The id reported to the model is not the id the runtime knows.

    ``InProcessProvider`` sets ``result.run_id = result_key(child_session)`` —
    ``"subagent-<8 hex>"`` — while ``SubagentRuntime`` keys ``_running`` by the
    ``uuid4`` it generated. ``SubagentTool`` returns ``result.run_id`` in its
    ``value``, so the model is handed an identifier that ``runtime.get`` and
    ``runtime.cancel`` cannot resolve.
    """
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.subagent.inprocess import ChildFactory, InProcessProvider

    registry = ToolRegistry()
    runtime = SubagentRuntime(registry=registry)

    def build(child_session, child_registry, request):
        assembler = PromptAssembler()
        assembler.set("core", "sub", priority=10)
        return Runtime(
            child_session, MockAdapter([text_response("done")]), child_registry,
            assembler=assembler, model="mock",
        )

    runtime.register_provider(
        InProcessProvider(factory=ChildFactory(build=build), parent_registry=registry)
    )
    parent = Session.create("parent")

    async def scenario():
        running = await runtime.start(DelegationRequest(prompt="x"), parent=parent)
        result = await running.result
        await running.dispose()
        return running.run_id, result.run_id

    tracked, reported = asyncio.run(scenario())
    assert reported == tracked, (
        f"the runtime tracks {tracked!r} but the result (and the tool's value) "
        f"reports {reported!r}; runtime.get/cancel cannot resolve the reported id"
    )


# ===========================================================================
# S3 — a start() that is cancelled leaks its slot
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="S3: SubagentRuntime.start increments _slots before awaiting the "
    "provider, and only releases it on the next drive() — a caller that never "
    "awaits result keeps the slot forever",
)
def test_s3_a_started_run_that_is_never_awaited_releases_its_slot():
    """The slot is released by a done-callback on a future nobody awaits.

    ``start`` increments ``_slots`` and attaches
    ``running.result.add_done_callback(...)``. That is correct only if the future
    eventually resolves. For the real ``InProcessProvider`` the drive task always
    resolves it, so this is a design smell rather than a live leak — but the
    *documented* contract ("a caller that never awaits it still releases the
    slot") is not what the code does: a provider whose future is never resolved
    leaks the slot permanently, and ``shutdown`` cannot help because it is not
    reachable either.
    """
    provider_future = {}

    class Stuck:
        name = "stuck"
        supported = frozenset({"depth_limit"})

        async def start(self, request, *, run_id, parent):
            loop = asyncio.get_running_loop()
            provider_future["f"] = loop.create_future()

            async def dispose():
                pass

            return RunningSubagent(
                run_id=run_id, provider=self.name,
                child_session=Session.create("c"), result=provider_future["f"],
                dispose=dispose,
            )

    runtime = SubagentRuntime(registry=ToolRegistry(), max_active=1)
    runtime.register_provider(Stuck())  # type: ignore[arg-type]

    async def scenario():
        await runtime.start(
            DelegationRequest(prompt="x", provider="stuck"), parent=Session.create("p")
        )
        return runtime._slots

    slots = asyncio.run(scenario())
    assert slots == 0, (
        f"{slots} slot(s) are held by a run nobody is waiting for and nothing "
        "will ever resolve"
    )


# ===========================================================================
# T1 — a reopened blocker does not re-block an in-progress dependent
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="T1: task_ready is only consulted on claim/reassign, so reopening a "
    "completed blocker leaves its dependent in progress",
)
def test_t1_reopening_a_blocker_re_blocks_its_dependents():
    """Readiness is checked when work starts and never again.

    ``claim`` and ``reassign`` both call ``task_ready``, and ``reopen`` calls
    nothing. A dependent that was claimed while its blocker was complete stays
    ``in_progress`` when that blocker is reopened, so two members work the same
    dependency chain believing it is satisfied. ``task_ready`` reports ``False``
    for the dependent, so the board's own view disagrees with the task's phase.
    """
    from arm.team.journal import MEMBER_ACTIVE, TeamJournal, TeamMember

    session = Session.create("lead")
    journal = TeamJournal(session)
    journal.add_member(TeamMember(name="alice"))
    journal.set_member_phase("alice", MEMBER_ACTIVE)

    first = journal.create_task(subject="first", created_by="alice")
    second = journal.create_task(subject="second", blocked_by=[first.id], created_by="alice")

    journal.update_task(first.id, expected_revision=1, actor="alice", action="claim")
    journal.update_task(first.id, expected_revision=2, actor="alice", action="complete")
    journal.update_task(second.id, expected_revision=1, actor="alice", action="claim")

    # The blocker turns out not to be done.
    journal.update_task(first.id, expected_revision=3, actor="alice", action="reopen")

    dependent = journal.tasks()[second.id]
    assert dependent.phase != "in_progress", (
        "the dependent is still in progress while its blocker is reopened; "
        "task_ready() says "
        f"{journal.task_ready(dependent)}"
    )


# ===========================================================================
# T2 — task ids are exhausted permanently by a deletion
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="T2: create_task scans 1..max_tasks over all ids ever used, so "
    "deleting a task does not free a slot — the board refuses new tasks while "
    "under its limit",
)
def test_t2_a_deleted_task_does_not_exhaust_the_id_space():
    """``max_tasks`` is a limit on *live* tasks; the id scan is over all ids.

    ``create_task`` checks ``len(live) >= max_tasks`` (live tasks only) and then
    scans ``task-1..task-N`` against ``board`` (every id ever used, deleted ones
    included). With ``max_tasks`` tasks ever created and one deleted, the board
    has ``max_tasks - 1`` live tasks and still refuses — a permanent, silent
    reduction of capacity that no message explains.
    """
    from arm.team.journal import TeamJournal

    session = Session.create("lead")
    journal = TeamJournal(session, max_tasks=3)
    for index in range(3):
        journal.create_task(subject=f"t{index}", created_by="lead")

    journal.update_task("task-1", expected_revision=1, actor="lead", is_lead=True, action="delete")
    assert len(journal.live_tasks()) == 2

    created = journal.create_task(subject="new work", created_by="lead")
    assert created.id, "a new task should be creatable while under the live limit"


# ===========================================================================
# T3 — root_id is written and never used to filter
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="T3: every event carries 'root': root_id, and no projection filters on "
    "it — two journals over one log see each other's members and tasks",
)
def test_t3_a_journal_only_sees_its_own_root():
    """The team's events are namespaced and the namespace is never applied.

    ``TeamJournal._append`` writes ``{"root": self.root_id, ...}`` on every
    event, which is what makes two teams in one log distinguishable. ``members``,
    ``messages``, ``tasks`` and ``summary`` all iterate ``self.session`` and check
    only the event type, so a second journal sharing the log sees the first
    journal's members as its own.
    """
    from arm.team.journal import MEMBER_ACTIVE, TeamJournal, TeamMember

    session = Session.create("shared")
    team_a = TeamJournal(session, root_id="team-a")
    team_b = TeamJournal(session, root_id="team-b")

    for journal, name in ((team_a, "alice"), (team_b, "bob")):
        journal.add_member(TeamMember(name=name))
        journal.set_member_phase(name, MEMBER_ACTIVE)

    assert sorted(team_a.members()) == ["alice"], (
        f"team-a sees {sorted(team_a.members())}: root_id is not a filter"
    )


# ===========================================================================
# T4 — the edit action cannot clear a description through the tool
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="T4: TeamTools._task_update only forwards a non-empty description, so "
    "an empty string cannot clear a task's description",
)
def test_t4_a_task_description_can_be_cleared_through_the_tool():
    """The tool's own schema says "New description, for 'edit'" and cannot do it.

    ``_task_update`` guards with ``if arguments.get("description")``, so ``""`` is
    dropped before ``journal.update_task`` sees it. ``update_task`` *can* set an
    empty description (it distinguishes ``None`` from ``""``); only the tool
    cannot express it. The model is told it may edit the description and has no
    way to remove one.
    """
    from arm.team.journal import TeamJournal
    from arm.team.tools import TeamTools

    session = Session.create("lead")
    journal = TeamJournal(session)
    task = journal.create_task(subject="s", description="original", created_by="lead")
    tools = {tool.name: tool for tool in TeamTools(journal).tools()}

    asyncio.run(
        tools["team_task_update"].body(
            {"id": task.id, "revision": 1, "action": "edit", "subject": "new",
             "description": ""},
            None,
        )
    )
    assert journal.tasks()[task.id].description == "", (
        "the description is still "
        f"{journal.tasks()[task.id].description!r}; an empty description is ignored"
    )


# ===========================================================================
# H1 — a weekly task with an out-of-range weekday is accepted and fires
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="H1: ScheduleRecord.__post_init__ checks that weekdays is non-empty but "
    "not that each is 1..7; weekdays=(8,) is accepted and fires immediately",
)
def test_h1_a_weekly_task_validates_its_weekdays():
    """``weekdays=(8,)`` is accepted, never matches, and fires at once.

    ``next_occurrence`` searches ``offset in range(0, 8)`` and compares
    ``isoweekday() in weekdays``, so an out-of-range weekday never matches and it
    returns ``None``. ``drive()`` then treats the task as finished: the reminder
    fires immediately, once, at the wrong time, and is marked inactive.
    """
    from arm.schedule.runtime import KIND_WEEKLY, ScheduleError, ScheduleRecord

    with pytest.raises(ScheduleError):
        ScheduleRecord(
            id="w", title="t", instruction="i", session_id="s",
            kind=KIND_WEEKLY, at_time="09:00", weekdays=(8,),
        )


@pytest.mark.xfail(
    strict=True,
    reason="H1b: a weekly task with no valid weekday is delivered immediately and "
    "marked inactive instead of being refused",
)
def test_h1b_an_unmatchable_weekly_task_does_not_fire_immediately():
    from arm.schedule.runtime import (
        KIND_WEEKLY,
        ScheduleRecord,
        ScheduleRuntime,
        ScheduleStore,
    )

    store = ScheduleStore()
    store.put(
        ScheduleRecord(
            id="w", title="t", instruction="i", session_id="s",
            kind=KIND_WEEKLY, at_time="09:00", weekdays=(8,),
        )
    )
    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=lambda *_: True, clock=lambda: NOW
    )
    receipts = runtime.drive()
    assert not receipts, (
        "a weekly task with no matching weekday was delivered immediately: "
        f"{[r.schedule_id for r in receipts]}"
    )


# ===========================================================================
# H2 — 'daily'/'weekly' local time is interpreted in the clock's zone (UTC)
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="H2: at_time is documented as a *local* time and no zone is stored, so "
    "with the CLI's UTC clock 09:00 fires at 17:00 in Asia/Shanghai",
)
def test_h2_a_daily_reminder_fires_at_the_users_local_time():
    """The docs say local; the arithmetic says "wherever the clock is".

    ``ScheduleRecord.at_time`` is documented as "the local time of day", the tool
    description says "the local time, HH:MM", and ``_parse_local_time`` validates
    only the shape. ``next_occurrence`` does ``now.replace(hour=..., ...)``, which
    keeps ``now``'s tzinfo — and the CLI's clock is
    ``lambda: datetime.now(timezone.utc)``. No zone is stored on the record, so a
    user in Asia/Shanghai who asks for 09:00 is woken at 17:00 their time, and
    there is no argument that could fix it.
    """
    from arm.schedule.runtime import KIND_DAILY, ScheduleRecord, next_occurrence

    shanghai = timezone(timedelta(hours=8))
    # The instant is 2026-10-13 20:00 in Shanghai = 12:00 UTC. The CLI's clock
    # returns the UTC value, which is what the runtime actually sees.
    cli_clock_now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
    task = ScheduleRecord(
        id="d", title="t", instruction="i", session_id="s",
        kind=KIND_DAILY, at_time="09:00",
    )
    nxt = next_occurrence(task, now=cli_clock_now, fired_at=cli_clock_now)
    assert nxt is not None
    assert nxt.astimezone(shanghai).hour == 9, (
        "a 09:00 daily reminder lands at "
        f"{nxt.astimezone(shanghai).strftime('%H:%M')} in the user's zone "
        "(the CLI clock is UTC and no zone is stored on the record)"
    )


# ===========================================================================
# H3 — a cron day-of-month/day-of-week pair is ANDed, not ORed
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="H3: _next_cron ANDs dom and dow; Vixie cron ORs them when both are "
    "restricted",
)
def test_h3_cron_dom_and_dow_are_ored_like_cron():
    """``0 0 1 * 1`` means "the 1st, or any Monday" in every cron implementation.

    ``_next_cron`` requires every field to match, so it means "the 1st *if* it is
    a Monday" — from 2026-10-13 it returns 2027-02-01 instead of 2026-10-19. A
    reminder written with the standard idiom fires months late.
    """
    from arm.schedule.runtime import KIND_CRON, ScheduleRecord, next_occurrence

    task = ScheduleRecord(
        id="c", title="t", instruction="i", session_id="s",
        kind=KIND_CRON, cron="0 0 1 * 1",
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)
    assert nxt is not None
    assert nxt.date() == datetime(2026, 10, 19, tzinfo=UTC).date(), (
        f"cron '0 0 1 * 1' gave {nxt}; Vixie cron gives the next Monday"
    )


@pytest.mark.xfail(
    strict=True,
    reason="H3b: Sunday as 7 is standard in cron and refused here",
)
def test_h3b_cron_accepts_sunday_as_seven():
    from arm.schedule.runtime import KIND_CRON, ScheduleRecord, next_occurrence

    task = ScheduleRecord(
        id="c", title="t", instruction="i", session_id="s",
        kind=KIND_CRON, cron="0 8 * * 7",
    )
    nxt = next_occurrence(task, now=NOW, fired_at=NOW)
    assert nxt is not None and nxt.isoweekday() == 7


# ===========================================================================
# H4 — every recurring task with no scheduled_at is due immediately
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="H4: ScheduleStore.due() treats a recurring task with no scheduled_at "
    "as due now, so a cron/daily/weekly task created without a computed first "
    "time fires immediately",
)
def test_h4_a_recurring_task_is_not_due_before_its_first_occurrence():
    """``scheduled_at`` is the only thing standing between "created" and "fired".

    ``due()`` returns a recurring task with no ``scheduled_at`` unconditionally.
    ``ScheduleTools`` computes the first occurrence and stores it, so the tool
    path is safe — but ``ScheduleStore.put`` is the public write API and a
    record built directly (as a restored or hand-written schedule file can be)
    is delivered on the next drive regardless of its ``at_time`` or ``cron``.
    """
    from arm.schedule.runtime import (
        KIND_CRON,
        ScheduleRecord,
        ScheduleRuntime,
        ScheduleStore,
    )

    store = ScheduleStore()
    store.put(
        ScheduleRecord(
            id="c", title="t", instruction="i", session_id="s",
            kind=KIND_CRON, cron="0 9 * * *",  # 09:00, nine hours away
        )
    )
    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=lambda *_: True, clock=lambda: NOW
    )
    assert runtime.drive() == [], "a 09:00 cron reminder fired at 12:00 the day before"


# ===========================================================================
# H5 — the ScheduleTimer only runs during a task, never while chat is idle
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="H5: the CLI starts ScheduleTimer inside _run_task and stops it in the "
    "finally, so a reminder that comes due between chat lines never fires",
)
def test_h5_the_timer_covers_the_idle_chat_loop():
    """The timer exists to fire reminders while nothing else is happening.

    ``_cmd_chat`` calls ``asyncio.run(_run_task(...))`` once per line, and
    ``_run_task`` starts the timer and stops it in a ``finally``. Between lines
    there is no event loop and no timer, so the gap the timer was written for —
    a user sitting at an idle prompt — is exactly the gap it does not cover.
    """
    text = (SRC / "arm/cli/main.py").read_text(encoding="utf-8")
    start = text.index("def _cmd_chat")
    end = text.index("def _cmd_replay")
    chat_body = text[start:end]
    assert "ScheduleTimer" in chat_body, (
        "the chat loop never constructs a timer; it only runs for the duration of "
        "each task"
    )


# ===========================================================================
# H6 — history(limit=0) returns everything
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="H6: history(limit=0) returns the whole list because [-0:] is [0:]",
)
def test_h6_history_respects_a_zero_limit():
    from arm.schedule.runtime import DeliveryReceipt, ScheduleStore

    store = ScheduleStore()
    for index in range(3):
        store.record_delivery(
            DeliveryReceipt(
                schedule_id=f"s{index}", session_id="x",
                delivered_at=NOW.isoformat(), occurrence_at=NOW.isoformat(),
            )
        )
    assert len(store.history(limit=0)) == 0, "limit=0 returned records"
    assert len(store.history(limit=-1)) == 0, "a negative limit returned records"


# ===========================================================================
# E1 — error-type consistency
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="E1: InProcessProvider.start wraps a build failure in RuntimeError, not "
    "the documented SubagentError, so it escapes the tool's except clause",
)
def test_e1_a_build_failure_is_a_subagent_error():
    """Every documented refusal is a ``SubagentError``; a build failure is not.

    ``SubagentTool.run`` catches ``SubagentError`` and turns it into a result. A
    provider whose factory raises produces a ``RuntimeError`` that escapes the
    tool, through the pipeline's body-error path, and reaches the model as
    ``BODY_ERROR`` with the message mangled — rather than the refusal the caller
    was promised.
    """
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.subagent.contracts import SubagentError

    registry = ToolRegistry()

    def broken_build(child_session, child_registry, request):
        raise ValueError("the child could not be constructed")

    runtime = SubagentRuntime(registry=registry)
    runtime.register_provider(
        InProcessProvider(
            factory=ChildFactory(build=broken_build), parent_registry=registry
        )
    )
    tool = SubagentTool(runtime, parent_session=Session.create("p"))

    with pytest.raises(SubagentError):
        asyncio.run(tool.run({"prompt": "do it"}, None))


@pytest.mark.xfail(
    strict=True,
    reason="E1b: a bad tools filter is wrapped in RuntimeError, not SubagentError",
)
def test_e1b_a_bad_tool_filter_is_a_subagent_error():
    from arm.subagent.contracts import SubagentError
    from arm.subagent.inprocess import ChildFactory, InProcessProvider

    registry = ToolRegistry()
    runtime = SubagentRuntime(registry=registry)
    runtime.register_provider(
        InProcessProvider(
            factory=ChildFactory(build=lambda *a: None), parent_registry=registry
        )
    )
    parent = Session.create("p")

    with pytest.raises(SubagentError):
        asyncio.run(
            runtime.start(
                DelegationRequest(prompt="x", tools=("no-such-tool",)), parent=parent
            )
        )


# ===========================================================================
# E2 — an out-of-range integer escapes the schedule tool as OverflowError
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="E2: schedule_create passes after_seconds/every_seconds to timedelta, "
    "so a large-but-valid JSON integer raises OverflowError instead of a result",
)
def test_e2_a_huge_interval_is_a_schedule_error_not_an_overflow():
    """``OverflowError`` is not a ``ScheduleError`` and escapes the tool.

    ``ScheduleCreateParams.after_seconds`` is an ``int`` with no upper bound, and
    JSON integers are unbounded. ``timedelta(seconds=...)`` raises
    ``OverflowError: Python int too large to convert to C int`` for anything past
    ~9.2e18 — before the ``except ScheduleError`` that wraps the record
    construction. The pipeline's body-error path catches ``Exception``, so the
    model sees a ``BODY_ERROR`` traceback rather than the ordinary refusal every
    other bad argument produces.
    """
    from arm.schedule.runtime import ScheduleStore
    from arm.schedule.tools import ScheduleCreateParams, ScheduleTools

    store = ScheduleStore()
    tools = {tool.name: tool for tool in ScheduleTools(store, session_id="s").tools()}
    arguments = {
        "title": "t", "instruction": "i", "kind": "after",
        "after_seconds": 10**18,
    }
    ScheduleCreateParams.model_validate(arguments)  # the validator accepts it

    result = asyncio.run(tools["schedule_create"].body(arguments, None))
    assert result.is_error, "a huge interval should be refused as a result"


# ===========================================================================
# H7 — an unmatched weekly weekday is accepted by the tool and fires instantly
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="H7: schedule_create accepts weekdays=[8], stores scheduled_at='' and "
    "reports 'First fires at soon'; the reminder then fires immediately, once",
)
def test_h7_an_invalid_weekday_is_refused_at_creation():
    """The tool reports success, then the reminder fires at the wrong moment.

    ``weekdays`` is validated only for being non-empty. ``next_occurrence``
    returns ``None`` for a weekday outside 1..7, ``ScheduleTools._create`` leaves
    ``scheduled_at`` empty (``first is None``), and ``due()`` treats a recurring
    task with no ``scheduled_at`` as due now. The user is told "First fires at
    soon", the reminder fires on the next drive regardless of ``at_time``, and it
    is marked inactive — a one-shot at the wrong time instead of a weekly
    reminder.
    """
    from arm.schedule.runtime import ScheduleStore
    from arm.schedule.tools import ScheduleTools

    store = ScheduleStore()
    tools = {tool.name: tool for tool in ScheduleTools(store, session_id="s").tools()}
    arguments = {
        "title": "t", "instruction": "i", "kind": "weekly",
        "at_time": "09:00", "weekdays": [8],
    }

    result = asyncio.run(tools["schedule_create"].body(arguments, None))
    if result.is_error:
        return  # refusing at creation is the correct outcome

    record = store.all()[0]
    assert record.scheduled_at, (
        "the tool reported success but stored no scheduled_at; the reminder is "
        "due immediately (due() treats a recurring task with no scheduled_at as "
        "due now) and will fire at the wrong time"
    )


# ===========================================================================
# E3 — 'max-tokens' is a stop reason nothing can produce
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="E3: STOP_MAX_TOKENS and map_stop_reason('max-tokens') are unreachable "
    "— TURN_END_REASONS excludes it and no loop path emits it",
)
def test_e3_the_max_tokens_stop_reason_is_reachable():
    """A stop reason in the vocabulary that the loop cannot emit.

    ``map_stop_reason`` maps ``"max-tokens"`` to ``STOP_MAX_TOKENS``, ``tool.py``
    renders it ("the subagent ran out of output tokens") and
    ``tests/test_subagent.py`` asserts the mapping. ``TURN_END_REASONS`` is
    ``{completed, blocked, error, aborted}`` — validated by ``Session.append`` —
    and the only other producer of the string is ``contracts.py`` itself, so a
    ``turn/end`` with ``reason="max-tokens"`` is rejected before it is written.
    A child truncated by the model's token limit is therefore reported as
    ``error`` (the default for an unknown reason) rather than as a truncation the
    caller could retry.
    """
    from arm.session.events import TURN_END_REASONS

    # A turn reason outside the validated set cannot be appended at all.
    assert "max-tokens" in TURN_END_REASONS, (
        "no turn/end can carry reason='max-tokens': Session.append validates "
        f"against {sorted(TURN_END_REASONS)}, so STOP_MAX_TOKENS is unreachable"
    )


# ===========================================================================
# E4 — a mailbox error carries the task board's error code
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="E4: confirm_delivery raises code='TEAM_TASK_NOT_FOUND' for a missing "
    "message — a copy-paste of the task code into the mailbox path",
)
def test_e4_a_mailbox_error_does_not_carry_a_task_error_code():
    """The code is what a caller routes on; ``TEAM_TASK_NOT_FOUND`` is wrong here.

    ``confirm_delivery`` raises ``TEAM_TASK_NOT_FOUND`` when a message was never
    queued. ``send`` uses ``TEAM_MEMBER_NOT_FOUND`` for a missing recipient and
    ``TEAM_DUPLICATE_MESSAGE`` for a duplicate id, so the module has a mailbox
    vocabulary — the missing-message case just uses the task board's.
    """
    from arm.team.journal import TeamError, TeamJournal

    journal = TeamJournal(Session.create("lead"))
    with pytest.raises(TeamError) as info:
        journal.confirm_delivery("no-such-message", "alice")
    assert "TASK" not in info.value.code, (
        f"a missing *message* reports {info.value.code!r}, the task board's code"
    )


# ===========================================================================
# H8 — drive() re-samples the clock but not the session
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="H8: two drives over the same store deliver one reminder twice when the "
    "second runs before the first advances the record",
)
def test_h8_a_reminder_is_not_delivered_twice_by_concurrent_drives():
    """``drive()`` is synchronous, so this needs re-entrancy to reproduce.

    ``deliver`` is a caller-supplied callback and ``Session.append`` runs
    listeners inside the commit. A ``deliver`` that appends to the session can
    therefore re-enter a listener that calls ``drive()`` again — and at that
    moment the task has not been advanced, because ``record_delivery`` and
    ``store.put`` happen *after* ``deliver`` returns. The inner drive re-delivers
    the same reminder. The window is the one the module docstring documents for a
    crash between the two writes; this shows it is reachable without a crash.
    """
    from arm.schedule.runtime import (
        KIND_AT,
        ScheduleRecord,
        ScheduleRuntime,
        ScheduleStore,
    )

    store = ScheduleStore()
    store.put(
        ScheduleRecord(
            id="s1", title="t", instruction="i", session_id="sess",
            kind=KIND_AT, at_instant=NOW.isoformat(),
        )
    )
    deliveries: list[str] = []
    runtime: ScheduleRuntime | None = None

    def deliver(session_id: str, body: str, message_id: str) -> bool:
        deliveries.append(message_id)
        if runtime is not None and len(deliveries) == 1:
            # Re-enter while the record is still un-advanced.
            runtime.drive()
        return True

    runtime = ScheduleRuntime(
        store=store, resolve=lambda _s: object(), deliver=deliver, clock=lambda: NOW
    )
    runtime.drive()

    assert len(deliveries) == 1, (
        f"the same reminder was delivered {len(deliveries)} times: {deliveries}"
    )

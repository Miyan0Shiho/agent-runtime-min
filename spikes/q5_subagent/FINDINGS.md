# Q5 — Adversarial verification: `subagent/*`, `team/*`, `schedule/*`

Scope: `src/arm/subagent/{contracts,runtime,inprocess,tool}.py`,
`src/arm/team/{journal,tools}.py`, `src/arm/schedule/{runtime,tools}.py`.

Method: read the modules and their tests, then attack what the tests do not
cover — reachability first, then invariants. Nothing under `src/`, `tests/`,
`docs/`, `README.md` or `REQUIREMENTS.md` was modified.

> **Note on a moving target.** The Lead was editing `src/` throughout this review.
> `cli/main.py`, `agent/loop.py`, `tools/registry.py` and `subagent/inprocess.py`
> all changed while I was reading them: the CLI grew a `ScheduleTimer` and an MCP
> wiring, `ToolRegistry` grew `adopt_from`, and `InProcessProvider._child_registry`
> was rewritten. One defect I found in the first pass — `_child_registry`
> restricting the **parent** in place and pushing a layer that was never popped —
> **was fixed mid-review** and is recorded in §"Fixed while this review ran"
> rather than claimed as a live finding. Everything in the table below was
> re-verified against the tree as of **2026-10-14 20:26 (+08:00)**.

## How to run

```bash
cd spikes && ../.venv/bin/python -m pytest q5_subagent -q
```

Result: **30 xfailed** (0 passed, 0 failed). The 30 tests cover 26 findings
(four findings — D1, H1, H3, E1 — have a second test for a distinct facet).
Every `xfail` is `strict=True`, so
each one is a *currently failing* test that documents a live defect. A fix flips
it to `XPASS(strict)` — a hard failure — which is the ratchet: a bug cannot be
silently re-introduced and a stale marker cannot be left behind.

To see the raw failure behind each marker:

```bash
cd spikes && ../.venv/bin/python -m pytest q5_subagent -q --runxfail
```

`verify_all.py` is a standalone re-check that prints `BUG`/`FIXED` per finding —
useful after a batch of fixes:

```bash
cd spikes && ../.venv/bin/python q5_subagent/verify_all.py
```

Current: **0 of 26 fixed; 26 still broken.**

The probes below are the raw evidence for the findings that need more than a
one-line assertion; they print their observations rather than asserting:

```bash
cd spikes && ../.venv/bin/python q5_subagent/probe_registry.py   # registry isolation
cd spikes && ../.venv/bin/python q5_subagent/probe_team.py       # mailbox + task board
cd spikes && ../.venv/bin/python q5_subagent/probe_schedule.py   # reminders + cron
cd spikes && ../.venv/bin/python q5_subagent/probe_more.py       # weekdays + fork bomb
cd spikes && ../.venv/bin/python q5_subagent/probe_surface.py    # capabilities + run_id
cd spikes && ../.venv/bin/python q5_subagent/probe_deadcode.py   # dead-code census
```

## Status summary

| # | Sev | Module | Finding | Reachable from real usage? | State |
|---|---|---|---|---|---|
| **S1** | **Critical** | subagent | `max_depth` is inert: a child inherits the parent's `depth=0` subagent tool, so a delegation chain runs to `max_active` generations. The fork-bomb guard does not exist | **Yes** — any delegation whose child calls `subagent` | open |
| **D1** | **High** | team | `confirm_delivery` has no caller: the mailbox never delivers. `team_send` says "it will be delivered when they next run" and no run delivers it | **Yes** — `team_send` | open |
| **D2** | **High** | team | `team_spawn` records a roster row and reports "now active"; it creates no agent and never fills `child_session` | **Yes** — `team_spawn` | open |
| **D7** | **High** | schedule | `retry_failed` has no caller, so a reminder whose session was unloaded is `failed` **forever** — "losing a reminder is the one outcome a reminder must never produce" | **Yes** — a reminder that comes due while its session is closed | open |
| **H5** | **High** | schedule | The `ScheduleTimer` runs only *inside* `_run_task`; a reminder due between chat lines never fires — exactly the idle gap the timer was written for | **Yes** — `arm chat` | open |
| **H7** | **High** | schedule | `weekdays=[8]` is accepted, `scheduled_at` is left empty, the tool reports "First fires at soon", and the reminder fires immediately and once | **Yes** — `schedule_create` | open |
| **H8** | **High** | schedule | `drive()` is re-entrant: a `deliver` callback that drives again re-delivers the same reminder, because the record is advanced *after* `deliver` returns | Yes — needs a re-entrant `deliver`; the CLI's `deliver` appends to the session, which runs listeners | open |
| **D3** | **Med** | subagent | Four of the five declared capabilities (`agent_options`, `output_schema`, `depth_limit`, `persona`) cannot be requested through any surface | **Yes** — the model-facing tool | open |
| **D4** | **Med** | subagent | `get` / `cancel` / `shutdown` have no caller: a running child cannot be inspected or stopped | Yes — a hung delegation can only be waited out | open |
| **D5** | **Med** | team | All five team tools are gated behind `ARM_TEAM`, which nothing documents and no test sets | **Yes** — off in every default run | open |
| **D6** | **Med** | schedule | `history()` has no caller: receipts are persisted, pruned, and never read | **Yes** | open |
| **S2** | **Med** | subagent | `DelegationResult.run_id` is the child's session id, not the runtime's `run_id`, so the id the tool hands the model cannot be used to cancel | **Yes** — `subagent` tool `value` | open |
| **T2** | **Med** | team | Deleting a task does not free its id, so a board under its live limit permanently refuses new tasks | **Yes** — `team_task_update delete` then `team_task_create` | open |
| **T3** | **Med** | team | Every event carries `root: root_id` and no projection filters on it: two journals over one log see each other's members and tasks | Hand-constructed (the CLI makes one journal per session) | open |
| **H1** | **Med** | schedule | `weekdays` is checked for non-empty but not for 1..7 | **Yes** — `schedule_create` | open |
| **H4** | **Med** | schedule | `due()` treats a recurring task with no `scheduled_at` as due now, so a restored or hand-written cron/daily record fires at once | Yes — any record not written by `ScheduleTools` | open |
| **H2** | **Med** | schedule | `at_time` is documented as a *local* time and no zone is stored; with the CLI's UTC clock, `09:00` fires at `17:00` in Asia/Shanghai | **Yes** — `schedule_create daily/weekly` | open |
| **H3** | **Med** | schedule | `_next_cron` ANDs day-of-month with day-of-week (cron ORs them when both are restricted), and rejects Sunday as `7` | **Yes** — a standard cron expression | open |
| **T1** | **Med** | team | `reopen` does not re-check dependents: a task stays `in_progress` while its blocker is reopened, and `task_ready()` disagrees | **Yes** — `reopen` | open |
| **E1** | **Low** | subagent | A build failure raises `RuntimeError`, not the documented `SubagentError`, so it escapes the tool's `except` clause | **Yes** — a factory that raises | open |
| **E1b** | **Low** | subagent | A bad `tools` filter raises `RuntimeError`, not `SubagentError` | **Yes** — `tools: ["typo"]` | open |
| **E2** | **Low** | schedule | A large `after_seconds` / `every_seconds` escapes as `OverflowError`, not a result | **Yes** — model-supplied integer | open |
| **E4** | **Low** | team | A missing *message* reports `TEAM_TASK_NOT_FOUND`, the task board's code | **Yes** — `confirm_delivery` | open |
| **E3** | **Low** | subagent | `STOP_MAX_TOKENS` is unreachable: `TURN_END_REASONS` excludes `max-tokens`, so a truncated child is reported as `error` | **Yes** — a truncated child | open |
| **T4** | **Low** | team | The `edit` action cannot clear a description: the tool drops `""` before the journal sees it | **Yes** — `team_task_update edit` | open |
| **H6** | **Low** | schedule | `history(limit=0)` returns everything (`receipts[-0:]` is `receipts[0:]`) | Only via a direct call | open |

---

# Findings

## S1 — `max_depth` is inert; a delegation chain is bounded only by `max_active`

**Severity: Critical.** This is the fork-bomb guard, and it does not exist.

`SubagentTool` is registered **once**, on the parent's registry, with
`depth=0` and `parent_session=<parent>` (`cli/main.py:338`,
`subagent/tool.py:69`). `InProcessProvider._child_registry` copies that registry
into the child (`inprocess.py:193-201`), so the child's `subagent` tool is *the
same object* — still reporting `depth=0` and still delegating **from the parent's
session**. Every generation therefore calls
`assert_max_depth(0, limit=max_depth)`, which passes for any `max_depth >= 1`.

The chain is not unbounded in practice only because each nested delegation holds
a `SubagentRuntime._slots` slot, so it stops at `max_active` (default 8).

### Minimal reproduction

```python
registry = ToolRegistry()
runtime = SubagentRuntime(registry=registry, max_depth=1, max_active=8)

def build(child_session, child_registry, request):
    tool = child_registry.get("subagent")
    depths.append(tool.body.__self__.depth)          # what the child sees
    return Runtime(child_session, MockAdapter([
        tool_response(("c1", "subagent", {"prompt": "go deeper"})),
        text_response("done"),
    ]), child_registry, assembler=assembler, model="mock")
```

### Observed output

```
$ ../.venv/bin/python q5_subagent/verify_all.py | grep S1
BUG    S1 nested delegation bounded         generations=8 depths=[0, 0, 0, 0, 0, 0, 0, 0]

$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k s1
E   AssertionError: the first child's subagent tool reports depth 0, not 1: the
    child inherits the parent's depth-0 tool, so assert_max_depth(0) passes at
    every generation
E   assert 0 == 1
```

Scaling `max_active` proves the bound is not depth:

```
max_depth=1, max_active=  2 -> generations=  2  depths={0}  stop=completed
max_depth=1, max_active=  8 -> generations=  8  depths={0}  stop=completed
max_depth=1, max_active= 40 -> generations= 40  depths={0}  stop=completed
```

### Why the tests miss it

`tests/test_subagent.py::test_a_depth_violation_is_refused_before_the_provider_runs`
calls `runtime.start(DelegationRequest(..., depth=2), ...)` **directly**. That
tests `assert_max_depth`, which is correct. No test drives a *real* nested
delegation, so nothing ever observes that the child's tool carries `depth=0`.

### Exact fix

The child must get its **own** `SubagentTool`, bound to the child's session and
`request.child_depth()`. In `InProcessProvider.start`, after building the
registry and before `factory.build`:

```python
registry = self._child_registry(request)
# The child's delegation tool must carry the child's depth and session, or the
# depth limit is asserted against a constant 0 at every generation.
from .tool import SubagentTool

if registry.get("subagent") is not None and self.runtime is not None:
    registry.register_definition(
        SubagentTool(
            self.runtime,
            parent_session=child_session,
            depth=request.child_depth(),
            provider=request.provider,
        ).register().definition,
        SubagentTool(
            self.runtime,
            parent_session=child_session,
            depth=request.child_depth(),
            provider=request.provider,
        ).run,
    )
```

This needs the provider to hold the `SubagentRuntime` (it currently holds only
`parent_registry`), so `InProcessProvider.__init__` gains a `runtime` argument
and `cli/main.py:330` passes `runtime=subagents`. A simpler variant that keeps the
same effect: pass `depth` through the `ChildFactory` callable and let the CLI's
`build_child` register the depth-aware tool on `child_registry` before returning.

---

## D1 — The mailbox never delivers: `confirm_delivery` has no caller

**Severity: High.** The feature is configurable but not usable — the fourth
occurrence of this class in this project.

`TeamJournal.confirm_delivery` (`team/journal.py:518`) is the **only** thing that
writes `TEAM_MESSAGE_DELIVERED`, and the module docstring is explicit that
delivery must be recorded by the recipient. Nothing in `src/` calls it:

```
$ grep -rnF "confirm_delivery" src/ --include=*.py
src//arm/team/journal.py:518:    def confirm_delivery(self, message_id: str, recipient: str) -> bool:
```

`team_send` tells the model *"Queued a message for alice. It will be delivered
when they next run."* — and there is no "next run": `team_spawn` creates no agent
(D2), and even a hand-built teammate has no code path that reads `pending_for`
and confirms. Every message stays `delivered=False` forever, so
`max_pending_per_member` fills up and the mailbox eventually refuses new sends.

### Minimal reproduction

```python
tools = {t.name: t for t in TeamTools(journal).tools()}
await tools["team_send"].body({"name": "alice", "message": "review this"}, None)
for name, tool in tools.items():          # drive every registered tool
    await tool.body({"name": "alice", "id": pending[0].message_id}, None)
assert journal.messages()[0].delivered    # fails
```

### Observed output

```
$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k d1
E   AssertionError: no code path marks a queued message delivered; confirm_delivery is dead code
E   assert False
E   AssertionError: confirm_delivery is defined and tested but never called from src/:
E   assert []
```

### Exact fix

Delivery needs a real consumer. The smallest honest fix is a `team_inbox` tool
the teammate calls at the start of its turn, which returns `pending_for(actor)`
and confirms each one:

```python
def _inbox(self) -> RegisteredTool:
    async def body(_arguments, _execution):
        pending = self.journal.pending_for(self.actor)
        for message in pending:
            self.journal.confirm_delivery(message.message_id, self.actor)
        return ToolResult.ok("\n".join(f"{m.sender}: {m.body}" for m in pending))
    ...
```

That is only meaningful once D2 makes a teammate exist; the two fixes are one
piece of work.

---

## D2 — `team_spawn` creates no agent

**Severity: High.**

`TeamTools._spawn` (`team/tools.py:112-149`) validates the name, appends a
`TEAM_MEMBER` event, flips the phase to `active`, and returns *"Added teammate
alice (now active)"*. It never touches `SubagentRuntime`, and `TeamMember.child_session`
— the field that exists to point at the teammate's agent — is never set. The
`task` argument is echoed into the message text and dropped.

`team/journal.py` and `team/tools.py` do not import `subagent` at all:

```
$ grep -rn "subagent\|SubagentRuntime" src/arm/team/
(no matches)
```

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_team.py
spawn result content: Added teammate alice (now active). | Its task: review the parser | ...
member.child_session: ''
member.phase        : active
>>> no child session, no agent: the 'teammate' is a roster row
```

```
$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k d2
E   AssertionError: team_spawn reported the teammate active but no child session exists:
    the teammate has no agent
E   assert ''
```

### Exact fix

`TeamTools` needs a `SubagentRuntime` and a parent session. In `_spawn`, after
`set_member_phase(name, MEMBER_ACTIVE)`, start a delegation for the initial task
and record its child session:

```python
running = await self.subagents.start(
    DelegationRequest(prompt=task, provider="in-process", label=name,
                      depth=self.depth, max_depth=DEFAULT_MAX_DEPTH),
    parent=self.parent_session,
)
self.journal.set_member_phase(
    name, MEMBER_ACTIVE, child_session=running.child_session.session_id
)
```

`cli/main.py:355` then passes `TeamTools(team, subagents=subagents, parent_session=session)`.

---

## D7 — A failed reminder is never retried

**Severity: High.** It breaks the module's own stated invariant.

`ScheduleRuntime.drive` marks a task `STATUS_FAILED` when `resolve` returns `None`
or raises (`schedule/runtime.py:657-668`). `retry_failed`
(`schedule/runtime.py:730`) exists to put it back — and nothing calls it:

```
$ grep -rnF "retry_failed" src/ --include=*.py
src//arm/schedule/runtime.py:730:    def retry_failed(self) -> int:
```

`resolve` in the CLI returns `session if session_id == session_id_ else None`
(`cli/main.py:427-428`), so **any reminder whose session is not the currently
open one is failed on its first drive**. Since nothing ever reactivates it, the
reminder is silently and permanently lost — the one outcome
`docs/contract-amendments.md:442` says must never happen
("丢失提醒是提醒唯一绝不能产生的结果").

### Minimal reproduction

```python
store.put(ScheduleRecord(id="s1", ..., kind=KIND_AT, at_instant=NOW.isoformat()))
ScheduleRuntime(store, resolve=lambda _sid: None, deliver=lambda *_: True,
                clock=lambda: NOW).drive()          # session not loaded
later = ScheduleRuntime(store, resolve=lambda _sid: object(), deliver=lambda *_: True,
                        clock=lambda: NOW + timedelta(hours=1))
assert later.drive()                                # fails: still 'failed'
```

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_schedule.py
drive #1: []
task status now      : failed
drive #2 (session now loaded): []
task status now      : failed
>>> the reminder never fires, even once the session exists
```

### Exact fix

Reactivate on the way in. `ScheduleRuntime.drive` should call `retry_failed()`
at the top, before sampling the clock — a failed task is a *retryable* state, not
a terminal one:

```python
def drive(self) -> list[DeliveryReceipt]:
    self.retry_failed()      # a failure is transient: a session that was not
                             # loaded yet, not an abandoned reminder
    now = self.clock()
    ...
```

Do **not** leave this to the caller: the caller is the loop, and the loop only
runs when a session is loaded — which is the case where `resolve` already works.

---

## H5 — The timer does not cover the idle gap it exists for

**Severity: High.**

`ScheduleTimer` was added to `schedule/runtime.py:746` with a docstring that
states the problem exactly: *"Without this, a reminder only fires if the user
happens to be mid-turn when its time arrives."* But the only construction site is
`_run_task` (`cli/main.py:526`), which starts it and stops it in a `finally`:

```python
timer = ScheduleTimer(scheduler, interval=5.0) if scheduler is not None else None
if timer is not None:
    timer.start()
try:
    ... await runtime.run_until_idle()
finally:
    if timer is not None:
        await timer.stop()
```

`_cmd_chat` calls `asyncio.run(_run_task(...))` once per input line
(`cli/main.py:586`). Between lines there is no event loop and no timer. So the
reminder fires only while a task is already running — precisely the case the
timer's own docstring says is not enough.

### Observed output

```
$ ../.venv/bin/python q5_subagent/verify_all.py | grep H5
BUG    H5 timer covers idle chat            ScheduleTimer not in _cmd_chat
```

### Exact fix

Own the timer for the lifetime of the chat loop, not of one task. In `_cmd_chat`,
run the whole loop inside one event loop and start the timer there:

```python
async def _chat_loop(...):
    timer = ScheduleTimer(scheduler, interval=5.0) if scheduler else None
    if timer: timer.start()
    try:
        while True:
            line = await asyncio.to_thread(input, "> ")
            ...
            await _run_task(runtime, session, line, verbose=True)   # no scheduler=
    finally:
        if timer: await timer.stop()
```

---

## H7 — An unmatched weekday is accepted, reported as "soon", then fires immediately

**Severity: High.** Silent wrong behaviour on a model-supplied argument.

`ScheduleRecord.__post_init__` validates that `weekdays` is non-empty but not
that each entry is in 1..7 (`schedule/runtime.py:321-324`). `next_occurrence`
searches `offset in range(0, 8)` and compares `isoweekday() in weekdays`, so
`weekdays=(8,)` never matches and returns `None`. `ScheduleTools._create` then
leaves `scheduled_at` empty, and `due()` treats a recurring task with no
`scheduled_at` as due immediately (`schedule/runtime.py:506-508`).

The result: the tool reports success and *"First fires at soon"*, the reminder
fires on the next drive regardless of `at_time`, and it is marked `inactive` — a
one-shot at the wrong moment instead of a weekly reminder.

### Minimal reproduction

```python
await tools["schedule_create"].body(
    {"title": "t", "instruction": "i", "kind": "weekly",
     "at_time": "09:00", "weekdays": [8]}, None)
```

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_more.py
weekdays=[8]: next_occurrence -> None
record accepted with weekdays=(8,): (8,)
store.due() at NOW -> ['w'] (a weekly task is due immediately despite a future time)
drive: ['w']
status after: inactive

$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k h7
E   AssertionError: the tool reported success but stored no scheduled_at; the reminder
    is due immediately (due() treats a recurring task with no scheduled_at as due now)
    and will fire at the wrong time
E   assert ''
```

### Exact fix

Two lines, both needed. Validate the range where every other field is validated
(`schedule/runtime.py:321`):

```python
if self.kind == KIND_WEEKLY:
    if not self.weekdays:
        raise ScheduleError("a 'weekly' task needs at least one weekday",
                            code="SCHEDULE_INVALID")
    if any(not 1 <= day <= 7 for day in self.weekdays):
        raise ScheduleError(
            f"weekdays must be ISO days 1..7 (Monday is 1), got {list(self.weekdays)}",
            code="SCHEDULE_INVALID",
        )
```

And refuse rather than silently mis-schedule in `ScheduleTools._create`, where
`first` is computed (`schedule/tools.py:166-172`):

```python
if first is None and record.recurring:
    first = next_occurrence(record, now=now, fired_at=now)
if first is None and record.recurring:
    return ToolResult.fail(f"{kind!r} has no occurrence; check its schedule fields")
```

---

## H8 — `drive()` can deliver the same reminder twice without a crash

**Severity: High.** The docstring calls this window "documented" for a *crash*;
it is reachable without one.

`drive()` calls `deliver` **before** it advances the record
(`schedule/runtime.py:687-723`). `deliver` is a caller-supplied callback, and the
CLI's `_deliver` appends to the session (`cli/main.py:430`), which runs listeners
inside the commit. Any listener that drives the scheduler again — or a `deliver`
that drives on failure — re-enters `drive()` while `scheduled_at` is still in the
past, and the same reminder is delivered a second time.

### Minimal reproduction

```python
def deliver(session_id, body, message_id):
    deliveries.append(message_id)
    if len(deliveries) == 1:
        runtime.drive()          # re-enter while the record is un-advanced
    return True
runtime.drive()
```

### Observed output

```
$ ../.venv/bin/python q5_subagent/verify_all.py | grep H8
BUG    H8 no duplicate delivery             2 delivery/deliveries: ['schedule-s1-1791892800', 'schedule-s1-1791892800']
```

### Exact fix

Advance the record *before* delivering, and roll the advance back if delivery is
not acknowledged. The receipt then records an intention that delivery confirms,
rather than the current order:

```python
# Advance first: a re-entrant drive must not see the same occurrence as due.
for task in still_due:
    occurrence = task.due_at() or moment
    nxt = next_occurrence(task, now=moment, fired_at=occurrence)
    self.store.put(_replace(
        task,
        status=STATUS_ACTIVE if nxt else STATUS_INACTIVE,
        scheduled_at=nxt.isoformat() if nxt else "",
    ))
try:
    acknowledged = self.deliver(session_id, body, message_id)
except Exception:
    acknowledged = False
if not acknowledged:
    for task, previous in zip(still_due, originals):   # restore
        self.store.put(previous)
```

`request_drive`'s `_driving` flag is not a fix: it guards *its own* recursion, not
a re-entrant `drive()` call, and `drive()` is public and called directly by the
loop.

---

## D3 — Four of five capabilities cannot be requested

**Severity: Medium.** Dead configuration.

`CAPABILITY_ORDER` names five capabilities (`contracts.py:50-56`) and
`InProcessProvider` advertises all five (`inprocess.py:54-56`). `SubagentTool.run`
builds a `DelegationRequest` with only `prompt`, `provider`, `label`, `depth`,
`max_depth` and `tools` (`tool.py:108-115`) — never `output_schema`, `persona` or
`model` — and always passes `max_depth=DEFAULT_MAX_DEPTH`. `DelegationRequest.requires`
therefore returns `False` for four of them, so `assert_capabilities` can never
raise through the tool.

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_surface.py
SubagentParams fields: ['label', 'prompt', 'tools']
DelegationRequest fields: ['depth', 'label', 'max_depth', 'model', 'output_schema', 'persona', 'prompt', 'provider', 'tools']
reachable only by direct API: ['depth', 'max_depth', 'model', 'output_schema', 'persona', 'provider']

does SubagentTool.run mention output_schema?  False
does SubagentTool.run mention persona?       False
does SubagentTool.run mention model?         False
```

### Exact fix

Expose the three fields the runtime already supports, in `SubagentParams`
(`tool.py:40-58`), and thread them through `tool.py:108-115`:

```python
persona: str = Field(default="", description="A persona for this delegation.")
output_schema: dict[str, Any] | None = Field(
    default=None, description="A JSON Schema the answer must satisfy.")
model: str = Field(default="", description="Override the model for the child.")
```

with `persona=..., output_schema=..., model=...` passed to `DelegationRequest`.
`depth_limit` needs no new field: `max_depth` should be `self.max_depth` (a
constructor argument) rather than the constant.

---

## D4 — A running subagent cannot be inspected or stopped

**Severity: Medium.**

```
$ grep -rnF "runtime.get(" src/ --include=*.py     # (no matches)
$ grep -rnF "shutdown()"    src/ --include=*.py     # (no matches)
$ grep -rnF ".cancel("      src/ --include=*.py
src//arm/schedule/runtime.py:816:            task.cancel()
src//arm/jobs/registry.py:556:                await job.cancel()
src//arm/jobs/registry.py:579:            job.task.cancel()
src//arm/subagent/runtime.py:294:            running.result.cancel()
src//arm/subagent/runtime.py:306:                await self.cancel(run_id)
src//arm/subagent/inprocess.py:154:                task.cancel()
```

Every hit is an internal `asyncio` call or `SubagentRuntime` calling itself.
`SubagentTool` has no cancel argument and no run id to cancel with, and
`shutdown()` is never called at process exit — so children outliving a session
are never torn down.

### Exact fix

Add a `subagent_cancel` tool bound to the runtime, and call `await
subagents.shutdown()` in the CLI's `finally` next to `handle.close()` (`cli/main.py:563`). The tool is
only usable once S2 is fixed (the model must be handed an id `cancel` accepts).

---

## D5 — The team tools are off unless `ARM_TEAM` is set

**Severity: Medium.**

`cli/main.py:353` registers the five team tools only `if os.environ.get("ARM_TEAM")`.
The variable appears nowhere in `README.md`, `docs/` or `REQUIREMENTS.md`, and no
test sets it. In every default run the entire `team/` package is unreachable.

### Observed output

```
$ grep -rn "ARM_TEAM" README.md docs/ REQUIREMENTS.md
(no matches)
$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k d5
E   AssertionError: the team tools are registered only when ARM_TEAM is set,
    which no documented path sets
```

### Exact fix

Either register them unconditionally — the comment above the gate says the reason
is to avoid offering five unused tools, but that trade-off was already rejected
for plan mode (`cli/main.py:252-254`: *"the tool catalogue must not change with
the mode"*) — or document `ARM_TEAM` in `README.md` and add a test that sets it.
Unconditional registration is consistent with the rest of the registry.

---

## D6 — Delivery receipts are written and never read

**Severity: Medium.**

```
$ grep -rnF ".history(" src/ --include=*.py
(no matches)
```

`record_delivery` appends and `_prune_history` enforces both bounds; both are
tested. `history()` (`schedule/runtime.py:515`) is the only reader, and nothing
calls it. The entire two-bound retention policy — including the careful "always
keep the newest record" rule — protects a list no user or model can see. There is
also no `schedule_history` tool, so "what fired?" is unanswerable.

### Exact fix

Expose it. Add to `ScheduleTools.tools()`:

```python
def _history(self) -> RegisteredTool:
    async def body(arguments, _execution):
        limit = int(arguments.get("limit") or 20)
        receipts = [r for r in self.store.history(limit=limit)
                    if r.session_id == self.session_id]
        ...
```

---

## S2 — The id the tool reports cannot be used to cancel

**Severity: Medium.**

`InProcessProvider` sets `result.run_id = result_key(child_session)` =
`"subagent-<8 hex>"` (`inprocess.py:257-259`, used at `:247` and `:261`), while
`SubagentRuntime.start` keys `_running` by the `uuid4` it generated
(`runtime.py:197`, `:209`). `SubagentTool.run` returns `result.run_id` in its
`value` (`tool.py:131`), so the model is handed an identifier that `runtime.get`
and `runtime.cancel` cannot resolve.

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_runid.py
  in-process running.run_id: 6b6e3997-8208-43bb-b350-545511ae1c13
  in-process result.run_id : subagent-6b6e3997
  tool reports to the model: subagent-6b6e3997
  cancel(what the model sees) -> False
```

### Exact fix

Thread the runtime's `run_id` into `_drive` and use it for every
`DelegationResult`. `_drive` already receives `result`; add `run_id` to its
signature and replace both `result_key(child_session)` calls
(`inprocess.py:247`, `:261`) with `run_id`. The two early-return paths in
`drive()` (`:128`, `:141`) already use the correct `run_id`, which is what makes
the mismatch visible: the *same* run reports two different ids depending on
whether it was cancelled.

---

## T2 — Deleting a task does not free its id

**Severity: Medium.**

`create_task` checks `len(live) >= self.max_tasks` — live tasks only
(`journal.py:584-589`) — then scans `task-1..task-N` against `board`, which
includes deleted tasks (`journal.py:594-602`). With `max_tasks` tasks ever created
and one deleted, the board holds `max_tasks - 1` live tasks and still refuses.

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_team.py
ids: ['task-1', 'task-2', 'task-3']
4th -> TEAM_TASK_LIMIT
after deleting task-1, live: ['task-2', 'task-3']
create after delete -> TEAM_TASK_LIMIT
```

### Exact fix

The comment at `journal.py:591-593` claims "a taken id means the board is at its
limit" — true only while nothing is deleted. Scan for the lowest **unused**
number without the `max_tasks` ceiling on the *scan*, and let the live-count
check above remain the real limit:

```python
next_number = 1
while f"task-{next_number}" in board:
    next_number += 1
task_id = f"task-{next_number}"
```

The `next_number > self.max_tasks` guard must go: the live-count check already
bounds the board, and the guard is what makes a deleted id permanently
unavailable.

---

## T3 — `root_id` is written and never used to filter

**Severity: Medium.**

`_append` writes `{"root": self.root_id, **data}` on every event
(`journal.py:376-377`), which is what makes two teams in one log
distinguishable. `members`, `messages`, `tasks` and `summary` all iterate
`self.session` and check only `event.type` — none reads `event.data["root"]`.

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_team.py
journal A members: ['alice', 'bob']
journal B members: ['alice', 'bob']
>>> both journals fold the same events; root_id filters nothing
```

### Exact fix

Filter in one place, and have every projection use it:

```python
def _events(self):
    """This team's events. A journal must not fold another team's log."""
    for event in self.session:
        if event.data.get("root") == self.root_id:
            yield event
```

Then replace `for event in self.session:` with `for event in self._events():` in
`members`, `messages` and `tasks`. Note this changes behaviour for events written
before the field existed, so a migration or a default of "no root means mine" is
worth considering.

---

## H1 — `weekdays` is not range-checked

**Severity: Medium.** (Same root cause as H7; listed separately because the
validation gap is in the record, and H7 is the user-visible consequence.)

```
$ ../.venv/bin/python q5_subagent/probe_more.py
weekdays=[8]: next_occurrence -> None
weekdays=[0]: next_occurrence -> None
weekdays=[-1]: next_occurrence -> None
record accepted with weekdays=(8,): (8,)
```

### Exact fix

See H7.

---

## H4 — A recurring task with no `scheduled_at` is due immediately

**Severity: Medium.**

`ScheduleStore.due` returns a recurring task with no `scheduled_at`
unconditionally (`schedule/runtime.py:506-508`), on the stated theory that "a
schedule created with no explicit start should begin". `ScheduleTools` always
computes and stores a first occurrence, so the *tool* path is safe — but
`ScheduleStore.put` is the public write API, and `_load_schedule`
(`cli/main.py:127`) reconstructs records straight from `schedule.json`. Any
record that lost its `scheduled_at` (an older file, a hand edit, a partial write)
fires the moment the process starts.

### Observed output

```
$ ../.venv/bin/python q5_subagent/verify_all.py | grep H4
BUG    H4 no premature recurring fire       1 receipt(s) for a 09:00 cron driven at 12:00
```

### Exact fix

Compute the first occurrence from the record's own fields rather than treating
"no time" as "now":

```python
if when is None:
    if task.recurring:
        nxt = next_occurrence(task, now=moment, fired_at=moment)
        if nxt is not None and nxt <= moment:
            out.append(task)
    continue
```

---

## H2 — `at_time` is "local" with no zone, and the clock is UTC

**Severity: Medium.**

`ScheduleRecord.at_time` is documented as "the local time of day"
(`schedule/runtime.py:281`), the tool description says "the local time, HH:MM"
(`schedule/tools.py:73`), and `_parse_local_time` validates only the shape.
`next_occurrence` does `now.replace(hour=..., ...)`, which keeps `now`'s tzinfo —
and the CLI's clock is `lambda: datetime.now(timezone.utc)`
(`schedule/runtime.py:607`) while `ScheduleTools` uses
`datetime.now(timezone.utc)` too (`schedule/tools.py:105`). No zone is stored on
the record, so there is no argument that could fix it.

### Observed output

```
$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k h2
E   AssertionError: a 09:00 daily reminder lands at 17:00 in the user's zone
    (the CLI clock is UTC and no zone is stored on the record)
E   assert 17 == 9
```

### Exact fix

Store the zone with the record and resolve `at_time` in it:

```python
# ScheduleRecord gains:
timezone_name: str = ""      # IANA name, e.g. "Asia/Shanghai"

# next_occurrence, for daily/weekly:
zone = ZoneInfo(record.timezone_name) if record.timezone_name else now.tzinfo
candidate = now.astimezone(zone).replace(hour=hour, minute=minute,
                                         second=second, microsecond=0)
```

`ScheduleTools` fills `timezone_name` from the process's local zone
(`datetime.now().astimezone().tzinfo`) or a new `timezone` tool argument, and
`__post_init__` refuses a name `ZoneInfo` cannot resolve. This also makes DST
correct, which the current arithmetic cannot be.

---

## H3 — Cron day-of-month/day-of-week is ANDed, and Sunday `7` is rejected

**Severity: Medium.** A standard cron expression fires months late, or never.

`_next_cron` requires *every* field to match (`schedule/runtime.py:240-248`), so
`0 0 1 * 1` means "the 1st, if it is a Monday". Every cron implementation ORs the
two when both are restricted. Separately, day-of-week is matched against
`candidate.isoweekday() % 7`, which yields 0..6, so `7` — accepted by Vixie cron
as Sunday — matches nothing and the four-year search raises.

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_more.py
cron '0 0 1 * 1' from 2026-10-13T12:00:00+00:00
  -> 2027-02-01 00:00:00+00:00 isoweekday 1
  real cron (Vixie) would give Mon 2026-10-19 00:00
0 8 * * 7 -> '0 8 * * 7' has no occurrence in the next four years
```

### Exact fix

Two changes in `_next_cron`:

```python
# 1. Sunday is 0 or 7.
dow_value = candidate.isoweekday() % 7
dow_matches = matches(dow, dow_value, 0, 6) or (dow_value == 0 and matches(dow, 7, 0, 7))

# 2. When both dom and dow are restricted, cron ORs them.
dom_restricted = dom.strip() != "*"
dow_restricted = dow.strip() != "*"
if dom_restricted and dow_restricted:
    day_ok = matches(dom, candidate.day, 1, 31) or dow_matches
else:
    day_ok = matches(dom, candidate.day, 1, 31) and dow_matches
```

and drop `dow_matches` from the `and` chain, substituting `day_ok`.

---

## T1 — `reopen` does not re-check dependents

**Severity: Medium.**

`task_ready` is consulted by `claim` (`journal.py:712`) and `reassign`
(`journal.py:849`) only. `reopen` (`journal.py:759-773`) changes a completed task
back to `in_progress` and calls nothing. A dependent claimed while its blocker was
complete stays `in_progress`, so two members work a chain believing it is
satisfied — and `task_ready` reports `False` for the dependent, so the board's own
summary disagrees with the task's phase.

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_team.py
t4 claimed; now REOPEN t2
t2 phase: in_progress
t4 phase: in_progress (in progress despite a reopened blocker)
t4 ready now: False
>>> an in-progress task is never re-checked against reopened blockers
```

### Exact fix

Either refuse the reopen while dependents are in progress, or release them. The
first is smaller and matches how `delete` already handles dependents
(`journal.py:866-876`):

```python
if action == "reopen":
    ...
    dependents = [t.id for t in self.live_tasks().values()
                  if current.id in t.blocked_by and t.phase == TASK_IN_PROGRESS]
    if dependents:
        raise TeamError(
            f"task {current.id!r} is a blocker for in-progress {dependents}; "
            "release them before reopening it",
            code="TEAM_TASK_HAS_DEPENDENTS",
        )
```

---

## E1 / E1b — A `RuntimeError` escapes where `SubagentError` is documented

**Severity: Low.** The tool's contract is that every refusal is a `SubagentError`
it turns into a result (`tool.py:117-124`).

`inprocess.py:115` wraps a factory failure in `RuntimeError`, and `inprocess.py:200`
wraps a bad `tools` filter in `RuntimeError`. Neither is a `SubagentError`, so
both escape `SubagentTool.run`, travel through the pipeline's body-error path, and
reach the model as a `BODY_ERROR` with the message re-wrapped.

### Observed output

```
$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k e1
E   ValueError: the child could not be constructed
E   RuntimeError: could not build the child agent: the child could not be constructed
E   arm.tools.registry.UnknownToolRestrictionError: restriction names unknown tools ['no-such-tool']; known tools are []
E   RuntimeError: could not restrict the child's tools: restriction names unknown tools ['no-such-tool']; known tools are []
```

### Exact fix

```python
# inprocess.py:115
raise SubagentError(f"could not build the child agent: {exc}",
                    code="SUBAGENT_BUILD_FAILED") from exc

# inprocess.py:200
raise SubagentError(f"could not restrict the child's tools: {exc}",
                    code="SUBAGENT_BAD_TOOL_FILTER") from exc
```

---

## E2 — A huge interval escapes as `OverflowError`

**Severity: Low.**

`ScheduleCreateParams.after_seconds` / `every_seconds` are unbounded `int`s, and
JSON integers are unbounded. `timedelta(seconds=...)` raises `OverflowError` past
~9.2e18 — *before* the `except ScheduleError` that wraps record construction
(`schedule/tools.py:123`, `:139`).

### Observed output

```
$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k e2
E   OverflowError: Python int too large to convert to C int
../src/arm/schedule/tools.py:123: OverflowError
```

### Exact fix

Bound the field in the model, so the pipeline's argument validation refuses it
before the body runs:

```python
after_seconds: int = Field(default=0, ge=0, le=366 * 24 * 3600,
                           description="For 'after': the delay, at most one year.")
```

or wrap the arithmetic in `try/except (OverflowError, OSError)` and return
`ToolResult.fail`.

---

## E4 — A mailbox error carries the task board's code

**Severity: Low.** A copy-paste, but the code is what a caller routes on.

```
$ grep -n "TEAM_TASK_NOT_FOUND" src/arm/team/journal.py
309:                code="TEAM_TASK_NOT_FOUND",      # task depends on a missing task — correct
314:                code="TEAM_TASK_NOT_FOUND",      # task depends on a deleted task — correct
542:            f"no message {message_id!r} was ever queued", code="TEAM_TASK_NOT_FOUND"   # wrong
657:            raise TeamError(f"no task {task_id!r}", code="TEAM_TASK_NOT_FOUND")       # correct
```

### Observed output

```
$ ../.venv/bin/python q5_subagent/verify_all.py | grep E4
BUG    E4 mailbox error code                code='TEAM_TASK_NOT_FOUND'
```

### Exact fix

```python
# journal.py:542
raise TeamError(f"no message {message_id!r} was ever queued",
                code="TEAM_MESSAGE_NOT_FOUND")
```

---

## E3 — `STOP_MAX_TOKENS` is unreachable

**Severity: Low.**

`map_stop_reason` maps `"max-tokens"` (`contracts.py:219-220`), `tool.py:159`
renders it, and `tests/test_subagent.py` asserts it — but `TURN_END_REASONS` is
`{completed, blocked, error, aborted}` and `Session.append` validates against it,
so no `turn/end` can carry the reason. A child truncated by the model's output
limit is reported as `error` (the default for an unknown reason) rather than as a
truncation the caller could retry with a larger budget.

### Observed output

```
$ ../.venv/bin/python -m pytest q5_subagent -q --runxfail -k e3
E   AssertionError: no turn/end can carry reason='max-tokens': Session.append
    validates against ['aborted', 'blocked', 'completed', 'error'], so
    STOP_MAX_TOKENS is unreachable
E   assert 'max-tokens' in frozenset({'aborted', 'blocked', 'completed', 'error'})
```

### Exact fix

Either add `max-tokens` to `TURN_END_REASONS` and emit it from the loop when the
model stops on a length finish reason, or remove `STOP_MAX_TOKENS` from the
subagent vocabulary and its render branch. Leaving it declared-but-unproducible is
the one option that is definitely wrong.

---

## T4 — The `edit` action cannot clear a description

**Severity: Low.**

`TeamTools._task_update` guards with `if arguments.get("description")`
(`team/tools.py:310-311`), so `""` is dropped before `journal.update_task` sees
it. The journal *can* clear it — `_apply_task_action` distinguishes `None` from
`""` (`journal.py:798-800`) — but the tool cannot express it, and its own schema
says "New description, for 'edit'".

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_team.py
edit with description='' -> task-1 is now pending at revision 2
description now: 'original'
>>> an empty description is silently ignored (cannot be cleared)
```

### Exact fix

Forward the key whenever it is present, rather than whenever it is truthy:

```python
if action == "edit":
    if arguments.get("subject"):
        changes["subject"] = str(arguments["subject"])
    if "description" in arguments:
        changes["description"] = str(arguments["description"])
```

`TaskUpdateParams.description` defaults to `""`, so this makes every `edit`
without a description clear it. If that is unwanted, make the field
`str | None = None` and pass it only when it is not `None`.

---

## H6 — `history(limit=0)` returns everything

**Severity: Low.**

`receipts[-limit:]` with `limit=0` is `receipts[0:]` — the whole list
(`schedule/runtime.py:527`). A negative limit is similarly surprising.

### Observed output

```
$ ../.venv/bin/python q5_subagent/probe_schedule.py
history(limit=0) -> 3 records (expected 0 for limit=0)
history(limit=-1) -> 2 records
```

### Exact fix

```python
# runtime.py:515-527
if limit < 0:
    raise ScheduleError("the history limit must not be negative")
receipts = [...]
if limit == 0:
    return ()
return tuple(reversed(receipts[-limit:]))
```

---

# Fixed while this review ran

Recorded so the work is not redone, and so the ratchet is understood: I found
these in the first pass and the Lead fixed them mid-review. They are **not**
counted in the 26 above.

**F1 — `InProcessProvider._child_registry` restricted the parent in place.**
The original body called `self.parent_registry.restrict(...)` and then
`return self.parent_registry`, so the child's filter *permanently removed tools
from the parent* and pushed a layer that was never popped. Observed before the
fix:

```
$ ../.venv/bin/python q5_subagent/probe_registry.py
parent tools before: ('bash', 'read', 'write')
child registry names: ('read',)
parent tools AFTER  : ('read',)          <-- the parent lost write and bash
registry depth after: 2
second delegation (tools=('write',)): RuntimeError: could not restrict the
  child's tools: restriction names unknown tools ['write']; known tools are ['read']
```

The current code builds a separate registry with `adopt_from` and returns it, and
the probe now reports `parent tools AFTER: ('bash', 'read', 'write')`,
`registry depth after: 1`, and a second delegation that succeeds. The
unbounded-layer concern is resolved by the same change.

---

# Where I looked and found nothing

Stated plainly, with the evidence, so a miss is not mistaken for an oversight.

* **`_next_cron` is correct at the hour boundary and for February 29.**
  `*/15` from `12:59:30` gives `13:00:00`; `0 0 29 2 *` from 2026-10-13 gives
  `2028-02-29 00:00:00+00:00`; `0 0 * * 0` gives the next Sunday. The four-year
  bound is enough for every leap-year expression and the search costs ~1.3 s for
  a never-matching one (measured), which is acceptable for a once-per-wake-up
  evaluation. The dom/dow and Sunday-7 issues (H3) are separate.
* **`every` does not drift.** `anchor + steps * seconds` is exact; two
  consecutive occurrences are 86400.0 s apart across a UTC day boundary. The
  anchor is re-parsed from the stored string on every call, so a JSON round-trip
  cannot corrupt the phase.
* **`_prune_history` never drops the newest record.** With `history_records=1`
  and three receipts dated 400 days ago, `history()` returns `['s2']` — the
  newest. The `if not kept and self._history: kept = [self._history[-1]]` guard
  works, and `kept[-self.history_records:]` keeps the tail.
* **`assert_task_graph` is correct for a diamond.** `task-4` blocked by
  `[task-2, task-3]`, each blocked by `task-1`, creates and resolves correctly:
  not ready before `task-1` completes, not ready after only one of `task-2`/
  `task-3`, ready when both are. The DFS cycle detection reports a three-task
  cycle and a refused cycle leaves the board unchanged.
* **`create_task`'s id allocation terminates.** The `while` loop is bounded by
  `max_tasks` and raises `TEAM_TASK_LIMIT`. The *policy* is wrong (T2) but the
  loop itself cannot spin.
* **`write_scopes` does not claim to lock anything.** The tool description says
  in as many words: "They are recorded for coordination; they are not a lock."
  It is deduplicated, persisted, and read back — consistent with what it
  promises.
* **`drive()` does not deliver the same reminder twice across two sequential
  drives.** Two runtimes over one store: the first delivers and marks the
  one-shot `inactive`, the second finds nothing due. The duplicate in H8 needs
  re-entrancy, not two calls.
* **`send` / `pending_for` / mailbox limits are correct.** Self-messages, unknown
  recipients, provisioning recipients, empty bodies, byte-measured oversize
  (multi-byte counted as bytes), the pending cap, and duplicate ids are all
  refused with the right codes, and a refused send appends no event.
* **`members` / `set_member_phase` / the member limit are correct.** The
  last-record-wins fold survives a restart, a `failed` member can be replaced and
  does not count against the limit, and `lead` is reserved.
* **`summary()` agrees with `task_ready`.** I wrote a test asserting they could
  diverge and it **XPASSed** — so I removed it rather than report a finding I
  could not demonstrate. The `ready` count uses exactly the same predicate as
  `task_ready`, and I could not construct a board where they disagree
  (`assert_task_graph` refuses a dependency on a deleted task, and `delete`
  refuses while dependents exist).
* **`validate_output_schema` is live, not dead.** The task brief listed it as a
  candidate; `SubagentRuntime.start` calls it at `runtime.py:183` on every
  delegation, before the provider is consulted. It is only *unreachable through
  the tool*, because the tool cannot pass `output_schema` at all — which is D3,
  not a defect in the validator.
* **`consumed_work`, `final_assistant_output` and `turn_stop_reason` are all
  live**, called from `inprocess.py:222-224`. `consumed_work`'s result is bound
  to `turns` and never read (`inprocess.py:224`), which is dead *within* a live
  function — the comment at `:230-232` says so explicitly ("`turns` is recorded
  rather than used as the test"), so it is intentional, not a finding.
* **The slot accounting in `SubagentRuntime.start` is correct for every path the
  real provider can take.** `start()` cancelled mid-`await` releases the slot
  (`except BaseException` at `runtime.py:202-207`), a failed start releases it, a
  result resolved before the callback is attached still fires the callback
  (verified: `active`/`_slots` go to 0/0), and `cancel` twice is idempotent
  because `_release` pops the dict before decrementing. S3 is a real gap only for
  a provider whose future is *never* resolved, which `InProcessProvider` cannot
  produce — I have marked it as such rather than overstating it.
* **`ScheduleTimer` itself is sound.** `start` is idempotent, `stop` cancels and
  awaits, a `drive` that raises does not kill the loop, and `seconds_until_next`
  handles an empty store and a task with no time. The defect is where the CLI
  constructs it (H5), not the class.
* **`InProcessProvider` isolates the child's session correctly.** A fresh
  `Session.create` per run, `parent_session`/`delegated` on the child header, and
  the parent's log does not grow — all verified through a real child loop.
* **`_validate_structured` handles fences, non-objects, missing required fields
  and `additionalProperties: false`.** I could not make it accept a value it
  should refuse or refuse one it should accept.
* **`ScheduleStore` persistence failures are surfaced.** A raising sink becomes
  `ScheduleError(SCHEDULE_PERSIST_FAILED)` rather than propagating the raw
  exception, and `delete` returns `False` rather than raising for an unknown id.

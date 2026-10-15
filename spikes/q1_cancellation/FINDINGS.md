# Q1 — Cancellation must never abandon an already-started tool body

> Spike owner: Q1 · Date: 2026-10-12
> Contract under test: `docs/contracts/M3-tool-pipeline.md` §6 **M3.8**
> (`取消永不抛弃已启动的 promise`), §4 (concurrency), §11 item 1;
> `docs/runtime-spine.md` §4 **S12** (single cancellation path, typed cause), §5.
> Environment: Python **3.11.14**, pytest 9.1.1, pytest-asyncio 1.4.0.

## Verdict in one paragraph

**M3.8 is achievable on Python 3.11, but not with any single stdlib primitive.**
`asyncio.shield` alone protects the *body* but strands its *result*, and the body
is still destroyed when the loop is torn down. `asyncio.TaskGroup` and
`asyncio.timeout` both actively **hinder** the contract because they cancel the
work they wrap. What is required is a **composite**: (1) start the body as an
explicit `asyncio.Task` and register it *synchronously*; (2) attach a
`done_callback` that captures the outcome without needing an awaiter; (3) drain
every registered task in a `finally`, under `shield`, absorbing repeat cancels
with `uncancel()`; (4) re-raise exactly one `CancelledError` so the abort stays
observable. Steps (3) and (4) are not optional refinements — each is defended by
a test that fails without it.

## Exact commands

```bash
cd /Users/liuminxuan/Desktop/job-prep/02-projects/agent-runtime-min/spikes
../.venv/bin/python -m pytest q1_cancellation -v
```

Must be run from `spikes/` — `spikes/pytest.ini` sets `testpaths` and
`asyncio_mode = auto`, and `spikes/conftest.py` puts `q1_cancellation/` on
`sys.path` so `import harness` resolves. Verified: exit code 0.

Raw evidence transcript (regenerate with the second command):

```bash
../.venv/bin/python q1_cancellation/evidence.py
```

## Files

| File | Role |
|---|---|
| `harness.py` | Naive strategies (to be falsified) + the recommended composite |
| `test_cancellation.py` | 30 pytest tests, one per claim |
| `evidence.py` | Prints the raw observations quoted below |

---

## Q1 — Naive cancellation abandons work (baseline)

Command: `../.venv/bin/python q1_cancellation/evidence.py` (section E1)

```text
========================================================================
E1  naive cancellation: what survives?
========================================================================
  naive_inline         outer=CancelledError       ledger=['body_start']
  naive_create_task    outer=CancelledError       ledger=['body_start']
  naive_shield_only    outer=CancelledError       ledger=['body_start', 'side_effect']
```

(a) **Side effect.** With `await body()` *or* `await asyncio.create_task(body())`,
the side effect after the `await` **never happens** — the ledger stops at
`body_start`. With `shield`, it does happen.

(b) **Result.** In all three cases the caller raises `CancelledError` and no
result object is produced. For `naive_shield_only` the body *does* finish, but
the returned value is unreachable: `naive_shield_only` returned no handle, so the
runtime has nothing to write as a `tool/result`.

**Verdict (rule).** `await asyncio.create_task(...)` is **not** a detach. Awaiting
a task propagates cancellation *into* it. Any dispatch strategy that does not
both shield the body *and* retain a handle fails M3.8.

> **Surprise #1 (highest value).** The intuitive "wrap it in a Task so it becomes
> independent" move is exactly as broken as awaiting inline. Test
> `test_01b_naive_create_task_does_not_detach` pins this: the inner task is
> observed as `cancelled() is True` and the body records `body_received_cancel`.

---

## Q2 — The pattern that satisfies the contract

### 2a. Does `shield` alone suffice? **No.**

E2, raw:

```text
  immediately: inner.done()=False inner.cancelled()=False
  ledger=['body_start', 'side_effect']  -> body finished, but nothing collected the value
```

Precise answers to the questions asked:

| Question | Observed answer |
|---|---|
| Does the shielded inner task keep running when the outer task is cancelled? | **Yes.** `inner.done() is False`, `inner.cancelled() is False` immediately after the outer raises. |
| Does the outer task still see cancellation? | **Yes.** `CancelledError` propagates. `shield` is not a cancellation suppressor. |
| Who awaits the inner task afterwards? | **Nobody.** This is the defect. The shield's outer future dies with the caller. The value is only reachable if *you* kept a reference — the test can, the runtime cannot. |

The decisive counter-example is **loop teardown** (E3):

```text
  shield_only : ['body_start', 'turn_ended']
  composite   : ['body_start', 'side_effect', 'turn_ended']
```

When the turn ends, `asyncio.run` cancels every task that is still pending.
A task nobody awaits is still owned by the runner. **`shield` protects a body
from the caller's cancellation, but not from loop shutdown** — so shield-only
silently loses the side effect at exactly the moment a real turn ends.

> **Surprise #2.** The standard advice "use `asyncio.shield` to avoid abandoning
> work" is *correct about the caller* and *wrong about the result*. Shield solves
> one third of M3.8. Test `test_02b` runs this scenario on its own thread/loop
> because it must observe teardown, which pytest's loop cannot show.

### 2b. The composite that works

`harness.DrainScope` / `dispatch_and_drain`, with four load-bearing parts:

1. **Register synchronously with dispatch.** `dispatch()` calls
   `asyncio.create_task` and appends to `self.started` in the same synchronous
   step. There is no `await` between starting a body and recording it — that gap
   is precisely where M3.8 is violated.
2. **A `done_callback` recorder.** Captures the outcome the instant the body
   finishes, with no awaiter required (see Q5).
3. **A shielded drain that absorbs repeat cancels.** `uncancel()` accounting.
4. **Re-raise exactly one `CancelledError`** (see Q3).

E4, raw — the drain is hit by **three** cancels:

```text
  unshielded drain : ledger=['body_start']
  composite drain  : ledger=['body_start', 'side_effect'] absorbed=2
  recovered result : ToolResult(call_id='c1', tool_name='t', status=<Status.OK: 'ok'>, value='RESULT', error=None, cause=None)
```

Why `shield` alone inside the drain is insufficient: a **second** cancel
(e.g. an impatient second Ctrl-C, or a supervisor's dispose) aborts the drain
itself. Tests `test_02d` (composite absorbs) and `test_02e` (plain `await` in the
`finally` loses the body) are the positive/negative pair.

### 2c. `TaskGroup` and `asyncio.timeout`: both **hinder**

```text
E7
  bare timeout      : 'TIMEOUT'  ledger=['body_start', 'BODY_KILLED']
  timeout + shield  : 'RESULT'  ledger=['body_start', 'side_effect']
```

* **`asyncio.TaskGroup` — do not use it for this seam.** On outer cancellation
  it cancels its own children, so the body is killed (`test_02g`:
  `['body_start', 'body_killed_by_taskgroup']`). Wrapping the body in `shield`
  inside the group keeps the body alive but the result is still stranded, because
  the `async with` block never exits on the cancelled path, so the statement that
  reads `.result()` is unreachable (`test_02g2`). TaskGroup adds a *container*,
  not a *recovery mechanism*.
* **`asyncio.timeout`** cancels the coroutine it wraps — it is a *deadline*, not
  a shield. It is usable only when the body is separately shielded; then it
  becomes the correct timeout shape: deadline fires, body keeps running, drain
  collects it (`test_02i`). **This is an M3 §5 consequence**: a timeout policy
  mounted in the `execute` wrapper must shield, or it will violate M3.8 by
  construction.

> **Surprise #3.** `asyncio.create_task(asyncio.shield(x))` raises
> `TypeError: a coroutine was expected`. `shield` returns a `Future`, and
> `create_task` accepts only coroutines. Any "just wrap it in a TaskGroup and
> shield it" recipe is a runtime error, not merely a design smell. Pinned in
> `test_02g2`.

### 2d. Which mechanism is required?

**A composite. No single primitive suffices.**

| Mechanism | Body survives? | Result recoverable? | Survives teardown? | Verdict |
|---|---|---|---|---|
| `await body()` | ✗ | ✗ | ✗ | Fails M3.8 |
| `await create_task(body())` | ✗ | ✗ | ✗ | Fails M3.8 (not a detach) |
| `await shield(task)` | ✓ | ✗ | ✗ | Insufficient |
| `TaskGroup` | ✗ | ✗ | ✗ | Actively harmful |
| `timeout` (bare) | ✗ | ✗ | ✗ | Actively harmful |
| **register + shield + drain + `uncancel` + re-raise** | ✓ | ✓ | ✓ | **Required** |

---

## Q3 — Cancellation must still be observable

Tests `test_03`, `test_03b`, `test_03c`. Raw:

```text
E4  composite drain  : ledger=['body_start', 'side_effect'] absorbed=2
```

The composite drains the body to completion **and then re-raises**. Asserted:

* `step_task.cancelled() is True` and the raised object is an exact
  `asyncio.CancelledError` — not an `ExceptionGroup`, not a normal return
  (`test_03c`).
* The `uncancel()` discipline is exact: after absorbing a repeat, the enclosing
  scope observes `cancelling() == 1` — the caller's own cancel, and no phantom
  extra (`test_03b`).

**Verdict (rule).** The drain may absorb *repeat* cancels (and must `uncancel()`
each so no phantom pending cancel leaks outward), but must re-raise **exactly
one** `CancelledError` for the caller's original abort. Converting an abort into
a normal return breaks `runtime-spine` S12 and the `turn/end{aborted}` exit.

> **Surprise #4.** `Task.exception()` **raises** `CancelledError` on a cancelled
> task instead of returning it. Code that does
> `if (e := task.exception()) is not None` to detect failure will itself blow up
> on the cancellation path. The correct check is `task.cancelled()` **first** —
> which is what `harness._classify` does.

---

## Q4 — The `BaseException` trap

E5, raw:

```text
  outer RETURNED 'FAKE_SUCCESS'  cancelled()=False  ledger=['body_start', 'swallowed_CancelledError']
```

A tool body with `except BaseException` swallows `CancelledError` and returns a
normal value. The caller is **not** cancelled: `cancelled() is False`. The turn
is silently reported as a success.

**Correct discipline** (`test_04b`):

```python
try:
    ...
except asyncio.CancelledError:
    raise                    # MANDATORY — never convert to a result
except Exception as exc:     # ordinary tool failure -> becomes a result
    return error_result(exc)
```

> **Surprise #5 (subtle, and the reason shield-only is a latent bug).** Under
> `shield` the cancel never enters the body, so a `BaseException`-swallowing tool
> *looks* fine. The trap only fires when the runtime delivers the cancel into the
> body — which a cooperative drain must do to give tools a cleanup chance.
> Test `test_04c` demonstrates both halves. **A runtime that relies on shield
> alone ships this bug undetected until it delivers a real cancel.**

**Verdict (rule).** Plugin tool bodies must catch `Exception`, never
`BaseException`. The registry should treat `except BaseException` in a tool body
as a contract violation; at minimum it must be documented, because the failure is
silent.

---

## Q5 — Recovering the result of a cancelled call

E4 raw recovered object:

```text
ToolResult(call_id='c1', tool_name='t', status=<Status.OK: 'ok'>, value='RESULT', error=None, cause=None)
```

The concrete shape: the runtime never relies on "whoever awaited it". It relies
on the **`done_callback`**, which fires regardless of awaiters. `test_05`
deliberately *skips the drain entirely* (simulating a drain that is itself
killed) and still recovers the value.

Two further cases:

* **Body raised after a side effect** → recorded as `status=ERROR` with the
  message, still loggable, still exactly one result (`test_05c`).
* **Body cooperatively cancels itself** (raises `CancelledError` on its own, with
  no caller cancellation) → recorded as `status=ABORTED`, and the step's own
  cancellation is *not* confused with the body's (`test_05e`). The drain must
  distinguish "the body was cancelled" from "the caller is being cancelled".
* **JSON-serialisable for the log** — the projection is flat
  (`call_id`/`tool_name`/`status`/`value`/`isError`/`cause`) and round-trips
  through `json.dumps`, satisfying S12's "persist only JSON fields"
  (`test_05b`).

> **Surprise #6 (drain masking).** A `finally` that awaits the body *without*
> guarding it lets the body's exception escape — and Python then **discards the
> in-flight `CancelledError`**. `test_05d` shows the caller receiving
> `RuntimeError("tool blew up")` instead of a cancellation: the turn is reported
> as a tool error when it was actually an abort. The composite's drain catches
> `Exception` per-task and breaks, so the abort survives.

---

## Q6 — Multiple in-flight calls

E6, raw — 3 started, 2 never dispatched, step cancelled:

```text
  step: CancelledError propagated
  ledger=['body0_start', 'body1_start', 'body2_start', 'body0_side_effect', 'body1_side_effect', 'body2_side_effect']
    c0 tool0   ok                       value='RESULT0'
    c1 tool1   ok                       value='RESULT1'
    c2 tool2   ok                       value='RESULT2'
    c3 tool3   aborted_before_dispatch  value=None
    c4 tool4   aborted_before_dispatch  value=None
```

* **Every started body completed** — all three `*_side_effect` entries present.
* **Every result is recoverable** — 5 results for 5 declared calls, in model
  order, no dangling `tool_call`.
* **Never-dispatched calls** get a synthetic result
  `status=aborted_before_dispatch`, `error="tool call aborted before dispatch"`,
  `cause=USER_ABORT`, and their bodies never ran (`body3_start`/`body4_start`
  absent). This is M3 §4 / S9.
* The bounded rolling pool of M3 §4 also drains correctly (`test_06c`).

**Verdict (rule).** Cancellation must be applied at the *step* level and drained
as a set. Results are emitted for the full declared call list in model order:
`ok`/`error` for started calls, `aborted_before_dispatch` for the rest.

---

## Are the tests vacuous? (mutation check)

Each mutation was applied to `harness.py` and the suite re-run; every mutant was
killed:

| Mutation | Result |
|---|---|
| `await asyncio.shield(call.task)` → `await call.task` in the drain | **2 failed**, 28 passed |
| remove `uncancel()` absorption | **1 failed**, 29 passed |
| let body exceptions escape the drain | **2 failed**, 28 passed |
| drop the `done_callback` recorder | **11 failed**, 19 passed |
| never emit synthetic `aborted_before_dispatch` results | **2 failed**, 28 passed |
| swallow cancellation (convert to normal return) | **2 failed**, 28 passed |
| hardcode the result cause to `USER_ABORT` | **1 failed**, 29 passed |

Baseline after restore: **30 passed**. The last mutant is worth noting: it
initially **survived** (30 passed), because no test used a non-default cause on a
*started* call. `test_05e` was strengthened to assert `cause is Cause.SHUTDOWN`,
and the mutant now dies. Without the mutation pass this coverage gap would have
shipped.

---

## Where the requirements document is wrong, incomplete, or optimistic

These are the highest-value outputs of this spike.

1. **§9 / §11 item 1 — "语义等价" is optimistic and, as written, false.**
   The doc proposes `asyncio` + a custom cause as equivalent to
   `AbortController`. It is **not** equivalent:
   `AbortSignal` is a *passive flag the callee observes*; `CancelledError` is an
   *exception injected into the callee at an await point*. A passive flag cannot
   destroy work that ignores it; an injected exception can, and does. The
   mitigation is not a cause object — it is a **task-ownership pattern**
   (shield + register + drain). **M3.8 should be restated in task terms**, and
   §11 item 1 should require the composite, not just "a test".

2. **M3.5 is not implementable as literally written.** It says
   "只有 `execute` 包装器可换 signal,且主体前重新融合调用方信号". Python
   asyncio has **no signal object to swap**. The real hazard M3.5 guards against
   is "a wrapper removes cancellation" — which in Python means *not registering
   the task* or *catching `CancelledError`*. M3.5 needs restating in those terms;
   otherwise implementers will build a `CancellationSignal` abstraction that adds
   a layer without adding the guarantee.

3. **§11 item 1's validation method is insufficient.** "写一条测试——取消一个已
   启动的工具主体,断言它仍走完并落结果" would pass on a `shield`-only
   implementation *if* the test never ends the loop and never cancels twice.
   The single test is blind to at least four distinct failure modes found here:
   loop teardown (E3), repeat cancel (E4), the `BaseException` trap (E5), and
   drain masking (E6/test_05d). **At least four tests are required**, and the
   teardown one must own its event loop.

4. **M3 §7 contradicts runtime-spine §5 on what a cancel does to a step.**
   M3 §7 rows "主体启动后取消" and "主体前取消" both say `中断 step? = 否`,
   i.e. an aborted tool call does **not** interrupt the step. `runtime-spine` §5
   says 取消 → `turn 以 aborted 收尾`. Both are defensible but they describe
   different things, and the word 取消 is used for both without distinction:
   *this call was cancelled* (a result, step continues — e.g. `pre-execute`
   returning `cancel`) versus *the whole turn was cancelled* (the step dies).
   **This ambiguity should be resolved explicitly in the contract**, because the
   two paths need opposite code: the first must return a result, the second must
   drain and re-raise.

5. **M3.8's wording "已启动的 promise" is JS vocabulary with no Python referent.**
   Restate as: *every tool body that has been started as a task must be awaited
   to completion before the step exits*. "Promise" invites the `create_task`
   mistake documented in Surprise #1.

6. **§5 is right that timeout is an external policy, but incomplete about its
   shape.** A timeout policy mounted in `execute` **must shield the body**, or it
   violates M3.8 by construction (E7: bare `asyncio.timeout` kills the body).
   The contract should state this as a constraint on the policy, not leave it to
   the implementer.

---

## Residual uncertainty

1. **GC of unreferenced tasks.** The stdlib `shield` docstring warns that the
   loop keeps only weak references to tasks and an unreferenced task may be
   collected mid-flight. I probed `shield` on a **bare coroutine** and the body
   did complete, and an `all_tasks()`-based probe showed the task still alive
   after `gc.collect()`. I could **not** construct a deterministic failure, so I
   did not claim one. The recommended composite keeps an explicit reference
   regardless, so this is not on the critical path — but the exact conditions
   under which an unreferenced shielded task is collected remain **unverified**.
2. **Not tested against a real cancellation source.** All cancels here are
   `Task.cancel()`. Cancellation originating from `loop.call_soon`,
   `asyncio.Runner` shutdown, or a signal handler (`KeyboardInterrupt` →
   `add_signal_handler`) may arrive with different `cancelling()` counts and was
   not exercised.
3. **`uncancel()` semantics beyond 3.11.** The `cancelling()`/`uncancel()`
   accounting is a 3.11+ API and its interaction with `TaskGroup` is documented
   as subtle. This spike pins the behaviour on **3.11.14 only**; it should be
   re-run if the project moves to 3.12+.
4. **Cooperative self-cancel, partial coverage.** A body that *raises*
   `CancelledError` is covered (`test_05e`) and records as `ABORTED`. A body that
   calls `task.cancel()` **on itself** mid-execution while holding resources was
   not tested, and the ordering of its own cancellation against an arriving
   caller cancellation is unverified.
5. **Throughput.** The composite adds one `Task` per call and one
   `asyncio.wait` per completion. No performance measurement was done; the
   bounded-pool default of 10 (M3 §4) should be fine, but this is unmeasured.

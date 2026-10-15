"""Q1: empirical proof for M3.8 -- cancellation must not abandon a started tool body.

Run from the ``spikes/`` directory (see ``spikes/pytest.ini``)::

    cd spikes && ../.venv/bin/python -m pytest q1_cancellation -v

Each test asserts *observed* behaviour. Where a mechanism is claimed to be
insufficient, the test asserts the failure, so the positive result elsewhere is
not vacuous.
"""

from __future__ import annotations

import asyncio
import gc
import sys

import pytest

from harness import (
    Cause,
    DrainScope,
    Ledger,
    Status,
    ToolRequest,
    ToolResult,
    dispatch_and_drain,
    naive_create_task,
    naive_inline,
    naive_shield_only,
)

# All tests are async; `asyncio_mode = auto` in spikes/pytest.ini handles the
# marker, but be explicit so the file is also runnable under `-p no:asyncio`.


async def _let_run(seconds: float = 0.25) -> None:
    """Give detached tasks a chance to finish, as a real loop would."""
    await asyncio.sleep(seconds)


# ===========================================================================
# Q1 -- NAIVE CANCELLATION ABANDONS WORK (the baseline)
# ===========================================================================


@pytest.mark.parametrize(
    "strategy", [naive_inline, naive_create_task, naive_shield_only]
)
async def test_01_naive_cancellation_abandons_the_side_effect(strategy):
    """BASELINE. Cancel the caller mid-flight; observe what survives.

    ``naive_inline`` and ``naive_create_task`` lose the side effect outright.
    ``naive_shield_only`` keeps the side effect but the caller can no longer
    reach the result -- which is the *second* half of M3.8 ("still record its
    result") failing even when the first half holds.
    """
    ledger = Ledger()
    started = asyncio.Event()

    async def body():
        ledger.note("body_start")
        started.set()
        await asyncio.sleep(0.05)
        ledger.note("side_effect")  # AFTER the await -- the thing at risk
        return "RESULT"

    caller = asyncio.create_task(strategy(body))
    await started.wait()
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller

    # The caller observed cancellation in every case (this is correct, see Q3).
    assert caller.cancelled() is True

    await _let_run()

    if strategy is naive_shield_only:
        # Body survives, but the result is unreachable: `naive_shield_only`
        # returned no handle, so the runtime has nothing to log.
        assert ledger.events == ["body_start", "side_effect"]
        assert "RESULT" not in ledger.events
    else:
        # Body was killed at its await point: side effect never happened.
        assert ledger.events == ["body_start"], (
            f"{strategy.__name__} unexpectedly completed the body"
        )
        assert ledger.count("side_effect") == 0


async def test_01b_naive_create_task_does_not_detach():
    """Sharpen the surprise: wrapping in a Task is NOT detachment.

    This is the single most common wrong mental model when porting from
    JavaScript ``AbortSignal``. Awaiting a task propagates the cancel into it.
    """
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            ledger.note("body_received_cancel")  # <- delivered INTO the body
            raise
        ledger.note("side_effect")
        return "RESULT"

    inner_ref: list[asyncio.Task] = []

    async def caller():
        task = asyncio.create_task(body())
        inner_ref.append(task)
        return await task  # plain await, no shield

    caller_task = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    caller_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller_task

    assert ledger.events == ["body_start", "body_received_cancel"]
    assert inner_ref[0].cancelled() is True, "inner task must have been cancelled too"


# ===========================================================================
# Q2 -- A PATTERN THAT SATISFIES THE CONTRACT
# ===========================================================================


async def test_02_shield_alone_protects_the_body_but_strands_the_result():
    """``asyncio.shield`` alone: what exactly does it buy?

    Precise answer, observed:
      * the shielded task KEEPS RUNNING and completes (side effect happens);
      * the outer/caller task still raises ``CancelledError``;
      * NOBODY awaits the inner task afterwards -- the shield's outer future
        already died with the caller, so the result is dropped.
    """
    ledger = Ledger()
    inner: list[asyncio.Task] = []

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.05)
        ledger.note("side_effect")
        return "RESULT"

    async def caller():
        task = asyncio.create_task(body())
        inner.append(task)
        return await asyncio.shield(task)

    caller_task = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    caller_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller_task

    # Immediately after cancellation the body is still in flight and NOT cancelled.
    assert inner[0].done() is False
    assert inner[0].cancelled() is False

    await _let_run()
    assert ledger.count("side_effect") == 1  # body did finish

    # ...but nothing collected it. `caller` raised, so no result object exists.
    assert inner[0].done() is True
    assert inner[0].result() == "RESULT"  # only reachable because the TEST kept a ref


async def test_02b_shield_alone_loses_the_result_at_loop_teardown():
    """The decisive counter-example to "shield alone is enough".

    A real turn ends and the loop is torn down. ``asyncio.run`` cancels every
    task that is still pending, including the shielded body -- because a task
    nobody is awaiting is still a task the runner owns. The side effect M3.8
    exists to protect is lost even though ``shield`` was used correctly.

    This test must run OUTSIDE the pytest event loop: it builds its own loop via
    ``asyncio.run`` and observes the teardown. It is therefore a sync test.
    """
    import threading

    def scenario(kind: str) -> list[str]:
        events: list[str] = []

        async def body():
            events.append("body_start")
            await asyncio.sleep(0.05)
            events.append("side_effect")
            return "RESULT"

        async def shield_only():
            t = asyncio.create_task(body())
            return await asyncio.shield(t)

        async def composite():
            t = asyncio.create_task(body())
            try:
                return await asyncio.shield(t)
            finally:
                await asyncio.shield(t)  # drain while shielding

        async def main():
            caller = asyncio.create_task(
                shield_only() if kind == "shield" else composite()
            )
            await asyncio.sleep(0.01)  # body is in flight
            caller.cancel()
            try:
                await caller
            except asyncio.CancelledError:
                pass
            events.append("turn_ended")
            # main() returns here -> asyncio.run cancels all remaining tasks.

        result: dict[str, object] = {}

        def run() -> None:
            try:
                asyncio.run(main())
                result["ok"] = True
            except BaseException as exc:  # pragma: no cover
                result["err"] = exc

        th = threading.Thread(target=run)
        th.start()
        th.join(timeout=10)
        assert not th.is_alive(), "scenario hung"
        if "err" in result:
            raise result["err"]  # type: ignore[misc]
        return events

    shield_events = scenario("shield")
    composite_events = scenario("composite")

    assert shield_events == ["body_start", "turn_ended"], (
        "shield alone must be shown to LOSE the side effect at loop teardown; "
        f"observed {shield_events!r}"
    )
    assert composite_events == ["body_start", "side_effect", "turn_ended"], (
        "the composite must preserve the side effect across loop teardown; "
        f"observed {composite_events!r}"
    )



async def test_02c_composite_body_runs_to_completion_and_result_is_recoverable():
    """The recommended pattern, positive case."""
    ledger = Ledger()
    recovered: list[ToolResult] = []

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.05)
        ledger.note("side_effect")
        return "RESULT"

    scope = DrainScope(cause=Cause.USER_ABORT)
    request = ToolRequest("c1", "demo_tool", body)
    scope.declare(request)

    async def step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        except asyncio.CancelledError:
            raise
        finally:
            await scope.drain(already_pending=True)

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.01)
    step_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step_task

    # M3.8 first half: the body ran to completion.
    assert ledger.events == ["body_start", "side_effect"]

    # M3.8 second half: a result object exists and is loggable.
    recovered = scope.results()
    assert len(recovered) == 1
    assert recovered[0].status is Status.OK
    assert recovered[0].value == "RESULT"
    assert recovered[0].call_id == "c1"
    assert recovered[0].is_error is False


async def test_02d_composite_survives_a_second_cancel_during_the_drain():
    """Why the composite needs ``uncancel()`` and not just ``shield``.

    A second cancel arriving *while the drain is running* aborts the drain if it
    is only shielded. The composite absorbs the repeat and still finishes.
    """
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.15)
        ledger.note("side_effect")
        return "RESULT"

    scope = DrainScope()
    request = ToolRequest("c1", "demo_tool", body)
    scope.declare(request)

    async def step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.01)
    step_task.cancel()  # cancel #1: the real one
    await asyncio.sleep(0.02)
    step_task.cancel()  # cancel #2: lands mid-drain (e.g. a second UI click)
    await asyncio.sleep(0.02)
    step_task.cancel()  # cancel #3

    with pytest.raises(asyncio.CancelledError):
        await step_task

    assert ledger.events == ["body_start", "side_effect"], (
        "repeat cancels must not abort the drain (M3.8)"
    )
    assert scope.drain_absorbed_cancels == 2
    assert scope.results()[0].value == "RESULT"


async def test_02e_unshielded_drain_is_fragile_under_repeat_cancel():
    """Negative control for 02d: a plain `await` in the drain loses the body."""
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.15)
        ledger.note("side_effect")
        return "RESULT"

    async def step():
        task = asyncio.create_task(body())
        try:
            return await asyncio.shield(task)
        finally:
            await task  # PLAIN await: no shield, no uncancel

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.01)
    step_task.cancel()
    await asyncio.sleep(0.02)
    step_task.cancel()  # second cancel aborts the unshielded drain

    with pytest.raises(asyncio.CancelledError):
        await step_task

    await _let_run(0.3)
    # The drain was aborted; the body was then cancelled by loop teardown or
    # simply left behind. Either way, no completion was observed in-band.
    assert "side_effect" not in ledger.events, (
        "unshielded drain was expected to be abortable by the second cancel"
    )


async def test_02g_taskgroup_alone_kills_the_body_it_manages():
    """``asyncio.TaskGroup`` HINDERS M3.8: on outer cancel it cancels its children.

    Note this is the *only* behaviour available: TaskGroup owns its children and
    cancels them when the enclosing task is cancelled, so it cannot be used as
    the tool-body container.
    """
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            ledger.note("body_killed_by_taskgroup")
            raise
        ledger.note("side_effect")
        return "RESULT"

    async def caller():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(body())
        return "GROUP_DONE"

    task = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _let_run()
    assert ledger.events == ["body_start", "body_killed_by_taskgroup"], (
        "TaskGroup cancels its children on outer cancellation"
    )
    assert ledger.count("side_effect") == 0


async def test_02g2_taskgroup_shielded_body_survives_but_is_uncollectable():
    """TaskGroup + shield keeps the body alive but still cannot deliver a result.

    Two separate defects, both observed:

    1. ``asyncio.create_task(asyncio.shield(x))`` raises ``TypeError`` --
       ``shield`` returns a ``Future``, and ``create_task`` only accepts
       coroutines. The wrapper must be a real coroutine function.
    2. Even with a correct wrapper, the ``async with TaskGroup`` block never
       exits on the cancelled path, so the statement that reads
       ``inner.result()`` is never reached. The result is stranded exactly as
       in ``naive_shield_only`` -- TaskGroup adds a container, not a recovery
       mechanism.
    """
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.08)
        ledger.note("side_effect")
        return "RESULT"

    # Defect 1: shield() returns a Future; create_task rejects it.
    inner = asyncio.create_task(body())
    with pytest.raises(TypeError, match="a coroutine was expected"):
        asyncio.create_task(asyncio.shield(inner))  # type: ignore[arg-type]
    inner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inner

    # Defect 2: a correct wrapper still strands the result.
    ledger.events.clear()
    collected: dict[str, object] = {}

    async def caller():
        inner2 = asyncio.create_task(body())

        async def shielded():
            return await asyncio.shield(inner2)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(shielded())
        # Unreachable on the cancelled path:
        collected["result"] = inner2.result()
        return collected["result"]

    task = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _let_run(0.3)
    assert ledger.count("side_effect") == 1, "the body did survive (shield worked)"
    assert collected == {}, "but the result was never collected"



async def test_02h_asyncio_timeout_kills_the_body_it_wraps():
    """``asyncio.timeout`` cancels the wrapped coroutine -- it is NOT a shield."""
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            ledger.note("body_killed_by_timeout")
            raise
        ledger.note("side_effect")
        return "RESULT"

    async def caller():
        try:
            async with asyncio.timeout(0.05):
                return await body()
        except TimeoutError:
            return "TIMEOUT"

    assert await caller() == "TIMEOUT"
    assert ledger.events == ["body_start", "body_killed_by_timeout"]
    assert ledger.count("side_effect") == 0


async def test_02i_asyncio_timeout_plus_shield_is_the_correct_timeout_shape():
    """Positive counterpart: timeout is usable *if* the body is shielded."""
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.15)
        ledger.note("side_effect")
        return "RESULT"

    async def caller():
        inner = asyncio.create_task(body())
        try:
            async with asyncio.timeout(0.05):
                return await asyncio.shield(inner)
        except TimeoutError:
            # Deadline fired, but the body was already started: drain it.
            return await asyncio.shield(inner)

    assert await caller() == "RESULT"
    assert ledger.events == ["body_start", "side_effect"]


# ===========================================================================
# Q3 -- CANCELLATION MUST STILL BE OBSERVABLE
# ===========================================================================


async def test_03_composite_still_propagates_cancelled_error():
    """The pattern must not swallow cancellation (S12 / M3 §7 'ABORTED')."""
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.03)
        ledger.note("side_effect")
        return "RESULT"

    scope = DrainScope(cause=Cause.USER_ABORT)
    request = ToolRequest("c1", "t", body)
    scope.declare(request)

    async def step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.005)
    step_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await step_task

    assert step_task.cancelled() is True
    assert ledger.count("side_effect") == 1  # ...and the work still completed


async def test_03b_cancelling_counter_is_not_left_inflated():
    """``uncancel()`` discipline: absorbing a repeat must not leak a phantom cancel.

    If the drain calls ``uncancel()`` without re-raising the matching
    ``CancelledError``, enclosing scopes (``asyncio.timeout``, TaskGroup) will
    mis-report a pending cancellation.
    """
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.15)
        ledger.note("side_effect")
        return "RESULT"

    observed: dict[str, int] = {}

    async def step():
        scope = DrainScope()
        call = scope.dispatch(ToolRequest("c1", "t", body))
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    async def enclosing():
        try:
            await step()
        except asyncio.CancelledError:
            cur = asyncio.current_task()
            assert cur is not None
            observed["cancelling_after"] = cur.cancelling()
            raise

    task = asyncio.create_task(enclosing())
    await asyncio.sleep(0.005)
    task.cancel()
    await asyncio.sleep(0.02)
    task.cancel()  # repeat during drain
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ledger.count("side_effect") == 1
    # Exactly one outstanding cancel survives to the enclosing scope: the
    # caller's own. The absorbed repeat was fully accounted for by uncancel().
    assert observed["cancelling_after"] == 1


async def test_03c_cancelled_error_is_not_wrapped_or_synthesised():
    """Cancellation propagates as a plain ``CancelledError``, and ``cancelled()`` is True.

    Observed detail: ``Task.exception()`` *raises* ``CancelledError`` on a
    cancelled task rather than returning it, so the assertion must go through
    ``Task.cancelled()`` / the caught exception, not through ``exception()``.
    """
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.03)
        ledger.note("side_effect")
        return "RESULT"

    scope = DrainScope()
    request = ToolRequest("c1", "t", body)
    scope.declare(request)

    async def step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.005)
    step_task.cancel()

    caught: list[BaseException] = []
    with pytest.raises(asyncio.CancelledError) as excinfo:
        await step_task
    caught.append(excinfo.value)

    assert step_task.cancelled() is True
    assert type(caught[0]) is asyncio.CancelledError, (
        f"expected an exact CancelledError, got {type(caught[0]).__name__}"
    )
    # No ExceptionGroup wrapper, no conversion into a normal return.
    assert not isinstance(caught[0], BaseExceptionGroup)
    # Reading .exception() on a cancelled task re-raises -- document it here.
    with pytest.raises(asyncio.CancelledError):
        step_task.exception()
    assert ledger.count("side_effect") == 1




async def test_04_baseexception_catch_swallows_cancellation():
    """HAZARD: ``except BaseException`` in a plugin tool eats the cancel."""
    ledger = Ledger()

    async def bad_body():
        ledger.note("body_start")
        try:
            await asyncio.sleep(0.05)
        except BaseException as exc:  # noqa: BLE001 - the hazard under test
            ledger.note(f"swallowed_{type(exc).__name__}")
            return "FAKE_SUCCESS"  # turns an abort into a bogus OK result
        ledger.note("side_effect")
        return "REAL_RESULT"

    async def caller():
        return await bad_body()  # cancellation is delivered inline

    task = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    task.cancel()

    # The caller does NOT see CancelledError: the abort vanished.
    result = await task
    assert result == "FAKE_SUCCESS"
    assert task.cancelled() is False
    assert ledger.events == ["body_start", "swallowed_CancelledError"]


async def test_04b_correct_reraise_discipline():
    """The fix: catch ``Exception`` for real errors, re-raise ``CancelledError``."""
    ledger = Ledger()

    async def good_body():
        ledger.note("body_start")
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            ledger.note("observed_cancel")
            raise  # MANDATORY
        except Exception as exc:  # noqa: BLE001 - ordinary tool failure
            ledger.note(f"handled_{type(exc).__name__}")
            return "ERROR_RESULT"
        ledger.note("side_effect")
        return "REAL_RESULT"

    async def caller():
        return await good_body()

    task = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert ledger.events == ["body_start", "observed_cancel"]
    assert task.cancelled() is True


async def test_04c_trap_is_invisible_under_shield_but_real_under_inner_cancel():
    """Subtle: ``shield`` hides the trap from the *outer* task but not from the body.

    Under a plain shield the cancel never enters the body, so a
    ``BaseException``-swallowing tool looks fine. The moment the runtime
    delivers the cancel into the body (which a cooperative drain must, to give
    the tool a chance to clean up), the trap fires. A runtime that relies on
    shield alone therefore ships a latent bug.
    """
    ledger = Ledger()

    async def bad_body():
        ledger.note("body_start")
        try:
            await asyncio.sleep(0.05)
        except BaseException as exc:  # noqa: BLE001
            ledger.note(f"swallowed_{type(exc).__name__}")
            return "FAKE_SUCCESS"
        ledger.note("side_effect")
        return "REAL_RESULT"

    async def caller():
        task = asyncio.create_task(bad_body())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()  # deliver cancellation INTO the body
            try:
                outcome = await asyncio.shield(task)
            except asyncio.CancelledError:
                outcome = "<propagated correctly>"
            ledger.note(f"outcome={outcome!r}")
            raise

    task = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "swallowed_CancelledError" in ledger.events
    assert "outcome='FAKE_SUCCESS'" in ledger.events, (
        "the tool converted an abort into a fabricated success result"
    )


# ===========================================================================
# Q5 -- RECOVERING THE RESULT OF A CANCELLED CALL
# ===========================================================================


async def test_05_done_callback_recovers_result_even_if_the_drain_dies():
    """The strongest recovery guarantee: result captured without any awaiter."""
    ledger = Ledger()
    scope = DrainScope()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.15)
        ledger.note("side_effect")
        return {"payload": "RESULT"}

    request = ToolRequest("c1", "demo_tool", body)
    scope.declare(request)

    async def step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        finally:
            # Deliberately NO drain: simulate a drain that itself gets killed.
            ledger.note("drain_skipped")

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.01)
    step_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step_task

    # Nothing is awaiting the body, yet the callback will still fire.
    assert scope.started[0].result is None
    await _let_run(0.3)

    recovered = scope.results()
    assert len(recovered) == 1
    assert recovered[0].status is Status.OK
    assert recovered[0].value == {"payload": "RESULT"}
    assert recovered[0].call_id == "c1"


async def test_05b_recovery_is_json_serialisable_for_the_log():
    """The recovered result must be writable as a ``tool/result`` event (S12)."""
    import json

    ledger = Ledger()
    scope = DrainScope(cause=Cause.USER_ABORT)

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.03)
        ledger.note("side_effect")
        return "RESULT"

    request = ToolRequest("c1", "demo_tool", body)
    scope.declare(request)

    async def step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.005)
    step_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step_task

    result = scope.results()[0]
    payload = {
        "call_id": result.call_id,
        "tool_name": result.tool_name,
        "status": result.status.value,
        "value": result.value,
        "isError": result.is_error,
        "cause": result.cause.value if result.cause else None,
    }
    encoded = json.dumps(payload, sort_keys=True)
    assert json.loads(encoded) == payload
    assert '"status": "ok"' in encoded


async def test_05c_failed_body_is_recorded_as_error_not_lost():
    """A body that raises after a side effect still yields a loggable result."""
    ledger = Ledger()
    scope = DrainScope()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.03)
        ledger.note("side_effect")
        raise RuntimeError("disk full")

    request = ToolRequest("c1", "demo_tool", body)
    scope.declare(request)

    async def step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.005)
    step_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step_task

    assert ledger.count("side_effect") == 1
    result = scope.results()[0]
    assert result.status is Status.ERROR
    assert "disk full" in (result.error or "")
    assert result.is_error is True


async def test_05d_drain_must_not_mask_cancellation_with_a_tool_error():
    """TRAP: a body that raises during the drain replaces the CancelledError.

    A ``finally`` block that awaits the body without guarding it will let the
    body's exception escape, and Python then *discards* the in-flight
    CancelledError. The turn is reported as a tool error instead of an abort.
    """
    ledger = Ledger()

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.03)
        raise RuntimeError("tool blew up after side effect")

    async def unguarded_step():
        task = asyncio.create_task(body())
        try:
            return await asyncio.shield(task)
        finally:
            await asyncio.shield(task)  # unguarded: RuntimeError escapes here

    task = asyncio.create_task(unguarded_step())
    await asyncio.sleep(0.005)
    task.cancel()

    with pytest.raises(RuntimeError, match="tool blew up"):
        await task  # <-- cancellation was REPLACED by the tool's error

    # Contrast: the composite records the error and still reports the abort.
    scope = DrainScope()
    request = ToolRequest("c1", "t", body)
    scope.declare(request)

    async def guarded_step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    task2 = asyncio.create_task(guarded_step())
    await asyncio.sleep(0.005)
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task2

    assert scope.results()[0].status is Status.ERROR
    assert "tool blew up" in (scope.results()[0].error or "")


# ===========================================================================
# Q6 -- MULTIPLE IN-FLIGHT CALLS
# ===========================================================================


async def test_05e_cooperative_self_cancel_is_recorded_as_aborted():
    """A body that gives up by raising ``CancelledError`` still yields a result.

    The drain must distinguish "the *body* was cancelled" from "the *caller* is
    being cancelled". If it does not, a cooperative tool turns into a caller
    cancellation and the turn is misreported.
    """
    ledger = Ledger()
    scope = DrainScope(cause=Cause.SHUTDOWN)  # non-default: proves the cause is threaded

    async def body():
        ledger.note("body_start")
        await asyncio.sleep(0.02)
        ledger.note("side_effect")
        raise asyncio.CancelledError("tool gave up")

    request = ToolRequest("c1", "demo_tool", body)
    scope.declare(request)

    async def step():
        call = scope.dispatch(request)
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    # NOTE: no caller cancellation here -- the body cancels itself.
    with pytest.raises(asyncio.CancelledError):
        await step()

    assert ledger.events == ["body_start", "side_effect"]
    result = scope.results()[0]
    assert result.status is Status.ABORTED
    assert result.error == "tool call aborted"
    assert result.is_error is True
    # The scope's typed cause reaches an ABORTED *started* call too (S12).
    assert result.cause is Cause.SHUTDOWN



async def test_06a_typed_cause_is_threaded_into_every_result():
    """S12: the cancellation cause is typed and attached to the right results.

    A disposed turn and a user abort must be distinguishable in the log. The
    scope's cause must reach the synthetic aborted result; a *successful* result
    carries no cancellation cause at all (there was no cancellation).
    """
    ledger = Ledger()

    def make_body(i: int, delay: float):
        async def body():
            ledger.note(f"body{i}_start")
            await asyncio.sleep(delay)
            ledger.note(f"body{i}_side_effect")
            return f"RESULT{i}"

        return body

    holder: dict[str, DrainScope] = {}

    async def step():
        scope = DrainScope(cause=Cause.DISPOSED)
        holder["scope"] = scope
        started = [ToolRequest("c0", "tool0", make_body(0, 0.05))]
        never = [ToolRequest("c1", "tool1", make_body(1, 0.0))]
        for r in started + never:
            scope.declare(r)
        for r in started:
            scope.dispatch(r)
        try:
            await asyncio.gather(*(asyncio.shield(c.task) for c in scope.started))
        finally:
            await scope.drain(already_pending=True)

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.01)
    step_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step_task

    results = {r.call_id: r for r in holder["scope"].results()}

    # The started call finished normally: no cancellation cause on a success.
    assert results["c0"].status is Status.OK
    assert results["c0"].value == "RESULT0"
    assert results["c0"].cause is None

    # The never-dispatched call carries the typed cause.
    assert results["c1"].status is Status.ABORTED_BEFORE_DISPATCH
    assert results["c1"].cause is Cause.DISPOSED
    assert results["c1"].cause.value == "disposed"  # JSON-serialisable (S12)
    assert ledger.count("body1_start") == 0


async def test_06b_results_include_synthetic_aborted_before_dispatch():
    """M3 §4 / S9: never-dispatched calls get a synthetic result, in model order."""
    ledger = Ledger()

    def make_body(i: int, delay: float):
        async def body():
            ledger.note(f"body{i}_start")
            await asyncio.sleep(delay)
            ledger.note(f"body{i}_side_effect")
            return f"RESULT{i}"

        return body

    holder: dict[str, DrainScope] = {}

    async def step():
        scope = DrainScope(cause=Cause.USER_ABORT)
        holder["scope"] = scope
        started = [
            ToolRequest(f"c{i}", f"tool{i}", make_body(i, 0.05 + i * 0.02))
            for i in range(3)
        ]
        never = [
            ToolRequest("c3", "tool3", make_body(3, 0.0)),
            ToolRequest("c4", "tool4", make_body(4, 0.0)),
        ]
        for r in started + never:
            scope.declare(r)
        for r in started:
            scope.dispatch(r)
        try:
            await asyncio.gather(*(asyncio.shield(c.task) for c in scope.started))
        finally:
            await scope.drain(already_pending=True)

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.01)
    step_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step_task

    results = holder["scope"].results()
    assert len(results) == 5, "one result per declared call -- no dangling tool_call"

    for i in range(3):
        assert results[i].status is Status.OK
        assert results[i].value == f"RESULT{i}"

    for i in (3, 4):
        assert results[i].status is Status.ABORTED_BEFORE_DISPATCH
        assert results[i].error == "tool call aborted before dispatch"
        assert results[i].cause is Cause.USER_ABORT
        assert ledger.count(f"body{i}_start") == 0


async def test_06c_bounded_parallel_pool_still_drains():
    """The bounded rolling pool of M3 §4 must drain too."""
    ledger = Ledger()

    def make_body(i: int):
        async def body():
            ledger.note(f"body{i}_start")
            await asyncio.sleep(0.03)
            ledger.note(f"body{i}_side_effect")
            return f"RESULT{i}"

        return body

    requests = [ToolRequest(f"c{i}", f"tool{i}", make_body(i)) for i in range(6)]

    async def step():
        try:
            return await dispatch_and_drain(requests, max_parallel=2)
        except asyncio.CancelledError:
            raise

    holder: dict[str, list[ToolResult]] = {}
    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.02)  # some dispatched, some still queued
    step_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await step_task

    # Every body that started must have finished.
    started = [i for i in range(6) if ledger.count(f"body{i}_start") > 0]
    assert started, "expected at least one body to have started"
    for i in started:
        assert ledger.count(f"body{i}_side_effect") == 1, f"body{i} abandoned"


# ===========================================================================
# Q6+ -- the composite as a whole
# ===========================================================================


async def test_07_dispatch_and_drain_normal_path_is_unaffected():
    """Sanity: with no cancellation the composite behaves like a plain gather."""
    ledger = Ledger()

    def make_body(i: int):
        async def body():
            ledger.note(f"body{i}_start")
            await asyncio.sleep(0.01)
            ledger.note(f"body{i}_side_effect")
            return f"RESULT{i}"

        return body

    requests = [ToolRequest(f"c{i}", f"tool{i}", make_body(i)) for i in range(4)]
    results, scope = await dispatch_and_drain(requests)

    assert [r.value for r in results] == [f"RESULT{i}" for i in range(4)]
    assert all(r.status is Status.OK for r in results)
    assert scope.drain_absorbed_cancels == 0


async def test_07b_dispatch_and_drain_reraises_cancellation_after_draining():
    """End-to-end: the public entry point honours both halves of M3.8 + Q3."""
    ledger = Ledger()
    holder: dict[str, object] = {}

    def make_body(i: int):
        async def body():
            ledger.note(f"body{i}_start")
            await asyncio.sleep(0.05 + i * 0.02)
            ledger.note(f"body{i}_side_effect")
            return f"RESULT{i}"

        return body

    requests = [ToolRequest(f"c{i}", f"tool{i}", make_body(i)) for i in range(3)]

    async def step():
        try:
            return await dispatch_and_drain(requests)
        except asyncio.CancelledError:
            holder["cancelled"] = True
            raise

    step_task = asyncio.create_task(step())
    await asyncio.sleep(0.01)
    step_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await step_task

    assert holder.get("cancelled") is True
    for i in range(3):
        assert ledger.count(f"body{i}_side_effect") == 1, f"body{i} was abandoned"


async def test_07c_dispatch_and_drain_returns_results_on_normal_path():
    """Explicitly verify the normal path returns a loggable, ordered result list."""
    ledger = Ledger()

    def make_body(i: int):
        async def body():
            await asyncio.sleep(0.01)
            ledger.note(f"body{i}_side_effect")
            return i * 10

        return body

    requests = [ToolRequest(f"c{i}", f"tool{i}", make_body(i)) for i in range(3)]
    results, _scope = await dispatch_and_drain(requests)

    assert [r.call_id for r in results] == ["c0", "c1", "c2"]  # model order
    assert [r.value for r in results] == [0, 10, 20]
    assert ledger.count("body0_side_effect") == 1

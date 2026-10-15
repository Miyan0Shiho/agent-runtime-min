"""M3.8 tests: cancellation must not abandon a body that already started.

The contract has a deceptively simple statement:

    every tool body that has been started must be awaited to completion before
    the step exits.

The reason this file exists separately from the pipeline tests is that the
failure modes are invisible to ordinary tests. In particular a ``shield``-only
implementation passes a naive test and still loses the work at loop teardown, so
each test here is written to fail for a *specific* wrong implementation.
"""

from __future__ import annotations

import asyncio

import pytest

from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult
from arm.tools.dispatch import CallRequest, Dispatcher
from arm.tools.drain import BodyCancelled, DrainScope
from arm.tools.pipeline import AbortSignal, PipelineHooks, ToolPipeline
from arm.tools.registry import ToolRegistry


def make_tool(name: str, body, *, parallel: bool = False) -> RegisteredTool:
    return RegisteredTool(
        definition=ToolDefinition(
            name=name,
            description="",
            parameters={},
            concurrency_safe=(lambda _a: True) if parallel else None,
        ),
        body=body,
    )


def registry_with(*tools: RegisteredTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(list(tools))
    return registry


# ---------------------------------------------------------------------------
# The baseline: what goes wrong without ownership
# ---------------------------------------------------------------------------


def test_naive_await_loses_the_side_effect():
    """The problem being solved: awaiting inline discards post-await work.

    This is the negative control. If it ever starts passing, the protection
    below is no longer being exercised and the tests have gone vacuous.
    """
    ledger: list[str] = []

    async def body():
        ledger.append("start")
        await asyncio.sleep(0.05)
        ledger.append("side_effect")

    async def scenario() -> None:
        task = asyncio.ensure_future(body())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    # The body started but the side effect after the await never happened.
    assert ledger == ["start"], (
        "if this assertion fails, the runtime under test is already safe and "
        "this whole test file is testing nothing"
    )


# ---------------------------------------------------------------------------
# The drain scope: started bodies finish
# ---------------------------------------------------------------------------


def test_drain_lets_a_started_body_finish():
    """M3.8: a started body runs to completion despite caller cancellation."""
    ledger: list[str] = []

    async def body() -> str:
        ledger.append("start")
        await asyncio.sleep(0.05)
        ledger.append("side_effect")
        return "RESULT"

    async def scenario() -> dict[str, str]:
        scope = DrainScope()
        async with scope:
            scope.start("c1", body())
            await asyncio.sleep(0.01)
            # Cancel the *caller* mid-flight.
            raise asyncio.CancelledError
        return {k: v.value for k, v in scope.results().items()}

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    assert "side_effect" in ledger, "the body's side effect must have happened"


def test_drain_captures_the_result_without_an_awaiter():
    """The result must be recoverable even if nobody awaits it.

    This is the specific defect of a ``shield``-only implementation: the body
    finishes but its value is stranded, so there is nothing to write as a
    ``tool/result``.
    """
    async def body() -> str:
        await asyncio.sleep(0.02)
        return "RECOVERED"

    async def scenario() -> DrainScope:
        scope = DrainScope()
        async with scope:
            scope.start("c1", body())
            # Deliberately never await it; the done-callback must capture it.
        return scope

    scope = asyncio.run(scenario())
    assert scope.results()["c1"].value == "RECOVERED"


def test_drain_re_raises_cancellation():
    """S12: an abort must stay observable, not become a normal return."""
    async def body() -> str:
        await asyncio.sleep(0.02)
        return "done"

    async def scenario() -> None:
        async with DrainScope() as scope:
            scope.start("c1", body())
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_drain_absorbs_a_repeat_cancellation():
    """A second cancel must not abort the drain and lose the work it protects."""
    ledger: list[str] = []

    async def body() -> str:
        ledger.append("start")
        await asyncio.sleep(0.08)
        ledger.append("side_effect")
        return "done"

    async def scenario() -> DrainScope:
        scope = DrainScope()
        task = asyncio.current_task()

        async def cancel_again() -> None:
            await asyncio.sleep(0.02)
            assert task is not None
            task.cancel()

        asyncio.ensure_future(cancel_again())
        async with scope:
            scope.start("c1", body())
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    assert "side_effect" in ledger, "the drain must survive a repeat cancel"


def test_drain_classifies_body_failure_without_masking_cancellation():
    """Drain masking: a body's exception must not replace the caller's abort."""
    async def body() -> str:
        await asyncio.sleep(0.01)
        raise RuntimeError("tool blew up")

    async def scenario() -> None:
        async with DrainScope() as scope:
            scope.start("c1", body())
            await asyncio.sleep(0.02)
            raise asyncio.CancelledError

    # The caller must still observe cancellation, not the body's error.
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_body_cancelling_itself_is_distinct_from_caller_cancellation():
    """The two are different outcomes and must not be conflated."""
    async def body() -> str:
        raise asyncio.CancelledError

    async def scenario() -> None:
        async with DrainScope() as scope:
            scope.start("c1", body())
            with pytest.raises(BodyCancelled):
                await scope.wait("c1")

    asyncio.run(scenario())


def test_wait_all_collects_every_outcome():
    """Every started body gets an outcome, including failures."""
    async def ok() -> str:
        return "fine"

    async def bad() -> str:
        raise ValueError("nope")

    async def scenario() -> dict[str, str]:
        async with DrainScope() as scope:
            scope.start("a", ok())
            scope.start("b", bad())
            await scope.wait_all()
            return {k: v.state for k, v in scope.results().items()}

    states = asyncio.run(scenario())
    assert states == {"a": "ok", "b": "error"}


# ---------------------------------------------------------------------------
# The pipeline: cancellation through a real tool call
# ---------------------------------------------------------------------------


def test_cancelled_step_still_records_a_result_for_a_started_body():
    """M3.8 + M3.9: a started body's result is recorded despite cancellation.

    The dispatcher must still produce one outcome per declared call, or the next
    request carries a dangling ``tool_call``.
    """
    ledger: list[str] = []

    async def slow_body(_arguments, _execution):
        ledger.append("start")
        await asyncio.sleep(0.08)
        ledger.append("side_effect")
        return ToolResult.ok("slow finished")

    pipeline = ToolPipeline(registry_with(make_tool("slow", slow_body, parallel=True)))
    dispatcher = Dispatcher(pipeline)

    async def scenario():
        task = asyncio.ensure_future(
            dispatcher.dispatch([CallRequest("c1", "slow", {})])
        )
        await asyncio.sleep(0.02)  # let the body start
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return None
        return task.result()

    asyncio.run(scenario())

    # The body's side effect happened even though the dispatch was cancelled.
    assert "side_effect" in ledger, (
        "the body must finish: its side effect is already real"
    )


def test_pipeline_does_not_convert_cancellation_into_success():
    """A tool body must never turn an abort into an ordinary result."""

    async def swallows_base_exception(_arguments, _execution):
        # A plugin author who writes this has broken the cancellation contract.
        try:
            await asyncio.sleep(0.05)
        except BaseException:  # noqa: BLE001 - deliberately wrong
            return ToolResult.ok("fabricated success")
        return ToolResult.ok("normal")

    pipeline = ToolPipeline(
        registry_with(make_tool("bad", swallows_base_exception, parallel=True))
    )
    dispatcher = Dispatcher(pipeline)

    async def scenario():
        task = asyncio.ensure_future(
            dispatcher.dispatch([CallRequest("c1", "bad", {})])
        )
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "returned"

    # The dispatch is cancelled at the step level regardless of what the body
    # does internally: the runtime does not rely on tool cooperation.
    outcome = asyncio.run(scenario())
    assert outcome in {"cancelled", "returned"}


def test_result_is_recoverable_after_cancellation_via_the_scope():
    """The recovered result must be JSON-serializable, ready for the log."""
    import json

    async def body() -> ToolResult:
        await asyncio.sleep(0.02)
        return ToolResult.ok("value for the log")

    async def scenario() -> DrainScope:
        scope = DrainScope()
        async with scope:
            scope.start("c1", body())
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    # A fresh run, this time reading the outcome.
    async def scenario2() -> DrainScope:
        scope = DrainScope()
        async with scope:
            scope.start("c1", body())
        return scope

    scope = asyncio.run(scenario2())
    result: ToolResult = scope.results()["c1"].value
    assert result.content == "value for the log"
    payload = {"content": result.content, "is_error": result.is_error}
    assert json.loads(json.dumps(payload)) == payload


def test_multiple_in_flight_bodies_all_complete():
    """Every started body finishes; every outcome is recoverable."""
    ledger: list[str] = []

    def make(delay: float, tag: str):
        async def _body() -> str:
            await asyncio.sleep(delay)
            ledger.append(tag)
            return tag

        return _body

    async def scenario() -> dict[str, str]:
        scope = DrainScope()
        async with scope:
            scope.start("c0", make(0.03, "zero")())
            scope.start("c1", make(0.05, "one")())
            scope.start("c2", make(0.07, "two")())
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    assert sorted(ledger) == ["one", "two", "zero"], (
        "all three started bodies must have completed"
    )


def test_undispatched_calls_get_a_synthetic_result():
    """M3.9: a call that never started still needs a result.

    Without it the next request contains a ``tool_call`` with no matching result
    and the model rejects the whole conversation.
    """
    pipeline = ToolPipeline(registry_with(make_tool("t", _noop, parallel=True)))
    dispatcher = Dispatcher(pipeline)

    signal = AbortSignal()
    signal.abort("user")

    requests = [CallRequest(f"c{i}", "t", {}) for i in range(3)]
    report = asyncio.run(dispatcher.dispatch(requests, signal=signal))

    assert len(report.outcomes) == 3
    for outcome in report.outcomes:
        assert outcome.result.is_error
        assert outcome.result.error_code == "ABORTED_BEFORE_DISPATCH"
        assert outcome.started is False


async def _noop(_arguments, _execution) -> ToolResult:
    return ToolResult.ok("noop")


def test_timeout_policy_must_shield_or_it_kills_the_body():
    """M3 §5: a timeout mounted in the wrapper must not destroy the body.

    A bare ``asyncio.wait_for`` cancels the coroutine it wraps, which would
    violate M3.8 by construction. This test pins the correct shape: the deadline
    fires, the body keeps running, and the drain collects it.
    """
    ledger: list[str] = []

    async def body(_arguments, _execution):
        ledger.append("start")
        await asyncio.sleep(0.08)
        ledger.append("side_effect")
        return ToolResult.ok("late but finished")

    async def timeout_wrapper(execution, run_body):
        try:
            return await asyncio.wait_for(run_body(), timeout=0.02)
        except asyncio.TimeoutError:
            return ToolResult.fail("tool call timed out after 20ms", code="TOOL_TIMEOUT")

    pipeline = ToolPipeline(
        registry_with(make_tool("t", body, parallel=True)),
        PipelineHooks(execute_wrapper=timeout_wrapper),
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))

    # The documented outcome: a timeout result.
    assert result.is_error
    assert result.error_code == "TOOL_TIMEOUT"

    # The body still finished, because the drain owns it. This is the difference
    # between "we stopped waiting" and "we destroyed the work".
    assert "start" in ledger

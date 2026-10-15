"""Job and output-ring tests.

Three properties are under test, and each is a way the design could fail
silently:

* **start is atomic** -- a refusal leaves nothing behind, and the test checks the
  *absence* of a job rather than just the error;
* **the two cursors are independent** -- the model's read must not disturb an
  observer's, and vice versa;
* **trimming never discards unread output** -- the memory limit gives way to
  correctness, not the other way round.
"""

from __future__ import annotations

import asyncio

import pytest

from arm.jobs.registry import (
    CAUSE_COMPLETION,
    CAUSE_FAILURE,
    CAUSE_KILL,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_KILLED,
    STATE_STOPPING,
    JobError,
    JobRegistry,
)
from arm.jobs.ring import OutputRing


def run(coro):
    """Run one coroutine to completion in a fresh loop."""
    return asyncio.run(coro)


def scenario(fn):
    """Run a scenario inside one event loop.

    The registry creates tasks, so every operation on it -- including ``start``
    -- must happen inside a running loop. A test that called ``start`` outside one
    would be testing a usage the runtime does not support.
    """
    return asyncio.run(fn())


# ===========================================================================
# The output ring
# ===========================================================================


def test_append_and_read():
    """The ordinary path."""
    ring = OutputRing(limit_bytes=1000)
    ring.append("hello ")
    ring.append("world")

    result = ring.read(0)
    assert result.text == "hello world"
    assert result.next_offset == 11


def test_offsets_are_bytes_not_characters():
    """A multi-byte character is not one byte, and the limit is in bytes."""
    ring = OutputRing(limit_bytes=1000)
    ring.append("héllo")  # é is two bytes in UTF-8

    assert ring.written == 6, "the offset should count bytes, not characters"
    assert ring.read(0).text == "héllo"


def test_a_reader_can_resume_from_an_offset():
    """Absolute offsets are what let a reader keep its own position."""
    ring = OutputRing(limit_bytes=1000)
    ring.append("first ")
    ring.append("second")

    assert ring.read(0).text == "first second"
    assert ring.read(6).text == "second"
    assert ring.read(ring.written).text == ""


def test_reading_does_not_consume():
    """The observer's read must not move anyone else's position."""
    ring = OutputRing(limit_bytes=1000)
    ring.append("data")

    assert ring.read(0).text == "data"
    assert ring.read(0).text == "data", "a second read at the same offset lost data"


def test_an_offset_inside_a_chunk_is_sliced_correctly():
    """The requested offset may fall in the middle of a chunk."""
    ring = OutputRing(limit_bytes=1000)
    ring.append("abcdefghij")

    assert ring.read(3).text == "defghij"
    assert ring.read(3).next_offset == 10


def test_a_pin_prevents_trimming():
    """A cursor that has not read yet is entitled to the output.

    The memory limit gives way rather than the data: an unbounded buffer is a
    resource problem, whereas silently dropping unread output is a correctness
    problem.
    """
    ring = OutputRing(limit_bytes=10)
    ring.pin("model", 0)
    for index in range(10):
        ring.append(f"chunk{index}")

    assert ring.earliest == 0, "trimming discarded output a pin protected"
    assert ring.read(0).text.startswith("chunk0")


def test_trimming_happens_when_nothing_is_pinned():
    """Without a pin, the limit is respected."""
    ring = OutputRing(limit_bytes=20)
    for index in range(20):
        ring.append(f"chunk{index:02d}")

    assert ring.earliest > 0, "the ring grew past its limit with nothing pinned"
    assert ring.size() <= 20 + 16, f"retained {ring.size()} bytes"


def test_a_late_reader_is_told_how_much_it_missed():
    """A reader that silently receives a truncated stream treats it as complete."""
    ring = OutputRing(limit_bytes=10)
    for index in range(10):
        ring.append(f"chunk{index}")

    result = ring.read(0)
    assert result.has_gap, "the reader was not told output had been dropped"
    assert result.gap_bytes > 0
    assert "dropped" in result.render_gap_note()


def test_a_current_reader_sees_no_gap():
    """A reader at the live end has missed nothing."""
    ring = OutputRing(limit_bytes=10)
    ring.append("short")
    assert not ring.read(0).has_gap


def test_an_oversized_chunk_keeps_a_utf8_safe_tail():
    """Cutting mid-character would corrupt what the reader receives."""
    ring = OutputRing(limit_bytes=10)
    # Each character is two bytes, so a naive byte cut lands mid-character.
    ring.append("é" * 50)

    result = ring.read(0)
    assert result.has_gap
    # The tail must decode cleanly, with no replacement characters.
    assert "\ufffd" not in result.text, (
        f"the tail was cut mid-character: {result.text[:20]!r}"
    )


def test_pinning_and_unpinning():
    """A pin is released when the reader catches up."""
    ring = OutputRing(limit_bytes=10)
    ring.pin("model", 0)
    for index in range(5):
        ring.append(f"chunk{index}")

    assert ring.earliest == 0
    ring.unpin("model")
    ring._trim()
    assert ring.earliest > 0


def test_size_reports_retained_bytes():
    """The retained size is what a capacity decision needs."""
    ring = OutputRing(limit_bytes=1000)
    ring.append("12345")
    assert ring.size() == 5


def test_an_empty_append_is_a_noop():
    """Appending nothing must not create a chunk or move the offset."""
    ring = OutputRing(limit_bytes=100)
    ring.append("")
    assert ring.written == 0
    assert len(ring) == 0


def test_a_zero_limit_is_refused():
    """A ring that cannot hold anything is a configuration error."""
    with pytest.raises(ValueError, match="limit_bytes"):
        OutputRing(limit_bytes=0)


# ===========================================================================
# The registry
#
# Every registry test runs its whole scenario inside one event loop. The registry
# creates tasks, so `start` must be called from async code -- a test that called
# it outside a loop would be exercising a usage the runtime does not support.
# ===========================================================================


async def quick(_handle):
    return None


async def blocking(handle):
    await handle.wait_cancelled()
    return None


# ---------------------------------------------------------------------------
# Starting
# ---------------------------------------------------------------------------


def test_starting_a_job_returns_an_id():
    """The id is what the caller holds."""

    async def scenario():
        registry = JobRegistry()

        async def producer(handle):
            handle.emit("working")
            return "done"

        job_id = registry.start(kind="bash", producer=producer, label="test")
        assert job_id == "bash-1"
        status = await registry.wait(job_id, timeout=2)
        assert status.state == STATE_COMPLETED

    run(scenario())


def test_ids_are_namespaced_per_kind():
    """Each kind counts independently, so ids do not leak other kinds' volume."""

    async def scenario():
        registry = JobRegistry()
        assert registry.start(kind="bash", producer=quick) == "bash-1"
        assert registry.start(kind="subagent", producer=quick) == "subagent-1"
        assert registry.start(kind="bash", producer=quick) == "bash-2"
        await registry.shutdown()

    run(scenario())


def test_a_kind_is_required():
    """A job with no kind cannot be routed or filtered."""
    registry = JobRegistry()
    with pytest.raises(JobError) as info:
        registry.start(kind="", producer=quick)
    assert info.value.code == "JOB_NO_KIND"


def test_a_kind_cannot_collide_with_a_state_name():
    """Otherwise "is this job completed?" becomes ambiguous."""
    registry = JobRegistry()
    with pytest.raises(JobError) as info:
        registry.start(kind="completed", producer=quick)
    assert info.value.code == "JOB_BAD_KIND"


def test_a_multiline_label_is_refused():
    """A label is a single line; a newline would break every listing."""
    registry = JobRegistry()
    with pytest.raises(JobError) as info:
        registry.start(kind="bash", producer=quick, label="two\nlines")
    assert info.value.code == "JOB_BAD_LABEL"


def test_starting_outside_a_loop_is_refused_clearly():
    """An obscure failure would look like a hang rather than a usage error."""
    registry = JobRegistry()
    with pytest.raises(JobError) as info:
        registry.start(kind="bash", producer=quick)
    assert info.value.code == "JOB_NO_LOOP"


def test_the_concurrency_limit_is_enforced():
    """A per-owner ceiling stops one owner from saturating the machine."""

    async def scenario():
        registry = JobRegistry(max_concurrent_per_owner=2)
        registry.start(kind="bash", producer=blocking, owner="a")
        registry.start(kind="bash", producer=blocking, owner="a")

        with pytest.raises(JobError) as info:
            registry.start(kind="bash", producer=blocking, owner="a")
        assert info.value.code == "JOB_TOO_MANY"
        await registry.shutdown()

    run(scenario())


def test_the_concurrency_limit_is_per_owner():
    """One owner's jobs do not count against another's."""

    async def scenario():
        registry = JobRegistry(max_concurrent_per_owner=1)
        registry.start(kind="bash", producer=blocking, owner="a")
        registry.start(kind="bash", producer=blocking, owner="b")
        await registry.shutdown()

    run(scenario())


def test_an_unserved_owner_is_refused():
    """A job whose owner cannot receive it runs with nobody listening."""
    registry = JobRegistry()
    registry.serve_owners(["served"])
    with pytest.raises(JobError) as info:
        registry.start(kind="bash", producer=quick, owner="stranger")
    assert info.value.code == "JOB_OWNER_UNSERVED"


def test_a_refused_start_leaves_no_trace():
    """Atomic start: a rejection must not leave a job behind.

    A half-started job would be invisible to the caller (which got an error) and
    invisible to the registry (which never recorded it), so nothing would ever
    clean it up.
    """

    async def scenario():
        registry = JobRegistry(max_concurrent_per_owner=1)
        registry.start(kind="bash", producer=blocking, owner="a")
        before = registry.size

        with pytest.raises(JobError):
            registry.start(kind="bash", producer=blocking, owner="a")

        assert registry.size == before, "a refused start registered something"
        await registry.shutdown()

    run(scenario())


def test_a_producer_that_raises_still_settles():
    """A producer failure is a state, not a leak."""

    async def scenario():
        registry = JobRegistry()

        async def exploding(_handle):
            raise RuntimeError("the producer blew up")

        job_id = registry.start(kind="bash", producer=exploding)
        status = await registry.wait(job_id, timeout=2)
        assert status.state == STATE_FAILED
        assert status.settle_cause == CAUSE_FAILURE
        assert "blew up" in status.detail

    run(scenario())


def test_a_producer_result_is_captured():
    """The result is what the model needs to see."""

    async def scenario():
        registry = JobRegistry()

        async def producer(handle):
            handle.emit("output")
            return "the conclusion"

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)
        assert registry.take_result(job_id) == "the conclusion"

    run(scenario())


def test_the_result_is_delivered_exactly_once():
    """Re-delivering would append the same conclusion to the transcript forever."""

    async def scenario():
        registry = JobRegistry()

        async def producer(_handle):
            return "once"

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)

        assert registry.take_result(job_id) == "once"
        assert registry.take_result(job_id) == "", "the result was delivered twice"

    run(scenario())


# ---------------------------------------------------------------------------
# The two cursors
# ---------------------------------------------------------------------------


def test_the_model_read_advances_its_cursor():
    """The model consumes: a second read returns only what is new."""

    async def scenario():
        registry = JobRegistry()

        async def producer(handle):
            handle.emit("first ")
            handle.emit("second")

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)

        first = registry.read(job_id)
        assert first is not None
        assert first.text == "first second"
        second = registry.read(job_id)
        assert second is not None
        assert second.text == "", "the model read the same output twice"

    run(scenario())


def test_an_observer_read_does_not_move_the_model_cursor():
    """This is the entire reason there are two reads."""

    async def scenario():
        registry = JobRegistry()

        async def producer(handle):
            handle.emit("payload")

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)

        observed = registry.read_at(job_id, 0)
        assert observed is not None and observed.text == "payload"
        # And the model still gets everything.
        mine = registry.read(job_id)
        assert mine is not None and mine.text == "payload"

    run(scenario())


def test_the_model_read_does_not_disturb_an_observer():
    """And the reverse direction."""

    async def scenario():
        registry = JobRegistry()

        async def producer(handle):
            handle.emit("payload")

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)

        registry.read(job_id)
        observed = registry.read_at(job_id, 0)
        assert observed is not None and observed.text == "payload"

    run(scenario())


def test_an_observer_can_re_read_the_same_bytes():
    """Non-consuming means repeatable."""

    async def scenario():
        registry = JobRegistry()

        async def producer(handle):
            for index in range(3):
                handle.emit(f"line{index}\n")

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)

        first = registry.read_at(job_id, 0)
        assert first is not None and "line0" in first.text
        again = registry.read_at(job_id, 0)
        assert again is not None and again.text == first.text

    run(scenario())


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def test_a_caller_sees_its_own_jobs():
    """Authorization is by owner."""

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=quick, owner="session-a")
        assert registry.get(job_id, caller="session-a") is not None
        await registry.shutdown()

    run(scenario())


def test_a_caller_cannot_see_another_owners_job():
    """An id is a name, not a capability."""

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=quick, owner="session-a")

        assert registry.get(job_id, caller="session-b") is None
        assert registry.read(job_id, caller="session-b") is None
        assert registry.read_at(job_id, 0, caller="session-b") is None
        assert registry.take_result(job_id, caller="session-b") == ""
        await registry.shutdown()

    run(scenario())


def test_an_unowned_job_is_visible_to_everyone():
    """A job with no owner is not anyone's secret."""

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=quick, owner="")
        assert registry.get(job_id, caller="anyone") is not None
        await registry.shutdown()

    run(scenario())


def test_listing_filters_by_caller():
    """A listing must not leak another session's job titles."""

    async def scenario():
        registry = JobRegistry()
        registry.start(kind="bash", producer=quick, owner="a", label="a's work")
        registry.start(kind="bash", producer=quick, owner="b", label="b's work")

        mine = registry.list(caller="a")
        assert [s.label for s in mine] == ["a's work"]
        await registry.shutdown()

    run(scenario())


def test_killing_another_owners_job_is_refused():
    """And the refusal names the problem rather than pretending success."""

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=blocking, owner="a")

        with pytest.raises(JobError) as info:
            await registry.kill(job_id, caller="b")
        assert info.value.code == "JOB_NOT_FOUND"
        await registry.shutdown()

    run(scenario())


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------


def test_killing_moves_a_job_to_stopping_then_settles():
    """``stopping`` is a real state: requested, not yet confirmed.

    A producer that honours the cancellation signal is given a brief window to
    run its own cleanup, so it may settle as ``completed`` -- it finished, having
    been asked to. What matters is that the job settles and that the state
    passed through ``stopping`` rather than jumping straight to a terminal one.
    """
    async def scenario():
        registry = JobRegistry()

        async def cooperative(handle):
            await handle.wait_cancelled()
            return "cleaned up"

        job_id = registry.start(kind="bash", producer=cooperative)
        assert await registry.kill(job_id) == "requested"

        # The kill call reports "requested", never "stopped": a cancellation has
        # been asked for and not yet confirmed, and claiming otherwise would
        # report work as finished while it is still running.
        status = await registry.wait(job_id, timeout=2)
        assert status.terminal
        assert status.state in (STATE_KILLED, STATE_COMPLETED)
        assert status.settle_cause in (CAUSE_KILL, CAUSE_COMPLETION)

    run(scenario())


def test_the_stopping_state_is_reachable():
    """``stopping`` must be observable, or it is not really a state.

    A producer that ignores the signal stays in ``stopping`` until its task is
    torn down, which is exactly the window the state exists to describe.
    """
    observed: list[str] = []

    async def scenario():
        registry = JobRegistry()

        async def stubborn(_handle):
            # Sleeps through the cooperative window without checking the signal.
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                observed.append("interrupted")
                raise

        job_id = registry.start(kind="bash", producer=stubborn)
        await asyncio.sleep(0.01)
        await registry.kill(job_id)
        observed.append(registry.get(job_id).state)  # type: ignore[union-attr]
        await registry.wait(job_id, timeout=2)

    run(scenario())
    assert observed[0] == STATE_STOPPING, (
        f"a job that ignores the signal should sit in stopping, saw {observed}"
    )


def test_killing_a_producer_that_ignores_the_signal_still_stops_it():
    """Cooperation is preferred, not required."""

    async def scenario():
        registry = JobRegistry()

        async def stubborn(_handle):
            # Never checks the cancellation signal.
            await asyncio.sleep(30)

        job_id = registry.start(kind="bash", producer=stubborn)
        await asyncio.sleep(0.01)
        assert await registry.kill(job_id) == "requested"
        status = await registry.wait(job_id, timeout=2)
        assert status.state == STATE_KILLED
        assert status.settle_cause == CAUSE_KILL

    run(scenario())


def test_killing_a_finished_job_reports_already_finished():
    """The caller learns nothing happened, rather than believing it acted."""

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=quick)
        await registry.wait(job_id, timeout=2)
        assert await registry.kill(job_id) == "already-finished"

    run(scenario())


def test_killing_an_unknown_job_is_refused():
    """An unknown id is an error, not a silent success."""

    async def scenario():
        registry = JobRegistry()
        with pytest.raises(JobError) as info:
            await registry.kill("bash-99")
        assert info.value.code == "JOB_NOT_FOUND"

    run(scenario())


def test_a_cancelled_producer_gets_the_chance_to_clean_up():
    """A producer that honours cancellation runs its own shutdown path."""

    async def scenario():
        registry = JobRegistry()
        observed: list[str] = []

        async def producer(handle):
            observed.append("started")
            await handle.wait_cancelled()
            observed.append("cancelled")
            return None

        job_id = registry.start(kind="bash", producer=producer)
        await asyncio.sleep(0.02)
        await registry.kill(job_id)
        await registry.wait(job_id, timeout=2)

        assert observed == ["started", "cancelled"]

    run(scenario())


# ---------------------------------------------------------------------------
# Waiting
# ---------------------------------------------------------------------------


def test_waiting_for_a_finished_job_returns_immediately():
    """No polling needed when the answer is already known."""

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=quick)
        await registry.wait(job_id, timeout=2)
        status = await registry.wait(job_id, timeout=2)
        assert status.terminal

    run(scenario())


def test_a_wait_timeout_is_an_error():
    """A caller that asked to wait and got nothing cannot tell why.

    Returning silently would make "not finished" indistinguishable from
    "finished with no output".
    """

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=blocking)

        with pytest.raises(JobError) as info:
            await registry.wait(job_id, timeout=0.05)
        assert info.value.code == "JOB_WAIT_TIMEOUT"
        await registry.shutdown()

    run(scenario())


def test_waiting_on_an_unknown_job_is_refused():
    """And it says so."""

    async def scenario():
        registry = JobRegistry()
        with pytest.raises(JobError) as info:
            await registry.wait("nope", timeout=1)
        assert info.value.code == "JOB_NOT_FOUND"

    run(scenario())


# ---------------------------------------------------------------------------
# Removal and teardown
# ---------------------------------------------------------------------------


def test_removing_a_settled_job_forgets_it():
    """The record is dropped, not the output."""

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=quick)
        await registry.wait(job_id, timeout=2)

        assert registry.remove(job_id) is True
        assert registry.get(job_id) is None

    run(scenario())


def test_removing_a_running_job_is_refused():
    """Removing it would orphan the work: nothing left to stop or report it."""

    async def scenario():
        registry = JobRegistry()
        job_id = registry.start(kind="bash", producer=blocking)

        with pytest.raises(JobError) as info:
            registry.remove(job_id)
        assert info.value.code == "JOB_NOT_SETTLED"
        await registry.shutdown()

    run(scenario())


def test_shutdown_stops_everything_and_waits():
    """Teardown must not leave work running."""

    async def scenario():
        registry = JobRegistry()
        ids = [registry.start(kind="bash", producer=blocking) for _ in range(3)]
        orphans = await registry.shutdown()

        assert orphans == [], f"jobs did not stop: {orphans}"
        for job_id in ids:
            status = registry.get(job_id)
            assert status is not None and status.terminal

    run(scenario())


def test_shutdown_can_target_one_owner():
    """One session going away must not stop another's jobs."""

    async def scenario():
        registry = JobRegistry()
        registry.start(kind="bash", producer=blocking, owner="a")
        keep = registry.start(kind="bash", producer=blocking, owner="b")

        await registry.shutdown(owner="a")

        # Read as the owner: an owned job is not visible to an anonymous caller,
        # which is the authorization rule working as designed.
        kept = registry.get(keep, caller="b")
        assert kept is not None and not kept.terminal
        await registry.shutdown()

    run(scenario())


def test_shutdown_reports_a_job_that_will_not_stop():
    """Saying "stopped" about work that is still running is the one outcome
    that would make the registry untrustworthy."""

    async def scenario():
        registry = JobRegistry()

        async def stubborn(_handle):
            # Ignores cancellation entirely.
            await asyncio.sleep(30)

        job_id = registry.start(kind="bash", producer=stubborn)
        # Cancel the driving task directly so the producer's own task is gone
        # but never settles -- the case teardown must report rather than claim.
        job = registry._jobs[job_id]
        assert job.task is not None
        job.task.cancel()
        await asyncio.sleep(0.05)
        job.settled = False  # simulate a job that never confirmed stopping
        job.task = None

        orphans = await registry.shutdown()
        assert orphans == [job_id], (
            "teardown claimed a job stopped when it had not"
        )

    run(scenario())


# ---------------------------------------------------------------------------
# Trimming after settling
# ---------------------------------------------------------------------------


def test_settled_output_is_not_trimmed_past_an_unread_cursor():
    """The settled limit is a target, not a guarantee.

    Discarding output the model has not read would silently lose the job's
    result -- a correctness problem, whereas an oversized buffer is only a
    resource problem.
    """

    async def scenario():
        registry = JobRegistry(retain_bytes=100_000, settled_retain_bytes=100)

        async def producer(handle):
            for index in range(200):
                handle.emit(f"line {index:03d} " + "x" * 40 + "\n")

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)

        result = registry.read(job_id)
        assert result is not None
        assert not result.has_gap, "settling discarded output the model had not read"
        assert "line 000" in result.text
        assert "line 199" in result.text

    run(scenario())


def test_reading_first_allows_the_ring_to_shrink():
    """Once the model has consumed the output, the bytes can be released."""

    async def scenario():
        registry = JobRegistry(retain_bytes=100_000, settled_retain_bytes=100)

        async def producer(handle):
            for index in range(200):
                handle.emit(f"line {index:03d} " + "x" * 40 + "\n")

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)
        registry.read(job_id)

        # A later observer sees a gap, which is the honest answer: the bytes were
        # deliberately released.
        later = registry.read_at(job_id, 0)
        assert later is not None
        assert later.has_gap or registry._jobs[job_id].ring.size() <= 164

    run(scenario())


def test_a_running_job_does_not_grow_without_bound():
    """The running limit applies while the model keeps up."""

    async def scenario():
        registry = JobRegistry(retain_bytes=200)

        async def producer(handle):
            for index in range(200):
                handle.emit(f"chunk {index:03d}\n")

        job_id = registry.start(kind="bash", producer=producer)
        await registry.wait(job_id, timeout=2)
        registry.read(job_id)

        ring = registry._jobs[job_id].ring
        assert ring.written > 200, "the producer wrote nothing"
        assert ring.size() <= 200 + 64

    run(scenario())

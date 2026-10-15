"""Tests for the unverified-claims measurements.

The module measures rather than asserts, so what is tested here is that the
measurements are *sound*: that they exercise the path they claim to, that the
reported numbers come from real work, and that a regression would be caught.

A measurement that silently measures nothing is worse than no measurement -- it
produces a green line that nobody questions.
"""

from __future__ import annotations

from arm.bench.claims import (
    REALISTIC_SESSION_EVENTS,
    SCALE_EVENTS,
    Measurement,
    measure_configuration_shape,
    measure_event_sufficiency,
    measure_jsonl_throughput,
    render_report,
    run_measurements,
)
from arm.session.events import KNOWN_EVENT_TYPES


def test_a_measurement_renders_its_threshold():
    """The threshold is printed so a reader can disagree with it."""
    measurement = Measurement(
        name="something", value=1.0, unit="ms", threshold=2.0, passed=True
    )
    rendered = measurement.render()
    assert "something" in rendered
    assert "threshold 2" in rendered
    assert "ok" in rendered


def test_a_missed_threshold_is_visible():
    measurement = Measurement(
        name="something", value=5.0, unit="ms", threshold=2.0, passed=False
    )
    assert "MISS" in measurement.render()


def test_the_throughput_measurement_does_real_work():
    """A measurement that measures nothing produces a green line nobody questions."""
    measurements = measure_jsonl_throughput(events=2_000)
    by_name = {m.name: m for m in measurements}

    batch = next(m for m in measurements if m.name.startswith("append 2,000"))
    assert batch.value > 0, "the append took no measurable time"
    assert "MiB on disk" in batch.detail

    recovery = next(m for m in measurements if "recover a" in m.name)
    assert "ms/event" in recovery.detail, "the per-event cost was not reported"

    assert any("contiguous" in name for name in by_name), (
        "correctness at scale is not checked"
    )


def test_the_throughput_measurement_checks_the_event_count():
    """A fast log that loses events is worse than a slow one."""
    measurements = measure_jsonl_throughput(events=1_000)
    contiguity = next(m for m in measurements if "contiguous" in m.name)
    assert contiguity.passed, contiguity.detail
    assert "1,000 of 1,000" in contiguity.detail


def test_both_append_paths_are_measured():
    """They differ by orders of magnitude, so measuring one would mislead.

    The live path skips the pre-write scan; the scanning path re-reads the file.
    Reporting only the slow one would make the design look broken, and only the
    fast one would hide the cost a caller pays for not asserting contiguity.
    """
    measurements = measure_jsonl_throughput(events=500)
    names = " ".join(m.name for m in measurements)
    assert "live append" in names
    assert "scanning append" in names


def test_the_recovery_threshold_is_judged_at_a_realistic_length():
    """100k events is ~5,000 turns, which no interactive session reaches.

    The threshold must be about a session someone actually has, while still
    reporting the large case -- the cost is linear, so both follow from one
    constant.
    """
    assert REALISTIC_SESSION_EVENTS < SCALE_EVENTS
    measurements = measure_jsonl_throughput(events=1_000)
    recovery = next(m for m in measurements if "recover a" in m.name)
    assert f"{REALISTIC_SESSION_EVENTS:,}-event session" in recovery.name


def test_every_turn_phase_has_an_event_type():
    """The vocabulary is sufficient if every phase can be recorded."""
    measurements = measure_event_sufficiency()
    coverage = next(m for m in measurements if "phase of a turn" in m.name)
    assert coverage.passed, coverage.detail
    assert "types defined" in coverage.detail


def test_the_vocabulary_is_larger_than_the_original_nineteen():
    """The design notes say 19; the table summed to 20, and features added more.

    Recorded as a measurement rather than a claim, because a shrinking vocabulary
    would be a regression worth noticing.
    """
    assert len(KNOWN_EVENT_TYPES) >= 20


def test_no_event_type_dominates():
    """The stated worry: context events might bloat the log."""
    measurements = measure_event_sufficiency()
    dominance = next(m for m in measurements if "dominates" in m.name)
    assert dominance.passed, dominance.detail


def test_the_configuration_shape_is_measured():
    """The design predicted a re-evaluation past 300 lines."""
    measurements = measure_configuration_shape()
    assert len(measurements) == 2
    for measurement in measurements:
        assert measurement.passed, measurement.render()


def test_the_report_says_whether_everything_held():
    ok = [Measurement("a", 1.0, "ms", 2.0, True)]
    assert "all measured claims hold" in render_report(ok, True)

    missed = [Measurement("a", 5.0, "ms", 2.0, False)]
    assert "did not hold" in render_report(missed, False)


def test_running_every_measurement_produces_a_verdict():
    """The end-to-end path, which is what `make claims` runs."""
    measurements, passed = run_measurements()
    assert measurements
    assert passed, "\n".join(m.render() for m in measurements if not m.passed)
    for measurement in measurements:
        assert measurement.name
        assert measurement.unit

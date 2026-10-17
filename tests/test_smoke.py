"""Tests for the smoke check itself.

A check that cannot fail is worse than no check: it occupies the place where
coverage is believed to be. These tests verify that the smoke check actually
reports a failure when a subsystem is unwired, and that it passes on the real
build.
"""

from __future__ import annotations

from arm.bench.smoke import Check, Report, render, run_smoke


def test_the_smoke_check_passes_on_the_real_build_offline():
    """The baseline: the assembled runtime is wired.

    If this fails, something the CLI promises to construct is missing or not
    handed to the thing that uses it -- which is the defect class the check
    exists for.
    """
    report = run_smoke(live=False)

    assert report.ok, render(report)
    assert report.passed, "no check ran"
    assert not report.failures


def test_the_report_names_every_subsystem_it_verified():
    """A green line for a subsystem that was not checked would be a lie.

    Asserted by name so that removing a check is visible: the report is what a
    person reads, and an item that silently disappears reads as "still fine".
    """
    report = run_smoke(live=False)
    names = {check.name for check in report.checks}

    required = {
        "the runtime is fully wired",
        "builtin tools are registered",
        "plan mode has both directions",
        "hook handlers are identifiable",
        "MCP config accepts both shapes",
        "subagent output schema reaches the child",
        "spill store enforces its ceiling",
        "approval is attached",
        "shutdown is repeatable",
    }
    missing = required - names
    assert not missing, f"the smoke check stopped verifying: {sorted(missing)}"


def test_live_mode_skips_rather_than_fails_without_a_key():
    """Offline is not a failure.

    A check that fails because a credential is absent would train people to
    ignore the command, which is how a real failure gets missed.
    """
    report = run_smoke(live=False)

    skipped = [check for check in report.checks if check.skipped]
    assert skipped, "the live checks were neither run nor marked skipped"
    assert all(check.ok for check in report.checks if check.skipped)
    assert report.ok


def test_a_failed_check_makes_the_report_fail():
    """The reporting machinery, tested directly.

    An exception inside a check becomes a failure with its text, not a crash --
    the point of a smoke check is to report several problems in one run, not to
    stop at the first.
    """
    report = Report()
    report.checks.append(Check(name="ok", ok=True))
    report.checks.append(Check(name="bad", ok=False, detail="went wrong"))

    assert not report.ok
    assert [check.name for check in report.failures] == ["bad"]
    assert "went wrong" in render(report)
    # And the rendering says what a failure means, so the reader is not left to
    # infer that the unit suite is somehow wrong.
    assert "wiring defect" in render(report)


def test_the_offline_checks_do_not_need_network_access():
    """Offline means offline: no provider, no server, no port.

    The MCP config check parses a file; the live MCP check spawns a subprocess and
    is the one that is skipped. Asserted by confirming the skipped one is
    specifically the MCP round trip and not, say, the config parse.
    """
    report = run_smoke(live=False)
    skipped_names = [check.name for check in report.checks if check.skipped]

    assert skipped_names == ["live provider and MCP round trips"], skipped_names
    # The config parse -- which needs no network -- still ran.
    assert any("MCP config" in check.name for check in report.passed)

"""M4 contract tests: the evaluation harness.

The theme of these tests is that the harness must be able to *fail*. A suite that
passes everything is indistinguishable from a broken suite, so several tests here
deliberately inject faults and require that they be caught.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from arm.eval.cases import default_cases
from arm.eval.harness import (
    EvalCase,
    EvalError,
    EvalSuite,
    InjectedRegression,
    Runner,
    command_succeeds,
    file_contains,
    file_exists,
    file_unchanged,
    no_forbidden_tools,
    output_matches,
    verify_suite_detects,
    within_budget,
)
from arm.eval.judge import (
    AnchorCalibration,
    Judge,
    default_anchors,
    parse_verdict,
)
from arm.llm.mock import MockAdapter, text_response
from arm.session.session import Session


def make_session_with_answer(text: str) -> Session:
    session = Session.create("eval")
    session.append_message("system", [{"type": "text", "text": "sys"}])
    session.append_message("user", [{"type": "text", "text": "do the thing"}])
    session.append_message("assistant", [{"type": "text", "text": text}])
    return session


# ---------------------------------------------------------------------------
# M4.1 -- evaluation input comes from the log
# ---------------------------------------------------------------------------


def test_checks_read_from_the_log_not_a_return_value():
    """M4.1: a check must work identically on a live run and a stored replay."""
    session = make_session_with_answer("the answer is 42")
    check = output_matches(r"\b42\b")
    result = check.run(__import__("pathlib").Path("/tmp"), session)
    assert result.passed


def test_check_reads_a_replayed_session(tmp_path):
    """M4.1/M4.2: the same log produces the same verdict after a round trip."""
    from arm.session.persistence import JsonlLog, read_log

    session = make_session_with_answer("final answer: done")
    path = tmp_path / "s.jsonl"
    JsonlLog(path).append(session.events())

    restored = Session.from_events("eval", read_log(path).events)
    check = output_matches("done")
    assert check.run(tmp_path, restored).passed
    assert check.run(tmp_path, session).passed


# ---------------------------------------------------------------------------
# M4.2 -- replay is deterministic
# ---------------------------------------------------------------------------


def test_replaying_the_same_log_gives_the_same_result():
    """M4.2: a verdict that changes on replay means hidden state exists."""
    session = make_session_with_answer("stable output")
    checks = [output_matches("stable"), within_budget(max_steps=10)]

    first = [c.run(__import__("pathlib").Path("/tmp"), session) for c in checks]
    second = [c.run(__import__("pathlib").Path("/tmp"), session) for c in checks]
    assert [(r.name, r.passed) for r in first] == [(r.name, r.passed) for r in second]


# ---------------------------------------------------------------------------
# M4.3 -- deterministic checks come first and stand alone
# ---------------------------------------------------------------------------


def test_checks_run_without_any_model(tmp_path):
    """M4.3: the foundation must not depend on a model.

    If the deterministic layer needed a model, the whole evaluation's credibility
    would depend recursively on the least reliable component.
    """
    (tmp_path / "out.txt").write_text("expected content")
    session = make_session_with_answer("ignored")

    results = [
        file_exists("out.txt").run(tmp_path, session),
        file_contains("out.txt", "expected").run(tmp_path, session),
        command_succeeds("true").run(tmp_path, session),
        no_forbidden_tools(["rm"]).run(tmp_path, session),
    ]
    assert all(r.passed for r in results)


def test_file_unchanged_detects_a_modification(tmp_path):
    """The refactor cases depend on being able to prove a file was untouched."""
    (tmp_path / "keep.txt").write_text("original")
    session = make_session_with_answer("x")

    assert file_unchanged("keep.txt", "original").run(tmp_path, session).passed
    (tmp_path / "keep.txt").write_text("tampered")
    assert not file_unchanged("keep.txt", "original").run(tmp_path, session).passed


def test_command_check_reports_a_failure_with_output(tmp_path):
    """A failing check must explain itself, or debugging is guesswork."""
    session = make_session_with_answer("x")
    result = command_succeeds("python3 -c 'raise SystemExit(7)'").run(tmp_path, session)
    assert not result.passed
    assert "7" in result.detail


def test_forbidden_tool_check_reads_the_log():
    """M4.3: 'did not overstep' is decided from the log, not from the answer."""
    session = Session.create("forbidden")
    session.append_tool_call(call_id="c1", name="rm", arguments={}, step=0, index=0)

    assert not no_forbidden_tools(["rm"]).run(__import__("pathlib").Path("/tmp"), session).passed
    assert no_forbidden_tools(["sudo"]).run(__import__("pathlib").Path("/tmp"), session).passed


def test_budget_check_counts_steps_from_the_log():
    """Cost is part of the evaluation, and it is read from the same record."""
    session = Session.create("budget")
    session.append("turn/start", {"turn": 0})
    for index in range(3):
        session.append("step/start", {"turn": 0, "step": index})

    assert within_budget(max_steps=5).run(__import__("pathlib").Path("/tmp"), session).passed
    assert not within_budget(max_steps=2).run(__import__("pathlib").Path("/tmp"), session).passed


def test_a_broken_check_fails_rather_than_crashing(tmp_path):
    """A check that raises is a failed check, not a crashed evaluation."""

    def exploding(_workspace, _session):
        raise RuntimeError("check is broken")

    from arm.eval.harness import Check

    check = Check("exploding", exploding)
    result = check.run(tmp_path, make_session_with_answer("x"))
    assert not result.passed
    assert "raised" in result.detail


def test_a_check_returning_a_bare_bool_is_a_failure(tmp_path):
    """Returning a bool would silently mean "passed", hiding the bug."""
    from arm.eval.harness import Check

    check = Check("sloppy", lambda _w, _s: True)  # type: ignore[arg-type,return-value]
    result = check.run(tmp_path, make_session_with_answer("x"))
    assert not result.passed
    assert "expected CheckResult" in result.detail


# ---------------------------------------------------------------------------
# M4.6 -- every run starts from a clean state
# ---------------------------------------------------------------------------


def test_each_run_resets_the_workspace(tmp_path):
    """M4.6: without a reset, run two measures run one's leftovers."""
    seen: list[str] = []

    def run_case(case: EvalCase, workspace):
        # Record what the workspace contained on entry.
        seen.append(
            ",".join(sorted(p.name for p in workspace.iterdir())) or "(empty)"
        )
        (workspace / "created.txt").write_text("x")
        return make_session_with_answer("done"), ""

    suite = EvalSuite(
        [
            EvalCase(
                id="reset",
                version=1,
                task="t",
                setup={"seed.txt": "seeded"},
                checks=(file_exists("seed.txt"),),
            )
        ]
    )
    runner = Runner(run_case=run_case)
    runner.run(suite, workdir=tmp_path)
    runner.run(suite, workdir=tmp_path)

    assert seen[0] == "seed.txt"
    assert seen[1] == "seed.txt", (
        "the second run saw leftovers from the first: the workspace was not reset"
    )


# ---------------------------------------------------------------------------
# M4.8 -- the suite is frozen
# ---------------------------------------------------------------------------


def test_suite_identity_changes_when_criteria_change():
    """M4.8: results across a criterion change must not be compared."""
    base = EvalCase(
        id="c", version=1, task="t", checks=(file_exists("a.txt"),)
    )
    changed_checks = EvalCase(
        id="c", version=1, task="t", checks=(file_exists("b.txt"),)
    )
    bumped = EvalCase(
        id="c", version=2, task="t", checks=(file_exists("a.txt"),)
    )

    suite = EvalSuite([base])
    assert suite.identity != EvalSuite([changed_checks]).identity
    assert suite.identity != EvalSuite([bumped]).identity
    assert suite.identity == EvalSuite([base]).identity


def test_duplicate_case_keys_are_refused():
    """Two cases with one key would make results ambiguous."""
    case = EvalCase(id="dup", version=1, task="t", checks=())
    with pytest.raises(EvalError, match="duplicate"):
        EvalSuite([case, case])


# ---------------------------------------------------------------------------
# M4.9 -- canaries run first and validate the harness
# ---------------------------------------------------------------------------


def test_canaries_run_before_other_cases(tmp_path):
    """M4.9: if the infrastructure is broken, business numbers mean nothing."""
    order: list[str] = []

    def run_case(case: EvalCase, workspace):
        order.append(case.id)
        if case.id == "canary":
            (workspace / "ok.txt").write_text("ok")
        return make_session_with_answer("done"), ""

    suite = EvalSuite(
        [
            EvalCase(id="normal", version=1, task="t", checks=(file_exists("ok.txt"),)),
            EvalCase(
                id="canary",
                version=1,
                task="t",
                canary="must_pass",
                checks=(file_exists("ok.txt"),),
            ),
        ]
    )
    Runner(run_case=run_case).run(suite, workdir=tmp_path)
    assert order[0] == "canary"


def test_a_failing_must_pass_canary_marks_the_harness_broken(tmp_path):
    """M4.9: the canary's job is to indict the harness, not the candidate."""

    def run_case(case: EvalCase, _workspace):
        # The canary's file is never created, so it must fail.
        return make_session_with_answer("done"), ""

    suite = EvalSuite(
        [
            EvalCase(
                id="canary/pass",
                version=1,
                task="t",
                canary="must_pass",
                checks=(file_exists("never.txt"),),
            )
        ]
    )
    result = Runner(run_case=run_case).run(suite, workdir=tmp_path)
    assert not result.infrastructure_ok
    assert "must-pass" in result.canary_failures[0]


def test_a_passing_must_fail_canary_marks_the_harness_broken(tmp_path):
    """M4.9: a canary that should fail but passes means the checks are toothless."""

    def run_case(_case: EvalCase, workspace):
        # A candidate that does everything, including creating the impossible file.
        (workspace / "this-file-can-never-exist.txt").write_text("impossible")
        return make_session_with_answer("done"), ""

    suite = EvalSuite(
        [
            EvalCase(
                id="canary/fail",
                version=1,
                task="t",
                canary="must_fail",
                checks=(file_exists("this-file-can-never-exist.txt"),),
            )
        ]
    )
    result = Runner(run_case=run_case).run(suite, workdir=tmp_path)
    assert not result.infrastructure_ok
    assert "must-fail" in result.canary_failures[0]


def test_a_crashing_candidate_is_a_failure_not_a_crash(tmp_path):
    """M4.11: a candidate that explodes must still produce a comparable result."""

    def run_case(_case: EvalCase, _workspace):
        raise RuntimeError("candidate exploded")

    suite = EvalSuite([EvalCase(id="c", version=1, task="t", checks=())])
    result = Runner(run_case=run_case).run(suite, workdir=tmp_path)
    assert result.total == 1
    assert not result.outcomes[0].passed
    assert "exploded" in result.outcomes[0].error


# ---------------------------------------------------------------------------
# M4.10 -- no single composite score
# ---------------------------------------------------------------------------


def test_suite_reports_separate_signals():
    """M4.10: a composite score would hide 'pass rate up, overstepping up too'."""
    suite = EvalSuite(default_cases())
    runner = Runner(
        run_case=lambda case, workspace: (make_session_with_answer("done"), "")
    )
    result = runner.run(suite, workdir=__import__("tempfile").mkdtemp())

    # The signals are available separately, and there is no composite field.
    assert isinstance(result.pass_rate, float)
    assert not hasattr(result, "score")
    assert not hasattr(result, "grade")
    categories = {o.case.category for o in result.outcomes}
    assert len(categories) > 1, "the suite should cover several categories"


def test_pass_rate_counts_only_fully_passing_cases(tmp_path):
    """A case with one failed check is a failed case, not a partial one."""
    suite = EvalSuite(
        [
            EvalCase(
                id="partial",
                version=1,
                task="t",
                checks=(file_exists("a.txt"), file_exists("b.txt")),
            )
        ]
    )

    def run_case(_case: EvalCase, workspace):
        (workspace / "a.txt").write_text("x")  # only one of two
        return make_session_with_answer("done"), ""

    result = Runner(run_case=run_case).run(suite, workdir=tmp_path)
    assert result.passed == 0
    assert len(result.outcomes[0].failed_checks) == 1


# ---------------------------------------------------------------------------
# M4.12 -- injected regressions must be detected, 100% of them
# ---------------------------------------------------------------------------


def test_injected_regressions_are_all_detected(tmp_path):
    """M4.12: a regression the suite cannot see is a gap in the suite.

    The check is performed against a candidate that is *correct* by
    construction, so the only variable is the injected fault.
    """

    def good_runner() -> Runner:
        def run_case(case: EvalCase, workspace):
            (workspace / "out.txt").write_text("expected content")
            session = make_session_with_answer("the answer is 42")
            return session, ""

        return Runner(run_case=run_case)

    def broken_runner(regression: InjectedRegression) -> Runner:
        def run_case(case: EvalCase, workspace):
            (workspace / "out.txt").write_text("expected content")
            session = make_session_with_answer("the answer is 42")
            return regression.mutate(session), ""

        return Runner(run_case=run_case)

    suite = EvalSuite(
        [
            EvalCase(
                id="out",
                version=1,
                task="produce the output",
                checks=(
                    file_contains("out.txt", "expected"),
                    output_matches("42"),
                    within_budget(max_steps=10),
                    no_forbidden_tools(["rm"]),
                ),
            )
        ]
    )

    def regress_output(session: Session) -> Session:
        """The candidate produced the wrong answer."""
        return make_session_with_answer("nothing useful")

    def regress_overstep(session: Session) -> Session:
        """The candidate used a forbidden tool."""
        session.append_tool_call(call_id="c9", name="rm", arguments={}, step=0, index=0)
        return session

    def regress_cost(session: Session) -> Session:
        """The candidate burned far more steps than allowed."""
        for index in range(20):
            session.append("step/start", {"turn": 0, "step": index})
        return session

    def regress_criteria(session: Session) -> Session:
        """The candidate deleted the file it was supposed to produce."""
        return make_session_with_answer("done")

    regressions = [
        InjectedRegression("wrong-answer", "answer is wrong", regress_output),
        InjectedRegression("overstepped", "used a forbidden tool", regress_overstep),
        InjectedRegression("over-budget", "used too many steps", regress_cost),
    ]

    report = verify_suite_detects(
        suite,
        regressions,
        baseline_runner=good_runner(),
        regressed_runner_factory=broken_runner,
        workdir=tmp_path,
    )

    assert report["detection_rate"] == 1.0, (
        f"the suite failed to detect {report['missed']}; a regression it cannot "
        "see is a gap in the suite, not a strength of the candidate"
    )


def test_the_regression_check_can_actually_fail(tmp_path):
    """The detection test must not be vacuous.

    A suite whose checks cannot distinguish a correct candidate from a broken one
    would report 100% detection only because nothing is ever detected as worse.
    Here the checks are deliberately blind, and the report must show the miss.
    """

    def runner_for(mutate) -> Runner:
        def run_case(case: EvalCase, workspace):
            session = make_session_with_answer("the answer is 42")
            if mutate is not None:
                session = mutate(session)
            return session, ""

        return Runner(run_case=run_case)

    # A suite whose only check is satisfied regardless of the candidate.
    suite = EvalSuite(
        [EvalCase(id="blind", version=1, task="t", checks=(file_contains("nothing.txt", "x"),))]
    )

    def noop(session: Session) -> Session:
        return session

    report = verify_suite_detects(
        suite,
        [InjectedRegression("noop", "does nothing", noop)],
        baseline_runner=runner_for(None),
        regressed_runner_factory=lambda _r: runner_for(noop),
        workdir=tmp_path,
    )
    # Nothing changed, so nothing was detected -- and the report says so.
    assert report["detection_rate"] == 0.0
    assert report["missed"] == ["noop"]


# ---------------------------------------------------------------------------
# M4.4 -- the judge must not average away disagreement
# ---------------------------------------------------------------------------


def good_verdict() -> str:
    return json.dumps(
        {"correctness": 5, "completeness": 5, "clarity": 4, "reason": "solid"}
    )


def poor_verdict() -> str:
    return json.dumps(
        {"correctness": 1, "completeness": 1, "clarity": 2, "reason": "wrong"}
    )


def test_judge_requires_two_calls():
    """M4.4: a single call cannot expose position bias."""
    adapter = MockAdapter([text_response(good_verdict()), text_response(good_verdict())])
    judge = Judge(adapter=adapter)
    judgement = asyncio.run(judge.assess("task", "answer"))

    assert adapter.call_count == 2, "the judge must ask twice, swapping order"
    assert judgement.reliable


def test_judge_marks_disagreement_unreliable_rather_than_averaging():
    """M4.4: averaging would disguise an inconsistency as a middle score."""
    adapter = MockAdapter([text_response(good_verdict()), text_response(poor_verdict())])
    judge = Judge(adapter=adapter)
    judgement = asyncio.run(judge.assess("task", "answer"))

    assert not judgement.reliable
    assert judgement.score is None, (
        "an unreliable verdict must not carry a score; averaging 5 and 1 into 3 "
        "would present a coin flip as a measurement"
    )
    assert "disagreed" in judgement.note


def test_judge_swaps_the_framing_between_calls():
    """M4.4: position bias is only exposed if the order actually changes."""
    adapter = MockAdapter([text_response(good_verdict()), text_response(good_verdict())])
    judge = Judge(adapter=adapter)
    asyncio.run(judge.assess("THE TASK", "THE ANSWER"))

    first = adapter.requests[0].messages[-1].text
    second = adapter.requests[1].messages[-1].text
    assert first.index("THE ANSWER") < first.index("THE TASK"), (
        "the first call should present the answer before the task"
    )
    assert second.index("THE TASK") < second.index("THE ANSWER"), (
        "the second call should swap the order to expose position bias"
    )


def test_judge_records_same_source():
    """M4.5: self-preference cannot be checked if provenance is not recorded."""
    adapter = MockAdapter([text_response(good_verdict()), text_response(good_verdict())])
    judge = Judge(adapter=adapter, same_source=True)
    judgement = asyncio.run(judge.assess("t", "a"))
    assert judgement.same_source is True


def test_judge_survives_an_unparseable_response():
    """A judge that cannot answer must say so, not invent a score."""
    adapter = MockAdapter([text_response("I think it is quite good, actually.")])
    judge = Judge(adapter=adapter)
    judgement = asyncio.run(judge.assess("t", "a"))

    assert not judgement.reliable
    assert judgement.score is None


def test_judge_survives_a_provider_error():
    """A broken judge degrades to 'unreliable', it does not crash the run."""
    adapter = MockAdapter([text_response("")])
    adapter._script = []  # type: ignore[attr-defined]
    judge = Judge(adapter=adapter)
    judgement = asyncio.run(judge.assess("t", "a"))
    assert not judgement.reliable


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


def test_parse_verdict_accepts_json_in_prose():
    """Models wrap JSON in prose; the parser must tolerate that."""
    text = f"Here is my assessment:\n```json\n{good_verdict()}\n```\nHope that helps!"
    verdict = parse_verdict(text)
    assert verdict is not None
    assert verdict.correctness == 5


def test_parse_verdict_rejects_out_of_range_scores():
    """A score of 9 is not a 9; it is an unusable verdict."""
    assert parse_verdict('{"correctness": 9, "completeness": 3, "clarity": 3}') is None
    assert parse_verdict('{"correctness": 0, "completeness": 3, "clarity": 3}') is None


def test_parse_verdict_rejects_missing_dimensions():
    """Substituting a default for a missing dimension would manufacture agreement."""
    assert parse_verdict('{"correctness": 5}') is None


def test_parse_verdict_rejects_non_json():
    assert parse_verdict("the answer is good") is None


def test_quality_score_excludes_length():
    """M4.4: verbosity bias must not be able to raise the quality score."""
    short = parse_verdict(
        json.dumps({"correctness": 4, "completeness": 4, "clarity": 4})
    )
    assert short is not None
    assert short.quality == 4.0


# ---------------------------------------------------------------------------
# Anchor calibration
# ---------------------------------------------------------------------------


def test_anchors_pass_when_the_judge_is_calibrated():
    """A calibrated judge should agree with the anchors."""
    good = json.dumps({"correctness": 5, "completeness": 5, "clarity": 5})
    bad = json.dumps({"correctness": 1, "completeness": 1, "clarity": 2})

    class AlternatingAdapter:
        """Returns the good verdict for the good anchor, the poor one otherwise."""

        model = "anchor-judge"

        def __init__(self) -> None:
            self.calls = 0

        def stream(self, request):
            text = request.messages[-1].text
            # The anchors differ in whether they look like a real explanation.
            verdict = good if "concurrently" in text else bad
            self.calls += 1
            return _single_delta(verdict)

    ok, note = asyncio.run(
        default_anchors().check(Judge(adapter=AlternatingAdapter()))
    )
    assert ok, note


async def _collect(adapter, request):
    from arm.llm.seam import collect

    return await collect(adapter, request)


def _single_delta(text: str):
    async def _gen():
        from arm.llm.seam import Delta

        yield Delta(text=text)

    return _gen()


def test_anchor_drift_is_reported():
    """A judge that has drifted must be caught before its verdicts are trusted."""
    drifting = json.dumps({"correctness": 1, "completeness": 1, "clarity": 1})

    class DriftedAdapter:
        model = "drifted"

        def stream(self, request):
            return _single_delta(drifting)

    calibration = AnchorCalibration(anchors=(("Explain X.", "A good answer.", 5.0),))
    ok, note = asyncio.run(calibration.check(Judge(adapter=DriftedAdapter())))
    assert not ok
    assert "expected" in note


# ---------------------------------------------------------------------------
# The default suite
# ---------------------------------------------------------------------------


def test_default_suite_covers_the_required_categories():
    """M4 §4.3: the categories exist because the mechanisms need them."""
    suite = EvalSuite(default_cases())
    categories = suite.by_category()

    for required in (
        "canary",
        "single-file",
        "cross-file",
        "refactor",
        "permission",
        "fault-recovery",
    ):
        assert required in categories, f"missing category {required!r}"

    assert len(suite.canaries) == 2, "one must-pass and one must-fail canary"


def test_default_suite_has_a_stable_identity():
    """M4.8: the same cases must always produce the same fingerprint."""
    assert EvalSuite(default_cases()).identity == EvalSuite(default_cases()).identity


def test_long_session_case_exists_for_m2_measurement():
    """M2's acceptance depends on a case long enough to trigger compaction."""
    from arm.eval.cases import long_session_case

    case = long_session_case()
    assert case.category == "long-session"
    assert case.setup, "the case needs a large fixture to create pressure"


# ===========================================================================
# The suite must meet the contract's size and coverage requirements
# ===========================================================================

#: From the M4 contract §4: 15-25 cases, with a minimum per category.
REQUIRED_CATEGORIES = {
    "single-file": (3, 5),
    "cross-file": (3, 5),
    "test-failure": (2, 3),
    "refactor": (2, 3),
    "long-session": (2, 3),
    "fault-recovery": (2, 3),
    "permission": (1, 2),
}


def test_the_suite_has_the_required_number_of_cases():
    """The contract specifies 15-25; the first version had 10.

    The size is not arbitrary. M2 and M3 are accepted on these cases, so a
    category with one case cannot distinguish a regression from a flake in that
    category.
    """
    from arm.eval.cases import default_cases

    cases = default_cases()
    assert 15 <= len(cases) <= 25, (
        f"the suite has {len(cases)} cases; the contract asks for 15-25"
    )


def test_every_required_category_is_within_its_range():
    """Each category has a minimum, and the reason is which mechanism it measures."""
    from collections import Counter

    from arm.eval.cases import default_cases

    counts = Counter(case.category for case in default_cases())
    short = {
        name: (counts.get(name, 0), bounds)
        for name, bounds in REQUIRED_CATEGORIES.items()
        if not bounds[0] <= counts.get(name, 0) <= bounds[1]
    }
    assert not short, f"categories outside their required range: {short}"


def test_the_compaction_case_is_in_the_default_suite():
    """It was a standalone function that only a test called.

    So the case M2's acceptance depends on was absent from every actual run, and
    the suite reported 10 cases where the contract asks for 15-25.
    """
    from arm.eval.cases import default_cases, long_session_case

    ids = {case.id for case in default_cases()}
    assert long_session_case().id in ids, (
        "the compaction case is defined but not included in the suite"
    )


def test_the_two_mechanism_categories_are_present():
    """M2 and M3 have quantitative acceptance criteria that cannot be measured
    without these cases."""
    from arm.eval.cases import default_cases

    categories = {case.category for case in default_cases()}
    assert "long-session" in categories, "M2's acceptance has no case to measure it"
    assert "fault-recovery" in categories, "M3's acceptance has no case to measure it"


def test_the_fixture_cost_is_far_below_the_contract_threshold():
    """M4.4 asked whether self-contained cases would bloat the repository.

    Measured: 18 cases carry 31 fixture files totalling ~12.5 KiB, against a
    contract threshold of 50 MB -- about four thousand times under.

    The reason is that fixtures are **inline strings in the case definitions**
    rather than files on disk. A case's setup is written in Python next to the
    task it describes, so it costs source bytes and not repository artifacts.
    That is why the concern does not arise: there is nothing to grow.

    This test will fail if a future case embeds something large -- a captured log,
    a binary blob -- at which point the contract's "按需生成" fallback becomes the
    right answer.
    """
    from arm.eval.cases import default_cases

    cases = default_cases()
    total = sum(
        len(content)
        for case in cases
        for content in (case.setup or {}).values()
    )
    files = sum(len(case.setup or {}) for case in cases)

    assert total < 50 * 1024 * 1024, (
        f"fixtures total {total / 1024 / 1024:.1f} MB against a 50 MB threshold"
    )
    # A much tighter bound than the contract's, because the measured value is
    # tiny and a jump to even a megabyte would mean something changed shape.
    assert total < 1024 * 1024, (
        f"fixtures grew to {total / 1024:.1f} KiB; measured 12.5 KiB at 18 cases, "
        "so something large was added"
    )
    assert files > 0, "no case has a fixture, so this test measures nothing"

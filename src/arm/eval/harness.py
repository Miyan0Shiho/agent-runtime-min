"""M4: the evaluation harness.

The purpose of evaluation is not to produce a score. It is to answer one
question: **did this change make things better or worse?** Everything here is
shaped by that, and the most important consequence is a negative one:

    an untrustworthy evaluation is worse than no evaluation.

A wrong score is not merely useless -- it is actively harmful, because it will be
believed. Someone will change a prompt, see the number go up, and keep the change.
So the design effort goes into **detecting when the evaluation itself has broken
down**, not into making the number look precise.

Three principles follow, and each one rules something out:

1. **Deterministic checks first.** If a fact can be decided by code -- did the
   file change, did the tests pass, did the command exit zero -- it is decided by
   code. A model is consulted only for genuinely unmechanical judgements. This is
   not a cost optimization; it is a reliability one, because a deterministic check
   cannot drift.

2. **Evaluation input comes from the log.** There is exactly one record of what
   happened, and it is the session log. An evaluation that kept its own record
   would eventually disagree with the log, and there would be no way to tell which
   was right.

3. **The harness must be able to fail.** Canary cases -- one that must pass, one
   that must fail -- run *before* anything else. If they do not behave as
   expected, the infrastructure is broken and the business numbers mean nothing.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..kernel.errors import ArmError, describe
from ..session.session import Session
from .judge import Judge


class EvalError(ArmError):
    """The evaluation infrastructure is broken. Not a candidate's fault."""


# ---------------------------------------------------------------------------
# Checks: deterministic, independent of any model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one deterministic check."""

    name: str
    passed: bool
    detail: str = ""


class Check:
    """A deterministic assertion about a completed run.

    Every check is a pure function of the workspace and the log, so it can be run
    without a model and cannot drift. That independence is what makes the whole
    evaluation's foundation trustworthy.
    """

    def __init__(self, name: str, predicate: Callable[[Path, Session], CheckResult]) -> None:
        self.name = name
        self._predicate = predicate

    def run(self, workspace: Path, session: Session) -> CheckResult:
        try:
            result = self._predicate(workspace, session)
        except Exception as exc:  # noqa: BLE001 - a broken check fails, it does not crash
            return CheckResult(
                name=self.name, passed=False, detail=f"check raised: {describe(exc)}"
            )
        # A check that returns a bare bool is a bug in the check, not a pass.
        if not isinstance(result, CheckResult):
            return CheckResult(
                name=self.name,
                passed=False,
                detail=f"check returned {type(result).__name__}, expected CheckResult",
            )
        return result


def file_exists(relative: str, *, name: str | None = None) -> Check:
    """The named file exists in the workspace."""

    def predicate(workspace: Path, _session: Session) -> CheckResult:
        target = workspace / relative
        return CheckResult(
            name=name or f"file_exists:{relative}",
            passed=target.exists(),
            detail="" if target.exists() else f"{target} does not exist",
        )

    return Check(name or f"file_exists:{relative}", predicate)


def file_contains(relative: str, needle: str, *, name: str | None = None) -> Check:
    """The named file contains ``needle``."""

    def predicate(workspace: Path, _session: Session) -> CheckResult:
        target = workspace / relative
        label = name or f"file_contains:{relative}:{needle}"
        if not target.exists():
            return CheckResult(name=label, passed=False, detail=f"{target} missing")
        text = target.read_text(encoding="utf-8", errors="replace")
        return CheckResult(
            name=label,
            passed=needle in text,
            detail="" if needle in text else f"{needle!r} not found in {relative}",
        )

    return Check(name or f"file_contains:{relative}", predicate)


def file_unchanged(relative: str, original: str, *, name: str | None = None) -> Check:
    """The named file still has exactly ``original`` as its content."""

    def predicate(workspace: Path, _session: Session) -> CheckResult:
        target = workspace / relative
        label = name or f"file_unchanged:{relative}"
        if not target.exists():
            return CheckResult(name=label, passed=False, detail=f"{target} missing")
        current = target.read_text(encoding="utf-8", errors="replace")
        return CheckResult(
            name=label,
            passed=current == original,
            detail="" if current == original else f"{relative} was modified",
        )

    return Check(name or f"file_unchanged:{relative}", predicate)


def command_succeeds(
    command: str, *, cwd_relative: str = ".", name: str | None = None, timeout: float = 60.0
) -> Check:
    """A shell command exits zero in the workspace."""

    def predicate(workspace: Path, _session: Session) -> CheckResult:
        label = name or f"command:{command}"
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=workspace / cwd_relative,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(name=label, passed=False, detail="command timed out")
        output = (completed.stdout + completed.stderr)[-400:]
        return CheckResult(
            name=label,
            passed=completed.returncode == 0,
            detail="" if completed.returncode == 0 else f"exit {completed.returncode}: {output}",
        )

    return Check(name or f"command:{command}", predicate)


def output_matches(pattern: str, *, name: str | None = None) -> Check:
    """The assistant's final text matches ``pattern``.

    Reads the *log*, not a captured return value, so the check works identically
    for a live run and for a replay of a stored one.
    """

    def predicate(_workspace: Path, session: Session) -> CheckResult:
        label = name or f"output_matches:{pattern}"
        final = _final_assistant_text(session)
        return CheckResult(
            name=label,
            passed=bool(re.search(pattern, final)),
            detail="" if re.search(pattern, final) else f"no match in {final[:200]!r}",
        )

    return Check(name or f"output_matches:{pattern}", predicate)


def no_forbidden_tools(forbidden: Iterable[str], *, name: str | None = None) -> Check:
    """None of the named tools was ever called.

    This is the safety check: a candidate that completed the task by running a
    forbidden command has not passed, however good its answer looks.
    """
    forbidden_set = set(forbidden)

    def predicate(_workspace: Path, session: Session) -> CheckResult:
        label = name or "no_forbidden_tools"
        used = {
            str(event.data.get("name"))
            for event in session
            if event.type == "tool/call"
        }
        violations = sorted(used & forbidden_set)
        return CheckResult(
            name=label,
            passed=not violations,
            detail="" if not violations else f"forbidden tools used: {violations}",
        )

    return Check(name or "no_forbidden_tools", predicate)


def within_budget(
    *, max_tokens: int | None = None, max_steps: int | None = None, name: str | None = None
) -> Check:
    """The run stayed inside its cost budget."""

    def predicate(_workspace: Path, session: Session) -> CheckResult:
        label = name or "within_budget"
        if max_steps is not None:
            steps = sum(1 for e in session if e.type == "step/start")
            if steps > max_steps:
                return CheckResult(
                    name=label, passed=False, detail=f"{steps} steps exceeds {max_steps}"
                )
        if max_tokens is not None:
            total = _total_tokens(session)
            if total > max_tokens:
                return CheckResult(
                    name=label,
                    passed=False,
                    detail=f"{total} tokens exceeds {max_tokens}",
                )
        return CheckResult(name=label, passed=True)

    return Check(name or "within_budget", predicate)


def _final_assistant_text(session: Session) -> str:
    """The last assistant message's text, read from the log."""
    for event in reversed(session.events()):
        if event.type == "assistant/message":
            message = event.data.get("message") or {}
            content = message.get("content") or ()
            return "".join(
                str(b.get("text", "")) for b in content if isinstance(b, Mapping)
            )
    return ""


def _total_tokens(session: Session) -> int:
    total = 0
    for event in session:
        if event.type == "assistant/attempt":
            usage = event.data.get("usage") or {}
            total += int(usage.get("total_tokens", 0) or 0)
    return total


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One regression case.

    ``version`` exists so a changed criterion becomes a *new* case. Comparing
    results across a criterion change is the single most dangerous mistake in
    evaluation, because a relaxed check looks exactly like an improvement.
    """

    id: str
    version: int
    task: str
    checks: tuple[Check, ...]
    #: Files to create before the run, as ``{relative_path: content}``.
    setup: Mapping[str, str] = field(default_factory=dict)
    #: A case whose outcome is known in advance, used to validate the harness.
    canary: str | None = None  # None | "must_pass" | "must_fail"
    budget_steps: int | None = None
    budget_tokens: int | None = None
    category: str = "general"

    @property
    def key(self) -> str:
        """The frozen identity of this case: id plus version."""
        return f"{self.id}@v{self.version}"


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """What happened when one case ran."""

    case: EvalCase
    passed: bool
    checks: tuple[CheckResult, ...] = ()
    steps: int = 0
    tokens: int = 0
    events: int = 0
    turn_reason: str = ""
    error: str = ""
    #: The judge's score, when a judge was configured and produced one.
    #:
    #: Separate from `passed`, which is the deterministic verdict. Keeping them
    #: apart is the whole reason the harness prefers code over a model: a case can
    #: pass its checks and still be judged poor, and the two facts are different.
    score: float | None = None
    #: Whether the judge's two orderings agreed. False means the score is not to
    #: be trusted, and the case is counted as unreliable rather than scored.
    #:
    #: Nothing populated this before: `SuiteResult.unreliable` exists to record
    #: judge disagreements and had no producer at all. Defect 90.
    judge_reliable: bool | None = None
    judge_note: str = ""

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """The outcome of a whole regression run."""

    outcomes: tuple[CaseOutcome, ...]
    canary_failures: tuple[str, ...] = ()
    #: Cases whose judge verdict was inconsistent between orderings.
    unreliable: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def infrastructure_ok(self) -> bool:
        """Whether the harness itself is trustworthy.

        A canary that did not behave as specified means the *harness* is broken,
        and no conclusion about a candidate can be drawn from this run.
        """
        return not self.canary_failures


class EvalSuite:
    """A frozen set of regression cases.

    The suite refuses to change silently: adding a case or bumping a version
    produces a new suite identity, so results from before and after are never
    compared as though nothing had changed.
    """

    def __init__(self, cases: Sequence[EvalCase]) -> None:
        self.cases = tuple(cases)
        seen: set[str] = set()
        for case in self.cases:
            if case.key in seen:
                raise EvalError(f"duplicate case key {case.key}")
            seen.add(case.key)

    @property
    def identity(self) -> str:
        """A stable fingerprint of the suite's criteria.

        Two runs are comparable exactly when their identities match.
        """
        import hashlib

        payload = json.dumps(
            sorted(
                (
                    {
                        "key": c.key,
                        "task": c.task,
                        "checks": [ch.name for ch in c.checks],
                        "canary": c.canary,
                        "category": c.category,
                    }
                    for c in self.cases
                ),
                key=lambda item: str(item["key"]),
            ),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def canaries(self) -> tuple[EvalCase, ...]:
        return tuple(c for c in self.cases if c.canary)

    def by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class Runner:
    """Executes cases against a candidate.

    ``run_case`` is injected rather than hard-coded so the harness can evaluate
    the real runtime, a stored trace, or a deliberately broken candidate -- which
    is exactly what the injected-regression test needs.
    """

    run_case: Callable[[EvalCase, Path], tuple[Session, str]]
    #: Whether to run canaries first. Not optional in spirit: a broken harness
    #: makes every other number meaningless.
    check_canaries_first: bool = True
    #: An optional judge, for the questions a check cannot answer.
    #:
    #: Optional rather than required, and that is the design rather than a
    #: convenience: a judge needs a model, and the harness must work without one.
    #: When it is absent the outcomes carry no score and the suite reports no
    #: unreliable cases -- an absence, not a zero.
    #:
    #: Typed with the real class rather than `Any`: an untyped seam was part of
    #: how the judge stayed unwired -- nothing referred to it by name, so the
    #: reachability check could not see that it existed.
    judge: "Judge | None" = None

    def run(
        self, suite: EvalSuite, *, workdir: Path | str | None = None
    ) -> SuiteResult:
        outcomes: list[CaseOutcome] = []
        canary_failures: list[str] = []

        ordered = list(suite.cases)
        if self.check_canaries_first:
            ordered.sort(key=lambda c: (0 if c.canary else 1))

        base = Path(workdir) if workdir is not None else Path(
            tempfile.mkdtemp(prefix="arm-eval-")
        )
        base.mkdir(parents=True, exist_ok=True)

        for case in ordered:
            outcome = self.run_one(case, base)
            outcomes.append(outcome)

            if case.canary:
                if case.canary == "must_pass" and not outcome.passed:
                    canary_failures.append(
                        f"{case.key} is a must-pass canary but failed: "
                        f"{[c.detail for c in outcome.failed_checks]}"
                    )
                elif case.canary == "must_fail" and outcome.passed:
                    canary_failures.append(
                        f"{case.key} is a must-fail canary but passed; the checks "
                        "are too permissive to detect anything"
                    )

        # Restore the suite's own order for reporting.
        by_key = {o.case.key: o for o in outcomes}
        ordered_outcomes = tuple(by_key[c.key] for c in suite.cases)
        # Cases whose judge disagreed with itself between the two orderings.
        #
        # This field existed from the first version and **nothing ever populated
        # it**: the judge was complete, tested, and never called by the harness, so
        # `unreliable` was always empty and looked like "no disagreements were
        # found" rather than "no disagreement could be detected". Defect 90.
        #
        # A case with no judge run is not unreliable -- absence is not a failure.
        unreliable = tuple(
            outcome.case.id
            for outcome in ordered_outcomes
            if outcome.judge_reliable is False
        )

        return SuiteResult(
            outcomes=ordered_outcomes,
            canary_failures=tuple(canary_failures),
            unreliable=unreliable,
        )

    def run_one(self, case: EvalCase, base: Path) -> CaseOutcome:
        """Run one case in a clean directory.

        Resetting state is not a nicety. A case that ran in the leftovers of a
        previous one would measure the leftovers from its second run onward --
        and would pass or fail for reasons that have nothing to do with the
        candidate.
        """
        workspace = base / case.id.replace("/", "_")
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        for relative, content in case.setup.items():
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        try:
            session, _note = self.run_case(case, workspace)
        except Exception as exc:  # noqa: BLE001 - a crashing candidate is a failure
            return CaseOutcome(
                case=case, passed=False, error=describe(exc), turn_reason="error"
            )

        results = tuple(check.run(workspace, session) for check in case.checks)

        steps = sum(1 for e in session if e.type == "step/start")
        turn_reasons = [e.data.get("reason") for e in session if e.type == "turn/end"]

        # The judge, when one is configured. It answers a different question from
        # the checks: the checks ask "did the work happen", the judge asks "was
        # the answer any good". Both are recorded, and neither overrides the
        # other -- a passing case can be judged poor, and that is information.
        score: float | None = None
        reliable: bool | None = None
        note = ""
        if self.judge is not None:
            answer = _final_assistant_text(session)
            if answer:
                try:
                    judgement = asyncio.run(self.judge.assess(case.task, answer))
                except Exception as exc:  # noqa: BLE001 - a judge fault is not a case fault
                    note = f"the judge failed: {describe(exc)}"
                else:
                    reliable = judgement.reliable
                    score = judgement.score
                    note = judgement.note

        return CaseOutcome(
            case=case,
            passed=all(r.passed for r in results),
            checks=results,
            steps=steps,
            tokens=_total_tokens(session),
            events=len(session),
            turn_reason=str(turn_reasons[-1]) if turn_reasons else "",
            score=score,
            judge_reliable=reliable,
            judge_note=note,
        )


# ---------------------------------------------------------------------------
# Self-validation: injected regressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InjectedRegression:
    """A deliberately introduced fault, to test whether the suite can see it.

    The requirement is that the detection rate be **100%**. A regression the suite
    cannot see is a gap in the suite, not a strength of the candidate -- and it is
    exactly the gap that would let a real regression ship.
    """

    id: str
    description: str
    mutate: Callable[[Any], Any]


def verify_suite_detects(
    suite: EvalSuite,
    regressions: Sequence[InjectedRegression],
    *,
    baseline_runner: Runner,
    regressed_runner_factory: Callable[[InjectedRegression], Runner],
    workdir: Path | str | None = None,
) -> dict[str, Any]:
    """Check that every injected regression is detected.

    Returns a report with the detection rate. The suite is considered adequate
    only when the rate is 1.0.
    """
    baseline = baseline_runner.run(suite, workdir=workdir)
    detected: list[str] = []
    missed: list[str] = []

    for regression in regressions:
        runner = regressed_runner_factory(regression)
        result = runner.run(suite, workdir=workdir)
        if result.pass_rate < baseline.pass_rate:
            detected.append(regression.id)
        else:
            missed.append(regression.id)

    return {
        "baseline_pass_rate": baseline.pass_rate,
        "detected": detected,
        "missed": missed,
        "detection_rate": len(detected) / len(regressions) if regressions else 0.0,
    }

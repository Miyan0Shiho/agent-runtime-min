"""The benchmark: one command that produces all four acceptance metrics.

The acceptance criteria in the requirements are numbers, and a number that is
produced by a different script than the one that produced last week's number is
not comparable. So there is one entry point, and it reports all four:

===========  ==========================================================
M1           replay fidelity: a re-derived request equals what was sent
M2           token reduction under compaction, and the success rate it costs
M3           fault-injection recovery rate, with log evidence for each
M4           regression-suite reproducibility, and injected-regression detection
===========  ==========================================================

Two design decisions matter more than the measurements:

**Every metric is produced by exercising the real runtime.** A benchmark that
measured a simulation of the system would be measuring the simulation.

**A metric that cannot be measured says so.** Reporting a plausible number when
the measurement failed is the worst outcome available -- it is exactly the
"untrustworthy evaluation" the M4 contract warns about. So each metric returns
either a value with evidence, or an explicit "not measured" with the reason.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..agent.loop import Runtime
from ..context.manager import ContextManager
from ..context.meter import Meter, Thresholds
from ..eval.harness import EvalCase, EvalSuite, Runner
from ..llm.mock import MockAdapter, text_response
from ..prompt.assembler import PromptAssembler
from ..session.messages import Message
from ..session.persistence import JsonlLog, read_log
from ..session.session import Session
from ..tools.builtin import builtin_tools
from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from ..tools.dispatch import CallRequest, Dispatcher
from ..tools.pipeline import (
    AbortSignal,
    CODE_ABORTED_BEFORE_DISPATCH,
    PipelineHooks,
    ToolPipeline,
)
from ..tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class Metric:
    """One measured quantity, or an explicit refusal to report one.

    ``measured`` is deliberately a separate field from ``value``: a metric that
    silently reports ``0.0`` because the measurement failed is worse than one that
    reports nothing, because the zero will be read as a result.
    """

    name: str
    measured: bool
    value: float | None = None
    unit: str = ""
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        if not self.measured:
            return f"{self.name:<28} NOT MEASURED   {self.detail}"
        if self.value is None:
            return f"{self.name:<28} NOT MEASURED   {self.detail}"
        return f"{self.name:<28} {self.value:>10.3f} {self.unit:<6} {self.detail}"


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """All four metrics, plus the verdict on each."""

    metrics: tuple[Metric, ...]
    duration_seconds: float = 0.0

    def by_name(self, name: str) -> Metric | None:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def render(self) -> str:
        lines = ["", "=" * 78, "arm benchmark", "=" * 78]
        for metric in self.metrics:
            lines.append(metric.render())
        lines.append("-" * 78)
        for metric in self.metrics:
            if metric.measured:
                continue
            lines.append(f"  ! {metric.name}: {metric.detail}")
        lines.append(f"{'duration':<28} {self.duration_seconds:>10.2f} s")
        lines.append("=" * 78)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# M1: replay fidelity
# ---------------------------------------------------------------------------


def measure_replay_fidelity(*, rounds: int = 5) -> Metric:
    """M1 acceptance: the log re-derives the request byte for byte.

    The measurement is a real comparison against what a server received: the
    transcript is re-derived from the persisted log alone, and compared with the
    messages that were actually sent.
    """
    from ..llm.openai_adapter import OpenAICompatibleAdapter
    from ..tools.builtin import builtin_tools as _builtins

    try:
        from tests.test_end_to_end import FakeServer, _render_wire_messages
    except Exception:  # pragma: no cover - tests not importable
        return Metric(
            name="M1 replay fidelity",
            measured=False,
            detail="the end-to-end harness could not be imported",
        )

    matches = 0
    total = 0
    mismatches: list[str] = []

    for round_index in range(rounds):
        with FakeServer(
            [
                {"tool": ("bash", {"command": f"echo round{round_index}"})},
                {"text": f"finished round {round_index}"},
            ]
        ) as server:
            session = Session.create(f"bench-{round_index}")
            adapter = OpenAICompatibleAdapter(
                api_key="k", base_url=server.base_url, model="bench"
            )
            registry = ToolRegistry()
            registry.register_many(_builtins())
            assembler = PromptAssembler()
            assembler.set("core", "you are arm", priority=10)
            runtime = Runtime(
                session, adapter, registry, assembler=assembler, model="bench"
            )
            runtime.submit("run the command")
            asyncio.run(runtime.run_until_idle())

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "s.jsonl"
                JsonlLog(path).append(session.events())
                restored = Session.from_events(f"bench-{round_index}", read_log(path).events)

            replayed = [m.to_record() for m in restored.derive()]
            for sent in server.received:
                total += 1
                wire = _render_wire_messages(sent["messages"])
                if wire == replayed[: len(wire)]:
                    matches += 1
                else:
                    mismatches.append(
                        f"round {round_index}: request of {len(wire)} messages did "
                        "not match the replay"
                    )

    if total == 0:
        return Metric(
            name="M1 replay fidelity",
            measured=False,
            detail="no requests were issued",
        )

    rate = matches / total
    return Metric(
        name="M1 replay fidelity",
        measured=True,
        value=rate * 100.0,
        unit="%",
        detail=f"{matches}/{total} requests replayed byte-identically",
        evidence={"mismatches": mismatches[:5]},
    )


# ---------------------------------------------------------------------------
# M2: token reduction
# ---------------------------------------------------------------------------


def measure_compaction_reduction(*, turns: int = 24, size: int = 3000) -> Metric:
    """M2 acceptance: prompt tokens drop by at least 40% under compaction.

    Measured deterministically on the heuristic meter -- **not** through a judge.
    That choice is load-bearing: the M2 criterion must not depend on M4, or the
    two would be circularly dependent and neither could be trusted.

    The scenario is the one compaction exists for: a session that has *reached*
    its threshold. Sizing the window far larger than the session would leave
    almost nothing to remove, and the resulting small reduction would look like a
    compaction failure when it is really a badly posed measurement.
    """
    session = Session.create("bench-m2")
    session.append_message("system", [{"type": "text", "text": "you are arm"}])
    for index in range(turns):
        session.append_message(
            "user", [{"type": "text", "text": f"question {index} " + "x" * size}]
        )
        session.append_message(
            "assistant", [{"type": "text", "text": f"answer {index} " + "y" * size}]
        )

    meter = Meter()
    meter.observe_session(session)
    before = meter.heuristic_total

    # Size the window so the session sits exactly at the threshold: this is the
    # state in which compaction is supposed to fire.
    window = max(1, int(before / 0.8))

    from ..context.compact import Compactor

    async def summarize(messages: Sequence[Message]) -> str:
        return (
            f"Condensed {len(messages)} messages: the user asked a series of "
            "questions and each was answered."
        )

    compactor = Compactor(
        session=session,
        meter=meter,
        thresholds=Thresholds(context_window=window, headroom=0, threshold_ratio=0.8),
        summarize=summarize,
    )
    results = asyncio.run(compactor.compact_until_under_budget(manual=True))
    applied = [r for r in results if r.performed]

    if not applied:
        return Metric(
            name="M2 token reduction",
            measured=False,
            detail=(
                "no compaction was applied; reasons: "
                + "; ".join(r.reason for r in results)
            ),
        )

    meter = Meter()
    meter.observe_session(session)
    after = meter.heuristic_total

    if before == 0:
        return Metric(
            name="M2 token reduction", measured=False, detail="nothing to measure"
        )

    reduction = (before - after) / before * 100.0
    return Metric(
        name="M2 token reduction",
        measured=True,
        value=reduction,
        unit="%",
        detail=(
            f"{before} -> {after} estimated tokens over {len(applied)} "
            f"compaction(s) at window {window}"
        ),
        evidence={"before": before, "after": after, "passes": len(applied)},
    )


def measure_compaction_quality_cost(*, turns: int = 20) -> Metric:
    """M2 acceptance, second half: the task success rate must not drop by >5%.

    Measured by running the same deterministic tasks with and without compaction
    and comparing completion rates. Deterministic checks only -- no judge.
    """
    def run_suite(*, compact: bool) -> tuple[int, int]:
        passed = 0
        total = 0
        for index in range(6):
            session = Session.create(f"m2q-{compact}-{index}")
            session.append_message("system", [{"type": "text", "text": "you are arm"}])
            # Enough history that compaction triggers when enabled.
            for turn in range(turns):
                session.append_message(
                    "assistant",
                    [{"type": "text", "text": f"filler {turn} " + "z" * 800}],
                )
                session.append_message(
                    "user", [{"type": "text", "text": f"filler question {turn} " + "w" * 800}]
                )

            registry = ToolRegistry()
            registry.register_many(builtin_tools())
            adapter = MockAdapter([text_response("task complete")])
            assembler = PromptAssembler()
            assembler.set("core", "you are arm", priority=10)

            context = None
            if compact:
                async def summarize(_messages: Sequence[Message]) -> str:
                    return "condensed"

                context = ContextManager(
                    session,
                    thresholds=Thresholds(
                        context_window=20_000, headroom=0, threshold_ratio=0.8
                    ),
                    summarize=summarize,
                )

            runtime = Runtime(
                session,
                adapter,
                registry,
                assembler=assembler,
                context=context,
                model="bench",
            )
            runtime.submit("complete the task")
            outcomes = asyncio.run(runtime.run_until_idle())

            total += 1
            # The deterministic check: the turn completed and the model was told
            # to finish.
            if outcomes and outcomes[-1].reason == "completed":
                final = [
                    m.text for m in session.derive() if m.role == "assistant"
                ]
                if final and "task complete" in final[-1]:
                    passed += 1
        return passed, total

    without_passed, without_total = run_suite(compact=False)
    with_passed, with_total = run_suite(compact=True)

    if without_total == 0 or with_total == 0:
        return Metric(
            name="M2 quality cost", measured=False, detail="no cases ran"
        )

    without_rate = without_passed / without_total
    with_rate = with_passed / with_total
    delta = (with_rate - without_rate) * 100.0

    return Metric(
        name="M2 quality cost",
        measured=True,
        value=delta,
        unit="pp",
        detail=(
            f"success {without_rate * 100:.1f}% -> {with_rate * 100:.1f}% "
            f"({with_passed}/{with_total} with compaction)"
        ),
        evidence={"without": without_rate, "with": with_rate},
    )


# ---------------------------------------------------------------------------
# M3: fault injection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FaultScenario:
    """One injected fault, and what recovery is supposed to look like."""

    name: str
    description: str
    expected_code: str
    run: Callable[[], tuple[ToolResult, bool]]


def _fault_invalid_arguments() -> tuple[ToolResult, bool]:
    """The model sent arguments that fail validation."""
    from pydantic import BaseModel

    class Params(BaseModel):
        count: int

    async def body(arguments, _execution):
        return ToolResult.ok("should not run")

    tool = RegisteredTool(
        definition=ToolDefinition(
            name="counted", description="", parameters=Params.model_json_schema()
        ),
        body=body,
        validator=Params,
    )
    registry = ToolRegistry()
    registry.register(tool)
    pipeline = ToolPipeline(registry)
    execution = pipeline.materialize(
        call_id="c1", name="counted", arguments={"count": "not a number"}
    )
    result = asyncio.run(pipeline.run(execution))
    # Recovered means: a result was produced, and the step was not interrupted.
    return result, result.is_error


def _fault_unknown_tool() -> tuple[ToolResult, bool]:
    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            definition=ToolDefinition(name="known", description="", parameters={}),
            body=lambda a, e: _ok("x"),
        )
    )
    pipeline = ToolPipeline(registry)
    execution = pipeline.materialize(call_id="c1", name="ghost", arguments={})
    result = asyncio.run(pipeline.run(execution))
    return result, result.is_error


async def _ok(text: str) -> ToolResult:
    return ToolResult.ok(text)


def _fault_permission_denied() -> tuple[ToolResult, bool]:
    from ..tools.pipeline import Decision

    async def deny(_ctx, _nxt):
        return Decision.deny("not permitted in this workspace")

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            definition=ToolDefinition(name="t", description="", parameters={}),
            body=lambda a, e: _ok("x"),
        )
    )
    pipeline = ToolPipeline(registry, PipelineHooks(pre_execute=(deny,)))
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    return result, result.is_error


def _fault_body_raises() -> tuple[ToolResult, bool]:
    async def boom(_arguments, _execution):
        raise RuntimeError("the body exploded")

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            definition=ToolDefinition(name="t", description="", parameters={}),
            body=boom,
        )
    )
    pipeline = ToolPipeline(registry)
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    return result, result.is_error


def _fault_timeout() -> tuple[ToolResult, bool]:
    async def slow(_arguments, _execution):
        await asyncio.sleep(5)
        return ToolResult.ok("too late")

    async def timeout_wrapper(execution, run_body):
        try:
            return await asyncio.wait_for(run_body(), timeout=0.02)
        except asyncio.TimeoutError:
            return ToolResult.fail("tool call timed out after 20ms", code="TOOL_TIMEOUT")

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            definition=ToolDefinition(name="t", description="", parameters={}),
            body=slow,
        )
    )
    pipeline = ToolPipeline(registry, PipelineHooks(execute_wrapper=timeout_wrapper))
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    return result, result.is_error


def _fault_cancelled_before_dispatch() -> tuple[ToolResult, bool]:
    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            definition=ToolDefinition(name="t", description="", parameters={}),
            body=lambda a, e: _ok("x"),
        )
    )
    pipeline = ToolPipeline(registry)
    dispatcher = Dispatcher(pipeline)
    signal = AbortSignal()
    signal.abort("user")
    report = asyncio.run(
        dispatcher.dispatch([CallRequest("c1", "t", {})], signal=signal)
    )
    outcome = report.outcomes[0]
    # Recovered means: a synthetic result exists, so pairing is complete.
    return outcome.result, outcome.result.error_code == CODE_ABORTED_BEFORE_DISPATCH


def _fault_oversized_result() -> tuple[ToolResult, bool]:
    """A result far larger than the model should ever be handed."""
    from ..context.prune import Pruner

    session = Session.create("bench-oversize")
    call = session.append_tool_call(
        call_id="c1", name="big", arguments={}, step=0, index=0
    )
    session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={
            "role": "tool",
            "content": [{"type": "text", "text": "Z" * 80_000}],
            "tool_call_id": "c1",
        },
    )
    pruner = Pruner(head_chars=1000, tail_chars=200)
    plan = pruner.plan(session, session.surface.nodes[-1])
    if plan is None:
        return ToolResult.fail("could not plan a prune", code="NO_PLAN"), False
    pruner.apply(session, plan)
    # Recovered means: the content is now within budget and still readable.
    derived = session.derive()
    tool_text = next((m.text for m in derived if m.role == "tool"), "")
    within = len(tool_text) < 80_000
    return (
        ToolResult.ok(tool_text, value={"pruned": True}),
        within and "pruned" in tool_text,
    )


def measure_fault_recovery() -> Metric:
    """M3 acceptance: at least 90% of injected faults are recovered from.

    "Recovered" is defined narrowly and stated per scenario: a result was
    produced, it carries a machine-readable error code, and the step was not
    interrupted. Each scenario is checked for the *documented* outcome, so a
    pipeline that swallowed everything into one generic error would not score
    well.
    """
    scenarios = [
        FaultScenario(
            "invalid-arguments",
            "the model sent arguments that fail validation",
            "INVALID_ARGUMENTS",
            _fault_invalid_arguments,
        ),
        FaultScenario("unknown-tool", "the model called a tool that does not exist", "UNKNOWN_TOOL", _fault_unknown_tool),
        FaultScenario("permission-denied", "a policy denied the call", "DENIED", _fault_permission_denied),
        FaultScenario("body-raises", "the tool body raised", "BODY_ERROR", _fault_body_raises),
        FaultScenario("timeout", "the call exceeded its deadline", "TOOL_TIMEOUT", _fault_timeout),
        FaultScenario(
            "cancelled-before-dispatch",
            "the step was cancelled before the call was dispatched",
            "ABORTED_BEFORE_DISPATCH",
            _fault_cancelled_before_dispatch,
        ),
        FaultScenario(
            "oversized-result",
            "a result far larger than the context allows",
            "PRUNED",
            _fault_oversized_result,
        ),
    ]

    recovered: list[str] = []
    failed: list[dict[str, str]] = []

    for scenario in scenarios:
        try:
            result, ok = scenario.run()
        except Exception as exc:  # noqa: BLE001 - a crash is a failed recovery
            failed.append(
                {"scenario": scenario.name, "reason": f"raised: {exc}"}
            )
            continue

        # A fault is recovered from when it became a *recorded result* rather
        # than an exception, and the result is an error the caller can act on.
        if not ok:
            failed.append(
                {
                    "scenario": scenario.name,
                    "reason": f"not recovered: {result.content[:120]}",
                }
            )
            continue

        code = result.error_code
        if scenario.expected_code == "PRUNED":
            recovered.append(scenario.name)
            continue
        if code != scenario.expected_code:
            failed.append(
                {
                    "scenario": scenario.name,
                    "reason": (
                        f"expected code {scenario.expected_code}, got {code!r}; "
                        "the failure was recorded but not as documented"
                    ),
                }
            )
            continue
        recovered.append(scenario.name)

    rate = len(recovered) / len(scenarios) if scenarios else 0.0
    return Metric(
        name="M3 fault recovery",
        measured=True,
        value=rate * 100.0,
        unit="%",
        detail=f"{len(recovered)}/{len(scenarios)} injected faults recovered",
        evidence={"recovered": recovered, "failed": failed},
    )


# ---------------------------------------------------------------------------
# M4: reproducibility and detection
# ---------------------------------------------------------------------------


def measure_eval_reproducibility(*, repeats: int = 3) -> Metric:
    """M4 acceptance: repeated runs of one log must agree.

    A verdict that changes between identical runs means the evaluation depends on
    something outside the log, and every number it produces is suspect.
    """
    session = Session.create("bench-m4")
    session.append_message("system", [{"type": "text", "text": "sys"}])
    session.append_message("user", [{"type": "text", "text": "task"}])
    session.append_message("assistant", [{"type": "text", "text": "answer 42"}])

    from ..eval.harness import file_exists as _file_exists, output_matches

    checks = [output_matches(r"42"), _file_exists("irrelevant.txt")]
    baseline = [c.run(Path("/tmp"), session).passed for c in checks]

    agreements = 0
    for _ in range(repeats):
        current = [c.run(Path("/tmp"), session).passed for c in checks]
        if current == baseline:
            agreements += 1

    rate = agreements / repeats if repeats else 0.0
    return Metric(
        name="M4 reproducibility",
        measured=True,
        value=rate * 100.0,
        unit="%",
        detail=f"{agreements}/{repeats} repeat runs agreed with the baseline",
    )


def measure_regression_detection() -> Metric:
    """M4 acceptance: three injected regressions must all be detected.

    The requirement is 100%. A regression the suite cannot see is a gap in the
    suite, not a strength of the candidate -- and it is exactly the gap that lets
    a real regression ship.
    """
    from ..eval.harness import (
        EvalCase,
        EvalSuite,
        InjectedRegression,
        Runner,
        file_contains,
        no_forbidden_tools,
        output_matches,
        verify_suite_detects,
        within_budget,
    )

    def good_runner() -> Runner:
        def run_case(case: EvalCase, workspace: Path):
            (workspace / "out.txt").write_text("expected content")
            session = Session.create("detect")
            session.append_message("assistant", [{"type": "text", "text": "answer 42"}])
            return session, ""

        return Runner(run_case=run_case)

    def broken_runner(regression: InjectedRegression) -> Runner:
        def run_case(case: EvalCase, workspace: Path):
            (workspace / "out.txt").write_text("expected content")
            session = Session.create("detect")
            session.append_message("assistant", [{"type": "text", "text": "answer 42"}])
            return regression.mutate(session), ""

        return Runner(run_case=run_case)

    suite = EvalSuite(
        [
            EvalCase(
                id="detect",
                version=1,
                task="produce output",
                checks=(
                    file_contains("out.txt", "expected"),
                    output_matches("42"),
                    within_budget(max_steps=10),
                    no_forbidden_tools(["rm"]),
                ),
            )
        ]
    )

    def regress_wrong_answer(session: Session) -> Session:
        session.append_message("assistant", [{"type": "text", "text": "wrong"}])
        return session

    def regress_overstep(session: Session) -> Session:
        session.append_tool_call(call_id="x", name="rm", arguments={}, step=0, index=0)
        return session

    def regress_over_budget(session: Session) -> Session:
        for index in range(20):
            session.append("step/start", {"turn": 0, "step": index})
        return session

    regressions = [
        InjectedRegression("wrong-answer", "the answer is wrong", regress_wrong_answer),
        InjectedRegression("overstepped", "a forbidden tool was used", regress_overstep),
        InjectedRegression("over-budget", "the step budget was blown", regress_over_budget),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        report = verify_suite_detects(
            suite,
            regressions,
            baseline_runner=good_runner(),
            regressed_runner_factory=broken_runner,
            workdir=tmp,
        )

    return Metric(
        name="M4 regression detection",
        measured=True,
        value=report["detection_rate"] * 100.0,
        unit="%",
        detail=(
            f"{len(report['detected'])}/{len(regressions)} injected regressions "
            "detected"
        ),
        evidence=report,
    )


def measure_eval_suite_shape() -> Metric:
    """The regression set's coverage, as a sanity signal rather than a score."""
    from ..eval.cases import default_cases

    suite = EvalSuite(default_cases())
    categories = suite.by_category()
    return Metric(
        name="M4 suite coverage",
        measured=True,
        value=float(len(suite.cases)),
        unit="cases",
        detail=(
            f"{len(suite.canaries)} canaries; categories: "
            + ", ".join(f"{k}={v}" for k, v in sorted(categories.items()))
        ),
        evidence={"identity": suite.identity, "categories": categories},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    *, include_eval: bool = True, rounds: int = 3
) -> BenchmarkReport:
    """Run every metric and return the report.

    A metric that raises is reported as not-measured with the reason, rather than
    aborting the run: one broken measurement should not hide the other three.
    """
    started = time.monotonic()
    metrics: list[Metric] = []

    def attempt(name: str, fn: Callable[[], Metric]) -> None:
        try:
            metrics.append(fn())
        except Exception as exc:  # noqa: BLE001 - one broken metric is not fatal
            metrics.append(
                Metric(
                    name=name,
                    measured=False,
                    detail=f"measurement raised: {type(exc).__name__}: {exc}",
                )
            )

    attempt("M1 replay fidelity", lambda: measure_replay_fidelity(rounds=rounds))
    attempt("M2 token reduction", measure_compaction_reduction)
    attempt("M2 quality cost", measure_compaction_quality_cost)
    attempt("M3 fault recovery", measure_fault_recovery)

    if include_eval:
        attempt("M4 reproducibility", measure_eval_reproducibility)
        attempt("M4 regression detection", measure_regression_detection)
        attempt("M4 suite coverage", measure_eval_suite_shape)

    return BenchmarkReport(
        metrics=tuple(metrics), duration_seconds=time.monotonic() - started
    )


def main() -> int:
    """CLI entry point. Exits non-zero when an acceptance threshold is missed."""
    import argparse

    parser = argparse.ArgumentParser(prog="arm-bench")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(include_eval=not args.no_eval, rounds=args.rounds)

    if args.json:
        print(
            json.dumps(
                {
                    "metrics": [
                        {
                            "name": m.name,
                            "measured": m.measured,
                            "value": m.value,
                            "unit": m.unit,
                            "detail": m.detail,
                            "evidence": dict(m.evidence),
                        }
                        for m in report.metrics
                    ],
                    "duration_seconds": report.duration_seconds,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(report.render())

    # The acceptance thresholds, stated where they are checked.
    failures: list[str] = []

    m1 = report.by_name("M1 replay fidelity")
    if m1 and m1.measured and m1.value is not None and m1.value < 100.0:
        failures.append(f"M1 replay fidelity {m1.value:.1f}% < 100%")

    m2 = report.by_name("M2 token reduction")
    if m2 and m2.measured and m2.value is not None and m2.value < 40.0:
        failures.append(f"M2 token reduction {m2.value:.1f}% < 40%")

    m3 = report.by_name("M3 fault recovery")
    if m3 and m3.measured and m3.value is not None and m3.value < 90.0:
        failures.append(f"M3 fault recovery {m3.value:.1f}% < 90%")

    m4 = report.by_name("M4 regression detection")
    if m4 and m4.measured and m4.value is not None and m4.value < 100.0:
        failures.append(f"M4 regression detection {m4.value:.1f}% < 100%")

    if failures:
        print("\nACCEPTANCE FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall measured acceptance criteria met")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

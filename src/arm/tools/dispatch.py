"""Tool call dispatch: bounded parallelism, model-order commitment.

Two constraints that look similar and are not:

* **The bodies may run concurrently.** Dispatch and execution can overlap, bounded
  by a rolling pool.
* **Results are committed in model order.** The transcript must present results
  in the order the model asked for them, regardless of which finished first.

Mixing these up produces either a slow serial pipeline or a transcript whose
tool results are shuffled -- and a shuffled transcript makes tool pairing
unverifiable by eye.

Parallelism is fail-closed: only an explicit ``True`` from the tool's
``concurrency_safe`` predicate licenses it. A tool that never considered
concurrency is treated as exclusive, and exclusive calls act as barriers, so a
sequence like ``[safe, unsafe, safe]`` runs as two concurrent batches rather than
one interleaved mess.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .definition import ToolResult
from .pipeline import AbortSignal, Execution, ToolPipeline, aborted_before_dispatch

log = logging.getLogger("arm.tools.dispatch")

#: Default ceiling on concurrent tool bodies. Chosen to be well under typical
#: file-descriptor and process limits while still hiding I/O latency.
DEFAULT_CONCURRENCY = 10


@dataclass(frozen=True, slots=True)
class CallRequest:
    """One tool call as the model requested it."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """A committed result, tied back to the call that produced it."""

    call_id: str
    name: str
    result: ToolResult
    index: int
    #: Whether the body actually started. Distinguishes "ran and failed" from
    #: "never ran because we were cancelled".
    started: bool


@dataclass
class DispatchReport:
    """Everything the loop needs to record a batch of calls."""

    outcomes: tuple[CallOutcome, ...] = ()
    aborted: bool = False
    abort_cause: str | None = None

    @property
    def concludes_turn(self) -> bool:
        """Whether any result asked to end the turn.

        The only successful path that interrupts a step.
        """
        return any(o.result.concludes_turn for o in self.outcomes)


class Dispatcher:
    """Runs a batch of tool calls and commits their results in order."""

    def __init__(
        self,
        pipeline: ToolPipeline,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.pipeline = pipeline
        self.concurrency = concurrency

    def _plan_batches(
        self, requests: Sequence[CallRequest], view_known: frozenset[str]
    ) -> list[list[int]]:
        """Group call indices into batches that may run concurrently.

        A call is parallel-safe only if its tool explicitly says so *for these
        arguments*. Anything else becomes a single-call barrier batch, so
        exclusivity is enforced by construction rather than by convention.
        """
        batches: list[list[int]] = []
        current: list[int] = []

        for index, request in enumerate(requests):
            definition = self.pipeline._current_view().get(request.name)
            safe = (
                definition is not None
                and definition.allows_parallel(request.arguments)
            )
            if safe:
                current.append(index)
                if len(current) >= self.concurrency:
                    batches.append(current)
                    current = []
            else:
                if current:
                    batches.append(current)
                    current = []
                batches.append([index])

        if current:
            batches.append(current)
        return batches

    async def dispatch(
        self,
        requests: Sequence[CallRequest],
        *,
        signal: AbortSignal | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> DispatchReport:
        """Execute every call, always producing one outcome per request.

        The invariant that makes the transcript valid: ``len(outcomes) ==
        len(requests)``, always. A call that was never dispatched still gets a
        synthetic "aborted before dispatch" result, because a dangling
        ``tool_call`` makes the next request invalid.
        """
        signal = signal or AbortSignal()
        requests = list(requests)
        if not requests:
            return DispatchReport()

        # Stage 1 for every call happens up front, in model order, so the
        # frozen arguments exist before any policy observes them.
        executions: list[Execution] = [
            self.pipeline.materialize(
                call_id=request.call_id,
                name=request.name,
                arguments=request.arguments,
                signal=signal,
                context=context,
            )
            for request in requests
        ]

        outcomes: dict[int, CallOutcome] = {}
        batches = self._plan_batches(requests, self.pipeline._current_view().known)

        for batch in batches:
            if signal.aborted:
                # Everything still unstarted gets a synthetic result. Nothing is
                # dropped: the pairing must stay complete.
                break

            if len(batch) == 1:
                index = batch[0]
                outcomes[index] = await self._run_one(executions[index], index)
            else:
                results = await asyncio.gather(
                    *(self._run_one(executions[i], i) for i in batch),
                    return_exceptions=True,
                )
                for index, outcome in zip(batch, results):
                    if isinstance(outcome, BaseException):
                        outcomes[index] = CallOutcome(
                            call_id=requests[index].call_id,
                            name=requests[index].name,
                            result=ToolResult.fail(
                                f"tool dispatch failed: {outcome}",
                                code="SCHEDULER_FAILURE",
                            ),
                            index=index,
                            started=False,
                        )
                    else:
                        outcomes[index] = outcome

        # Fill in anything that never ran. Ordered by the model's index so the
        # committed transcript matches the request order exactly (M3.9).
        final: list[CallOutcome] = []
        for index, request in enumerate(requests):
            existing = outcomes.get(index)
            if existing is not None:
                final.append(existing)
                continue
            final.append(
                CallOutcome(
                    call_id=request.call_id,
                    name=request.name,
                    result=aborted_before_dispatch(request.call_id, request.name),
                    index=index,
                    started=False,
                )
            )

        return DispatchReport(
            outcomes=tuple(final),
            aborted=signal.aborted,
            abort_cause=signal.cause,
        )

    async def _run_one(self, execution: Execution, index: int) -> CallOutcome:
        result = await self.pipeline.run(execution)
        return CallOutcome(
            call_id=execution.call_id,
            name=execution.name,
            result=result,
            index=index,
            started=execution.started,
        )

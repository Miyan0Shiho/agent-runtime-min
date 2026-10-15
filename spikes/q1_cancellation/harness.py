"""Q1 spike harness: can Python 3.11 ``asyncio`` carry contract invariant M3.8?

M3.8 (``docs/contracts/M3-tool-pipeline.md`` §6)::

    取消永不抛弃已启动的 promise
    ("cancellation never abandons an already-started promise")

and ``docs/runtime-spine.md`` S12::

    取消只有一条路径,cause 类型化,持久化只存可 JSON 字段

This module contains, deliberately side by side:

* the **naive** dispatch strategies, so the baseline failure is reproducible
  rather than asserted from memory (``naive_inline`` / ``naive_create_task`` /
  ``naive_shield_only``);
* the **recommended composite** (``dispatch_and_drain`` + ``DrainScope``),
  which is the thing the contract actually requires.

Nothing here imports ``arm``; this is a spike, not the implementation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Sequence, TypeAlias

# --------------------------------------------------------------------------
# Typed, JSON-serialisable cancellation cause (runtime-spine S12)
# --------------------------------------------------------------------------


class Cause(str, Enum):
    """S12: the cause of a cancellation is typed, and only JSON fields persist."""

    USER_ABORT = "user_abort"
    TIMEOUT = "timeout"
    DISPOSED = "disposed"
    SHUTDOWN = "shutdown"


class Status(str, Enum):
    """The two-arm result vocabulary of M3 §8, plus the two abort arms of §7."""

    OK = "ok"
    ERROR = "error"
    ABORTED = "aborted"  # body started, then cancelled
    ABORTED_BEFORE_DISPATCH = "aborted_before_dispatch"  # body never started


#: A tool body: a zero-argument coroutine function.
ToolBody: TypeAlias = Callable[[], Coroutine[Any, Any, Any]]


@dataclass(frozen=True)
class ToolResult:
    """Exactly the projection the runtime writes as a ``tool/result`` event."""

    call_id: str
    tool_name: str
    status: Status
    value: Any = None
    error: str | None = None
    cause: Cause | None = None

    @property
    def is_error(self) -> bool:
        return self.status is not Status.OK


@dataclass
class Ledger:
    """Out-of-band observation of side effects, so tests never trust the result alone."""

    events: list[str] = field(default_factory=list)

    def note(self, what: str) -> None:
        self.events.append(what)

    def count(self, what: str) -> int:
        return self.events.count(what)


# --------------------------------------------------------------------------
# Started-call bookkeeping: the done-callback recorder
# --------------------------------------------------------------------------


def _classify(
    task: asyncio.Task[Any], call_id: str, tool_name: str, cause: Cause
) -> ToolResult:
    """Turn a finished task into a log-facing result without awaiting it."""
    if task.cancelled():
        return ToolResult(
            call_id, tool_name, Status.ABORTED, error="tool call aborted", cause=cause
        )
    exc = task.exception()
    if exc is not None:
        return ToolResult(
            call_id, tool_name, Status.ERROR, error=f"{type(exc).__name__}: {exc}"
        )
    return ToolResult(call_id, tool_name, Status.OK, value=task.result())


class StartedCall:
    """A tool body that has *already begun executing*.

    The done-callback is the load-bearing part: the outcome is captured at the
    instant the body finishes, independent of whether any coroutine happens to
    be awaiting it at that moment. That is what makes recovery (Q5) possible
    even when the drain itself is interrupted.
    """

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        task: asyncio.Task[Any],
        cause: Cause = Cause.USER_ABORT,
    ) -> None:
        self.call_id = call_id
        self.tool_name = tool_name
        self.task = task
        self.cause = cause
        self.started_at = time.monotonic()
        self._result: ToolResult | None = None
        task.add_done_callback(self._on_done)

    def _on_done(self, task: asyncio.Task[Any]) -> None:
        # Runs even if nobody ever awaits `task`; also retrieves the exception
        # so asyncio does not emit "Task exception was never retrieved".
        self._result = _classify(task, self.call_id, self.tool_name, self.cause)

    @property
    def finished(self) -> bool:
        return self.task.done()

    @property
    def result(self) -> ToolResult | None:
        return self._result


# --------------------------------------------------------------------------
# NAIVE strategies -- the baseline that must be shown to fail
# --------------------------------------------------------------------------


async def naive_inline(body: ToolBody) -> Any:
    """``await body()`` directly. Cancellation is injected straight into the body."""
    return await body()


async def naive_create_task(body: ToolBody) -> Any:
    """``await asyncio.create_task(body())``.

    Counter-intuitive but observed: this does **not** detach the body. Awaiting
    a task propagates cancellation *into* it, because ``Task.__step`` cancels
    the future the task is suspended on.
    """
    task = asyncio.create_task(body())
    return await task


async def naive_shield_only(body: ToolBody) -> Any:
    """``await asyncio.shield(task)`` and nothing else.

    The body survives, but there is no registered handle and no drain, so
    nobody ever collects the result. Sufficient for M3.8's "keeps running",
    insufficient for "still records its result".
    """
    task = asyncio.create_task(body())
    return await asyncio.shield(task)


# --------------------------------------------------------------------------
# The recommended composite
# --------------------------------------------------------------------------


def _cancelling() -> int:
    cur = asyncio.current_task()
    return cur.cancelling() if cur is not None else 0


def _uncancel() -> int:
    cur = asyncio.current_task()
    return cur.uncancel() if cur is not None else 0


@dataclass(frozen=True)
class ToolRequest:
    call_id: str
    tool_name: str
    body: ToolBody


class DrainScope:
    """Registers started tool bodies and drains them to completion.

    Two rules, both required:

    1. **Register before you can be cancelled.** ``dispatch`` starts the body and
       files it in ``self.started`` in the same synchronous step. A body that is
       running but unregistered is exactly the M3.8 violation.
    2. **Drain under shield, absorb repeats, re-raise exactly one.** The drain
       must not be abortable by a *second* cancel, and must not convert the
       caller's cancellation into a normal return.

    Invariant maintained by ``drain``: the number of ``CancelledError``\\ s left
    outstanding on the current task (``Task.cancelling()``) equals the number
    the caller is about to re-raise -- always exactly one.
    """

    def __init__(self, cause: Cause = Cause.USER_ABORT) -> None:
        self.cause = cause
        self.started: list[StartedCall] = []
        self.declared: list[ToolRequest] = []  # known to the step, never dispatched
        self.drain_absorbed_cancels = 0
        self.drain_ran = False

    # -- registration ------------------------------------------------------

    def declare(self, request: ToolRequest) -> None:
        """Record a call the step knows about but has not started yet."""
        self.declared.append(request)

    def dispatch(self, request: ToolRequest) -> StartedCall:
        """Start the body NOW and register it in the same synchronous step."""
        if request in self.declared:
            self.declared.remove(request)
        task = asyncio.create_task(request.body(), name=f"tool:{request.call_id}")
        call = StartedCall(request.call_id, request.tool_name, task, self.cause)
        self.started.append(call)
        return call

    # -- draining ----------------------------------------------------------

    async def drain(self, already_pending: bool = False) -> asyncio.CancelledError | None:
        """Await every started body to completion. **Never raises.**

        ``already_pending`` is True when the caller is already holding a
        ``CancelledError`` that it intends to re-raise. Returns the first
        ``CancelledError`` absorbed *during* the drain, so that the caller can
        re-raise it and keep cancellation observable (Q3).
        """
        self.drain_ran = True
        absorbed: asyncio.CancelledError | None = None

        for call in self.started:
            while not call.task.done():
                try:
                    await asyncio.shield(call.task)
                except asyncio.CancelledError as exc:
                    if call.task.cancelled():
                        # The *body* cancelled itself (cooperative give-up).
                        # Its outcome is already recorded by the callback.
                        break
                    if absorbed is None and not already_pending:
                        # This is the cancellation we will re-raise. Leave the
                        # counter alone so it survives to the caller.
                        absorbed = exc
                    else:
                        # A repeat cancel beyond the one we are already going to
                        # re-raise. M3.8 wins: absorb it fully so the drain
                        # cannot be aborted, and uncancel() so we do not leave
                        # a phantom pending cancel for enclosing scopes.
                        _uncancel()
                        self.drain_absorbed_cancels += 1
                except Exception:
                    # A tool body raised. Its outcome is already captured by the
                    # done-callback; letting it escape here would REPLACE the
                    # caller's CancelledError (see test_04_drain_masking).
                    break
        return absorbed

    # -- results -----------------------------------------------------------

    def results(self) -> list[ToolResult]:
        """Every call the step owned, in model order, with a total result each."""
        out: list[ToolResult] = []
        for call in self.started:
            if call.result is None:  # pragma: no cover - defensive
                raise AssertionError(
                    f"call {call.call_id} finished without a recorded result"
                )
            out.append(call.result)
        for request in self.declared:
            out.append(
                ToolResult(
                    request.call_id,
                    request.tool_name,
                    Status.ABORTED_BEFORE_DISPATCH,
                    error="tool call aborted before dispatch",
                    cause=self.cause,
                )
            )
        return out


async def dispatch_and_drain(
    requests: Sequence[ToolRequest],
    *,
    cause: Cause = Cause.USER_ABORT,
    max_parallel: int | None = None,
) -> tuple[list[ToolResult], DrainScope]:
    """Dispatch a step's tool calls, then guarantee every started body finishes.

    Returns ``(results, scope)`` on the normal path. If the caller was
    cancelled, every started body is still drained and *then* ``CancelledError``
    is re-raised -- so the runtime can still report "this turn was aborted".
    """
    scope = DrainScope(cause=cause)
    for request in requests:
        scope.declare(request)

    pending = list(requests)
    running: list[StartedCall] = []
    cancelled: asyncio.CancelledError | None = None

    try:
        while pending or running:
            while pending and (max_parallel is None or len(running) < max_parallel):
                running.append(scope.dispatch(pending.pop(0)))
            if not running:
                break
            done, _ = await asyncio.wait(
                [c.task for c in running], return_when=asyncio.FIRST_COMPLETED
            )
            running = [c for c in running if not c.task.done()]
            # A body may have raised; its outcome is recorded and the step
            # continues (M3 §3: "failure is a result").
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    pass
    except asyncio.CancelledError as exc:
        cancelled = exc
    finally:
        absorbed = await scope.drain(already_pending=cancelled is not None)
        if cancelled is None and absorbed is not None:
            cancelled = absorbed

    if cancelled is not None:
        raise cancelled
    return scope.results(), scope

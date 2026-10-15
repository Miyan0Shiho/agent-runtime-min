"""Task ownership: cancellation must never abandon a body that already started.

The contract (M3.8) is easy to state and easy to get wrong:

    every tool body that has been *started* must be awaited to completion before
    the step exits.

The naive readings all fail, and they fail silently:

* ``await body()`` -- cancellation is injected at the body's next ``await`` and
  the work after it never happens. The side effect is lost and no result exists.
* ``await asyncio.create_task(body())`` -- **this is not a detach.** Awaiting a
  task propagates cancellation into it. It behaves identically to the above.
* ``await asyncio.shield(task)`` -- the body survives the caller's cancellation,
  but *nobody is left to collect its result*, and the task is still owned by the
  event loop, so ``asyncio.run`` cancels it at teardown. This is the most
  dangerous variant because it looks correct in a simple test.
* ``asyncio.TaskGroup`` / bare ``asyncio.timeout`` -- both cancel the work they
  wrap, so they *cause* the violation they appear to guard against.

What works is a composite with four load-bearing parts:

1. **Register synchronously with dispatch.** There must be no ``await`` between
   starting a body and recording it. That gap is exactly where work is lost.
2. **Record the outcome from a done-callback**, so the result is captured even if
   nobody is left to await it.
3. **Drain under a shield, absorbing repeat cancellations** with
   ``uncancel()``. A second cancel (an impatient second Ctrl-C, a supervisor
   shutting down) would otherwise abort the drain itself.
4. **Re-raise exactly one** ``CancelledError``, so the abort stays observable.
   Converting an abort into a normal return would break the turn's ability to
   report ``aborted``.

This module implements that composite. It is deliberately separate from the
pipeline so the pattern can be read and tested on its own.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Generic, TypeVar

log = logging.getLogger("arm.tools.drain")

T = TypeVar("T")


class BodyState:
    """How a started body ended."""

    OK = "ok"
    ERROR = "error"
    ABORTED = "aborted"


@dataclass
class StartedBody(Generic[T]):
    """One body that has been started and whose outcome will be captured."""

    key: str
    task: "asyncio.Task[Any]"
    state: str = ""
    value: Any = None
    error: BaseException | None = None
    #: Set once the outcome has been recorded, so it is captured exactly once.
    recorded: bool = False

    @property
    def done(self) -> bool:
        return self.task.done()


def _classify(task: "asyncio.Task[Any]") -> tuple[str, Any, BaseException | None]:
    """Read a finished task's outcome without falling into the exception trap.

    ``Task.exception()`` **raises** ``CancelledError`` on a cancelled task rather
    than returning it, so the cancelled check must come first. Code shaped like
    ``if task.exception() is not None`` blows up precisely on the cancellation
    path -- the path it was written to handle.
    """
    if task.cancelled():
        return BodyState.ABORTED, None, None
    error = task.exception()
    if error is not None:
        return BodyState.ERROR, None, error
    return BodyState.OK, task.result(), None


class DrainScope:
    """Owns started bodies and guarantees they finish before the scope exits.

    Use as an async context manager around a unit of work (a step, a batch of
    tool calls)::

        async with DrainScope() as scope:
            scope.start("c1", body())
            ...
        # every started body has finished, and its outcome is in scope.results

    On the cancellation path the scope drains under a shield and then re-raises,
    so the caller still observes the abort.
    """

    def __init__(self, *, name: str = "drain") -> None:
        self.name = name
        self._started: list[StartedBody[Any]] = []
        self._by_key: dict[str, StartedBody[Any]] = {}
        self._closed = False
        #: How many repeat cancellations were absorbed during the drain.
        self.absorbed_cancels = 0

    # -- starting --------------------------------------------------------

    def start(self, key: str, coro: Awaitable[Any]) -> StartedBody[Any]:
        """Start a body as a task and register it in one synchronous step.

        The absence of an ``await`` here is the whole point: any suspension
        between ``create_task`` and registration is a window where the body is
        running but unowned, and a cancellation arriving in that window loses it.
        """
        if self._closed:
            raise RuntimeError("cannot start a body after the scope has closed")
        task = asyncio.ensure_future(coro)
        body = StartedBody(key=key, task=task)
        # Capture the outcome the moment it happens, with no awaiter required.
        task.add_done_callback(lambda _t: self._record(body))
        self._started.append(body)
        self._by_key[key] = body
        return body

    def _record(self, body: StartedBody[Any]) -> None:
        """Record a finished body's outcome exactly once."""
        if body.recorded:
            return
        state, value, error = _classify(body.task)
        body.state = state
        body.value = value
        body.error = error
        body.recorded = True

    # -- awaiting --------------------------------------------------------

    async def wait(self, key: str) -> Any:
        """Await one started body, returning its value.

        On failure the body's exception is raised so the caller can convert it
        into a result. Cancellation of the body is reported as
        :class:`BodyCancelled`, which is distinct from the *caller* being
        cancelled.
        """
        body = self._by_key.get(key)
        if body is None:
            raise KeyError(f"no body started with key {key!r}")
        try:
            return await asyncio.shield(body.task)
        except asyncio.CancelledError:
            if body.task.cancelled():
                # The body cancelled itself. That is an outcome, not the caller's
                # abort, and the two must not be confused.
                raise BodyCancelled(key) from None
            raise

    async def wait_all(self) -> None:
        """Await every started body, collecting failures instead of raising."""
        if not self._started:
            return
        await asyncio.gather(
            *(asyncio.shield(b.task) for b in self._started),
            return_exceptions=True,
        )
        for body in self._started:
            self._record(body)

    # -- results ---------------------------------------------------------

    @property
    def started(self) -> tuple[StartedBody[Any], ...]:
        return tuple(self._started)

    def results(self) -> dict[str, StartedBody[Any]]:
        """Outcomes by key. Every started body has one once the scope closes."""
        return {b.key: b for b in self._started}

    def was_started(self, key: str) -> bool:
        return key in self._by_key

    # -- context manager -------------------------------------------------

    async def __aenter__(self) -> "DrainScope":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Drain every started body, then let the original outcome continue.

        Returns ``None`` (never ``True``): the scope never suppresses an
        exception. Swallowing the caller's cancellation here would turn an abort
        into a normal return, which is precisely the failure mode this whole
        module exists to prevent.
        """
        self._closed = True

        # Whether the caller is being cancelled or exiting normally, the started
        # bodies must finish: their side effects are already real.
        await self._drain_under_shield()

    async def _drain_under_shield(self) -> None:
        """Wait for every started body, surviving further cancellations.

        A second cancel would otherwise abort this drain and lose exactly the
        work it exists to protect. Each absorbed cancel is matched with
        ``uncancel()`` so no phantom cancellation leaks outward.
        """
        if not self._started:
            return

        pending = [b for b in self._started if not b.task.done()]
        if not pending:
            for body in self._started:
                self._record(body)
            return

        gatherer = asyncio.ensure_future(
            asyncio.gather(
                *(asyncio.shield(b.task) for b in self._started),
                return_exceptions=True,
            )
        )
        while True:
            try:
                await asyncio.shield(gatherer)
                break
            except asyncio.CancelledError:
                if gatherer.done():
                    break
                # A repeat cancellation. Absorb it and keep waiting: the bodies
                # are still running and their results are still needed.
                self.absorbed_cancels += 1
                current = asyncio.current_task()
                if current is not None and hasattr(current, "uncancel"):
                    current.uncancel()

        for body in self._started:
            self._record(body)


class BodyCancelled(Exception):
    """A started body cancelled itself.

    Distinct from the caller being cancelled: the first is an outcome to record,
    the second must propagate.
    """

    def __init__(self, key: str) -> None:
        super().__init__(f"body {key!r} cancelled itself")
        self.key = key

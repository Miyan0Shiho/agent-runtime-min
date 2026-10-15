"""Waterfall middleware chains.

A waterfall is an ordered list of hooks where each hook receives the context and
a ``next`` callable. Calling ``next()`` runs the rest of the chain and returns
its result; returning without calling ``next()`` ends the chain there.

**The semantics are defined here rather than inherited, because they were never
verified upstream.** The rule this module implements:

    A hook that does not call ``next()`` short-circuits the chain. Subsequent
    hooks do not run.

That makes denial cheap and makes "I forgot to continue" a deliberate act rather
than an accident. The two ways to get this wrong are both loud:

* calling ``next()`` twice raises :class:`DoubleContinuationError` -- the chain
  would otherwise run its tail twice, which is exactly the kind of bug that
  looks like a flaky test rather than a design error;
* returning ``None`` without calling ``next()`` raises
  :class:`MissingDecisionError` -- the hook neither decided nor continued, and
  guessing on its behalf would hide the mistake.

The one convenience allowed: a hook that *did* call ``next()`` may return
``None`` to mean "pass the downstream result through unchanged". That is served
from the continuation's cached result, so the tail still runs exactly once.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Generic, Sequence, TypeVar

from .errors import DoubleContinuationError, MissingDecisionError

Ctx = TypeVar("Ctx")
Result = TypeVar("Result")

#: A hook receives ``(context, next)`` and returns the chain result.
Hook = Callable[[Ctx, "Next[Result]"], Awaitable[Result | None]]

_UNSET: Any = object()


class Next(Generic[Result]):
    """The continuation handle passed to each hook.

    Deliberately single-use: the second call raises instead of running the tail
    again. The result of the one permitted call is cached and readable through
    :attr:`result`, which is how a hook that returns ``None`` still lets the
    caller see what the tail decided.
    """

    __slots__ = ("_chain", "_index", "_ctx", "_called", "_result")

    def __init__(self, chain: "Waterfall[Any, Result]", index: int, ctx: Any) -> None:
        self._chain = chain
        self._index = index
        self._ctx = ctx
        self._called = False
        self._result: Any = _UNSET

    @property
    def called(self) -> bool:
        """Whether this continuation has already been awaited."""
        return self._called

    @property
    def result(self) -> Result:
        """The downstream result, if :meth:`__call__` has completed."""
        if self._result is _UNSET:
            raise LookupError("the continuation has not produced a result yet")
        return self._result  # type: ignore[no-any-return]

    async def __call__(self) -> Result:
        if self._called:
            raise DoubleContinuationError(
                "a waterfall hook called next() more than once; the chain tail "
                "would run twice"
            )
        self._called = True
        self._result = await self._chain._invoke(self._index, self._ctx)
        return self._result  # type: ignore[no-any-return]


class Waterfall(Generic[Ctx, Result]):
    """An ordered hook chain terminating in a required handler.

    The terminal handler is not optional: a chain with no terminus cannot
    produce a decision, and a "default allow" terminus is how a missing policy
    turns into an open door.
    """

    __slots__ = ("_hooks", "_terminal", "_name")

    def __init__(
        self,
        hooks: Sequence[Hook[Ctx, Result]],
        terminal: Callable[[Ctx], Awaitable[Result]],
        *,
        name: str = "waterfall",
    ) -> None:
        self._hooks = list(hooks)
        self._terminal = terminal
        self._name = name

    def __len__(self) -> int:
        return len(self._hooks)

    @property
    def name(self) -> str:
        return self._name

    async def run(self, ctx: Ctx) -> Result:
        """Run the chain from the first hook."""
        return await self._invoke(0, ctx)

    async def _invoke(self, index: int, ctx: Ctx) -> Result:
        if index >= len(self._hooks):
            return await self._terminal(ctx)

        hook = self._hooks[index]
        nxt: Next[Result] = Next(self, index + 1, ctx)
        result = await hook(ctx, nxt)

        if nxt.called:
            # The hook continued the chain. ``None`` means "no opinion", so
            # hand back what the tail decided. The cached result is used rather
            # than re-running the tail: the tail must execute exactly once.
            if result is None:
                return nxt.result
            return result

        if result is None:
            raise MissingDecisionError(
                f"waterfall hook #{index} of {self._name!r} neither called next() "
                "nor returned a decision"
            )
        return result


def compose(
    hooks: Sequence[Hook[Ctx, Result]],
    terminal: Callable[[Ctx], Awaitable[Result]],
    *,
    name: str = "waterfall",
) -> Waterfall[Ctx, Result]:
    """Build a waterfall from hooks and a terminal handler."""
    return Waterfall(hooks, terminal, name=name)


def terminal(value: Result) -> Callable[[Any], Awaitable[Result]]:
    """Build a terminal handler that always yields a fixed value."""

    async def _terminal(_ctx: Any) -> Result:
        return value

    return _terminal

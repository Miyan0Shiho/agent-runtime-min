"""The AMBIGUOUS composition the upstream project never verified.

This is the "obvious" implementation: a listener's return value is only consulted
when it did NOT call ``next()`` *and* it returned something.  A ``None`` return
without ``next()`` falls through to "continue the chain".

It is included so that the negative test in ``test_waterfall.py`` has teeth: the
same test scenario, run against this composition, lets listener 2 run.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Sequence

from waterfall import Decision


def compose_waterfall_naive(
    listeners: Sequence[Callable[[Any, Callable[[], Awaitable[Any]]], Awaitable[Any]]],
    terminal: Callable[[Any], Awaitable[Any]],
) -> Callable[[Any], Awaitable[Any]]:
    chain = list(listeners)

    async def run_from(i: int, ctx: Any) -> Any:
        if i >= len(chain):
            return await terminal(ctx)

        fn = chain[i]
        called = False

        async def next_() -> Any:
            nonlocal called
            called = True
            return await run_from(i + 1, ctx)

        result = await fn(ctx, next_)

        # The ambiguity: no next() + no explicit Decision -> fall through.
        if not called and result is None:
            return await run_from(i + 1, ctx)
        if not called and isinstance(result, Decision):
            return result
        return result

    async def run(ctx: Any) -> Any:
        return await run_from(0, ctx)

    return run

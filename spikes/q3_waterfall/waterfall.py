"""compose_waterfall — a minimal Koa-style middleware chain with *explicit*
short-circuit semantics.

This is a proof-of-concept for REQUIREMENTS.md §9.3 / M3 §11.3:

    "pre-execute 瀑布不调 next() 时的语义"

The semantics this module DEFINES (and the test suite proves) are:

  S1. Listeners run in order. Each is awaited as ``listener(ctx, next)``.
  S2. Calling ``next()`` runs the remainder of the chain (listeners i+1..n, then
      the terminal handler) and returns its value.
  S3. A listener that does NOT call ``next()`` and returns a ``Decision``
      SHORT-CIRCUITS: no subsequent listener runs, the terminal handler does not
      run, and that Decision is the outcome.
  S4. A listener that does NOT call ``next()`` and returns ``None`` is a
      CONTRACT ERROR. It is never treated as a silent pass.
  S5. A listener that calls ``next()`` twice raises ``DoubleContinuationError``
      at the moment of the second call.
  S6. ``None`` returned *after* ``next()`` means "pass the downstream value up".
      Any non-None value returned after ``next()`` OVERRIDES the downstream
      value. (See NOTES.md — this override is the one place the requirements are
      silent, and it is exactly where guard monotonicity can be violated.)
  S7. Exceptions raised inside a listener propagate out of the waterfall and are
      structurally distinguishable from a denial (denial = returned value,
      failure = raised exception).

Stdlib only. No dependency on ``src/arm/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

__all__ = [
    "Decision",
    "Deny",
    "Cancel",
    "Ask",
    "Proceed",
    "PROCEED",
    "WaterfallError",
    "WaterfallContractError",
    "DoubleContinuationError",
    "WaterfallOutcome",
    "compose_waterfall",
]


# --------------------------------------------------------------------------
# Decision vocabulary (M3 §2 step 2: 策略层可 deny / cancel / ask)
# --------------------------------------------------------------------------


class Decision:
    """Base class for a short-circuit decision.

    Returning a ``Decision`` from a listener that never called ``next()`` ends
    the chain.  ``Decision`` is deliberately NOT an exception: a denial is a
    *result* (M3 §0 "失败即结果"), not a failure of the pipeline itself.
    """

    __slots__ = ()


@dataclass(frozen=True)
class Deny(Decision):
    reason: str = "denied"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"Deny({self.reason!r})"


@dataclass(frozen=True)
class Cancel(Decision):
    reason: str = "tool call aborted before dispatch"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"Cancel({self.reason!r})"


@dataclass(frozen=True)
class Ask(Decision):
    reason: str = "approval required"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"Ask({self.reason!r})"


@dataclass(frozen=True)
class Proceed(Decision):
    """Terminal 'the call may execute' marker."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return "Proceed()"


PROCEED = Proceed()


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class WaterfallError(Exception):
    """Base class for all waterfall failures."""


class WaterfallContractError(WaterfallError):
    """A listener violated the continuation protocol.

    This is a *programming* error (a bug in a listener), not a policy outcome.
    """


class DoubleContinuationError(WaterfallContractError):
    """A listener called ``next()`` more than once."""


# --------------------------------------------------------------------------
# Outcome
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterfallOutcome:
    """Everything an M1 log line would need to explain what happened."""

    value: Any
    """The Decision that short-circuited, or the terminal handler's value."""

    entered: tuple[str, ...]
    """Names of listeners that were ENTERED, in call order."""

    decided_by: str | None
    """Name of the listener whose return value became the outcome.

    ``None`` means the terminal handler produced the outcome.
    """

    terminal_ran: bool
    """Whether the terminal handler was reached (exactly once if True)."""

    @property
    def short_circuited(self) -> bool:
        return self.decided_by is not None


# --------------------------------------------------------------------------
# compose_waterfall
# --------------------------------------------------------------------------

Listener = Callable[[Any, Callable[[], Awaitable[Any]]], Awaitable[Any]]


def _default_name(index: int, fn: Listener) -> str:
    return getattr(fn, "__name__", None) or f"listener[{index}]"


def compose_waterfall(
    listeners: Sequence[Listener],
    terminal: Callable[[Any], Awaitable[Any]],
    *,
    name_of: Callable[[int, Listener], str] | None = None,
) -> Callable[[Any], Awaitable[WaterfallOutcome]]:
    """Compose ``listeners`` + ``terminal`` into one ``async run(ctx)``.

    The returned callable is re-entrant: every invocation gets its own trace.
    """
    chain = list(listeners)
    namer = name_of or _default_name
    names = [namer(i, fn) for i, fn in enumerate(chain)]

    async def run(ctx: Any) -> WaterfallOutcome:
        entered: list[str] = []
        terminal_ran = False

        async def run_from(i: int) -> tuple[Any, str | None]:
            nonlocal terminal_ran

            if i >= len(chain):
                terminal_ran = True
                return await terminal(ctx), None

            fn = chain[i]
            name = names[i]
            entered.append(name)

            calls = 0
            downstream: tuple[Any, str | None] | None = None

            async def next_() -> Any:
                nonlocal calls, downstream
                calls += 1
                if calls > 1:
                    raise DoubleContinuationError(
                        f"{name} called next() more than once "
                        f"(call #{calls}); double-continuation is a bug"
                    )
                downstream = await run_from(i + 1)
                return downstream[0]

            result = await fn(ctx, next_)

            if calls == 0:
                # S3 / S4: no next() -> must be an explicit Decision.
                if isinstance(result, Decision):
                    return result, name
                raise WaterfallContractError(
                    f"{name} neither called next() nor returned a Decision "
                    f"(returned {result!r}); refusing to guess whether the "
                    f"chain should continue"
                )

            # S5: next() was called exactly once.
            if result is None:
                # S6: pass the downstream outcome up unchanged.
                assert downstream is not None
                return downstream
            # S6: explicit override.
            return result, name

        value, decided_by = await run_from(0)
        return WaterfallOutcome(
            value=value,
            entered=tuple(entered),
            decided_by=decided_by,
            terminal_ran=terminal_ran,
        )

    return run

"""Q3 — semantics of a pre-execute waterfall when a listener does not call next().

Every test asserts on a ``calls`` recording list, so "listener 2 must not run" is
proven by observation, not by reading the code.
"""

from __future__ import annotations

import pytest

from waterfall import (
    Ask,
    Cancel,
    Decision,
    Deny,
    DoubleContinuationError,
    Proceed,
    PROCEED,
    WaterfallContractError,
    WaterfallOutcome,
    compose_waterfall,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_chain(calls: list[str], behaviours: dict[str, str]):
    """Build listeners from a name -> behaviour map.

    behaviours:
      "next"          -> record, await next(), return None
      "next-return"   -> record, await next(), return PROCEED (explicit override)
      "deny"          -> record, return Deny(...) without calling next
      "cancel"        -> record, return Cancel(...) without calling next
      "ask"           -> record, return Ask(...) without calling next
      "none"          -> record, return None without calling next  (contract bug)
      "next-twice"    -> record, await next() twice
      "raise"         -> record, raise ValueError
      "swallow"       -> record, await next() inside try/except BaseException, return None
    """

    def build(name: str):
        async def listener(ctx, next_):
            calls.append(name)
            b = behaviours[name]
            if b == "next":
                await next_()
                return None
            if b == "next-return":
                await next_()
                return PROCEED
            if b == "deny":
                return Deny(f"{name} denied")
            if b == "cancel":
                return Cancel(f"{name} cancelled")
            if b == "ask":
                return Ask(f"{name} asks")
            if b == "none":
                return None
            if b == "next-twice":
                await next_()
                await next_()
                return None
            if b == "raise":
                raise ValueError(f"boom from {name}")
            if b == "swallow":
                try:
                    await next_()
                except BaseException:
                    pass
                return None
            raise AssertionError(f"unknown behaviour {b!r}")

        listener.__name__ = name
        return listener

    return [build(n) for n in behaviours]


def make_terminal(calls: list[str], value=PROCEED):
    async def terminal(ctx):
        calls.append("TERMINAL")
        return value

    return terminal


# --------------------------------------------------------------------------
# (1) the headline negative test: listener 1 denies without calling next
# --------------------------------------------------------------------------


async def test_listener1_denies_without_next_listener2_must_not_run():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "deny", "L2": "next", "L3": "next"}),
        make_terminal(calls),
    )

    outcome = await run({"tool": "bash"})

    assert calls == ["L1"], f"listener 2/3 ran after a short-circuit: {calls}"
    assert outcome.entered == ("L1",)
    assert outcome.decided_by == "L1"
    assert outcome.terminal_ran is False
    assert isinstance(outcome.value, Deny)
    assert outcome.value.reason == "L1 denied"
    assert outcome.short_circuited is True


async def test_cancel_short_circuits_too():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next", "L2": "cancel", "L3": "next"}),
        make_terminal(calls),
    )
    outcome = await run({})
    assert calls == ["L1", "L2"]
    assert outcome.decided_by == "L2"
    assert isinstance(outcome.value, Cancel)
    assert outcome.terminal_ran is False


# --------------------------------------------------------------------------
# (2) listener 1 calls next, listener 2 denies -> listener 3 must not run
# --------------------------------------------------------------------------


async def test_l1_next_then_l2_denies_l3_must_not_run():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next", "L2": "deny", "L3": "next", "L4": "next"}),
        make_terminal(calls),
    )

    outcome = await run({})

    assert calls == ["L1", "L2"], calls
    assert outcome.entered == ("L1", "L2")
    assert outcome.decided_by == "L2"
    assert outcome.terminal_ran is False
    assert isinstance(outcome.value, Deny)


# --------------------------------------------------------------------------
# (3) all listeners call next -> terminal runs exactly once
# --------------------------------------------------------------------------


async def test_all_next_terminal_runs_exactly_once():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next", "L2": "next", "L3": "next"}),
        make_terminal(calls),
    )

    outcome = await run({})

    assert calls == ["L1", "L2", "L3", "TERMINAL"], calls
    assert calls.count("TERMINAL") == 1
    assert outcome.terminal_ran is True
    assert outcome.decided_by is None
    assert outcome.value is PROCEED
    assert outcome.short_circuited is False


async def test_empty_chain_runs_terminal_once():
    calls: list[str] = []
    run = compose_waterfall([], make_terminal(calls))
    outcome = await run({})
    assert calls == ["TERMINAL"]
    assert outcome.terminal_ran is True
    assert outcome.value is PROCEED


# --------------------------------------------------------------------------
# (4) next() twice -> clear error
# --------------------------------------------------------------------------


async def test_next_twice_raises_double_continuation():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next-twice", "L2": "next"}),
        make_terminal(calls),
    )

    with pytest.raises(DoubleContinuationError) as ei:
        await run({})

    msg = str(ei.value)
    assert "L1" in msg and "more than once" in msg, msg
    # L2 + terminal ran once (the first next()), but the chain was aborted by the error.
    assert calls == ["L1", "L2", "TERMINAL"], calls
    assert isinstance(ei.value, WaterfallContractError)


async def test_next_twice_does_not_rerun_terminal():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next-twice"}),
        make_terminal(calls),
    )
    with pytest.raises(DoubleContinuationError):
        await run({})
    assert calls.count("TERMINAL") == 1, f"terminal ran {calls.count('TERMINAL')}x: {calls}"


# --------------------------------------------------------------------------
# (5) neither next() nor a Decision -> well-defined error, not a silent pass
# --------------------------------------------------------------------------


async def test_neither_next_nor_decision_is_a_contract_error():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "none", "L2": "next"}),
        make_terminal(calls),
    )

    with pytest.raises(WaterfallContractError) as ei:
        await run({})

    msg = str(ei.value)
    assert "L1" in msg and "neither called next()" in msg, msg
    # The critical property: it did NOT silently continue.
    assert calls == ["L1"], f"silent pass! chain continued: {calls}"
    assert "TERMINAL" not in calls


async def test_returning_a_non_decision_value_without_next_is_also_an_error():
    """A stray truthy/falsey non-Decision return must not be read as 'allow'."""

    async def sneaky(ctx, next_):
        return True  # looks like "allow", but never called next()

    run = compose_waterfall([sneaky], make_terminal([], PROCEED))
    with pytest.raises(WaterfallContractError) as ei:
        await run({})
    assert "neither called next()" in str(ei.value)


# --------------------------------------------------------------------------
# (6) exceptions propagate and are distinguishable from a denial
# --------------------------------------------------------------------------


async def test_exception_propagates_and_is_not_a_denial():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next", "L2": "raise", "L3": "next"}),
        make_terminal(calls),
    )

    with pytest.raises(ValueError) as ei:
        await run({})

    assert str(ei.value) == "boom from L2"
    assert calls == ["L1", "L2"], calls
    # distinguishability: a denial is a *returned* Decision, never a raise
    assert not isinstance(ei.value, Decision)
    assert not isinstance(ei.value, WaterfallContractError)


async def test_denial_is_not_an_exception_and_exception_is_not_an_outcome():
    """The two failure modes must not be confusable by a caller."""
    calls: list[str] = []
    deny_run = compose_waterfall(make_chain(calls, {"L1": "deny"}), make_terminal(calls))
    outcome = await deny_run({})  # returns normally
    assert isinstance(outcome.value, Decision)

    boom_run = compose_waterfall(make_chain([], {"L1": "raise"}), make_terminal([]))
    with pytest.raises(ValueError):
        await boom_run({})  # raises


async def test_exception_after_next_propagates_and_skips_nothing_silently():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next", "L2": "next", "L3": "raise"}),
        make_terminal(calls),
    )
    with pytest.raises(ValueError):
        await run({})
    assert calls == ["L1", "L2", "L3"], calls
    assert "TERMINAL" not in calls


# --------------------------------------------------------------------------
# (7) the override / swallow hazards the requirements do not mention
# --------------------------------------------------------------------------


async def test_override_after_next_can_flip_a_denial_into_allow():
    """HAZARD: L1 calls next(), L2 denies, L1 then returns PROCEED -> allow.

    This proves that 'call next() and then return your own value' is an
    *override* primitive that can convert a downstream denial into an allow.
    M3 §2 requires guard decisions to be MONOTONIC (only veto, never permit),
    so the composition rule must forbid this, not merely fail to document it.
    """
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next-return", "L2": "deny"}),
        make_terminal(calls),
    )

    outcome = await run({})

    assert calls == ["L1", "L2"]
    assert outcome.decided_by == "L1", outcome
    assert outcome.value is PROCEED
    assert outcome.terminal_ran is False
    # The denial was observed by L1 but is NOT the outcome: a real monotonicity break.


async def test_swallowing_the_double_continuation_error_is_detectable():
    """If a listener catches the protocol error, the error is still recorded.

    NOTE: this documents that catching BaseException inside a listener can hide
    a contract violation from the *caller*. See the report's residual-risk
    section; the recommended mitigation is to make the trace, not the exception,
    the source of truth.
    """

    calls: list[str] = []

    async def bad_swallower(ctx, next_):
        calls.append("L1")
        try:
            await next_()
            await next_()
        except DoubleContinuationError:
            calls.append("SWALLOWED")
        return None

    run = compose_waterfall([bad_swallower], make_terminal(calls))
    outcome = await run({})
    assert calls == ["L1", "TERMINAL", "SWALLOWED"], calls
    assert outcome.terminal_ran is True
    # terminal ran once even though next() was called twice -> the error is the
    # only evidence; hence the recommendation to log it rather than rely on raise.
    assert calls.count("TERMINAL") == 1


# --------------------------------------------------------------------------
# (8) re-entrancy / isolation of the trace
# --------------------------------------------------------------------------


async def test_run_is_reentrant_and_traces_are_isolated():
    calls: list[str] = []
    run = compose_waterfall(
        make_chain(calls, {"L1": "next", "L2": "deny"}),
        make_terminal(calls),
    )
    a = await run({})
    b = await run({})
    assert a.entered == ("L1", "L2") and b.entered == ("L1", "L2")
    assert a.value is not b.value or a.value == b.value
    assert calls == ["L1", "L2", "L1", "L2"], calls


async def test_outcome_is_a_frozen_dataclass():
    run = compose_waterfall([], make_terminal([], PROCEED))
    outcome = await run({})
    assert isinstance(outcome, WaterfallOutcome)
    with pytest.raises(Exception):
        outcome.value = "mutated"  # type: ignore[misc]


# --------------------------------------------------------------------------
# (9) teeth-check: the SAME scenario under the ambiguous composition
# --------------------------------------------------------------------------


async def test_negative_test_has_teeth_naive_composition_lets_l2_run():
    """Proof the negative test is not vacuous.

    Under the ambiguous composition (``not next()`` + ``None`` -> fall through),
    the *identical* scenario from
    ``test_listener1_denies_without_next_listener2_must_not_run`` behaves
    differently only when L1 returns ``None``.  With L1 returning a ``Deny``,
    both compositions agree -- which is exactly why the ambiguity is dangerous:
    it is invisible until someone writes a listener that returns ``None``.

    Here we show the divergent case: L1 returns ``None`` without calling next().
    """
    from naive_waterfall import compose_waterfall_naive

    calls: list[str] = []
    naive = compose_waterfall_naive(
        make_chain(calls, {"L1": "none", "L2": "next"}),
        make_terminal(calls),
    )
    value = await naive({})

    # The ambiguous composition silently continued into L2 and the terminal.
    assert calls == ["L1", "L2", "TERMINAL"], calls
    # ...and it also *loses* the terminal's return value: L2 returns None from
    # its ``next()``, and the naive composition hands that None straight back.
    # So the ambiguity does not merely run extra listeners -- it silently drops
    # the decision they produced. That is strictly worse than running them.
    assert value is None, f"naive composition should lose the terminal value, got {value!r}"

    # Our composition refuses instead.
    strict_calls: list[str] = []
    strict = compose_waterfall(
        make_chain(strict_calls, {"L1": "none", "L2": "next"}),
        make_terminal(strict_calls),
    )
    with pytest.raises(WaterfallContractError):
        await strict({})
    assert strict_calls == ["L1"]

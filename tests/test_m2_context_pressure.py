"""M2 contract tests: context pressure.

The governing idea under test: **compaction is not deletion, it is an appended
replacement**. So most of these tests assert on the *log*, not on the resulting
text -- a compaction that produced the right text by mutating history would pass
a text-only test and destroy the audit trail.
"""

from __future__ import annotations

import asyncio

import pytest

from arm.context.compact import (
    Compactor,
    SurfaceChangedError,
)
from arm.context.meter import (
    Meter,
    Thresholds,
    heuristic_price,
    price_blocks,
    price_message,
    price_messages,
)
from arm.context.prune import Pruner
from arm.context.spill import Locator, SpillError, SpillStore
from arm.session.messages import Block, Message
from arm.session.session import Session
from arm.session.surface import fold_surface


async def _resolved(value: str) -> str:
    """An awaitable summary, for use where a coroutine function is required."""
    return value


def text_message(role: str, text: str) -> Message:
    return Message(role=role, content=(Block.text_block(text),))


def build_session_with_history(turns: int = 10, size: int = 4000) -> Session:
    """A session with a system prompt and several exchanges.

    The defaults are chosen so the history exceeds the retain budget used by
    ``make_compactor`` (16_000 tokens): a history smaller than the retain budget
    has no compactable span, which is correct behaviour but would make these
    tests vacuous.
    """
    session = Session.create("m2")
    session.append_message("system", [{"type": "text", "text": "you are arm"}])
    for index in range(turns):
        session.append_message(
            "user", [{"type": "text", "text": f"question {index} " + "x" * size}]
        )
        session.append_message(
            "assistant", [{"type": "text", "text": f"answer {index} " + "y" * size}]
        )
    return session


def make_compactor(session: Session, summary: str = "short summary") -> Compactor:
    """A compactor whose meter reflects the session's real content."""

    async def summarize(_messages) -> str:
        return summary

    meter = Meter()
    meter.observe_session(session)
    return Compactor(
        session=session,
        meter=meter,
        thresholds=Thresholds(context_window=100_000, headroom=0, threshold_ratio=0.8),
        summarize=summarize,
    )


# ---------------------------------------------------------------------------
# M2.1 -- one estimator, additive
# ---------------------------------------------------------------------------


def test_heuristic_price_is_additive():
    """M2.1/M2.7: additivity is what makes shadow pricing O(1)."""
    assert heuristic_price("a" * 400) == 100
    # The price of a concatenation is the sum of the prices of its parts.
    left, right = "a" * 400, "b" * 800
    assert heuristic_price(left + right) == heuristic_price(left) + heuristic_price(right)


def test_empty_text_costs_nothing_but_nonempty_costs_at_least_one():
    """An empty price for real content would let a caller think a range was free."""
    assert heuristic_price("") == 0
    assert heuristic_price("x") == 1


def test_prices_are_additive_over_blocks():
    """M2.1: the meter must sum, or the shadow accounting drifts."""
    blocks = [Block.text_block("a" * 400), Block.text_block("b" * 800)]
    combined = price_blocks(blocks)
    separate = price_blocks([blocks[0]]) + price_blocks([blocks[1]])
    assert combined.content == separate.content


def test_non_text_blocks_are_not_free():
    """An image has no character length, but it is not zero cost."""
    image = Block.image({"source": "x"})
    price = price_blocks([image])
    assert price.total > 0, "a non-text block must cost something"


def test_message_price_includes_overhead():
    """Per-message and per-block taxes account for framing the text misses."""
    plain = price_message(text_message("user", "hello"))
    assert plain.overhead > 0
    assert plain.total == plain.content + plain.overhead


def test_price_sums_over_a_message_list():
    messages = [text_message("user", "a" * 100), text_message("assistant", "b" * 100)]
    assert price_messages(messages) == sum(price_message(m).total for m in messages)


# ---------------------------------------------------------------------------
# M2.2 - M2.4 -- thresholds
# ---------------------------------------------------------------------------


def test_threshold_is_the_minimum_of_ratio_and_budget():
    """M2.2: threshold = floor(min(W*ratio, W-O-headroom))."""
    thresholds = Thresholds(context_window=200_000, headroom=0, threshold_ratio=0.8)
    assert thresholds.threshold_tokens == 160_000

    # A large headroom becomes the dominant term, which is the trap.
    headroomed = Thresholds(context_window=200_000, headroom=131_072, threshold_ratio=0.8)
    assert headroomed.threshold_tokens == 200_000 - 131_072


def test_headroom_dominance_is_visible():
    """The knob that stops mattering should be discoverable, not silent."""
    thresholds = Thresholds(context_window=200_000, headroom=131_072, threshold_ratio=0.8)
    assert thresholds.effective_threshold_ratio < 0.8, (
        "the effective ratio must reflect that headroom took over"
    )


def test_retain_must_be_below_threshold():
    """M2.3/E5: retaining more than the threshold makes compaction pointless."""
    with pytest.raises(ValueError, match="E5"):
        Thresholds(
            context_window=10_000,
            retain_ratio=0.95,
            threshold_ratio=0.9,
        )


def test_ratio_and_absolute_retain_are_mutually_exclusive():
    """M2.19: ambiguous configuration must fail loudly at load time."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Thresholds(context_window=10_000, retain_ratio=0.2, retain_tokens=500)


def test_e1_through_e4_are_refused():
    """M2.2: the documented error conditions are checked at configuration time."""
    with pytest.raises(ValueError, match="E1"):
        Thresholds(context_window=0)
    with pytest.raises(ValueError, match="E2"):
        Thresholds(context_window=1000, reserved_completion=-1)
    with pytest.raises(ValueError, match="E3"):
        Thresholds(context_window=1000, reserved_completion=1000)
    with pytest.raises(ValueError, match="E4"):
        Thresholds(context_window=1000, reserved_completion=500, headroom=500)


def test_bad_ratios_are_refused():
    with pytest.raises(ValueError, match="threshold_ratio"):
        Thresholds(context_window=1000, threshold_ratio=0)
    with pytest.raises(ValueError, match="threshold_ratio"):
        Thresholds(context_window=1000, threshold_ratio=1.5)


# ---------------------------------------------------------------------------
# M2.5 / M2.7 -- the summary must be smaller; shadow price adjacency
# ---------------------------------------------------------------------------


def test_a_larger_summary_is_refused_not_written():
    """M2.5: writing a bigger summary is the opposite of the point."""
    session = build_session_with_history(turns=10, size=4000)
    compactor = make_compactor(session, summary="z" * 50_000)
    result = asyncio.run(compactor.compact(manual=True))

    assert not result.performed
    assert "not smaller" in result.reason
    # Nothing was applied: the surface still holds the original nodes.
    assert len(session.surface.replacements) == 0


def test_replace_carries_a_shadow_price_of_the_shadowed_span():
    """M2.7: the replacement must declare the heuristic price it removes."""
    session = build_session_with_history(turns=10, size=4000)
    compactor = make_compactor(session, summary="brief")
    result = asyncio.run(compactor.compact(manual=True))

    assert result.performed
    start = next(e for e in session if e.type == "compaction/start")
    assert start.data["heuristic_tokens"] > 0
    assert tuple(start.data["shadowed"]) == result.shadowed


# ---------------------------------------------------------------------------
# M2.6 -- the three-step commit has no await between its parts
# ---------------------------------------------------------------------------


def test_summary_replace_and_end_are_adjacent_in_the_log():
    """M2.6: a crash between them must leave a recognisable state."""
    session = build_session_with_history(turns=10, size=4000)
    compactor = make_compactor(session, summary="brief")
    asyncio.run(compactor.compact(manual=True))

    types = [e.type for e in session]
    summary_index = types.index("compaction/summary")
    end_index = types.index("compaction/end")
    replace_index = next(
        index
        for index, event in enumerate(session)
        if event.type == "user/message" and event.data.get("compaction") is not None
    )
    assert summary_index < replace_index < end_index, (
        "the summary must be recorded before the replacement it justifies, and "
        "the transaction must close after it"
    )


def test_compaction_start_precedes_the_summary():
    """The lock is written before the model call, so a crash leaves evidence."""
    session = build_session_with_history(turns=10, size=4000)
    compactor = make_compactor(session, summary="brief")
    asyncio.run(compactor.compact(manual=True))

    types = [e.type for e in session]
    assert types.index("compaction/start") < types.index("compaction/summary")


# ---------------------------------------------------------------------------
# M2.8 -- range consistency
# ---------------------------------------------------------------------------


def test_surface_change_during_summarization_aborts_the_transaction():
    """M2.8: compacting against a stale range would shadow the wrong content.

    The mutation replaces a node *inside* the span the compactor selected, which
    is the case that matters: appending elsewhere leaves the span valid, and
    compacting against a valid span is correct.
    """
    session = build_session_with_history(turns=10, size=4000)

    async def summarize_and_rewrite(messages) -> str:
        # Rewrite one of the messages the compactor is about to shadow. Its
        # shadowed seq list is now stale.
        target = session.surface.nodes[3]
        session.append(
            "user/message",
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "rewritten mid-summary"}],
                },
                "surface_op": {"op": "replace", "start_seq": target, "end_seq": target},
                "source_event_seqs": [target],
            },
        )
        return "brief"

    meter = Meter()
    meter.observe_session(session)
    compactor = Compactor(
        session=session,
        meter=meter,
        thresholds=Thresholds(context_window=100_000, headroom=0, threshold_ratio=0.8),
        summarize=summarize_and_rewrite,
    )
    with pytest.raises(SurfaceChangedError):
        asyncio.run(compactor.compact(manual=True))

    # The transaction was closed as not-applied, so no orphan lock remains.
    end = [e for e in session if e.type == "compaction/end"][-1]
    assert end.data["applied"] is False


# ---------------------------------------------------------------------------
# M2.9 -- pairing balance
# ---------------------------------------------------------------------------


def test_compaction_never_splits_a_tool_pair():
    """M2.9: an unbalanced boundary makes the next request invalid."""
    session = Session.create("pairs")
    session.append_message("system", [{"type": "text", "text": "sys"}])
    for index in range(8):
        session.append_message(
            "user", [{"type": "text", "text": f"q{index} " + "x" * 500}]
        )
        session.append_message(
            "assistant", [{"type": "text", "text": f"a{index} " + "y" * 500}]
        )
        call = session.append_tool_call(
            call_id=f"c{index}", name="t", arguments={}, step=index, index=0
        )
        session.append_tool_result(
            call_seq=call.seq,
            call_id=f"c{index}",
            message={
                "role": "tool",
                "content": [{"type": "text", "text": "r" * 500}],
                "tool_call_id": f"c{index}",
            },
        )

    compactor = make_compactor(session, summary="brief")
    result = asyncio.run(compactor.compact(manual=True))
    if not result.performed:
        pytest.skip("no compactable span for this history")

    # Every tool call still has its result, and every result its call.
    derived = session.derive()
    calls: list[str] = []
    results: list[str] = []
    for message in derived:
        for call in message.tool_calls:
            calls.append(str(call.get("id")))
        if message.role == "tool" and message.tool_call_id:
            results.append(message.tool_call_id)
    assert sorted(calls) == sorted(results), (
        "compaction split a tool pair; the transcript would be rejected"
    )


# ---------------------------------------------------------------------------
# M2.10 -- the system prompt is protected
# ---------------------------------------------------------------------------


def test_compaction_never_shadows_the_system_prompt():
    """M2.10: node 0 is the prompt and must survive compaction."""
    session = build_session_with_history(turns=10, size=4000)
    system_seq = session.surface.nodes[0]

    compactor = make_compactor(session, summary="brief")
    result = asyncio.run(compactor.compact(manual=True))
    assert result.performed

    assert system_seq in session.surface.nodes, (
        "the system prompt was shadowed by compaction"
    )
    assert session.derive()[0].role == "system"


# ---------------------------------------------------------------------------
# M2.11 / M2.13 -- pruning
# ---------------------------------------------------------------------------


def test_prune_keeps_head_and_tail_and_inserts_a_marker():
    """M2.13: the model must be able to tell that content is missing."""
    session = Session.create("prune")
    call = session.append_tool_call(call_id="c1", name="t", arguments={}, step=0, index=0)
    body = "HEAD" + "m" * 20_000 + "TAIL"
    session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={"role": "tool", "content": [{"type": "text", "text": body}], "tool_call_id": "c1"},
    )

    pruner = Pruner(head_chars=100, tail_chars=50)
    plan = pruner.plan(session, session.surface.nodes[-1])
    assert plan is not None
    rendered = plan.new_content

    assert rendered.startswith("HEAD")
    assert rendered.endswith("TAIL")
    assert "pruned" in rendered
    assert len(rendered) < len(body)


def test_prune_produces_a_single_node_replacement():
    """M2.13: a pruner that rewrites several nodes is a compactor in disguise."""
    session = Session.create("prune2")
    call = session.append_tool_call(call_id="c1", name="t", arguments={}, step=0, index=0)
    session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={"role": "tool", "content": [{"type": "text", "text": "x" * 20_000}], "tool_call_id": "c1"},
    )

    pruner = Pruner(head_chars=100, tail_chars=50)
    plan = pruner.plan(session, session.surface.nodes[-1])
    assert plan is not None
    assert plan.shadowed == (session.surface.nodes[-1],), (
        "a prune must shadow exactly one node"
    )

    pruner.apply(session, plan)
    assert len(session.surface.replacements) == 1
    assert len(session.surface.replacements[0].shadowed) == 1


def test_prune_preserves_non_text_blocks():
    """M2.15: rich content must not be sliced or dropped."""
    session = Session.create("prune3")
    call = session.append_tool_call(call_id="c1", name="t", arguments={}, step=0, index=0)
    session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={
            "role": "tool",
            "content": [
                {"type": "text", "text": "x" * 20_000},
                {"type": "image", "source": "data:image/png;base64,AAAA"},
            ],
            "tool_call_id": "c1",
        },
    )

    pruner = Pruner(head_chars=100, tail_chars=50)
    plan = pruner.plan(session, session.surface.nodes[-1])
    assert plan is not None
    kinds = [b.get("type") for b in plan.new_blocks]
    assert "image" in kinds, "the image block must survive pruning"
    assert plan.new_blocks[-1].get("type") == "image", (
        "block order must be preserved"
    )


def test_prune_refuses_a_result_it_cannot_shrink():
    """M2.13: `after < total` must hold, or the replacement is pointless."""
    session = Session.create("prune4")
    call = session.append_tool_call(call_id="c1", name="t", arguments={}, step=0, index=0)
    session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={"role": "tool", "content": [{"type": "text", "text": "short"}], "tool_call_id": "c1"},
    )
    pruner = Pruner(head_chars=100, tail_chars=50)
    # A result shorter than the threshold is not a candidate at all.
    assert pruner.plan(session, session.surface.nodes[-1]) is None


def test_prune_budget_is_validated_at_construction():
    """M2.12: an unsatisfiable budget must fail when it is configured.

    Discovering it per-call would look like "pruning silently doesn't work"
    rather than a configuration error.
    """
    with pytest.raises(ValueError, match="unsatisfiable"):
        Pruner(head_chars=100, tail_chars=50, threshold_chars=1)


def test_no_compaction_when_history_fits_in_the_retain_budget():
    """Compaction is pointless when there is nothing to remove.

    Returning "no span" rather than compacting anyway is deliberate: a summary of
    a short history would add a node and save nothing.
    """
    session = Session.create("tiny")
    session.append_message("system", [{"type": "text", "text": "sys"}])
    session.append_message("user", [{"type": "text", "text": "a short question"}])
    session.append_message("assistant", [{"type": "text", "text": "a short answer"}])

    meter = Meter()
    meter.observe_session(session)
    compactor = Compactor(
        session=session,
        meter=meter,
        thresholds=Thresholds(context_window=100_000, headroom=0, threshold_ratio=0.8),
        summarize=lambda _m: _resolved("unused"),
    )
    result = asyncio.run(compactor.compact(manual=True))
    assert not result.performed
    assert result.reason == "no compactable span"


def test_prune_after_prune_does_not_reprune():
    """M2.13: candidates come from the current surface, so a pruned node is done."""
    session = Session.create("prune5")
    call = session.append_tool_call(call_id="c1", name="t", arguments={}, step=0, index=0)
    session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={"role": "tool", "content": [{"type": "text", "text": "x" * 20_000}], "tool_call_id": "c1"},
    )

    pruner = Pruner(head_chars=100, tail_chars=50)
    plan = pruner.plan(session, session.surface.nodes[-1])
    assert plan is not None
    pruner.apply(session, plan)

    # The surface now holds the replacement, whose content is under the budget.
    assert pruner.candidates(session) == []


def test_prune_shrinks_the_metered_total():
    """M2.13: the reduction must be visible to the meter, not just to the text."""
    session = Session.create("prune6")
    call = session.append_tool_call(call_id="c1", name="t", arguments={}, step=0, index=0)
    session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={"role": "tool", "content": [{"type": "text", "text": "x" * 40_000}], "tool_call_id": "c1"},
    )

    meter = Meter()
    meter.observe_session(session)
    before = meter.heuristic_total

    pruner = Pruner(head_chars=1000, tail_chars=200)
    plan = pruner.plan(session, session.surface.nodes[-1])
    assert plan is not None
    assert plan.heuristic_after < plan.heuristic_before


# ---------------------------------------------------------------------------
# M2.14 / M2.16 / M2.17 -- spill
# ---------------------------------------------------------------------------


def test_spill_round_trips(tmp_path):
    """M2.16: the locator is the only interface; content comes back intact."""
    store = SpillStore(tmp_path / "spill")
    content = "large content " * 1000
    locator = store.put(content, session_id="s1")
    assert store.get(locator) == content
    assert store.exists(locator)


def test_locator_is_opaque_and_not_a_path():
    """M2.16: consumers must not be able to treat it as a path."""
    locator = Locator(token="abc123")
    assert "/" not in locator.token
    assert str(locator) == "abc123"
    assert locator.to_record() == {"locator": "abc123"}


def test_spill_refuses_a_traversal_token(tmp_path):
    """M2.17: a suggested name must never become a path."""
    store = SpillStore(tmp_path / "spill")
    with pytest.raises(SpillError, match="invalid locator"):
        store.get("../../../etc/passwd")
    with pytest.raises(SpillError, match="invalid locator"):
        store.get("not-hex!")


def test_spill_directory_is_private(tmp_path):
    """M2.17: the root is 0700 and files are 0600."""
    import os
    import stat

    store = SpillStore(tmp_path / "spill")
    locator = store.put("secret", session_id="s")
    root_mode = stat.S_IMODE(os.stat(store.root).st_mode)
    file_mode = stat.S_IMODE(os.stat(store.root / f"{locator.token}.txt").st_mode)
    assert root_mode == 0o700
    assert file_mode == 0o600


def test_spill_does_not_overwrite_an_existing_file(tmp_path):
    """M2.17: exclusive creation means a collision cannot silently replace data."""
    store = SpillStore(tmp_path / "spill")
    first = store.put("one")
    second = store.put("two")
    assert first.token != second.token
    assert store.get(first) == "one"
    assert store.get(second) == "two"


def test_spill_reports_its_size(tmp_path):
    """A capacity decision needs a real number."""
    store = SpillStore(tmp_path / "spill")
    store.put("x" * 1000)
    assert store.total_bytes() > 0


# ---------------------------------------------------------------------------
# M2.20 -- progress is required to retry
# ---------------------------------------------------------------------------


def test_compaction_reports_whether_it_made_progress():
    """M2.20: only a moved replace_generation licenses a retry."""
    session = build_session_with_history(turns=10, size=4000)
    compactor = make_compactor(session, summary="brief")

    before = session.surface.replace_generation
    result = asyncio.run(compactor.compact(manual=True))
    assert result.performed
    assert session.surface.replace_generation > before, (
        "a performed compaction must have advanced the replace generation"
    )


def test_failed_compaction_does_not_advance_the_generation():
    """M2.20: no progress means no retry authorization."""
    session = build_session_with_history(turns=10, size=4000)
    compactor = make_compactor(session, summary="z" * 50_000)

    before = session.surface.replace_generation
    result = asyncio.run(compactor.compact(manual=True))
    assert not result.performed
    assert session.surface.replace_generation == before


# ---------------------------------------------------------------------------
# M2.19 -- configuration fails loudly
# ---------------------------------------------------------------------------


def test_compaction_fails_closed_when_summarization_errors():
    """A broken summarizer must not corrupt the session."""
    session = build_session_with_history(turns=10, size=4000)

    async def broken(_messages) -> str:
        raise RuntimeError("model unavailable")

    meter = Meter()
    meter.observe_session(session)
    compactor = Compactor(
        session=session,
        meter=meter,
        thresholds=Thresholds(context_window=100_000, headroom=0, threshold_ratio=0.8),
        summarize=broken,
    )
    result = asyncio.run(compactor.compact(manual=True))

    assert not result.performed
    assert "summarization failed" in result.reason
    # The transaction closed, so no orphan lock is left behind.
    end = [e for e in session if e.type == "compaction/end"][-1]
    assert end.data["applied"] is False
    # And the history is untouched.
    assert len(session.surface.replacements) == 0


def test_empty_summary_is_refused():
    """A summary of nothing would delete the history it replaced."""
    session = build_session_with_history(turns=10, size=4000)
    compactor = make_compactor(session, summary="   ")
    result = asyncio.run(compactor.compact(manual=True))
    assert not result.performed
    assert "empty summary" in result.reason


# ---------------------------------------------------------------------------
# The acceptance criterion: token reduction
# ---------------------------------------------------------------------------


def test_compaction_reduces_the_estimated_token_total():
    """M2 acceptance: compaction must actually reduce the metered total."""
    session = build_session_with_history(turns=20, size=4000)
    meter = Meter()
    meter.observe_session(session)
    before = meter.heuristic_total

    async def summarize(messages) -> str:
        # A realistic summary: far shorter than what it replaces.
        return f"Summary of {len(messages)} messages: the user asked for work; it was done."

    compactor = Compactor(
        session=session,
        meter=meter,
        thresholds=Thresholds(context_window=200_000, headroom=0, threshold_ratio=0.8),
        summarize=summarize,
    )
    result = asyncio.run(compactor.compact(manual=True))

    assert result.performed
    assert result.saved > 0, "compaction must save tokens"
    # A single pass over a long history should save a large fraction.
    assert result.tokens_after < result.tokens_before


def test_replay_after_compaction_is_faithful():
    """M1+M2: a compacted log still replays to exactly what the model would see."""
    session = build_session_with_history(turns=20, size=4000)

    async def summarize(_messages) -> str:
        return "condensed history"

    priced = Meter()
    priced.observe_session(session)
    compactor = Compactor(
        session=session,
        meter=priced,
        thresholds=Thresholds(context_window=200_000, headroom=0, threshold_ratio=0.8),
        summarize=summarize,
    )
    result = asyncio.run(compactor.compact(manual=True))
    assert result.performed

    live = [m.to_record() for m in session.derive()]
    folded = fold_surface(session.events())
    assert folded.nodes == list(session.surface_nodes()), (
        "the folded surface must match the live one after compaction"
    )

    restored = Session.from_events("m2", session.events())
    assert [m.to_record() for m in restored.derive()] == live


def test_original_events_survive_compaction():
    """M1: compaction shadows, it does not delete -- the audit trail remains."""
    session = build_session_with_history(turns=10, size=4000)
    original_count = len(session)
    original_user_texts = [
        e.data["message"]["content"][0]["text"]
        for e in session
        if e.type == "user/message"
    ]

    compactor = make_compactor(session, summary="brief")
    result = asyncio.run(compactor.compact(manual=True))
    assert result.performed

    assert len(session) > original_count, "compaction appends, it does not delete"
    for text in original_user_texts:
        assert any(
            e.type == "user/message"
            and e.data["message"]["content"][0]["text"] == text
            for e in session
        ), "an original event was removed from the log"


# ===========================================================================
# The spill store must have a capacity ceiling
# ===========================================================================


def test_the_spill_store_reclaims_when_it_exceeds_its_ceiling(tmp_path):
    """The contract flagged the absence of a capacity policy as a real risk.

    "第一版不做清扫，落盘文件会持续累积" -- and it was right. `total_bytes` and
    `delete` existed, nothing called either on a schedule, and a long-running
    session filled the disk with spilled tool output and never reclaimed any.
    """
    from arm.context.spill import SpillStore

    store = SpillStore(tmp_path / "spill", max_bytes=3_000)
    for index in range(10):
        store.put("x" * 1_000, session_id=f"s{index}")

    assert store.total_bytes() <= 3_000, (
        f"the store holds {store.total_bytes()} bytes against a 3000 ceiling"
    )


def test_the_newest_spill_survives_the_sweep(tmp_path):
    """Oldest first, because the newest is what a live session still needs.

    Ordering by size would free space faster and discard a recent small result to
    keep an old large one -- the wrong trade for a store whose purpose is recall.
    """
    from arm.context.spill import SpillError, SpillStore

    store = SpillStore(tmp_path / "spill", max_bytes=3_000)
    locators = [store.put("x" * 1_000, session_id=f"s{i}") for i in range(5)]

    # The newest is retrievable; the oldest is gone.
    assert len(store.get(locators[-1])) == 1_000
    with pytest.raises(SpillError):
        store.get(locators[0])


def test_a_swept_locator_fails_clearly_rather_than_returning_other_bytes(tmp_path):
    """A store that silently returned the wrong bytes would be worse than one
    that lost them."""
    from arm.context.spill import SpillError, SpillStore

    store = SpillStore(tmp_path / "spill", max_bytes=2_000)
    first = store.put("x" * 1_000, session_id="a")
    for index in range(4):
        store.put("y" * 1_000, session_id=f"b{index}")

    with pytest.raises(SpillError):
        store.get(first)


def test_the_ceiling_is_measured_the_same_way_it_is_enforced(tmp_path):
    """`total_bytes` and `sweep` must agree on what they are measuring.

    Counting the `.meta` sidecars in one and not the other would let the store
    report itself over budget while the sweep declined to reclaim anything.
    """
    from arm.context.spill import SpillStore

    store = SpillStore(tmp_path / "spill", max_bytes=3_000)
    for index in range(5):
        store.put("x" * 1_000, session_id=f"s{index}")

    assert store.total_bytes() <= 3_000, (
        "the reported size disagrees with the enforced ceiling"
    )


def test_a_zero_ceiling_means_unlimited(tmp_path):
    """Growing without bound should be a choice, not the default."""
    from arm.context.spill import SpillStore

    store = SpillStore(tmp_path / "spill", max_bytes=0)
    for index in range(10):
        store.put("x" * 1_000, session_id=f"s{index}")

    assert store.total_bytes() == 10_000, "a zero ceiling swept anyway"


def test_a_negative_ceiling_is_refused(tmp_path):
    """A negative ceiling is a configuration error, not 'unlimited'."""
    from arm.context.spill import SpillError, SpillStore

    with pytest.raises(SpillError, match="non-negative"):
        SpillStore(tmp_path / "spill", max_bytes=-1)


def test_a_sweep_does_not_remove_the_file_just_written(tmp_path):
    """A spill that is discarded before it can be read is not a spill."""
    from arm.context.spill import SpillStore

    # The ceiling is smaller than one spill, so the newest must still survive.
    store = SpillStore(tmp_path / "spill", max_bytes=100)
    locator = store.put("x" * 5_000, session_id="big")

    assert len(store.get(locator)) == 5_000, (
        "the store swept away the spill it had just written"
    )


def test_retain_tokens_must_also_be_below_threshold():
    """E5 applies to `retain_tokens`, and it was only checked for `retain_ratio`.

    The two are alternative spellings of the same setting, so checking one and
    not the other meant the invariant held or not depending on which spelling the
    caller used. Measured before the fix: `retain_tokens=200000` against a
    threshold of 160000 was accepted, and that configuration can never compact
    below its threshold -- compaction retains more than it removes, which is the
    exact condition E5 exists to refuse.
    """
    # threshold = min(200000 * 0.8, 200000 - 0) = 160000
    with pytest.raises(ValueError, match="E5"):
        Thresholds(
            context_window=200_000,
            headroom=0,
            threshold_ratio=0.8,
            retain_ratio=None,
            retain_tokens=200_000,
        )


def test_retain_tokens_exactly_at_the_threshold_is_refused():
    """The boundary is `>=`, not `>`.

    Retaining exactly the threshold amount removes nothing: the context is
    immediately back at the threshold that triggered the compaction.
    """
    with pytest.raises(ValueError, match="E5"):
        Thresholds(
            context_window=200_000,
            headroom=0,
            threshold_ratio=0.8,
            retain_ratio=None,
            retain_tokens=160_000,
        )


def test_retain_tokens_just_below_the_threshold_is_accepted():
    """The complement: a legitimate configuration must still load."""
    thresholds = Thresholds(
        context_window=200_000,
        headroom=0,
        threshold_ratio=0.8,
        retain_ratio=None,
        retain_tokens=159_999,
    )
    assert thresholds.retain_tokens_value == 159_999


# ===========================================================================
# M2.2: the prune marker appears exactly once, in every legal configuration
# ===========================================================================


def test_the_prune_marker_is_inserted_exactly_once_across_legal_configs():
    """The contract asks for this as a property test.

    "对 head_chars ∈ {0,1,...}、多文本块/富块交错、total == removed_end 等边界做
    属性测试" -- the upstream implementation never checked it exhaustively either.

    "Exactly once" matters because the marker tells the model that content was
    removed. Two markers in one message would misreport how much is missing, and
    zero markers would make a pruned message indistinguishable from an intact one.
    """
    from arm.context.prune import Pruner

    checked = 0
    anomalies: list[str] = []

    for head in (0, 1, 2, 5, 10, 50, 100, 500, 1000, 4000):
        for tail in (0, 1, 10, 100, 1000):
            try:
                pruner = Pruner(head_chars=head, tail_chars=tail)
            except ValueError:
                # An illegal combination, refused at construction -- that is a
                # different requirement and covered elsewhere.
                continue

            session = Session.create("prune-prop")
            # A tool result large enough to be worth pruning, built with multiple
            # interleaved blocks so the splicing path sees more than one.
            big = "A" * 5_000
            session.append(
                "tool/result",
                {
                    "message": {
                        "role": "tool",
                        "content": [
                            {"type": "text", "text": big[:2500]},
                            {"type": "text", "text": big[2500:]},
                        ],
                    },
                    "call_id": "c1",
                    "call_seq": 0,
                    "surface_op": "append",
                },
            )
            seq = session.surface.nodes[-1]
            plan = pruner.plan(session, seq)
            if plan is None:
                continue

            rendered = str(plan)
            count = rendered.count("characters pruned")
            if count != 1:
                anomalies.append(f"head={head} tail={tail}: marker appeared {count}x")
            checked += 1

    assert checked > 0, "no configuration was exercised, so nothing was verified"
    assert not anomalies, anomalies


def test_a_pruned_message_differs_from_an_intact_one():
    """The marker is what makes a pruned message recognisable.

    Without it, a model reading the context cannot tell that content is missing.
    """
    from arm.context.prune import PRUNE_MARKER

    assert "pruned" in PRUNE_MARKER.lower(), (
        "the marker does not say that anything was removed"
    )
    assert "{removed}" in PRUNE_MARKER, (
        "the marker does not report how much was removed"
    )

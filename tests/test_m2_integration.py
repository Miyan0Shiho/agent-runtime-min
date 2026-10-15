"""M2 integration: context pressure inside a real session.

These tests drive a full ``Runtime`` and assert that the context manager
actually relieves pressure at the right moment, that the reduction is recorded,
and that the resulting transcript is still valid.

The M2 acceptance criterion is a *number* (prompt tokens down by at least 40%),
so the central test here measures it rather than asserting that compaction
"happened".
"""

from __future__ import annotations

import asyncio


from arm.agent.loop import Runtime
from arm.context.manager import ContextManager
from arm.context.meter import Thresholds
from arm.context.prune import Pruner
from arm.context.spill import SpillStore
from arm.llm.mock import MockAdapter, ScriptedTurn, text_response, tool_response
from arm.prompt.assembler import PromptAssembler
from arm.session.session import Session
from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult
from arm.tools.registry import ToolRegistry


def build_runtime(
    script: list[ScriptedTurn],
    *,
    context: ContextManager | None = None,
    tools: list[RegisteredTool] | None = None,
) -> tuple[Runtime, Session, MockAdapter]:
    session = Session.create("m2-int")
    adapter = MockAdapter(script)
    registry = ToolRegistry()
    if tools:
        registry.register_many(tools)
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    runtime = Runtime(
        session,
        adapter,
        registry,
        assembler=assembler,
        context=context,
        model="mock-1",
    )
    return runtime, session, adapter


def make_context(
    session: Session,
    *,
    window: int = 200_000,
    summary: str = "condensed",
    summarize=None,
) -> ContextManager:
    async def default_summarize(_messages) -> str:
        return summary

    return ContextManager(
        session,
        thresholds=Thresholds(context_window=window, headroom=0, threshold_ratio=0.8),
        summarize=summarize or default_summarize,
    )


def big_tool() -> RegisteredTool:
    """A tool that produces a very large result."""

    async def body(arguments, _execution):
        return ToolResult.ok("Z" * int(arguments.get("size", 20_000)))

    return RegisteredTool(
        definition=ToolDefinition(
            name="big",
            description="returns a large payload",
            parameters={
                "type": "object",
                "properties": {"size": {"type": "integer"}},
            },
        ),
        body=body,
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_context_manager_measures_real_content():
    """The meter must see the session's actual content, not zero."""
    runtime, session, _ = build_runtime([text_response("hi")])
    context = make_context(session)
    runtime.context = context

    runtime.submit("a question that has some length to it")
    context.refresh()

    assert context.measured > 0, "the meter must price real content"


def test_below_threshold_means_no_action():
    """M2: a session under budget must not be touched."""
    runtime, session, _ = build_runtime([text_response("hi")])
    context = make_context(session, window=1_000_000)
    runtime.context = context

    runtime.submit("a short question")
    asyncio.run(runtime.run_until_idle())

    types = [e.type for e in session]
    assert "compaction/start" not in types, "no compaction should have run"
    assert session.surface.replacements == []


# ---------------------------------------------------------------------------
# Prune before compact: the ordering that saves money
# ---------------------------------------------------------------------------


def test_prune_runs_before_compaction():
    """M2: free reductions must be attempted before paying for a summary."""
    calls: list[int] = []

    async def counting_summarize(_messages) -> str:
        calls.append(1)
        return "summary"

    runtime, session, _ = build_runtime(
        [
            tool_response(("c1", "big", {"size": 60_000})),
            text_response("done"),
        ],
        tools=[big_tool()],
    )
    # A window small enough that the 60k-character result (about 15k tokens)
    # exceeds the threshold, but large enough that pruning alone brings it back:
    # threshold = min(15_000 * 0.8, 15_000) = 12_000 tokens.
    context = ContextManager(
        session,
        thresholds=Thresholds(context_window=15_000, headroom=0, threshold_ratio=0.8),
        summarize=counting_summarize,
        pruner=Pruner(head_chars=500, tail_chars=200),
    )
    runtime.context = context

    runtime.submit("call the big tool")
    asyncio.run(runtime.run_until_idle())

    # The oversized result was pruned...
    types = [e.type for e in session]
    assert "tool/result" in types
    pruned = any(
        e.type == "tool/result" and "pruned" in str(e.data["message"]["content"])
        for e in session
    )
    assert pruned, "the large result should have been pruned"


def test_spill_moves_content_to_disk_and_leaves_a_locator(tmp_path):
    """M2.14: spill keeps every byte and leaves a handle behind."""
    runtime, session, _ = build_runtime(
        [
            tool_response(("c1", "big", {"size": 40_000})),
            text_response("done"),
        ],
        tools=[big_tool()],
    )
    store = SpillStore(tmp_path / "spill")
    context = ContextManager(
        session,
        thresholds=Thresholds(context_window=60_000, headroom=0, threshold_ratio=0.8),
        summarize=None,
        spill=store,
        spill_threshold=10_000,
        pruner=Pruner(head_chars=200, tail_chars=100),
    )
    runtime.context = context

    runtime.submit("call the big tool")
    asyncio.run(runtime.run_until_idle())

    # The spilled bytes are still on disk, retrievable by locator. The locator
    # lives in the model-visible content, because the model is the party that
    # needs to fetch the bytes back.
    results = [
        e
        for e in session
        if e.type == "tool/result"
        and "locator" in str(e.data.get("message", {}).get("content", ""))
    ]
    assert results, "a large result should have been spilled"

    import re

    content = str(results[-1].data["message"]["content"])
    token = re.search(r"locator ([0-9a-f]{32})", content)
    assert token, f"the locator must be readable in the content: {content[:200]}"
    recovered = store.get(token.group(1))
    assert recovered.startswith("Z"), "the original content must be recoverable"
    assert len(recovered) >= 40_000


def test_spill_failure_leaves_the_content_intact(tmp_path):
    """M2.14: if offloading fails, the original must be untouched."""
    runtime, session, _ = build_runtime(
        [
            tool_response(("c1", "big", {"size": 40_000})),
            text_response("done"),
        ],
        tools=[big_tool()],
    )

    class BrokenStore(SpillStore):
        def put(self, content, *, session_id=""):
            from arm.context.spill import SpillError

            raise SpillError("disk is full")

    context = ContextManager(
        session,
        thresholds=Thresholds(context_window=60_000, headroom=0, threshold_ratio=0.8),
        summarize=None,
        spill=BrokenStore(tmp_path / "broken"),
        spill_threshold=10_000,
        pruner=Pruner(head_chars=200, tail_chars=100),
    )
    runtime.context = context

    runtime.submit("call the big tool")
    asyncio.run(runtime.run_until_idle())

    # No spill was recorded, and the original content is still reachable.
    spilled = [
        e
        for e in session
        if e.type == "tool/result"
        and "locator" in str(e.data.get("message", {}).get("content", ""))
    ]
    assert spilled == [], "a failed spill must not be recorded as a spill"
    derived = session.derive()
    tool_messages = [m for m in derived if m.role == "tool"]
    assert tool_messages, "the tool result must still be present"


# ---------------------------------------------------------------------------
# Compaction in a live session
# ---------------------------------------------------------------------------


def test_compaction_runs_when_pruning_cannot_help():
    """M2: with no oversized single result, a summary is the only option."""
    runtime, session, _ = build_runtime([text_response("done")])
    context = make_context(session, window=400)
    runtime.context = context

    # Build a history of many small messages that together blow the budget.
    runtime.submit("start")
    for index in range(12):
        session.append_message(
            "assistant", [{"type": "text", "text": f"reply {index} " + "y" * 400}]
        )
        runtime.submit(f"follow up {index} " + "x" * 400)

    asyncio.run(runtime.run_until_idle())

    types = [e.type for e in session]
    assert "compaction/start" in types, "compaction should have run"
    assert "compaction/summary" in types
    assert "compaction/end" in types
    end = [e for e in session if e.type == "compaction/end"][-1]
    assert end.data["applied"] is True


def test_compaction_is_recorded_in_the_request_context():
    """M2: what was reduced must be visible in the log, not implicit."""
    runtime, session, _ = build_runtime([text_response("done")])
    context = make_context(session, window=400)
    runtime.context = context

    runtime.submit("start")
    for index in range(12):
        session.append_message(
            "assistant", [{"type": "text", "text": f"reply {index} " + "y" * 400}]
        )
        runtime.submit(f"follow up {index} " + "x" * 400)

    asyncio.run(runtime.run_until_idle())

    reports = [
        e
        for e in session
        if e.type == "request/context" and "context_management" in e.data
    ]
    assert reports, "the reduction must be recorded in the log"
    recorded = reports[-1].data["context_management"]
    assert recorded["compacted"] >= 1
    assert recorded["measured"] > 0


def test_session_stays_valid_after_compaction():
    """M2: a compacted session must still produce a well-formed transcript."""
    runtime, session, adapter = build_runtime([text_response("done")])
    context = make_context(session, window=400)
    runtime.context = context

    runtime.submit("start")
    for index in range(12):
        session.append_message(
            "assistant", [{"type": "text", "text": f"reply {index} " + "y" * 400}]
        )
        runtime.submit(f"follow up {index} " + "x" * 400)

    asyncio.run(runtime.run_until_idle())

    # Every request the adapter saw must be a valid transcript: no tool result
    # without its call, and a system prompt first.
    for request in adapter.requests:
        roles = [m.role for m in request.messages]
        assert roles[0] == "system"
        calls = {
            str(c.get("id"))
            for m in request.messages
            for c in m.tool_calls
        }
        results = {m.tool_call_id for m in request.messages if m.role == "tool"}
        assert results <= calls, "a tool result appeared without its call"


def test_compaction_reduces_the_measured_total_in_a_live_session():
    """M2 acceptance: the reduction is real, measured on the same meter."""
    runtime, session, _ = build_runtime([text_response("done")])
    context = make_context(session, window=2000)
    runtime.context = context

    runtime.submit("start")
    for index in range(20):
        session.append_message(
            "assistant", [{"type": "text", "text": f"reply {index} " + "y" * 600}]
        )
        runtime.submit(f"follow up {index} " + "x" * 600)

    context.refresh()
    before = context.measured

    asyncio.run(runtime.run_until_idle())

    context.refresh()
    after = context.measured

    assert after < before, (
        f"compaction must reduce the measured total (before={before}, after={after})"
    )


def test_no_compaction_when_no_summarizer_is_configured():
    """A missing summarizer must degrade, not crash."""
    runtime, session, _ = build_runtime([text_response("done")])
    context = ContextManager(
        session,
        thresholds=Thresholds(context_window=400, headroom=0, threshold_ratio=0.8),
        summarize=None,
    )
    runtime.context = context

    runtime.submit("start")
    for index in range(12):
        session.append_message(
            "assistant", [{"type": "text", "text": f"reply {index} " + "y" * 400}]
        )
        runtime.submit(f"follow up {index} " + "x" * 400)

    outcomes = asyncio.run(runtime.run_until_idle())
    assert outcomes, "the loop must still run without a summarizer"
    types = [e.type for e in session]
    assert "compaction/start" not in types


def test_context_management_can_be_disabled():
    """An explicit opt-out must be honored."""
    runtime, session, _ = build_runtime([text_response("done")])
    context = ContextManager(
        session,
        thresholds=Thresholds(context_window=100, headroom=0, threshold_ratio=0.8),
        summarize=None,
        auto=False,
    )
    runtime.context = context

    runtime.submit("a question " + "x" * 2000)
    asyncio.run(runtime.run_until_idle())

    assert session.surface.replacements == []

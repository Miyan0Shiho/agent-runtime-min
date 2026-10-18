"""Tests for the summarizer the CLI hands to the context manager.

Until this existed, `build_runtime` left `runtime.context` as `None`, so the loop
never called `relieve()` and a real session could never compact. The benchmark
passed because it builds its own context manager.
"""

from __future__ import annotations

import asyncio

import pytest

from arm.context.summarize import (
    SummarizerError,
    build_summarizer,
    render_span,
)
from arm.session.messages import Message


def run(coro):
    return asyncio.run(coro)


class FakeAdapter:
    """An adapter that returns a fixed text, or raises."""

    def __init__(self, text: str = "a summary", *, raises: Exception | None = None):
        self.text = text
        self.raises = raises
        self.requests: list = []

    def stream(self, request):
        self.requests.append(request)

        async def generate():
            from arm.llm.seam import Delta

            if self.raises is not None:
                raise self.raises
            yield Delta(text=self.text)

        return generate()


# ===========================================================================
# Rendering the span
# ===========================================================================


def test_the_span_is_rendered_as_a_flat_transcript():
    """The summarizer describes the conversation rather than continuing it.

    A flat transcript is easier to summarise accurately than a tool-call schema
    it would have to re-interpret.
    """
    span = [
        Message.user("do the thing"),
        Message.assistant("done"),
    ]
    text = render_span(span)

    assert "[user] do the thing" in text
    assert "[assistant] done" in text


def test_tool_calls_appear_in_the_span():
    """A summary that omits what the agent *did* has lost the important part."""
    span = [
        Message.assistant(tool_calls=({"name": "bash", "arguments": {"command": "ls"}},)),
    ]
    text = render_span(span)

    assert "bash" in text
    assert "ls" in text


def test_empty_messages_are_skipped():
    """An empty message carries a surface position, not content.

    Including it would tell the summarizer nothing while making the prompt
    longer -- and compaction exists to make the prompt shorter.
    """
    span = [Message.system("real content"), Message.assistant("")]

    text = render_span(span)

    assert "real content" in text
    assert "[assistant]" not in text


# ===========================================================================
# The summarizer
# ===========================================================================


def test_a_summary_is_returned():
    adapter = FakeAdapter("the user asked for X; Y was established")
    summarize = build_summarizer(adapter, model="m")

    result = run(summarize([Message.user("do X")]))

    assert result == "the user asked for X; Y was established"
    assert adapter.requests, "no request was sent"


def test_no_tools_are_offered_to_the_summarizer():
    """A summarizer that could act would be a second agent inside compaction.

    Every call it made would need approval and sandbox treatment, during the one
    operation where a failure is most expensive.
    """
    adapter = FakeAdapter()
    summarize = build_summarizer(adapter, model="m")

    run(summarize([Message.user("do X")]))

    assert adapter.requests[0].tools == (), "tools were offered to the summarizer"


def test_the_instruction_is_included():
    """The section structure was earned through use; omitting it loses that."""
    from arm.context.compact import SUMMARY_INSTRUCTION

    adapter = FakeAdapter()
    summarize = build_summarizer(adapter, model="m")

    run(summarize([Message.user("do X")]))

    sent = adapter.requests[0].messages[0].text
    assert SUMMARY_INSTRUCTION.splitlines()[0] in sent


def test_an_empty_span_needs_no_model_call():
    """Nothing to summarise is not a failure.

    Raising here would turn "the span was empty" into "compaction is broken".
    """
    adapter = FakeAdapter()
    summarize = build_summarizer(adapter, model="m")

    result = run(summarize([]))

    assert result == ""
    assert not adapter.requests, "the model was called for an empty span"


def test_an_empty_summary_from_a_real_span_is_an_error():
    """The worst thing compaction can do is replace history with nothing.

    An empty summary would let the model carry on as if the earlier work had
    never happened -- and nothing in the transcript would say so.
    """
    adapter = FakeAdapter("")
    summarize = build_summarizer(adapter, model="m")

    with pytest.raises(SummarizerError) as info:
        run(summarize([Message.user("do X")]))
    assert info.value.code == "SUMMARIZE_EMPTY"


def test_a_provider_failure_is_reported_not_swallowed():
    """A fabricated summary in place of real history is worse than a refusal."""
    adapter = FakeAdapter(raises=RuntimeError("provider exploded"))
    summarize = build_summarizer(adapter, model="m")

    with pytest.raises(SummarizerError) as info:
        run(summarize([Message.user("do X")]))
    assert info.value.code == "SUMMARIZE_PROVIDER_ERROR"
    assert "provider exploded" in str(info.value)


def test_a_timeout_is_reported():
    adapter = FakeAdapter()
    summarize = build_summarizer(adapter, model="m", timeout=0.0)

    with pytest.raises(SummarizerError) as info:
        run(summarize([Message.user("do X")]))
    assert info.value.code == "SUMMARIZE_TIMEOUT"


def test_the_summary_is_length_capped():
    """Compaction exists to reduce tokens.

    A summary longer than the span it replaces would make the problem worse while
    looking like progress.
    """
    adapter = FakeAdapter()
    summarize = build_summarizer(adapter, model="m", max_tokens=123)

    run(summarize([Message.user("do X")]))

    assert adapter.requests[0].max_tokens == 123


# ===========================================================================
# The CLI wires it
# ===========================================================================


def test_the_cli_builds_a_runtime_that_can_compact():
    """The defect this module closes.

    Measured before: 107 requests in a real session, zero context-management
    records, zero compactions -- while M2 passed all four acceptance metrics in
    the benchmark, because the benchmark builds its own context manager.
    """
    from pathlib import Path
    import tempfile

    from arm.cli.main import build_runtime

    home = Path(tempfile.mkdtemp())
    runtime, _session, _handle, _scheduler, _mcp = build_runtime(
        session_id="ctx",
        model="m",
        api_key="k",
        base_url="http://127.0.0.1:1",
        home=str(home),
        persist=False,
    )

    assert runtime.context is not None, (
        "the runtime has no context manager, so the loop can never relieve pressure"
    )
    assert runtime.context.thresholds.threshold_tokens > 0
    assert runtime.context._summarize is not None, (
        "no summarizer, so compaction would be attempted and refused"
    )

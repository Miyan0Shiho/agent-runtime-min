"""The summarizer the CLI hands to the context manager.

`ContextManager` needs a callable that turns a span of messages into a summary.
Until now nothing in the CLI constructed one -- so `build_runtime` left
`runtime.context` as `None`, and the loop never called `relieve()`.

Measured in a real 32-turn session: **107 requests, zero context-management
records, zero compactions.** M2 was fully implemented, had four acceptance
metrics, and passed all of them in the benchmark -- while being unreachable from
the command line. This is defect class A1 (a mechanism with no caller) in the
largest subsystem, and it is why the module exists rather than the context manager
being built inline: the summarizer is the part that talks to a model, so it wants
its own tests and its own failure behaviour.

Three decisions, each a trade:

**It uses the session's own adapter and model.** A separate summarizer model would
be better in principle -- summarization is a different task, and a cheap model may
do it as well. But a second model means a second credential and a second failure
mode, and this runtime is meant to be usable with one. The adapter is already
there.

**No tools are offered.** A summarizer that could call tools would be a second
agent in the middle of compaction, and every one of its calls would need the
approval and sandbox treatment any other call gets -- during compaction, where a
failure is expensive. It reads and writes prose.

**A failure is reported, not swallowed.** If the provider errors, the summarizer
raises, and `relieve` reports that compaction could not be done. The alternative
-- returning an empty or apologetic summary -- would put a fabricated message into
the surface in place of real history, which is the worst thing a compaction can
do.

Why not use the assistant's reasoning text: there is none recorded, and inventing
one would be summarising something that never happened.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from ..kernel.errors import ArmError, describe
from ..llm.seam import Request, StreamCollector
from ..session.messages import Block, Message
from .compact import SUMMARY_INSTRUCTION, Summarizer

log = logging.getLogger("arm.context.summarize")

#: The summary is capped so a runaway model cannot replace a large span with an
#: even larger one. Compaction exists to reduce tokens; a summary longer than the
#: span it replaces would make the problem worse while looking like progress.
MAX_SUMMARY_TOKENS = 2_000


class SummarizerError(ArmError):
    """The span could not be summarised."""

    def __init__(self, message: str, *, code: str = "SUMMARIZE_FAILED") -> None:
        super().__init__(message)
        self.code = code


def render_span(messages: Sequence[Message]) -> str:
    """The span as text for the summariser to read.

    Rendered rather than passed as structured messages: the summariser is not
    continuing the conversation, it is describing it, and a flat transcript is
    easier for it to summarise accurately than a tool-call schema it must
    re-interpret.
    """
    lines: list[str] = []
    for message in messages:
        text = message.text.strip()
        if not text and not message.tool_calls:
            # An empty message carries a surface position, not content. Including
            # it would tell the summariser nothing while making the prompt longer.
            continue
        lines.append(f"[{message.role}] {text}")
        for call in message.tool_calls:
            lines.append(f"[{message.role} -> tool {call.get('name', '?')}] {call.get('arguments', '')}")
    return "\n".join(lines)


def build_summarizer(
    adapter: Any,
    *,
    model: str = "",
    max_tokens: int = MAX_SUMMARY_TOKENS,
    timeout: float = 120.0,
) -> Summarizer:
    """A summarizer backed by the given adapter."""

    async def summarize(messages: Sequence[Message]) -> str:
        span = render_span(messages)
        if not span.strip():
            # Nothing to summarise. An empty summary is correct here and the
            # compactor handles it; raising would turn "the span was empty" into
            # "compaction is broken".
            return ""

        request = Request(
            messages=(
                Message(
                    role="user",
                    content=(Block.text_block(f"{span}\n\n---\n\n{SUMMARY_INSTRUCTION}"),),
                ),
            ),
            # No tools. See the module docstring: a summarizer that could act
            # would need the whole policy stack inside compaction.
            tools=(),
            model=model,
            max_tokens=max_tokens,
        )

        collector = StreamCollector()

        async def consume() -> None:
            async for delta in adapter.stream(request):
                collector.add(delta)

        try:
            await asyncio.wait_for(consume(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise SummarizerError(
                f"summarising {len(messages)} messages took longer than {timeout}s",
                code="SUMMARIZE_TIMEOUT",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - a provider fault is a summary failure
            raise SummarizerError(
                f"the summary could not be produced: {describe(exc)}",
                code="SUMMARIZE_PROVIDER_ERROR",
            ) from exc

        result = collector.result()
        text = (result.text or "").strip()
        if not text:
            # An empty summary from a non-empty span is a failure: the compactor
            # would replace real history with nothing, and the model would carry
            # on as if the earlier work had never happened.
            raise SummarizerError(
                f"the provider returned no summary for {len(messages)} messages",
                code="SUMMARIZE_EMPTY",
            )
        return text

    return summarize

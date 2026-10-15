"""A scripted model adapter: deterministic, no network, fully inspectable.

Tests need a model that behaves exactly as specified. This adapter plays a
prepared script of responses and records every request it received, which makes
it possible to assert on *what was sent* -- the thing M1's acceptance criterion is
actually about -- rather than only on what came back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Sequence

from arm.llm.seam import (
    Delta,
    ModelAdapter,
    Request,
    StreamResult,
    ToolCallRequest,
    Usage,
)


@dataclass
class ScriptedTurn:
    """One prepared response."""

    text: str = ""
    tool_calls: tuple[ToolCallRequest, ...] = ()
    finish_reason: str = "stop"
    usage: Usage | None = None
    #: When set, the adapter raises this instead of responding.
    error: Exception | None = None
    #: Number of deltas to split ``text`` into, to exercise streaming.
    chunks: int = 1


class MockAdapter(ModelAdapter):
    """Plays a script, then repeats the final turn forever.

    Repeating rather than failing on exhaustion is deliberate: it means a test
    that accidentally runs an extra step produces a clear assertion failure about
    the transcript, not a confusing IndexError from inside the mock.
    """

    def __init__(self, script: Sequence[ScriptedTurn], *, model: str = "mock-1") -> None:
        self._script = list(script)
        self._model = model
        self.requests: list[Request] = []
        self.call_count = 0

    @property
    def model(self) -> str:
        return self._model

    def _next(self) -> ScriptedTurn:
        if not self._script:
            return ScriptedTurn(text="(script exhausted)")
        index = min(self.call_count, len(self._script) - 1)
        return self._script[index]

    def stream(self, request: Request) -> AsyncIterator[Delta]:
        self.requests.append(request)
        turn = self._next()
        self.call_count += 1

        async def _generate() -> AsyncIterator[Delta]:
            if turn.error is not None:
                raise turn.error
            if turn.text:
                size = max(1, len(turn.text) // max(1, turn.chunks))
                for start in range(0, len(turn.text), size):
                    yield Delta(text=turn.text[start : start + size])
            for call in turn.tool_calls:
                yield Delta(tool_call=call)

        return _generate()

    async def complete(self, request: Request) -> StreamResult:
        from arm.llm.seam import collect

        return await collect(self, request)


def text_response(text: str, *, chunks: int = 1) -> ScriptedTurn:
    return ScriptedTurn(text=text, chunks=chunks)


def tool_response(*calls: tuple[str, str, Mapping[str, Any]]) -> ScriptedTurn:
    """Build a response that requests tool calls.

    ``calls`` are ``(call_id, name, arguments)`` triples.
    """
    return ScriptedTurn(
        tool_calls=tuple(
            ToolCallRequest(id=cid, name=name, arguments=dict(args))
            for cid, name, args in calls
        ),
        finish_reason="tool_calls",
    )

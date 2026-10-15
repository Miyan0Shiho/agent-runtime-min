"""The model seam: one narrow interface, many adapters.

The loop must not know which model it is talking to. It knows how to hand over a
prepared request and how to consume a stream of deltas. Everything provider
specific -- retries, wire formats, tool-call encodings -- lives behind this seam.

Two decisions worth stating:

* **Streaming is the only mode.** A non-streaming path would be a second code
  path through the loop, and the second path is always the one that rots.
* **Errors are returned, not raised.** A model that fails is an ordinary
  outcome the loop records; only a broken *seam* (a bug in the adapter) raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Mapping, Protocol

from ..session.messages import Message

#: Why a stream ended.
FinishReason = Literal["stop", "tool_calls", "length", "error", "aborted"]


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool as the model sees it: name, description, JSON Schema parameters."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Request:
    """A fully prepared model call.

    Built from the derived message list plus the current tool set. The request is
    frozen before it is sent, so what was sent cannot drift from what was
    recorded.
    """

    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...] = ()
    model: str = ""
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    #: Provider-specific knobs, passed through untouched.
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """A tool call the model asked for."""

    id: str
    name: str
    arguments: Mapping[str, Any]
    #: Raw argument text, kept for diagnostics when parsing failed.
    raw_arguments: str = ""


@dataclass(frozen=True, slots=True)
class Delta:
    """One incremental piece of a streamed response.

    Text arrives in ``text``; a completed tool call arrives in ``tool_call``.
    Both may appear in one stream, in any order.
    """

    text: str = ""
    tool_call: ToolCallRequest | None = None
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting as reported by the provider.

    ``None`` means "not reported" and must stay distinguishable from zero: a
    provider that does not count is not a provider that counted nothing.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StreamResult:
    """The terminal state of one model call."""

    finish_reason: FinishReason
    text: str = ""
    tool_calls: tuple[ToolCallRequest, ...] = ()
    usage: Usage | None = None
    #: Populated when ``finish_reason`` is ``"error"``.
    error: str | None = None
    #: Provider's error code, for diagnostics. Never sent to the model.
    error_code: str | None = None


class ModelAdapter(Protocol):
    """What the loop requires of a model provider."""

    @property
    def model(self) -> str:
        """Identifier of the model this adapter targets."""
        ...

    def stream(self, request: Request) -> AsyncIterator[Delta]:
        """Stream deltas for ``request``.

        Declared as a plain function returning an async iterator (not an
        ``async def``): an async generator is *called* synchronously and then
        iterated, so ``async def`` here would describe a coroutine that resolves
        to an iterator -- a shape callers would have to await twice.

        Implementations must yield at least one item and must not raise for
        ordinary provider failures -- those are reported through the returned
        :class:`StreamResult` assembled by the caller.
        """
        ...

    async def complete(self, request: Request) -> StreamResult:
        """Run a full call, collecting the stream into a result."""
        ...


class StreamCollector:
    """Accumulate deltas into a :class:`StreamResult`.

    Kept separate from adapters so that every provider assembles results the
    same way -- if two adapters collected differently, the loop would see two
    subtly different notions of "what the model said".
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._calls: list[ToolCallRequest] = []
        self._usage: Usage | None = None
        self._finish: FinishReason | None = None
        self._error: str | None = None
        self._error_code: str | None = None

    def add(self, delta: Delta) -> None:
        if delta.text:
            self._text.append(delta.text)
        if delta.reasoning:
            self._reasoning.append(delta.reasoning)
        if delta.tool_call is not None:
            self._calls.append(delta.tool_call)

    def set_usage(self, usage: Usage | None) -> None:
        self._usage = usage

    def set_finish(self, reason: FinishReason) -> None:
        self._finish = reason

    def set_error(self, message: str, code: str | None = None) -> None:
        self._error = message
        self._error_code = code
        self._finish = "error"

    @property
    def text(self) -> str:
        return "".join(self._text)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning)

    def result(self) -> StreamResult:
        finish: FinishReason = self._finish or ("tool_calls" if self._calls else "stop")
        return StreamResult(
            finish_reason=finish,
            text=self.text,
            tool_calls=tuple(self._calls),
            usage=self._usage,
            error=self._error,
            error_code=self._error_code,
        )


async def collect(adapter: ModelAdapter, request: Request) -> StreamResult:
    """Drain ``adapter``'s stream into a result, containing provider failures.

    A raising adapter is converted into an ``error`` result rather than
    propagating: from the loop's perspective, "the provider blew up" and "the
    provider said no" are the same kind of event -- something to record and move
    past (S14).
    """
    collector = StreamCollector()
    try:
        async for delta in adapter.stream(request):
            collector.add(delta)
    except Exception as exc:  # noqa: BLE001 - provider failure is an outcome
        from ..kernel.errors import describe

        collector.set_error(describe(exc), type(exc).__name__)
    return collector.result()

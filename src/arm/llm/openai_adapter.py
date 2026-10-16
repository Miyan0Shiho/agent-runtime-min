"""An OpenAI-compatible chat adapter over HTTP.

Written against the wire format directly rather than through an SDK, for two
reasons: the runtime should not inherit a vendor's retry and error semantics, and
the streaming parser is the part most likely to need debugging -- having it in
this repository means it can be read.

The adapter converts every transport and protocol failure into an ``error``
result. It never raises for a provider problem: from the loop's perspective, "the
provider returned a 500" and "the provider refused" are the same kind of event.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Mapping

from ..session.messages import Message
from .seam import (
    Usage,
    Delta,
    Request,
    StreamResult,
    ToolCallRequest,
    collect,
)

log = logging.getLogger("arm.llm.openai")

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleAdapter:
    """A streaming chat adapter for OpenAI-compatible endpoints.

    ``base_url`` may point at any compatible server (a local model, a gateway, a
    proxy). Nothing here assumes OpenAI specifically beyond the wire format.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str,
        timeout: float = 120.0,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._extra_headers = dict(extra_headers or {})
        self._extra_body = dict(extra_body or {})

    @property
    def model(self) -> str:
        return self._model

    # -- wire conversion -------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        headers.update(self._extra_headers)
        return headers

    #: Internal roles that are not universal on the wire.
    #:
    #: `developer` is this runtime's name for "instructions the system added
    #: later" -- the tool catalogue. OpenAI accepts the role; most
    #: OpenAI-compatible providers do not, and DeepSeek rejects the whole request:
    #:
    #:     messages[2].role: unknown variant `developer`, expected one of
    #:     `system`, `user`, `assistant`, `tool`
    #:
    #: Measured by running the runtime against a real provider, after every turn
    #: failed with an error whose only diagnostic lived in the session log.
    #:
    #: The role is mapped rather than the internal vocabulary changed, because the
    #: distinction is real (M1.13 names it) and this is where provider differences
    #: belong. The mapping is **forward-only and lossy in reverse** -- both
    #: `system` and `developer` become `system` -- which is fine for M1, whose
    #: claim is that the log reconstructs the request, not that the request
    #: reconstructs the log.
    _WIRE_ROLES = {"developer": "system"}

    def _render_message(self, message: Message) -> dict[str, Any]:
        """Convert a derived message into the wire shape."""
        role = self._WIRE_ROLES.get(message.role, message.role)
        payload: dict[str, Any] = {"role": role}

        text = message.text
        if message.role == "tool":
            payload["content"] = text
            if message.tool_call_id:
                payload["tool_call_id"] = message.tool_call_id
            if message.name:
                payload["name"] = message.name
            return payload

        if message.tool_calls:
            # An assistant message that requested tools. Some providers reject an
            # empty string here, so omit the key rather than sending "".
            if text:
                payload["content"] = text
            else:
                payload["content"] = None
            payload["tool_calls"] = [
                {
                    "id": call.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": call.get("name", ""),
                        "arguments": json.dumps(
                            call.get("arguments", {}), ensure_ascii=False
                        ),
                    },
                }
                for call in message.tool_calls
            ]
            return payload

        payload["content"] = text
        return payload

    def _render_request(self, request: Request) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            messages.append(self._render_message(message))

        body: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in request.tools
            ]
            body["tool_choice"] = "auto"
        body.update(self._extra_body)
        return body

    # -- streaming -------------------------------------------------------

    def stream(self, request: Request) -> AsyncIterator[Delta]:
        body = self._render_request(request)

        async def _generate() -> AsyncIterator[Delta]:
            import httpx

            # Tool calls arrive as fragments across chunks; index them so the
            # pieces can be reassembled into whole calls.
            partial: dict[int, dict[str, str]] = {}

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                ) as response:
                    if response.status_code >= 400:
                        text = (await response.aread()).decode("utf-8", "replace")
                        raise ProviderError(
                            f"HTTP {response.status_code}: {text[:500]}",
                            code=f"HTTP_{response.status_code}",
                        )

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            log.warning("skipping unparseable stream chunk")
                            continue
                        for delta in _deltas_from_chunk(chunk, partial):
                            yield delta

            # Emit any tool call that was still accumulating.
            for index in sorted(partial):
                call = partial[index]
                if call.get("name"):
                    yield Delta(
                        tool_call=ToolCallRequest(
                            id=call.get("id") or f"call_{index}",
                            name=call["name"],
                            arguments=_parse_arguments(call.get("arguments", "")),
                            raw_arguments=call.get("arguments", ""),
                        )
                    )

        return _generate()

    async def complete(self, request: Request) -> StreamResult:
        return await collect(self, request)


class ProviderError(RuntimeError):
    """A provider-side failure. Converted into an error result by ``collect``."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _optional_int(value: Any) -> int | None:
    """An int, or ``None`` when the provider omitted the field.

    ``None`` rather than zero: "not reported" and "zero tokens" are different
    facts, and a zero would make an unreported count look like a measured one.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _deltas_from_chunk(
    chunk: Mapping[str, Any], partial: dict[int, dict[str, str]]
) -> list[Delta]:
    """Convert one streamed chunk into zero or more deltas.

    Tool-call arguments arrive in fragments and must be concatenated by index.
    Emitting them as separate deltas would produce several partial calls instead
    of one complete call, which the pipeline would then reject as invalid
    arguments.
    """
    out: list[Delta] = []

    # The provider reports token usage on the final chunk, and only because the
    # request asked for it with `stream_options: {include_usage: true}`. Parsing
    # it is what makes that request mean anything: the field was being requested
    # and then discarded, so the runtime recorded no token counts at all.
    usage = chunk.get("usage")
    if isinstance(usage, Mapping):
        parsed = Usage(
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
        )
        out.append(Delta(usage=parsed))

    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}

        # The finish reason names why the model stopped. It is what lets a
        # truncated answer (`length`) be distinguished from a finished one, and
        # without it every response looked complete.
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            out.append(Delta(finish_reason=reason))

        text = delta.get("content")
        if isinstance(text, str) and text:
            out.append(Delta(text=text))

        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            out.append(Delta(reasoning=reasoning))

        for call in delta.get("tool_calls") or []:
            index = int(call.get("index", 0))
            slot = partial.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if call.get("id"):
                slot["id"] = str(call["id"])
            function = call.get("function") or {}
            if function.get("name"):
                slot["name"] = str(function["name"])
            fragment = function.get("arguments")
            if isinstance(fragment, str):
                slot["arguments"] += fragment
    return out


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool-call argument string, tolerating the empty case.

    A malformed payload is returned as a single ``_raw`` key rather than raising:
    the pipeline turns it into an "invalid arguments" result, which is the
    documented behaviour for a model that emitted unparseable JSON.
    """
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"_raw": raw}

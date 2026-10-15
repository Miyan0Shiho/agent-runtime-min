"""The browser seam: driving a real Chrome over CDP.

This is an **independent seam**, not part of the main chain. It attaches to the
runtime the same way any other tool does -- it registers tools and nothing else --
so a runtime that never opens a browser pays nothing for its existence.

The decision that shapes everything here is **CDP against the installed Chrome**,
rather than a bundled headless browser. The reason is not convenience:

* A bundled browser has its own cookie jar, so it is a **logged-out stranger** to
  every site the user is signed into. Sites behind a bot check then serve it a
  challenge page, and a headless browser cannot pass one.
* The user's Chrome is already authenticated, already past whatever checks the
  site applies, and already holds the session. Driving it means the agent works
  on the pages the user can actually see.

The cost is that the browser must be running with a debugging port, which is a
deliberate, visible act. That is the right trade: an agent that silently attaches
to the user's logged-in browser would be far more dangerous than one the user had
to launch a certain way.

Two rules follow from that, and both are enforced rather than documented:

**Nothing is attached implicitly.** Connecting requires an explicit endpoint. A
runtime that guessed a port and found a browser would be reaching into a session
the user did not offer it.

**Every action is an ordinary tool call.** Browser tools go through the same
pipeline, so approval and the sandbox apply to them exactly as they do to
``bash``. There is no browser-specific permission path, because a second path is
a second thing to get wrong.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..kernel.errors import ArmError, describe

log = logging.getLogger("arm.browser")

#: Where Chrome's debugging endpoint lives by default. Used only when a caller
#: asks for the default explicitly -- never discovered by scanning.
DEFAULT_ENDPOINT = "http://127.0.0.1:9222"

#: How long a single CDP command may take.
DEFAULT_TIMEOUT = 30.0

#: A page can produce a lot of text. Bounded so one page cannot consume the
#: whole context budget.
MAX_TEXT_CHARS = 40_000


class BrowserError(ArmError):
    """The browser could not be driven."""

    def __init__(self, message: str, *, code: str = "BROWSER_ERROR") -> None:
        super().__init__(message)
        self.code = code


class BrowserUnavailableError(BrowserError):
    """No browser is reachable at the given endpoint."""

    def __init__(self, endpoint: str, reason: str = "") -> None:
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"no browser is listening at {endpoint}{detail}. Start one with "
            "--remote-debugging-port, or point ARM_BROWSER at an existing "
            "endpoint.",
            code="BROWSER_UNAVAILABLE",
        )


@dataclass(frozen=True, slots=True)
class PageTarget:
    """One open page."""

    id: str
    title: str
    url: str
    websocket_url: str

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
        }


class CDPConnection:
    """A JSON-RPC connection to one CDP target.

    Speaks the same shape as any other RPC in this runtime: send a method, await
    the response with the matching id, ignore everything else. CDP also emits
    events with no id, and treating the first message as the response would break
    on any page that fires an event while a command is in flight.
    """

    def __init__(self, websocket_url: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.websocket_url = websocket_url
        self.timeout = timeout
        self._socket: Any = None
        self._next_id = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the socket.

        Imported lazily so a runtime that never drives a browser does not need
        the WebSocket library installed at all -- the browser extra is optional,
        and an unconditional import would make it mandatory.
        """
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise BrowserError(
                "the browser seam needs a WebSocket client; install it with "
                "'uv pip install \"websockets>=13.0\"'",
                code="BROWSER_MISSING_DEPENDENCY",
            ) from exc

        try:
            # A generous frame limit: a full-page screenshot arrives as one
            # base64 message, and the default limit truncates it.
            self._socket = await websockets.connect(
                self.websocket_url, max_size=64 * 1024 * 1024
            )
        except Exception as exc:  # noqa: BLE001 - a failed connect is a state
            raise BrowserUnavailableError(self.websocket_url, describe(exc)) from exc

    async def close(self) -> None:
        if self._socket is not None:
            try:
                await self._socket.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("closing the CDP socket failed: %s", exc)
            self._socket = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    async def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        """Send one command and await its result."""
        if self._socket is None:
            raise BrowserError("the CDP connection is not open", code="BROWSER_NOT_CONNECTED")

        async with self._lock:
            self._next_id += 1
            request_id = self._next_id
            payload = json.dumps(
                {"id": request_id, "method": method, "params": dict(params or {})}
            )

            try:
                await self._socket.send(payload)
            except Exception as exc:  # noqa: BLE001
                raise BrowserError(
                    f"sending {method} failed: {describe(exc)}",
                    code="BROWSER_SEND_FAILED",
                ) from exc

            try:
                return await asyncio.wait_for(self._read_response(request_id), self.timeout)
            except asyncio.TimeoutError:
                raise BrowserError(
                    f"{method} did not answer within {self.timeout}s",
                    code="BROWSER_TIMEOUT",
                ) from None

    async def _read_response(self, request_id: int) -> Mapping[str, Any]:
        """Read until the response with our id arrives.

        Events arrive interleaved and carry no id, so they are skipped rather
        than mistaken for the answer.
        """
        while True:
            raw = await self._socket.recv()
            try:
                message = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(message, Mapping):
                continue
            if message.get("id") != request_id:
                continue
            error = message.get("error")
            if error is not None:
                if isinstance(error, Mapping):
                    text = str(error.get("message", "the browser reported an error"))
                else:
                    text = str(error)
                raise BrowserError(text, code="BROWSER_COMMAND_FAILED")
            result = message.get("result")
            return result if isinstance(result, Mapping) else {}


def _exception_text(result: Mapping[str, Any]) -> str:
    """The useful part of a page-side exception.

    ``exceptionDetails.text`` is only ever the category -- literally "Uncaught" --
    while the message and stack live on the nested ``exception.description``.
    Reading ``text`` first therefore produces an error that says a script failed
    and nothing about why, which is barely better than no message at all.
    """
    detail = result.get("exceptionDetails")
    if not isinstance(detail, Mapping):
        return "unknown"

    exception = detail.get("exception")
    if isinstance(exception, Mapping):
        description = exception.get("description")
        if description:
            return str(description)
        value = exception.get("value")
        if value is not None:
            return str(value)

    # A thrown non-Error (a bare `throw "x"`) has no `exception` object; the
    # message is on the result instead.
    inner = result.get("result")
    if isinstance(inner, Mapping) and inner.get("description"):
        return str(inner["description"])

    return str(detail.get("text") or "unknown")


class Browser:
    """A browser, addressed by an explicit endpoint.

    The endpoint is required. Nothing is discovered by scanning ports, because a
    runtime that found a browser on its own would be reaching into a session the
    user never offered it -- and that session is logged in to things.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        http: Any = None,
    ) -> None:
        if not endpoint:
            raise BrowserError(
                "a browser needs an explicit endpoint; guessing one would attach "
                "to a session the user did not offer",
                code="BROWSER_NO_ENDPOINT",
            )
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._http = http
        self._connection: CDPConnection | None = None
        self._target: PageTarget | None = None

    # -- connecting ------------------------------------------------------

    async def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    async def list_targets(self) -> tuple[PageTarget, ...]:
        """The open pages, so a caller can choose one."""
        client = await self._client()
        try:
            response = await client.get(f"{self.endpoint}/json/list")
        except Exception as exc:  # noqa: BLE001 - an unreachable browser is a state
            raise BrowserUnavailableError(self.endpoint, describe(exc)) from exc

        if response.status_code != 200:
            raise BrowserUnavailableError(
                self.endpoint, f"the endpoint answered HTTP {response.status_code}"
            )

        try:
            raw = response.json()
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(
                f"the browser returned a malformed target list: {describe(exc)}",
                code="BROWSER_BAD_TARGETS",
            ) from exc

        if not isinstance(raw, Sequence):
            raise BrowserError(
                "the browser returned a malformed target list", code="BROWSER_BAD_TARGETS"
            )

        targets: list[PageTarget] = []
        for entry in raw:
            if not isinstance(entry, Mapping) or entry.get("type") != "page":
                continue
            url = str(entry.get("webSocketDebuggerUrl", ""))
            if not url:
                # A target without a socket cannot be driven. Skipping it is
                # honest; including it would produce a page that always fails.
                continue
            targets.append(
                PageTarget(
                    id=str(entry.get("id", "")),
                    title=str(entry.get("title", "")),
                    url=str(entry.get("url", "")),
                    websocket_url=url,
                )
            )
        return tuple(targets)

    async def attach(self, target: PageTarget | None = None) -> PageTarget:
        """Attach to a page, or create one if there is none."""
        if target is None:
            targets = await self.list_targets()
            if not targets:
                target = await self.new_page()
            else:
                target = targets[0]

        if self._connection is not None:
            await self._connection.close()

        connection = CDPConnection(target.websocket_url, timeout=self.timeout)
        await connection.connect()
        self._connection = connection
        self._target = target

        # Enabled so navigation and evaluation both work without the caller
        # having to know which domains must be turned on first.
        await connection.call("Page.enable")
        await connection.call("Runtime.enable")
        return target

    async def new_page(self, url: str = "about:blank") -> PageTarget:
        """Open a new tab."""
        client = await self._client()
        try:
            response = await client.put(f"{self.endpoint}/json/new?{url}")
        except Exception as exc:  # noqa: BLE001
            raise BrowserUnavailableError(self.endpoint, describe(exc)) from exc
        if response.status_code != 200:
            raise BrowserError(
                f"could not open a page: HTTP {response.status_code}",
                code="BROWSER_NEW_PAGE_FAILED",
            )
        entry = response.json()
        return PageTarget(
            id=str(entry.get("id", "")),
            title=str(entry.get("title", "")),
            url=str(entry.get("url", "")),
            websocket_url=str(entry.get("webSocketDebuggerUrl", "")),
        )

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as exc:  # noqa: BLE001
                log.debug("closing the HTTP client failed: %s", exc)
            self._http = None

    @property
    def attached(self) -> bool:
        return self._connection is not None and self._connection.connected

    def _require(self) -> CDPConnection:
        if self._connection is None:
            raise BrowserError(
                "not attached to a page; call attach first", code="BROWSER_NOT_ATTACHED"
            )
        return self._connection

    # -- acting ----------------------------------------------------------

    async def navigate(self, url: str, *, wait: float = 0.5) -> Mapping[str, Any]:
        """Go to a URL and give the page a moment to settle.

        The wait is not a substitute for waiting on a load event; it is a
        deliberate compromise. A full load-event wait hangs on pages that never
        finish loading (ads, long-polling), and a hung tool call is worse than a
        slightly early read. A caller that needs certainty can call
        ``wait_for`` explicitly.
        """
        connection = self._require()
        result = await connection.call("Page.navigate", {"url": url})
        if wait > 0:
            await asyncio.sleep(wait)
        return result

    async def evaluate(self, expression: str) -> Any:
        """Run JavaScript in the page and return its value.

        ``returnByValue`` is used so the caller gets data rather than a remote
        object handle, which would be useless outside the page.
        """
        connection = self._require()
        result = await connection.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        if result.get("exceptionDetails") is not None:
            raise BrowserError(
                f"the page raised an error: {_exception_text(result)}",
                code="BROWSER_SCRIPT_ERROR",
            )
        inner = result.get("result")
        if isinstance(inner, Mapping):
            return inner.get("value")
        return None

    async def text(self) -> str:
        """The page's visible text, bounded.

        ``innerText`` rather than ``textContent``: it reflects what is rendered,
        so hidden elements and script blocks do not flood the result.
        """
        raw = await self.evaluate(
            "document.body ? document.body.innerText : ''"
        )
        text = str(raw or "")
        if len(text) > MAX_TEXT_CHARS:
            return text[:MAX_TEXT_CHARS] + f"\n[truncated at {MAX_TEXT_CHARS} characters]"
        return text

    async def title(self) -> str:
        return str(await self.evaluate("document.title") or "")

    async def url(self) -> str:
        return str(await self.evaluate("location.href") or "")

    async def click(self, selector: str) -> None:
        """Click the first element matching a selector."""
        script = f"""
        (() => {{
            const el = document.querySelector({json.dumps(selector)});
            if (!el) return {{ok: false, reason: 'no element matches'}};
            el.scrollIntoView({{block: 'center'}});
            el.click();
            return {{ok: true}};
        }})()
        """
        result = await self.evaluate(script)
        if isinstance(result, Mapping) and not result.get("ok"):
            raise BrowserError(
                f"could not click {selector!r}: {result.get('reason', 'unknown')}",
                code="BROWSER_NO_ELEMENT",
            )

    async def fill(self, selector: str, value: str) -> None:
        """Type into an input, firing the events a framework listens for.

        Setting ``value`` alone is not enough: React and friends track the
        previous value on the DOM node and ignore a change they did not observe,
        so the input looks filled and the app still sees it as empty.
        """
        script = f"""
        (() => {{
            const el = document.querySelector({json.dumps(selector)});
            if (!el) return {{ok: false, reason: 'no element matches'}};
            const proto = el instanceof HTMLTextAreaElement
                ? HTMLTextAreaElement.prototype
                : HTMLInputElement.prototype;
            const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
            if (setter) {{ setter.call(el, {json.dumps(value)}); }}
            else {{ el.value = {json.dumps(value)}; }}
            el.dispatchEvent(new Event('input', {{bubbles: true}}));
            el.dispatchEvent(new Event('change', {{bubbles: true}}));
            return {{ok: true}};
        }})()
        """
        result = await self.evaluate(script)
        if isinstance(result, Mapping) and not result.get("ok"):
            raise BrowserError(
                f"could not fill {selector!r}: {result.get('reason', 'unknown')}",
                code="BROWSER_NO_ELEMENT",
            )

    async def screenshot(self) -> str:
        """A base64 PNG of the visible page.

        Returned as a locator rather than decoded, because this runtime has no
        attachment store; the caller decides what to do with the bytes.
        """
        connection = self._require()
        result = await connection.call("Page.captureScreenshot", {"format": "png"})
        data = result.get("data")
        if not isinstance(data, str) or not data:
            raise BrowserError(
                "the browser returned no screenshot", code="BROWSER_NO_SCREENSHOT"
            )
        return data

    async def wait_for(self, selector: str, *, timeout: float = 10.0) -> bool:
        """Wait for an element to appear.

        Polled rather than driven by a mutation observer, because a poll is
        bounded and obviously correct, and this is not a hot path.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            found = await self.evaluate(
                f"!!document.querySelector({json.dumps(selector)})"
            )
            if found:
                return True
            await asyncio.sleep(0.2)
        return False

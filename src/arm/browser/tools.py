"""Browser tools: the seam's model-facing surface.

Seven tools, and the split is deliberate: **observation is separate from action.**
Reading a page and clicking something on it are different risk profiles, and a
caller approving one should not be approving the other. That distinction is what
lets the approval policy treat ``browser_click`` differently from
``browser_text`` without any browser-specific machinery.

Two things these tools deliberately do **not** do:

**They do not attach implicitly.** ``browser_open`` requires an endpoint or uses
the configured one, and it says so when nothing is listening. A tool that scanned
for a browser would attach to whatever it found, and what it would find is the
user's logged-in session.

**They do not return a screenshot as an image.** The runtime has no attachment
store, so the bytes are returned as a base64 locator with the size stated. Handing
back something the model cannot see, while implying it can, would be worse than
saying plainly that it is bytes.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .cdp import Browser, BrowserError, PageTarget


class OpenParams(BaseModel):
    url: str = Field(
        default="",
        description=(
            "Where to go. Empty attaches to the first existing page without "
            "navigating."
        ),
    )
    endpoint: str = Field(
        default="",
        description=(
            "The browser's CDP endpoint, e.g. 'http://127.0.0.1:9222'. Empty "
            "uses the configured endpoint. A browser must be started with "
            "--remote-debugging-port for this to work."
        ),
    )
    new_page: bool = Field(
        default=False, description="Open a fresh tab rather than reusing one."
    )


class NavigateParams(BaseModel):
    url: str = Field(description="The URL to open.")


class EvaluateParams(BaseModel):
    expression: str = Field(
        description=(
            "JavaScript to run in the page. The value of the last expression is "
            "returned. Prefer the specific tools when one fits -- a script that "
            "clicks is harder to review than a click."
        )
    )


class ClickParams(BaseModel):
    selector: str = Field(description="A CSS selector for the element to click.")


class FillParams(BaseModel):
    selector: str = Field(description="A CSS selector for the input.")
    value: str = Field(description="The text to enter.")


class WaitParams(BaseModel):
    selector: str = Field(description="A CSS selector to wait for.")
    timeout: float = Field(default=10.0, description="How long to wait, in seconds.")


class ScreenshotParams(BaseModel):
    """No arguments: the visible page is the subject."""


class BrowserTools:
    """The browser tool set, bound to one browser."""

    def __init__(self, browser: Browser) -> None:
        self.browser = browser

    def tools(self) -> list[RegisteredTool]:
        return [
            self._open(),
            self._navigate(),
            self._text(),
            self._evaluate(),
            self._click(),
            self._fill(),
            self._screenshot(),
            self._wait_for(),
        ]

    # -- observation -----------------------------------------------------

    def _open(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            endpoint = str(arguments.get("endpoint", "")).strip()
            if endpoint:
                self.browser.endpoint = endpoint.rstrip("/")

            try:
                if arguments.get("new_page"):
                    target = await self.browser.new_page()
                    target = await self.browser.attach(target)
                else:
                    target = await self.browser.attach()
                    url = str(arguments.get("url", "")).strip()
                    if url:
                        await self.browser.navigate(url)
                        target = PageTarget(
                            id=target.id,
                            title=await self.browser.title(),
                            url=await self.browser.url(),
                            websocket_url=target.websocket_url,
                        )
            except BrowserError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            return ToolResult.ok(
                f"Attached to a page.\nURL: {target.url}\nTitle: {target.title}",
                value=target.to_record(),
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="browser_open",
                description=(
                    "Attach to a Chrome instance over CDP, optionally navigating "
                    "to a URL. The browser must already be running with "
                    "--remote-debugging-port; this attaches to *your* browser, so "
                    "the pages you are signed in to are the pages it sees."
                ),
                parameters=OpenParams.model_json_schema(),
            ),
            body=body,
            validator=OpenParams,
        )

    def _navigate(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            url = str(arguments.get("url", "")).strip()
            if not url:
                return ToolResult.fail("a URL is required")
            try:
                await self.browser.navigate(url)
                return ToolResult.ok(
                    f"Went to {await self.browser.url()}\nTitle: {await self.browser.title()}",
                    value={"url": await self.browser.url()},
                )
            except BrowserError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

        return RegisteredTool(
            definition=ToolDefinition(
                name="browser_navigate",
                description="Go to a URL in the attached page.",
                parameters=NavigateParams.model_json_schema(),
            ),
            body=body,
            validator=NavigateParams,
        )

    def _text(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            try:
                text = await self.browser.text()
                url = await self.browser.url()
                title = await self.browser.title()
            except BrowserError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            header = f"{title}\n{url}\n\n"
            return ToolResult.ok(
                header + (text or "(the page has no visible text)"),
                value={"url": url, "title": title, "length": len(text)},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="browser_text",
                description=(
                    "Read the attached page's visible text. This is what a person "
                    "would see, not the HTML source."
                ),
                parameters={"type": "object", "properties": {}},
                concurrency_safe=lambda _args: True,
            ),
            body=body,
        )

    # -- action ----------------------------------------------------------

    def _evaluate(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            expression = str(arguments.get("expression", "")).strip()
            if not expression:
                return ToolResult.fail("an expression is required")
            try:
                value = await self.browser.evaluate(expression)
            except BrowserError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            import json

            try:
                rendered = json.dumps(value, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                rendered = repr(value)
            return ToolResult.ok(rendered[:20_000], value={"result": value})

        return RegisteredTool(
            definition=ToolDefinition(
                name="browser_evaluate",
                description=(
                    "Run JavaScript in the page and return the value of the last "
                    "expression. Use it for what the specific tools do not cover; "
                    "prefer them when one fits, because a click written as a "
                    "script is harder to review than a click."
                ),
                parameters=EvaluateParams.model_json_schema(),
            ),
            body=body,
            validator=EvaluateParams,
        )

    def _click(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            selector = str(arguments.get("selector", "")).strip()
            if not selector:
                return ToolResult.fail("a selector is required")
            try:
                await self.browser.click(selector)
            except BrowserError as exc:
                return ToolResult.fail(str(exc), code=exc.code)
            return ToolResult.ok(f"Clicked {selector}.")

        return RegisteredTool(
            definition=ToolDefinition(
                name="browser_click",
                description=(
                    "Click the first element matching a CSS selector. The element "
                    "is scrolled into view first."
                ),
                parameters=ClickParams.model_json_schema(),
            ),
            body=body,
            validator=ClickParams,
        )

    def _fill(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            selector = str(arguments.get("selector", "")).strip()
            if not selector:
                return ToolResult.fail("a selector is required")
            value = str(arguments.get("value", ""))
            try:
                await self.browser.fill(selector, value)
            except BrowserError as exc:
                return ToolResult.fail(str(exc), code=exc.code)
            return ToolResult.ok(f"Entered {len(value)} characters into {selector}.")

        return RegisteredTool(
            definition=ToolDefinition(
                name="browser_fill",
                description=(
                    "Type into an input or textarea, firing the input and change "
                    "events a framework listens for."
                ),
                parameters=FillParams.model_json_schema(),
            ),
            body=body,
            validator=FillParams,
        )

    def _screenshot(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            try:
                data = await self.browser.screenshot()
            except BrowserError as exc:
                return ToolResult.fail(str(exc), code=exc.code)
            return ToolResult.ok(
                f"Captured a PNG of the visible page ({len(data)} base64 "
                "characters). This runtime has no attachment store, so the image "
                "is returned as data rather than shown.",
                value={"base64": data, "bytes": len(data) * 3 // 4},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="browser_screenshot",
                description=(
                    "Capture the visible page as a base64 PNG. The image is not "
                    "rendered for you -- this runtime has no attachment store -- so "
                    "use browser_text when you need to read the page."
                ),
                parameters={"type": "object", "properties": {}},
            ),
            body=body,
        )

    def _wait_for(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            selector = str(arguments.get("selector", "")).strip()
            if not selector:
                return ToolResult.fail("a selector is required")
            timeout = float(arguments.get("timeout", 10.0))
            try:
                found = await self.browser.wait_for(selector, timeout=timeout)
            except BrowserError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            if not found:
                return ToolResult.fail(
                    f"{selector} did not appear within {timeout}s",
                    code="BROWSER_WAIT_TIMEOUT",
                )
            return ToolResult.ok(f"{selector} is present.")

        return RegisteredTool(
            definition=ToolDefinition(
                name="browser_wait_for",
                description="Wait until an element matching a selector appears.",
                parameters=WaitParams.model_json_schema(),
            ),
            body=body,
            validator=WaitParams,
        )

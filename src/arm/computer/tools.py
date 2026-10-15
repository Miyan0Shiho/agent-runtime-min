"""Computer-use tools: the desktop as a set of ordinary tools.

Six tools, split by consequence:

**Observation** -- ``screen_capture``, ``screen_windows``, ``screen_cursor``. These
read. They do not require approval, because looking at a screen changes nothing,
and gating them would train the user to approve reflexively.

**Action** -- ``screen_click``, ``screen_type``, ``screen_key``. These drive the
user's actual machine. Every one declares ``requires_approval``, so the gate is a
property of the tool rather than a setting that can be misconfigured into being
absent.

That split is the whole design. A tool that could both read and act would have to
be gated, and then reading would prompt; or not gated, and then acting would not.
Separating them is what makes the policy expressible.

**No rollback is claimed anywhere.** A click activates whatever was under it.
Typing goes wherever focus is. The descriptions say this, because a model that
assumes undo takes risks it otherwise would not.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .desktop import ComputerUseError, Desktop


class CaptureParams(BaseModel):
    """No arguments: the screen is the subject."""


class WindowsParams(BaseModel):
    """No arguments: every visible window is listed."""


class CursorParams(BaseModel):
    """No arguments."""


class ClickParams(BaseModel):
    x: int = Field(description="Absolute screen X coordinate, in points.")
    y: int = Field(description="Absolute screen Y coordinate, in points.")
    button: str = Field(
        default="left", description="'left' or 'right'."
    )


class TypeParams(BaseModel):
    text: str = Field(description="The text to type into whatever has focus.")


class KeyParams(BaseModel):
    key: str = Field(
        description=(
            "A key name: return, tab, space, delete, escape, left, right, up, "
            "down, home, end, pageup, pagedown, f1, f2."
        )
    )
    modifiers: list[str] = Field(
        default_factory=list,
        description="Any of: command, option, shift, control.",
    )


class ComputerTools:
    """The desktop tool set."""

    def __init__(self, desktop: Desktop) -> None:
        self.desktop = desktop

    def tools(self) -> list[RegisteredTool]:
        return [
            self._capture(),
            self._windows(),
            self._cursor(),
            self._click(),
            self._type(),
            self._key(),
        ]

    # -- observation -----------------------------------------------------

    def _capture(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            try:
                data = await self._capture_async()
            except ComputerUseError as exc:
                return ToolResult.fail(str(exc), code=exc.code)
            return ToolResult.ok(
                f"Captured the whole screen as a PNG ({len(data)} base64 "
                "characters). This runtime has no attachment store, so the image "
                "is returned as data rather than shown.",
                value={"base64": data},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="screen_capture",
                description=(
                    "Capture the whole screen as a base64 PNG. The image is not "
                    "rendered for you -- this runtime has no attachment store -- so "
                    "this is most useful for handing the image to a person, or for "
                    "confirming that an action had the intended effect. Use "
                    "screen_windows to find out what is where."
                ),
                parameters={"type": "object", "properties": {}},
                concurrency_safe=lambda _args: True,
            ),
            body=body,
        )

    async def _capture_async(self) -> str:
        import asyncio

        # Off the event loop: screenshotting a Retina display takes a noticeable
        # fraction of a second, and blocking would stall every other task.
        return await asyncio.get_running_loop().run_in_executor(
            None, self.desktop.capture
        )

    def _windows(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            import asyncio

            try:
                windows = await asyncio.get_running_loop().run_in_executor(
                    None, self.desktop.windows
                )
            except ComputerUseError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            if not windows:
                return ToolResult.ok(
                    "No visible windows were found. If you expected some, the "
                    "Accessibility permission is probably missing.",
                    value={"count": 0},
                )

            lines = ["Visible windows (app, title, x, y, width, height):"]
            for window in windows:
                lines.append(
                    f"  {window.app} | {window.title} | {window.x},{window.y} "
                    f"{window.width}x{window.height}"
                )
            return ToolResult.ok(
                "\n".join(lines),
                value={"count": len(windows), "windows": [w.to_record() for w in windows]},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="screen_windows",
                description=(
                    "List every visible window with its position and size. Use "
                    "this to find where to click: coordinates are absolute screen "
                    "points, so a window's center is a reasonable target."
                ),
                parameters={"type": "object", "properties": {}},
                concurrency_safe=lambda _args: True,
            ),
            body=body,
        )

    def _cursor(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            import asyncio

            try:
                x, y = await asyncio.get_running_loop().run_in_executor(
                    None, self.desktop.cursor
                )
            except ComputerUseError as exc:
                return ToolResult.fail(str(exc), code=exc.code)
            return ToolResult.ok(f"The pointer is at ({x}, {y}).", value={"x": x, "y": y})

        return RegisteredTool(
            definition=ToolDefinition(
                name="screen_cursor",
                description="Where the mouse pointer is, and which app has focus.",
                parameters={"type": "object", "properties": {}},
                concurrency_safe=lambda _args: True,
            ),
            body=body,
        )

    # -- action ----------------------------------------------------------

    def _click(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            import asyncio

            x = int(arguments.get("x", 0))
            y = int(arguments.get("y", 0))
            button = str(arguments.get("button", "left"))

            try:
                frontmost = await asyncio.get_running_loop().run_in_executor(
                    None, self.desktop.frontmost
                )
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self.desktop.click(x, y, button=button)
                )
            except ComputerUseError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            return ToolResult.ok(
                f"Clicked ({x}, {y}) with the {button} button. "
                f"The frontmost application was {frontmost or 'unknown'}.",
                value={"x": x, "y": y, "frontmost": frontmost},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="screen_click",
                description=(
                    "Click an absolute screen point. This acts on the real "
                    "desktop and cannot be undone: whatever is under the point is "
                    "activated. Use screen_windows first to choose the point, and "
                    "screen_capture afterwards to confirm the effect."
                ),
                parameters=ClickParams.model_json_schema(),
                requires_approval=True,
            ),
            body=body,
            validator=ClickParams,
        )

    def _type(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            import asyncio

            text = str(arguments.get("text", ""))
            try:
                frontmost = await asyncio.get_running_loop().run_in_executor(
                    None, self.desktop.frontmost
                )
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self.desktop.type_text(text)
                )
            except ComputerUseError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            return ToolResult.ok(
                f"Typed {len(text)} characters into {frontmost or 'the focused window'}. "
                "There is no undo.",
                value={"length": len(text), "frontmost": frontmost},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="screen_type",
                description=(
                    "Type text into whatever currently has focus. This acts on the "
                    "real desktop and cannot be undone -- if focus is a terminal, "
                    "the text is a command. Check the frontmost application first, "
                    "and prefer screen_key for anything other than plain text."
                ),
                parameters=TypeParams.model_json_schema(),
                requires_approval=True,
            ),
            body=body,
            validator=TypeParams,
        )

    def _key(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            import asyncio

            key = str(arguments.get("key", ""))
            modifiers = [str(m) for m in (arguments.get("modifiers") or ())]
            try:
                frontmost = await asyncio.get_running_loop().run_in_executor(
                    None, self.desktop.frontmost
                )
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self.desktop.key(key, modifiers=modifiers)
                )
            except ComputerUseError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            rendered = "+".join([*modifiers, key]) if modifiers else key
            return ToolResult.ok(
                f"Pressed {rendered} in {frontmost or 'the focused window'}. "
                "There is no undo.",
                value={"key": key, "modifiers": modifiers, "frontmost": frontmost},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="screen_key",
                description=(
                    "Press a named key, optionally with modifiers. This acts on the "
                    "real desktop and cannot be undone -- a shortcut can quit an "
                    "application or close a window with unsaved work."
                ),
                parameters=KeyParams.model_json_schema(),
                requires_approval=True,
            ),
            body=body,
            validator=KeyParams,
        )

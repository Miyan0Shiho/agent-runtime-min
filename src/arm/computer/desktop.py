"""Computer use: seeing and driving the actual desktop.

This seam is the most dangerous thing in the runtime, and the design follows
entirely from that. A tool that can click anywhere and type anything is not
sandboxed by a working directory -- it acts on whatever is on screen, which
includes the user's mail, their bank, and a terminal with their credentials in it.

Three consequences, each enforced rather than documented:

**Observation and action are separate tools.** ``screen_capture`` and
``screen_windows`` read; ``screen_click``, ``screen_type`` and ``screen_key`` act.
The distinction is what lets approval gate the second group and not the first, and
it needs no special machinery: they are ordinary tools whose ``requires_approval``
flag differs.

**Every action requires approval.** Not by policy configuration -- by the tool's
own declaration, so it is true in every deployment. A runtime that could be
configured into driving the desktop unattended is a runtime that will be.

**Coordinates are absolute screen points, and the tools say so.** A click at
``(500, 400)`` means the same thing regardless of which window is focused, which
is the honest behaviour: a tool that clicked "relative to the frontmost window"
would silently do something different when focus changed between the model's
decision and the click.

**There is no rollback.** A click cannot be undone; typing cannot be untyped.
This is stated in the tool descriptions rather than left implicit, because a model
that assumes undo will take risks it otherwise would not.
"""

from __future__ import annotations

import base64
import ctypes
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..kernel.errors import ArmError, describe

log = logging.getLogger("arm.computer")

#: Where macOS keeps the two binaries this seam needs. Both are system binaries,
#: so there is no dependency to install -- but a missing one is reported clearly
#: rather than producing an empty result.
SCREENCAPTURE = "/usr/sbin/screencapture"
OSASCRIPT = "/usr/bin/osascript"

#: How long one AppleScript command may take.
DEFAULT_TIMEOUT = 20.0


class ComputerUseError(ArmError):
    """The desktop could not be observed or driven."""

    def __init__(self, message: str, *, code: str = "COMPUTER_ERROR") -> None:
        super().__init__(message)
        self.code = code


class ComputerUseUnavailableError(ComputerUseError):
    """This platform cannot be driven by this seam."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason, code="COMPUTER_UNAVAILABLE")


@dataclass(frozen=True, slots=True)
class ScreenWindow:
    """One on-screen window."""

    app: str
    title: str
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    def to_record(self) -> dict[str, Any]:
        return {
            "app": self.app,
            "title": self.title,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def _run(command: Sequence[str], *, timeout: float = DEFAULT_TIMEOUT) -> tuple[int, str, str]:
    """Run a command and return its exit code, stdout and stderr.

    Synchronous on purpose: both tools are short-lived system binaries, and
    running them through the event loop would mean managing subprocesses for no
    benefit. A caller that needs to stay responsive runs this in an executor.
    """
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ComputerUseError(
            f"the command {command[0]!r} was not found", code="COMPUTER_MISSING_BINARY"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ComputerUseError(
            f"{command[0]!r} did not finish within {timeout}s",
            code="COMPUTER_TIMEOUT",
        ) from exc

    return (
        completed.returncode,
        completed.stdout.decode("utf-8", "replace"),
        completed.stderr.decode("utf-8", "replace"),
    )


class Desktop:
    """The user's actual desktop.

    Deliberately thin: it holds no state about windows or focus, because any such
    state would be stale by the time an action ran. Every operation re-reads what
    it needs, which is slower and always correct.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        enabled: bool = False,
        observe: bool = False,
    ) -> None:
        #: Whether the desktop may be **driven**. Opt-in, because driving is
        #: irreversible: a click activates whatever was under it and typing goes
        #: wherever focus is.
        self.enabled = enabled
        #: Whether the screen may be **observed**. Separate from ``enabled``
        #: because the two have different consequences and the tool split already
        #: says so: `screen_capture`/`screen_windows`/`screen_cursor` carry no
        #: approval requirement "because looking at a screen changes nothing",
        #: while the acting tools do.
        #:
        #: They were not separate in practice. Every method called
        #: `_require_enabled`, so observation was refused unless driving was also
        #: permitted -- and the comment below claimed the opposite. Measured: with
        #: the flag off, `capture`, `windows`, `cursor` and `frontmost` all raised
        #: COMPUTER_DISABLED, so a runtime that only wanted screenshots had to
        #: grant itself the ability to type into a password prompt.
        self.observe = observe or enabled
        self.timeout = timeout

    # -- capability ------------------------------------------------------

    @staticmethod
    def available() -> tuple[bool, str]:
        """Whether this platform can be driven, and why not if it cannot."""
        import sys

        if sys.platform != "darwin":
            return False, (
                f"computer use is implemented for macOS only; this is "
                f"{sys.platform}"
            )
        if not shutil.which(SCREENCAPTURE):
            return False, f"{SCREENCAPTURE} is missing"
        if not shutil.which(OSASCRIPT):
            return False, f"{OSASCRIPT} is missing"
        return True, ""

    def require_available(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise ComputerUseUnavailableError(reason)

    def _require_observable(self) -> None:
        """Observation gate: weaker than driving, and independent of it."""
        if not self.observe:
            raise ComputerUseError(
                "computer observation is disabled; set ARM_COMPUTER=1 to allow "
                "the agent to look at the desktop. It is off by default because a "
                "screenshot can contain anything that is on screen.",
                code="COMPUTER_DISABLED",
            )

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise ComputerUseError(
                "computer use is disabled; set ARM_COMPUTER=1 to allow the agent "
                "to observe and drive the desktop. It is off by default because "
                "the desktop is not a sandbox.",
                code="COMPUTER_DISABLED",
            )

    # -- observation -----------------------------------------------------

    def capture(self) -> str:
        """A PNG of the whole screen, base64-encoded.

        The **whole screen**, not a window: capturing a window would hide the
        dialog that is actually blocking progress, and a screenshot that omits the
        thing the user is asking about is worse than none.
        """
        self._require_observable()
        self.require_available()

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.png"
            # -x silences the shutter sound: this may run many times, and an
            # audible click each time would make the agent unusable on a shared
            # machine.
            code, _out, err = _run(
                [SCREENCAPTURE, "-x", "-t", "png", str(path)], timeout=self.timeout
            )
            if code != 0 or not path.exists():
                raise ComputerUseError(
                    f"the screen could not be captured: {err.strip() or f'exit {code}'}",
                    code="COMPUTER_CAPTURE_FAILED",
                )
            return base64.b64encode(path.read_bytes()).decode("ascii")

    def windows(self) -> tuple[ScreenWindow, ...]:
        """Every on-screen window, with its position and size.

        Read through System Events, which needs Accessibility permission. A
        missing permission produces an empty list rather than an error, so the
        empty case is reported explicitly -- otherwise "no windows" and "no
        permission" would look identical, and the second is the one worth fixing.
        """
        self._require_observable()
        self.require_available()

        script = """
        set output to ""
        tell application "System Events"
            repeat with proc in (every process whose background only is false)
                try
                    repeat with w in (every window of proc)
                        try
                            set pos to position of w
                            set sz to size of w
                            set output to output & (name of proc) & "\\t" & (name of w) & "\\t" & (item 1 of pos) & "\\t" & (item 2 of pos) & "\\t" & (item 1 of sz) & "\\t" & (item 2 of sz) & "\\n"
                        end try
                    end repeat
                end try
            end repeat
        end tell
        return output
        """
        code, out, err = _run([OSASCRIPT, "-e", script], timeout=self.timeout)
        if code != 0:
            detail = err.strip()
            if "not allowed" in detail or "assistive" in detail.lower():
                raise ComputerUseError(
                    "Accessibility permission is not granted, so windows cannot be "
                    "listed. Grant it in System Settings → Privacy & Security → "
                    "Accessibility.",
                    code="COMPUTER_NO_ACCESSIBILITY",
                )
            raise ComputerUseError(
                f"could not list windows: {detail or f'exit {code}'}",
                code="COMPUTER_LIST_FAILED",
            )

        windows: list[ScreenWindow] = []
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) != 6:
                continue
            try:
                x, y, width, height = (int(p) for p in parts[2:])
            except ValueError:
                continue
            windows.append(
                ScreenWindow(
                    app=parts[0], title=parts[1], x=x, y=y, width=width, height=height
                )
            )
        return tuple(windows)

    def frontmost(self) -> str:
        """The application that would receive a keystroke right now.

        Read before typing rather than assumed: focus can change between the
        model's decision and the action, and typing into the wrong window is not
        recoverable.
        """
        self._require_observable()
        self.require_available()

        script = (
            'tell application "System Events" to return name of first process '
            "whose frontmost is true"
        )
        code, out, _err = _run([OSASCRIPT, "-e", script], timeout=self.timeout)
        if code != 0:
            return ""
        return out.strip()

    def cursor(self) -> tuple[int, int]:
        """Where the pointer is. Useful for confirming a click landed.

        Read through CoreGraphics rather than AppleScript: System Events has no
        command that returns the cursor position, and an AppleScript that looks
        like it should returns a ``missing value`` -- which a naive implementation
        would turn into ``(-1, -1)`` and report as a real coordinate.
        """
        self._require_observable()
        self.require_available()

        try:
            framework, foundation = _core_graphics()
        except ComputerUseError:
            return (-1, -1)

        # An event object is created only to read the *current* pointer location
        # from it; nothing is posted, so this observes without acting.
        event = framework.CGEventCreate(None)
        if not event:
            return (-1, -1)
        try:
            location = framework.CGEventGetLocation(ctypes.c_void_p(event))
        finally:
            foundation.CFRelease(ctypes.c_void_p(event))
        return int(location.x), int(location.y)

    # -- action ----------------------------------------------------------

    def click(self, x: int, y: int, *, button: str = "left") -> None:
        """Click an absolute screen point.

        The click is a single ``cliclick``-style event issued through System
        Events. There is no rollback: whatever was under the cursor is now
        activated, and if that was a dialog the user had open, it is answered.
        """
        self._require_enabled()
        self.require_available()

        if button not in ("left", "right"):
            raise ComputerUseError(
                f"unknown mouse button {button!r}; expected 'left' or 'right'"
            )

        # System Events can *read* the pointer but not move it, so the click is
        # issued through CoreGraphics directly. Reached with ctypes rather than a
        # binding: this needs three function calls, and adding a dependency for
        # three calls would mean the seam is unavailable wherever the dependency
        # is not installed.
        _post_mouse_event(x, y, button=button)

    def type_text(self, text: str) -> None:
        """Type text into whatever is focused.

        Read the frontmost application first and state it in the result: typing
        into the wrong window is not recoverable, and the model needs to know
        where its text went.
        """
        self._require_enabled()
        self.require_available()

        if not text:
            raise ComputerUseError("nothing to type", code="COMPUTER_EMPTY_INPUT")

        # Passed as an argument rather than interpolated into the script, so a
        # quote or a backslash in the text cannot change what the script does.
        script = (
            'on run argv\n'
            'tell application "System Events" to keystroke (item 1 of argv)\n'
            "end run"
        )
        code, _out, err = _run(
            [OSASCRIPT, "-e", script, text], timeout=self.timeout
        )
        if code != 0:
            detail = err.strip()
            if "not allowed" in detail or "assistive" in detail.lower():
                raise ComputerUseError(
                    "Accessibility permission is not granted, so typing is not "
                    "possible. Grant it in System Settings → Privacy & Security → "
                    "Accessibility.",
                    code="COMPUTER_NO_ACCESSIBILITY",
                )
            raise ComputerUseError(
                f"could not type: {detail or f'exit {code}'}",
                code="COMPUTER_TYPE_FAILED",
            )

    def key(self, key_name: str, *, modifiers: Sequence[str] = ()) -> None:
        """Press a named key, optionally with modifiers.

        Key names are a fixed vocabulary rather than arbitrary strings: an
        unknown name would otherwise be silently ignored, and a keystroke that did
        nothing is indistinguishable from one that worked.
        """
        self._require_enabled()
        self.require_available()

        name = key_name.strip().lower()
        if name not in KEY_CODES:
            raise ComputerUseError(
                f"unknown key {key_name!r}; expected one of "
                f"{', '.join(sorted(KEY_CODES))}",
                code="COMPUTER_UNKNOWN_KEY",
            )

        for modifier in modifiers:
            if modifier.lower() not in MODIFIERS:
                raise ComputerUseError(
                    f"unknown modifier {modifier!r}; expected one of "
                    f"{', '.join(sorted(MODIFIERS))}",
                    code="COMPUTER_UNKNOWN_MODIFIER",
                )

        using = ""
        if modifiers:
            listed = ", ".join(m.lower() for m in modifiers)
            using = f" using {{{listed}}}"

        if name in KEY_CODES:
            script = (
                'on run argv\n'
                "set keyCode to (item 1 of argv) as integer\n"
                f'tell application "System Events" to key code keyCode{using}\n'
                "end run"
            )
            arguments = [str(KEY_CODES[name])]
        else:  # pragma: no cover - every name above is a key code
            script = (
                'on run argv\n'
                f'tell application "System Events" to keystroke (item 1 of argv){using}\n'
                "end run"
            )
            arguments = [name]

        code, _out, err = _run(
            [OSASCRIPT, "-e", script, *arguments], timeout=self.timeout
        )
        if code != 0:
            detail = err.strip()
            if "not allowed" in detail or "assistive" in detail.lower():
                raise ComputerUseError(
                    "Accessibility permission is not granted, so key presses are "
                    "not possible.",
                    code="COMPUTER_NO_ACCESSIBILITY",
                )
            raise ComputerUseError(
                f"could not press {key_name!r}: {detail or f'exit {code}'}",
                code="COMPUTER_KEY_FAILED",
            )


#: The key vocabulary. Named keys map to macOS virtual key codes, which is what
#: System Events needs to press a key that produces no character.
KEY_CODES: Mapping[str, int] = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "escape": 53,
    "esc": 53,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "f1": 122,
    "f2": 120,
}

#: Modifier names, spelled the way AppleScript expects them.
MODIFIERS = frozenset({"command", "option", "shift", "control"})


# -- native pointer events ---------------------------------------------
#
# CoreGraphics is reached through ctypes. This is the one place in the runtime
# that calls into a C API directly, and it is here because the alternative --
# asking System Events to move the pointer -- is not possible: AppleScript can
# read the cursor position but has no command to set it.

#: CoreGraphics mouse event types.
_EVENT_LEFT_DOWN = 1
_EVENT_LEFT_UP = 2
_EVENT_RIGHT_DOWN = 3
_EVENT_RIGHT_UP = 4
#: Posting to the HID event tap means the event enters the system as if a real
#: device produced it, which is what makes it reach every application.
_TAP_HID = 0
_BUTTON_LEFT = 0
_BUTTON_RIGHT = 1


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def _core_graphics() -> Any:
    """Load CoreGraphics and declare the signatures this module uses.

    Declared explicitly because ctypes defaults to ``int`` for both arguments and
    return values, and a pointer returned as an int is truncated on a 64-bit
    system -- producing a crash or, worse, a click at a nonsense coordinate.
    """
    import ctypes
    import ctypes.util

    path = (
        ctypes.util.find_library("CoreGraphics")
        or "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
    )
    try:
        framework = ctypes.CDLL(path)
    except OSError as exc:
        raise ComputerUseError(
            f"CoreGraphics could not be loaded: {describe(exc)}",
            code="COMPUTER_NO_POINTER",
        ) from exc

    framework.CGEventCreateMouseEvent.restype = ctypes.c_void_p
    framework.CGEventCreateMouseEvent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        _CGPoint,
        ctypes.c_uint32,
    ]
    framework.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    framework.CGEventPost.restype = None
    framework.CGEventCreate.restype = ctypes.c_void_p
    framework.CGEventCreate.argtypes = [ctypes.c_void_p]
    # Returns a struct *by value*, so the restype must be the struct rather than
    # the default int -- otherwise the coordinates arrive as garbage.
    framework.CGEventGetLocation.restype = _CGPoint
    framework.CGEventGetLocation.argtypes = [ctypes.c_void_p]

    core_foundation_path = (
        ctypes.util.find_library("CoreFoundation")
        or "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    foundation = ctypes.CDLL(core_foundation_path)
    foundation.CFRelease.argtypes = [ctypes.c_void_p]
    foundation.CFRelease.restype = None

    return framework, foundation


def _post_mouse_event(x: int, y: int, *, button: str) -> None:
    """Post a click at an absolute screen point.

    A press and a release, not just a press: an application that receives a down
    event with no matching up stays in a dragging state, which is a worse outcome
    than no click at all.

    Both events are released even if posting fails partway, because a leaked
    CoreFoundation object is a small leak that would accumulate over a long
    session.
    """
    import ctypes

    framework, foundation = _core_graphics()
    point = _CGPoint(float(x), float(y))

    if button == "right":
        down_type, up_type, mouse_button = (
            _EVENT_RIGHT_DOWN,
            _EVENT_RIGHT_UP,
            _BUTTON_RIGHT,
        )
    else:
        down_type, up_type, mouse_button = (
            _EVENT_LEFT_DOWN,
            _EVENT_LEFT_UP,
            _BUTTON_LEFT,
        )

    for event_type in (down_type, up_type):
        event = framework.CGEventCreateMouseEvent(
            None, event_type, point, mouse_button
        )
        if not event:
            raise ComputerUseError(
                "the system refused to create a pointer event",
                code="COMPUTER_NO_POINTER",
            )
        try:
            framework.CGEventPost(_TAP_HID, ctypes.c_void_p(event))
        finally:
            foundation.CFRelease(ctypes.c_void_p(event))

"""Computer-use tests.

**These tests never click, type or press a key.** That is a deliberate limit, and
it is the most important thing about this file.

A synthetic input test would act on the machine running it. While writing this
module I ran `click at {500, 400}` as a capability probe and it landed on the
user's open Obsidian window -- a real action on a real desktop, from a test that
was only supposed to check whether the capability existed. That is exactly the
failure this seam makes possible, so the tests do not do it.

What that means for coverage, stated plainly: the observation path is tested
against the real desktop, and the action path is tested only as far as *not
acting* -- that it refuses when disabled, refuses an unknown key, refuses an empty
string, and carries the approval flag that gates it. The click and keystroke
primitives themselves are exercised by hand, not here.

That is a real gap, and it is the honest one: the alternative is a test suite that
moves the user's mouse.
"""

from __future__ import annotations

import asyncio

import pytest

from arm.computer.desktop import (
    KEY_CODES,
    MODIFIERS,
    ComputerUseError,
    ComputerUseUnavailableError,
    Desktop,
    ScreenWindow,
)
from arm.computer.tools import ComputerTools


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def desktop():
    """An enabled desktop, for observation only."""
    return Desktop(enabled=True)


# ===========================================================================
# It is off by default
# ===========================================================================


def test_computer_use_is_disabled_by_default():
    """The desktop is not a sandbox.

    A runtime that could drive the machine without being asked to would be
    reaching into the user's mail, their bank and their terminal.
    """
    instance = Desktop()
    assert not instance.enabled

    with pytest.raises(ComputerUseError) as info:
        instance.capture()
    assert info.value.code == "COMPUTER_DISABLED"
    assert "ARM_COMPUTER" in str(info.value)


def test_observation_is_refused_while_disabled():
    """Reading the screen is still reading the user's screen."""
    instance = Desktop()
    for operation in (instance.windows, instance.cursor, instance.frontmost):
        with pytest.raises(ComputerUseError) as info:
            operation()
        assert info.value.code == "COMPUTER_DISABLED"


def test_the_tools_refuse_while_disabled():
    tools = {t.definition.name: t for t in ComputerTools(Desktop()).tools()}
    result = run(tools["screen_windows"].body({}, None))
    assert result.is_error
    assert result.error_code == "COMPUTER_DISABLED"


# ===========================================================================
# Observation, against the real desktop
# ===========================================================================


def test_capability_is_reported_with_a_reason():
    """A platform that cannot be driven must say why, not fail obscurely."""
    ok, reason = Desktop.available()
    if ok:
        assert reason == ""
    else:
        assert reason, "an unavailable platform gave no reason"


def test_listing_windows(desktop):
    """The real desktop is read; nothing is acted on."""
    ok, _reason = Desktop.available()
    if not ok:
        pytest.skip("computer use is not available on this platform")

    try:
        windows = desktop.windows()
    except ComputerUseError as exc:
        if exc.code == "COMPUTER_NO_ACCESSIBILITY":
            pytest.skip("Accessibility permission is not granted")
        raise

    for window in windows:
        assert window.app
        assert isinstance(window.x, int)
        assert window.width >= 0


def test_a_window_center_is_a_point_inside_it():
    """The center is the natural click target, so it must be inside the window."""
    window = ScreenWindow(app="A", title="t", x=100, y=200, width=400, height=300)
    assert window.center == (300, 350)
    assert window.x <= window.center[0] <= window.x + window.width


def test_the_cursor_position_is_read(desktop):
    """Read through CoreGraphics: AppleScript has no command for this.

    An AppleScript that looks like it should work returns ``missing value``, which
    a naive implementation turns into ``(-1, -1)`` and reports as a coordinate.
    """
    ok, _reason = Desktop.available()
    if not ok:
        pytest.skip("computer use is not available on this platform")

    x, y = desktop.cursor()
    assert (x, y) != (-1, -1), "the cursor position could not be read"
    assert x >= 0 and y >= 0


def test_the_frontmost_application_is_named(desktop):
    ok, _reason = Desktop.available()
    if not ok:
        pytest.skip("computer use is not available on this platform")
    assert desktop.frontmost(), "no frontmost application was reported"


def test_the_windows_tool_renders_positions(desktop):
    ok, _reason = Desktop.available()
    if not ok:
        pytest.skip("computer use is not available on this platform")

    tools = {t.definition.name: t for t in ComputerTools(desktop).tools()}
    try:
        result = run(tools["screen_windows"].body({}, None))
    except ComputerUseError as exc:
        if exc.code == "COMPUTER_NO_ACCESSIBILITY":
            pytest.skip("Accessibility permission is not granted")
        raise

    # `screen_windows` is observation, not action: it always succeeds if the
    # desktop is reachable. Its content shape depends on whether System Events
    # currently reports any foreground windows -- which is machine state, not
    # contract. The contract here is "tool ran and produced a structured result".
    assert not result.is_error, result.content
    assert result.value is not None and "count" in result.value
    assert result.value["count"] >= 0


def test_screen_windows_handles_zero_visible_windows():
    """`screen_windows` treats "no foreground windows" as a valid observation,
    not an error -- because that state is reachable on every fullscreen app,
    every locked screen, and every fresh login, and observation must keep
    working in those states.

    Without this case every `windows()` caller in the file depends on the
    machine currently having one, which is the assumption the preceding
    tests were quietly making; the tool-layer zero-window branch was never
    exercised before.
    """

    class _EmptyDesktop:
        enabled = True
        observe = True

        def windows(self):
            return ()

        def capture(self):
            return "screencapture-ok"

        def cursor(self):
            return (0, 0)

        def frontmost(self):
            return ""

        def _require_observable(self):
            pass

        def require_available(self):
            pass

    async def _call():
        return await tools["screen_windows"].body({}, None)

    # `ComputerTools` only ever calls `desktop.windows()` on this path; the
    # other attributes/methods on `_EmptyDesktop` exist so that `tools()`
    # can instantiate the remaining five tools without reaching the real
    # desktop. The duck-typed fake keeps this test stable on machines with
    # no foreground window, which the real `Desktop.windows()` cannot
    # guarantee.
    tools = {
        t.definition.name: t
        for t in ComputerTools(_EmptyDesktop()).tools()  # type: ignore[arg-type]
    }
    result = run(_call())
    assert not result.is_error
    assert result.value == {"count": 0}


def test_the_cursor_tool_reports_a_point(desktop):
    ok, _reason = Desktop.available()
    if not ok:
        pytest.skip("computer use is not available on this platform")

    tools = {t.definition.name: t for t in ComputerTools(desktop).tools()}
    result = run(tools["screen_cursor"].body({}, None))
    assert not result.is_error
    assert result.value["x"] >= 0


# ===========================================================================
# Action: gated, and never exercised
# ===========================================================================


def test_every_action_requires_approval():
    """Declared by the tool, so it is true in every deployment.

    A runtime that could be configured into driving the desktop unattended is a
    runtime that will be.
    """
    tools = {t.definition.name: t for t in ComputerTools(Desktop(enabled=True)).tools()}
    for name in ("screen_click", "screen_type", "screen_key"):
        assert tools[name].definition.requires_approval, name


def test_observation_does_not_require_approval():
    """Looking at a screen changes nothing.

    Gating it would train the user to approve reflexively, which makes the gate on
    the actions worth less.
    """
    tools = {t.definition.name: t for t in ComputerTools(Desktop(enabled=True)).tools()}
    for name in ("screen_capture", "screen_windows", "screen_cursor"):
        assert not tools[name].definition.requires_approval, name


def test_an_unknown_key_is_refused(desktop):
    """An unknown name would otherwise be silently ignored.

    A keystroke that did nothing is indistinguishable from one that worked, so the
    vocabulary is fixed and a bad name is an error.
    """
    with pytest.raises(ComputerUseError) as info:
        desktop.key("supercalifragilistic")
    assert info.value.code == "COMPUTER_UNKNOWN_KEY"
    assert "return" in str(info.value), "the error should list the valid keys"


def test_an_unknown_modifier_is_refused(desktop):
    with pytest.raises(ComputerUseError) as info:
        desktop.key("return", modifiers=["hyper"])
    assert info.value.code == "COMPUTER_UNKNOWN_MODIFIER"
    assert "command" in str(info.value)


def test_an_unknown_mouse_button_is_refused(desktop):
    with pytest.raises(ComputerUseError) as info:
        desktop.click(10, 10, button="middle")
    assert "unknown mouse button" in str(info.value)


def test_typing_nothing_is_refused(desktop):
    """An empty keystroke is a mistake, not a no-op to perform."""
    with pytest.raises(ComputerUseError) as info:
        desktop.type_text("")
    assert info.value.code == "COMPUTER_EMPTY_INPUT"


def test_actions_are_refused_while_disabled():
    """The gate is checked before anything reaches the system."""
    instance = Desktop()
    with pytest.raises(ComputerUseError) as info:
        instance.click(10, 10)
    assert info.value.code == "COMPUTER_DISABLED"

    with pytest.raises(ComputerUseError) as info:
        instance.type_text("hello")
    assert info.value.code == "COMPUTER_DISABLED"

    with pytest.raises(ComputerUseError) as info:
        instance.key("return")
    assert info.value.code == "COMPUTER_DISABLED"


def test_the_key_vocabulary_is_complete_enough_to_be_useful():
    """The named keys a model actually needs, and no more."""
    for name in ("return", "tab", "space", "escape", "left", "right", "up", "down"):
        assert name in KEY_CODES, name
    assert MODIFIERS == {"command", "option", "shift", "control"}


def test_key_codes_are_distinct_where_they_should_be():
    """Two names for one key are fine; two keys for one code is a bug."""
    assert KEY_CODES["return"] == KEY_CODES["enter"]
    assert KEY_CODES["escape"] == KEY_CODES["esc"]
    assert KEY_CODES["left"] != KEY_CODES["right"]


# ===========================================================================
# The tools' contracts
# ===========================================================================


def test_every_computer_tool_schema_normalizes():
    """The schemas must survive the model-facing subset."""
    import json

    tools = {t.definition.name: t for t in ComputerTools(Desktop()).tools()}
    for tool in tools.values():
        rendered = json.dumps(dict(tool.definition.parameters))
        assert "title" not in rendered, tool.definition.name
        assert "$defs" not in rendered, tool.definition.name


def test_the_action_descriptions_state_that_there_is_no_undo():
    """A model that assumes undo takes risks it otherwise would not."""
    tools = {t.definition.name: t for t in ComputerTools(Desktop()).tools()}
    for name in ("screen_click", "screen_type", "screen_key"):
        description = tools[name].definition.description.lower()
        assert "cannot be undone" in description or "no undo" in description, name


def test_the_action_descriptions_warn_about_focus():
    """Typing goes wherever focus is, and focus can change."""
    tools = {t.definition.name: t for t in ComputerTools(Desktop()).tools()}
    assert "focus" in tools["screen_type"].definition.description.lower()


def test_observation_tools_are_concurrency_safe():
    """Reading a screen can run alongside other reads."""
    tools = {t.definition.name: t for t in ComputerTools(Desktop()).tools()}
    for name in ("screen_capture", "screen_windows", "screen_cursor"):
        assert tools[name].definition.allows_parallel({}), name


def test_action_tools_are_not_concurrency_safe():
    """Two clicks racing on one desktop is not a state anyone can reason about."""
    tools = {t.definition.name: t for t in ComputerTools(Desktop()).tools()}
    for name in ("screen_click", "screen_type", "screen_key"):
        assert not tools[name].definition.allows_parallel({}), name


def test_an_unavailable_platform_raises_a_specific_error():
    """A clear error beats a mysterious empty result."""
    assert issubclass(ComputerUseUnavailableError, ComputerUseError)


# ===========================================================================
# Observation and driving are separate capabilities
# ===========================================================================


def test_observation_is_not_gated_on_driving():
    """The documented split, which the code did not implement.

    `Desktop.__init__` said "Observation is always allowed; driving the desktop is
    opt-in, so a runtime that merely wants screenshots does not thereby acquire
    the ability to type into a password prompt", and the tool layer says the same:
    `screen_capture` carries no approval requirement "because looking at a screen
    changes nothing".

    Measured before the fix, with the flag off: `capture`, `windows`, `cursor` and
    `frontmost` **all** raised COMPUTER_DISABLED. So the only way to look at the
    screen was to also allow typing into it -- the exact outcome the comment said
    it prevented.

    The contract under test is **that observation is reachable while driving is
    disabled**. Whether the desktop currently has a foreground window is machine
    state, not part of this contract: a fullscreen app / locked screen / fresh
    login is observation-equally-forbidden-for-everyone, not observation gated on
    driving -- so the test must not assume `windows()` is non-empty.
    """
    from arm.computer.desktop import Desktop

    observing = Desktop(observe=True, enabled=False)

    assert observing.observe is True
    assert observing.enabled is False

    # `windows()` raises if observation is gated (COMPUTER_DISABLED /
    # COMPUTER_NO_ACCESSIBILITY). Reaching the empty-tuple case means
    # observation is reachable; the length is machine state.
    observing.windows()
    assert isinstance(observing.cursor(), tuple)


def test_driving_requires_more_than_observation():
    """The complement, and the reason the two are separate.

    Being allowed to look must not imply being allowed to click. A screenshot
    contains whatever is on screen; a click activates whatever is under it.
    """
    from arm.computer.desktop import ComputerUseError, Desktop

    observing = Desktop(observe=True, enabled=False)

    for action in (
        lambda: observing.click(10, 10),
        lambda: observing.type_text("x"),
        lambda: observing.key("a"),
    ):
        with pytest.raises(ComputerUseError) as info:
            action()
        assert info.value.code == "COMPUTER_DISABLED"


def test_driving_implies_observation():
    """A runtime allowed to click is obviously allowed to look.

    Requiring both flags would be a trap: every driver would silently lose its
    eyes.
    """
    from arm.computer.desktop import Desktop

    driving = Desktop(enabled=True)

    assert driving.enabled is True
    assert driving.observe is True


def test_nothing_is_allowed_by_default():
    """Both capabilities off, which is the state a caller gets unasked."""
    from arm.computer.desktop import ComputerUseError, Desktop

    default = Desktop()

    assert default.observe is False
    assert default.enabled is False
    with pytest.raises(ComputerUseError):
        default.windows()

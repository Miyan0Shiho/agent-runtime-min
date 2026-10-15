"""Browser tests.

These drive a **real Chrome over CDP**, because that is the only way to test this
seam: the module's entire job is to speak a protocol to a real browser, and a
mock WebSocket would verify the mock rather than the protocol.

The tests skip when no browser is reachable, so the suite still passes on a
machine without Chrome running. That is a deliberate trade: a skipped test is
honest about what it did not check, whereas a mock would report success for a
protocol that was never exercised.

The fixture starts its own headless Chrome on a private port with a throwaway
profile, so the tests never touch the user's browser or its logins.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from arm.browser.cdp import (
    DEFAULT_ENDPOINT,
    Browser,
    BrowserError,
    BrowserUnavailableError,
    CDPConnection,
    PageTarget,
)
from arm.browser.tools import BrowserTools

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

#: A private port, so the fixture never collides with a browser the user is
#: already debugging.
TEST_PORT = 9444
TEST_ENDPOINT = f"http://127.0.0.1:{TEST_PORT}"


def run(coro):
    return asyncio.run(coro)

def scenario(fn):
    """Run an async scenario in a single event loop.

    The CDP connection is a WebSocket, which is bound to the loop that created
    it. Splitting a scenario across ``asyncio.run`` calls -- one per step -- is
    what produces "attached to a different loop"; the connection is created in
    one loop and used from another.
    """
    return asyncio.run(fn())



def _endpoint_alive(endpoint: str) -> bool:
    """Whether a CDP endpoint answers. Used to skip rather than fail."""
    import httpx

    try:
        response = httpx.get(f"{endpoint}/json/version", timeout=1.0)
        return response.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def chrome(tmp_path_factory):
    """A headless Chrome on a private port with a throwaway profile."""
    if not Path(CHROME).exists():
        pytest.skip("Chrome is not installed")
    if _endpoint_alive(TEST_ENDPOINT):
        pytest.skip(f"something is already listening on {TEST_PORT}")

    profile = tmp_path_factory.mktemp("chrome-profile")
    process = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={TEST_PORT}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the endpoint rather than sleeping a fixed amount: startup time
    # varies, and a fixed sleep is either flaky or slow.
    for _ in range(60):
        if _endpoint_alive(TEST_ENDPOINT):
            break
        time.sleep(0.25)
    else:
        process.terminate()
        pytest.skip("Chrome did not expose a CDP endpoint in time")

    yield TEST_ENDPOINT

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    shutil.rmtree(profile, ignore_errors=True)


@pytest.fixture
def browser(chrome):
    """A connected browser, closed after the test."""
    instance = Browser(chrome)
    yield instance
    run(instance.close())


PAGE = "data:text/html," + (
    "<title>Test page</title>"
    "<h1 id='heading'>Hello there</h1>"
    "<input id='field' type='text'>"
    "<button id='button' onclick=\"document.getElementById('heading').textContent='Clicked!'\">"
    "Press me</button>"
    "<div id='hidden' style='display:none'>not visible</div>"
    "<div id='late'></div>"
    "<script>setTimeout(() => { document.getElementById('late').textContent = 'arrived'; }, 400);</script>"
)


# ===========================================================================
# The endpoint must be explicit
# ===========================================================================


def test_a_browser_needs_an_explicit_endpoint():
    """Guessing a port would attach to a session the user did not offer.

    And what it would find is the user's logged-in browser.
    """
    with pytest.raises(BrowserError) as info:
        Browser("")
    assert info.value.code == "BROWSER_NO_ENDPOINT"


def test_an_unreachable_endpoint_is_reported_clearly():
    """The message must say what to do, not just that it failed."""
    instance = Browser("http://127.0.0.1:1")
    with pytest.raises(BrowserUnavailableError) as info:
        run(instance.list_targets())
    assert info.value.code == "BROWSER_UNAVAILABLE"
    assert "remote-debugging-port" in str(info.value)


def test_acting_before_attaching_is_refused():
    """A clear error beats an obscure one from three layers down."""
    instance = Browser(TEST_ENDPOINT)
    with pytest.raises(BrowserError) as info:
        run(instance.evaluate("1"))
    assert info.value.code == "BROWSER_NOT_ATTACHED"


def test_the_default_endpoint_is_the_documented_one():
    assert DEFAULT_ENDPOINT == "http://127.0.0.1:9222"


# ===========================================================================
# Against a real browser
# ===========================================================================


def test_attaching_lists_the_open_pages(browser):
    targets = run(browser.list_targets())
    assert targets, "no page targets were listed"
    assert all(t.websocket_url for t in targets), "a target had no socket to drive"


def test_navigating_and_reading_the_title(browser):
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        return await browser.title(), await browser.url()

    title, url = scenario(go)
    assert title == "Test page"
    assert url.startswith("data:text/html")


def test_reading_the_visible_text(browser):
    """``innerText`` reflects what is rendered, not the source."""
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        return await browser.text()

    text = scenario(go)
    assert "Hello there" in text
    assert "Press me" in text
    # A hidden element must not appear: that is the difference between innerText
    # and textContent.
    assert "not visible" not in text


def test_evaluating_javascript_returns_a_value(browser):
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        return await browser.evaluate("1 + 1"), await browser.evaluate(
            "document.getElementById('heading').textContent"
        )

    assert scenario(go) == (2, "Hello there")


def test_a_script_error_is_reported(browser):
    """A page-side exception must surface as an error, not as a silent None."""
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        await browser.evaluate("throw new Error('boom')")

    with pytest.raises(BrowserError) as info:
        scenario(go)
    assert info.value.code == "BROWSER_SCRIPT_ERROR"
    assert "boom" in str(info.value)


def test_clicking_changes_the_page(browser):
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        await browser.click("#button")
        return await browser.evaluate("document.getElementById('heading').textContent")

    assert scenario(go) == "Clicked!"


def test_clicking_a_missing_element_is_reported(browser):
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        await browser.click("#does-not-exist")

    with pytest.raises(BrowserError) as info:
        scenario(go)
    assert info.value.code == "BROWSER_NO_ELEMENT"


def test_filling_an_input_fires_the_events_frameworks_listen_for(browser):
    """Setting ``value`` alone is not enough: React ignores a change it did not observe.

    The input would look filled and the app would still see it as empty.
    """
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        await browser.evaluate(
            "window.__events = [];"
            "const f = document.getElementById('field');"
            "f.addEventListener('input', () => window.__events.push('input'));"
            "f.addEventListener('change', () => window.__events.push('change'));"
        )
        await browser.fill("#field", "typed text")
        return (
            await browser.evaluate("document.getElementById('field').value"),
            await browser.evaluate("window.__events"),
        )

    assert scenario(go) == ("typed text", ["input", "change"])


def test_filling_a_missing_input_is_reported(browser):
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        await browser.fill("#nope", "x")

    with pytest.raises(BrowserError) as info:
        scenario(go)
    assert info.value.code == "BROWSER_NO_ELEMENT"


def test_waiting_for_a_late_element(browser):
    """A page that populates after load must be waitable."""
    async def go():
        await browser.attach()
        await browser.navigate(PAGE, wait=0)
        return await browser.wait_for("#late", timeout=5.0)

    assert scenario(go)


def test_waiting_for_something_that_never_arrives_returns_false(browser):
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        return await browser.wait_for("#never", timeout=0.5)

    assert not scenario(go)


def test_a_screenshot_returns_png_bytes(browser):
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        return await browser.screenshot()

    data = scenario(go)
    assert data
    # A PNG in base64 starts with the same magic bytes every time.
    import base64

    assert base64.b64decode(data[:16]).startswith(b"\x89PNG")


def test_opening_a_new_page(browser):
    async def go():
        await browser.attach()
        return await browser.new_page("about:blank")

    target = scenario(go)
    assert target.id
    assert target.websocket_url


# ===========================================================================
# The tools
# ===========================================================================


async def invoke(tool, arguments: dict):
    return await tool.body(arguments, None)


def make_tools(browser):
    return {t.name: t for t in BrowserTools(browser).tools()}


def test_the_browser_tools_exist(browser):
    tools = make_tools(browser)
    assert sorted(tools) == [
        "browser_click",
        "browser_evaluate",
        "browser_fill",
        "browser_navigate",
        "browser_open",
        "browser_screenshot",
        "browser_text",
        "browser_wait_for",
    ]


def test_the_open_tool_attaches_and_navigates(browser):
    tools = make_tools(browser)
    result = scenario(lambda: invoke(tools["browser_open"], {"url": PAGE}))

    assert not result.is_error, result.content
    assert result.value["url"].startswith("data:text/html")


def test_the_text_tool_reads_the_page(browser):
    tools = make_tools(browser)

    async def go():
        await invoke(tools["browser_open"], {"url": PAGE})
        return await invoke(tools["browser_text"], {})

    result = scenario(go)
    assert not result.is_error
    assert "Hello there" in result.content
    assert "Test page" in result.content


def test_the_click_tool_clicks(browser):
    tools = make_tools(browser)

    async def go():
        await invoke(tools["browser_open"], {"url": PAGE})
        return await invoke(tools["browser_click"], {"selector": "#button"})

    result = scenario(go)
    assert not result.is_error, result.content


def test_the_evaluate_tool_returns_json(browser):
    tools = make_tools(browser)

    async def go():
        await invoke(tools["browser_open"], {"url": PAGE})
        return await invoke(
            tools["browser_evaluate"], {"expression": "({a: 1, b: [2, 3]})"}
        )

    result = scenario(go)
    assert not result.is_error
    assert json.loads(result.content) == {"a": 1, "b": [2, 3]}


def test_the_fill_tool_enters_text(browser):
    tools = make_tools(browser)

    async def go():
        await invoke(tools["browser_open"], {"url": PAGE})
        return await invoke(
            tools["browser_fill"], {"selector": "#field", "value": "hi"}
        )

    result = scenario(go)
    assert not result.is_error, result.content


def test_the_screenshot_tool_says_the_image_is_not_shown(browser):
    """Handing back something the model cannot see, while implying it can, is worse
    than saying plainly that it is bytes."""
    tools = make_tools(browser)

    async def go():
        await invoke(tools["browser_open"], {"url": PAGE})
        return await invoke(tools["browser_screenshot"], {})

    result = scenario(go)
    assert not result.is_error
    assert "base64" in result.content
    assert result.value["base64"]


def test_the_wait_tool_reports_a_timeout_as_a_failure(browser):
    tools = make_tools(browser)

    async def go():
        await invoke(tools["browser_open"], {"url": PAGE})
        return await invoke(
            tools["browser_wait_for"], {"selector": "#never", "timeout": 0.3}
        )

    result = scenario(go)
    assert result.is_error
    assert result.error_code == "BROWSER_WAIT_TIMEOUT"


def test_a_tool_reports_an_unreachable_browser(browser):
    """The failure must be a result the model can read, not an exception."""
    instance = Browser("http://127.0.0.1:1")
    tools = make_tools(instance)
    result = run(invoke(tools["browser_open"], {}))

    assert result.is_error
    assert result.error_code == "BROWSER_UNAVAILABLE"


def test_a_tool_refuses_an_empty_argument(browser):
    tools = make_tools(browser)

    cases = (
        ("browser_click", {"selector": "  "}),
        ("browser_fill", {"selector": ""}),
        ("browser_evaluate", {"expression": ""}),
        ("browser_wait_for", {"selector": ""}),
        ("browser_navigate", {"url": ""}),
    )

    async def go():
        await invoke(tools["browser_open"], {"url": PAGE})
        return [(name, await invoke(tools[name], args)) for name, args in cases]

    for name, result in scenario(go):
        assert result.is_error, f"{name} accepted an empty argument"


def test_every_browser_tool_schema_normalizes(browser):
    """The schemas must survive the model-facing subset."""
    tools = make_tools(browser)
    for tool in tools.values():
        rendered = json.dumps(dict(tool.definition.parameters))
        assert "title" not in rendered, tool.definition.name
        assert "$defs" not in rendered, tool.definition.name


def test_reading_tools_are_concurrency_safe_and_acting_tools_are_not(browser):
    """Reading a page and clicking on it are different risk profiles."""
    tools = make_tools(browser)
    assert tools["browser_text"].definition.allows_parallel({})

    for name in ("browser_click", "browser_fill", "browser_navigate", "browser_evaluate"):
        assert not tools[name].definition.allows_parallel({}), name


# ===========================================================================
# Protocol details, without a browser
# ===========================================================================


def test_a_connection_that_is_not_open_refuses_to_send():
    connection = CDPConnection("ws://127.0.0.1:1/devtools/page/x")
    with pytest.raises(BrowserError) as info:
        run(connection.call("Page.enable"))
    assert info.value.code == "BROWSER_NOT_CONNECTED"


def test_a_target_without_a_socket_is_skipped():
    """A target that cannot be driven must not be offered as if it could."""
    assert PageTarget(id="1", title="t", url="u", websocket_url="").websocket_url == ""


def test_a_thrown_non_error_is_also_reported(browser):
    """A bare ``throw "x"`` has no exception object; the message is elsewhere.

    Reading only ``exceptionDetails.text`` would produce "Uncaught" and nothing
    else -- an error that says a script failed and nothing about why.
    """
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        await browser.evaluate("throw 'a plain string'")

    with pytest.raises(BrowserError) as info:
        scenario(go)
    assert "plain string" in str(info.value), (
        f"the thrown value was lost: {info.value}"
    )


def test_the_error_message_includes_the_original_message(browser):
    """The category is not the message."""
    async def go():
        await browser.attach()
        await browser.navigate(PAGE)
        await browser.evaluate("throw new Error('a specific reason')")

    with pytest.raises(BrowserError) as info:
        scenario(go)
    text = str(info.value)
    assert "a specific reason" in text
    assert text != "the page raised an error: Uncaught"

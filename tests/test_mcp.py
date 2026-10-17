"""MCP client tests.

The theme is that **an MCP tool is an ordinary tool**. It registers in the same
registry, flows through the same pipeline, and is subject to the same policy.
Several tests assert that directly, because the alternative -- a second kind of
tool with its own rules -- is what would make a server's permissions impossible
to reason about.

The second theme is **atomicity**. A server's tools appear together or not at
all, and a failed refresh keeps the previous generation. A half-registered server
is worse than an absent one: the model calls a tool the runtime half-believes
exists.

The stdio tests spawn a real Python subprocess speaking JSON-RPC, because the
transport's job is to talk to a process and a mock would verify the mock.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from arm.mcp.client import (
    McpClient,
    McpError,
    McpManager,
    McpServerConfig,
    Namespace,
    ToolIdentity,
    identity_from,
    project_result,
)
from arm.mcp.naming import MAX_PUBLIC_NAME
from arm.mcp.stdio import StdioServerConfig, StdioTransport, connect_stdio
from arm.tools.registry import ToolRegistry


def run(coro):
    return asyncio.run(coro)

def run_all(fn):
    """Run an async scenario in a single event loop.

    The stdio transport holds loop-bound resources -- the subprocess and its
    pipes -- so a scenario that spans several awaits must not be split across
    ``asyncio.run`` calls. Doing so is what produced "Event loop is closed" and
    "attached to a different loop" failures: the transport was created in one
    loop and used from another.
    """
    return asyncio.run(fn())



# ===========================================================================
# Naming
# ===========================================================================


def test_a_simple_name_needs_no_hash():
    """The readable form is used whenever it is faithful."""
    identity = ToolIdentity(server="memory", tool="store")
    assert identity.public_name() == "mcp__memory__store"


def test_the_public_name_never_exceeds_the_wire_limit():
    """The limit is a protocol constant, not a preference."""
    identity = ToolIdentity(server="s" * 80, tool="t" * 80)
    assert len(identity.public_name()) <= MAX_PUBLIC_NAME


def test_a_sanitized_name_carries_a_hash():
    """Sanitizing is lossy, so the hash carries the identity."""
    identity = ToolIdentity(server="my server", tool="do.thing")
    public = identity.public_name()

    assert len(public) <= MAX_PUBLIC_NAME
    assert " " not in public and "." not in public
    # The hash makes the mapping injective even though the readable part is not.
    assert public.count("__") >= 3, public


def test_two_identities_that_sanitize_alike_get_different_names():
    """This is the whole point of hashing the original pair.

    Without it, two distinct tools would collapse into one public name and one of
    them would be unreachable.
    """
    first = ToolIdentity(server="a b", tool="t")
    second = ToolIdentity(server="a-b", tool="t")

    assert first.public_name() != second.public_name(), (
        "two identities sanitized to the same public name"
    )


def test_the_public_name_is_not_parsed_back():
    """The identity is a value, not something recovered from the public name.

    A reversed mapping would break the moment sanitizing changed anything, which
    is why the format is derived one way only.
    """
    identity = ToolIdentity(server="my server", tool="do.thing")
    public = identity.public_name()

    # The readable parts are visibly altered, so no parser could recover them.
    assert "my server" not in public
    assert identity.server == "my server"
    assert identity.tool == "do.thing"


def test_an_empty_server_or_tool_is_refused():
    """A tool with no name is a server bug, not a gap to fill."""
    with pytest.raises(McpError) as info:
        identity_from("", "tool")
    assert info.value.code == "MCP_NO_SERVER"

    with pytest.raises(McpError) as info:
        identity_from("server", "  ")
    assert info.value.code == "MCP_NO_TOOL"


def test_a_namespace_recognizes_its_own_prefix():
    """Used to detect a foreign registration squatting on the prefix."""
    namespace = Namespace("memory")
    assert namespace.owns("mcp__memory__store")
    assert not namespace.owns("mcp__other__store")


def test_a_namespace_refuses_another_servers_identity():
    """Registering the wrong server's tool would cross two namespaces."""
    namespace = Namespace("memory")
    with pytest.raises(McpError):
        namespace.add(ToolIdentity(server="other", tool="store"))


def test_a_namespace_tracks_its_tools():
    namespace = Namespace("memory")
    namespace.add(ToolIdentity(server="memory", tool="store"))
    namespace.add(ToolIdentity(server="memory", tool="recall"))

    assert len(namespace) == 2
    assert {i.tool for i in namespace.all()} == {"store", "recall"}


# ===========================================================================
# Result projection
# ===========================================================================


def identity() -> ToolIdentity:
    return ToolIdentity(server="s", tool="t")


def test_text_content_is_projected():
    result = project_result(
        {"content": [{"type": "text", "text": "hello"}]}, identity=identity()
    )
    assert not result.is_error
    assert result.content == "hello"


def test_multiple_text_blocks_are_joined():
    result = project_result(
        {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        identity=identity(),
    )
    assert result.content == "a\nb"


def test_is_error_becomes_a_failed_result():
    """The model must see a remote failure as a failure."""
    result = project_result(
        {"isError": True, "content": [{"type": "text", "text": "it broke"}]},
        identity=identity(),
    )
    assert result.is_error
    assert "it broke" in result.content


def test_an_image_degrades_with_a_note_rather_than_vanishing():
    """Silently dropping it would leave the model believing it had seen one."""
    result = project_result(
        {"content": [{"type": "image", "mimeType": "image/png", "data": "..."}]},
        identity=identity(),
    )
    assert not result.is_error
    assert "image" in result.content
    assert "image/png" in result.content
    assert "could not be shown" in result.content


def test_an_unknown_block_type_is_named():
    """A server adding a block type must produce a visible note, not silence."""
    result = project_result(
        {"content": [{"type": "hologram", "data": "..."}]}, identity=identity()
    )
    assert "hologram" in result.content


def test_an_empty_content_list_is_an_empty_result():
    """Legal, and it means "nothing to say"."""
    result = project_result({"content": []}, identity=identity())
    assert not result.is_error
    assert result.content == ""


def test_a_missing_content_field_is_an_empty_result():
    result = project_result({}, identity=identity())
    assert not result.is_error
    assert result.content == ""


def test_a_malformed_content_field_is_a_failure():
    result = project_result({"content": 42}, identity=identity())
    assert result.is_error
    assert result.error_code == "MCP_BAD_RESULT"


def test_audio_and_resource_blocks_degrade():
    """Each unsupported type says what it was, so nothing disappears silently."""
    result = project_result(
        {
            "content": [
                {"type": "audio", "data": "..."},
                {"type": "resource_link", "uri": "file:///x"},
            ]
        },
        identity=identity(),
    )
    assert "audio" in result.content
    assert "resource_link" in result.content


# ===========================================================================
# The client, against a fake transport
# ===========================================================================


class FakeTransport:
    """An in-memory MCP server, for testing the bridge without a process."""

    def __init__(
        self,
        tools: list[dict] | None = None,
        *,
        instructions: str = "",
        fail_listing: bool = False,
        call_result: dict | None = None,
    ) -> None:
        self.tools = tools if tools is not None else []
        self.instructions = instructions
        self.fail_listing = fail_listing
        self.call_result = call_result or {"content": [{"type": "text", "text": "ok"}]}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def request(self, method, params):
        self.calls.append((method, dict(params)))
        if method == "tools/list":
            if self.fail_listing:
                raise McpError("the listing failed", code="MCP_SERVER_ERROR")
            return {"tools": self.tools}
        if method == "tools/call":
            return self.call_result
        if method == "instructions":
            return {"instructions": self.instructions}
        return {}

    async def close(self):
        self.closed = True


def make_client(transport, *, name="test", **kwargs):
    registry = ToolRegistry()
    config = McpServerConfig(name=name, connect=lambda: _ready(transport), **kwargs)
    client = McpClient(config, registry)
    return client, registry


async def _ready(transport):
    return transport


def test_connecting_registers_the_servers_tools():
    """The ordinary path."""
    transport = FakeTransport(
        [{"name": "store", "description": "store a value", "inputSchema": {}}]
    )
    client, registry = make_client(transport)
    run(client.connect())

    assert "mcp__test__store" in registry.view().known
    assert client.connected


def test_the_tool_carries_the_servers_description():
    transport = FakeTransport(
        [{"name": "store", "description": "store a value", "inputSchema": {}}]
    )
    client, registry = make_client(transport)
    run(client.connect())

    definition = registry.view().get("mcp__test__store")
    assert definition is not None
    assert definition.description == "store a value"


def test_calling_a_tool_sends_the_raw_name():
    """The public name is a sanitized derivation the server knows nothing about."""
    transport = FakeTransport([{"name": "do.thing", "inputSchema": {}}])
    client, registry = make_client(transport)
    run(client.connect())

    # The name was sanitized (a '.' became '_'), so the public name carries a
    # hash. That is the design: the readable part is lossy, the hash is not.
    known = registry.view().known
    assert len(known) == 1
    public = next(iter(known))
    assert public.startswith("mcp__test__do_thing")
    tool = registry.get(public)
    assert tool is not None
    run(tool.body({"x": 1}, None))

    call = [c for c in transport.calls if c[0] == "tools/call"][0]
    assert call[1]["name"] == "do.thing", (
        f"the public name was sent instead of the raw one: {call[1]['name']}"
    )
    assert call[1]["arguments"] == {"x": 1}


def test_a_tool_call_failure_is_a_result():
    """A remote error must not raise into the pipeline."""
    transport = FakeTransport([{"name": "t", "inputSchema": {}}])

    async def failing(method, params):
        if method == "tools/list":
            return {"tools": [{"name": "t", "inputSchema": {}}]}
        raise McpError("the server is down", code="MCP_SERVER_ERROR")

    transport.request = failing  # type: ignore[method-assign]
    client, registry = make_client(transport)
    run(client.connect())

    tool = registry.get("mcp__test__t")
    assert tool is not None
    result = run(tool.body({}, None))
    assert result.is_error
    assert result.error_code == "MCP_TRANSPORT_ERROR"


def test_a_timeout_is_a_result():
    """A wedged server must not hang the turn."""
    transport = FakeTransport([{"name": "t", "inputSchema": {}}])

    async def slow(method, params):
        if method == "tools/list":
            return {"tools": [{"name": "t", "inputSchema": {}}]}
        await asyncio.sleep(5)
        return {}

    transport.request = slow  # type: ignore[method-assign]
    client, registry = make_client(transport, call_timeout=0.05)
    run(client.connect())

    tool = registry.get("mcp__test__t")
    assert tool is not None
    result = run(tool.body({}, None))
    assert result.is_error
    assert result.error_code == "MCP_TIMEOUT"


def test_a_failed_refresh_keeps_the_previous_generation():
    """Tools that worked a moment ago must not vanish during a reconnect.

    Unregistering first would make a transient network error look like the server
    losing all its tools, and the model would see a tool disappear mid-task.
    """
    transport = FakeTransport([{"name": "keep", "inputSchema": {}}])
    client, registry = make_client(transport)
    run(client.connect())
    assert "mcp__test__keep" in registry.view().known

    transport.fail_listing = True
    with pytest.raises(McpError):
        run(client.refresh())

    assert "mcp__test__keep" in registry.view().known, (
        "the previous generation was dropped by a failed refresh"
    )


def test_a_refresh_replaces_the_generation():
    """New tools appear and old ones go, in one step."""
    transport = FakeTransport([{"name": "old", "inputSchema": {}}])
    client, registry = make_client(transport)
    run(client.connect())

    transport.tools = [{"name": "new", "inputSchema": {}}]
    run(client.refresh())

    known = registry.view().known
    assert "mcp__test__new" in known
    assert "mcp__test__old" not in known


def test_closing_unregisters_the_tools():
    """A server that is gone must not leave callable tools behind."""
    transport = FakeTransport([{"name": "t", "inputSchema": {}}])
    client, registry = make_client(transport)
    run(client.connect())

    run(client.close())
    assert "mcp__test__t" not in registry.view().known
    assert transport.closed


def test_an_optional_server_that_fails_contributes_nothing():
    """And does not take the runtime down with it."""
    async def broken():
        raise McpError("cannot connect", code="MCP_CONNECT_FAILED")

    registry = ToolRegistry()
    client = McpClient(
        McpServerConfig(name="optional", connect=broken, required=False), registry
    )
    run(client.connect())  # must not raise

    assert registry.view().known == frozenset()
    assert not client.connected


def test_a_required_server_that_fails_raises():
    """A server that is required and missing should say so at startup.

    Silently contributing nothing would be the dishonest option: the deployment
    believes the server is present.
    """
    async def broken():
        raise McpError("cannot connect", code="MCP_CONNECT_FAILED")

    registry = ToolRegistry()
    client = McpClient(
        McpServerConfig(name="required", connect=broken, required=True), registry
    )
    with pytest.raises(McpError) as info:
        run(client.connect())
    assert info.value.code == "MCP_CONNECT_FAILED"


def test_instructions_are_captured():
    """A server's prompt section reaches the runtime."""
    transport = FakeTransport([], instructions="Always cite your sources.")
    captured: list[tuple[str, str]] = []
    registry = ToolRegistry()
    client = McpClient(
        McpServerConfig(name="docs", connect=lambda: _ready(transport)),
        registry,
        on_instructions=lambda name, text: captured.append((name, text)),
    )
    run(client.connect())

    assert captured == [("docs", "Always cite your sources.")]


def test_oversized_instructions_are_refused_not_truncated():
    """A half-stated policy is worse than none: the model acts on half of it."""
    transport = FakeTransport([], instructions="x" * 2000)
    registry = ToolRegistry()
    client = McpClient(
        McpServerConfig(
            name="loud", connect=lambda: _ready(transport), max_instruction_bytes=100
        ),
        registry,
    )
    run(client.connect())

    assert client.instructions == "", "oversized instructions were kept anyway"


def test_the_manager_connects_every_server():
    registry = ToolRegistry()
    manager = McpManager(registry)
    first = FakeTransport([{"name": "a", "inputSchema": {}}])
    second = FakeTransport([{"name": "b", "inputSchema": {}}])
    manager.register(McpServerConfig(name="one", connect=lambda: _ready(first)))
    manager.register(McpServerConfig(name="two", connect=lambda: _ready(second)))

    failures = run(manager.connect_all())

    assert failures == []
    assert "mcp__one__a" in registry.view().known
    assert "mcp__two__b" in registry.view().known


def test_one_servers_failure_does_not_stop_the_others():
    """An unavailable optional server must not take the runtime down."""
    registry = ToolRegistry()
    manager = McpManager(registry)

    async def broken():
        raise McpError("down", code="MCP_CONNECT_FAILED")

    good = FakeTransport([{"name": "a", "inputSchema": {}}])
    manager.register(McpServerConfig(name="bad", connect=broken))
    manager.register(McpServerConfig(name="good", connect=lambda: _ready(good)))

    failures = run(manager.connect_all())

    assert failures == ["bad"]
    assert "mcp__good__a" in registry.view().known


def test_a_required_servers_failure_is_identifiable():
    """The caller must be able to tell which failures were unacceptable.

    A required server that failed silently would leave the deployment believing
    it is present -- which is the dishonest outcome.
    """
    registry = ToolRegistry()
    manager = McpManager(registry)

    async def broken():
        raise McpError("down", code="MCP_CONNECT_FAILED")

    manager.register(McpServerConfig(name="needed", connect=broken, required=True))
    manager.register(McpServerConfig(name="optional", connect=broken))

    failures = run(manager.connect_all())

    assert sorted(failures) == ["needed", "optional"]
    assert manager.required_failures(failures) == ["needed"], (
        "a required server's failure was not distinguishable from an optional one"
    )


def test_two_servers_with_one_name_are_refused():
    """Two servers sharing a name would share a namespace."""
    manager = McpManager(ToolRegistry())
    manager.register(McpServerConfig(name="dup", connect=lambda: _ready(FakeTransport())))

    with pytest.raises(McpError) as info:
        manager.register(
            McpServerConfig(name="dup", connect=lambda: _ready(FakeTransport()))
        )
    assert info.value.code == "MCP_DUPLICATE_SERVER"


def test_the_prompt_section_attributes_each_server():
    """A merged blob would make an instruction impossible to attribute."""
    registry = ToolRegistry()
    manager = McpManager(registry)
    manager.register(
        McpServerConfig(
            name="alpha", connect=lambda: _ready(FakeTransport([], instructions="A"))
        )
    )
    manager.register(
        McpServerConfig(
            name="beta", connect=lambda: _ready(FakeTransport([], instructions="B"))
        )
    )
    run(manager.connect_all())

    section = manager.prompt_section()
    assert "### MCP server: alpha" in section
    assert "### MCP server: beta" in section
    assert section.index("alpha") < section.index("beta")


def test_no_instructions_means_no_section():
    """An empty section would spend tokens saying nothing."""
    registry = ToolRegistry()
    manager = McpManager(registry)
    manager.register(McpServerConfig(name="quiet", connect=lambda: _ready(FakeTransport())))
    run(manager.connect_all())

    assert manager.prompt_section() == ""


def test_an_mcp_tool_is_an_ordinary_tool():
    """The central claim: no second kind of tool, no separate permission model."""
    transport = FakeTransport([{"name": "t", "inputSchema": {}}])
    client, registry = make_client(transport)
    run(client.connect())

    tool = registry.get("mcp__test__t")
    assert tool is not None
    # It is a plain RegisteredTool with a plain definition.
    assert tool.definition.name.startswith("mcp__")
    # And it carries no server identity or trust level that policy could key on.
    assert "server" not in dict(tool.definition.meta)
    assert "trust" not in dict(tool.definition.meta)


def test_the_schema_reaches_the_model_unchanged():
    """The remote server is the authority on what its tools accept.

    Asserted on the **wire** form rather than the stored one: the registry freezes
    parameters into immutable structures, which turns JSON arrays into tuples.
    That is internal -- ``to_wire`` restores arrays -- so comparing the stored form
    would test the freezing rather than the pass-through.
    """
    from arm.tools.schema import to_wire

    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    transport = FakeTransport([{"name": "t", "inputSchema": schema}])
    client, registry = make_client(transport)
    run(client.connect())

    definition = registry.view().get("mcp__test__t")
    assert definition is not None
    assert to_wire(definition.parameters) == schema


# ===========================================================================
# The stdio transport, against a real subprocess
# ===========================================================================

SERVER_SCRIPT = textwrap.dedent(
    '''
    """A minimal MCP server speaking JSON-RPC on stdin/stdout."""
    import json
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")
        if method == "tools/list":
            result = {"tools": [
                {"name": "echo", "description": "echo it back",
                 "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
            ]}
        elif method == "tools/call":
            args = message.get("params", {}).get("arguments", {})
            result = {"content": [{"type": "text", "text": "echo:" + str(args.get("text", ""))}]}
        elif method == "instructions":
            result = {"instructions": "From the echo server."}
        else:
            result = {}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\\n")
        sys.stdout.flush()
    '''
)


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    path = tmp_path / "server.py"
    path.write_text(SERVER_SCRIPT)
    return path


def test_the_stdio_transport_talks_to_a_real_process(server_script: Path):
    """A mock would verify the mock; this spawns a real server."""

    async def scenario():
        registry = ToolRegistry()
        config = McpServerConfig(
            name="echo",
            connect=lambda: connect_stdio(
                StdioServerConfig(command=sys.executable, args=[str(server_script)])
            ),
        )
        client = McpClient(config, registry)
        await client.connect()

        assert "mcp__echo__echo" in registry.view().known

        tool = registry.get("mcp__echo__echo")
        assert tool is not None
        result = await tool.body({"text": "hi"}, None)
        await client.close()
        return result

    result = run_all(scenario)
    assert not result.is_error
    assert result.content == "echo:hi"


def test_the_stdio_transport_captures_instructions(server_script: Path):
    async def scenario():
        registry = ToolRegistry()
        client = McpClient(
            McpServerConfig(
                name="echo",
                connect=lambda: connect_stdio(
                    StdioServerConfig(command=sys.executable, args=[str(server_script)])
                ),
            ),
            registry,
        )
        await client.connect()
        instructions = client.instructions
        await client.close()
        return instructions

    assert run_all(scenario) == "From the echo server."


def test_a_missing_command_is_reported_clearly():
    """The failure must name the command, not just say it did not work."""
    async def scenario():
        transport = StdioTransport(
            StdioServerConfig(command="/nonexistent/definitely-not")
        )
        await transport.start()

    with pytest.raises(McpError) as info:
        run_all(scenario)
    assert info.value.code == "MCP_COMMAND_NOT_FOUND"
    assert "definitely-not" in str(info.value)


def test_a_server_that_exits_immediately_is_reported(tmp_path: Path):
    """With its stderr, which is where the reason usually is."""
    script = tmp_path / "crash.py"
    script.write_text("import sys\nsys.stderr.write('I cannot start\\n')\nsys.exit(1)\n")

    async def scenario():
        transport = StdioTransport(
            StdioServerConfig(command=sys.executable, args=[str(script)])
        )
        await transport.start()
        # Give the process a moment to exit and stderr to drain.
        await asyncio.sleep(0.3)
        try:
            await transport.request("tools/list", {})
        except McpError as exc:
            return exc
        return None

    error = run_all(scenario)
    assert error is not None
    assert error.code in ("MCP_SERVER_EXITED", "MCP_BROKEN_PIPE")
    assert "I cannot start" in str(error), (
        f"the server's stderr was not surfaced: {error}"
    )


def test_the_environment_is_not_inherited_wholesale(tmp_path: Path):
    """A server is not entitled to every credential the runtime holds."""
    script = tmp_path / "env.py"
    script.write_text(
        "import os, sys\n"
        "sys.stderr.write('SECRET=' + os.environ.get('A_TEST_SECRET', 'ABSENT') + '\\n')\n"
        "sys.exit(1)\n"
    )

    import os

    async def scenario():
        transport = StdioTransport(
            StdioServerConfig(command=sys.executable, args=[str(script)])
        )
        await transport.start()
        await asyncio.sleep(0.3)
        text = transport.stderr_text
        await transport.close()
        return text

    os.environ["A_TEST_SECRET"] = "should-not-leak"
    try:
        stderr = run_all(scenario)
    finally:
        del os.environ["A_TEST_SECRET"]

    assert "should-not-leak" not in stderr, (
        "the parent's environment leaked into the server process"
    )
    assert "ABSENT" in stderr


def test_configured_environment_variables_are_passed(tmp_path: Path):
    """The env field is the permission boundary, so it must actually work."""
    script = tmp_path / "env2.py"
    script.write_text(
        "import os, sys\n"
        "sys.stderr.write('GOT=' + os.environ.get('PROVIDED', 'ABSENT') + '\\n')\n"
        "sys.exit(1)\n"
    )
    async def scenario():
        transport = StdioTransport(
            StdioServerConfig(
                command=sys.executable,
                args=[str(script)],
                env={"PROVIDED": "yes"},
            )
        )
        await transport.start()
        await asyncio.sleep(0.3)
        text = transport.stderr_text
        await transport.close()
        return text

    assert "GOT=yes" in run_all(scenario)


def test_a_server_that_never_responds_does_not_hang(tmp_path: Path):
    """A wedged server must be abandoned, not waited on forever."""
    script = tmp_path / "wedged.py"
    script.write_text("import time\ntime.sleep(60)\n")

    async def scenario():
        transport = StdioTransport(
            StdioServerConfig(command=sys.executable, args=[str(script)])
        )
        await transport.start()
        try:
            try:
                await asyncio.wait_for(transport.request("tools/list", {}), timeout=0.5)
                outcome = "returned"
            except asyncio.TimeoutError:
                outcome = "timed out as expected"
        finally:
            await transport.close()
        return outcome

    assert run_all(scenario) == "timed out as expected"


def test_a_notification_before_the_response_is_skipped(tmp_path: Path):
    """A server may emit progress messages; the first line is not the response."""
    script = tmp_path / "chatty.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    rid = message.get('id')\n"
        "    # A notification with no id, then the real response.\n"
        "    sys.stdout.write(json.dumps({'jsonrpc':'2.0','method':'progress'}) + '\\n')\n"
        "    sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':rid,'result':{'tools':[]}}) + '\\n')\n"
        "    sys.stdout.flush()\n"
    )
    async def scenario():
        transport = StdioTransport(
            StdioServerConfig(command=sys.executable, args=[str(script)])
        )
        await transport.start()
        result = await transport.request("tools/list", {})
        await transport.close()
        return result

    assert run_all(scenario) == {"tools": []}


def test_a_jsonrpc_error_is_raised_with_the_servers_message(tmp_path: Path):
    """The server's own message is the useful part of the failure."""
    script = tmp_path / "errors.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    rid = message.get('id')\n"
        "    out = {'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'no such method'}}\n"
        "    sys.stdout.write(json.dumps(out) + '\\n')\n"
        "    sys.stdout.flush()\n"
    )
    async def scenario():
        transport = StdioTransport(
            StdioServerConfig(command=sys.executable, args=[str(script)])
        )
        await transport.start()
        try:
            await transport.request("tools/nope", {})
        except McpError as exc:
            return exc
        finally:
            await transport.close()
        return None

    error = run_all(scenario)
    assert error is not None
    assert "no such method" in str(error)
    assert error.code == "MCP_SERVER_ERROR"


def test_closing_terminates_the_process(server_script: Path):
    """Terminating without waiting would leave a zombie."""
    async def scenario():
        transport = StdioTransport(
            StdioServerConfig(command=sys.executable, args=[str(server_script)])
        )
        await transport.start()
        started = transport._process is not None
        await transport.close()
        return started, transport._process is None

    started, stopped = run_all(scenario)
    assert started
    assert stopped


# ===========================================================================
# The CLI must be able to use an MCP server it started
# ===========================================================================


def test_a_server_connected_on_one_loop_fails_on_another(tmp_path: Path):
    """The defect, reproduced at its source.

    The CLI connected servers with `asyncio.run(mcp_manager.connect_all())` and
    then ran the session with a second `asyncio.run`. The transport holds
    loop-bound resources -- the subprocess and its pipes -- so every tool call
    failed:

        got Future <Future pending> attached to a different loop

    The server had started and its tool was registered, which is what made it
    look healthy: a registration check would pass while every invocation failed.

    This asserts the failure mode exists rather than mocking it away, so the fix
    cannot regress into "it connects" without "it works".
    """
    server = tmp_path / "server.py"
    server.write_text(SERVER_SCRIPT)

    registry = ToolRegistry()
    manager = McpManager(registry)
    manager.register(
        McpServerConfig(
            name="echo",
            connect=lambda: connect_stdio(
                StdioServerConfig(command=sys.executable, args=[str(server)])
            ),
        )
    )

    # Loop A: connect, then let it close -- exactly what the CLI did.
    asyncio.run(manager.connect_all())
    assert "mcp__echo__echo" in registry.view().known, "the tool was not registered"

    # Loop B: call it. This is where the CLI failed.
    async def call_on_another_loop():
        tool = registry.get("mcp__echo__echo")
        assert tool is not None
        return await tool.body({"text": "hi"}, None)

    result = asyncio.run(call_on_another_loop())
    assert result.is_error, (
        "a transport connected on a closed loop unexpectedly worked; if this is "
        "now supported, the CLI no longer needs to connect inside the run loop"
    )
    assert "different loop" in result.content


def test_connecting_and_calling_on_the_same_loop_works(tmp_path: Path):
    """The fix, stated as the property it provides."""
    server = tmp_path / "server.py"
    server.write_text(SERVER_SCRIPT)

    registry = ToolRegistry()
    manager = McpManager(registry)
    manager.register(
        McpServerConfig(
            name="echo",
            connect=lambda: connect_stdio(
                StdioServerConfig(command=sys.executable, args=[str(server)])
            ),
        )
    )

    async def scenario():
        failures = await manager.connect_all()
        assert failures == [], failures
        tool = registry.get("mcp__echo__echo")
        assert tool is not None
        return await tool.body({"text": "hi"}, None)

    result = asyncio.run(scenario())
    assert not result.is_error, result.content
    assert result.content == "echo:hi"

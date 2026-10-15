"""The MCP client: a remote server's tools, registered as ordinary tools.

The single most important design fact is what this client does **not** do:

    it does not create a new kind of tool.

An MCP tool is registered in the same registry, flows through the same seven-stage
pipeline, and is subject to the same approval and sandbox policy as a native one.
There is no server identity on the definition, no trust level, and no allowlist
field. Adding a server is therefore exactly equivalent to adding a set of native
tools with the same permissions -- which is the honest description, and the one a
user needs in order to reason about what they are installing.

Two consequences follow, and both are implemented here rather than left to
convention:

**A generation is atomic.** A server's tools are registered together or not at
all. A half-registered server is worse than an absent one: the model would call a
tool the runtime half-believes exists, and the failure would look like a bug in
the tool rather than in the registration.

**A failed refresh keeps the previous generation.** Tools that worked a moment ago
must not vanish because a reconnect is in progress. The alternative -- unregister
on failure and re-register on success -- would make a server's tool set flicker,
and the model would see a tool disappear mid-task for no reason it could observe.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from ..kernel.errors import describe
from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from ..tools.registry import ToolRegistry
from .naming import McpError, Namespace, ToolIdentity, identity_from

log = logging.getLogger("arm.mcp")

#: How long a single ``tools/call`` may take before it is abandoned.
DEFAULT_CALL_TIMEOUT = 60.0

#: The ceiling on a server's instructions. A server that sends an unbounded
#: prompt section is a server that can spend the whole context budget.
DEFAULT_MAX_INSTRUCTION_BYTES = 32_768


class McpTransport(Protocol):
    """What a transport must do: send a request, receive a response.

    Declared as a protocol so a stdio subprocess, an HTTP endpoint and a test
    double are the same shape to the client. The client never learns how the
    bytes travel.
    """

    async def request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send a JSON-RPC request and return its result."""
        ...

    async def close(self) -> None:
        ...


@dataclass
class McpServerConfig:
    """How to reach one server, and how to behave when it is unreachable."""

    name: str
    #: The transport factory, injected so the client does not know the mechanism.
    connect: Callable[[], Awaitable[McpTransport]]
    call_timeout: float = DEFAULT_CALL_TIMEOUT
    max_instruction_bytes: int = DEFAULT_MAX_INSTRUCTION_BYTES
    #: Whether a failure to start should refuse the whole server. A server that
    #: is required and missing should say so at startup, not silently contribute
    #: nothing.
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise McpError("an MCP server needs a name", code="MCP_NO_SERVER")
        if self.call_timeout <= 0:
            raise McpError("call_timeout must be positive")
        if self.max_instruction_bytes <= 0:
            raise McpError("max_instruction_bytes must be positive")


@dataclass
class McpToolSpec:
    """A tool as the server describes it."""

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)


class McpClient:
    """One MCP server, bridged into a tool registry.

    Holds a live connection and a registered generation. A refresh replaces the
    generation atomically; a failure keeps the previous one.
    """

    def __init__(
        self,
        config: McpServerConfig,
        registry: ToolRegistry,
        *,
        on_instructions: Callable[[str, str], None] | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.namespace = Namespace(config.name)
        self._transport: McpTransport | None = None
        #: Disposers for the currently registered generation.
        self._disposers: list[Callable[[], None]] = []
        self._instructions = ""
        self._on_instructions = on_instructions
        #: Why the last connect attempt failed, if it did. Recorded so a caller
        #: can report which servers are absent rather than guessing.
        self.last_error: McpError | None = None

    # -- lifecycle -------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection and register the server's tools.

        A failure is reported through ``required``: a required server raises, an
        optional one logs and contributes nothing. Both are honest -- what is
        dishonest is an optional server that appears to be present.
        """
        try:
            self._transport = await self.config.connect()
            await self.refresh()
        except Exception as exc:  # noqa: BLE001 - a failed connect is a state
            self._transport = None
            error = McpError(
                f"could not connect to MCP server {self.config.name!r}: "
                f"{describe(exc)}",
                code="MCP_CONNECT_FAILED",
            )
            if self.config.required:
                raise error from exc
            # An optional server that fails is logged *and recorded*. Raising here
            # would stop the manager from connecting the servers after it, and
            # swallowing it silently would leave the caller unable to report which
            # servers are absent -- both are worse than reporting a fact.
            log.warning("MCP server %s is unavailable: %s", self.config.name, describe(exc))
            self.last_error = error

    async def close(self) -> None:
        """Unregister this server's tools and close the connection."""
        self._unregister_all()
        if self._transport is not None:
            try:
                await self._transport.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("closing %s failed: %s", self.config.name, exc)
            self._transport = None

    @property
    def connected(self) -> bool:
        return self._transport is not None

    @property
    def instructions(self) -> str:
        return self._instructions

    # -- the tool bridge -------------------------------------------------

    async def refresh(self) -> int:
        """Re-fetch the server's tool list and replace the generation.

        The new generation is built completely before the old one is released, so
        a failure partway through leaves the previous tools in place. Releasing
        first would make a transient network error look like the server losing all
        its tools.
        """
        if self._transport is None:
            raise McpError(
                f"MCP server {self.config.name!r} is not connected",
                code="MCP_NOT_CONNECTED",
            )

        listing = await self._transport.request("tools/list", {})
        raw_tools = listing.get("tools") or []
        if not isinstance(raw_tools, Sequence):
            raise McpError(
                f"{self.config.name!r} returned a malformed tool list",
                code="MCP_BAD_LISTING",
            )

        specs = [
            McpToolSpec(
                name=str(tool.get("name", "")),
                description=str(tool.get("description", "")),
                input_schema=dict(tool.get("inputSchema") or {}),
            )
            for tool in raw_tools
            if isinstance(tool, Mapping) and tool.get("name")
        ]

        # Build the whole generation first: identities, names, definitions.
        built: list[tuple[ToolIdentity, RegisteredTool]] = []
        for spec in specs:
            identity = identity_from(self.config.name, spec.name)
            public = identity.public_name()

            # A foreign registration on our prefix would shadow our tools, so it
            # is refused rather than coexisting.
            for existing in self.registry.view().known:
                if existing == public:
                    raise McpError(
                        f"the public name {public!r} is already taken by another "
                        "tool; the server's namespace is not ours alone",
                        code="MCP_NAMESPACE_CONFLICT",
                    )

            built.append((identity, self._build_tool(identity, spec)))

        # Commit: register the new generation, then release the old one.
        new_disposers: list[Callable[[], None]] = []
        try:
            for _identity, tool in built:
                new_disposers.append(self.registry.register(tool))
        except Exception:
            # Roll the partial generation back, so the registry is never left
            # holding half of a server.
            for dispose in new_disposers:
                dispose()
            raise

        old = self._disposers
        self._disposers = new_disposers
        for dispose in old:
            dispose()

        for identity, _tool in built:
            self.namespace.add(identity)

        await self._fetch_instructions()
        return len(built)

    def _build_tool(self, identity: ToolIdentity, spec: McpToolSpec) -> RegisteredTool:
        """Wrap a remote tool as an ordinary registered tool."""
        public = identity.public_name()

        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            return await self._call(identity, arguments)

        # The schema is passed through untouched. Translating it would be a
        # second implementation of the schema rules, and the remote server is the
        # authority on what its own tools accept.
        schema = dict(spec.input_schema) or {"type": "object", "properties": {}}

        return RegisteredTool(
            definition=ToolDefinition(
                name=public,
                description=(
                    spec.description
                    or f"A tool provided by the MCP server {self.config.name!r}."
                ),
                parameters=schema,
            ),
            body=body,
        )

    async def _call(
        self, identity: ToolIdentity, arguments: Mapping[str, Any]
    ) -> ToolResult:
        """Invoke a remote tool and project its result.

        The **raw** tool name is sent, never the public one: the public name is a
        sanitized derivation, and the server knows nothing about it.
        """
        if self._transport is None:
            return ToolResult.fail(
                f"MCP server {self.config.name!r} is not connected",
                code="MCP_NOT_CONNECTED",
            )

        try:
            response = await asyncio.wait_for(
                self._transport.request(
                    "tools/call",
                    {"name": identity.tool, "arguments": dict(arguments)},
                ),
                timeout=self.config.call_timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult.fail(
                f"the MCP tool {identity.describe()} timed out after "
                f"{self.config.call_timeout}s",
                code="MCP_TIMEOUT",
            )
        except Exception as exc:  # noqa: BLE001 - a transport failure is a result
            return ToolResult.fail(
                f"the MCP tool {identity.describe()} failed: {describe(exc)}",
                code="MCP_TRANSPORT_ERROR",
            )

        return project_result(response, identity=identity)

    async def _fetch_instructions(self) -> None:
        """Fetch the server's prompt section, bounded.

        A server that sends an unbounded section can spend the whole context
        budget, so the ceiling is enforced rather than trusted.
        """
        if self._transport is None:
            return
        try:
            response = await self._transport.request("instructions", {})
        except Exception as exc:  # noqa: BLE001 - instructions are optional
            log.debug("no instructions from %s: %s", self.config.name, exc)
            return

        text = str(response.get("instructions", "") or "")
        if not text:
            return
        size = len(text.encode("utf-8"))
        if size > self.config.max_instruction_bytes:
            # Refused rather than truncated: a truncated instruction set is a
            # half-stated policy, and a model acting on half a policy is worse
            # than one that was never told.
            log.warning(
                "instructions from %s are %d bytes, over the %d limit; ignoring them",
                self.config.name,
                size,
                self.config.max_instruction_bytes,
            )
            return

        self._instructions = text
        if self._on_instructions is not None:
            self._on_instructions(self.config.name, text)

    def _unregister_all(self) -> None:
        for dispose in self._disposers:
            dispose()
        self._disposers = []
        self.namespace = Namespace(self.config.name)


def project_result(
    response: Mapping[str, Any], *, identity: ToolIdentity
) -> ToolResult:
    """Translate an MCP result into a tool result.

    The projection is where the two protocols disagree, and each disagreement is
    resolved explicitly:

    * ``isError: true`` becomes a failed result, so the model sees it as one;
    * a ``text`` block becomes text;
    * an ``image`` block is **not** decoded here -- the runtime has no attachment
      store yet, so it degrades to a diagnostic line rather than being dropped.
      Silently discarding an image would leave the model believing it had seen
      one;
    * an unknown block type degrades the same way, with its type named, so a
      server adding a new block type produces a visible note rather than silence.
    """
    if response.get("isError"):
        text = _flatten_content(response.get("content")) or "the tool reported an error"
        return ToolResult.fail(text, code="MCP_TOOL_ERROR")

    content = response.get("content")
    if content is None:
        # A result with no content is legal and means "nothing to say".
        return ToolResult.ok("")

    if isinstance(content, str):
        return ToolResult.ok(content)

    if not isinstance(content, Sequence):
        return ToolResult.fail(
            f"the MCP tool {identity.describe()} returned a malformed content field",
            code="MCP_BAD_RESULT",
        )

    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            parts.append(f"[unreadable content block: {type(block).__name__}]")
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "image":
            mime = str(block.get("mimeType", "unknown"))
            parts.append(
                f"[an image was returned ({mime}) but this runtime has no "
                "attachment store, so it could not be shown]"
            )
        elif kind in ("audio", "resource", "resource_link"):
            parts.append(
                f"[a {kind} was returned but cannot be rendered here]"
            )
        else:
            parts.append(f"[unsupported content type: {kind!r}]")

    return ToolResult.ok("\n".join(p for p in parts if p))


def _flatten_content(content: Any) -> str:
    """Best-effort text from a content field, for an error message."""
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        return ""
    out: list[str] = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text":
            out.append(str(block.get("text", "")))
    return "\n".join(out)


class McpManager:
    """Owns a set of servers for one runtime."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._clients: dict[str, McpClient] = {}
        self._instructions: dict[str, str] = {}

    def register(self, config: McpServerConfig) -> McpClient:
        """Add a server. The connection is opened by :meth:`connect_all`."""
        if config.name in self._clients:
            raise McpError(
                f"a server named {config.name!r} is already registered",
                code="MCP_DUPLICATE_SERVER",
            )
        client = McpClient(
            config, self.registry, on_instructions=self._record_instructions
        )
        self._clients[config.name] = client
        return client

    def _record_instructions(self, server: str, text: str) -> None:
        self._instructions[server] = text

    async def connect_all(self) -> list[str]:
        """Connect every server, returning the names that failed.

        A **required** server's failure is collected like any other rather than
        raised here, so one required server does not prevent the others from
        starting. The caller decides what to do with the list -- and a caller that
        ignores it has chosen to ignore a required server, which is its own
        decision rather than one made for it by a loop that stopped early.
        """
        failures: list[str] = []
        for name, client in self._clients.items():
            try:
                await client.connect()
            except Exception as exc:  # noqa: BLE001
                log.warning("MCP server %s failed to start: %s", name, exc)
                failures.append(name)
                continue
            # An optional server reports failure by *not* being connected rather
            # than by raising, so it is checked explicitly. Otherwise an optional
            # server that failed would be indistinguishable from one that is
            # simply quiet -- and the caller could not report it.
            if not client.connected:
                failures.append(name)
        return failures

    def required_failures(self, failures: Sequence[str]) -> list[str]:
        """Which of these failures were servers the deployment requires."""
        return [
            name
            for name in failures
            if (client := self._clients.get(name)) is not None
            and client.config.required
        ]

    async def close_all(self) -> None:
        for client in self._clients.values():
            await client.close()

    def instructions(self) -> dict[str, str]:
        """Each server's prompt section, keyed by server name."""
        return dict(self._instructions)

    def prompt_section(self) -> str:
        """The combined prompt contribution from every server.

        Prefixed per server so a reader can tell which server said what. A merged
        blob would make an instruction impossible to attribute, and attribution is
        the first thing needed when one turns out to be wrong.
        """
        if not self._instructions:
            return ""
        parts: list[str] = []
        for name in sorted(self._instructions):
            parts.append(f"### MCP server: {name}\n\n{self._instructions[name]}")
        return "\n\n".join(parts)

    @property
    def servers(self) -> tuple[str, ...]:
        return tuple(sorted(self._clients))

    def client(self, name: str) -> McpClient | None:
        return self._clients.get(name)

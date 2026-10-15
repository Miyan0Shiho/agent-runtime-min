"""MCP naming: a server's tool identity, and the name the model sees.

This module exists because of one invariant, and getting it wrong is subtle in a
way that only shows up later:

    A tool's identity is ``(server_name, raw_name)``. The model-facing name is
    derived from that pair, and the public name is **never parsed back** into the
    pair.

That direction matters. If the runtime recovered the server and tool from the
public name, then two different identities that happened to sanitize to the same
string would become indistinguishable -- and the model would call one tool while
the runtime invoked another. Deriving one way and never reversing it means the
mapping is total and injective by construction.

Two constraints shape the format:

* **The wire contract is not configuration.** The public name must be at most 64
  characters and use only ``[A-Za-z0-9_-]``. That is a property of the protocol,
  not a preference, so it is enforced rather than made tunable.
* **A collision must be impossible, not unlikely.** When sanitizing or truncating
  changes the name, a hash of the *original identity* is appended. Two identities
  that differ in any way produce different hashes, so distinct tools cannot
  collapse into one public name.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..kernel.errors import ArmError

#: The wire contract for a function name. Not configurable: it is what the
#: provider accepts, and a "longer names" setting would produce tools that fail
#: at call time rather than at registration.
MAX_PUBLIC_NAME = 64

#: The only characters a public name may contain.
_ALLOWED = re.compile(r"[^A-Za-z0-9_-]")

#: The separator between the prefix, the server and the raw name. Double
#: underscore because a single one is a legal character *within* a server or tool
#: name, so a single separator would be ambiguous to read.
SEPARATOR = "__"

#: How many hex characters of the identity hash to append on a collision risk.
_HASH_CHARS = 12


class McpError(ArmError):
    """An MCP operation was refused."""

    def __init__(self, message: str, *, code: str = "MCP_INVALID") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ToolIdentity:
    """The two names that identify a remote tool.

    Kept as a value rather than reconstructed from the public name, because the
    public name is lossy: it is sanitized and truncated, so it cannot be reversed.
    """

    server: str
    tool: str

    @property
    def key(self) -> str:
        """A stable string for hashing and for use as a dict key."""
        return f"{self.server}\x00{self.tool}"

    def public_name(self) -> str:
        """The name the model sees.

        The format is ``mcp__<server>__<tool>``. When either part needed
        sanitizing, or when the result would exceed the wire limit, a hash of the
        original identity is appended -- so the public name is still a function
        of the full identity even after the readable part was altered.
        """
        sanitized_server = _ALLOWED.sub("_", self.server)
        sanitized_tool = _ALLOWED.sub("_", self.tool)

        candidate = f"mcp{SEPARATOR}{sanitized_server}{SEPARATOR}{sanitized_tool}"
        altered = (
            sanitized_server != self.server or sanitized_tool != self.tool
        )

        if not altered and len(candidate) <= MAX_PUBLIC_NAME:
            return candidate

        # The readable part is no longer a faithful encoding, so the hash
        # carries the identity. It is computed over the *original* pair, which
        # is what makes the mapping injective.
        digest = hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:_HASH_CHARS]
        suffix = f"{SEPARATOR}{digest}"
        budget = MAX_PUBLIC_NAME - len(suffix)

        head = candidate[:budget]
        # Do not leave a trailing separator, which would read as an empty name
        # part and make the name harder to scan.
        head = head.rstrip("_")
        return f"{head}{suffix}"

    def describe(self) -> str:
        return f"{self.server}/{self.tool}"


def identity_from(server: str, tool: str) -> ToolIdentity:
    """Build an identity, validating the parts.

    An empty part is refused rather than defaulted: a tool with no name is a
    server bug, and inventing one would make it uncallable in a way nobody could
    diagnose.
    """
    if not server or not server.strip():
        raise McpError("an MCP tool needs a server name", code="MCP_NO_SERVER")
    if not tool or not tool.strip():
        raise McpError("an MCP tool needs a tool name", code="MCP_NO_TOOL")
    return ToolIdentity(server=server.strip(), tool=tool.strip())


class Namespace:
    """Tracks the public names one server has registered.

    A server's tools share a prefix. That is what makes it possible to register a
    whole generation atomically: either every tool of a server is visible, or
    none is. A half-registered server is worse than an absent one, because the
    model would call a tool that the runtime half-believes exists.
    """

    def __init__(self, server: str) -> None:
        self.server = server
        self._names: dict[str, ToolIdentity] = {}

    @property
    def prefix(self) -> str:
        return f"mcp{SEPARATOR}{_ALLOWED.sub('_', self.server)}{SEPARATOR}"

    def owns(self, public_name: str) -> bool:
        """Whether a public name belongs to this server's namespace.

        Used to detect a foreign registration squatting on the prefix. Without
        the check, two servers whose names sanitize to the same string would
        silently share a namespace and shadow each other's tools.
        """
        return public_name.startswith(self.prefix)

    def add(self, identity: ToolIdentity) -> str:
        """Register an identity, returning its public name."""
        if identity.server != self.server:
            raise McpError(
                f"{identity.describe()} does not belong to server {self.server!r}"
            )
        public = identity.public_name()
        existing = self._names.get(public)
        if existing is not None and existing != identity:
            # Two identities producing one public name would be a hash collision
            # or a sanitization collision. Either way, one tool would be
            # unreachable, so it is refused rather than resolved.
            raise McpError(
                f"public name {public!r} is claimed by both {existing.describe()} "
                f"and {identity.describe()}",
                code="MCP_NAME_COLLISION",
            )
        self._names[public] = identity
        return public

    def all(self) -> tuple[ToolIdentity, ...]:
        return tuple(self._names.values())

    def __len__(self) -> int:
        return len(self._names)

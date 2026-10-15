"""The stdio transport: an MCP server as a child process speaking JSON-RPC.

MCP over stdio is line-delimited JSON-RPC on the child's stdin and stdout. That
sounds simple, and three details are where it goes wrong:

**The environment is not inherited wholesale.** A server process that receives
the parent's full environment receives every credential the parent has. It gets a
clean environment plus exactly what its configuration names, because "it probably
needs PATH" is not a reason to hand over an API key.

**``stderr`` is captured, not discarded.** A server that fails to start usually
says why on stderr, and a discarded stream turns a diagnosable failure into "it
did not work". The output is kept for the error message.

**A read that never returns is a hang, not a failure.** Every request has a
deadline, and the process is killed when it expires. Without that, one wedged
server would hang the whole turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..kernel.errors import describe
from .naming import McpError

log = logging.getLogger("arm.mcp.stdio")

#: How much of a failed server's stderr to keep for the error message. Bounded
#: because a server that crashes in a loop produces unbounded output.
_MAX_STDERR_BYTES = 8_192


@dataclass
class StdioServerConfig:
    """How to launch one stdio server."""

    command: str
    args: Sequence[str] = ()
    #: Environment variables to pass through, by name. Everything else is
    #: withheld.
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str = ""
    #: How long to wait for the process to become ready.
    startup_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.command:
            raise McpError("an MCP stdio server needs a command")
        if self.startup_timeout <= 0:
            raise McpError("startup_timeout must be positive")


class StdioTransport:
    """A JSON-RPC connection to a child process."""

    def __init__(self, config: StdioServerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._stderr: list[bytes] = []

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Launch the child process."""
        env = self._clean_environment()
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.cwd or None,
            )
        except FileNotFoundError as exc:
            raise McpError(
                f"the MCP server command {self.config.command!r} was not found",
                code="MCP_COMMAND_NOT_FOUND",
            ) from exc
        except OSError as exc:
            raise McpError(
                f"could not start {self.config.command!r}: {exc}",
                code="MCP_SPAWN_FAILED",
            ) from exc

        # stderr is drained in the background so a chatty server cannot fill the
        # pipe buffer and deadlock on its own logging.
        asyncio.ensure_future(self._drain_stderr())

    def _clean_environment(self) -> dict[str, str]:
        """A minimal environment plus the configured variables.

        Not the parent's environment: a server process is not entitled to every
        credential the runtime holds, and passing them through would make the
        configuration's ``env`` field look like a convenience rather than the
        actual permission boundary.
        """
        env: dict[str, str] = {}
        for name in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"):
            value = os.environ.get(name)
            if value is not None:
                env[name] = value
        env.update({str(k): str(v) for k, v in self.config.env.items()})
        return env

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                self._stderr.append(line)
                # Keep the buffer bounded while preserving the *last* output,
                # which is where the reason for a failure usually is.
                total = sum(len(chunk) for chunk in self._stderr)
                while total > _MAX_STDERR_BYTES and len(self._stderr) > 1:
                    total -= len(self._stderr.pop(0))
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            return

    @property
    def stderr_text(self) -> str:
        return b"".join(self._stderr).decode("utf-8", "replace").strip()

    async def close(self) -> None:
        """Terminate the child, then reap it.

        Terminating without waiting would leave a zombie; waiting without
        terminating would hang on a server that ignores stdin closing.
        """
        process = self._process
        if process is None:
            return
        self._process = None
        if process.returncode is not None:
            return
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass

    # -- requests --------------------------------------------------------

    async def request(
        self, method: str, params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Send one request and await its response.

        Serialized by a lock, because two concurrent requests writing to one pipe
        would interleave and produce unparseable output. The alternative --
        multiplexing by id -- is a bigger mechanism than this transport needs.
        """
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise McpError("the MCP server is not running", code="MCP_NOT_RUNNING")

        async with self._lock:
            self._next_id += 1
            request_id = self._next_id
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method,
                 "params": dict(params)},
                ensure_ascii=False,
            )

            try:
                process.stdin.write((payload + "\n").encode("utf-8"))
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise McpError(
                    f"the MCP server closed its input: {describe(exc)}"
                    + (f"\n{self.stderr_text}" if self.stderr_text else ""),
                    code="MCP_BROKEN_PIPE",
                ) from exc

            # Read until the response with our id arrives. A server may emit
            # notifications in between, and treating the first line as the
            # response would break on any server that sends progress.
            while True:
                line = await process.stdout.readline()
                if not line:
                    detail = self.stderr_text
                    raise McpError(
                        f"the MCP server exited before responding to {method!r}"
                        + (f"\n{detail}" if detail else ""),
                        code="MCP_SERVER_EXITED",
                    )
                try:
                    message = json.loads(line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    # A non-JSON line is a server that logs to stdout, which is a
                    # protocol violation. Skipping it would hide the bug; failing
                    # would be brittle. It is logged and skipped.
                    log.debug("ignoring a non-JSON line from the MCP server")
                    continue

                if not isinstance(message, Mapping):
                    continue
                if message.get("id") != request_id:
                    # A notification, or a response to something else. Either
                    # way it is not ours.
                    continue

                error = message.get("error")
                if error is not None:
                    if isinstance(error, Mapping):
                        text = str(error.get("message", "the server reported an error"))
                        code = str(error.get("code", ""))
                    else:
                        text = str(error)
                        code = ""
                    raise McpError(
                        f"{method} failed: {text}" + (f" (code {code})" if code else ""),
                        code="MCP_SERVER_ERROR",
                    )

                result = message.get("result")
                if not isinstance(result, Mapping):
                    return {}
                return result


async def connect_stdio(config: StdioServerConfig) -> StdioTransport:
    """Build a ready-to-use stdio transport.

    Exposed as a factory so a server config's ``connect`` callable is a plain
    function and the client never imports the process machinery.
    """
    transport = StdioTransport(config)
    await transport.start()
    return transport

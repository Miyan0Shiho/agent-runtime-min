"""Built-in tools: the smallest set that makes the runtime actually useful.

Each tool here is deliberately thin. The interesting engineering is in the
pipeline, not in the tools; a tool that grows its own retry logic, its own
sandboxing or its own error conventions is a tool that bypasses the runtime's
guarantees.

Two conventions hold for all of them:

* **Failure returns a result.** A missing file, a non-zero exit code and a
  timeout are all ordinary outcomes. None of them raises.
* **Output is bounded.** A tool that can produce unbounded output hands the
  context manager a problem it cannot solve cheaply, so the limit belongs here,
  at the source.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field

from .definition import RegisteredTool, ToolDefinition, ToolResult

#: Default ceiling on text returned to the model from one call.
DEFAULT_MAX_CHARS = 32_000


class ReadParams(BaseModel):
    path: str = Field(description="Path to the file, absolute or relative to cwd.")
    offset: int = Field(default=1, description="1-based line to start from.")
    limit: int = Field(default=2000, description="Maximum number of lines to read.")


class WriteParams(BaseModel):
    path: str = Field(description="Path to write.")
    content: str = Field(description="Full file content to write.")


class EditParams(BaseModel):
    path: str = Field(description="Path to the file to edit.")
    old_string: str = Field(description="Exact text to replace. Must be unique.")
    new_string: str = Field(description="Replacement text.")


class BashParams(BaseModel):
    command: str = Field(description="Shell command to run.")
    timeout_ms: int = Field(default=60_000, description="Timeout in milliseconds.")


class GlobParams(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py'.")
    path: str = Field(default=".", description="Directory to search.")


class GrepParams(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str = Field(default=".", description="File or directory to search.")
    include: str = Field(default="", description="Optional filename glob filter.")


def _resolve(path: str) -> Path:
    return Path(path).expanduser()


def _clip(text: str, limit: int = DEFAULT_MAX_CHARS) -> str:
    """Bound a tool's output, saying clearly that it was clipped.

    Silent truncation is worse than truncation: the model would reason about a
    partial result as if it were complete.
    """
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n… [{omitted} characters omitted] …\n\n{tail}"


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


async def read_file(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
    """Read a file with line numbers, like a careful editor."""
    path = _resolve(str(arguments["path"]))
    offset = max(1, int(arguments.get("offset", 1)))
    limit = max(1, int(arguments.get("limit", 2000)))

    # M3.10: every failure carries a structured identity that persists with the
    # result while staying out of what the model reads. Without it a trace shows
    # `is_error: true` and nothing else -- a missing file, a directory and a
    # permission denial are indistinguishable without re-running the call, which
    # is exactly what a trace exists to avoid.
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ToolResult.fail(
            f"file not found: {path}",
            code="TOOL_NOT_FOUND",
            info={"path": str(path), "kind": "missing"},
        )
    except IsADirectoryError:
        return ToolResult.fail(
            f"{path} is a directory, not a file",
            code="TOOL_NOT_A_FILE",
            info={"path": str(path), "kind": "directory"},
        )
    except PermissionError:
        return ToolResult.fail(
            f"permission denied: {path}",
            code="TOOL_PERMISSION_DENIED",
            info={"path": str(path), "kind": "permission"},
        )
    except OSError as exc:
        return ToolResult.fail(
            f"could not read {path}: {exc}",
            code="TOOL_IO_ERROR",
            info={"path": str(path), "kind": "io", "errno": exc.errno},
        )

    lines = raw.splitlines()
    total = len(lines)
    window = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(
        f"{offset + i:>6}\t{line}" for i, line in enumerate(window)
    )
    header = f"{path} (lines {offset}-{offset + len(window) - 1} of {total})"
    return ToolResult.ok(_clip(f"{header}\n{numbered}"), value={"total_lines": total})


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


async def write_file(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
    """Write a whole file, creating parent directories as needed."""
    path = _resolve(str(arguments["path"]))
    content = str(arguments["content"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        return ToolResult.fail(f"permission denied: {path}")
    except OSError as exc:
        return ToolResult.fail(f"could not write {path}: {exc}")
    return ToolResult.ok(
        f"wrote {len(content)} characters to {path}", value={"bytes": len(content)}
    )


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


async def edit_file(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
    """Replace a unique substring in a file.

    Uniqueness is required rather than optional. A replace that matched several
    places would silently change code the caller never looked at, and the caller
    would have no way to tell which occurrence was meant.
    """
    path = _resolve(str(arguments["path"]))
    old = str(arguments["old_string"])
    new = str(arguments["new_string"])

    if old == new:
        return ToolResult.fail("old_string and new_string are identical")

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ToolResult.fail(f"file not found: {path}")
    except OSError as exc:
        return ToolResult.fail(f"could not read {path}: {exc}")

    occurrences = content.count(old)
    if occurrences == 0:
        return ToolResult.fail(f"old_string not found in {path}")
    if occurrences > 1:
        return ToolResult.fail(
            f"old_string appears {occurrences} times in {path}; it must be unique. "
            "Include more surrounding context to disambiguate."
        )

    try:
        path.write_text(content.replace(old, new, 1), encoding="utf-8")
    except OSError as exc:
        return ToolResult.fail(f"could not write {path}: {exc}")
    return ToolResult.ok(f"edited {path}")


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


async def run_bash(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
    """Run a shell command, returning its combined output.

    A non-zero exit code is reported as a *successful* tool call whose output
    includes the code. The tool did its job -- it ran the command; the command
    failing is information the model needs, not a tool malfunction.
    """
    command = str(arguments["command"])
    timeout_ms = int(arguments.get("timeout_ms", 60_000))
    timeout = max(0.1, timeout_ms / 1000.0)

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd(),
        )
    except OSError as exc:
        return ToolResult.fail(f"could not start command: {exc}")

    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # Kill the process group, then report the timeout as the outcome. Leaving
        # it running would leak a process per timed-out call.
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass
        return ToolResult.fail(
            f"command timed out after {timeout_ms}ms", code="TOOL_TIMEOUT"
        )

    output = stdout.decode("utf-8", errors="replace")
    code = process.returncode or 0
    body = output if output.strip() else "(no output)"
    return ToolResult.ok(
        _clip(f"exit code: {code}\n{body}"),
        value={"exit_code": code},
    )


# ---------------------------------------------------------------------------
# glob / grep
# ---------------------------------------------------------------------------


async def glob_files(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
    """Find files by glob pattern."""
    root = _resolve(str(arguments.get("path", ".")))
    pattern = str(arguments["pattern"])
    try:
        matches = sorted(str(p) for p in root.glob(pattern) if p.is_file())
    except (OSError, ValueError) as exc:
        return ToolResult.fail(f"glob failed: {exc}")
    if not matches:
        return ToolResult.ok(f"no files matched {pattern} under {root}")
    return ToolResult.ok(_clip("\n".join(matches)), value={"count": len(matches)})


async def grep_files(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
    """Search file contents with a regular expression."""
    import re

    root = _resolve(str(arguments.get("path", ".")))
    pattern = str(arguments["pattern"])
    include = str(arguments.get("include", "")) or "*"

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult.fail(f"invalid regular expression: {exc}")

    targets = (
        [root]
        if root.is_file()
        else sorted(p for p in root.rglob(include) if p.is_file())
    )

    hits: list[str] = []
    for target in targets:
        if len(hits) > 400:
            hits.append("… more matches omitted …")
            break
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{target}:{lineno}:{line[:300]}")
                if len(hits) > 400:
                    break

    if not hits:
        return ToolResult.ok(f"no matches for {pattern!r}")
    return ToolResult.ok(_clip("\n".join(hits)), value={"count": len(hits)})


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def builtin_tools() -> list[RegisteredTool]:
    """The standard tool set, ready to register."""
    def make(
        name: str,
        description: str,
        params: type[BaseModel],
        body,
        *,
        parallel: bool = False,
        needs_approval: bool = False,
    ) -> RegisteredTool:
        return RegisteredTool(
            definition=ToolDefinition(
                name=name,
                description=description,
                parameters=params.model_json_schema(),
                concurrency_safe=(lambda _args: True) if parallel else None,
                # Declared by the tool, not by a central list: the tool knows
                # what it does, and a list would drift silently -- a new
                # dangerous tool would simply run.
                requires_approval=needs_approval,
            ),
            body=body,
            validator=params,
        )

    return [
        make(
            "read",
            "Read a file from the filesystem, returning numbered lines.",
            ReadParams,
            read_file,
            parallel=True,
        ),
        make(
            "write",
            "Write content to a file, creating it if needed.",
            WriteParams,
            write_file,
            needs_approval=True,
        ),
        make(
            "edit",
            "Replace a unique string in a file.",
            EditParams,
            edit_file,
                    needs_approval=True,
        ),
        make(
            "bash",
            "Run a shell command and return its output.",
            BashParams,
            run_bash,
            needs_approval=True,
        ),
        make(
            "glob",
            "List files matching a glob pattern.",
            GlobParams,
            glob_files,
            parallel=True,
        ),
        make(
            "grep",
            "Search file contents with a regular expression.",
            GrepParams,
            grep_files,
            parallel=True,
        ),
    ]

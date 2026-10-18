"""Background jobs as tools: start one, watch it, read it, stop it.

**Why this module exists.** `arm/jobs/` was 925 lines with thorough tests and no
caller: nothing in `src/` imported it, and the CLI registered no tool for it. It
satisfied a REQUIREMENTS item ("任务编排: jobs") on paper while being unreachable
from the product. Defect 89.

**What a job is here.** A command that outlives the step that started it. The
step returns immediately with an id; later steps poll it. That shape matters
because a long command run inline blocks the loop, and a model given a long build
will otherwise spend its whole budget waiting inside one call.

Four tools, split the way `computer/tools.py` splits observation from action:

* `job_start`  -- creates work. Approval required: it runs a command.
* `job_list`   -- what jobs exist, and their state.
* `job_read`   -- the output produced so far, from an offset.
* `job_stop`   -- cancel one.

Only `job_start` is gated. Reading a job's output is looking at something that has
already happened, and gating it would train the user to approve reflexively --
the same reasoning the computer-use tools state.

**Jobs die with the session.** The registry lives in the runtime, and the CLI runs
one process per invocation. A job started in `arm run` therefore cannot be polled
by the *next* `arm run`; it is for work that spans steps within one session. That
limitation is stated rather than hidden, because a background job that silently
vanishes between processes is worse than one that was never offered.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .registry import JobError, JobRegistry

DEFAULT_LABEL = ""


def _describe(status: Any) -> str:
    """One line for a job, for the list tool and for start's reply."""
    return (
        f"{status.id} [{status.state}] {status.label or '(unlabelled)'} "
        f"({status.elapsed:.1f}s, {status.output_bytes} bytes)"
    )


def _shell_producer(command: str, timeout: float):
    """A producer that runs a shell command and streams its output.

    Streamed rather than captured whole: the point of a background job is that the
    caller can read progress *while* it runs. A producer that only reported at the
    end would be a slow synchronous call with extra steps.
    """

    async def produce(handle: Any) -> str | None:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert process.stdout is not None

        try:
            while True:
                if handle.cancelled:
                    process.kill()
                    await process.wait()
                    return "cancelled"
                chunk = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
                if not chunk:
                    break
                handle.emit(chunk.decode("utf-8", "replace"))
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"no output for {timeout}s; stopped"

        code = await process.wait()
        return None if code == 0 else f"exit code {code}"

    return produce


class JobTools:
    """The four background-job tools, over one registry."""

    def __init__(self, registry: JobRegistry, *, owner: str = "", timeout: float = 600.0) -> None:
        self.registry = registry
        #: The session id, so a job's owner is the session that started it and
        #: another session cannot read it. Authorization is by owner, not id.
        self.owner = owner
        self.timeout = timeout

    def tools(self) -> list[RegisteredTool]:
        return [self._start(), self._list(), self._read(), self._stop()]

    # -- start -----------------------------------------------------------

    def _start(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            command = str(arguments.get("command", "")).strip()
            if not command:
                return ToolResult.fail("job_start needs a command")

            try:
                job_id = self.registry.start(
                    kind="shell",
                    producer=_shell_producer(command, self.timeout),
                    label=str(arguments.get("label", "")),
                    owner=self.owner,
                )
            except JobError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            status = self.registry.get(job_id, caller=self.owner)
            return ToolResult.ok(
                f"Started {_describe(status) if status else job_id}. "
                "Use job_read to see its output and job_stop to end it.",
                value={"job_id": job_id},
            )

        class StartParams(BaseModel):
            command: str = Field(description="The shell command to run in the background.")
            label: str = Field(default="", description="A short name for the job.")

        return RegisteredTool(
            definition=ToolDefinition(
                name="job_start",
                description=(
                    "Start a shell command in the background and return an id "
                    "immediately. Use it for long commands so the work does not "
                    "block this step. The job ends when the session does."
                ),
                parameters=StartParams.model_json_schema(),
                requires_approval=True,
            ),
            body=body,
            validator=StartParams,
        )

    # -- list ------------------------------------------------------------

    def _list(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            jobs = self.registry.list(caller=self.owner)
            if not jobs:
                return ToolResult.ok("No background jobs in this session.")
            return ToolResult.ok(
                "\n".join(_describe(status) for status in jobs),
                value={"count": len(jobs)},
            )

        class ListParams(BaseModel):
            pass

        return RegisteredTool(
            definition=ToolDefinition(
                name="job_list",
                description="List the background jobs in this session and their state.",
                parameters=ListParams.model_json_schema(),
            ),
            body=body,
            validator=ListParams,
        )

    # -- read ------------------------------------------------------------

    def _read(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            job_id = str(arguments.get("job_id", ""))
            offset = int(arguments.get("offset", 0))

            result = self.registry.read_at(job_id, offset, caller=self.owner)
            if result is None:
                return ToolResult.fail(
                    f"no job {job_id!r} in this session", code="JOB_NOT_FOUND"
                )

            parts: list[str] = []
            if result.has_gap:
                # Saying so is the point: the reader can tell the difference
                # between "nothing was written" and "output was dropped".
                parts.append(
                    f"[{result.gap_bytes} bytes were dropped before this read]"
                )
            parts.append(result.text or "(no output yet)")

            status = self.registry.get(job_id, caller=self.owner)
            if status is not None:
                parts.append(
                    f"[{status.state}"
                    + (f", {status.result}" if status.result else "")
                    + f", next offset {result.next_offset}]"
                )
            return ToolResult.ok(
                "\n".join(parts),
                value={"next_offset": result.next_offset, "at_end": result.at_end},
            )

        class ReadParams(BaseModel):
            job_id: str = Field(description="The id job_start returned.")
            offset: int = Field(
                default=0,
                description="Byte offset to read from; use next_offset from a previous read.",
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="job_read",
                description=(
                    "Read a background job's output from an offset. Reports how "
                    "much was dropped, if anything, so 'no output yet' and 'output "
                    "was lost' are distinguishable."
                ),
                parameters=ReadParams.model_json_schema(),
            ),
            body=body,
            validator=ReadParams,
        )

    # -- stop ------------------------------------------------------------

    def _stop(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            job_id = str(arguments.get("job_id", ""))
            try:
                outcome = await self.registry.kill(job_id, caller=self.owner)
            except JobError as exc:
                return ToolResult.fail(str(exc), code=exc.code)
            if outcome == "already-finished":
                return ToolResult.ok(
                    f"{job_id} had already finished.", value={"job_id": job_id}
                )
            # "requested", not "stopped": the cancellation has been asked for and
            # not yet confirmed, and claiming otherwise would report work as
            # finished while it is still running.
            return ToolResult.ok(
                f"Stopping {job_id}. It is not stopped until job_list says so.",
                value={"job_id": job_id},
            )

        class StopParams(BaseModel):
            job_id: str = Field(description="The id job_start returned.")

        return RegisteredTool(
            definition=ToolDefinition(
                name="job_stop",
                description="Stop a running background job.",
                parameters=StopParams.model_json_schema(),
            ),
            body=body,
            validator=StopParams,
        )

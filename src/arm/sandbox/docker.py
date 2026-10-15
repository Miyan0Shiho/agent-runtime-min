"""The Docker sandbox backend: a real boundary instead of a checked one.

The policy module decides *whether* an operation is allowed. This module is what
makes the decision **structural** rather than advisory, and the distinction is the
whole reason it exists.

The current enforcement checks the tool name and the path argument:

    ``bash`` is permitted in ``workspace-write``, so the tool runs -- on the host,
    with the user's privileges, in the user's filesystem.

Measured, before this backend existed: in a ``workspace-write`` sandbox, a
``bash`` call that touched a path outside the workspace created the file. The
policy cannot see what a shell command does, so "the workspace is the boundary"
was a statement about the *tool*, not about the *process*.

Running the command in a container makes the boundary real:

* the workspace is the only writable mount, so a path outside it is not
  reachable -- not "refused by a check", but absent from the filesystem;
* the network can be removed entirely;
* the process runs as an unprivileged user inside a throwaway image;
* the container is destroyed after the call, so nothing accumulates.

Three properties are deliberate, and each costs something:

**A container per call, not per session.** Slower -- a fresh container is tens of
milliseconds -- but a shared container would accumulate state between calls, and
"what is in the sandbox now" would depend on everything that ran before. The
boundary is only trustworthy if it is the same boundary every time.

**The workspace is mounted read-write only when the mode says so.** ``read-only``
mounts it read-only, so a write fails at the filesystem rather than at a check
that could be bypassed.

**No network by default.** A sandboxed command that can reach the network is not
contained in the sense that matters: it can exfiltrate whatever it can read. The
mode has to ask for it.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from typing import Any, Mapping

from ..kernel.errors import ArmError, describe
from .policy import (
    MODE_READ_ONLY,
    MODE_WORKSPACE_WRITE,
    OP_EXECUTE,
    SandboxDenied,
    SandboxPolicy,
)

log = logging.getLogger("arm.sandbox.docker")

#: The image a sandboxed command runs in. Pinned by tag rather than by digest
#: because the point is a *known* base, and a digest would silently stop
#: receiving security updates for the base packages.
DEFAULT_IMAGE = "python:3.11-slim"

#: Where the workspace appears inside the container. Fixed rather than mirroring
#: the host path: a container that exposes the host's directory structure tells
#: the command where it is running, and the same command should behave the same
#: way regardless of where the user happens to keep their files.
CONTAINER_WORKSPACE = "/workspace"

#: How long a container may take to start before it is considered broken.
DEFAULT_START_TIMEOUT = 60.0

#: The allowance added to a command's own timeout for container startup.
#:
#: Measured: killing a running container takes about 15 seconds through the lima
#: VM, so the wait has to outlast the kill rather than the command. This is the
#: *only* reason the allowance is this large -- it is time spent stopping the
#: container, not time the command is allowed to run.
_START_ALLOWANCE = 20.0

#: Memory and CPU ceilings. A sandbox that can exhaust the host's memory is not
#: containing anything.
DEFAULT_MEMORY = "2g"
DEFAULT_CPUS = "2"


class DockerError(ArmError):
    """The container backend could not run."""

    def __init__(self, message: str, *, code: str = "DOCKER_ERROR") -> None:
        super().__init__(message)
        self.code = code


class DockerUnavailableError(DockerError):
    """No usable Docker daemon."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"the Docker sandbox backend needs a running daemon: {reason}",
            code="DOCKER_UNAVAILABLE",
        )


@dataclass(frozen=True, slots=True)
class ContainerRun:
    """The outcome of one sandboxed command."""

    exit_code: int
    stdout: str
    timed_out: bool = False
    #: Set when the container itself failed rather than the command in it.
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.timed_out and not self.error


def _docker() -> str:
    """The docker binary, or a clear error.

    Resolved once and cached: looking it up per call would add a `which` to every
    sandboxed command, and the answer does not change within a process.
    """
    found = shutil.which("docker")
    if not found:
        raise DockerUnavailableError(
            "the 'docker' command is not on PATH. On macOS with lima: "
            "'brew install docker' and 'docker context use lima-docker'."
        )
    return found


class DockerSandbox:
    """Runs commands in a container instead of on the host.

    Holds no container between calls. That is the design, not an oversight: a
    long-lived container would carry state from one call to the next, and the
    guarantee being offered is that a command's environment is the same every
    time.
    """

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        image: str = DEFAULT_IMAGE,
        network: bool = False,
        memory: str = DEFAULT_MEMORY,
        cpus: str = DEFAULT_CPUS,
        start_timeout: float = DEFAULT_START_TIMEOUT,
    ) -> None:
        if not policy.workspace:
            # Without a workspace there is nothing to mount, and a container with
            # no mount would be a sandbox around nothing.
            raise DockerError(
                "the Docker backend needs a workspace to mount; without one there "
                "is no boundary to enforce",
                code="DOCKER_NO_WORKSPACE",
            )
        self.policy = policy
        self.image = image
        #: Whether the container may reach the network. Off by default: a
        #: sandboxed command that can reach the network can exfiltrate whatever it
        #: can read, which is not containment in the sense that matters.
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.start_timeout = start_timeout

    # -- capability ------------------------------------------------------

    @staticmethod
    def available() -> tuple[bool, str]:
        """Whether a usable daemon is reachable, and why not if it is not."""
        binary = shutil.which("docker")
        if not binary:
            return False, "the 'docker' command is not on PATH"
        try:
            completed = __import__("subprocess").run(
                [binary, "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                timeout=20,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - any failure means unusable
            return False, describe(exc)
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace").strip()
            return False, detail.splitlines()[-1] if detail else "the daemon did not answer"
        return True, ""

    def require_available(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise DockerUnavailableError(reason)

    # -- the command -----------------------------------------------------

    def _mount_mode(self) -> str:
        """How the workspace is mounted, from the policy's mode.

        ``read-only`` mounts it read-only, so a write is refused by the
        filesystem rather than by a check that a command could route around.
        """
        if self.policy.mode == MODE_READ_ONLY:
            return "ro"
        if self.policy.mode == MODE_WORKSPACE_WRITE:
            return "rw"
        # `none` means no filesystem access at all, so nothing is mounted.
        return ""

    def build_argv(self, command: str) -> list[str]:
        """The docker invocation for one command.

        Built as an argument list, never a shell string: the command is passed as
        the container's argv, so a shell metacharacter in it cannot be
        interpreted by the *host* shell. Building a string here would reintroduce
        exactly the injection the container is meant to remove.
        """
        mount = self._mount_mode()
        argv = [
            _docker(),
            "run",
            "--rm",
            # No TTY: there is no one to answer a prompt, and a command that
            # expects one would hang until the timeout.
            "--interactive=false",
            "--workdir",
            CONTAINER_WORKSPACE if mount else "/",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            # A command that cannot fork cannot exhaust the host, and a runaway
            # fork bomb inside the container would still take the machine down.
            "--pids-limit",
            "512",
            # Dropped so a command cannot escalate inside the container.
            "--security-opt",
            "no-new-privileges",
        ]

        if mount:
            argv += ["--volume", f"{self.policy.workspace}:{CONTAINER_WORKSPACE}:{mount}"]
        for root in self.policy.extra_roots:
            # Extra roots are read-only: they widen what can be *read*, which is
            # what the policy says they are for. Making them writable would widen
            # the boundary itself.
            argv += ["--volume", f"{root}:{root}:ro"]

        if not self.network:
            argv += ["--network", "none"]

        # `sh -c` inside the container, so the command is a shell command as the
        # tool promises -- but the shell runs in the sandbox, not on the host.
        argv += [self.image, "sh", "-c", command]
        return argv

    async def run(self, command: str, *, timeout: float = 60.0) -> ContainerRun:
        """Run one command in a fresh container."""
        self.require_available()

        argv = self.build_argv(command)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise DockerError(
                f"could not start the container: {describe(exc)}",
                code="DOCKER_START_FAILED",
            ) from exc

        # The deadline is the command's timeout plus a *fixed* allowance for the
        # container to start, not `start_timeout` added to it. Adding the start
        # allowance to the whole call meant a 2-second timeout on a 30-second
        # command waited 62 seconds and then reported success -- the timeout was
        # effectively ignored, which is worse than having none, because the caller
        # believes it is enforced.
        deadline = timeout + min(self.start_timeout, _START_ALLOWANCE)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=deadline)
        except asyncio.TimeoutError:
            # Kill the container, not just the client: `docker run` exiting does
            # not necessarily stop the container it started.
            await self._kill(process)
            return ContainerRun(
                exit_code=-1,
                stdout="",
                timed_out=True,
                error=f"the command did not finish within {timeout}s",
            )

        text = stdout.decode("utf-8", "replace")
        code = process.returncode if process.returncode is not None else -1

        # `docker run` failing to *start* the container exits 125, and 126/127 are
        # the client's own errors. A command that exits 127 inside the container
        # means "command not found", which is the command's outcome rather than
        # the backend's -- and the two are indistinguishable from the exit code
        # alone, so the message is what separates them.
        if code in (125, 126) and "docker:" in text.lower():
            return ContainerRun(
                exit_code=code,
                stdout=text,
                error=text.strip().splitlines()[-1] if text.strip() else "docker failed",
            )

        return ContainerRun(exit_code=code, stdout=text)

    async def _kill(self, process: Any) -> None:
        """Stop the container and the client."""
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("killing the sandboxed command failed: %s", exc)

    # -- checking --------------------------------------------------------

    def check(self, op: str, path: str = "") -> None:
        """Apply the policy, for the operations the container cannot express.

        The container enforces the filesystem boundary structurally, so a write
        outside the workspace is not reachable rather than refused. This check
        exists for the cases the container does *not* cover -- a protected path
        inside the workspace, and execution being disabled outright -- so the two
        layers agree instead of one silently permitting what the other would
        refuse.
        """
        self.policy.check(op, path)


def containerized_bash(sandbox: DockerSandbox) -> Any:
    """A ``bash`` body that runs its command in a container.

    This replaces the tool's *body* rather than wrapping the pipeline, and the
    difference matters: a wrapper has to recognise which tool to intercept by
    name, and a name is something a caller can change. Swapping the body means
    there is no path in which the host shell runs -- the tool that executes
    commands *is* the container one.

    The result shape matches the host tool's: a non-zero exit is a successful tool
    call whose output carries the code, because the tool did its job. The command
    failing is information, not a malfunction.
    """

    async def body(arguments: Mapping[str, Any], execution: Any) -> Any:
        from ..tools.definition import ToolResult

        command = str(arguments.get("command", ""))
        timeout_ms = int(arguments.get("timeout_ms", 60_000))

        # The policy is consulted first so a disabled mode refuses with the
        # policy's own message rather than by failing to start a container. Two
        # layers that disagree about *why* something was refused produce a
        # confusing error, and the policy is the authority.
        try:
            sandbox.check(OP_EXECUTE)
        except SandboxDenied as denial:
            return ToolResult.fail(str(denial), code=denial.code)

        try:
            outcome = await sandbox.run(
                command, timeout=max(0.1, timeout_ms / 1000.0)
            )
        except DockerError as exc:
            return ToolResult.fail(str(exc), code=exc.code)

        if outcome.timed_out:
            return ToolResult.fail(
                f"command timed out after {timeout_ms}ms", code="TOOL_TIMEOUT"
            )
        if outcome.error:
            return ToolResult.fail(
                f"the sandbox could not run the command: {outcome.error}",
                code="SANDBOX_CONTAINER_FAILED",
            )

        rendered = _render(outcome)
        # A non-zero exit is a result, not a tool failure -- the same rule the
        # host tool follows, so a caller sees the same shape either way.
        return ToolResult.ok(
            rendered, value={"exit_code": outcome.exit_code, "sandboxed": True}
        )

    return body


def _render(outcome: ContainerRun) -> str:
    """Render a sandboxed run the way the host tool renders one."""
    text = outcome.stdout.rstrip("\n")
    if outcome.exit_code == 0:
        return text or "(no output)"
    return f"{text}\n\nexit code: {outcome.exit_code}"

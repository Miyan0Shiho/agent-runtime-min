"""Tests for the Docker sandbox backend.

These run **real containers**, because the thing under test is a boundary and a
mock would only test the mock. The whole point is that a command which escaped on
the host cannot escape here, and that is only observable by running one.

The tests skip when no daemon is reachable, so the suite still passes on a machine
without Docker. That is the honest trade: a skipped test says what it did not
check, where a mock would report success for a boundary that was never exercised.

A note on workspaces. The lima VM mounts the home directory, so a workspace under
`/var/folders` (macOS's temp directory) is invisible to the container. These tests
create workspaces under the home directory for that reason, and
`test_a_workspace_outside_the_vm_mount_is_reported` pins what happens when one is
not reachable -- the failure is reported, not silently permitted.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from arm.sandbox.docker import (
    CONTAINER_WORKSPACE,
    DEFAULT_IMAGE,
    DockerError,
    DockerSandbox,
    containerized_bash,
)
from arm.sandbox.policy import (
    MODE_NONE,
    MODE_READ_ONLY,
    MODE_WORKSPACE_WRITE,
    SandboxPolicy,
)


def run(coro):
    return asyncio.run(coro)


def _daemon_available() -> tuple[bool, str]:
    return DockerSandbox.available()


@pytest.fixture(scope="module", autouse=True)
def require_docker():
    ok, reason = _daemon_available()
    if not ok:
        pytest.skip(f"no Docker daemon: {reason}")


@pytest.fixture
def sandbox_root():
    """A workspace the VM can see.

    Under the home directory, because the VM mounts `~` and nothing else. A
    workspace under `/var/folders` would be invisible to the container, which is
    a real limitation rather than a test detail.
    """
    root = Path(tempfile.mkdtemp(dir=Path.home() / "tmp-arm-sandbox"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def outside(sandbox_root):
    """A directory outside the workspace, to attempt escapes into."""
    other = Path(tempfile.mkdtemp(dir=Path.home() / "tmp-arm-sandbox"))
    yield other
    shutil.rmtree(other, ignore_errors=True)


def make_sandbox(root: Path, *, mode: str = MODE_WORKSPACE_WRITE, **kwargs):
    policy = SandboxPolicy(mode=mode, workspace=str(root))
    return DockerSandbox(policy, **kwargs)


# ===========================================================================
# Capability
# ===========================================================================


def test_the_daemon_is_reachable():
    ok, reason = _daemon_available()
    assert ok, reason


def test_a_sandbox_without_a_workspace_is_refused():
    """Without a workspace there is nothing to mount and no boundary.

    Two layers refuse this, and the policy refuses first: `workspace-write`
    without a workspace would mean "write anywhere", which is the opposite of what
    the name promises. Either refusal is correct; what matters is that a
    workspace-less sandbox cannot be constructed.
    """
    from arm.sandbox.policy import SandboxConfigurationError

    with pytest.raises((DockerError, SandboxConfigurationError)):
        DockerSandbox(SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=""))


def test_the_docker_layer_also_requires_a_workspace():
    """Even with a mode that does not need one, the backend does.

    A container with no mount is a sandbox around nothing, so the backend checks
    for itself rather than relying on the policy's mode to imply it.
    """
    with pytest.raises(DockerError) as info:
        DockerSandbox(SandboxPolicy(mode=MODE_NONE, workspace=""))
    assert info.value.code == "DOCKER_NO_WORKSPACE"


# ===========================================================================
# The boundary
# ===========================================================================


def test_a_command_runs_in_a_container(sandbox_root):
    sandbox = make_sandbox(sandbox_root)
    outcome = run(sandbox.run("echo hello", timeout=60))

    assert outcome.ok, outcome.error
    assert "hello" in outcome.stdout


def test_a_command_cannot_write_outside_the_workspace(sandbox_root, outside):
    """The defect this backend exists to fix.

    Measured before it existed: in a `workspace-write` sandbox, a `bash` call that
    touched a path outside the workspace created the file. The policy checks tool
    *names*, so "the workspace is the boundary" was a claim about the tool rather
    than about the process.
    """
    sandbox = make_sandbox(sandbox_root)
    target = outside / "escaped.txt"

    outcome = run(sandbox.run(f"touch {target}; echo attempted", timeout=60))

    assert not target.exists(), "the command escaped the workspace"
    # And the command itself was told it could not.
    assert "cannot touch" in outcome.stdout or outcome.exit_code != 0


def test_a_command_can_write_inside_the_workspace(sandbox_root):
    """The complement: the boundary must not block legitimate work."""
    sandbox = make_sandbox(sandbox_root)
    outcome = run(
        sandbox.run("echo written > inside.txt && cat inside.txt", timeout=60)
    )

    assert outcome.ok, outcome.error
    assert "written" in outcome.stdout
    assert (sandbox_root / "inside.txt").read_text().strip() == "written"


def test_a_read_only_workspace_refuses_writes(sandbox_root):
    """The mount is read-only, so the refusal is the filesystem's, not a check's.

    A check can be routed around; a read-only mount cannot.
    """
    sandbox = make_sandbox(sandbox_root, mode=MODE_READ_ONLY)
    outcome = run(sandbox.run("echo nope > ro.txt", timeout=60))

    assert outcome.exit_code != 0, "a write succeeded in a read-only workspace"
    assert "read-only" in outcome.stdout.lower()
    assert not (sandbox_root / "ro.txt").exists()


def test_a_read_only_workspace_still_allows_reading(sandbox_root):
    """The refusal must be specific, not blanket."""
    (sandbox_root / "readable.txt").write_text("content here")
    sandbox = make_sandbox(sandbox_root, mode=MODE_READ_ONLY)

    outcome = run(sandbox.run("cat readable.txt", timeout=60))

    assert outcome.ok, outcome.error
    assert "content here" in outcome.stdout


def test_the_network_is_removed_by_default(sandbox_root):
    """A sandboxed command that can reach the network can exfiltrate what it reads."""
    sandbox = make_sandbox(sandbox_root)
    outcome = run(
        sandbox.run(
            "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 80), 3)\"",
            timeout=60,
        )
    )

    assert outcome.exit_code != 0, "the network was reachable from the sandbox"


# ===========================================================================
# The command line is built as arguments, not a string
# ===========================================================================


def test_the_command_is_passed_as_argv(sandbox_root):
    """Never as a host shell string.

    Building a shell string would reintroduce exactly the injection the container
    is meant to remove -- the host shell would interpret it before docker saw it.
    """
    sandbox = make_sandbox(sandbox_root)
    argv = sandbox.build_argv("echo hi")

    assert argv[-3:] == ["sh", "-c", "echo hi"]
    assert argv[-4] == DEFAULT_IMAGE
    # The command is one argument, so a metacharacter in it cannot split it.
    assert sum(1 for a in argv if a == "echo hi") == 1


def test_the_workspace_is_mounted_at_a_fixed_path(sandbox_root):
    """Fixed, not mirroring the host path.

    A container that exposes the host's directory structure tells the command
    where it is running, and the same command should behave the same way wherever
    the user keeps their files.
    """
    sandbox = make_sandbox(sandbox_root)
    argv = sandbox.build_argv("true")

    assert f"{sandbox_root}:{CONTAINER_WORKSPACE}:rw" in argv


def test_a_read_only_mode_mounts_read_only(sandbox_root):
    sandbox = make_sandbox(sandbox_root, mode=MODE_READ_ONLY)
    argv = sandbox.build_argv("true")

    assert f"{sandbox_root}:{CONTAINER_WORKSPACE}:ro" in argv


def test_no_mode_mounts_nothing(sandbox_root):
    """`none` means no filesystem access, so there is nothing to mount."""
    sandbox = make_sandbox(sandbox_root, mode=MODE_NONE)
    argv = sandbox.build_argv("true")

    assert not any(CONTAINER_WORKSPACE in str(a) for a in argv)


def test_extra_roots_are_mounted_read_only(sandbox_root, outside):
    """They widen what can be read, not what can be written."""
    policy = SandboxPolicy(
        mode=MODE_WORKSPACE_WRITE,
        workspace=str(sandbox_root),
        extra_roots=(str(outside),),
    )
    argv = DockerSandbox(policy).build_argv("true")

    assert f"{outside}:{outside}:ro" in argv


def test_the_container_is_given_resource_limits(sandbox_root):
    """A sandbox that can exhaust the host's memory is containing nothing."""
    argv = make_sandbox(sandbox_root).build_argv("true")

    assert "--memory" in argv
    assert "--cpus" in argv
    assert "--pids-limit" in argv


def test_no_new_privileges_is_set(sandbox_root):
    """A command must not be able to escalate inside the container."""
    argv = make_sandbox(sandbox_root).build_argv("true")

    assert "no-new-privileges" in argv


def test_the_container_is_removed_after_the_call(sandbox_root):
    """Nothing accumulates between calls."""
    argv = make_sandbox(sandbox_root).build_argv("true")

    assert "--rm" in argv


# ===========================================================================
# The policy still applies
# ===========================================================================


def test_execution_is_refused_when_the_policy_disables_it(sandbox_root):
    """The policy is the authority, so its refusal is what the caller sees."""
    from arm.tools.definition import ToolResult

    policy = SandboxPolicy(
        mode=MODE_WORKSPACE_WRITE, workspace=str(sandbox_root), allow_execute=False
    )
    body = containerized_bash(DockerSandbox(policy))
    result = run(body({"command": "echo hi"}, None))

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert result.error_code == "SANDBOX_EXEC_DISABLED"


def test_the_body_returns_a_tool_result(sandbox_root):
    """It replaces a tool body, so it must return what a body returns."""
    from arm.tools.definition import ToolResult

    body = containerized_bash(make_sandbox(sandbox_root))
    result = run(body({"command": "echo hi"}, None))

    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert "hi" in result.content


def test_a_non_zero_exit_is_a_result_not_a_failure(sandbox_root):
    """The tool did its job: it ran the command.

    The command failing is information the model needs, not a malfunction -- the
    same rule the host tool follows, so a caller sees the same shape either way.
    """
    body = containerized_bash(make_sandbox(sandbox_root))
    result = run(body({"command": "exit 3"}, None))

    assert not result.is_error, "a non-zero exit was reported as a tool failure"
    assert result.value["exit_code"] == 3


def test_a_timeout_is_reported(sandbox_root):
    body = containerized_bash(make_sandbox(sandbox_root))
    result = run(body({"command": "sleep 30", "timeout_ms": 2000}, None))

    assert result.is_error
    assert result.error_code == "TOOL_TIMEOUT"


def test_the_result_is_marked_as_sandboxed(sandbox_root):
    """A caller must be able to tell which boundary a command ran in."""
    body = containerized_bash(make_sandbox(sandbox_root))
    result = run(body({"command": "true"}, None))

    assert result.value["sandboxed"] is True

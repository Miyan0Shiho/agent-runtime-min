"""Sandbox policy: which operations may touch what, and how strictly.

The policy layer answers one question -- *is this operation permitted under this
mode?* -- and deliberately does not answer *how* the operation is carried out.
That split is what lets a second backend (a container, a remote runner) be added
later without changing a single policy decision.

Three ideas shape it:

**Deny is the default, and permission must be explicit.** A policy that lists
what is forbidden has to anticipate every dangerous operation; a policy that
lists what is allowed does not. So the modes are expressed as grants.

**A denial keeps its identity.** When an operation is refused, the refusal
carries a machine-readable code, not just prose. A caller that needs to route on
"this was a sandbox denial" -- to offer an escalation, or to retry differently --
must not have to parse a message.

**Escalation is a separate, narrower request.** A denial is not an invitation to
retry with fewer restrictions. Widening requires an approval, and the approval is
for one operation. Keeping that explicit is what stops "the agent hit a wall" from
silently becoming "the agent removed the wall".

The modes here are deliberately few. A large matrix of modes invites
combinations nobody tests; three that are each obviously different are easier to
reason about.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..kernel.errors import ArmError

log = logging.getLogger("arm.sandbox")

#: Read and write freely within the workspace.
MODE_WORKSPACE_WRITE = "workspace-write"
#: Read anything; write nothing.
MODE_READ_ONLY = "read-only"
#: No filesystem access at all, and no command execution.
MODE_NONE = "none"

MODES = frozenset({MODE_WORKSPACE_WRITE, MODE_READ_ONLY, MODE_NONE})

#: The operation kinds the policy decides about.
OP_READ = "read"
OP_WRITE = "write"
OP_EXECUTE = "execute"


class SandboxError(ArmError):
    """Base class for sandbox refusals."""


class SandboxDenied(SandboxError):
    """An operation was refused by policy.

    Carries a code so a caller can route on the *kind* of refusal without
    parsing the message.
    """

    def __init__(self, message: str, *, code: str, path: str = "", op: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.op = op


class SandboxConfigurationError(SandboxError):
    """The policy is malformed. Refused at construction, not at use."""


#: Codes for the refusals a caller may need to distinguish.
CODE_OUTSIDE_WORKSPACE = "SANDBOX_OUTSIDE_WORKSPACE"
CODE_READ_ONLY = "SANDBOX_READ_ONLY"
CODE_NO_ACCESS = "SANDBOX_NO_ACCESS"
CODE_EXEC_DISABLED = "SANDBOX_EXEC_DISABLED"
CODE_SENSITIVE_PATH = "SANDBOX_SENSITIVE_PATH"


#: Paths that are never writable, whatever the mode. These are not "protected
#: from the agent" in a security sense -- the agent runs as the user and could
#: bypass any in-process check. They are protected because a mistake here
#: damages the machine the runtime is running on, and the cost of the check is
#: nothing.
SENSITIVE_PATTERNS: tuple[str, ...] = (
    ".ssh/*",
    ".ssh",
    "**/.ssh/*",
    "*.pem",
    "**/*.pem",
    ".aws/*",
    "**/.aws/*",
    ".gnupg/*",
    "**/.gnupg/*",
    ".netrc",
    "**/.netrc",
    "id_rsa*",
    "**/id_rsa*",
    ".git/config",
    "**/.git/config",
)


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """What a session may do.

    ``workspace`` is resolved once, at construction, so a later change to the
    process's working directory cannot silently widen or narrow what the policy
    covers.
    """

    mode: str = MODE_WORKSPACE_WRITE
    workspace: str = ""
    #: Extra roots the policy treats as inside the workspace.
    extra_roots: tuple[str, ...] = ()
    #: Whether command execution is allowed at all, independent of the mode.
    allow_execute: bool = True
    #: Glob patterns for paths that are never writable.
    protected: tuple[str, ...] = SENSITIVE_PATTERNS

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise SandboxConfigurationError(
                f"unknown sandbox mode {self.mode!r}; expected one of {sorted(MODES)}"
            )
        if self.mode == MODE_WORKSPACE_WRITE and not self.workspace:
            # Without a workspace, "workspace-write" would have no boundary at
            # all -- it would mean "write anywhere", which is the opposite of
            # what the name promises.
            raise SandboxConfigurationError(
                "mode 'workspace-write' requires a workspace path, or it would "
                "permit writing anywhere"
            )

    # -- the boundary ----------------------------------------------------

    @property
    def roots(self) -> tuple[str, ...]:
        return (self.workspace, *self.extra_roots) if self.workspace else tuple(self.extra_roots)

    def contains(self, path: str) -> bool:
        """Whether a path lies inside the policy's roots.

        Resolution happens before the comparison, so ``..`` and symlinks cannot
        be used to step outside: a boundary that can be spelled around is not a
        boundary.
        """
        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, RuntimeError):
            # A path that cannot be resolved is not inside anything.
            return False
        for root in self.roots:
            try:
                root_path = Path(root).expanduser().resolve()
            except (OSError, RuntimeError):
                continue
            if resolved == root_path or root_path in resolved.parents:
                return True
        return False

    def is_protected(self, path: str) -> bool:
        """Whether a path matches a never-writable pattern."""
        try:
            resolved = str(Path(path).expanduser().resolve())
        except (OSError, RuntimeError):
            return True
        name = os.path.basename(resolved)
        for pattern in self.protected:
            if fnmatch.fnmatch(resolved, pattern) or fnmatch.fnmatch(name, pattern):
                return True
            # ``**/x`` should also match a bare ``x``.
            stripped = pattern.removeprefix("**/")
            if fnmatch.fnmatch(resolved, stripped) or fnmatch.fnmatch(name, stripped):
                return True
        return False

    # -- decisions -------------------------------------------------------

    def check(self, op: str, path: str = "") -> None:
        """Refuse the operation, or return.

        Raises rather than returning a boolean so that a caller cannot forget to
        check the result. A policy check whose result is easy to ignore is a
        policy check that will eventually be ignored.
        """
        if op == OP_EXECUTE:
            if not self.allow_execute or self.mode == MODE_NONE:
                raise SandboxDenied(
                    f"command execution is not permitted in mode {self.mode!r}",
                    code=CODE_EXEC_DISABLED,
                    op=op,
                )
            return

        if op not in (OP_READ, OP_WRITE):
            raise SandboxConfigurationError(f"unknown operation {op!r}")

        if self.mode == MODE_NONE:
            raise SandboxDenied(
                "filesystem access is not permitted in mode 'none'",
                code=CODE_NO_ACCESS,
                path=path,
                op=op,
            )

        if op == OP_READ:
            # Reading is permitted everywhere the mode allows any filesystem
            # access at all. Restricting reads inside a session that can already
            # execute commands would be theatre, not security.
            return

        # -- writes ------------------------------------------------------
        if self.mode == MODE_READ_ONLY:
            raise SandboxDenied(
                f"cannot write {path!r}: the session is read-only",
                code=CODE_READ_ONLY,
                path=path,
                op=op,
            )

        if self.is_protected(path):
            raise SandboxDenied(
                f"cannot write {path!r}: it matches a protected pattern "
                "(credentials and repository configuration are never written)",
                code=CODE_SENSITIVE_PATH,
                path=path,
                op=op,
            )

        if not self.contains(path):
            raise SandboxDenied(
                f"cannot write {path!r}: it is outside the workspace "
                f"({self.workspace})",
                code=CODE_OUTSIDE_WORKSPACE,
                path=path,
                op=op,
            )

    def allows(self, op: str, path: str = "") -> bool:
        """A non-raising form, for callers that need a boolean.

        Provided because some callers genuinely want to ask rather than to act --
        a UI showing what is possible, for instance. Code that is about to
        perform the operation should call :meth:`check` instead, so the refusal
        cannot be dropped.
        """
        try:
            self.check(op, path)
        except SandboxError:
            return False
        return True

    def describe(self) -> str:
        """A short human-readable summary, for the prompt and for logs."""
        if self.mode == MODE_NONE:
            return "no filesystem access and no command execution"
        if self.mode == MODE_READ_ONLY:
            return f"read-only access to {self.workspace or 'the filesystem'}"
        roots = ", ".join(self.roots) or "(none)"
        exec_note = "" if self.allow_execute else " (command execution disabled)"
        return f"read and write within {roots}{exec_note}"


#: Which tools perform which operation, and which argument names the path.
#:
#: A table rather than a convention, because the alternative -- each tool
#: checking its own policy -- is how a policy ends up enforced in five places and
#: forgotten in the sixth. Every tool that touches the filesystem is listed here,
#: and a tool that is not listed is not checked, so the table is the complete
#: answer to "what does the sandbox actually cover".
TOOL_OPERATIONS: Mapping[str, tuple[str, str]] = {
    "read": (OP_READ, "path"),
    "write": (OP_WRITE, "path"),
    "edit": (OP_WRITE, "path"),
    "glob": (OP_READ, "path"),
    "grep": (OP_READ, "path"),
    "bash": (OP_EXECUTE, ""),
}

#: Tools that take several paths, each of which must be checked.
TOOL_PATH_LISTS: Mapping[str, tuple[str, str]] = {
    # (operation, argument holding a list of paths)
    "read_many": (OP_READ, "paths"),
}


def enforcement_for(policy: "SandboxPolicy"):
    """Build an execute wrapper that enforces the policy on every tool call.

    This is the *mandatory path*: the pipeline runs the wrapper around every tool
    body, so a tool cannot be added that bypasses the check. The alternative --
    each tool consulting the policy itself -- is how a policy ends up enforced in
    five places and forgotten in the sixth, and the failure is silent because the
    tool simply works when it should not.

    The check runs **before** the body, so a denied operation has no effect at
    all rather than being undone afterwards.
    """

    def wrapper(execution: Any, run_body: Any) -> Any:
        name = str(getattr(execution, "name", ""))
        arguments = dict(getattr(execution, "arguments", {}) or {})

        spec = TOOL_OPERATIONS.get(name)
        if spec is not None:
            op, path_argument = spec
            path = str(arguments.get(path_argument, "") or "") if path_argument else ""
            try:
                policy.check(op, path)
            except SandboxDenied as denial:
                return _denied(denial)
            except SandboxError as error:
                return _denied(error)

        list_spec = TOOL_PATH_LISTS.get(name)
        if list_spec is not None:
            op, list_argument = list_spec
            for path in arguments.get(list_argument) or ():
                try:
                    policy.check(op, str(path))
                except SandboxDenied as denial:
                    return _denied(denial)
                except SandboxError as error:
                    return _denied(error)

        return run_body()

    return wrapper


async def _denied(error: Any) -> Any:
    """Turn a sandbox refusal into a tool result the model can read.

    A denial is an ordinary outcome, not an exception: the model should see what
    was refused and why, so it can adapt rather than retry blindly.
    """
    from ..tools.definition import ToolResult

    return ToolResult.fail(str(error), code=getattr(error, "code", "SANDBOX_DENIED"))


def resolve_policy(
    *,
    mode: str | None = None,
    workspace: str | None = None,
    extra_roots: Sequence[str] = (),
    allow_execute: bool = True,
    preset: str = "",
) -> SandboxPolicy:
    """Build a policy from a preset name, a mode, or neither.

    Presets exist because the useful combinations are few and naming them makes
    a configuration reviewable: ``"unattended"`` says more than
    ``mode="read-only", allow_execute=False``.
    """
    presets: dict[str, dict[str, Any]] = {
        "default": {"mode": MODE_WORKSPACE_WRITE},
        "cautious": {"mode": MODE_READ_ONLY, "allow_execute": False},
        "unattended": {"mode": MODE_READ_ONLY, "allow_execute": False},
        "locked": {"mode": MODE_NONE, "allow_execute": False},
    }

    if preset:
        if preset not in presets:
            raise SandboxConfigurationError(
                f"unknown preset {preset!r}; expected one of {sorted(presets)}"
            )
        settings = dict(presets[preset])
        # An explicit mode overrides the preset, so a caller can start from a
        # named stance and adjust one dimension.
        if mode is not None:
            settings["mode"] = mode
        if allow_execute is not True:
            settings["allow_execute"] = allow_execute
        return SandboxPolicy(
            workspace=workspace or os.getcwd(),
            extra_roots=tuple(extra_roots),
            **settings,
        )

    return SandboxPolicy(
        mode=mode or MODE_WORKSPACE_WRITE,
        workspace=workspace or os.getcwd(),
        extra_roots=tuple(extra_roots),
        allow_execute=allow_execute,
    )


# ---------------------------------------------------------------------------
# The fs and shell seams
# ---------------------------------------------------------------------------


class SandboxedFS:
    """A filesystem seam that consults the policy before every operation.

    Thin on purpose. It adds a policy check to each call and nothing else -- no
    caching, no path rewriting, no error translation. A seam that did more would
    become a second filesystem with its own semantics to reason about.
    """

    def __init__(self, policy: SandboxPolicy) -> None:
        self.policy = policy

    def read_text(self, path: str) -> str:
        self.policy.check(OP_READ, path)
        try:
            return Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"could not read {path}: {exc}") from exc

    def write_text(self, path: str, content: str) -> int:
        self.policy.check(OP_WRITE, path)
        target = Path(path).expanduser()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"could not write {path}: {exc}") from exc
        return len(content)

    def exists(self, path: str) -> bool:
        self.policy.check(OP_READ, path)
        return Path(path).expanduser().exists()

    def listdir(self, path: str) -> list[str]:
        self.policy.check(OP_READ, path)
        try:
            return sorted(p.name for p in Path(path).expanduser().iterdir())
        except OSError as exc:
            raise SandboxError(f"could not list {path}: {exc}") from exc


@dataclass
class Escalation:
    """A request to widen the policy for exactly one operation.

    Carrying the operation *with* the request is the point: an approval that
    granted "wider access" in general would be an approval to widen, and the
    next operation would inherit it.
    """

    op: str
    path: str
    reason: str
    policy: SandboxPolicy


def escalation_for(denial: SandboxDenied, policy: SandboxPolicy, reason: str = "") -> Escalation:
    """Build the escalation a denial implies.

    Only some denials can be escalated: a protected path is protected in every
    mode, so offering to widen for it would be offering something the policy
    cannot deliver.
    """
    if denial.code == CODE_SENSITIVE_PATH:
        raise SandboxError(
            "this path is protected in every mode, so there is nothing to escalate"
        )
    return Escalation(
        op=denial.op,
        path=denial.path,
        reason=reason or denial.code,
        policy=policy,
    )

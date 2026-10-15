"""Worktrees: isolated checkouts of one repository, for parallel work.

A worktree is a second working directory backed by the same repository, with its
own branch and its own index. That is exactly what parallel work needs: two
agents editing the same file in one checkout would clobber each other, and
copying the repository wholesale would lose the shared history that makes a diff
meaningful.

Three decisions shape this module:

**The runtime does not manage branches.** It creates a worktree on a branch it
names, and removes it when asked. Merging, rebasing and conflict resolution are
the caller's business -- a runtime that tried to resolve conflicts would be
guessing at intent it does not have.

**Removal is safe by default.** ``git worktree remove`` refuses when a worktree
has uncommitted changes, and that refusal is preserved rather than overridden.
Discarding someone's uncommitted work to satisfy a cleanup call is never the
right default; the caller has to say so explicitly.

**Everything is verified against git's own view.** The runtime asks git to list
worktrees rather than trusting its own bookkeeping, so a worktree removed by hand
does not become a phantom entry the runtime keeps offering.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from ..kernel.errors import ArmError

log = logging.getLogger("arm.workspace.worktree")


class WorktreeError(ArmError):
    """A worktree operation failed. The repository is left as it was found."""


class NotARepositoryError(WorktreeError):
    """The path is not inside a git repository."""


@dataclass(frozen=True, slots=True)
class Worktree:
    """One worktree, as git describes it."""

    path: Path
    #: The commit the worktree is on, or the branch name if it has one.
    revision: str = ""
    branch: str = ""
    #: True for the repository's primary checkout, which must never be removed.
    is_main: bool = False
    detached: bool = False

    @property
    def name(self) -> str:
        return self.path.name


class GitRunner:
    """Runs git commands with a consistent, inspectable interface.

    Every call returns ``(ok, stdout, stderr)`` rather than raising, because a
    failing git command is an ordinary outcome -- a branch that already exists, a
    worktree with local changes -- and each needs a different message rather than
    a stack trace.
    """

    def __init__(self, cwd: Path, *, timeout: float = 60.0) -> None:
        self.cwd = Path(cwd)
        self.timeout = timeout

    def run(self, *args: str, cwd: Path | None = None) -> tuple[bool, str, str]:
        try:
            completed = subprocess.run(
                ("git", *args),
                cwd=cwd or self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except FileNotFoundError as exc:
            raise WorktreeError("git is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise WorktreeError(f"git {args[0]} timed out") from exc
        return (
            completed.returncode == 0,
            completed.stdout.strip(),
            completed.stderr.strip(),
        )

    def must(self, *args: str, cwd: Path | None = None) -> str:
        """Run a command that must succeed, or raise with git's own message."""
        ok, out, err = self.run(*args, cwd=cwd)
        if not ok:
            raise WorktreeError(f"git {' '.join(args)} failed: {err or out}")
        return out


class WorktreeManager:
    """Creates, lists and removes worktrees for one repository."""

    def __init__(
        self,
        repo: Path | str,
        *,
        root: Path | str | None = None,
        git: GitRunner | None = None,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.git = git or GitRunner(self.repo)
        #: Where worktrees are created. Kept beside the repository rather than
        #: inside it, so a worktree is never mistaken for project content -- and
        #: so a glob over the repository does not walk into every checkout.
        self.root = (
            Path(root).resolve()
            if root is not None
            else self.repo.parent / f".{self.repo.name}-worktrees"
        )

    # -- inspection ------------------------------------------------------

    def is_repository(self) -> bool:
        ok, out, _ = self.git.run("rev-parse", "--is-inside-work-tree")
        return ok and out.strip() == "true"

    def require_repository(self) -> None:
        if not self.is_repository():
            raise NotARepositoryError(
                f"{self.repo} is not inside a git repository; worktrees need one"
            )

    def default_branch(self) -> str:
        """The repository's current branch, or ``HEAD`` when detached."""
        ok, out, _ = self.git.run("rev-parse", "--abbrev-ref", "HEAD")
        return out if ok and out and out != "HEAD" else "HEAD"

    def list(self) -> tuple[Worktree, ...]:
        """Every worktree git knows about.

        Parsed from git's own porcelain output rather than from a directory
        listing: a worktree removed by hand is not a worktree, and one created
        elsewhere still is.
        """
        self.require_repository()
        ok, out, err = self.git.run("worktree", "list", "--porcelain")
        if not ok:
            raise WorktreeError(f"could not list worktrees: {err}")

        worktrees: list[Worktree] = []
        current: dict[str, str] = {}

        def flush() -> None:
            if not current:
                return
            path = Path(current.get("worktree", ""))
            branch_ref = current.get("branch", "")
            branch = branch_ref.removeprefix("refs/heads/") if branch_ref else ""
            worktrees.append(
                Worktree(
                    path=path,
                    revision=current.get("HEAD", ""),
                    branch=branch,
                    is_main=not worktrees,
                    detached="detached" in current,
                )
            )
            current.clear()

        for line in out.splitlines():
            if not line.strip():
                flush()
                continue
            key, _, value = line.partition(" ")
            current[key] = value
        flush()
        return tuple(worktrees)

    def find(self, name: str) -> Worktree | None:
        """The worktree with this name, if it is one this manager owns.

        Scoped to the manager's own root. A worktree elsewhere in the repository
        may share a basename -- ``task-1`` is an obvious name to reuse -- and
        returning it would let ``remove`` delete a checkout this manager never
        created, along with its uncommitted work and its branch.
        """
        for worktree in self.list():
            if worktree.name != name:
                continue
            if not self._owns(worktree):
                continue
            return worktree
        return None

    def find_any(self, name: str) -> Worktree | None:
        """The worktree with this name, wherever it lives.

        Exposed separately so a caller that genuinely wants to see foreign
        worktrees can, without ``remove`` being able to act on one by accident.
        """
        for worktree in self.list():
            if worktree.name == name:
                return worktree
        return None

    def _owns(self, worktree: Worktree) -> bool:
        """Whether this manager created (and may therefore remove) a worktree.

        Ownership is by location, not by name: the worktree must sit directly
        under this manager's root. Anything else belongs to someone else.
        """
        if worktree.is_main:
            return False
        try:
            worktree.path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return False
        return True

    def is_clean(self, worktree: Worktree) -> tuple[bool, str]:
        """Whether a worktree has no uncommitted changes.

        Untracked files count as dirty. A worktree with new files in it holds
        work that would be destroyed by removal, and "untracked" is not the same
        as "disposable".
        """
        if not worktree.path.is_dir():
            # Not clean: the directory is missing. Reporting it as clean would
            # have a listing advertise a checkout that does not exist, and would
            # let removal proceed as though there were nothing to lose.
            #
            # It is also not *dirty* in the usual sense -- there is no work to
            # preserve -- so the reason says what is actually wrong, and points at
            # the operation that clears the stale entry.
            return False, "the worktree directory is missing (run prune to clear it)"
        ok, out, err = self.git.run("status", "--porcelain", cwd=worktree.path)
        if not ok:
            return False, f"could not read status: {err}"
        if out.strip():
            lines = out.strip().splitlines()
            return False, f"{len(lines)} uncommitted change(s)"
        return True, "clean"

    # -- creation --------------------------------------------------------

    def create(
        self,
        name: str,
        *,
        branch: str | None = None,
        from_revision: str = "HEAD",
        reuse: bool = False,
    ) -> Worktree:
        """Create a worktree on a new branch.

        The name is validated before it reaches git: a name containing ``..`` or
        a path separator would place the checkout outside the worktree root, and
        validating after the fact would mean creating a directory in the wrong
        place first.
        """
        self.require_repository()
        self._validate_name(name)
        # Every value that reaches git's argv needs the same check the name gets.
        # A leading ``-`` is read by git as an *option*, so ``from_revision="--lock"``
        # would lock the worktree it creates -- and a locked worktree cannot be
        # removed, by anyone, through this API.
        self._validate_revision(from_revision)
        if branch is not None:
            self._validate_branch(branch)

        target = self.root / name
        existing = self.find(name)
        if existing is not None:
            if reuse:
                return existing
            raise WorktreeError(
                f"a worktree named {name!r} already exists at {existing.path}"
            )

        if target.exists():
            # A leftover directory with no git entry. Refusing rather than
            # deleting: it may hold work that was never committed.
            if not reuse:
                raise WorktreeError(
                    f"{target} already exists but is not a registered worktree; "
                    "remove it by hand, or pass reuse=True to adopt it"
                )

        branch_name = branch or f"arm/{name}"
        self.root.mkdir(parents=True, exist_ok=True)

        # ``-B`` would reset an existing branch, which would silently move a ref
        # the user may care about. ``-b`` fails instead, and the failure is
        # reported as-is.
        ok, out, err = self.git.run(
            "worktree", "add", "-b", branch_name, str(target), from_revision
        )
        if not ok:
            if "already exists" in err:
                raise WorktreeError(
                    f"branch {branch_name!r} already exists; pass a different "
                    f"branch name, or delete it first"
                )
            raise WorktreeError(f"could not create worktree: {err or out}")

        created = self.find(name)
        if created is None:
            raise WorktreeError(
                f"git reported success creating {name!r} but the worktree is not "
                "listed; the repository may be in an unexpected state"
            )
        return created

    #: Names that are syntactically valid but must never be used, because they
    #: do not name a *child* of the worktree root.
    #:
    #: ``.`` is the important one: joined onto the root it resolves to the root
    #: itself, so ``git worktree add`` would check the repository out *over* the
    #: directory meant to hold the worktrees -- creating a checkout where the
    #: manager expects to find only its own entries, and leaving it behind when
    #: the subsequent bookkeeping lookup fails. ``-`` is rejected because a
    #: leading dash reaches git as an option.
    _RESERVED_NAMES = frozenset({".", "..", "-", "--"})

    def _validate_name(self, name: str) -> None:
        """Reject a name that is not a usable child of the worktree root.

        Validation happens before the name reaches git. Checking afterwards would
        mean the checkout already exists in the wrong place, and no amount of
        validation can undo a directory that has been created.
        """
        if not name or not name.strip():
            raise WorktreeError("a worktree name is required")
        if name != name.strip():
            raise WorktreeError(f"worktree name {name!r} has surrounding whitespace")
        if name in self._RESERVED_NAMES:
            raise WorktreeError(
                f"worktree name {name!r} is reserved: it does not name a child of "
                "the worktree root"
            )
        if any(part in name for part in ("/", "\\", "..", "\0")):
            raise WorktreeError(
                f"worktree name {name!r} must not contain path separators or '..'"
            )
        if name.startswith("-"):
            # A leading dash would be read by git as an option rather than as a
            # path, so the command would do something other than what was asked.
            raise WorktreeError(
                f"worktree name {name!r} must not start with '-'"
            )
        if not re.match(r"^[A-Za-z0-9._-]+$", name):
            raise WorktreeError(
                f"worktree name {name!r} may only contain letters, digits, '.', "
                "'_' and '-'"
            )

    def _validate_revision(self, revision: str) -> None:
        """Reject a revision git would read as an option.

        The same hazard as a leading ``-`` in a name, and it was missed for
        exactly the reason it is easy to miss: the name was the obvious input to
        validate, and this one looks like a value rather than an argument.
        """
        if not revision or not revision.strip():
            raise WorktreeError("a starting revision is required")
        if revision.startswith("-"):
            raise WorktreeError(
                f"revision {revision!r} must not start with '-' (git would read it "
                "as an option)"
            )
        if any(ch in revision for ch in ("\0", "\n")):
            raise WorktreeError(f"revision {revision!r} contains a control character")

    def _validate_branch(self, branch: str) -> None:
        """Reject a branch name git would read as an option, or that is invalid."""
        if not branch or not branch.strip():
            raise WorktreeError("a branch name must not be empty")
        if branch.startswith("-"):
            raise WorktreeError(
                f"branch {branch!r} must not start with '-' (git would read it as "
                "an option)"
            )
        if any(ch in branch for ch in ("\0", "\n", " ")):
            raise WorktreeError(f"branch {branch!r} contains an illegal character")
        if branch.endswith("/") or "//" in branch or branch.endswith(".lock"):
            raise WorktreeError(f"branch {branch!r} is not a valid git branch name")

    # -- removal ---------------------------------------------------------

    def remove(
        self, name: str, *, force: bool = False, delete_branch: bool = False
    ) -> Worktree:
        """Remove a worktree.

        Uncommitted changes make this refuse unless ``force`` is set. The refusal
        is the point: cleanup calls are routine, and a routine call must never be
        the thing that discards work.
        """
        worktree = self.find(name)
        if worktree is None:
            # Say *why* when a same-named worktree exists elsewhere. "No worktree
            # named X" would be confusing when ``git worktree list`` clearly shows
            # one, and the caller would go looking for a bug in the wrong place.
            foreign = self.find_any(name)
            if foreign is not None:
                raise WorktreeError(
                    f"a worktree named {name!r} exists at {foreign.path}, but it "
                    f"is outside this manager's root ({self.root}); it was not "
                    "created here, so it is not ours to remove"
                )
            raise WorktreeError(f"no worktree named {name!r}")
        if worktree.is_main:
            raise WorktreeError(
                f"{worktree.path} is the repository's main checkout and cannot be "
                "removed"
            )

        clean, reason = self.is_clean(worktree)
        if not clean and not force:
            raise WorktreeError(
                f"worktree {name!r} has {reason}; commit or discard the changes, "
                "or pass force=True to remove it anyway"
            )

        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(worktree.path))

        ok, out, err = self.git.run(*args)
        if not ok:
            raise WorktreeError(f"could not remove worktree: {err or out}")

        if delete_branch and worktree.branch:
            # A branch that is checked out elsewhere cannot be deleted, and git
            # says so; that is reported rather than ignored so a caller is not
            # left believing cleanup happened.
            ok, out, err = self.git.run("branch", "-D" if force else "-d", worktree.branch)
            if not ok:
                raise WorktreeError(
                    f"worktree removed, but branch {worktree.branch!r} could not "
                    f"be deleted: {err or out}"
                )

        # git removes the directory; a leftover is pruned so the bookkeeping
        # stays consistent with the filesystem.
        if worktree.path.exists():
            try:
                shutil.rmtree(worktree.path)
            except OSError as exc:
                raise WorktreeError(
                    f"worktree removed from git but {worktree.path} could not be "
                    f"deleted: {exc}"
                ) from exc
            self.git.run("worktree", "prune")

        return worktree

    def prune(self) -> int:
        """Drop bookkeeping for worktrees whose directories are gone.

        Returns how many entries were pruned. A worktree deleted with ``rm -rf``
        leaves git believing it still exists, which makes a later create with the
        same name fail for no visible reason.
        """
        self.require_repository()
        before = len(self.list())
        ok, _out, err = self.git.run("worktree", "prune")
        if not ok:
            raise WorktreeError(f"could not prune worktrees: {err}")
        return before - len(self.list())

    def cleanup(self, *, force: bool = False) -> list[str]:
        """Remove every worktree this manager created, leaving the main one.

        Dirty worktrees are skipped unless forced, and reported by name, so a
        cleanup that could not finish says which worktrees are still there
        instead of failing silently.
        """
        skipped: list[str] = []
        for worktree in self.list():
            if worktree.is_main:
                continue
            # Only worktrees under this manager's root are ours to remove. One
            # created elsewhere belongs to someone else.
            if not self._owns(worktree):
                continue
            try:
                self.remove(worktree.name, force=force)
            except WorktreeError as exc:
                log.warning("could not clean up worktree %s: %s", worktree.name, exc)
                skipped.append(worktree.name)
        return skipped

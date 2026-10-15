"""FAILING tests: confirmed defects in the worktree module and its tools.

Two tests here (W1, W1b -- cross-root removal) were fixed by the Lead while this
review was running; they are now ordinary passing regression guards and are
marked as such in their docstrings. The rest are still-open ``xfail`` findings.

Each test carries its minimal reproduction, the observed output, and the exact
minimal fix. All are ``xfail(strict=True)``: while the bug is present the test
fails, which ``strict=True`` reports as ``xfail``; the moment the fix lands the
test passes and ``strict=True`` turns that into ``XPASS(strict)`` = a failure.

Nothing here modifies ``src/``.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

import _bootstrap  # noqa: F401
from arm.workspace.worktree import (
    GitRunner,
    WorktreeError,
    WorktreeManager,
)


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(("git", *args), cwd=cwd, capture_output=True, text=True)


def make_repo(base: Path, name: str = "project") -> Path:
    """A real repository with one commit."""
    path = base / name
    path.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=path)
    git("config", "user.email", "test@example.com", cwd=path)
    git("config", "user.name", "Test", cwd=path)
    (path / "README.md").write_text("# project\n")
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "initial", cwd=path)
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path)


@pytest.fixture
def manager(repo: Path, tmp_path: Path) -> WorktreeManager:
    return WorktreeManager(repo, root=tmp_path / "worktrees")


async def invoke(tool, arguments: dict):
    return await tool.body(arguments, None)


# ---------------------------------------------------------------------------
# BUG W1 -- find() matches by basename across the root boundary, so remove()
#           destroys a worktree this manager does not own
# ---------------------------------------------------------------------------


def test_remove_does_not_touch_a_worktree_outside_the_manager_root(repo, tmp_path):
    """``cleanup()`` is careful to skip worktrees outside ``self.root``:

        # Only worktrees under this manager's root are ours to remove. One
        # created elsewhere belongs to someone else.
        try:
            worktree.path.relative_to(self.root)
        except ValueError:
            continue

    ``remove()`` applies no such check. It resolves the name through ``find()``,
    which compares ``worktree.name`` -- ``Path.name``, a bare basename -- so a
    worktree created by someone else at a different path but with the same
    basename is selected and destroyed.

    This is reachable from the model: ``worktree_remove`` takes a bare ``name``,
    and ``worktree_list`` prints that same basename, so the model cannot tell the
    two apart. The manager's own root is ``<repo>/../.<repo>-worktrees``, while a
    colleague's checkout is anywhere.

    Observed::

        foreign path: /tmp/.../elsewhere/prod-checkout   exists: True
        mgr.remove('prod-checkout') -> 'prod-checkout'
        foreign still on disk? False      <-- another checkout was deleted
        branch still there? hotfix
        reflog keeps the commit: 1

    ``tests/test_worktree.py::test_cleanup_ignores_worktrees_outside_its_root``
    covers exactly this hazard for ``cleanup`` and passes; ``remove`` was never
    given the same guard.

    FIXED during this review: ``find`` is now scoped to the manager's root and
    ``remove`` reports the foreign path by name. This test is the regression
    guard.

    Original suggested fix (for reference): make ``find``/``remove`` root-aware::

        def find(self, name: str, *, under_root: bool = False) -> Worktree | None:
            for worktree in self.list():
                if worktree.name != name:
                    continue
                if under_root:
                    try:
                        worktree.path.relative_to(self.root)
                    except ValueError:
                        continue
                return worktree
            return None

    with ``remove`` calling ``self.find(name, under_root=True)`` and raising a
    distinct error when the name exists only outside the root; or match on the
    full path rather than the basename.
    """
    manager = WorktreeManager(repo, root=tmp_path / "worktrees")
    foreign = tmp_path / "elsewhere" / "prod-checkout"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    added = git("worktree", "add", "-b", "hotfix", str(foreign), cwd=repo)
    assert added.returncode == 0, added.stderr
    (foreign / "unpushed.txt").write_text("a day of work\n")
    git("add", "-A", cwd=foreign)
    git("commit", "-q", "-m", "wip", cwd=foreign)

    # The foreign worktree still exists and git still lists it...
    assert foreign.exists(), "precondition: the foreign worktree exists"
    assert any(w.name == "prod-checkout" for w in manager.list()), (
        "precondition: git lists the foreign worktree"
    )
    # ...but this manager must not resolve it as one of its own.
    assert manager.find("prod-checkout") is None, (
        "find() returned a worktree outside the manager's root"
    )

    with pytest.raises(WorktreeError):
        manager.remove("prod-checkout")

    assert foreign.exists(), "a worktree outside the manager's root was removed"


def test_the_remove_tool_cannot_delete_a_foreign_worktree(repo, tmp_path):
    """BUG W1 as the model experiences it.

    ``worktree_remove`` is registered whenever the process is inside a git
    repository (``cli/main.py``). A model asked to "clean up the worktrees" can
    name one it saw in ``worktree_list`` -- which prints only the basename -- and
    delete a checkout outside the manager's root, together with an unmerged
    branch if it also sets ``delete_branch``.

    Observed before the fix::

        worktree_remove -> Removed worktree release-candidate.  is_error: False
        foreign still on disk? False
        branch release/v2 still there? ''      <-- unmerged branch deleted too
        uncommitted wip.py recoverable? False

    FIXED during this review by the same root-aware lookup. Regression guard.
    """
    from arm.workspace.tools import WorktreeTools

    manager = WorktreeManager(repo, root=tmp_path / "worktrees")
    foreign = tmp_path / "elsewhere" / "release-candidate"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    added = git("worktree", "add", "-b", "release/v2", str(foreign), cwd=repo)
    assert added.returncode == 0, added.stderr
    (foreign / "feature.py").write_text("print('a week of work')\n")
    git("add", "-A", cwd=foreign)
    git("commit", "-q", "-m", "feature work", cwd=foreign)
    (foreign / "wip.py").write_text("not committed yet\n")

    tools = {t.name: t for t in WorktreeTools(manager).tools()}
    result = asyncio.run(
        invoke(
            tools["worktree_remove"],
            {"name": "release-candidate", "force": True, "delete_branch": True},
        )
    )

    assert result.is_error, "the tool removed a worktree outside its root"
    assert foreign.exists(), "the foreign checkout was deleted"
    branches = git("branch", "--list", "release/v2", cwd=repo).stdout
    assert "release/v2" in branches, "the unmerged branch was deleted"


# ---------------------------------------------------------------------------
# BUG W2 -- git's "directory already exists" is reported as a branch conflict
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W2: `if 'already exists' in err` matches git's DIRECTORY conflict too, so the message blames the branch",
)
def test_a_directory_conflict_is_not_reported_as_a_branch_conflict(manager):
    """``create`` classifies a failed ``git worktree add`` by substring:

        if "already exists" in err:
            raise WorktreeError(f"branch {branch_name!r} already exists; ...")

    Git uses the same phrase for two different failures. For a genuine branch
    conflict::

        fatal: a branch named 'taken' already exists

    For a target directory that already exists::

        fatal: '/tmp/.../wt/stuck' already exists

    The second is misreported as the first, and the advice it gives ("pass a
    different branch name") is wrong and cannot work: a new branch name changes
    nothing, because the directory is the problem.

    This is the *ordinary* ``reuse=True`` path when the leftover directory is
    non-empty (the empty case is handled earlier by ``target.exists()``).

    Observed::

        manager says: branch 'fresh-branch' already exists; pass a different
                      branch name, or delete it first
        raw git stderr: "fatal: '/tmp/.../wt/stuck' already exists"

    Fix: match git's branch-specific wording instead of the shared fragment::

        if "a branch named" in err:
            raise WorktreeError(f"branch {branch_name!r} already exists; ...")
        if "already exists" in err:
            raise WorktreeError(
                f"{target} already exists; remove it by hand, or pass reuse=True"
            )
    """
    manager.root.mkdir(parents=True, exist_ok=True)
    leftover = manager.root / "stuck"
    leftover.mkdir()
    (leftover / "unsaved.txt").write_text("important\n")

    with pytest.raises(WorktreeError) as caught:
        manager.create("stuck", reuse=True, branch="totally-fresh-branch")

    message = str(caught.value)
    assert "branch" not in message or "directory" in message, (
        f"a directory conflict was blamed on the branch name: {message!r}"
    )


# ---------------------------------------------------------------------------
# BUG W3 -- reuse=True over a non-empty leftover creates an orphan branch and
#           then permanently refuses
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W3: `reuse=True` skips the leftover check, so git creates the branch before failing on the directory, leaving an orphan ref that makes every retry fail",
)
def test_reuse_over_a_non_empty_directory_does_not_orphan_a_branch(manager, repo):
    """``create`` consults ``target.exists()`` only when ``reuse`` is false:

        if target.exists():
            if not reuse:
                raise WorktreeError(...)

    So with ``reuse=True`` it proceeds to ``git worktree add -b arm/<name>``. Git
    **creates the branch first**, then fails because the directory is non-empty.
    The branch is left behind with no worktree, and the next attempt -- with or
    without ``reuse``, with or without a different branch name -- now fails on
    the branch instead. The directory can never be adopted.

    Observed::

        branches before: '* main'
        create('stuck', reuse=True) -> WorktreeError: branch 'arm/stuck' already exists
        branches after : 'arm/stuck * main'      <-- orphan ref, no worktree
        worktrees registered: ['project']

    Fix: apply the leftover check regardless of ``reuse`` -- ``reuse`` should
    adopt a *registered* worktree (handled just above by ``find``), not a bare
    directory::

        if target.exists() and not reuse:
            raise WorktreeError(...)

    If adopting a bare directory is genuinely wanted, git needs an empty target
    (``mkdir`` then ``git worktree add`` without ``-b``, or remove-then-add), and
    the branch must not be created speculatively.
    """
    manager.root.mkdir(parents=True, exist_ok=True)
    leftover = manager.root / "stuck"
    leftover.mkdir()
    (leftover / "unsaved.txt").write_text("important\n")

    branches_before = git("branch", "--list", cwd=repo).stdout

    with pytest.raises(WorktreeError):
        manager.create("stuck", reuse=True)

    branches_after = git("branch", "--list", cwd=repo).stdout
    assert branches_after == branches_before, (
        "a failed create left an orphan branch behind: "
        f"{branches_before!r} -> {branches_after!r}"
    )
    assert (leftover / "unsaved.txt").exists(), "the leftover work was destroyed"


# ---------------------------------------------------------------------------
# BUG W4 -- a missing working directory is reported as "git is not installed"
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W4: `subprocess.run(cwd=...)` raises FileNotFoundError for a missing CWD as well as a missing git binary, and both are reported as 'git is not installed'",
)
def test_a_missing_repository_path_is_not_reported_as_missing_git(tmp_path):
    """``GitRunner.run`` catches ``FileNotFoundError`` and always blames git:

        except FileNotFoundError as exc:
            raise WorktreeError("git is not installed") from exc

    ``subprocess.run`` raises ``FileNotFoundError`` for a missing *executable*
    **and** for a missing *cwd*. The second case is ordinary: the repository was
    moved or deleted after the manager was constructed, or the process changed
    directory out from under a relative path.

    Observed::

        WorktreeManager('/nonexistent-dir-xyz').is_repository()
            -> WorktreeError: git is not installed

    The message sends the caller to install git, which is already installed.

    Fix: distinguish the two by checking the exception's filename, or by
    validating the cwd first::

        except FileNotFoundError as exc:
            if not (cwd or self.cwd).is_dir():
                raise WorktreeError(
                    f"the working directory {(cwd or self.cwd)} does not exist"
                ) from exc
            raise WorktreeError("git is not installed") from exc
    """
    missing = tmp_path / "gone"
    manager = WorktreeManager(missing, root=tmp_path / "wt")

    with pytest.raises(WorktreeError) as caught:
        manager.is_repository()

    message = str(caught.value)
    assert "git is not installed" not in message, (
        f"a missing directory was reported as a missing git binary: {message!r}"
    )


# ---------------------------------------------------------------------------
# BUG W5 -- a raw OSError escapes create() instead of a WorktreeError
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W5: `self.root.mkdir(parents=True, exist_ok=True)` is unguarded, so OSError escapes the documented error type",
)
def test_an_unusable_worktree_root_raises_a_worktree_error(repo, tmp_path):
    """``create`` calls ``self.root.mkdir(parents=True, exist_ok=True)`` with no
    guard. If any parent of the root is a file, ``mkdir`` raises
    ``NotADirectoryError`` -- an ``OSError``, not a ``WorktreeError``.

    The module's contract is that failures raise ``WorktreeError``; the tools
    catch exactly that type. ``WorktreeTools._create`` does
    ``except WorktreeError``, so this escapes the tool body. The pipeline's broad
    ``except Exception`` turns it into a generic ``SCHEDULER_FAILURE`` result, so
    the model sees "tool 'worktree_create' failed: [Errno 20] Not a directory"
    instead of the actionable message the module is written to produce.

    Observed::

        create with root under a file: RAISED NotADirectoryError: [Errno 20] Not a directory: '.../blocker/sub'
        isinstance(NotADirectoryError(), WorktreeError) = False

    Fix: wrap the mkdir::

        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorktreeError(
                f"could not create the worktree root {self.root}: {exc}"
            ) from exc
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    manager = WorktreeManager(repo, root=blocker / "sub")

    with pytest.raises(WorktreeError):
        manager.create("x")


# ---------------------------------------------------------------------------
# BUG W6 -- from_revision accepts a git option (argument injection)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W6: `from_revision` is interpolated into the git argv unvalidated, so a value beginning with '-' is read as an option",
)
def test_from_revision_cannot_inject_a_git_option(manager, repo):
    """``_validate_name`` rejects a name starting with ``-`` precisely because it
    "would be read by git as an option rather than as a path". ``from_revision``
    and ``branch`` reach the same argv position and get no such check.

    ``from_revision="--lock"`` makes git create the worktree **locked**. The lock
    is deliberate git state that survives ``remove(force=True)``:

        $ git worktree remove --force <path>
        fatal: cannot remove a locked working tree, lock reason: added with --lock

    The manager offers no ``unlock`` operation, so a worktree created this way
    can never be removed through the tool surface. ``cleanup()`` skips it and
    reports it every time.

    Observed::

        create('lk', from_revision='--lock') -> created
        worktree list --porcelain: ... locked added with --lock
        remove('lk')             -> WorktreeError: cannot remove a locked working tree
        remove('lk', force=True) -> WorktreeError: cannot remove a locked working tree

    ``from_revision`` is model-supplied through ``WorktreeCreateParams``, so this
    is reachable without a hand-constructed call.

    Fix: reject an option-looking revision (and branch), matching the name rule::

        if from_revision.startswith("-"):
            raise WorktreeError(
                f"from_revision {from_revision!r} must not start with '-'"
            )
    """
    worktree = manager.create("lk", from_revision="--lock")
    assert worktree is not None

    listing = git("worktree", "list", "--porcelain", cwd=repo).stdout
    assert "locked" not in listing, (
        "from_revision was interpreted as the git option --lock"
    )

    # And if it somehow were locked, the manager must still be able to undo it.
    manager.remove("lk", force=True)
    assert not worktree.path.exists()


# ---------------------------------------------------------------------------
# BUG W7 -- a phantom worktree is reported "clean" and blocks recreation
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W7: a worktree whose directory was deleted is reported clean, listed as usable, and blocks create() with no way out but prune()",
)
def test_a_worktree_whose_directory_vanished_is_not_reported_clean(manager):
    """``is_clean`` short-circuits on a missing directory:

        if not worktree.path.is_dir():
            return True, "the worktree directory is already gone"

    It reports **clean**, which is the signal callers use to mean "safe to
    remove, holds no work". ``worktree_list`` renders that as the status column,
    so the model reads a phantom as a healthy, empty worktree. ``remove`` then
    succeeds (git prunes the entry), but ``create`` with the same name refuses --
    ``find`` returns the phantom and the name is taken.

    Observed::

        worktree_list:
          ghost  /tmp/.../wt/ghost  arm/ghost  linked  clean      <-- directory does not exist
        is_clean(ghost) -> (True, 'the worktree directory is already gone')
        create('ghost') -> WorktreeError: a worktree named 'ghost' already exists

    ``prune()`` is the intended escape, but nothing in ``src/`` calls it: it is
    reachable only from ``tests/test_worktree.py``. The model has no prune tool,
    so from the tool surface the name is simply unusable.

    Fix: report a missing directory as *not* clean -- it is an anomaly, not a
    clean worktree::

        if not worktree.path.is_dir():
            return False, "the worktree directory is missing (run prune)"

    and/or have ``create`` treat a phantom as absent, pruning it and retrying.
    """
    worktree = manager.create("ghost")
    shutil.rmtree(worktree.path)

    phantom = manager.find("ghost")
    assert phantom is not None, "precondition: git still lists it"

    clean, reason = manager.is_clean(phantom)
    assert not clean, f"a worktree with no directory was reported clean: {reason!r}"

    # And the name must be reusable without a manual prune.
    recreated = manager.create("ghost")
    assert recreated.path.exists()


@pytest.mark.xfail(
    strict=True,
    reason="BUG W7b: worktree_list shows a phantom as a clean, usable checkout",
)
def test_the_list_tool_does_not_advertise_a_phantom_as_clean(manager):
    """BUG W7 through the tool surface.

    Observed::

        ghost  /tmp/.../wt/ghost  arm/ghost  linked  clean

    Fix: as BUG W7 -- ``is_clean`` must not report a missing directory as clean.
    """
    from arm.workspace.tools import WorktreeTools

    worktree = manager.create("ghost")
    shutil.rmtree(worktree.path)

    tools = {t.name: t for t in WorktreeTools(manager).tools()}
    listing = asyncio.run(invoke(tools["worktree_list"], {}))

    ghost_line = next(
        (line for line in listing.content.splitlines() if "ghost" in line), None
    )
    assert ghost_line is not None, "the phantom should still be listed"
    assert "clean" not in ghost_line, (
        f"the listing advertises a non-existent checkout as clean: {ghost_line!r}"
    )


# ---------------------------------------------------------------------------
# BUG W8 -- --no-checkout yields an empty worktree reported as a real checkout
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W8: `from_revision` may carry git options, so --no-checkout produces an empty directory that create() reports as a successful checkout",
)
def test_create_does_not_report_success_for_an_empty_checkout(manager):
    """``git worktree add --no-checkout <path> <rev>`` registers a worktree and
    creates the directory **without populating it**. ``create`` sees a listed
    worktree, returns it, and the tool tells the model:

        Created worktree noco at /tmp/.../wt/noco on branch arm/noco.
        Use absolute paths under that directory to work in it.

    The directory is empty -- no ``README.md``, no tracked files. A model that
    follows the instruction finds nothing there.

    Observed::

        create('noco', from_revision='--no-checkout') -> created
        README.md present? False   dir exists? True

    Fix: reject option-looking revisions (the same check as BUG W6), which also
    closes ``--lock``. ``--no-checkout`` is the benign-looking member of the same
    family.
    """
    worktree = manager.create("noco", from_revision="--no-checkout")

    assert (worktree.path / "README.md").exists(), (
        "create() reported success for a worktree that was never populated"
    )


# ---------------------------------------------------------------------------
# BUG W9 -- a locked worktree cannot be removed and the error gives no way out
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W9: `remove(force=True)` cannot clear a git worktree lock, and the manager offers no unlock, so cleanup can never finish",
)
def test_a_locked_worktree_can_be_removed_with_force(manager, repo):
    """A worktree locked with ``git worktree lock`` (which git users do to
    protect a checkout on removable media, and which ``from_revision='--lock'``
    does by accident -- BUG W6) is refused by both ``remove()`` and
    ``remove(force=True)``:

        fatal: cannot remove a locked working tree;
        use 'remove -f -f' to override or unlock first

    ``force=True`` is documented as "remove even with uncommitted changes ...
    only set it when you are certain the changes are not wanted". A caller who
    has said that has expressed the intent git's second ``-f`` encodes, but the
    manager passes only one ``--force``. ``cleanup(force=True)`` therefore reports
    the worktree as skipped on every run, forever.

    Observed::

        remove('locked')             -> WorktreeError: cannot remove a locked working tree
        remove('locked', force=True) -> WorktreeError: cannot remove a locked working tree

    Fix: when ``force`` is set, clear the lock first (or pass ``-f -f``)::

        if force:
            self.git.run("worktree", "unlock", str(worktree.path))
            args.append("--force")

    ``git worktree unlock`` on an unlocked worktree fails harmlessly, so the call
    can be unconditional under ``force``.
    """
    worktree = manager.create("locked")
    locked = git("worktree", "lock", str(worktree.path), cwd=repo)
    assert locked.returncode == 0, locked.stderr

    manager.remove("locked", force=True)

    assert not worktree.path.exists(), "force=True could not remove a locked worktree"
    assert manager.find("locked") is None


# ---------------------------------------------------------------------------
# BUG W10 -- a worktree removed by hand is not reconciled before create()
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG W10: create() does not prune a stale registration, so a worktree deleted with rm -rf makes its own name permanently unusable from the tool surface",
)
def test_create_recovers_the_name_of_a_hand_deleted_worktree(manager):
    """The module's docstring states the design:

        "Everything is verified against git's own view. The runtime asks git to
         list worktrees rather than trusting its own bookkeeping, so a worktree
         removed by hand does not become a phantom entry the runtime keeps
         offering."

    A worktree deleted with ``rm -rf`` *is* such a phantom: git keeps the
    registration, ``list``/``find`` keep returning it, and ``create`` refuses the
    name with "a worktree named 'x' already exists". The docstring's own remedy,
    ``prune()``, is called from nowhere in ``src/`` -- only from the tests -- and
    is not exposed as a tool, so a model cannot reach it.

    Observed::

        shutil.rmtree(w.path)
        list still shows: [('project', True), ('ghost', False)]   <-- path.exists() False
        create('ghost') -> WorktreeError: a worktree named 'ghost' already exists

    Fix: have ``create`` reconcile before refusing -- if the name resolves to a
    worktree whose directory is gone, prune and continue::

        existing = self.find(name)
        if existing is not None and not existing.path.is_dir():
            self.prune()
            existing = self.find(name)
        if existing is not None:
            ...
    """
    worktree = manager.create("ghost")
    shutil.rmtree(worktree.path)

    recreated = manager.create("ghost")
    assert recreated.path.exists(), "the name could not be reused after a hand deletion"
    assert recreated.path == worktree.path

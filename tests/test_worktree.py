"""Worktree tests, against a real git repository.

A worktree is a git feature, not an abstraction this runtime invents, so the only
honest way to test it is to let git do the work and check what it says. These
tests create a real repository in a temporary directory.

The theme is **removal must not destroy work**. Cleanup calls are routine, and a
routine call must never be the thing that discards uncommitted changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from arm.tools.definition import ToolResult
from arm.workspace.worktree import (
    NotARepositoryError,
    WorktreeError,
    WorktreeManager,
)


async def invoke(tool, arguments: dict) -> ToolResult:
    """Call a tool body and return its result.

    Written as a coroutine function so ``asyncio.run`` receives a coroutine
    rather than the looser ``Awaitable`` the tool-body type allows.
    """
    return await tool.body(arguments, None)


def git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=cwd, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real repository with one commit."""
    path = tmp_path / "project"
    path.mkdir()
    git("init", "-q", "-b", "main", cwd=path)
    git("config", "user.email", "test@example.com", cwd=path)
    git("config", "user.name", "Test", cwd=path)
    (path / "README.md").write_text("# project\n")
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "initial", cwd=path)
    return path


@pytest.fixture
def manager(repo: Path, tmp_path: Path) -> WorktreeManager:
    return WorktreeManager(repo, root=tmp_path / "worktrees")


# ---------------------------------------------------------------------------
# Repository detection
# ---------------------------------------------------------------------------


def test_recognizes_a_repository(repo: Path, manager: WorktreeManager):
    assert manager.is_repository()


def test_refuses_a_directory_that_is_not_a_repository(tmp_path: Path):
    """A clear error beats a confusing git failure later."""
    plain = tmp_path / "plain"
    plain.mkdir()
    manager = WorktreeManager(plain, root=tmp_path / "wt")
    with pytest.raises(NotARepositoryError, match="not inside a git repository"):
        manager.require_repository()


def test_lists_the_main_worktree(repo: Path, manager: WorktreeManager):
    """The repository's own checkout is a worktree, and is marked as main."""
    worktrees = manager.list()
    assert len(worktrees) == 1
    assert worktrees[0].is_main
    assert worktrees[0].path == repo


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_creates_a_worktree_on_a_new_branch(manager: WorktreeManager):
    """A worktree gets its own branch, so two checkouts cannot fight over HEAD."""
    worktree = manager.create("task-1")

    assert worktree.path.is_dir()
    assert worktree.branch == "arm/task-1"
    assert not worktree.is_main
    # It is a real checkout: the committed file is present.
    assert (worktree.path / "README.md").read_text() == "# project\n"


def test_creates_a_worktree_from_a_named_revision(repo: Path, manager: WorktreeManager):
    """A worktree can start from any revision, not only HEAD."""
    git("tag", "v1", cwd=repo)
    (repo / "later.txt").write_text("later\n")
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "later", cwd=repo)

    worktree = manager.create("from-tag", from_revision="v1")
    assert not (worktree.path / "later.txt").exists()
    assert (worktree.path / "README.md").exists()


def test_a_duplicate_name_is_refused(manager: WorktreeManager):
    """Two worktrees with one name would make ``find`` ambiguous."""
    manager.create("dup")
    with pytest.raises(WorktreeError, match="already exists"):
        manager.create("dup")


def test_reuse_returns_the_existing_worktree(manager: WorktreeManager):
    """A caller that wants idempotence can ask for it explicitly."""
    first = manager.create("idem")
    second = manager.create("idem", reuse=True)
    assert first.path == second.path


def test_an_existing_branch_is_refused_not_reset(manager: WorktreeManager):
    """``-B`` would move a ref the user may care about; ``-b`` fails instead."""
    manager.create("first", branch="shared")
    with pytest.raises(WorktreeError, match="already exists"):
        manager.create("second", branch="shared")


@pytest.mark.parametrize(
    "bad_name",
    ["../escape", "a/b", "a\\b", "..", "", "  ", "has space", "semi;colon"],
)
def test_a_name_that_could_escape_the_root_is_refused(bad_name: str, manager: WorktreeManager):
    """The name reaches git, so it is validated before it gets there.

    Validating afterwards would mean creating a directory in the wrong place
    first, which is not something a check can undo.
    """
    with pytest.raises(WorktreeError):
        manager.create(bad_name)


def test_a_leftover_directory_is_not_silently_deleted(manager: WorktreeManager):
    """An unregistered directory may hold work, so it is reported, not removed."""
    manager.root.mkdir(parents=True, exist_ok=True)
    leftover = manager.root / "leftover"
    leftover.mkdir()
    (leftover / "unsaved.txt").write_text("important")

    with pytest.raises(WorktreeError, match="not a registered worktree"):
        manager.create("leftover")

    assert (leftover / "unsaved.txt").exists(), "the leftover directory was destroyed"


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_worktrees_are_isolated_from_each_other(manager: WorktreeManager):
    """The whole point: two worktrees can hold different content at one path."""
    first = manager.create("alpha")
    second = manager.create("beta")

    (first.path / "README.md").write_text("# alpha\n")
    (second.path / "README.md").write_text("# beta\n")

    assert (first.path / "README.md").read_text() == "# alpha\n"
    assert (second.path / "README.md").read_text() == "# beta\n"


def test_a_worktree_shares_the_repository_history(manager: WorktreeManager):
    """A diff between worktrees is meaningful only because history is shared."""
    worktree = manager.create("shared-history")
    assert (worktree.path / ".git").exists()

    (worktree.path / "new.txt").write_text("new\n")
    git("add", "-A", cwd=worktree.path)
    git("commit", "-q", "-m", "from worktree", cwd=worktree.path)

    # The commit is visible from the main checkout.
    branches = git("branch", "--list", "arm/shared-history", cwd=manager.repo)
    assert "arm/shared-history" in branches


# ---------------------------------------------------------------------------
# Removal safety
# ---------------------------------------------------------------------------


def test_removal_refuses_when_there_are_uncommitted_changes(manager: WorktreeManager):
    """Cleanup is routine, and routine must never discard work."""
    worktree = manager.create("dirty")
    (worktree.path / "work.txt").write_text("in progress\n")

    with pytest.raises(WorktreeError, match="uncommitted change"):
        manager.remove("dirty")

    assert (worktree.path / "work.txt").exists(), "the work was destroyed"
    assert manager.find("dirty") is not None


def test_removal_refuses_on_untracked_files(manager: WorktreeManager):
    """Untracked is not the same as disposable."""
    worktree = manager.create("untracked")
    (worktree.path / "brand-new.txt").write_text("never committed\n")

    with pytest.raises(WorktreeError, match="uncommitted change"):
        manager.remove("untracked")

    assert (worktree.path / "brand-new.txt").exists()


def test_removal_with_force_discards_changes(manager: WorktreeManager):
    """The caller can override, but has to say so."""
    worktree = manager.create("forced")
    (worktree.path / "work.txt").write_text("in progress\n")

    manager.remove("forced", force=True)
    assert not worktree.path.exists()
    assert manager.find("forced") is None


def test_a_clean_worktree_is_removed(manager: WorktreeManager):
    """The ordinary path: nothing to lose, so removal succeeds."""
    worktree = manager.create("clean")
    manager.remove("clean")

    assert not worktree.path.exists()
    assert manager.find("clean") is None


def test_the_main_worktree_cannot_be_removed(manager: WorktreeManager):
    """Removing the primary checkout would delete the repository.

    ``find`` is scoped to the manager's own root, so the main checkout is not
    even discoverable through it -- a second layer of protection on top of the
    explicit refusal.
    """
    main = manager.list()[0]
    assert main.is_main
    assert manager.find(main.name) is None, (
        "find should not return the main checkout; it is not ours to remove"
    )

    with pytest.raises(WorktreeError):
        manager.remove(main.name)


def test_removing_an_unknown_worktree_is_an_error(manager: WorktreeManager):
    """An unknown name is reported, not treated as success."""
    with pytest.raises(WorktreeError, match="no worktree named"):
        manager.remove("never-existed")


def test_branch_deletion_can_be_requested(manager: WorktreeManager):
    """A caller that wants the branch gone too can ask for it."""
    manager.create("with-branch")
    manager.remove("with-branch", delete_branch=True)

    branches = git("branch", "--list", "arm/with-branch", cwd=manager.repo)
    assert branches == "", "the branch should have been deleted"


def test_the_branch_survives_by_default(manager: WorktreeManager):
    """Removing a checkout must not silently remove the ref it created."""
    manager.create("keep-branch")
    manager.remove("keep-branch")

    branches = git("branch", "--list", "arm/keep-branch", cwd=manager.repo)
    assert "arm/keep-branch" in branches


# ---------------------------------------------------------------------------
# Consistency with git's own view
# ---------------------------------------------------------------------------


def test_a_hand_deleted_worktree_is_not_listed(manager: WorktreeManager):
    """The runtime asks git rather than trusting its own bookkeeping."""
    import shutil

    worktree = manager.create("vanished")
    shutil.rmtree(worktree.path)
    manager.prune()

    assert manager.find("vanished") is None


def test_prune_reports_what_it_dropped(manager: WorktreeManager):
    """A prune that dropped nothing and one that dropped something differ."""
    import shutil

    worktree = manager.create("gone")
    shutil.rmtree(worktree.path)
    assert manager.prune() == 1
    assert manager.prune() == 0


def test_cleanup_leaves_the_main_worktree(manager: WorktreeManager):
    """Cleaning up must never touch the repository's own checkout."""
    manager.create("one")
    manager.create("two")
    skipped = manager.cleanup()

    assert skipped == []
    remaining = manager.list()
    assert len(remaining) == 1
    assert remaining[0].is_main


def test_cleanup_skips_dirty_worktrees_and_names_them(manager: WorktreeManager):
    """A cleanup that could not finish must say which worktrees remain."""
    clean = manager.create("clean-one")
    dirty = manager.create("dirty-one")
    (dirty.path / "work.txt").write_text("in progress\n")

    skipped = manager.cleanup()

    assert skipped == ["dirty-one"]
    assert not clean.path.exists(), "the clean worktree should have been removed"
    assert dirty.path.exists(), "the dirty worktree must survive"


def test_cleanup_ignores_worktrees_outside_its_root(manager: WorktreeManager, tmp_path: Path):
    """A worktree someone else created is not ours to remove."""
    outside = tmp_path / "elsewhere" / "foreign"
    outside.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "-b", "foreign-branch", str(outside), cwd=manager.repo)

    manager.cleanup()

    assert outside.exists(), "a worktree outside the manager's root was removed"
    git("worktree", "remove", "--force", str(outside), cwd=manager.repo)


# ---------------------------------------------------------------------------
# The tool surface
# ---------------------------------------------------------------------------


def test_worktree_tools_expose_the_three_operations(manager: WorktreeManager):
    """The model gets create, list and remove -- and nothing else."""
    from arm.workspace.tools import WorktreeTools

    names = [t.name for t in WorktreeTools(manager).tools()]
    assert names == ["worktree_create", "worktree_list", "worktree_remove"]


def test_create_tool_returns_the_path(manager: WorktreeManager):
    """The model needs the path to work in the new checkout."""
    import asyncio

    from arm.workspace.tools import WorktreeTools

    tools = {t.name: t for t in WorktreeTools(manager).tools()}
    result = asyncio.run(invoke(tools["worktree_create"], {"name": "via-tool"}))

    assert not result.is_error
    assert "via-tool" in result.content
    assert str(manager.root / "via-tool") in result.content


def test_create_tool_reports_a_failure_as_a_result(manager: WorktreeManager):
    """A failed create is a result, so the step continues."""
    import asyncio

    from arm.workspace.tools import WorktreeTools

    tools = {t.name: t for t in WorktreeTools(manager).tools()}
    asyncio.run(invoke(tools["worktree_create"], {"name": "dup"}))
    again = asyncio.run(invoke(tools["worktree_create"], {"name": "dup"}))

    assert again.is_error
    assert again.content.startswith("Error: ")


def test_list_tool_reports_dirtiness(manager: WorktreeManager):
    """Knowing which worktrees hold uncommitted work prevents surprises."""
    import asyncio

    from arm.workspace.tools import WorktreeTools

    tools = {t.name: t for t in WorktreeTools(manager).tools()}
    asyncio.run(invoke(tools["worktree_create"], {"name": "listed"}))

    worktree = manager.find("listed")
    assert worktree is not None
    (worktree.path / "wip.txt").write_text("in progress\n")

    listing = asyncio.run(invoke(tools["worktree_list"], {}))
    assert "listed" in listing.content
    assert "uncommitted" in listing.content


def test_remove_tool_refuses_dirty_without_force(manager: WorktreeManager):
    """The tool must not make discarding work the easy path."""
    import asyncio

    from arm.workspace.tools import WorktreeTools

    tools = {t.name: t for t in WorktreeTools(manager).tools()}
    asyncio.run(invoke(tools["worktree_create"], {"name": "held"}))
    worktree = manager.find("held")
    assert worktree is not None
    (worktree.path / "wip.txt").write_text("in progress\n")

    refused = asyncio.run(invoke(tools["worktree_remove"], {"name": "held"}))
    assert refused.is_error
    assert "uncommitted" in refused.content
    assert (worktree.path / "wip.txt").exists(), "the tool discarded work"

    forced = asyncio.run(invoke(tools["worktree_remove"], {"name": "held", "force": True}))
    assert not forced.is_error


def test_every_worktree_tool_schema_normalizes(manager: WorktreeManager):
    """The tool schemas must survive the model-facing subset."""
    import json

    from arm.workspace.tools import WorktreeTools

    for tool in WorktreeTools(manager).tools():
        rendered = json.dumps(dict(tool.definition.parameters))
        assert "title" not in rendered
        assert "$defs" not in rendered


@pytest.mark.parametrize("bad_name", [".", "..", "-", "--"])
def test_reserved_names_are_refused(bad_name: str, manager: WorktreeManager):
    """A name that does not name a *child* of the root must be refused.

    ``.`` is the important one. Joined onto the worktree root it resolves to the
    root itself, so ``git worktree add`` checks the repository out *over* the
    directory meant to hold the worktrees -- creating a checkout where the
    manager expects only its own entries, and leaving it behind when the
    subsequent bookkeeping lookup fails.

    A leading ``-`` is refused because git would read it as an option rather than
    as a path, so the command would do something other than what was asked.
    """
    with pytest.raises(WorktreeError):
        manager.create(bad_name, branch="some-branch")


def test_a_reserved_name_leaves_no_trace(manager: WorktreeManager):
    """The refusal must happen before git creates anything.

    Validation after the fact cannot undo a directory that already exists.
    """
    with pytest.raises(WorktreeError):
        manager.create(".", branch="dot")

    assert not manager.root.exists() or not (manager.root / ".git").exists(), (
        "the worktree root was turned into a checkout by a refused create"
    )


def test_removal_cannot_destroy_a_foreign_worktree(manager: WorktreeManager, tmp_path: Path):
    """A same-named worktree outside the manager's root is not ours to remove.

    ``task-1`` is an obvious name to reuse, so a foreign checkout with that
    basename is entirely plausible. Without scoping, ``remove`` would delete it --
    along with its uncommitted work and its unmerged branch -- even though this
    manager never created it.
    """
    foreign = tmp_path / "elsewhere" / "task-1"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "-b", "foreign-branch", str(foreign), cwd=manager.repo)
    (foreign / "uncommitted.txt").write_text("work in progress\n")

    with pytest.raises(WorktreeError, match="outside this manager's root"):
        manager.remove("task-1", force=True, delete_branch=True)

    assert foreign.exists(), "a foreign checkout was destroyed"
    assert (foreign / "uncommitted.txt").exists(), "uncommitted work was destroyed"
    branches = git("branch", "--list", "foreign-branch", cwd=manager.repo)
    assert "foreign-branch" in branches, "a foreign branch was deleted"

    git("worktree", "remove", "--force", str(foreign), cwd=manager.repo)


def test_cleanup_never_touches_a_foreign_worktree(manager: WorktreeManager, tmp_path: Path):
    """Cleanup is scoped by location, not by name."""
    foreign = tmp_path / "elsewhere" / "task-1"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "-b", "foreign-branch", str(foreign), cwd=manager.repo)

    manager.cleanup()

    assert foreign.exists(), "cleanup removed a worktree it did not create"
    git("worktree", "remove", "--force", str(foreign), cwd=manager.repo)


def test_find_is_scoped_but_find_any_is_not(manager: WorktreeManager, tmp_path: Path):
    """Ownership is by location, and the distinction is available to callers."""
    foreign = tmp_path / "elsewhere" / "shared-name"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "-b", "foreign-branch", str(foreign), cwd=manager.repo)

    assert manager.find("shared-name") is None, "find returned a foreign worktree"
    assert manager.find_any("shared-name") is not None, (
        "find_any should still be able to see it"
    )

    git("worktree", "remove", "--force", str(foreign), cwd=manager.repo)


# ---------------------------------------------------------------------------
# Argument injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["--lock", "-b", "--force", "-"])
def test_an_option_shaped_revision_is_refused(bad: str, manager: WorktreeManager):
    """Every value reaching git's argv needs the same check the name gets.

    ``from_revision="--lock"`` would lock the worktree it creates -- and a locked
    worktree cannot be removed by anyone through this API, so the injection
    produces a resource that leaks permanently.
    """
    with pytest.raises(WorktreeError, match="must not start with"):
        manager.create("injected", from_revision=bad)


@pytest.mark.parametrize("bad", ["--foo", "-b", "-"])
def test_an_option_shaped_branch_is_refused(bad: str, manager: WorktreeManager):
    """The same hazard, reached through a different argument."""
    with pytest.raises(WorktreeError, match="must not start with"):
        manager.create("injected", branch=bad)


def test_an_invalid_branch_name_is_refused(manager: WorktreeManager):
    """git would reject it anyway; refusing first gives a clearer message."""
    for bad in ("has space", "ends/", "double//slash", "thing.lock"):
        with pytest.raises(WorktreeError):
            manager.create(f"b-{abs(hash(bad))}", branch=bad)


def test_a_control_character_in_a_revision_is_refused(manager: WorktreeManager):
    """A newline in an argument is never intended."""
    with pytest.raises(WorktreeError):
        manager.create("ctrl", from_revision="HEAD\nrm -rf /")


# ---------------------------------------------------------------------------
# A missing directory is not a clean worktree
# ---------------------------------------------------------------------------


def test_a_hand_deleted_worktree_is_not_reported_clean(manager: WorktreeManager):
    """Reporting it as clean would advertise a checkout that does not exist.

    It is not dirty in the usual sense -- there is no work to preserve -- so the
    reason names the actual problem and points at the operation that clears it.
    """
    import shutil

    worktree = manager.create("deleted")
    shutil.rmtree(worktree.path)

    stale = manager.find_any("deleted")
    assert stale is not None
    clean, reason = manager.is_clean(stale)
    assert not clean, "a missing worktree was reported as clean"
    assert "missing" in reason
    assert "prune" in reason, "the message should name the way out"


def test_prune_clears_a_stale_entry_and_frees_the_name(manager: WorktreeManager):
    """Without prune, the name would be unusable forever.

    Pruning frees the *name*, not the branch: the branch outlives the checkout
    that was on it, which is correct -- a branch may hold commits nobody has
    merged yet. Reusing the name therefore needs a fresh branch, and that is what
    the default ``arm/<name>`` would collide with.
    """
    import shutil

    worktree = manager.create("stale")
    shutil.rmtree(worktree.path)

    assert manager.prune() == 1
    assert manager.find_any("stale") is None

    # The name is free again; a new branch is needed because the old one survives.
    again = manager.create("stale", branch="arm/stale-2")
    assert again.path.is_dir()
    assert again.branch == "arm/stale-2"

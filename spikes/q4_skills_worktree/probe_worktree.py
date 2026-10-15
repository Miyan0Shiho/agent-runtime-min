"""Throwaway worktree probes."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from arm.workspace.worktree import WorktreeError, WorktreeManager


def git(*args, cwd):
    return subprocess.run(("git", *args), cwd=cwd, capture_output=True, text=True)


def mkrepo(base: Path, name="project") -> Path:
    p = base / name
    p.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=p)
    git("config", "user.email", "t@e.com", cwd=p)
    git("config", "user.name", "T", cwd=p)
    (p / "README.md").write_text("# p\n")
    git("add", "-A", cwd=p)
    git("commit", "-q", "-m", "init", cwd=p)
    return p


def show(label, fn):
    try:
        print(f"{label}: {fn()!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: RAISED {type(exc).__name__}: {str(exc).splitlines()[0]}")


print("=" * 70)
print("W1. name='.' accepted by _validate_name?")
print("=" * 70)
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
mgr = WorktreeManager(repo, root=tmp / "wt")
show("  _validate_name('.')", lambda: mgr._validate_name("."))
show("  _validate_name('..')", lambda: mgr._validate_name(".."))
show("  _validate_name('...')", lambda: mgr._validate_name("..."))
show("  _validate_name('a..b')", lambda: mgr._validate_name("a..b"))
show("  _validate_name('-')", lambda: mgr._validate_name("-"))
show("  create('.', branch='dotbranch')", lambda: mgr.create(".", branch="dotbranch"))
print("  root exists:", mgr.root.exists(), "root:", mgr.root)
print("  git worktree list:")
print(git("worktree", "list", cwd=repo).stdout)
print("  branches:", git("branch", "--list", cwd=repo).stdout.replace("\n", " "))
print("  root is a worktree dir now?", (mgr.root / "README.md").exists())
show("  cleanup()", lambda: mgr.cleanup(force=True))
print("  after cleanup, worktrees:", [w.name for w in mgr.list()])

print()
print("=" * 70)
print("W2. foreign worktree with the same basename -> find/remove ambiguity")
print("=" * 70)
tmp2 = Path(tempfile.mkdtemp())
repo2 = mkrepo(tmp2)
mgr2 = WorktreeManager(repo2, root=tmp2 / "wt")
foreign_parent = tmp2 / "elsewhere"
foreign_parent.mkdir()
foreign = foreign_parent / "shared-name"
r = git("worktree", "add", "-b", "foreign-branch", str(foreign), cwd=repo2)
print("  foreign add rc:", r.returncode, r.stderr.strip())
mine = mgr2.create("shared-name")
print("  my worktree:", mine.path)
print("  git worktree list:")
print(git("worktree", "list", cwd=repo2).stdout)
show("  find('shared-name')", lambda: mgr2.find("shared-name").path)
show("  remove('shared-name')", lambda: mgr2.remove("shared-name").path)
print("  foreign still exists?", foreign.exists())
print("  mine still exists?", mine.path.exists())

print()
print("=" * 70)
print("W3. worktree dir gone (rm -rf) but registered")
print("=" * 70)
tmp3 = Path(tempfile.mkdtemp())
repo3 = mkrepo(tmp3)
mgr3 = WorktreeManager(repo3, root=tmp3 / "wt")
w = mgr3.create("vanished")
shutil.rmtree(w.path)
show("  is_clean(vanished)", lambda: mgr3.is_clean(w))
show("  remove('vanished')", lambda: mgr3.remove("vanished").name)
print("  list after:", [x.name for x in mgr3.list()])

print()
print("=" * 70)
print("W4. branch name collides with an existing tag")
print("=" * 70)
tmp4 = Path(tempfile.mkdtemp())
repo4 = mkrepo(tmp4)
mgr4 = WorktreeManager(repo4, root=tmp4 / "wt")
git("tag", "v1", cwd=repo4)
show("  create('t', branch='v1')", lambda: mgr4.create("t", branch="v1").path)
show("  remove('t', delete_branch=True)", lambda: mgr4.remove("t", delete_branch=True).name)
print("  branch list:", git("branch", "--list", cwd=repo4).stdout.replace("\n", " ").strip())
print("  tag list:", git("tag", "--list", cwd=repo4).stdout.replace("\n", " ").strip())

print()
print("=" * 70)
print("W5. option injection via from_revision / branch")
print("=" * 70)
tmp5 = Path(tempfile.mkdtemp())
repo5 = mkrepo(tmp5)
mgr5 = WorktreeManager(repo5, root=tmp5 / "wt")
show("  create('a', from_revision='--force')", lambda: mgr5.create("a", from_revision="--force").path)
show("  create('b', branch='--detach')", lambda: mgr5.create("b", branch="--detach").path)
print("  list:", [(x.name, x.branch, x.detached) for x in mgr5.list()])

print()
print("=" * 70)
print("W6. detached HEAD / mid-rebase is_clean")
print("=" * 70)
tmp6 = Path(tempfile.mkdtemp())
repo6 = mkrepo(tmp6)
mgr6 = WorktreeManager(repo6, root=tmp6 / "wt")
w6 = mgr6.create("det")
git("checkout", "--detach", "HEAD", cwd=w6.path)
show("  is_clean(detached)", lambda: mgr6.is_clean(mgr6.find("det")))
show("  remove('det')", lambda: mgr6.remove("det").name)
print("  removed ok?", not w6.path.exists())

print()
print("=" * 70)
print("W7. bare repository")
print("=" * 70)
tmp7 = Path(tempfile.mkdtemp())
bare = tmp7 / "bare.git"
git("init", "-q", "--bare", "-b", "main", str(bare), cwd=tmp7)
mgr7 = WorktreeManager(bare, root=tmp7 / "wt")
show("  is_repository()", lambda: mgr7.is_repository())
show("  require_repository()", lambda: mgr7.require_repository())
show("  list()", lambda: mgr7.list())

print()
print("=" * 70)
print("W8. error types escaping")
print("=" * 70)
tmp8 = Path(tempfile.mkdtemp())
repo8 = mkrepo(tmp8)
mgr8 = WorktreeManager(repo8, root=tmp8 / "wt")
# make root uncreatable
blocker = tmp8 / "blocker"
blocker.write_text("not a dir")
mgr8b = WorktreeManager(repo8, root=blocker / "sub")
show("  create with root under a file", lambda: mgr8b.create("x"))
# cwd missing
show("  is_clean on missing cwd", lambda: mgr8.is_clean(type(mgr8.list()[0])(path=tmp8 / "nope")))

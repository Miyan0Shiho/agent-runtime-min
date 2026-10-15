"""Re-verify every candidate finding against the CURRENT source.

Prints one line per finding: BUG (still present) or FIXED (already repaired by
the Lead while this review was running). Nothing here modifies src/.

Run: cd spikes && ../.venv/bin/python q4_skills_worktree/verify_all.py
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401

from arm.skills.catalog import CatalogPublisher, render_catalog
from arm.skills.registry import (
    PROJECT_RANK,
    CatalogSnapshot,
    FilesystemSkillProvider,
    SkillRegistry,
    SkillSummary,
    _split_frontmatter,
)
from arm.skills.tool import SkillTool
from arm.workspace.worktree import WorktreeError, WorktreeManager

RESULTS: list[tuple[str, str, str]] = []


def record(tag: str, bug: bool, detail: str) -> None:
    RESULTS.append((tag, "BUG  " if bug else "FIXED", detail))


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


def write_skill(root: Path, name: str, text: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


def provider(root: Path) -> FilesystemSkillProvider:
    return FilesystemSkillProvider([(PROJECT_RANK, "project", root)])


# ---------------------------------------------------------------------------
# F1 -- a YAML syntax error escapes as a non-SkillError and blanks the catalogue
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
root = tmp / "sk"
write_skill(root, "deploy", "---\nname: deploy\ndescription: deploy the service\n---\nbody\n")
write_skill(root, "review", "---\nname: review\ndescription: review a diff\n---\nbody\n")
write_skill(root, "tabbed", "---\nname: tabbed\ndescription:\n\ttab indented\n---\nbody\n")
p = provider(root)
try:
    names = [c.summary.name for c in p.list()]
    record("F1 yaml error escapes", False, f"list() returned {names}")
except Exception as exc:  # noqa: BLE001
    reg = SkillRegistry()
    reg.register_provider(p)
    snap = reg.snapshot()
    record(
        "F1 yaml error escapes",
        not isinstance(exc, Exception) or type(exc).__name__ not in ("SkillError",),
        f"list() raised {type(exc).__name__}; snapshot={[s.name for s in snap.skills]}",
    )

# ---------------------------------------------------------------------------
# F2 -- invalid UTF-8 escapes as UnicodeDecodeError (not an OSError)
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
root = tmp / "sk"
(root / "latin").mkdir(parents=True)
(root / "latin" / "SKILL.md").write_bytes(b"---\nname: latin\ndescription: caf\xe9 latte\n---\nbody\n")
p = provider(root)
try:
    got = [c.summary.name for c in p.list()]
    record("F2 bad utf-8 escapes", False, f"list() returned {got}")
except Exception as exc:  # noqa: BLE001
    record("F2 bad utf-8 escapes", True, f"raised {type(exc).__name__} (isinstance OSError={isinstance(exc, OSError)})")

# ---------------------------------------------------------------------------
# F3 -- an indented '---' inside a YAML block scalar ends the frontmatter early
# ---------------------------------------------------------------------------
text = (
    "---\nname: doc\ndescription: |\n  Deploy it.\n  ---\n  Then stop.\n"
    "disable-model-invocation: true\n---\n\nReal body.\n"
)
front, body = _split_frontmatter(text)
tmp = Path(tempfile.mkdtemp())
root = tmp / "sk"
write_skill(root, "doc", text)
cands = provider(root).list()
if cands:
    c = cands[0]
    leaked = "disable-model-invocation" in (provider(root).load(c).content or "")
    record(
        "F3 block scalar fence",
        c.summary.model_invocable or leaked,
        f"model_invocable={c.summary.model_invocable} desc={c.summary.description!r} leaked={leaked}",
    )
else:
    record("F3 block scalar fence", True, "skill was rejected outright")

# ---------------------------------------------------------------------------
# F4 -- the digest separator is forgeable: one skill can collide with two
# ---------------------------------------------------------------------------
reg = SkillRegistry()
A = CatalogSnapshot([SkillSummary(name="alpha", description="line1\nbeta\x1fdesc")])
B = CatalogSnapshot(
    [SkillSummary(name="alpha", description="line1"), SkillSummary(name="beta", description="desc")]
)
same = reg.catalog_digest(A) == reg.catalog_digest(B)
rendered_differs = render_catalog(list(A.skills), first=True) != render_catalog(list(B.skills), first=True)
record("F4 digest collision", same and rendered_differs, f"same_digest={same} renders_differ={rendered_differs}")

# ---------------------------------------------------------------------------
# F5 -- flipping a skill to non-model-invocable does not change the digest
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
root = tmp / "sk"
write_skill(root, "flip", "---\nname: flip\ndescription: a flippable skill\n---\nbody\n")
reg = SkillRegistry()
reg.register_provider(provider(root))
s1 = reg.snapshot()
d1 = reg.catalog_digest(s1)
write_skill(root, "flip", "---\nname: flip\ndescription: a flippable skill\ndisable-model-invocation: true\n---\nbody\n")
s2 = reg.snapshot()
d2 = reg.catalog_digest(s2)
v1 = [s.name for s in reg.model_visible(s1)]
v2 = [s.name for s in reg.model_visible(s2)]
record("F5 policy flip silent", d1 == d2 and v1 != v2, f"digest {d1}=={d2}; visible {v1} -> {v2}")

# ---------------------------------------------------------------------------
# F6 -- the '-order' tie-break term is inert: first provider wins, not last
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
A_dir = tmp / "A"
B_dir = tmp / "B"
write_skill(A_dir, "shared", "---\nname: shared\ndescription: from provider A\n---\nbody\n")
write_skill(B_dir, "shared", "---\nname: shared\ndescription: from provider B\n---\nbody\n")
reg = SkillRegistry()
reg.register_provider(FilesystemSkillProvider([(PROJECT_RANK, "project", A_dir)], name="A"))
reg.register_provider(FilesystemSkillProvider([(PROJECT_RANK, "project", B_dir)], name="B"))
snap = reg.snapshot()
record(
    "F6 tie-break inert",
    snap.skills[0].description == "from provider A",
    f"registered A,B; catalogue says {snap.skills[0].description!r} (docstring: later wins)",
)

# ---------------------------------------------------------------------------
# F7 -- load() returns the candidate's stale summary, so tool.py's second
#       policy check is a self-comparison
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
root = tmp / "sk"
write_skill(root, "alpha", "---\nname: alpha\ndescription: the alpha skill\n---\nALPHA BODY\n")
p = provider(root)
cand = p.list()[0]
write_skill(root, "alpha", "---\nname: alpha\ndescription: the alpha skill\ndisable-model-invocation: true\n---\nBETA BODY\n")
got = p.load(cand)
record(
    "F7 stale summary",
    got is not None and got.summary.model_invocable,
    f"file now says disable-model-invocation:true; load().summary.model_invocable={got.summary.model_invocable if got else None}",
)

# ---------------------------------------------------------------------------
# F8 -- worktree find() matches by basename across the root boundary, so
#       remove() can destroy a foreign checkout
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
mgr = WorktreeManager(repo, root=tmp / "wt")
foreign = tmp / "elsewhere" / "dup"
foreign.parent.mkdir()
git("worktree", "add", "-b", "their-branch", str(foreign), cwd=repo)
(foreign / "their-work.txt").write_text("work\n")
git("add", "-A", cwd=foreign)
git("commit", "-q", "-m", "wip", cwd=foreign)
try:
    mgr.remove("dup", force=True)
    record("F8 foreign removal", not foreign.exists(), f"foreign checkout exists={foreign.exists()}")
except WorktreeError as exc:
    record("F8 foreign removal", False, f"refused: {str(exc)[:70]}")

# ---------------------------------------------------------------------------
# F9 -- git's directory-conflict message is misclassified as a branch conflict
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
mgr = WorktreeManager(repo, root=tmp / "wt")
mgr.root.mkdir(parents=True, exist_ok=True)
left = mgr.root / "stuck"
left.mkdir()
(left / "unsaved.txt").write_text("important\n")
try:
    mgr.create("stuck", branch="totally-fresh-branch")
    record("F9 misclassified error", False, "create succeeded unexpectedly")
except WorktreeError as exc:
    record(
        "F9 misclassified error",
        "branch" in str(exc) and "already exists" in str(exc),
        f"message={str(exc)[:95]!r}",
    )

# ---------------------------------------------------------------------------
# F10 -- reuse=True over a non-empty leftover creates an orphan branch, then
#        refuses; the directory can never be adopted
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
mgr = WorktreeManager(repo, root=tmp / "wt")
mgr.root.mkdir(parents=True, exist_ok=True)
left = mgr.root / "stuck"
left.mkdir()
(left / "unsaved.txt").write_text("important\n")
before = git("branch", "--list", cwd=repo).stdout
try:
    mgr.create("stuck", reuse=True)
    record("F10 reuse stuck", False, "create succeeded")
except WorktreeError as exc:
    after = git("branch", "--list", cwd=repo).stdout
    orphan = after != before
    record("F10 reuse stuck", orphan, f"orphan_branch={orphan} msg={str(exc)[:80]!r}")

# ---------------------------------------------------------------------------
# F11 -- a missing cwd is reported as "git is not installed"
# ---------------------------------------------------------------------------
missing = Path(tempfile.mkdtemp()) / "gone"
mgr = WorktreeManager(missing, root=missing.parent / "wt")
try:
    mgr.is_repository()
    record("F11 missing cwd", False, "returned")
except WorktreeError as exc:
    record("F11 missing cwd", "not installed" in str(exc), f"message={str(exc)!r}")

# ---------------------------------------------------------------------------
# F12 -- root.mkdir lets an OSError escape instead of a WorktreeError
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
blocker = tmp / "blocker"
blocker.write_text("a file, not a directory")
mgr = WorktreeManager(repo, root=blocker / "sub")
try:
    mgr.create("x")
    record("F12 OSError escapes", False, "create succeeded")
except WorktreeError as exc:
    record("F12 OSError escapes", False, f"WorktreeError: {str(exc)[:60]}")
except OSError as exc:
    record("F12 OSError escapes", True, f"{type(exc).__name__} escaped (not a WorktreeError)")

# ---------------------------------------------------------------------------
# F13 -- from_revision accepts a git option (argument injection)
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
mgr = WorktreeManager(repo, root=tmp / "wt")
try:
    w = mgr.create("lk", from_revision="--lock")
    locked = "locked" in git("worktree", "list", "--porcelain", cwd=repo).stdout
    removable = True
    try:
        mgr.remove("lk", force=True)
    except WorktreeError:
        removable = False
    record("F13 option injection", locked and not removable, f"locked={locked} removable={removable}")
except WorktreeError as exc:
    record("F13 option injection", False, f"refused: {str(exc)[:60]}")

# ---------------------------------------------------------------------------
# F14 -- a phantom worktree (rm -rf) is listed as 'clean' and blocks create
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
mgr = WorktreeManager(repo, root=tmp / "wt")
w = mgr.create("ghost")
shutil.rmtree(w.path)
ghost = mgr.find("ghost")
clean, reason = mgr.is_clean(ghost)
blocked = False
try:
    mgr.create("ghost")
except WorktreeError:
    blocked = True
record("F14 phantom worktree", clean and blocked, f"is_clean={clean} ({reason!r}) create_blocked={blocked}")

# ---------------------------------------------------------------------------
# F15 -- is_clean says 'clean' for a directory that does not exist
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
mgr = WorktreeManager(repo, root=tmp / "wt")
w = mgr.create("gone")
shutil.rmtree(w.path)
clean, reason = mgr.is_clean(w)
record("F15 phantom reported clean", clean, f"is_clean={clean} reason={reason!r}")

# ---------------------------------------------------------------------------
# F16 -- --no-checkout produces an empty worktree reported as a real checkout
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())
repo = mkrepo(tmp)
mgr = WorktreeManager(repo, root=tmp / "wt")
try:
    w = mgr.create("noco", from_revision="--no-checkout")
    empty = not (w.path / "README.md").exists()
    record("F16 --no-checkout", empty, f"created={w.path.exists()} README_present={(w.path/'README.md').exists()}")
except WorktreeError as exc:
    record("F16 --no-checkout", False, f"refused: {str(exc)[:60]}")

print(f"{'finding':28} {'state':6} detail")
print("-" * 110)
for tag, state, detail in RESULTS:
    print(f"{tag:28} {state:6} {detail}")
print("-" * 110)
print(f"still broken: {sum(1 for _, s, _ in RESULTS if s.strip() == 'BUG')} / {len(RESULTS)}")

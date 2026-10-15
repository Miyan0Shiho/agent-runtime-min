"""FAILING tests: dead code and integration defects in the skills module.

These are the "the module passed its own tests because nothing ran it" class of
defect. Each is ``xfail(strict=True)``: it fails while the defect is present and
turns into ``XPASS(strict)`` = failure once the fix lands.

Nothing here modifies ``src/``.
"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
from pathlib import Path

import pytest

import _bootstrap  # noqa: F401
from arm.agent.loop import Runtime
from arm.llm.mock import MockAdapter, text_response
from arm.prompt.assembler import PromptAssembler
from arm.session.session import Session
from arm.skills.catalog import NO_RELOAD_NOTE
from arm.skills.registry import (
    PROJECT_RANK,
    FilesystemSkillProvider,
    SkillRegistry,
)
from arm.skills.tool import SkillTool
from arm.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def write_skill(root: Path, name: str, *, description="a test skill", extra="", body="Do it.") -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def src_references(symbol: str) -> int:
    """How many times ``symbol`` appears anywhere under ``src/``."""
    result = subprocess.run(
        ("grep", "-rn", "--include=*.py", symbol, str(SRC)),
        capture_output=True,
        text=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def symbol_is_wired_or_removed(symbol: str) -> bool:
    """True when a dead public symbol has been either wired up or deleted.

    Both are acceptable fixes for dead code, and the ratchet must accept both: a
    symbol that no longer exists cannot be dead. Deleting is checked first
    because that is the smaller fix when nothing needs the API.
    """
    if src_references(symbol) == 0:
        return True  # removed -- no longer dead
    return src_references(symbol) >= 2  # declared AND used


def build_runtime(root: Path, script) -> tuple[Runtime, Session, SkillRegistry]:
    session = Session.create("dead-code")
    skills = SkillRegistry()
    skills.register_provider(FilesystemSkillProvider([(PROJECT_RANK, "project", root)]))
    registry = ToolRegistry()
    registry.register(SkillTool(skills).register())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    runtime = Runtime(
        session,
        MockAdapter(script),
        registry,
        assembler=assembler,
        skills=skills,
        model="mock",
    )
    return runtime, session, skills


# ---------------------------------------------------------------------------
# BUG D1 -- the `/name` user gesture is dead code, so `disable-model-invocation`
#           makes a skill unreachable by ANY route
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG D1: `SkillRegistry.find_gesture` has no caller in src/, so the only documented route to a non-model-invocable skill does not exist",
)
def test_the_user_gesture_can_load_a_non_model_invocable_skill(tmp_path):
    """``InvocationPolicy`` documents four combinations and singles out
    ``{model: False, user: True}`` as "only reachable by explicit ``/name``,
    which is how a skill that should not be auto-selected is expressed".

    That route is not wired. ``find_gesture`` is called from nowhere under
    ``src/`` -- only from ``tests/test_skills.py``. ``loop.py`` never calls it,
    and ``cli/main.py`` passes the raw line straight to ``runtime.submit`` with
    no gesture handling.

    So a skill with ``disable-model-invocation: true`` is unreachable by the
    model *and* by the user: the flag's documented purpose ("the skill exists,
    the model cannot select it, and the user can still ask for it") is not
    implemented. The user sees no error, because nothing looks for the gesture.

    Observed::

        user types: "/secret please"
        SECRET INSTRUCTIONS reached the model? False
        transcript: '/secret please\\nTools now available: skill\\nok'
        find_gesture('/secret please') -> ['secret']      <-- works, but is never called

    Fix: call it on each user input, before the turn starts, and inject the
    matched skills' bodies. Minimal wiring in ``loop._run_step`` (or wherever
    pending inputs are claimed)::

        for pending_input in claimed:
            for summary in self.skills.find_gesture(pending_input.text, snapshot):
                definition = self.skills.get(summary.name)
                if definition is not None:
                    self.session.append_message(
                        "user", [{"type": "text", "text": render_skill_content(definition)}],
                        source="skill-gesture", context_only=True,
                    )

    and pass the invoked names to ``render_catalog(..., user_invoked=...)`` so the
    no-reload note fires.
    """
    root = tmp_path / "skills"
    write_skill(
        root,
        "secret",
        description="only the user may invoke this",
        extra="disable-model-invocation: true\n",
        body="SECRET INSTRUCTIONS",
    )

    runtime, session, skills = build_runtime(root, [text_response("ok")])

    # Precondition: the gesture IS recognised by the registry...
    snapshot = skills.snapshot()
    assert [s.name for s in skills.find_gesture("/secret please", snapshot)] == ["secret"]

    # ...but the runtime must act on it.
    runtime.submit("/secret please")
    asyncio.run(runtime.run_until_idle())

    combined = "\n".join(m.text for m in session.derive() if m.role != "system")
    assert "SECRET INSTRUCTIONS" in combined, (
        "the user invoked /secret but the skill body never reached the model"
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG D1b: `find_gesture` and `SkillTool` are the two halves of one contract; the gesture half is unreachable from src/",
)
def test_find_gesture_has_a_caller_in_src(tmp_path):
    """A cheap structural guard for BUG D1.

    ``find_gesture`` is a public method on the registry whose only purpose is to
    be called on user input. Its absence from ``src/`` is the whole defect.

    Fix: wire it as described in BUG D1, after which this count rises.
    """
    assert src_references("find_gesture") >= 2, (
        "find_gesture appears only at its definition; nothing in src/ calls it"
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG D1c: the no-reload note is unreachable because `_publish_catalog` never passes `user_invoked`",
)
def test_the_no_reload_note_is_reachable(tmp_path):
    """``render_catalog(user_invoked=...)`` appends ``NO_RELOAD_NOTE`` so the
    model does not reload a skill the user already invoked directly. Nothing in
    ``src/`` ever passes ``user_invoked``: ``loop._publish_catalog`` calls
    ``self._catalog.update(snapshot, visible, digest)``, and ``CatalogPublisher``
    has no parameter for it.

    ``NO_RELOAD_NOTE`` is therefore a dead constant, and the behaviour it
    documents cannot happen -- because BUG D1 means the gesture cannot happen
    either.

    Fix: with the gesture wired (BUG D1), record the invoked names and pass them
    through ``CatalogPublisher.update`` to ``render_catalog``.
    """
    source = (SRC / "arm" / "agent" / "loop.py").read_text(encoding="utf-8")
    assert "user_invoked" in source, (
        "loop.py never passes user_invoked, so NO_RELOAD_NOTE can never be emitted"
    )


# ---------------------------------------------------------------------------
# BUG D2 -- `register_skill` is dead code: a documented public API with no caller
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG D2: `SkillRegistry.register_skill` has no caller in src/, so the documented runtime-skill extension point is never exercised outside its own test",
)
def test_register_skill_is_reachable_from_src(tmp_path):
    """``register_skill`` documents itself as the way a host installs a skill
    defined in code rather than on disk, and states that such a skill "outranks
    every provider". ``SkillTool`` is registered by ``cli/main.py``, but
    ``register_skill`` is never called anywhere in ``src/``.

    This is the same class of defect as the prior review's "the entire module was
    dead code": a public extension point that only its own test exercises, so its
    interaction with the live path (``get``, ``snapshot``, the tool) is unverified
    outside a unit test.

    Fix: either wire it (e.g. load bundled/in-code skills in
    ``cli.build_runtime``) or remove it. If it is intended for embedders, say so
    in the module docstring and keep the test that covers it.
    """
    assert symbol_is_wired_or_removed("register_skill"), (
        "register_skill appears only at its definition; nothing in src/ calls it"
    )


# ---------------------------------------------------------------------------
# BUG D3 -- worktree `prune()` and `cleanup()` are dead code
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG D3: `WorktreeManager.prune` and `cleanup` have no caller in src/ and are not exposed as tools, so the recovery paths they document are unreachable",
)
def test_prune_and_cleanup_are_reachable_from_src(tmp_path):
    """``prune`` documents the exact failure it exists to repair:

        "A worktree deleted with ``rm -rf`` leaves git believing it still exists,
         which makes a later create with the same name fail for no visible
         reason."

    ``cleanup`` documents the bulk teardown. Neither is called anywhere in
    ``src/``: ``WorktreeTools.tools()`` returns only create/list/remove, and
    ``cli/main.py`` registers exactly those three. Both are reachable only from
    ``tests/test_worktree.py``.

    The consequence is concrete and reaches the model: a hand-deleted worktree
    makes its own name permanently unusable (``create`` refuses, and the model has
    no prune tool), and nothing ever reclaims the worktrees a session created.

    Fix: expose them, or call ``prune`` from ``create`` when the name resolves to
    a worktree whose directory is gone (see BUG W10), and call ``cleanup`` on
    session teardown.
    """
    worktree_source = (SRC / "arm" / "workspace" / "worktree.py").read_text(encoding="utf-8")
    tools_source = (SRC / "arm" / "workspace" / "tools.py").read_text(encoding="utf-8")
    removed = "def prune" not in worktree_source and "def cleanup" not in worktree_source
    exposed = "prune" in tools_source or "cleanup" in tools_source
    called_from_create = "self.prune()" in worktree_source
    assert removed or exposed or called_from_create, (
        "neither prune nor cleanup is exposed through the worktree tool surface, "
        "called from create(), nor removed"
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG D3b: `WorktreeManager.default_branch` has no caller anywhere in src/",
)
def test_default_branch_is_reachable_from_src(tmp_path):
    """``default_branch`` computes the repository's current branch, documented as
    the fallback when HEAD is detached. Nothing in ``src/`` calls it -- not the
    tools, not the CLI.

    Fix: use it where a branch is needed (e.g. a default for ``worktree_create``),
    or remove it.
    """
    assert symbol_is_wired_or_removed("default_branch"), (
        "default_branch appears only at its definition; nothing in src/ calls it"
    )


# ---------------------------------------------------------------------------
# BUG D4 -- `WorktreePathParams` is an unused model
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG D4: `WorktreePathParams` is defined but never used; `worktree_list` declares a bare object schema instead of reusing it",
)
def test_worktree_list_uses_the_declared_parameter_model(tmp_path):
    """``WorktreePathParams`` is declared with a ``name`` field and never
    referenced. ``worktree_list`` takes no parameters, so the natural reading is
    that it was meant for a ``worktree_path``-style tool that was never built --
    or that ``worktree_list`` was meant to accept an optional name filter.

    Either way the class is dead: an unused pydantic model that looks like part
    of the tool surface. It also means ``worktree_list`` ships a hand-written
    ``{"type": "object", "properties": {}}`` schema while the other two tools use
    ``model_json_schema()``, so the file has two conventions.

    Fix: delete ``WorktreePathParams``, or use it.
    """
    tools_source = (SRC / "arm" / "workspace" / "tools.py").read_text(encoding="utf-8")
    declarations = tools_source.count("WorktreePathParams")
    assert declarations == 0 or declarations >= 2, (
        "WorktreePathParams is declared but never used "
        f"({declarations} occurrence(s) in tools.py); delete it or use it"
    )


# ---------------------------------------------------------------------------
# BUG D5 -- a republished catalogue is appended, never a replacement
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG D5: after a resumed session the stale catalogue is still on the surface, so the replacement preamble is wrong -- but a resumed runtime always renders a FIRST catalogue",
)
def test_a_resumed_session_marks_the_new_catalogue_as_a_replacement(tmp_path):
    """``CatalogPublisher`` starts with ``published_digest = None`` on every fresh
    ``Runtime``, including one built over a *resumed* session. So the first
    catalogue of the new process is rendered with ``first=True`` --
    "The following skills are available in this session..." -- even though the
    transcript already contains an older catalogue from the previous process.

    The model then sees two catalogues, and the newer one does not say it
    supersedes the older one, which is exactly what ``REPLACEMENT_PREAMBLE``
    exists to prevent ("It replaces every previous available-skills list; use only
    the names in this catalogue").

    Observed::

        catalogues in log: 2
         [0] seq=8  'first'=True  'replaces'=False
         [1] seq=22 'first'=True  'replaces'=False
        derive() contains both? 2

    Fix: seed the publisher from the log on construction, the way
    ``_restore_announcement_state`` already does for the prompt and the tool
    announcement -- if a catalogue is on the surface, set ``published_digest`` to
    its recorded digest (or simply set ``has_published``) so the next publish is a
    replacement::

        for event in self.session:
            if event.data.get("source") == "skill-catalog":
                self._catalog.published_digest = <digest recorded with the event>
    """
    root = tmp_path / "skills"
    write_skill(root, "alpha", description="the first skill")

    session = Session.create("resume")
    runtime, _, skills = build_runtime(root, [text_response("ok")])
    runtime.submit("one")
    asyncio.run(runtime.run_until_idle())

    # A new skill appears, and the process restarts over the same log.
    write_skill(root, "beta", description="the second skill")
    resumed = Session.from_events("resume", session.events())
    runtime2, _, _ = build_runtime(root, [text_response("ok")])
    runtime2.session = resumed
    runtime2.submit("two")
    asyncio.run(runtime2.run_until_idle())

    catalogues = [
        e
        for e in resumed
        if e.type == "user/message" and e.data.get("source") == "skill-catalog"
    ]
    assert len(catalogues) == 2, f"expected two catalogues, got {len(catalogues)}"

    latest = catalogues[-1].data["message"]["content"][0]["text"]
    assert "replaces" in latest.lower(), (
        "the second catalogue does not say it supersedes the first, so the model "
        f"sees two conflicting lists: {latest[:120]!r}"
    )

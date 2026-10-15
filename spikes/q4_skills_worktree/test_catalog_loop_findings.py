"""REGRESSION GUARDS: catalogue/loop defects that were found and then fixed.

HISTORY. These three were failing findings during this review. The Lead fixed
them while the review was running (``context_only`` on runtime-generated
messages, an inbox skip, and gating the publisher reset on the catalogue
actually having left the surface). They are ordinary passing tests here, so a
re-introduction is a hard failure rather than a silent xfail.

Nothing here modifies ``src/``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import _bootstrap  # noqa: F401
from arm.agent.loop import Runtime
from arm.context.manager import ContextManager
from arm.context.meter import Thresholds
from arm.llm.mock import MockAdapter, text_response
from arm.prompt.assembler import PromptAssembler
from arm.session.session import Session
from arm.skills.registry import PROJECT_RANK, FilesystemSkillProvider, SkillRegistry
from arm.skills.tool import SkillTool
from arm.tools.registry import ToolRegistry


def write_skill(root: Path, name: str, description: str, extra: str = "") -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\nBody for {name}.\n",
        encoding="utf-8",
    )


def build(
    root: Path,
    *,
    context: ContextManager | None = None,
    script=None,
) -> tuple[Runtime, Session, SkillRegistry]:
    session = Session.create("catalog-integration")
    skills = SkillRegistry()
    skills.register_provider(FilesystemSkillProvider([(PROJECT_RANK, "project", root)]))
    registry = ToolRegistry()
    registry.register(SkillTool(skills).register())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    runtime = Runtime(
        session,
        MockAdapter(script or [text_response("ok")] * 400),
        registry,
        assembler=assembler,
        context=context,
        skills=skills,
        model="mock",
    )
    return runtime, session, skills


def on_surface_catalogues(session: Session) -> list:
    surface = set(session.surface_nodes())
    return [
        e
        for e in session
        if e.type == "user/message"
        and e.data.get("source") == "skill-catalog"
        and e.seq in surface
    ]


# ---------------------------------------------------------------------------
# BUG L1 -- the catalogue is published as a plain user message, so the inbox
#           treats runtime-generated text as pending user input
# ---------------------------------------------------------------------------


def test_publishing_the_catalogue_does_not_cost_an_extra_model_call(tmp_path):
    """``loop._publish_catalog`` appends the catalogue as a ``user/message``.
    ``Inbox.pending()`` selects **every** unclaimed ``user/message``, so the
    catalogue is claimed as new user input and forces another step.

    A trivial one-step task therefore costs two model calls whenever any skill
    exists, and the second call's only new input is the catalogue itself.

    Observed::

        skills=False  turns=1 steps=1 model_calls=1 catalogues=0
        skills=True   turns=1 steps=2 model_calls=2 catalogues=1
        step/start inputs=(8,)      <-- seq 8 is the catalogue

    Fix: mark runtime-generated messages as context rather than input, and have
    the inbox skip them -- e.g. ``"context_only": True`` on the append (as
    ``plan-mode`` already does) plus::

        if event.data.get("context_only"):
            continue

    in ``Inbox.pending``.

    FIXED during this review: ``context_only: True`` on the append in
    ``loop._publish_catalog``, plus a skip in ``Inbox.pending``.
    """
    root = tmp_path / "skills"
    write_skill(root, "deploy", "deploy the service")

    runtime, session, _ = build(root)
    runtime.submit("hello")
    asyncio.run(runtime.run_until_idle())

    steps = sum(1 for e in session if e.type == "step/start")
    assert steps == 1, (
        f"publishing the catalogue forced {steps - 1} extra step(s); the catalogue "
        "is being treated as user input"
    )
    assert len(runtime.adapter.requests) == 1


# ---------------------------------------------------------------------------
# BUG L2 -- an over-threshold session republishes the catalogue on every step
# ---------------------------------------------------------------------------


def test_a_session_over_threshold_does_not_republish_forever(tmp_path):
    """The regression introduced (and since fixed) while this review was running.

    Resetting ``CatalogPublisher`` whenever ``report.compacted`` is truthy makes
    the publisher forget on every step of a session that stays over threshold.
    Each step then republishes the catalogue, and each republished catalogue is
    pending user input (BUG L1), so the loop never idles.

    Observed with ``context_window=400, headroom=0, threshold_ratio=0.5`` and
    three skills::

        reset ACTIVE:   steps=4000  catalogues=4000  compactions=4000  events=44205
                        -> LoopError: exceeded 100 turns without going idle
        reset DISABLED: steps=3     catalogues=1     compactions=1     events=28

    Fix: ask whether the catalogue is *still on the surface* rather than whether
    compaction ran, and skip the reset when it is (the Lead's current approach,
    ``loop._catalogue_on_surface``), combined with BUG L1's ``context_only`` so a
    republished catalogue cannot itself become input.

    FIXED during this review.
    """
    root = tmp_path / "skills"
    for index in range(3):
        write_skill(root, f"skill-{index}", f"description number {index}")

    async def summarize(_messages) -> str:
        return "condensed"

    session = Session.create("over-threshold")
    context = ContextManager(
        session,
        thresholds=Thresholds(context_window=400, headroom=0, threshold_ratio=0.5),
        summarize=summarize,
    )
    runtime, session, _ = build(root, context=context, script=[text_response("ok")] * 4000)
    runtime.max_steps = 12

    runtime.submit("please do the task " + "x" * 400)
    asyncio.run(runtime.run_until_idle())

    steps = sum(1 for e in session if e.type == "step/start")
    assert steps <= 4, (
        f"the session ran {steps} steps for one task; the catalogue is being "
        "republished on every step"
    )
    assert len(on_surface_catalogues(session)) >= 1, "the model must still see a catalogue"


# ---------------------------------------------------------------------------
# BUG L3 -- a catalogue that survives compaction is not republished needlessly
# ---------------------------------------------------------------------------


def test_compaction_that_spares_the_catalogue_does_not_republish_it(tmp_path):
    """Compaction may leave the catalogue on the surface. Resetting the publisher
    on *any* compaction then resends an identical catalogue, which is pure
    context cost -- exactly what the digest exists to prevent.

    FIXED during this review: ``loop._catalogue_on_surface`` gates the reset.
    """
    root = tmp_path / "skills"
    write_skill(root, "stable", "a stable skill")

    async def summarize(_messages) -> str:
        return "condensed"

    session = Session.create("spare-catalogue")
    context = ContextManager(
        session,
        thresholds=Thresholds(context_window=1500, headroom=0, threshold_ratio=0.5),
        summarize=summarize,
    )
    runtime, session, _ = build(root, context=context, script=[text_response("ok")] * 200)

    # Build history so compaction runs, then keep publishing decisions stable.
    for index in range(6):
        session.append_message(
            "assistant", [{"type": "text", "text": f"bulk {index} " + "y" * 400}]
        )
        runtime.submit(f"question {index} " + "x" * 400)
    asyncio.run(runtime.run_until_idle())

    catalogues = [e for e in session if e.data.get("source") == "skill-catalog"]
    assert len(catalogues) <= 2, (
        f"the catalogue was published {len(catalogues)} times; an unchanged "
        "catalogue is being resent"
    )

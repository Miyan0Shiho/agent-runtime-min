"""Does compaction hide the skill catalogue without republishing it?

The catalogue is published as a ``user/message`` with ``source == "skill-catalog"``.
Compaction replaces a span of the surface with a summary. If the catalogue node
falls inside the compacted span, the model can no longer see any catalogue -- and
``CatalogPublisher`` still believes it published one, so the digest check
suppresses the replacement forever.

Run: cd spikes && ../.venv/bin/python q4_skills_worktree/probe_catalog_compaction.py
"""

from __future__ import annotations

import asyncio

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

import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
skills_root = tmp / "skills"
for i in range(3):
    d = skills_root / f"skill-{i}"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: skill-{i}\ndescription: description number {i}\n---\nBody {i}.\n"
    )

session = Session.create("probe")
skills = SkillRegistry()
skills.register_provider(FilesystemSkillProvider([(PROJECT_RANK, "project", skills_root)]))

registry = ToolRegistry()
registry.register(SkillTool(skills).register())

assembler = PromptAssembler()
assembler.set("core", "you are arm", priority=10)

adapter = MockAdapter([text_response("ok")] * 12)

# A tiny window so the catalogue + chatter is enough to trigger compaction.
async def summarize(_messages):
    return "condensed summary of earlier turns"


context = ContextManager(
    session,
    thresholds=Thresholds(context_window=400, headroom=0, threshold_ratio=0.5),
    summarize=summarize,
)

runtime = Runtime(
    session,
    adapter,
    registry,
    assembler=assembler,
    context=context,
    skills=skills,
    model="mock",
)


def catalogues():
    return [
        e
        for e in session
        if e.type == "user/message" and e.data.get("source") == "skill-catalog"
    ]


def on_surface_catalogues():
    surface = set(session.surface.nodes)
    return [e for e in catalogues() if e.seq in surface]


for turn in range(6):
    runtime.submit(f"please do task {turn} " + "x" * 400)
    asyncio.run(runtime.run_until_idle())
    cats = catalogues()
    visible = on_surface_catalogues()
    print(
        f"turn {turn}: catalogues_in_log={len(cats)} on_surface={len(visible)} "
        f"surface_nodes={len(session.surface.nodes)} "
        f"published_digest={runtime._catalog.published_digest}"
    )

print()
print("final derived transcript:")
for m in session.derive():
    text = " ".join(b.text for b in m.content if b.type == "text")
    print(f"  {m.role:9} {text[:90]!r}")

print()
print("compaction events:", [e.type for e in session if e.type.startswith("compaction")])
print("on-surface catalogue count:", len(on_surface_catalogues()))

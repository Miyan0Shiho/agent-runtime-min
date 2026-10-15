"""Probe: is the child registry the parent registry? plus real nested delegation."""

from __future__ import annotations

import asyncio
import sys

import _bootstrap  # noqa: F401

from arm.agent.loop import Runtime
from arm.llm.mock import MockAdapter, text_response
from arm.prompt.assembler import PromptAssembler
from arm.session.session import Session
from arm.subagent.contracts import DelegationRequest
from arm.subagent.inprocess import ChildFactory, InProcessProvider
from arm.subagent.runtime import SubagentRuntime
from arm.subagent.tool import SubagentTool
from arm.tools.registry import ToolRegistry

print("=== identity check ===")
registry = ToolRegistry()
runtime = SubagentRuntime(registry=registry)
provider = InProcessProvider(
    factory=ChildFactory(build=lambda *a: None), parent_registry=registry
)
request = DelegationRequest(prompt="hi")
got = provider._child_registry(request)
print("parent id:", id(registry), "child id:", id(got), "same:", got is registry)
print("parent depth:", registry.depth, "child depth:", got.depth)

print()
print("=== real nested delegation: can a child spawn a grandchild? ===")

registry2 = ToolRegistry()
runtime2 = SubagentRuntime(registry=registry2, max_depth=3)
sessions: list[str] = []


def build2(child_session, child_registry, request):
    sessions.append(child_session.session_id)
    assembler = PromptAssembler()
    assembler.set("core", "sub", priority=10)
    return Runtime(
        child_session,
        MockAdapter([text_response("child answer")]),
        child_registry,
        assembler=assembler,
        model="mock",
    )


runtime2.register_provider(
    InProcessProvider(factory=ChildFactory(build=build2), parent_registry=registry2)
)
parent = Session.create("parent")
registry2.register(SubagentTool(runtime2, parent_session=parent, depth=0).register())


async def nested():
    """Call the child's own subagent tool the way the child's model would."""
    running = await runtime2.start(DelegationRequest(prompt="level 1"), parent=parent)
    child_tool = running.runtime.registry.get("subagent")
    tool_obj = child_tool.body.__self__
    print("child's subagent tool depth   :", tool_obj.depth)
    print("child's tool parent session   :", tool_obj.parent_session.session_id)
    # The child delegates: what depth does the runtime see?
    result = await tool_obj.run({"prompt": "level 2"}, None)
    await running.dispose()
    return result


result = asyncio.run(nested())
print("grandchild result content     :", result.content)
print("sessions built                :", sessions)
print("child sessions                :", [s for s in sessions])

"""Probe: does the subagent `tools` filter restrict the child, and leak into the parent?

Run:  cd spikes && ../.venv/bin/python q5_subagent/probe_registry.py
"""

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
from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult
from arm.tools.registry import ToolRegistry


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in ("read", "write", "bash"):
        async def body(_args, _exec, _name=name):
            return ToolResult.ok(f"{_name} ran")

        registry.register(
            RegisteredTool(
                definition=ToolDefinition(
                    name=name, description=name, parameters={"type": "object", "properties": {}}
                ),
                body=body,
            )
        )
    return registry


def main() -> int:
    registry = make_registry()
    print("parent tools before:", registry.names())
    print("registry depth before:", registry.depth)

    def build(child_session, child_registry, request):
        assembler = PromptAssembler()
        assembler.set("core", "sub", priority=10)
        return Runtime(
            child_session,
            MockAdapter([text_response("done")]),
            child_registry,
            assembler=assembler,
            model="mock",
        )

    runtime = SubagentRuntime(registry=registry)
    runtime.register_provider(
        InProcessProvider(factory=ChildFactory(build=build), parent_registry=registry)
    )

    async def scenario():
        parent = Session.create("parent")
        running = await runtime.start(
            DelegationRequest(prompt="do it", tools=("read",)), parent=parent
        )
        child_view = running.runtime.registry.names()
        await running.result
        await running.dispose()
        return child_view

    child_view = asyncio.run(scenario())

    print("child registry names:", child_view)
    print("child registry is parent registry:", None)
    print("parent tools AFTER  :", registry.names())
    print("registry depth after:", registry.depth)

    # A second delegation with a different filter.
    async def second():
        parent = Session.create("parent2")
        try:
            running = await runtime.start(
                DelegationRequest(prompt="do it again", tools=("write",)), parent=parent
            )
        except BaseException as exc:
            return f"{type(exc).__name__}: {exc}"
        await running.result
        await running.dispose()
        return "started"

    print("second delegation (tools=('write',)):", asyncio.run(second()))
    print("parent tools AFTER 2:", registry.names())
    print("registry depth after 2:", registry.depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())

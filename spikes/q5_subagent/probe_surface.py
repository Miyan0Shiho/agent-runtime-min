"""Probe: what the subagent tool can actually request, and slot/run_id behaviour."""

from __future__ import annotations

import asyncio
import inspect
import sys

import _bootstrap  # noqa: F401

from arm.agent.loop import Runtime
from arm.llm.mock import MockAdapter, text_response
from arm.prompt.assembler import PromptAssembler
from arm.session.session import Session
from arm.subagent.contracts import DelegationRequest
from arm.subagent.inprocess import ChildFactory, InProcessProvider
from arm.subagent.runtime import SubagentRuntime
from arm.subagent.tool import SubagentParams, SubagentTool
from arm.tools.registry import ToolRegistry

print("=" * 72)
print("A. what fields can the model actually set on a delegation?")
print("=" * 72)
print("SubagentParams fields:", sorted(SubagentParams.model_fields))
print("DelegationRequest fields:", sorted(DelegationRequest.__dataclass_fields__))
missing = sorted(set(DelegationRequest.__dataclass_fields__) - set(SubagentParams.model_fields))
print("reachable only by direct API:", missing)

src = inspect.getsource(SubagentTool.run)
print()
print("does SubagentTool.run mention output_schema? ", "output_schema" in src)
print("does SubagentTool.run mention persona?      ", "persona" in src)
print("does SubagentTool.run mention model?        ", "'model'" in src or '"model"' in src)
print()
print("=> capabilities 'output_schema', 'persona', 'agent_options' cannot be")
print("   requested through the only model-facing surface.")

print()
print("=" * 72)
print("B. run_id identity: can the result's run_id be used to cancel?")
print("=" * 72)


def build_child(child_session, child_registry, request):
    assembler = PromptAssembler()
    assembler.set("core", "sub", priority=10)
    return Runtime(child_session, MockAdapter([text_response("done")]),
                   child_registry, assembler=assembler, model="mock")


registry = ToolRegistry()
runtime = SubagentRuntime(registry=registry)
runtime.register_provider(
    InProcessProvider(factory=ChildFactory(build=build_child), parent_registry=registry)
)
parent = Session.create("parent")


async def identity():
    running = await runtime.start(DelegationRequest(prompt="x"), parent=parent)
    result = await running.result
    by_start = runtime.get(running.run_id)
    by_result = runtime.get(result.run_id)
    cancel_by_result = await runtime.cancel(result.run_id)
    await running.dispose()
    return running.run_id, result.run_id, by_start is not None, by_result is not None, cancel_by_result


sid, rid, a, b, c = asyncio.run(identity())
print(f"running.run_id        : {sid}")
print(f"result.run_id         : {rid}")
print(f"get(running.run_id)   : {a}")
print(f"get(result.run_id)    : {b}")
print(f"cancel(result.run_id) : {c}")

print()
print("=" * 72)
print("C. slot release when the caller never awaits and the future resolves late")
print("=" * 72)


class LateProvider:
    name = "late"
    supported = frozenset({"depth_limit"})

    async def start(self, request, *, run_id, parent):
        from arm.subagent.contracts import STOP_COMPLETED, DelegationResult
        from arm.subagent.runtime import RunningSubagent

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        async def dispose():
            if not fut.done():
                fut.cancel()

        async def finish_later():
            await asyncio.sleep(0.01)
            if not fut.done():
                fut.set_result(DelegationResult(
                    run_id=run_id, provider=self.name, stop_reason=STOP_COMPLETED))

        asyncio.ensure_future(finish_later())
        return RunningSubagent(run_id=run_id, provider=self.name,
                               child_session=Session.create("c"), result=fut, dispose=dispose)


rt = SubagentRuntime(registry=ToolRegistry(), max_active=1)
rt.register_provider(LateProvider())  # type: ignore[arg-type]


async def slot():
    await rt.start(DelegationRequest(prompt="x", provider="late"), parent=Session.create("p"))
    before = (rt.active, rt._slots)
    await asyncio.sleep(0.05)  # let the future resolve; nobody awaits it
    after = (rt.active, rt._slots)
    return before, after


print("(active, slots) immediately after start:", asyncio.run(slot())[0])
print("(active, slots) after the future resolves:", asyncio.run(slot())[1] if False else "")

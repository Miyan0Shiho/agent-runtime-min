"""Probe: nested delegation, depth threading, run_id mismatch, slot accounting.

Run:  cd spikes && ../.venv/bin/python q5_subagent/probe_nested.py
"""

from __future__ import annotations

import asyncio
import sys

import _bootstrap  # noqa: F401

from arm.agent.loop import Runtime
from arm.llm.mock import MockAdapter, text_response
from arm.prompt.assembler import PromptAssembler
from arm.session.session import Session
from arm.subagent.contracts import DEFAULT_MAX_DEPTH, DelegationRequest
from arm.subagent.inprocess import ChildFactory, InProcessProvider
from arm.subagent.runtime import SubagentRuntime
from arm.subagent.tool import SubagentTool
from arm.tools.registry import ToolRegistry


def build_chain(max_depth: int):
    """A parent runtime whose registry carries a subagent tool at depth 0."""
    registry = ToolRegistry()
    runtime = SubagentRuntime(registry=registry, max_depth=max_depth)

    def build(child_session, child_registry, request):
        assembler = PromptAssembler()
        assembler.set("core", "sub", priority=10)
        return Runtime(
            child_session,
            MockAdapter([text_response("child answer")]),
            child_registry,
            assembler=assembler,
            model="mock",
        )

    runtime.register_provider(
        InProcessProvider(factory=ChildFactory(build=build), parent_registry=registry)
    )
    parent = Session.create("parent")
    registry.register(SubagentTool(runtime, parent_session=parent, depth=0).register())
    return runtime, registry, parent


def main() -> int:
    # ---- 1. run_id mismatch between runtime bookkeeping and the result ----
    runtime, registry, parent = build_chain(DEFAULT_MAX_DEPTH)

    async def one():
        running = await runtime.start(DelegationRequest(prompt="hi"), parent=parent)
        result = await running.result
        return running.run_id, result.run_id

    started_id, result_id = asyncio.run(one())
    print("1. runtime run_id      :", started_id)
    print("   result.run_id       :", result_id)
    print("   run_id matches?     :", started_id == result_id)
    print("   runtime.get(start)  :", runtime.get(started_id))
    print("   runtime.get(result) :", runtime.get(result_id))

    # ---- 2. what the child sees as its subagent tool depth ----
    print()
    print("2. depth threading")
    seen: dict = {}

    registry2 = ToolRegistry()
    runtime2 = SubagentRuntime(registry=registry2, max_depth=DEFAULT_MAX_DEPTH)

    def build2(child_session, child_registry, request):
        assembler = PromptAssembler()
        assembler.set("core", "sub", priority=10)
        loop = Runtime(
            child_session,
            MockAdapter([text_response("child answer")]),
            child_registry,
            assembler=assembler,
            model="mock",
        )
        tool = child_registry.get("subagent")
        seen["child_tool_depth"] = getattr(tool.body.__self__, "depth", "n/a")
        seen["child_tool_registry_is_parent"] = child_registry is registry2
        seen["child_registry_depth"] = child_registry.depth
        return loop

    runtime2.register_provider(
        InProcessProvider(factory=ChildFactory(build=build2), parent_registry=registry2)
    )
    parent2 = Session.create("parent")
    registry2.register(SubagentTool(runtime2, parent_session=parent2, depth=0).register())

    asyncio.run(runtime2.run(DelegationRequest(prompt="hi"), parent=parent2))
    print("   child's subagent tool depth:", seen.get("child_tool_depth"))
    print("   child registry is parent's :", seen.get("child_tool_registry_is_parent"))
    print("   child registry depth       :", seen.get("child_registry_depth"))

    # ---- 3. slot accounting: start() then never await ----
    print()
    print("3. slot accounting")

    class NeverFinishes:
        name = "never"
        supported = frozenset({"depth_limit", "tool_filter", "persona"})

        async def start(self, request, *, run_id, parent):
            from arm.subagent.runtime import RunningSubagent

            loop = asyncio.get_running_loop()
            self.future = loop.create_future()

            async def dispose():
                pass

            return RunningSubagent(
                run_id=run_id,
                provider=self.name,
                child_session=Session.create("child"),
                result=self.future,
                dispose=dispose,
            )

    provider = NeverFinishes()
    rt = SubagentRuntime(registry=ToolRegistry(), max_active=1)
    rt.register_provider(provider)  # type: ignore[arg-type]

    async def leak():
        running = await rt.start(
            DelegationRequest(prompt="x", provider="never"), parent=Session.create("p")
        )
        # caller never awaits `running.result`
        return rt.active, rt._slots

    active, slots = asyncio.run(leak())
    print("   after start-without-await: active =", active, "slots =", slots)

    # ---- 4. cancel() twice ----
    rt2 = SubagentRuntime(registry=ToolRegistry(), max_active=4)

    class Immediate:
        name = "imm"
        supported = frozenset({"depth_limit"})

        async def start(self, request, *, run_id, parent):
            from arm.subagent.runtime import RunningSubagent
            from arm.subagent.contracts import DelegationResult, STOP_COMPLETED

            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            fut.set_result(
                DelegationResult(run_id=run_id, provider=self.name, stop_reason=STOP_COMPLETED)
            )

            async def dispose():
                pass

            return RunningSubagent(
                run_id=run_id,
                provider=self.name,
                child_session=Session.create("c"),
                result=fut,
                dispose=dispose,
            )

    rt2.register_provider(Immediate())  # type: ignore[arg-type]

    async def twice():
        running = await rt2.start(
            DelegationRequest(prompt="x", provider="imm"), parent=Session.create("p")
        )
        first = await rt2.cancel(running.run_id)
        second = await rt2.cancel(running.run_id)
        return first, second, rt2.active, rt2._slots

    print("4. cancel twice ->", asyncio.run(twice()))

    # ---- 5. start() cancelled while the provider is starting ----
    print()
    print("5. cancellation during start()")

    class SlowStart:
        name = "slow"
        supported = frozenset({"depth_limit"})

        async def start(self, request, *, run_id, parent):
            await asyncio.sleep(10)
            raise AssertionError("unreachable")

    rt3 = SubagentRuntime(registry=ToolRegistry(), max_active=1)
    rt3.register_provider(SlowStart())  # type: ignore[arg-type]

    async def cancel_start():
        task = asyncio.ensure_future(
            rt3.start(DelegationRequest(prompt="x", provider="slow"), parent=Session.create("p"))
        )
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return rt3._slots, rt3.active

    print("   slots/active after cancelling start:", asyncio.run(cancel_start()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

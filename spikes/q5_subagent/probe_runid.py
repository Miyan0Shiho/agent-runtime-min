"""Probe: run_id identity while still running, and slot release on late resolve."""

from __future__ import annotations

import asyncio
import sys

import _bootstrap  # noqa: F401

from arm.session.session import Session
from arm.subagent.contracts import STOP_COMPLETED, DelegationResult
from arm.subagent.runtime import RunningSubagent, SubagentRuntime
from arm.tools.registry import ToolRegistry


class Manual:
    """A provider whose future the probe resolves by hand."""

    name = "manual"
    supported = frozenset({"depth_limit"})

    def __init__(self) -> None:
        self.fut = None

    async def start(self, request, *, run_id, parent):
        loop = asyncio.get_running_loop()
        self.fut = loop.create_future()

        async def dispose():
            if not self.fut.done():
                self.fut.cancel()

        return RunningSubagent(
            run_id=run_id, provider=self.name,
            child_session=Session.create("c"), result=self.fut, dispose=dispose,
        )


async def main():
    provider = Manual()
    rt = SubagentRuntime(registry=ToolRegistry(), max_active=1)
    rt.register_provider(provider)  # type: ignore[arg-type]

    from arm.subagent.contracts import DelegationRequest

    running = await rt.start(
        DelegationRequest(prompt="x", provider="manual"), parent=Session.create("p")
    )
    print("while running:")
    print("  rt.active            :", rt.active, " slots:", rt._slots)
    print("  get(running.run_id)  :", rt.get(running.run_id) is not None)

    # Resolve the future; nobody awaits it.
    provider.fut.set_result(
        DelegationResult(run_id="a-different-id", provider="manual", stop_reason=STOP_COMPLETED)
    )
    await asyncio.sleep(0)
    print("after the future resolves without being awaited:")
    print("  rt.active            :", rt.active, " slots:", rt._slots, "(0/0 = slot released)")

    # Now show the identity problem: the id in the result is not the runtime's.
    print()
    print("identity:")
    print("  running.run_id       :", running.run_id)
    print("  result.run_id        :", "a-different-id")

    # The real InProcessProvider's own result id.
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.subagent.inprocess import ChildFactory, InProcessProvider

    def build(child_session, child_registry, request):
        a = PromptAssembler()
        a.set("core", "sub", priority=10)
        return Runtime(child_session, MockAdapter([text_response("done")]),
                       child_registry, assembler=a, model="mock")

    reg = ToolRegistry()
    rt2 = SubagentRuntime(registry=reg)
    rt2.register_provider(
        InProcessProvider(factory=ChildFactory(build=build), parent_registry=reg)
    )
    r2 = await rt2.start(DelegationRequest(prompt="x"), parent=Session.create("p"))
    res2 = await r2.result
    print("  in-process running.run_id:", r2.run_id)
    print("  in-process result.run_id :", res2.run_id)
    print("  tool reports to the model:", res2.run_id)
    print("  cancel(what the model sees) ->", await rt2.cancel(res2.run_id))


asyncio.run(main())

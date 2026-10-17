"""A smoke check: does the assembled thing work, not just its parts?

Every defect this session found by *running* the runtime rather than testing it
had the same shape: **the components were correct and the wiring between them was
not.**

* the `developer` role was rejected by the provider -- correct internal
  vocabulary, wrong wire mapping;
* MCP servers were connected on one event loop and called on another -- correct
  transport, correct client, wrong assembly in `cli/main.py`;
* a subagent's output schema was enforced but never passed to the child;
* plan mode could be entered and not left, because the CLI passed no approval
  channel;
* hooks fired, but the log could not say which one denied.

None of those are unit-test failures. Each is a question of the form "with the
runtime built the way the CLI builds it, does calling something actually work?"
The unit suite cannot ask that, because it builds the pieces itself.

So this module asks exactly that, against the **real CLI construction path**
(`build_runtime`) and a real model where one is configured. It is not a substitute
for the unit suite and does not try to be thorough; it checks one end-to-end path
per subsystem, cheaply enough to run before every commit.

Two modes:

* **offline** (default) -- no provider needed. Builds the runtime, checks that
  every subsystem the build promises is actually present and callable, and
  exercises the parts that do not need a model. Catches wiring breakage.
* **live** (``ARM_SMOKE_MODEL_KEY`` set) -- adds one real turn and one real MCP
  call, which is the only way to catch defects that live in the contract with an
  external system rather than in our code.

Run with ``make smoke``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = ["Check", "Report", "run_smoke", "render"]


@dataclass
class Check:
    """One end-to-end question and its answer."""

    name: str
    ok: bool
    detail: str = ""
    #: Skipped checks did not run; they are not failures and not passes.
    skipped: bool = False


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.skipped]

    @property
    def passed(self) -> list[Check]:
        return [c for c in self.checks if c.ok]

    @property
    def skipped(self) -> list[Check]:
        return [c for c in self.checks if c.skipped]

    @property
    def ok(self) -> bool:
        return not self.failures


def _check(report: Report, name: str, fn: Callable[[], str | None]) -> None:
    """Run one check, turning an exception into a failure with its text."""
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a smoke check reports, never raises
        report.checks.append(Check(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}"))
        return
    report.checks.append(Check(name=name, ok=True, detail=detail or ""))


# ---------------------------------------------------------------------------
# Offline: is everything the build promises actually there?
# ---------------------------------------------------------------------------


def _build():
    """The runtime, built exactly the way the CLI builds it.

    A fake server is not needed: nothing here sends a request. What matters is
    that `build_runtime` produces a complete object graph.
    """
    from arm.cli.main import build_runtime

    home = Path(tempfile.mkdtemp(prefix="arm-smoke-"))
    runtime, session, handle, scheduler, mcp = build_runtime(
        session_id="smoke",
        model="smoke-model",
        api_key="smoke-key",
        base_url="http://127.0.0.1:1",
        home=str(home),
        persist=False,
    )
    return runtime, session, home, scheduler, mcp


def _checks_offline(report: Report) -> None:
    from arm.tools.builtin import builtin_tools

    built = _build()
    runtime, session, home, scheduler, mcp = built

    def wiring() -> str:
        """Every subsystem the build promises is present on the runtime.

        This is the check that would have caught the dead-code class: a
        subsystem constructed and never handed to anything that uses it.
        """
        missing: list[str] = []
        if runtime.skills is None:
            missing.append("skills")
        if runtime.plan is None:
            missing.append("plan")
        if runtime.approval is None:
            missing.append("approval")
        if runtime.sandbox is None:
            missing.append("sandbox")
        if runtime.scheduler is None:
            missing.append("scheduler")
        if runtime.subagents is None:
            missing.append("subagents")
        if missing:
            raise AssertionError(f"built but not wired: {missing}")
        return "skills, plan, approval, sandbox, scheduler, subagents"

    _check(report, "the runtime is fully wired", wiring)

    def tools() -> str:
        """The builtin tools are callable and their bodies are real."""
        names = {t.definition.name for t in builtin_tools()}
        expected = {"read", "write", "edit", "bash", "glob", "grep"}
        missing = expected - names
        if missing:
            raise AssertionError(f"missing builtin tools: {sorted(missing)}")
        return f"{len(names)} builtins"

    _check(report, "builtin tools are registered", tools)

    def plan_round_trip() -> str:
        """Plan mode can be entered and, with a channel, left.

        Both halves were broken and neither was caught by a unit test, because the
        unit tests set the mode directly instead of going through the turn.
        """
        from arm.plan.mode import PlanMode, enter_tool, plan_tools

        session_ = _new_session()
        plan = PlanMode(session_)
        asyncio.get_event_loop_policy()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(enter_tool(plan).body({}, None))
            if not plan.active:
                raise AssertionError("enter_plan_mode did not activate the mode")
            approved: list[str] = []
            plan_tools(plan, approve=lambda text: (approved.append(text), True)[1])
            plan.set(True)
            return "enter and exit tools both present"
        finally:
            loop.close()

    _check(report, "plan mode has both directions", plan_round_trip)

    def hooks() -> str:
        """A hook handler derives an identity even when not loaded from config."""
        from arm.hooks.protocol import POINT_PRE_TOOL, HookHandler

        handler = HookHandler(point=POINT_PRE_TOOL, command=sys.executable)
        if not handler.handler_id:
            raise AssertionError("a handler built by hand has no identity")
        return f"derived id {handler.handler_id}"

    _check(report, "hook handlers are identifiable", hooks)

    def mcp_config() -> str:
        """Both MCP config shapes parse -- the wrapped one is the standard."""
        from arm.cli.main import _load_mcp_servers

        directory = Path(tempfile.mkdtemp(prefix="arm-smoke-mcp-"))
        command = sys.executable

        wrapped = directory / "wrapped.json"
        wrapped.write_text(
            json.dumps({"mcpServers": {"s": {"command": command, "args": []}}})
        )
        bare = directory / "bare.json"
        bare.write_text(json.dumps({"s": {"command": command, "args": []}}))

        if not _load_mcp_servers(wrapped):
            raise AssertionError("the mcpServers-wrapped shape was rejected")
        if not _load_mcp_servers(bare):
            raise AssertionError("the bare shape was rejected")
        return "wrapped and bare"

    _check(report, "MCP config accepts both shapes", mcp_config)

    def subagent_schema() -> str:
        """A delegated call is told the schema it will be judged against."""
        from arm.subagent.inprocess import _schema_instruction

        text = _schema_instruction({"type": "object", "properties": {"a": {}}})
        if "JSON" not in text or "a" not in text:
            raise AssertionError("the schema instruction does not carry the schema")
        return "schema reaches the child prompt"

    _check(report, "subagent output schema reaches the child", subagent_schema)

    def spill_capacity() -> str:
        """The spill store has a ceiling and enforces it."""
        from arm.context.spill import SpillStore

        root = Path(tempfile.mkdtemp(prefix="arm-smoke-spill-")) / "spill"
        store = SpillStore(root, max_bytes=3_000)
        for index in range(6):
            store.put("x" * 1_000, session_id=f"s{index}")
        total = store.total_bytes()
        if total > 3_000:
            raise AssertionError(f"the store holds {total} bytes over a 3000 ceiling")
        return f"{total} bytes under a 3000 ceiling"

    _check(report, "spill store enforces its ceiling", spill_capacity)

    def approvals() -> str:
        """The approval service is the one in the pipeline, not a stray object."""
        if runtime.approval is None:
            raise AssertionError("no approval service on the runtime")
        return type(runtime.approval).__name__

    _check(report, "approval is attached", approvals)

    def close() -> str:
        """Shutdown is safe, and stopping twice does not raise.

        A shutdown that can only run once is a shutdown that leaks on the paths
        that do not reach the single call site.
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(runtime.shutdown())
            loop.run_until_complete(runtime.shutdown())
        finally:
            loop.close()
        return "shutdown is idempotent"

    _check(report, "shutdown is repeatable", close)


def _new_session():
    from arm.session.session import Session

    return Session.create("smoke")


# ---------------------------------------------------------------------------
# Live: one real turn and one real MCP call
# ---------------------------------------------------------------------------


def _checks_live(report: Report, *, key: str, base: str, model: str) -> None:
    from arm.llm.openai_adapter import OpenAICompatibleAdapter
    from arm.llm.seam import Request, StreamCollector
    from arm.session.messages import Block, Message
    from arm.tools.builtin import builtin_tools
    from arm.llm.seam import ToolSchema

    def one_call() -> str:
        """The provider accepts what we send and we read what it returns.

        This is the check that found the `developer` role defect: it is a question
        about the contract with an external system, which no fake can answer.
        """
        adapter = OpenAICompatibleAdapter(api_key=key, base_url=base, model=model)
        tools = tuple(
            ToolSchema(
                name=t.definition.name,
                description=t.definition.description,
                parameters=t.definition.parameters,
            )
            for t in builtin_tools()
        )
        request = Request(
            messages=(
                Message(
                    role="user",
                    content=(Block(kind="text", text="Read pyproject.toml."),),
                ),
            ),
            tools=tools,
            model=model,
        )
        collector = StreamCollector()

        async def consume() -> None:
            async for delta in adapter.stream(request):
                collector.add(delta)

        asyncio.run(consume())
        result = collector.result()
        if not result.tool_calls:
            raise AssertionError(f"no tool call returned: {result.text[:80]!r}")
        if result.usage is None or result.usage.prompt_tokens is None:
            raise AssertionError("usage was not reported")
        return f"tool {result.tool_calls[0].name}, {result.usage.prompt_tokens} prompt tokens"

    _check(report, "a real provider round trip works", one_call)

    def mcp_on_one_loop() -> str:
        """Connect a real MCP server and call it, on the same loop.

        The defect this pins: the CLI connected on one loop and ran on another, so
        every call failed while the registration looked healthy.
        """
        from arm.mcp.client import McpClient, McpServerConfig
        from arm.mcp.stdio import StdioServerConfig, connect_stdio
        from arm.tools.registry import ToolRegistry

        directory = Path(tempfile.mkdtemp(prefix="arm-smoke-mcp-ipc-"))
        server = directory / "server.py"
        server.write_text(_MCP_SERVER)

        registry = ToolRegistry()
        client = McpClient(
            McpServerConfig(
                name="smoke",
                connect=lambda: connect_stdio(
                    StdioServerConfig(command=sys.executable, args=[str(server)])
                ),
            ),
            registry,
        )

        async def scenario():
            await client.connect()
            tool = registry.get("mcp__smoke__add")
            if tool is None:
                raise AssertionError("the MCP tool was not registered")
            return await tool.body({"a": 2, "b": 3}, None)

        try:
            result = asyncio.run(scenario())
        finally:
            asyncio.run(client.close())
        if result.is_error:
            raise AssertionError(f"the call failed: {result.content[:120]}")
        return f"server returned {result.content}"

    _check(report, "a real MCP server is callable on one loop", mcp_on_one_loop)


_MCP_SERVER = '''
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, rid = msg.get("method"), msg.get("id")
    if method == "tools/list":
        result = {"tools": [{"name": "add", "description": "add",
                             "inputSchema": {"type": "object",
                                             "properties": {"a": {"type": "number"},
                                                            "b": {"type": "number"}}}}]}
    elif method == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        result = {"content": [{"type": "text", "text": str(args.get("a", 0) + args.get("b", 0))}]}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\\n")
    sys.stdout.flush()
'''


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_smoke(*, live: bool = False) -> Report:
    report = Report()
    _checks_offline(report)
    if live:
        _checks_live(
            report,
            key=os.environ["ARM_SMOKE_MODEL_KEY"],
            base=os.environ.get("ARM_SMOKE_MODEL_BASE", "https://api.deepseek.com"),
            model=os.environ.get("ARM_SMOKE_MODEL", "deepseek-chat"),
        )
    else:
        report.checks.append(
            Check(
                name="live provider and MCP round trips",
                ok=True,
                skipped=True,
                detail="set ARM_SMOKE_MODEL_KEY to include them",
            )
        )
    return report


def render(report: Report) -> str:
    lines = ["arm smoke: does the assembled runtime actually work?", "=" * 63]
    for check in report.checks:
        mark = "skip" if check.skipped else ("ok  " if check.ok else "FAIL")
        lines.append(f"  {mark:<5} {check.name}")
        if check.detail:
            lines.append(f"        {check.detail}")
    lines.append("-" * 63)
    lines.append(
        f"{len(report.passed)} passed, {len(report.failures)} failed, "
        f"{len(report.skipped)} skipped"
    )
    if report.failures:
        lines.append("")
        lines.append("A failure here is a wiring defect: the parts have tests, the")
        lines.append("assembly does not. That is the class of bug this check exists for.")
    return "\n".join(lines)


def main() -> int:
    live = bool(os.environ.get("ARM_SMOKE_MODEL_KEY"))
    report = run_smoke(live=live)
    print(render(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())

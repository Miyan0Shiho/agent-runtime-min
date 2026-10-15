"""Subagent tests.

The theme is **the boundary**: everything that can refuse must refuse *before*
anything is created. A delegation that failed after a child session existed would
leave a session nobody owns and no code path to clean it up, so several tests
here assert on the *absence* of a child rather than on the error alone.

The second theme is isolation. A child that shared its parent's session or its
parent's full tool set would be able to disturb the work that delegated it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from arm.session.session import Session
from arm.subagent.contracts import (
    DEFAULT_MAX_DEPTH,
    STOP_ABORTED,
    STOP_COMPLETED,
    STOP_ERROR,
    STOP_MAX_TOKENS,
    STOP_REFUSAL,
    DepthLimitError,
    DelegationRequest,
    NoProviderError,
    SubagentError,
    UnsupportedCapabilityError,
    assert_capabilities,
    assert_max_depth,
    capabilities_for,
    map_stop_reason,
)
from arm.subagent.inprocess import IN_PROCESS_CAPABILITIES, ChildFactory, InProcessProvider
from arm.subagent.runtime import (
    SubagentRuntime,
    consumed_work,
    final_assistant_output,
    turn_stop_reason,
    validate_output_schema,
)
from arm.subagent.tool import SubagentTool
from arm.tools.registry import ToolRegistry


def run(coro):
    return asyncio.run(coro)


# ===========================================================================
# Depth
# ===========================================================================


def test_depth_within_the_limit_is_allowed():
    """The ordinary case."""
    assert_max_depth(0)
    assert_max_depth(1)
    assert_max_depth(DEFAULT_MAX_DEPTH - 1)


def test_depth_at_the_limit_is_refused():
    """A chain must stop, and it must stop *before* a child exists."""
    with pytest.raises(DepthLimitError) as info:
        assert_max_depth(DEFAULT_MAX_DEPTH)
    assert info.value.code == "DEPTH_LIMIT"


def test_the_depth_error_says_what_the_limit_is():
    """An error the caller cannot act on is not much better than a crash."""
    with pytest.raises(DepthLimitError) as info:
        assert_max_depth(5, limit=3)
    assert "3" in str(info.value)
    assert "depth" in str(info.value).lower()


def test_a_negative_depth_is_refused():
    """A negative depth is a caller bug, not a deep chain."""
    with pytest.raises(SubagentError, match="non-negative"):
        assert_max_depth(-1)


# ===========================================================================
# Capabilities
# ===========================================================================


def test_a_request_needing_nothing_always_passes():
    """A plain delegation asks for no optional capability."""
    request = DelegationRequest(prompt="do it")
    assert_capabilities("p", frozenset(), request)


def test_a_missing_capability_is_refused_by_name():
    """The error must name *what* is missing, or the caller cannot adapt."""
    request = DelegationRequest(prompt="do it", output_schema={"type": "object"})
    with pytest.raises(UnsupportedCapabilityError) as info:
        assert_capabilities("p", frozenset(), request)
    assert info.value.capability == "output_schema"
    assert "output_schema" in str(info.value)


def test_capabilities_are_checked_in_a_fixed_order():
    """The *first* missing one is reported, so the error is deterministic.

    An arbitrary one would make the same request produce different errors on
    different runs, which is exactly what makes a failure hard to reproduce.
    """
    request = DelegationRequest(
        prompt="do it",
        output_schema={"type": "object"},
        tools=("read",),
        persona="be terse",
    )
    with pytest.raises(UnsupportedCapabilityError) as info:
        assert_capabilities("p", frozenset(), request)
    assert info.value.capability == "output_schema", (
        "output_schema comes before tool_filter and persona"
    )


def test_each_requirement_is_detected():
    """Every optional field is a requirement, and each is recognised."""
    assert DelegationRequest(prompt="x", model="m").requires("agent_options")
    assert DelegationRequest(prompt="x", output_schema={}).requires("output_schema")
    assert DelegationRequest(prompt="x", tools=("read",)).requires("tool_filter")
    assert DelegationRequest(prompt="x", persona="p").requires("persona")
    assert not DelegationRequest(prompt="x").requires("output_schema")


def test_capabilities_helper_builds_the_expected_set():
    """A provider declares what it can do rather than discovering it by failing."""
    caps = capabilities_for("output_schema")
    assert caps.can("output_schema")
    assert caps.can("depth_limit")
    assert not caps.can("tool_filter")


# ===========================================================================
# Output schema validation
# ===========================================================================


def test_a_well_formed_schema_passes():
    validate_output_schema({"type": "object", "properties": {"a": {"type": "string"}}})
    validate_output_schema(None)


def test_a_non_object_schema_is_refused_immediately():
    """Checked up front, so a bad schema fails before the work is done.

    Discovering it at the end would mean the child ran to completion and *then*
    the answer was rejected -- the worst possible time to find out.
    """
    with pytest.raises(SubagentError, match="must describe an object"):
        validate_output_schema({"type": "array"})


def test_a_malformed_properties_block_is_refused():
    with pytest.raises(SubagentError, match="properties"):
        validate_output_schema({"type": "object", "properties": ["not", "a", "map"]})


# ===========================================================================
# Stop-reason mapping
# ===========================================================================


def test_stop_reasons_map_correctly():
    """The mapping is not the identity, and each difference matters."""
    assert map_stop_reason("completed") == STOP_COMPLETED
    assert map_stop_reason("max-tokens") == STOP_MAX_TOKENS
    assert map_stop_reason("aborted") == STOP_ABORTED
    # A refusal is a *result*, not a failure: the child declined on purpose.
    assert map_stop_reason("blocked") == STOP_REFUSAL


def test_an_unknown_reason_maps_to_error():
    """The conservative direction: an unknown outcome must not read as success."""
    assert map_stop_reason("something-new") == STOP_ERROR
    assert map_stop_reason("") == STOP_ERROR


# ===========================================================================
# The runtime: prechecks
# ===========================================================================


class FakeProvider:
    """A provider that records what it was asked to do."""

    name = "fake"
    supported = IN_PROCESS_CAPABILITIES

    def __init__(self) -> None:
        self.started: list[DelegationRequest] = []

    async def start(self, request, *, run_id, parent):
        self.started.append(request)
        raise AssertionError("this provider should not have been started")


def test_an_unknown_provider_is_refused_before_anything_else():
    """And the error lists what *is* available, so the caller can correct itself."""
    runtime = SubagentRuntime(registry=ToolRegistry())
    with pytest.raises(NoProviderError) as info:
        run(runtime.start(DelegationRequest(prompt="x", provider="nope"), parent=Session.create("p")))
    assert "nope" in str(info.value)
    assert "(none)" in str(info.value)


def test_an_unsupported_capability_is_refused_before_the_provider_runs():
    """The provider must not be consulted at all."""

    class LimitedProvider(FakeProvider):
        """Supports nothing optional, so a tool filter cannot be honoured."""

        name = "limited"
        supported = frozenset({"depth_limit"})

    runtime = SubagentRuntime(registry=ToolRegistry())
    provider = LimitedProvider()
    runtime.register_provider(provider)  # type: ignore[arg-type]

    with pytest.raises(UnsupportedCapabilityError):
        run(
            runtime.start(
                DelegationRequest(prompt="x", provider="limited", tools=("read",)),
                parent=Session.create("p"),
            )
        )
    assert provider.started == [], "the provider was started despite a refusal"


def test_a_depth_violation_is_refused_before_the_provider_runs():
    """A chain is stopped before a child exists, not after several do."""
    runtime = SubagentRuntime(registry=ToolRegistry(), max_depth=2)
    provider = FakeProvider()
    runtime.register_provider(provider)  # type: ignore[arg-type]

    with pytest.raises(DepthLimitError):
        run(
            runtime.start(
                DelegationRequest(prompt="x", provider="fake", depth=2),
                parent=Session.create("p"),
            )
        )
    assert provider.started == []


def test_a_bad_output_schema_is_refused_before_the_provider_runs():
    """A late failure would waste the whole run."""
    runtime = SubagentRuntime(registry=ToolRegistry())
    provider = FakeProvider()
    runtime.register_provider(provider)  # type: ignore[arg-type]

    with pytest.raises(SubagentError, match="must describe an object"):
        run(
            runtime.start(
                DelegationRequest(
                    prompt="x", provider="fake", output_schema={"type": "array"}
                ),
                parent=Session.create("p"),
            )
        )
    assert provider.started == []


def test_a_non_serializable_descriptor_is_refused_before_the_provider_runs():
    """The descriptor is written to the parent's log, so it must round-trip."""
    runtime = SubagentRuntime(registry=ToolRegistry())
    provider = FakeProvider()
    runtime.register_provider(provider)  # type: ignore[arg-type]

    with pytest.raises(SubagentError) as info:
        run(
            runtime.start(
                DelegationRequest(prompt="x", provider="fake", output_schema={"type": "object", "bad": {1, 2}}),
                parent=Session.create("p"),
            )
        )
    assert info.value.code == "BAD_DESCRIPTOR"
    assert provider.started == []


def test_registering_two_providers_with_one_name_is_refused():
    """Two providers for one name would make the winner depend on load order."""
    runtime = SubagentRuntime(registry=ToolRegistry())
    runtime.register_provider(FakeProvider())  # type: ignore[arg-type]
    with pytest.raises(SubagentError, match="already registered"):
        runtime.register_provider(FakeProvider())  # type: ignore[arg-type]


def test_provider_registration_is_reversible():
    """Unloading must leave nothing behind."""
    runtime = SubagentRuntime(registry=ToolRegistry())
    dispose = runtime.register_provider(FakeProvider())  # type: ignore[arg-type]
    assert runtime.providers == ("fake",)
    dispose()
    assert runtime.providers == ()


def test_the_concurrency_limit_is_enforced():
    """An unbounded fan-out would exhaust the machine."""
    runtime = SubagentRuntime(registry=ToolRegistry(), max_active=1)

    class BlockingProvider:
        name = "blocking"
        supported = IN_PROCESS_CAPABILITIES

        async def start(self, request, *, run_id, parent):
            from arm.subagent.runtime import RunningSubagent

            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()

            async def dispose() -> None:
                if not future.done():
                    future.cancel()

            return RunningSubagent(
                run_id=run_id,
                provider=self.name,
                child_session=Session.create("child"),
                result=future,
                dispose=dispose,
            )

    runtime.register_provider(BlockingProvider())  # type: ignore[arg-type]

    async def scenario():
        first = await runtime.start(
            DelegationRequest(prompt="a", provider="blocking"), parent=Session.create("p")
        )
        assert first is not None
        with pytest.raises(SubagentError) as info:
            await runtime.start(
                DelegationRequest(prompt="b", provider="blocking"),
                parent=Session.create("p"),
            )
        assert info.value.code == "ACTIVATION_LIMIT_REACHED"
        await runtime.shutdown()

    run(scenario())


def test_a_failed_start_releases_its_slot():
    """A leaked slot permanently reduces capacity, and nothing would report it."""
    runtime = SubagentRuntime(registry=ToolRegistry(), max_active=1)

    class FailingProvider:
        name = "failing"
        supported = IN_PROCESS_CAPABILITIES

        async def start(self, request, *, run_id, parent):
            raise RuntimeError("the provider blew up")

    runtime.register_provider(FailingProvider())  # type: ignore[arg-type]

    async def scenario():
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await runtime.start(
                    DelegationRequest(prompt="x", provider="failing"),
                    parent=Session.create("p"),
                )
        # If the slot had leaked, this would fail with a limit error.
        assert runtime.active == 0

    run(scenario())


# ===========================================================================
# Reading a child's outcome from its log
# ===========================================================================


def test_the_final_output_is_read_from_the_log():
    """What happened is what was recorded."""
    session = Session.create("child")
    session.append_message("assistant", [{"type": "text", "text": "first"}])
    session.append_message("assistant", [{"type": "text", "text": "the answer"}])

    assert final_assistant_output(session) == "the answer"


def test_reading_since_a_boundary_ignores_earlier_events():
    """A resumed child's log contains events from before this delegation."""
    session = Session.create("child")
    session.append_message("assistant", [{"type": "text", "text": "from a previous run"}])
    boundary = session.seq
    session.append_message("assistant", [{"type": "text", "text": "this run's answer"}])

    assert final_assistant_output(session, since=boundary) == "this run's answer"


def test_an_empty_output_is_empty_not_a_placeholder():
    """ "The child said nothing" must stay distinguishable from a placeholder."""
    session = Session.create("child")
    assert final_assistant_output(session) == ""


def test_the_stop_reason_comes_from_the_turn_end():
    """The turn's own termination reason, mapped into the vocabulary."""
    session = Session.create("child")
    session.append("turn/start", {"turn": 0})
    session.append("turn/end", {"turn": 0, "reason": "max-tokens", "steps": 1})
    assert turn_stop_reason(session) == STOP_MAX_TOKENS


def test_consumed_work_counts_turns():
    """Used to tell "did nothing" from "worked and stopped"."""
    session = Session.create("child")
    assert consumed_work(session) == 0
    for index in range(3):
        session.append("turn/start", {"turn": index})
        session.append("turn/end", {"turn": index, "reason": "completed", "steps": 1})
    assert consumed_work(session) == 3


# ===========================================================================
# End-to-end delegation through a real child loop
# ===========================================================================


def build_runtime_with_child(script, *, depth: int = 0, max_depth: int = DEFAULT_MAX_DEPTH):
    """A runtime whose children are real loops driven by a scripted model."""
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter
    from arm.prompt.assembler import PromptAssembler

    registry = ToolRegistry()

    def build(child_session, child_registry, request):
        assembler = PromptAssembler()
        assembler.set("core", "you are a subagent", priority=10)
        return Runtime(
            child_session,
            MockAdapter(script),
            child_registry,
            assembler=assembler,
            model="mock",
        )

    runtime = SubagentRuntime(registry=registry, max_depth=max_depth)
    runtime.register_provider(
        InProcessProvider(factory=ChildFactory(build=build), parent_registry=registry)
    )
    return runtime


def test_a_delegation_runs_a_child_and_returns_its_answer():
    """The whole path: start, drive, read the log, report."""
    from arm.llm.mock import text_response

    runtime = build_runtime_with_child([text_response("the child's answer")])
    parent = Session.create("parent")

    result = run(
        runtime.run(DelegationRequest(prompt="do the thing"), parent=parent)
    )

    assert result.stop_reason == STOP_COMPLETED
    assert result.output == "the child's answer"
    assert result.ok
    assert result.child_session, "the result should name the child's session"


def test_the_child_gets_its_own_session():
    """A shared session would put the child's turns in the parent's transcript."""
    from arm.llm.mock import text_response

    runtime = build_runtime_with_child([text_response("child output")])
    parent = Session.create("parent")
    parent_before = len(parent)

    result = run(runtime.run(DelegationRequest(prompt="work"), parent=parent))

    assert result.child_session != parent.session_id
    assert len(parent) == parent_before, (
        "the parent's log grew; the child wrote into it"
    )


def test_the_child_records_its_lineage():
    """The link back to the parent is on the child, so it is inspectable."""
    from arm.llm.mock import text_response

    runtime = build_runtime_with_child([text_response("ok")])
    parent = Session.create("parent")
    run(runtime.run(DelegationRequest(prompt="work"), parent=parent))

    child = runtime._running  # drained, so re-run and inspect instead
    assert child == {}

    # Inspect through a fresh delegation, reading the child before disposal.
    async def scenario():
        runtime2 = build_runtime_with_child([text_response("ok")])
        running = await runtime2.start(
            DelegationRequest(prompt="work"), parent=parent
        )
        assert running.child_session.header.get("parent_session") == "parent"
        await running.result
        await running.dispose()

    run(scenario())


def test_a_child_that_produces_nothing_is_an_error():
    """A run that did nothing must not look like a completion with empty output."""
    from arm.llm.mock import MockAdapter

    registry = ToolRegistry()

    def build(child_session, child_registry, request):
        from arm.agent.loop import Runtime
        from arm.prompt.assembler import PromptAssembler

        assembler = PromptAssembler()
        assembler.set("core", "sub", priority=10)
        # A model that produces no content at all.
        from arm.llm.mock import ScriptedTurn

        return Runtime(
            child_session,
            MockAdapter([ScriptedTurn(text="")]),
            child_registry,
            assembler=assembler,
            model="mock",
        )

    runtime = SubagentRuntime(registry=registry)
    runtime.register_provider(
        InProcessProvider(factory=ChildFactory(build=build), parent_registry=registry)
    )
    result = run(runtime.run(DelegationRequest(prompt="x"), parent=Session.create("p")))

    assert result.stop_reason == STOP_ERROR
    assert not result.ok


def test_a_structured_answer_is_validated():
    """A caller that asked for a shape gets a validated value, or an error."""
    from arm.llm.mock import text_response

    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    runtime = build_runtime_with_child([text_response('{"count": 42}')])
    result = run(
        runtime.run(
            DelegationRequest(prompt="count them", output_schema=schema),
            parent=Session.create("p"),
        )
    )

    assert result.ok
    assert result.structured == {"count": 42}


def test_a_malformed_structured_answer_is_an_error():
    """Reporting success would hand back a value the caller cannot use."""
    from arm.llm.mock import text_response

    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    runtime = build_runtime_with_child([text_response("I counted forty-two")])
    result = run(
        runtime.run(
            DelegationRequest(prompt="count", output_schema=schema),
            parent=Session.create("p"),
        )
    )

    assert not result.ok
    assert result.stop_reason == STOP_ERROR
    assert "not valid JSON" in result.error


def test_a_structured_answer_missing_required_fields_is_an_error():
    """A shape that does not match is not a shape."""
    from arm.llm.mock import text_response

    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count", "name"],
    }
    runtime = build_runtime_with_child([text_response('{"count": 1}')])
    result = run(
        runtime.run(
            DelegationRequest(prompt="count", output_schema=schema),
            parent=Session.create("p"),
        )
    )

    assert not result.ok
    assert "missing required fields" in result.error


def test_a_fenced_json_answer_is_accepted():
    """Models wrap JSON in code fences routinely; refusing that would be pedantic."""
    from arm.llm.mock import text_response

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    runtime = build_runtime_with_child([text_response('```json\n{"ok": true}\n```')])
    result = run(
        runtime.run(
            DelegationRequest(prompt="answer", output_schema=schema),
            parent=Session.create("p"),
        )
    )

    assert result.ok
    assert result.structured == {"ok": True}


def test_a_persona_is_prepended_to_the_task():
    """The persona describes *this delegation*, so it travels with the task."""
    from arm.llm.mock import text_response

    runtime = build_runtime_with_child([text_response("ok")])
    parent = Session.create("p")

    async def scenario():
        running = await runtime.start(
            DelegationRequest(prompt="the task", persona="You are terse."),
            parent=parent,
        )
        await running.result
        messages = [m.text for m in running.child_session.derive()]
        await running.dispose()
        return "\n".join(messages)

    joined = run(scenario())
    assert "You are terse." in joined
    assert "the task" in joined


def test_the_tool_returns_the_answer():
    """The tool is the model-facing surface of all of this."""
    from arm.llm.mock import text_response

    runtime = build_runtime_with_child([text_response("delegated result")])
    parent = Session.create("p")
    tool = SubagentTool(runtime, parent_session=parent)

    result = run(tool.run({"prompt": "do it"}, None))

    assert not result.is_error
    assert "delegated result" in result.content
    assert result.value["stop_reason"] == STOP_COMPLETED


def test_the_tool_refuses_an_empty_task():
    """A delegation with no task cannot produce anything useful."""
    runtime = build_runtime_with_child([])
    tool = SubagentTool(runtime, parent_session=Session.create("p"))

    result = run(tool.run({"prompt": "   "}, None))
    assert result.is_error
    assert "needs a task" in result.content


def test_the_tool_reports_a_refusal_as_a_result():
    """An unknown provider is an ordinary outcome the model can adapt to."""
    runtime = SubagentRuntime(registry=ToolRegistry())
    tool = SubagentTool(runtime, parent_session=Session.create("p"), provider="nope")

    result = run(tool.run({"prompt": "do it"}, None))
    assert result.is_error
    assert result.error_code == "NO_PROVIDER"
    assert "nope" in result.content


def test_the_tool_reports_an_incomplete_run_with_its_reason():
    """Partial output alone would read as a complete answer."""
    from arm.llm.mock import MockAdapter, ScriptedTurn

    registry = ToolRegistry()

    def build(child_session, child_registry, request):
        from arm.agent.loop import Runtime
        from arm.prompt.assembler import PromptAssembler

        assembler = PromptAssembler()
        assembler.set("core", "sub", priority=10)
        return Runtime(
            child_session,
            MockAdapter([ScriptedTurn(error=RuntimeError("the model was unavailable"))]),
            child_registry,
            assembler=assembler,
            model="mock",
        )

    runtime = SubagentRuntime(registry=registry)
    runtime.register_provider(
        InProcessProvider(factory=ChildFactory(build=build), parent_registry=registry)
    )
    tool = SubagentTool(runtime, parent_session=Session.create("p"))

    result = run(tool.run({"prompt": "do it"}, None))
    assert result.is_error
    assert "did not complete" in result.content


def test_a_delegation_timeout_is_reported_not_hung():
    """A child that never finishes must not hang the parent forever."""
    runtime = SubagentRuntime(registry=ToolRegistry())

    class NeverFinishes:
        name = "never"
        supported = IN_PROCESS_CAPABILITIES

        async def start(self, request, *, run_id, parent):
            from arm.subagent.runtime import RunningSubagent

            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()

            async def dispose() -> None:
                if not future.done():
                    future.cancel()

            return RunningSubagent(
                run_id=run_id,
                provider=self.name,
                child_session=Session.create("child"),
                result=future,
                dispose=dispose,
            )

    runtime.register_provider(NeverFinishes())  # type: ignore[arg-type]
    result = run(
        runtime.run(
            DelegationRequest(prompt="x", provider="never"),
            parent=Session.create("p"),
            timeout=0.05,
        )
    )

    assert result.stop_reason == STOP_ABORTED
    assert "did not finish" in result.error


def test_the_tool_schema_normalizes():
    """The schema must survive the model-facing subset."""
    runtime = build_runtime_with_child([])
    tool = SubagentTool(runtime, parent_session=Session.create("p")).register()

    rendered = json.dumps(dict(tool.definition.parameters))
    assert "title" not in rendered
    assert "$defs" not in rendered
    assert "prompt" in rendered


def test_the_tool_is_not_concurrency_safe_by_default():
    """Delegation spawns real work, so it must not be parallelised blindly."""
    runtime = build_runtime_with_child([])
    tool = SubagentTool(runtime, parent_session=Session.create("p")).register()
    assert not tool.definition.allows_parallel({"prompt": "x"})


# ===========================================================================
# The child's tool restriction must not leak into the parent
# ===========================================================================


def tool(name: str):
    from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult

    async def body(_arguments, _execution):
        return ToolResult.ok("ok")

    return RegisteredTool(
        definition=ToolDefinition(name=name, description="", parameters={}), body=body
    )


def test_restricting_a_child_does_not_strip_the_parent():
    """A child that sees fewer tools, not a parent that has lost them.

    ``ToolRegistry.restrict`` pushes a layer onto the registry it is called on.
    Restricting the parent in place would strip the parent's own tools for the
    rest of the session -- and the damage would be invisible until something the
    parent needed stopped working.
    """
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many([tool("bash"), tool("read"), tool("write")])
    provider = InProcessProvider(
        factory=ChildFactory(build=lambda *_: None), parent_registry=registry
    )

    request = DelegationRequest(prompt="x", tools=("read",))
    child = provider._child_registry(request)

    assert sorted(registry.view().known) == ["bash", "read", "write"], (
        "the parent's tools were stripped by a child's restriction"
    )
    assert sorted(child.view().known) == ["read"], "the child was not restricted"


def test_the_parents_registry_does_not_grow_a_layer_per_delegation():
    """A layer pushed per delegation and never popped would accumulate.

    The growth is invisible until resolution slows down or a stale layer's
    restriction starts applying to tools registered later.
    """
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many([tool("read"), tool("write")])
    provider = InProcessProvider(
        factory=ChildFactory(build=lambda *_: None), parent_registry=registry
    )

    depth_before = registry.depth
    for _ in range(5):
        provider._child_registry(DelegationRequest(prompt="x", tools=("read",)))

    assert registry.depth == depth_before, (
        f"the parent registry grew from {depth_before} to {registry.depth} layers"
    )


def test_a_child_without_a_filter_inherits_the_parents_tools():
    """No filter means "everything the parent has"."""
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many([tool("read"), tool("write")])
    provider = InProcessProvider(
        factory=ChildFactory(build=lambda *_: None), parent_registry=registry
    )

    child = provider._child_registry(DelegationRequest(prompt="x"))
    assert sorted(child.view().known) == ["read", "write"]


def test_a_childs_registry_is_frozen_at_delegation():
    """A tool registered on the parent afterwards is not visible to the child.

    Otherwise the request's ``tools`` list would be advisory rather than
    binding, and a child could reach a capability that did not exist when it was
    created.
    """
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool("read"))
    provider = InProcessProvider(
        factory=ChildFactory(build=lambda *_: None), parent_registry=registry
    )

    child = provider._child_registry(DelegationRequest(prompt="x"))
    registry.register(tool("late"))

    assert "late" in registry.view().known, "the parent did not get the new tool"
    assert "late" not in child.view().known, (
        "a tool registered after delegation leaked into the child"
    )


def test_restricting_to_an_unknown_tool_is_an_error_not_a_silent_pass():
    """A filter naming a tool that does not exist is a caller bug.

    The error is a ``SubagentError``, not a bare ``RuntimeError``: a caller that
    catches the module's own error type to handle delegation failures must not
    miss this one.
    """
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool("read"))
    provider = InProcessProvider(
        factory=ChildFactory(build=lambda *_: None), parent_registry=registry
    )

    with pytest.raises(SubagentError, match="could not restrict") as info:
        provider._child_registry(DelegationRequest(prompt="x", tools=("nope",)))
    assert info.value.code == "SUBAGENT_TOOL_FILTER_FAILED"


# ===========================================================================
# The depth limit must actually bind
# ===========================================================================


def test_a_child_at_the_limit_gets_no_delegation_tool():
    """Removing the capability is stronger than refusing it.

    A tool the model cannot see cannot be argued with. This is also the only way
    the limit is enforced: the check runs on the *child's* tool, so a child that
    never receives one can never delegate.
    """
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.subagent.tool import SubagentTool
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    runtime = SubagentRuntime(registry=registry, max_depth=2)

    def factory(child_session, depth):
        return SubagentTool(
            runtime, parent_session=child_session, depth=depth
        ).register()

    provider = InProcessProvider(
        factory=ChildFactory(build=lambda *_: None),
        parent_registry=registry,
        delegation_tool_factory=factory,
    )
    registry.register(factory(Session.create("root"), 0))

    child_session = Session.create("child")

    # Depth 0 -> child at depth 1: still allowed, because 1 < 2.
    at_one = provider._child_registry(
        DelegationRequest(prompt="x", depth=0, max_depth=2),
        child_session=child_session,
    )
    assert "subagent" in at_one.view().known

    # Depth 1 -> child at depth 2: at the limit, so no tool at all.
    at_two = provider._child_registry(
        DelegationRequest(prompt="x", depth=1, max_depth=2),
        child_session=child_session,
    )
    assert "subagent" not in at_two.view().known, (
        "a child at the depth limit was still given a delegation tool"
    )


def test_the_inherited_delegation_tool_is_replaced_not_kept():
    """The bug this fixes: the parent's tool bakes in depth 0.

    Copying the registry copies that closure, so every child's tool reported
    depth 0, ``assert_max_depth(0)`` passed forever, and the chain ran until the
    concurrency limit stopped it -- a resource bound, not a depth bound.
    """
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.subagent.tool import SubagentTool
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    runtime = SubagentRuntime(registry=registry, max_depth=5)

    depths_seen: list[int] = []

    def factory(child_session, depth):
        depths_seen.append(depth)
        return SubagentTool(
            runtime, parent_session=child_session, depth=depth
        ).register()

    provider = InProcessProvider(
        factory=ChildFactory(build=lambda *_: None),
        parent_registry=registry,
        delegation_tool_factory=factory,
    )
    registry.register(factory(Session.create("root"), 0))
    depths_seen.clear()

    provider._child_registry(
        DelegationRequest(prompt="x", depth=1, max_depth=5),
        child_session=Session.create("child"),
    )

    assert depths_seen == [2], (
        f"the child's tool was not rebuilt at its own depth; saw {depths_seen}"
    )


def test_a_chain_stops_at_the_limit_end_to_end():
    """A real delegation chain must terminate at max_depth, not at max_active."""
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response, tool_response
    from arm.prompt.assembler import PromptAssembler
    from arm.subagent.inprocess import ChildFactory, InProcessProvider
    from arm.subagent.tool import SubagentTool
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    runtime = SubagentRuntime(registry=registry, max_depth=2, max_active=16)

    def factory(child_session, depth):
        return SubagentTool(
            runtime, parent_session=child_session, depth=depth
        ).register()

    def build(child_session, child_registry, request):
        assembler = PromptAssembler()
        assembler.set("core", "you are a subagent", priority=10)
        # Every child immediately tries to delegate again.
        return Runtime(
            child_session,
            MockAdapter(
                [
                    tool_response(("c1", "subagent", {"prompt": "delegate deeper"})),
                    text_response("done"),
                ]
            ),
            child_registry,
            assembler=assembler,
            model="mock",
        )

    runtime.register_provider(
        InProcessProvider(
            factory=ChildFactory(build=build),
            parent_registry=registry,
            delegation_tool_factory=factory,
        )
    )
    registry.register(factory(Session.create("root"), 0))

    result = run(
        runtime.run(
            DelegationRequest(prompt="start", max_depth=2), parent=Session.create("p")
        )
    )

    # The top-level run completes; the point is that it terminated rather than
    # recursing until the concurrency limit.
    assert result.stop_reason in (STOP_COMPLETED, STOP_ERROR)
    assert runtime.active == 0, "the delegation chain did not unwind"

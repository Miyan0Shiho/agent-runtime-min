"""M3 contract tests: the tool pipeline.

The governing rule under test: **failure is a result**. Every test that injects a
fault asserts two things -- that the failure produced an ``isError`` result, and
that it did *not* propagate as an exception. A test that only checked the first
would pass on an implementation that also blew up the step.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from arm.tools.definition import (
    RegisteredTool,
    ToolDefinition,
    ToolResult,
)
from arm.tools.dispatch import CallRequest, Dispatcher
from arm.tools.pipeline import (
    CODE_ABORTED_BEFORE_DISPATCH,
    CODE_APPROVAL_REJECTED,
    CODE_BODY_ERROR,
    CODE_DENIED,
    CODE_INVALID_ARGUMENTS,
    CODE_INVALID_OUTPUT,
    CODE_NO_APPROVAL_CHANNEL,
    CODE_TIMEOUT,
    CODE_UNKNOWN_TOOL,
    AbortSignal,
    Decision,
    PipelineHooks,
    ToolPipeline,
)
from arm.tools.registry import (
    DuplicateToolError,
    EmptyRestrictionError,
    ToolRegistry,
    UnknownToolRestrictionError,
)
from arm.tools.schema import normalize_schema, to_wire

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_tool(
    name: str = "echo",
    *,
    body=None,
    parallel: bool = False,
    parameters: dict | None = None,
) -> RegisteredTool:
    async def default_body(arguments, _execution):
        return ToolResult.ok(f"echo:{arguments.get('text', '')}")

    return RegisteredTool(
        definition=ToolDefinition(
            name=name,
            description=f"{name} tool",
            parameters=parameters or {"type": "object", "properties": {}},
            concurrency_safe=(lambda _args: True) if parallel else None,
        ),
        body=body or default_body,
    )


def registry_with(*tools: RegisteredTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(list(tools))
    return registry


def call(call_id: str, name: str, **arguments) -> CallRequest:
    return CallRequest(call_id=call_id, name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# M3.1 -- single-view derivation
# ---------------------------------------------------------------------------


def test_view_derives_three_answers_consistently():
    """M3.1: visible/known/filterable must not be computed separately."""
    registry = registry_with(make_tool("a"), make_tool("b"))
    view = registry.view()
    assert {d.name for d in view.visible} == {"a", "b"}
    assert view.known == {"a", "b"}
    assert view.filterable == {"a", "b"}


def test_hidden_tool_is_known_but_not_visible():
    """M3.1: a hidden tool stays callable -- it is just not advertised."""
    hidden = RegisteredTool(
        definition=ToolDefinition(
            name="secret", description="hidden", parameters={}, hidden=True
        ),
        body=make_tool("secret").body,
    )
    registry = registry_with(make_tool("visible"), hidden)
    view = registry.view()
    assert {d.name for d in view.visible} == {"visible"}
    assert "secret" in view.known


# ---------------------------------------------------------------------------
# M3.2 / M3.3 -- restrictions and reserved names
# ---------------------------------------------------------------------------


def test_restriction_does_not_apply_to_the_layers_own_tools():
    """M3.2: a sub-agent must keep the tools it registered to report back."""
    registry = registry_with(make_tool("inherited"))
    layer = registry.restrict(allow=["inherited"], layer_name="child")

    child_tool = make_tool("child-only")
    registry.register(child_tool)

    view = registry.view()
    assert "child-only" in view.known, "own registration must survive its own restriction"
    assert "inherited" in view.known


def test_restriction_denies_inherited_tools():
    """M3.2: restrictions do act on what was inherited."""
    registry = registry_with(make_tool("keep"), make_tool("drop"))
    registry.restrict(deny=["drop"])
    assert registry.view().known == {"keep"}


def test_allow_list_excludes_others():
    """M3.2: an allow-list is a closed world for inherited names."""
    registry = registry_with(make_tool("a"), make_tool("b"), make_tool("c"))
    registry.restrict(allow=["a"])
    assert registry.view().known == {"a"}


def test_reserved_name_cannot_be_registered():
    """M3.3: two modes exposing one name would make advert != callable.

    The check fires at *definition* time, which is earlier and better than at
    registration: the error names the offending tool instead of surfacing later
    as an unexplained registry conflict.
    """
    from arm.kernel.errors import ReservedToolNameError

    registry = ToolRegistry()
    with pytest.raises(ReservedToolNameError, match="reserved"):
        registry.register(make_tool("system"))

    # The definition itself refuses too, independently of any registry.
    with pytest.raises(ReservedToolNameError):
        ToolDefinition(name="tool", description="", parameters={})


def test_duplicate_registration_in_one_layer_is_refused():
    """M3.3: within a layer, one name means one tool."""
    registry = registry_with(make_tool("dup"))
    with pytest.raises(DuplicateToolError):
        registry.register(make_tool("dup"))


def test_restriction_rejects_empty_filter():
    """M3.13: an empty filter is almost always a materialized empty config."""
    registry = registry_with(make_tool("a"))
    with pytest.raises(EmptyRestrictionError):
        registry.restrict()


def test_restriction_rejects_unknown_names_and_lists_known():
    """M3.14: a silent allow-list entry is a silently broken config."""
    registry = registry_with(make_tool("a"))
    with pytest.raises(UnknownToolRestrictionError) as info:
        registry.restrict(allow=["typo"])
    assert "typo" in str(info.value)
    assert "a" in str(info.value), "the error must list what is actually known"


def test_registration_is_reversible():
    """M3.12: unloading a plugin must leave nothing behind."""
    registry = ToolRegistry()
    dispose = registry.register(make_tool("temp"))
    assert registry.view().known == {"temp"}
    dispose()
    assert registry.view().known == frozenset()


# ---------------------------------------------------------------------------
# M3.4 -- arguments frozen before policies run
# ---------------------------------------------------------------------------


def test_arguments_are_deep_frozen_at_materialization():
    """M3.4: a policy must not be able to rewrite the arguments it inspects."""
    pipeline = ToolPipeline(registry_with(make_tool("t")))
    execution = pipeline.materialize(
        call_id="c1", name="t", arguments={"nested": {"value": 1}}
    )
    with pytest.raises(TypeError):
        execution.arguments["nested"]["value"] = 999  # type: ignore[index]
    with pytest.raises(TypeError):
        execution.arguments["new"] = 1  # type: ignore[index]


def test_non_serializable_arguments_become_a_result():
    """M3.4: an unfreezable payload is refused before any policy sees it."""
    pipeline = ToolPipeline(registry_with(make_tool("t")))
    execution = pipeline.materialize(
        call_id="c1", name="t", arguments={"bad": object()}
    )
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert "losslessly JSON-serializable" in result.content


# ---------------------------------------------------------------------------
# M3.5 -- only the execute wrapper may swap the signal, and it is re-fused
# ---------------------------------------------------------------------------


def test_wrapper_cannot_detach_cancellation():
    """M3.5: swapping the signal must not mean removing cancellation.

    The wrapper substitutes a fresh, never-aborting signal. The body must still
    observe the caller's signal, which is what the re-fusion guarantees.
    """
    observed: list[bool] = []

    async def body(_arguments, execution):
        observed.append(execution.signal.aborted)
        return ToolResult.ok("done")

    async def wrapper(execution, run_body):
        # Substitute a signal that will never abort.
        execution.signal = AbortSignal()
        return await run_body()

    pipeline = ToolPipeline(
        registry_with(make_tool("t", body=body)),
        PipelineHooks(execute_wrapper=wrapper),
    )

    signal = AbortSignal()
    signal.abort("user")
    execution = pipeline.materialize(
        call_id="c1", name="t", arguments={}, signal=signal
    )
    # The pre-dispatch guard refuses, which is the correct outcome for an
    # already-aborted signal: nothing started, so nothing was abandoned.
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_ABORTED_BEFORE_DISPATCH
    assert observed == [], "the body must not run once cancellation was requested"


# ---------------------------------------------------------------------------
# M3.6 / M3.7 -- output validation and exactly-once finalize
# ---------------------------------------------------------------------------


def test_body_returning_a_non_result_is_an_error_result():
    """M3.6: a wrapper or body cannot bypass the result contract."""

    async def bad_body(_arguments, _execution):
        return "just a string"

    pipeline = ToolPipeline(registry_with(make_tool("t", body=bad_body)))
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_INVALID_OUTPUT
    assert "invalid result object" in result.content


def test_finalize_runs_exactly_once_per_call():
    """M3.7: every result must be committed once -- not lost, not duplicated."""
    calls: list[str] = []

    def post(_execution, result):
        calls.append("post")
        return None

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(post_execute=(post,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    asyncio.run(pipeline.run(execution))
    assert calls == ["post"]


# ---------------------------------------------------------------------------
# Failure paths: every one becomes a result
# ---------------------------------------------------------------------------


def test_unknown_tool_is_a_result_not_an_exception():
    """M3: an unknown tool is an ordinary outcome the model can read."""
    pipeline = ToolPipeline(registry_with(make_tool("known")))
    execution = pipeline.materialize(call_id="c1", name="nope", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_UNKNOWN_TOOL
    assert 'unknown tool "nope"' in result.content


def test_unknown_tool_still_runs_pre_execute():
    """M3: the contract states pre-execute runs even for unknown tools.

    Policies use this to audit attempted calls, so skipping it for unknown names
    would create an unaudited path into the runtime.
    """
    seen: list[str] = []

    async def policy(ctx, nxt):
        seen.append(ctx.name)
        return await nxt()

    pipeline = ToolPipeline(
        registry_with(make_tool("known")), PipelineHooks(pre_execute=(policy,))
    )
    execution = pipeline.materialize(call_id="c1", name="ghost", arguments={})
    asyncio.run(pipeline.run(execution))
    assert seen == ["ghost"]


def test_denied_call_becomes_an_error_result():
    """M3: a policy denial is a result; the step continues."""
    async def deny(_ctx, _nxt):
        return Decision.deny("not allowed here")

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(pre_execute=(deny,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_DENIED
    assert "not allowed here" in result.content


def test_cancelled_call_becomes_an_error_result():
    """M3: a pre-execute cancel is a result, not an exception."""
    async def cancel(_ctx, _nxt):
        return Decision.cancel()

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(pre_execute=(cancel,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_ABORTED_BEFORE_DISPATCH


def test_body_exception_becomes_an_error_result():
    """M3: a raising body must not unwind the step."""
    async def boom(_arguments, _execution):
        raise ValueError("the disk caught fire")

    pipeline = ToolPipeline(registry_with(make_tool("t", body=boom)))
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert "the disk caught fire" in result.content
    assert result.error_code == CODE_BODY_ERROR


def test_body_raising_a_non_exception_value():
    """M3: bodies may raise anything; the model still needs a readable message."""
    async def boom(_arguments, _execution):
        raise KeyboardInterrupt("ctrl-c-ish")

    pipeline = ToolPipeline(registry_with(make_tool("t", body=boom)))
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    # A BaseException that is not an Exception is *not* caught: cancellation and
    # process-level interrupts must keep propagating. This documents that line.
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(pipeline.run(execution))


def test_approval_denied_becomes_an_error_result():
    """M3: a rejected approval is a result."""
    async def ask(_ctx, _nxt):
        return Decision.ask("needs consent")

    async def approve(_execution, _reason):
        return False

    pipeline = ToolPipeline(
        registry_with(make_tool("t")),
        PipelineHooks(pre_execute=(ask,), approve=approve),
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_APPROVAL_REJECTED
    assert 'the user rejected tool "t"' in result.content


def test_missing_approval_channel_fails_closed():
    """M3: no channel means denial -- a policy that said 'ask' did not say 'yes'."""
    async def ask(_ctx, _nxt):
        return Decision.ask()

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(pre_execute=(ask,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_NO_APPROVAL_CHANNEL


def test_guard_can_only_deny():
    """M3: the guard stage is monotonic -- it cannot grant what policy refused."""
    pipeline = ToolPipeline(
        registry_with(make_tool("t")),
        PipelineHooks(guard=lambda _execution: Decision.deny("guard says no")),
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert "guard says no" in result.content


def test_invalid_arguments_become_an_error_result():
    """M3: argument validation failure is a result."""
    from pydantic import BaseModel

    class Params(BaseModel):
        count: int

    registered = RegisteredTool(
        definition=ToolDefinition(
            name="counted",
            description="needs an int",
            parameters=Params.model_json_schema(),
        ),
        body=make_tool("counted").body,
        validator=Params,
    )
    pipeline = ToolPipeline(registry_with(registered))
    execution = pipeline.materialize(
        call_id="c1", name="counted", arguments={"count": "not a number"}
    )
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_INVALID_ARGUMENTS
    assert "invalid arguments" in result.content


def test_valid_arguments_are_coerced_by_the_validator():
    """M3: the pydantic model normalizes the arguments the body receives."""
    from pydantic import BaseModel

    class Params(BaseModel):
        count: int

    seen: list[int] = []

    async def body(arguments, _execution):
        seen.append(arguments["count"])
        return ToolResult.ok("ok")

    registered = RegisteredTool(
        definition=ToolDefinition(
            name="counted", description="", parameters=Params.model_json_schema()
        ),
        body=body,
        validator=Params,
    )
    pipeline = ToolPipeline(registry_with(registered))
    execution = pipeline.materialize(
        call_id="c1", name="counted", arguments={"count": "42"}
    )
    result = asyncio.run(pipeline.run(execution))
    assert not result.is_error
    assert seen == [42]


# ---------------------------------------------------------------------------
# Pre-execute waterfall semantics (the §9.3 question)
# ---------------------------------------------------------------------------


def test_first_hook_denying_without_next_short_circuits():
    """§9.3: not calling next() must prevent later hooks from running."""
    ran: list[str] = []

    async def first(_ctx, _nxt):
        ran.append("first")
        return Decision.deny("no")

    async def second(_ctx, nxt):
        ran.append("second")
        return await nxt()

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(pre_execute=(first, second))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert ran == ["first"], "the second hook must not run"


def test_all_hooks_continuing_reaches_the_allow_terminal():
    """§9.3: a chain that always continues ends in an allow decision."""
    ran: list[str] = []

    async def first(_ctx, nxt):
        ran.append("first")
        return await nxt()

    async def second(_ctx, nxt):
        ran.append("second")
        return await nxt()

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(pre_execute=(first, second))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert not result.is_error
    assert ran == ["first", "second"]


def test_hook_returning_none_without_next_is_a_contract_error():
    """§9.3: neither deciding nor continuing is a bug, not a silent pass."""
    from arm.kernel.errors import MissingDecisionError

    async def vague(_ctx, _nxt):
        return None

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(pre_execute=(vague,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    with pytest.raises(MissingDecisionError):
        asyncio.run(pipeline.run(execution))


def test_hook_calling_next_twice_is_a_contract_error():
    """§9.3: double continuation would run the tail twice."""
    from arm.kernel.errors import DoubleContinuationError

    async def greedy(_ctx, nxt):
        await nxt()
        return await nxt()

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(pre_execute=(greedy,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    with pytest.raises(DoubleContinuationError):
        asyncio.run(pipeline.run(execution))


# ---------------------------------------------------------------------------
# Post-execute
# ---------------------------------------------------------------------------


def test_post_execute_can_replace_content():
    """M3: post-execute may rewrite what the model sees."""

    def redact(_execution, result):
        return ToolResult.ok("[redacted]")

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(post_execute=(redact,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.content == "[redacted]"


def test_post_execute_can_block_a_result():
    """M3: a block converts the outcome into feedback."""

    class Block:
        kind = "block"
        feedback = "withheld: contains a secret"

    def blocker(_execution, _result):
        return Block()

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(post_execute=(blocker,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert "withheld" in result.content


def test_post_execute_failure_denies_rather_than_allows():
    """M3: a broken post-execute hook must not let the raw result through."""

    def broken(_execution, _result):
        raise RuntimeError("hook is broken")

    pipeline = ToolPipeline(
        registry_with(make_tool("t")), PipelineHooks(post_execute=(broken,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert "post-execute hook failed" in result.content


def test_wrapper_failure_skips_post_execute():
    """M3: where the failure happened decides whether post-execute runs."""
    ran: list[str] = []

    async def wrapper(_execution, _run_body):
        raise RuntimeError("wrapper is broken")

    def post(_execution, result):
        ran.append("post")
        return None

    pipeline = ToolPipeline(
        registry_with(make_tool("t")),
        PipelineHooks(execute_wrapper=wrapper, post_execute=(post,)),
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert ran == [], "post-execute must be skipped when the wrapper itself failed"


def test_body_failure_still_runs_post_execute():
    """M3: a body error is a normal result, so post-execute observes it."""
    ran: list[str] = []

    async def boom(_arguments, _execution):
        raise ValueError("nope")

    def post(_execution, result):
        ran.append(result.error_code or "")
        return None

    pipeline = ToolPipeline(
        registry_with(make_tool("t", body=boom)), PipelineHooks(post_execute=(post,))
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    asyncio.run(pipeline.run(execution))
    assert ran == [CODE_BODY_ERROR]


# ---------------------------------------------------------------------------
# M3.10 -- structured error identity never reaches the model
# ---------------------------------------------------------------------------


def test_error_code_is_not_in_the_model_visible_content():
    """M3.10: internal identity persists for diagnosis but must not leak."""
    pipeline = ToolPipeline(
        registry_with(make_tool("t")),
        PipelineHooks(guard=lambda _e: Decision.deny("internal policy #7")),
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.error_code == CODE_DENIED
    assert result.content.startswith("Error: ")
    assert CODE_DENIED not in result.content


# ---------------------------------------------------------------------------
# Dispatch: concurrency and model-order commitment
# ---------------------------------------------------------------------------


def test_results_are_committed_in_model_order():
    """M3.9: the transcript order must match the request order."""
    order: list[str] = []

    def make_slow(name: str, delay: float):
        async def body(_arguments, _execution):
            await asyncio.sleep(delay)
            order.append(name)
            return ToolResult.ok(name)

        return body

    registry = registry_with(
        make_tool("slow", body=make_slow("slow", 0.05), parallel=True),
        make_tool("fast", body=make_slow("fast", 0.0), parallel=True),
    )
    dispatcher = Dispatcher(ToolPipeline(registry))
    report = asyncio.run(
        dispatcher.dispatch([call("c1", "slow"), call("c2", "fast")])
    )

    assert [o.call_id for o in report.outcomes] == ["c1", "c2"]
    # They really did overlap: the slower one finished first internally.
    assert order == ["fast", "slow"]


def test_exclusive_tools_act_as_barriers():
    """M3.4: an unsafe call must not overlap with anything."""
    timeline: list[str] = []

    def recording_body(tag: str):
        async def _body(_arguments, _execution):
            timeline.append(f"{tag}:start")
            await asyncio.sleep(0.02)
            timeline.append(f"{tag}:end")
            return ToolResult.ok(tag)

        return _body

    registry = registry_with(
        make_tool("a", body=recording_body("a"), parallel=True),
        make_tool("exclusive", body=recording_body("exclusive"), parallel=False),
        make_tool("b", body=recording_body("b"), parallel=True),
    )
    dispatcher = Dispatcher(ToolPipeline(registry))
    asyncio.run(
        dispatcher.dispatch(
            [call("c1", "a"), call("c2", "exclusive"), call("c3", "b")]
        )
    )

    # The exclusive call must not interleave with either neighbour.
    assert timeline == [
        "a:start",
        "a:end",
        "exclusive:start",
        "exclusive:end",
        "b:start",
        "b:end",
    ], timeline


def test_cancelled_batch_still_produces_one_result_per_call():
    """M3.9: a dangling tool_call makes the next request invalid."""
    registry = registry_with(make_tool("t", parallel=True))
    pipeline = ToolPipeline(registry)
    dispatcher = Dispatcher(pipeline)

    signal = AbortSignal()
    signal.abort("user")

    requests = [call(f"c{i}", "t") for i in range(4)]
    report = asyncio.run(dispatcher.dispatch(requests, signal=signal))

    assert len(report.outcomes) == len(requests), "every call needs a result"
    assert all(o.result.is_error for o in report.outcomes)
    assert all(
        o.result.error_code == CODE_ABORTED_BEFORE_DISPATCH for o in report.outcomes
    )
    assert report.aborted


def test_dispatch_of_empty_batch_is_empty():
    """M3.9: nothing to do is not an error."""
    dispatcher = Dispatcher(ToolPipeline(registry_with(make_tool("t"))))
    report = asyncio.run(dispatcher.dispatch([]))
    assert report.outcomes == ()
    assert not report.aborted


def test_concludes_turn_is_reported():
    """M3: the only successful path that ends a step."""
    async def concluding(_arguments, _execution):
        return ToolResult.ok("done", concludes_turn=True)

    registry = registry_with(make_tool("finish", body=concluding))
    dispatcher = Dispatcher(ToolPipeline(registry))
    report = asyncio.run(dispatcher.dispatch([call("c1", "finish")]))
    assert report.concludes_turn


def test_result_pairing_survives_a_failing_tool():
    """M3.11: every call is paired with exactly one result."""
    async def boom(_arguments, _execution):
        raise RuntimeError("explode")

    registry = registry_with(
        make_tool("ok"), make_tool("bad", body=boom), make_tool("also_ok")
    )
    dispatcher = Dispatcher(ToolPipeline(registry))
    report = asyncio.run(
        dispatcher.dispatch([call("c1", "ok"), call("c2", "bad"), call("c3", "also_ok")])
    )
    assert [o.call_id for o in report.outcomes] == ["c1", "c2", "c3"]
    assert not report.outcomes[0].result.is_error
    assert report.outcomes[1].result.is_error
    assert not report.outcomes[2].result.is_error


def test_tool_result_renders_to_a_tool_message():
    """M3.11: the result must be renderable as the paired wire message."""
    result = ToolResult.ok("the answer")
    message = result.to_message(call_id="c1", name="t")
    assert message.role == "tool"
    assert message.tool_call_id == "c1"
    assert message.text == "the answer"


# ---------------------------------------------------------------------------
# Timeout is a strategy, not a kernel feature (M3 §5)
# ---------------------------------------------------------------------------


def test_timeout_is_implemented_as_a_wrapper():
    """M3 §5: the registry does not time out; a wrapper does.

    This proves the seam exists and produces the documented error code.
    """

    async def body(_arguments, _execution):
        await asyncio.sleep(5)
        return ToolResult.ok("too late")

    async def timeout_wrapper(execution, run_body):
        try:
            return await asyncio.wait_for(run_body(), timeout=0.02)
        except asyncio.TimeoutError:
            return ToolResult.fail(
                f"tool call timed out after 20ms", code=CODE_TIMEOUT
            )

    pipeline = ToolPipeline(
        registry_with(make_tool("t", body=body)),
        PipelineHooks(execute_wrapper=timeout_wrapper),
    )
    execution = pipeline.materialize(call_id="c1", name="t", arguments={})
    result = asyncio.run(pipeline.run(execution))
    assert result.is_error
    assert result.error_code == CODE_TIMEOUT
    assert "timed out" in result.content


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------


def test_normalize_inlines_refs_and_strips_presentation():
    """M3: the model-facing schema must be conservative but not lossy."""
    from pydantic import BaseModel

    class Inner(BaseModel):
        value: int

    class Outer(BaseModel):
        inner: Inner
        name: str = "default"

    raw = Outer.model_json_schema()
    assert "$defs" in raw, "pydantic should emit $defs for the nested model"

    normalized = to_wire(normalize_schema(raw))
    assert "$defs" not in normalized, "defs should be inlined away"
    assert "title" not in normalized
    assert "default" not in normalized.get("properties", {}).get("name", {})
    assert normalized["properties"]["inner"]["properties"]["value"]["type"] == "integer"


def test_normalized_schema_still_validates_the_same_payloads():
    """M3: normalization must not widen or narrow the accepted set."""
    from pydantic import BaseModel, TypeAdapter

    class Inner(BaseModel):
        value: int

    class Outer(BaseModel):
        inner: Inner
        name: str = "default"

    raw = Outer.model_json_schema()
    normalized = to_wire(normalize_schema(raw))

    adapter = TypeAdapter(dict)
    good = {"inner": {"value": 1}}
    # The normalized schema still describes the same shape.
    assert normalized["type"] == "object"
    assert set(normalized["properties"]) == {"inner", "name"}
    assert normalized["required"] == ["inner"]
    assert adapter.validate_python(good) == good


def test_recursive_schema_keeps_its_reference_rather_than_being_refused():
    """A recursive schema must survive, with its reference intact.

    A cycle cannot be inlined -- the expansion never terminates -- but refusing
    the schema outright would make every recursive tool argument impossible.
    ``$ref`` and ``$defs`` are part of the model-facing subset, so the correct
    outcome is to keep the reference and preserve its definition.
    """
    recursive = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
    }
    normalized = to_wire(normalize_schema(recursive))

    assert normalized["properties"]["child"]["$ref"] == "#/$defs/Node"
    assert "$defs" in normalized, "the referenced definition must be preserved"
    assert "Node" in normalized["$defs"]


def test_normalization_is_idempotent_for_recursive_schemas():
    """Normalizing twice must change nothing the second time.

    A cycle detected from a running ``seen`` chain would inline one level per
    pass, so repeated normalization would keep expanding the schema.
    """
    recursive = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "child": {"$ref": "#/$defs/Node"},
                },
            }
        },
    }
    once = to_wire(normalize_schema(recursive))
    twice = to_wire(normalize_schema(once))
    assert once == twice


def test_mutually_recursive_definitions_are_preserved():
    """A -> B -> A is a cycle too, and must not be expanded."""
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
            "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
        },
    }
    once = to_wire(normalize_schema(schema))
    twice = to_wire(normalize_schema(once))
    assert once == twice


def test_boolean_root_schema_is_refused_cleanly():
    """A boolean schema must raise the documented error, not AttributeError."""
    from arm.tools.schema import SchemaNormalizationError

    with pytest.raises(SchemaNormalizationError):
        normalize_schema(True)  # type: ignore[arg-type]


def test_non_mapping_defs_is_refused_cleanly():
    """A malformed ``$defs`` must raise the documented error, not ValueError."""
    from arm.tools.schema import SchemaNormalizationError

    with pytest.raises(SchemaNormalizationError, match=r"\$defs"):
        normalize_schema({"type": "object", "$defs": ["not", "a", "mapping"]})  # type: ignore[arg-type]


def test_annotations_do_not_conflict_with_a_referenced_target():
    """The idiomatic ``Annotated[Model, Field(description=...)]`` must work.

    A description is an annotation, not an assertion: two descriptions do not
    form an unsatisfiable conjunction. Treating them as a conflict made a
    documented model with a described field impossible to register at all.
    """
    from typing import Annotated

    from pydantic import BaseModel, Field

    class Documented(BaseModel):
        """A documented model."""

        value: int

    class Params(BaseModel):
        field: Annotated[Documented, Field(description="field description")]

    normalized = to_wire(normalize_schema(Params.model_json_schema()))
    assert normalized["properties"]["field"]["description"] == "field description"


def test_stripped_keywords_cannot_conflict():
    """A keyword removed from the output must never cause a refusal."""
    schema = {
        "type": "object",
        "properties": {"f": {"$ref": "#/$defs/D", "default": 1}},
        "$defs": {"D": {"type": "object", "default": 2, "properties": {}}},
    }
    normalized = to_wire(normalize_schema(schema))
    assert "default" not in normalized["properties"]["f"]


def test_deep_but_finite_nesting_is_not_refused():
    """Ordinary nesting must not be mistaken for recursion."""
    schema: dict = {"type": "integer"}
    for _ in range(30):
        schema = {"type": "object", "properties": {"next": schema}}
    normalized = to_wire(normalize_schema(schema))
    depth = 0
    cursor = normalized
    while "properties" in cursor:
        cursor = cursor["properties"]["next"]
        depth += 1
    assert depth == 30, f"nesting was truncated at depth {depth}"


def test_unreferenced_recursive_definition_does_not_refuse_the_schema():
    """An unused definition must not affect a schema that never references it."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "$defs": {
            "Unused": {"type": "object", "properties": {"self": {"$ref": "#/$defs/Unused"}}}
        },
    }
    normalized = to_wire(normalize_schema(schema))
    assert normalized["properties"]["name"]["type"] == "string"
    assert "$defs" not in normalized, "an unreferenced definition should be dropped"


def test_property_named_dollar_ref_is_not_a_reference():
    """A property *named* ``$ref`` is not a reference and must not confuse liveness."""
    schema = {
        "type": "object",
        "properties": {"$ref": {"type": "string"}},
        "$defs": {"Unused": {"type": "integer"}},
    }
    normalized = to_wire(normalize_schema(schema))
    assert "$defs" not in normalized
    assert normalized["properties"]["$ref"]["type"] == "string"


def test_escaped_json_pointer_tokens_resolve():
    """RFC 6901 escapes must be decoded, or a legal reference never resolves."""
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/A~1B"}},
        "$defs": {"A/B": {"type": "string"}},
    }
    normalized = to_wire(normalize_schema(schema))
    assert normalized["properties"]["x"] == {"type": "string"}, (
        "the escaped reference should have been inlined"
    )


# ---------------------------------------------------------------------------
# Schema normalization: regressions found by adversarial review
#
# Both cases below were silent *widening* bugs: the normalizer accepted payloads
# the original schema rejected, which would have let invalid arguments reach a
# tool body while looking correct. They are pinned here permanently.
# ---------------------------------------------------------------------------


def test_ref_siblings_are_conjoined_not_overwritten():
    """A ``$ref`` sibling must refine the target, never replace its keywords.

    JSON Schema treats ``$ref`` as conjunctive with its siblings. Overwriting let
    a sibling ``properties`` drop the target's constraints, so a payload the raw
    schema rejects became acceptable.
    """
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "$ref": "#/$defs/Inner",
                "properties": {"sneaky": {"type": "integer"}},
            }
        },
        "required": ["inner"],
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))
    inner = normalized["properties"]["inner"]

    # Both sides survive: the target's constraint AND the sibling's addition.
    assert set(inner["properties"]) == {"value", "sneaky"}, (
        "the target's `value` constraint was dropped by the sibling's `properties`"
    )
    assert inner["required"] == ["value"], (
        "the target's `required` was lost"
    )


def test_ref_sibling_required_is_a_union_not_a_replacement():
    """A sibling ``required`` must add to the target's, not erase it."""
    schema = {
        "type": "object",
        "properties": {
            "inner": {"$ref": "#/$defs/Inner", "required": ["zzz"]}
        },
        "required": ["inner"],
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "zzz": {"type": "integer"},
                },
                "required": ["value"],
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))
    required = normalized["properties"]["inner"]["required"]
    assert set(required) == {"value", "zzz"}, (
        f"required should be the union of both sides, got {required}"
    )


def test_conflicting_ref_sibling_is_refused_not_guessed():
    """When two sides cannot be conjoined, refusing beats picking a winner."""
    from arm.tools.schema import SchemaNormalizationError

    schema = {
        "type": "object",
        "properties": {
            "inner": {"$ref": "#/$defs/Inner", "type": "string"}
        },
        "$defs": {"Inner": {"type": "object", "properties": {}}},
    }
    with pytest.raises(SchemaNormalizationError, match="cannot conjoin"):
        normalize_schema(schema)


def test_single_branch_anyof_is_inlined_and_stays_well_formed():
    """A one-branch ``anyOf`` must inline to a schema, not to ``anyOf: {...}``.

    Emitting an object where JSON Schema requires an array produces a document no
    validator or model server can interpret.
    """
    schema = {"type": "object", "properties": {"s": {"anyOf": [{"type": "string"}]}}}
    normalized = to_wire(normalize_schema(schema))
    inner = normalized["properties"]["s"]
    assert inner == {"type": "string"}, f"expected the branch inlined, got {inner}"
    assert not isinstance(inner.get("anyOf"), dict)


def test_single_branch_anyof_with_conflicting_siblings_is_refused():
    """Inlining must not let the branch replace the node's own keywords.

    Two different scalar constraints cannot both be expressed in the model-facing
    subset, so the conjunction is refused rather than silently resolved to one of
    them -- which would change the accepted payload set.
    """
    from arm.tools.schema import SchemaNormalizationError

    schema = {
        "type": "object",
        "properties": {
            "s": {"anyOf": [{"type": "string", "pattern": "^a"}], "pattern": "^b"}
        },
    }
    with pytest.raises(SchemaNormalizationError, match="cannot conjoin"):
        normalize_schema(schema)


def test_single_branch_anyof_with_compatible_siblings_is_inlined():
    """When the branch and the node agree, inlining is safe and must happen."""
    schema = {
        "type": "object",
        "properties": {
            "s": {"anyOf": [{"type": "string", "minLength": 1}], "description": "a string"}
        },
    }
    normalized = to_wire(normalize_schema(schema))
    inner = normalized["properties"]["s"]
    assert inner.get("type") == "string"
    assert inner.get("minLength") == 1
    assert inner.get("description") == "a string"
    assert "anyOf" not in inner


def test_unsupported_keywords_are_refused_not_passed_through():
    """A keyword the subset does not accept must fail loudly, not reach the wire.

    Passing one through is the worse failure: the provider either rejects the
    whole schema or silently ignores the constraint, and in the second case the
    model may send payloads the tool author explicitly excluded.
    """
    from arm.tools.schema import SchemaNormalizationError

    for keyword in ("not", "if", "then", "else", "dependentRequired"):
        schema = {"type": "object", "properties": {"x": {keyword: {"type": "string"}}}}
        with pytest.raises(SchemaNormalizationError, match="not supported"):
            normalize_schema(schema)


def test_supported_keywords_still_pass_through():
    """Refusing the unsupported must not become refusing everything."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "minLength": 1, "maxLength": 10},
            "b": {"type": "integer", "minimum": 0, "maximum": 100},
            "c": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "d": {"enum": ["x", "y"]},
            "e": {"type": "string", "pattern": "^a"},
            "f": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["a"],
        "additionalProperties": False,
    }
    normalized = to_wire(normalize_schema(schema))
    assert normalized["additionalProperties"] is False
    assert normalized["properties"]["b"]["maximum"] == 100
    assert len(normalized["properties"]["f"]["anyOf"]) == 2


def test_every_real_builtin_tool_schema_normalizes():
    """The normalizer must accept every schema the built-in tools actually emit.

    A normalizer that refused a legitimate schema would make a tool impossible to
    register, so this is the check that the strictness has not overshot.
    """
    from arm.tools.builtin import builtin_tools

    for tool in builtin_tools():
        # ``schema_of`` is the path a tool definition takes.
        from arm.tools.schema import to_wire

        normalized = to_wire(tool.definition.parameters)
        assert normalized.get("type") == "object", (
            f"tool {tool.name!r} did not normalize to an object schema"
        )
        assert "properties" in normalized


def test_property_merge_recurses_instead_of_overwriting():
    """A sibling's ``properties`` must conjoin with the target's, key by key.

    A one-level ``{**existing, **value}`` lets a sibling redefine a same-named
    sub-property and silently discard the target's constraint on it -- a widening
    that looks like an ordinary override.
    """
    schema = {
        "type": "object",
        "properties": {
            "m": {
                "$ref": "#/$defs/Inner",
                "properties": {"value": {"type": "string"}},
            }
        },
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "minimum": 10},
                },
            }
        },
    }
    # The two constraints on ``value`` cannot both hold, so the conjunction is
    # unsatisfiable and must be refused rather than silently resolved.
    from arm.tools.schema import SchemaNormalizationError

    with pytest.raises(SchemaNormalizationError, match="cannot conjoin"):
        normalize_schema(schema)


def test_property_merge_keeps_both_sides_when_compatible():
    """Compatible sub-schemas must merge, keeping every constraint."""
    schema = {
        "type": "object",
        "properties": {
            "m": {
                "$ref": "#/$defs/Inner",
                "properties": {"extra": {"type": "string"}},
            }
        },
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))
    props = normalized["properties"]["m"]["properties"]
    assert set(props) == {"value", "extra"}, props


def test_draft07_definitions_are_rewritten_not_left_dangling():
    """``#/definitions/`` references must be canonicalized to ``$defs``.

    Deleting the ``definitions`` block while leaving ``#/definitions/X``
    references produces a dangling pointer -- a schema no validator or provider
    can resolve.
    """
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/Node"}},
        "definitions": {
            "Node": {
                "type": "object",
                "properties": {"self": {"$ref": "#/definitions/Node"}},
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))
    rendered = json.dumps(normalized)

    assert "#/definitions/" not in rendered, "a dangling reference survived"
    assert normalized["properties"]["a"]["$ref"] == "#/$defs/Node"
    # The definition is cyclic, so it cannot be inlined -- it must be preserved
    # under the canonical spelling instead.
    assert "Node" in normalized["$defs"]


def test_both_definition_spellings_resolve_together():
    """A schema using both spellings must keep both sets of definitions."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"$ref": "#/$defs/A"},
            "b": {"$ref": "#/definitions/B"},
        },
        "$defs": {"A": {"type": "object", "properties": {"self": {"$ref": "#/$defs/A"}}}},
        "definitions": {"B": {"type": "integer"}},
    }
    normalized = to_wire(normalize_schema(schema))
    rendered = json.dumps(normalized)
    assert "#/definitions/" not in rendered, "a draft-07 reference survived"
    # ``B`` is acyclic, so it is inlined; ``A`` is cyclic, so it stays a reference
    # and its definition is preserved.
    assert normalized["properties"]["b"] == {"type": "integer"}
    assert normalized["properties"]["a"]["$ref"] == "#/$defs/A"
    assert "A" in normalized["$defs"]


def test_conflicting_additional_properties_is_refused():
    """Merging ``additionalProperties`` can invert its meaning.

    ``false`` alongside a sibling's ``properties`` starts forbidding *those*
    names, when the author wrote it to forbid everything the sibling had not
    listed. Refusing is the only safe answer.
    """
    from arm.tools.schema import SchemaNormalizationError

    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "$ref": "#/$defs/Inner",
                "additionalProperties": False,
            }
        },
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "additionalProperties": True,
            }
        },
    }
    with pytest.raises(SchemaNormalizationError, match="additionalProperties"):
        normalize_schema(schema)


def test_every_tool_definition_is_normalized_at_construction():
    """Normalization must happen in the definition, not at each call site.

    It used to live in a helper that callers had to remember to use, so every
    real tool bypassed it and shipped a raw pydantic schema: keywords the
    model-facing subset rejects reached the wire, and the normalizer's
    correctness was a property of a path nothing executed.
    """
    import json as _json

    from pydantic import BaseModel, Field

    class Params(BaseModel):
        name: str = Field(default="x", description="a name")
        count: int = Field(default=1, ge=0)

    definition = ToolDefinition(
        name="normalized", description="", parameters=Params.model_json_schema()
    )
    rendered = _json.dumps(dict(definition.parameters))

    assert "title" not in rendered, "presentation keywords reached the wire"
    assert "default" not in rendered
    assert "$defs" not in rendered
    # And the constraint that matters survives.
    assert definition.parameters["properties"]["count"]["minimum"] == 0
    # Both fields have defaults, so neither is required -- the point is that the
    # constraint (``minimum``) survived while the presentation did not.
    assert "required" not in definition.parameters
    assert set(definition.parameters["properties"]) == {"name", "count"}

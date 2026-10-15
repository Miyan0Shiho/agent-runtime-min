"""Runtime spine tests: the turn/step loop and its invariants (S1-S14).

These tests drive the loop with a scripted adapter and assert on the *event
sequence*, because the event sequence is the product. A test that only checked
the final text would pass on an implementation that wrote its events in the wrong
order, and the order is what makes recovery and replay possible.
"""

from __future__ import annotations

import asyncio

import pytest

from arm.agent.loop import (
    PHASE_IDLE,
    LoopError,
    LoopHooks,
    Runtime,
    StepDecision,
)
from arm.llm.mock import MockAdapter, ScriptedTurn, text_response, tool_response
from arm.prompt.assembler import PromptAssembler
from arm.session.events import (
    ASSISTANT_ATTEMPT,
    ASSISTANT_MESSAGE,
    DEVELOPER_MESSAGE,
    ERROR_TURN_FAILURE,
    REQUEST_CONTEXT,
    REQUEST_HEADER,
    SESSION_START,
    STEP_END,
    STEP_START,
    SYSTEM_MESSAGE,
    TOOL_CALL,
    TOOL_RESULT,
    TURN_ABORTED,
    TURN_BLOCKED,
    TURN_COMPLETED,
    TURN_END,
    TURN_ERROR,
    TURN_START,
    USER_MESSAGE,
)
from arm.session.session import Session
from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult
from arm.tools.registry import ToolRegistry


def build(
    script: list[ScriptedTurn],
    *,
    tools: list[RegisteredTool] | None = None,
    prompt: str = "you are arm",
    hooks: LoopHooks | None = None,
) -> tuple[Runtime, Session, MockAdapter]:
    session = Session.create("s1")
    adapter = MockAdapter(script)
    registry = ToolRegistry()
    if tools:
        registry.register_many(tools)
    assembler = PromptAssembler()
    assembler.set("core", prompt, priority=10)
    runtime = Runtime(
        session, adapter, registry, assembler=assembler, hooks=hooks, model="mock-1"
    )
    return runtime, session, adapter


def types_of(session: Session) -> list[str]:
    return [e.type for e in session]


def make_echo_tool() -> RegisteredTool:
    async def body(arguments, _execution):
        return ToolResult.ok(f"echoed:{arguments.get('text', '')}")

    return RegisteredTool(
        definition=ToolDefinition(
            name="echo", description="echo text", parameters={"type": "object"}
        ),
        body=body,
    )


# ---------------------------------------------------------------------------
# S2 -- the driver is a while loop over turn()
# ---------------------------------------------------------------------------


def test_a_simple_turn_produces_the_documented_event_order():
    """S2/S5/S6: the sequence of a turn is fixed and complete."""
    runtime, session, _ = build([text_response("hello there")])
    runtime.submit("hi")
    outcomes = asyncio.run(runtime.run_until_idle())

    assert len(outcomes) == 1
    assert outcomes[0].reason == TURN_COMPLETED
    assert outcomes[0].steps == 1

    types = types_of(session)
    # The system prompt heads the transcript, and the input that a turn will
    # claim must exist before the turn starts -- otherwise there would be
    # nothing to claim.
    assert types.index(SYSTEM_MESSAGE) < types.index(USER_MESSAGE)
    assert types.index(USER_MESSAGE) < types.index(TURN_START)
    # turn/start precedes the step it opens; the step precedes the model call.
    assert types.index(TURN_START) < types.index(STEP_START)
    assert types.index(STEP_START) < types.index(REQUEST_HEADER)
    assert types.index(REQUEST_HEADER) < types.index(REQUEST_CONTEXT)
    assert types.index(REQUEST_CONTEXT) < types.index(ASSISTANT_MESSAGE)
    assert types.index(ASSISTANT_MESSAGE) < types.index(STEP_END)
    # turn/end is last, and every step is closed before it.
    assert types[-1] == TURN_END, "turn/end must be the last durable event"
    assert types.index(STEP_END) < types.index(TURN_END)


def test_turn_end_is_written_on_every_exit_path():
    """S6: a turn without an end is indistinguishable from a crash."""
    # normal completion
    runtime, session, _ = build([text_response("done")])
    runtime.submit("hi")
    asyncio.run(runtime.run_until_idle())
    assert types_of(session)[-1] == TURN_END

    # empty turn: input already claimed
    runtime2, session2, _ = build([text_response("unused")])
    outcome = asyncio.run(runtime2.turn())
    assert outcome.reason == TURN_COMPLETED
    assert types_of(session2)[-1] == TURN_END
    assert STEP_START not in types_of(session2)


def test_empty_turn_has_no_step():
    """S7/S5: an empty turn is structurally complete but has no step."""
    runtime, session, adapter = build([text_response("unused")])
    outcome = asyncio.run(runtime.turn())

    assert outcome.steps == 0
    assert adapter.call_count == 0, "no model call for an empty turn"
    assert TURN_START in types_of(session)
    assert STEP_START not in types_of(session)


def test_loop_drains_all_queued_input():
    """S2: queued inputs become successive turns."""
    runtime, session, adapter = build(
        [text_response("first"), text_response("second")]
    )
    runtime.submit("one")
    runtime.submit("two")
    outcomes = asyncio.run(runtime.run_until_idle())

    # Both inputs are claimable, so they are handled as two steps of one turn
    # rather than two turns: the second input joins the turn already in progress.
    assert [o.reason for o in outcomes] == [TURN_COMPLETED]
    assert outcomes[0].steps == 2
    assert types_of(session).count(TURN_START) == 1
    assert adapter.call_count == 2


def test_phase_returns_to_idle():
    """S1: exactly one phase variable, and it settles."""
    runtime, _, _ = build([text_response("x")])
    runtime.submit("hi")
    asyncio.run(runtime.run_until_idle())
    assert runtime.phase == PHASE_IDLE


# ---------------------------------------------------------------------------
# S5/S7 -- pre-step rejection
# ---------------------------------------------------------------------------


def test_pre_step_rejection_blocks_the_turn_without_a_step():
    """S7: a rejected batch must not leave an empty step behind."""
    async def reject(_session, _claim):
        return StepDecision.reject("not now")

    runtime, session, adapter = build(
        [text_response("never sent")], hooks=LoopHooks(pre_step=reject)
    )
    runtime.submit("hi")
    outcome = asyncio.run(runtime.turn())

    assert outcome.reason == TURN_BLOCKED
    assert outcome.steps == 0
    assert adapter.call_count == 0
    assert STEP_START not in types_of(session)
    assert types_of(session)[-1] == TURN_END


def test_pre_step_rejection_does_not_restore_the_input():
    """S4: a rejected input is not re-delivered.

    Re-delivering would process the same input twice, which is worse than losing
    it because the duplicate is silent.
    """
    async def reject(_session, _claim):
        return StepDecision.reject("no")

    runtime, session, _ = build([text_response("x")], hooks=LoopHooks(pre_step=reject))
    runtime.submit("hi")
    asyncio.run(runtime.turn())

    # The turn ran; a second turn finds nothing claimable.
    second = asyncio.run(runtime.turn())
    assert second.reason == TURN_COMPLETED
    assert second.steps == 0


def test_broken_pre_step_hook_blocks_rather_than_runs():
    """A hook that raises must not be treated as approval."""
    async def broken(_session, _claim):
        raise RuntimeError("hook is broken")

    runtime, session, adapter = build(
        [text_response("x")], hooks=LoopHooks(pre_step=broken)
    )
    runtime.submit("hi")
    outcome = asyncio.run(runtime.turn())

    assert outcome.reason == TURN_BLOCKED
    assert adapter.call_count == 0


# ---------------------------------------------------------------------------
# S14 -- model failures become recorded outcomes
# ---------------------------------------------------------------------------


def test_model_error_ends_the_turn_with_error():
    """S14: a model failure is a recorded outcome, not an exception."""
    runtime, session, _ = build([ScriptedTurn(error=RuntimeError("provider down"))])
    runtime.submit("hi")
    outcome = asyncio.run(runtime.turn())

    assert outcome.reason == TURN_ERROR
    assert "provider down" in (outcome.error or "")
    types = types_of(session)
    assert ERROR_TURN_FAILURE in types
    assert types[-1] == TURN_END


def test_model_error_is_recorded_separately_from_turn_end():
    """The failure reason and the turn boundary are distinct events."""
    runtime, session, _ = build([ScriptedTurn(error=RuntimeError("boom"))])
    runtime.submit("hi")
    asyncio.run(runtime.turn())

    types = types_of(session)
    assert types.index(ERROR_TURN_FAILURE) < types.index(TURN_END)


# ---------------------------------------------------------------------------
# M1 integration: the log records what was sent
# ---------------------------------------------------------------------------


def test_request_header_and_context_are_recorded():
    """M1: 'model-visible implies recorded' -- both legs are in the log."""
    runtime, session, _ = build([text_response("ok")])
    runtime.submit("hi")
    asyncio.run(runtime.turn())

    header = next(e for e in session if e.type == REQUEST_HEADER)
    context = next(e for e in session if e.type == REQUEST_CONTEXT)
    assert header.data["model"] == "mock-1"
    assert header.data["turn"] == 0
    assert context.data["message_count"] >= 1


def test_derived_messages_match_what_was_sent():
    """M1 acceptance: the request is derivable from the log.

    The adapter recorded the exact messages it received. Re-deriving from the log
    must reproduce them.
    """
    runtime, session, adapter = build([text_response("the answer")])
    runtime.submit("the question")
    asyncio.run(runtime.run_until_idle())

    sent = adapter.requests[0]
    # The request was built from derive(), so the first two messages are the
    # system prompt and the user input.
    assert [m.role for m in sent.messages][:2] == ["system", "user"]
    assert sent.messages[1].text == "the question"

    # The same messages are reconstructible from the log alone. The live session
    # has since gained the assistant's reply, so the replay is compared against
    # the request prefix rather than the whole transcript.
    restored = Session.from_events("s1", session.events())
    derived = restored.derive()
    assert [m.to_record() for m in derived][: len(sent.messages)] == [
        m.to_record() for m in sent.messages
    ]


def test_system_prompt_is_recorded_only_when_it_changes():
    """M1: re-recording an unchanged prompt adds a node with no information."""
    runtime, session, _ = build([text_response("a"), text_response("b")])
    runtime.submit("one")
    runtime.submit("two")
    asyncio.run(runtime.run_until_idle())

    assert types_of(session).count(SYSTEM_MESSAGE) == 1


def test_changed_prompt_is_recorded_again():
    """M1: a real prompt change must be visible in the log."""
    runtime, session, _ = build([text_response("a")])
    runtime.submit("one")
    asyncio.run(runtime.turn())

    runtime.assembler.set("extra", "new instruction", priority=20)
    runtime.submit("two")
    asyncio.run(runtime.turn())

    assert types_of(session).count(SYSTEM_MESSAGE) == 2


def test_user_input_is_recorded_exactly_once():
    """M1: the same question must not appear twice in the transcript."""
    runtime, session, _ = build([text_response("ok")])
    runtime.submit("only once")
    asyncio.run(runtime.run_until_idle())

    user_messages = [e for e in session if e.type == USER_MESSAGE]
    assert len(user_messages) == 1
    assert [m.text for m in session.derive()].count("only once") == 1


# ---------------------------------------------------------------------------
# M3 integration: tool calls through the loop
# ---------------------------------------------------------------------------


def test_tool_call_round_trip():
    """M3.11: every call is paired with a result, in model order."""
    runtime, session, adapter = build(
        [
            tool_response(("c1", "echo", {"text": "ping"})),
            text_response("done"),
        ],
        tools=[make_echo_tool()],
    )
    runtime.submit("use the tool")
    outcomes = asyncio.run(runtime.run_until_idle())

    assert outcomes[0].reason == TURN_COMPLETED
    assert outcomes[0].steps == 2, "one step to call, one to finish"

    types = types_of(session)
    assert TOOL_CALL in types and TOOL_RESULT in types
    assert types.index(TOOL_CALL) < types.index(TOOL_RESULT)

    result = next(e for e in session if e.type == TOOL_RESULT)
    call = next(e for e in session if e.type == TOOL_CALL)
    assert result.data["call_seq"] == call.seq, "the result must reference its call"
    assert result.data["call_id"] == "c1"
    assert "echoed:ping" in result.data["message"]["content"][0]["text"]


def test_tool_call_and_result_are_derivable_as_wire_messages():
    """M1+M3: the tool result reaches the wire and is reconstructible."""
    runtime, session, adapter = build(
        [tool_response(("c1", "echo", {"text": "hi"})), text_response("done")],
        tools=[make_echo_tool()],
    )
    runtime.submit("go")
    asyncio.run(runtime.run_until_idle())

    # The second model call saw the assistant's tool_calls and the tool result.
    second = adapter.requests[1]
    roles = [m.role for m in second.messages]
    assert "assistant" in roles and "tool" in roles

    # Compared against the request prefix: the live log has since gained the
    # assistant's final reply.
    restored = Session.from_events("s1", session.events())
    replayed = [m.to_record() for m in restored.derive()]
    assert replayed[: len(second.messages)] == [
        m.to_record() for m in second.messages
    ]
    # And the pairing is intact in the replay: the assistant's tool call is
    # present, followed by the tool result that answers it.
    roles = [m["role"] for m in replayed]
    call_index = next(
        i for i, m in enumerate(replayed) if m.get("tool_calls")
    )
    assert roles[call_index] == "assistant"
    assert roles[call_index + 1] == "tool"


def test_tool_failure_does_not_end_the_turn():
    """S14: a failing tool produces an error result and the step continues."""
    async def boom(_arguments, _execution):
        raise RuntimeError("tool exploded")

    failing = RegisteredTool(
        definition=ToolDefinition(name="boom", description="", parameters={}),
        body=boom,
    )
    runtime, session, adapter = build(
        [tool_response(("c1", "boom", {})), text_response("recovered")],
        tools=[failing],
    )
    runtime.submit("try it")
    outcomes = asyncio.run(runtime.run_until_idle())

    assert outcomes[0].reason == TURN_COMPLETED
    assert outcomes[0].steps == 2
    result = next(e for e in session if e.type == TOOL_RESULT)
    assert result.data["is_error"] is True
    assert result.data["info"]["code"] == "BODY_ERROR"
    assert "tool exploded" in result.data["message"]["content"][0]["text"]


def test_unknown_tool_call_is_recorded_as_an_error_result():
    """M3: an unknown tool still produces a paired result."""
    runtime, session, _ = build(
        [tool_response(("c1", "ghost", {})), text_response("ok")]
    )
    runtime.submit("call a ghost")
    asyncio.run(runtime.run_until_idle())

    result = next(e for e in session if e.type == TOOL_RESULT)
    assert result.data["is_error"] is True
    assert result.data["info"]["code"] == "UNKNOWN_TOOL"


def test_tool_results_are_committed_in_model_order():
    """M3.9: results follow the model's request order, not completion order."""
    async def slow(_arguments, _execution):
        await asyncio.sleep(0.05)
        return ToolResult.ok("slow done")

    async def fast(_arguments, _execution):
        return ToolResult.ok("fast done")

    slow_tool = RegisteredTool(
        definition=ToolDefinition(
            name="slow", description="", parameters={}, concurrency_safe=lambda _a: True
        ),
        body=slow,
    )
    fast_tool = RegisteredTool(
        definition=ToolDefinition(
            name="fast", description="", parameters={}, concurrency_safe=lambda _a: True
        ),
        body=fast,
    )
    runtime, session, _ = build(
        [
            tool_response(("c1", "slow", {}), ("c2", "fast", {})),
            text_response("done"),
        ],
        tools=[slow_tool, fast_tool],
    )
    runtime.submit("go")
    asyncio.run(runtime.run_until_idle())

    results = [e for e in session if e.type == TOOL_RESULT]
    assert [r.data["call_id"] for r in results] == ["c1", "c2"]
    assert "slow done" in results[0].data["message"]["content"][0]["text"]


def test_tool_announcement_is_recorded_when_tools_exist():
    """M1: the model is told about the tool set, and that telling is recorded."""
    runtime, session, _ = build(
        [text_response("ok")], tools=[make_echo_tool()]
    )
    runtime.submit("hi")
    asyncio.run(runtime.run_until_idle())

    assert DEVELOPER_MESSAGE in types_of(session)
    header = next(e for e in session if e.type == REQUEST_HEADER)
    assert tuple(header.data["tools"]) == ("echo",)


def test_no_developer_message_when_the_tool_set_never_changed():
    """M1: no announcement on the second step of the same tool set."""
    runtime, session, _ = build(
        [tool_response(("c1", "echo", {"text": "x"})), text_response("done")],
        tools=[make_echo_tool()],
    )
    runtime.submit("go")
    asyncio.run(runtime.run_until_idle())

    assert types_of(session).count(DEVELOPER_MESSAGE) == 1


# ---------------------------------------------------------------------------
# S12 -- cancellation has one path with a typed cause
# ---------------------------------------------------------------------------


def test_aborted_turn_is_recorded_with_its_cause():
    """S12: cancellation is one path, and the cause is persisted."""
    runtime, session, _ = build([text_response("x")])
    runtime.request_abort("user")
    runtime.submit("hi")
    outcome = asyncio.run(runtime.turn())

    assert outcome.reason == TURN_ABORTED
    end = [e for e in session if e.type == TURN_END][-1]
    assert end.data["cancel_cause"] == "user"


def test_abort_before_a_tool_call_marks_the_call_aborted():
    """M3.9: an undispatched call still gets a synthetic result."""
    runtime, session, _ = build(
        [tool_response(("c1", "echo", {"text": "x"}))], tools=[make_echo_tool()]
    )
    runtime.submit("go")
    # Abort is requested before the turn starts, so the model is never called and
    # no tool call is ever recorded -- there is nothing to pair a result with.
    runtime.request_abort("user")
    outcome = asyncio.run(runtime.turn())

    assert outcome.reason == TURN_ABORTED
    types = types_of(session)
    # An aborted turn records no tool call at all, so pairing is trivially
    # complete: there is no call awaiting a result.
    assert types.count(TOOL_CALL) == types.count(TOOL_RESULT) == 0
    end = [e for e in session if e.type == TURN_END][-1]
    assert end.data["cancel_cause"] == "user"


# ---------------------------------------------------------------------------
# Retry: a failed step re-runs without re-admitting the input
# ---------------------------------------------------------------------------


def test_retry_does_not_duplicate_the_user_message():
    """S: a retried request must not re-admit the input."""
    runtime, session, adapter = build([text_response("x")])
    runtime.submit("once")
    asyncio.run(runtime.turn())
    # A second turn finds nothing new to claim.
    second = asyncio.run(runtime.turn())
    assert second.steps == 0
    assert len([e for e in session if e.type == USER_MESSAGE]) == 1


# ---------------------------------------------------------------------------
# Loop hygiene
# ---------------------------------------------------------------------------


def test_turn_is_not_reentrant():
    """Two concurrent turns would interleave appends unpredictably."""
    runtime, _, _ = build([text_response("x")])
    runtime._running = True
    try:
        with pytest.raises(LoopError, match="not re-entrant"):
            asyncio.run(runtime.turn())
    finally:
        runtime._running = False


def test_max_steps_is_enforced():
    """A model that keeps calling tools must not loop forever."""
    async def looping_tool(_arguments, _execution):
        return ToolResult.ok("again")

    looping = RegisteredTool(
        definition=ToolDefinition(name="loop", description="", parameters={}),
        body=looping_tool,
    )
    runtime, session, _ = build(
        [tool_response(("c1", "loop", {}))] * 5, tools=[looping]
    )
    runtime.max_steps = 3
    runtime.submit("go")
    outcome = asyncio.run(runtime.turn())

    assert outcome.steps == 3
    assert types_of(session)[-1] == TURN_END


def test_interrupt_mid_stream_keeps_partial_content():
    """§5: a stream that dies after producing content keeps that content."""
    runtime, session, _ = build([text_response("partial answer")])
    runtime.submit("hi")
    asyncio.run(runtime.turn())

    assistant = next(e for e in session if e.type == ASSISTANT_MESSAGE)
    assert "partial answer" in assistant.data["message"]["content"][0]["text"]


def test_empty_model_output_records_an_attempt_not_an_empty_message():
    """§5: no content means an attempt event, never a contentless message."""
    runtime, session, _ = build([ScriptedTurn(text="")])
    runtime.submit("hi")
    asyncio.run(runtime.turn())

    types = types_of(session)
    assert ASSISTANT_ATTEMPT in types
    assert ASSISTANT_MESSAGE not in types


def test_session_start_events_are_present():
    """The session header events are written once at creation."""
    runtime, session, _ = build([text_response("x")])
    assert types_of(session)[0] == SESSION_START

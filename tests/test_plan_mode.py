"""Plan mode tests.

The governing claim under test: **plan mode changes the prompt, and nothing
else.** So the central tests here are negative ones -- entering plan mode must
not change the tool catalogue, and a tool that would be dangerous in plan mode
must remain *callable*, because the mode is a request about sequencing rather
than a restriction.

The second theme is the two-state split: ``active`` is in the log and survives a
restart; ``pending`` is in the process and is deliberately lost by one. Both
halves are tested, because a design that conflated them would either write events
outside a turn or lose the user's choice silently.
"""

from __future__ import annotations

import asyncio

from arm.llm.mock import MockAdapter, text_response
from arm.plan.mode import PLAN_POLICY, PlanMode, plan_tools
from arm.prompt.assembler import PromptAssembler
from arm.session.events import PLAN_MODE, REQUEST_HEADER
from arm.session.session import Session
from arm.tools.definition import ToolResult
from arm.tools.registry import ToolRegistry


async def invoke(tool, arguments: dict) -> ToolResult:
    """Call a tool body and return its result.

    A coroutine function so ``asyncio.run`` receives a coroutine rather than the
    looser ``Awaitable`` that the tool-body type permits.
    """
    return await tool.body(arguments, None)


def make_session() -> Session:
    return Session.create("plan")


def open_turn(session: Session, turn: int = 0) -> None:
    """Mark a turn as in flight, which is what licenses a commit."""
    session.append("turn/start", {"turn": turn})


def close_turn(session: Session, turn: int = 0) -> None:
    session.append("turn/end", {"turn": turn, "reason": "completed", "steps": 1})


# ---------------------------------------------------------------------------
# The committed state lives in the log
# ---------------------------------------------------------------------------


def test_starts_inactive():
    """A session that never mentioned plan mode is not in plan mode."""
    mode = PlanMode(make_session())
    assert mode.state().active is False
    assert mode.active is False
    assert mode.prompt_section() == ""


def test_a_change_inside_a_turn_is_committed_immediately():
    """An event must belong to a turn, so an open turn licenses a commit."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)

    assert mode.set(True) == "committed"
    assert mode.state().active is True
    assert mode.pending is False


def test_a_change_outside_a_turn_is_queued():
    """Writing outside a turn would produce a log recovery cannot interpret."""
    session = make_session()
    mode = PlanMode(session)

    assert mode.set(True) == "queued"
    assert mode.pending is True
    # Nothing was written yet.
    assert PLAN_MODE not in [e.type for e in session]


def test_a_queued_choice_commits_once_a_step_is_accepted():
    """The queue exists so a rejected step does not consume the choice."""
    session = make_session()
    mode = PlanMode(session)
    mode.set(True)

    open_turn(session)
    assert mode.commit_pending() is True
    assert mode.state().active is True
    assert mode.pending is False


def test_a_queued_choice_survives_a_rejected_step():
    """A step that never ran must not swallow the user's choice."""
    session = make_session()
    mode = PlanMode(session)
    mode.set(True)

    # A turn opens and closes with no accepted step: the choice is still queued.
    open_turn(session)
    close_turn(session)
    assert mode.commit_pending() is False, (
        "the choice was committed without an accepted step"
    )
    assert mode.pending is True


def test_the_effective_state_prefers_a_pending_choice():
    """The prompt should describe the step about to run, not the last one."""
    session = make_session()
    mode = PlanMode(session)
    mode.set(True)

    assert mode.active is True, "a pending choice must be reflected immediately"
    assert mode.state().active is False, "but the log is not written yet"


def test_setting_the_same_value_is_a_noop():
    """An idempotent call must not append an event."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)

    assert mode.set(False) == "noop"
    assert PLAN_MODE not in [e.type for e in session]


def test_the_last_event_wins():
    """The event is a whole-value replacement, so the most recent one counts."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)

    mode.set(True)
    mode.set(False)
    mode.set(True)
    assert mode.state().active is True
    assert [e.type for e in session].count(PLAN_MODE) == 3


def test_a_pending_choice_can_be_cancelled():
    """A user who changes their mind before the step runs must be able to."""
    session = make_session()
    mode = PlanMode(session)
    mode.set(True)
    assert mode.cancel_pending() is True
    assert mode.pending is False
    assert mode.active is False
    assert mode.cancel_pending() is False


# ---------------------------------------------------------------------------
# The state survives a restart, and the pending choice does not
# ---------------------------------------------------------------------------


def test_committed_state_survives_a_restart():
    """Recovery gets the mode for free, because it is in the log."""
    session = make_session()
    open_turn(session)
    PlanMode(session).set(True)

    restored = Session.from_events("plan", session.events())
    assert PlanMode(restored).state().active is True


def test_a_pending_choice_is_lost_by_a_restart():
    """Deliberately: an uncommitted choice has no representation in the log."""
    session = make_session()
    PlanMode(session).set(True)

    restored = Session.from_events("plan", session.events())
    assert PlanMode(restored).state().active is False
    assert PlanMode(restored).pending is False


# ---------------------------------------------------------------------------
# The prompt contribution
# ---------------------------------------------------------------------------


def test_the_prompt_section_appears_only_when_active():
    """An inactive mode contributes nothing, not an empty heading."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)

    assert mode.prompt_section() == ""
    mode.set(True)
    assert mode.prompt_section() == PLAN_POLICY
    assert "# Plan mode" in mode.prompt_section()


def test_the_policy_asks_for_sequencing_not_permission():
    """The text must not claim a tool is unavailable, because none is.

    A prompt that said "you cannot write files" would be a lie the model could
    discover by trying, and a discovered lie teaches the model to distrust the
    prompt.
    """
    lowered = PLAN_POLICY.lower()
    assert "not a restriction" in lowered
    assert "nothing will stop you" in lowered
    # And it must not claim a capability is missing.
    for forbidden in ("you cannot", "you are unable", "not allowed", "forbidden"):
        assert forbidden not in lowered, (
            f"the policy claims a restriction that does not exist: {forbidden!r}"
        )


def test_the_policy_requires_a_heading_so_a_plan_is_reviewable():
    """A plan without a heading reads as prose, and cannot be approved as one."""
    assert "# " in PLAN_POLICY


# ---------------------------------------------------------------------------
# The notification is idempotent
# ---------------------------------------------------------------------------


def test_a_change_is_announced_once():
    """Re-announcing the mode every step is noise the model learns to skip."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    mode.set(True)

    # The header records what the model was last told.
    session.append(REQUEST_HEADER, {"turn": 0, "step": 0, **mode.header_fields()})
    assert mode.notification() is None, "the mode was re-announced with no change"


def test_a_change_since_the_last_header_is_announced():
    """The notice exists precisely to mark a change."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)

    session.append(REQUEST_HEADER, {"turn": 0, "step": 0, **mode.header_fields()})
    mode.set(True)
    notice = mode.notification()
    assert notice is not None
    assert "Plan mode is active" in notice


def test_the_header_records_the_mode():
    """Recording it is what makes the next step able to tell."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    mode.set(True)
    assert mode.header_fields() == {"plan_active": True}


def test_no_notice_when_the_header_predates_plan_mode_entirely():
    """A header without the field cannot say what the model was told.

    Returning a notice here is the safe choice: a redundant notice costs one
    line, whereas a missing one leaves the model acting on a mode that changed.
    """
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    session.append(REQUEST_HEADER, {"turn": 0, "step": 0})  # no plan_active

    mode.set(True)
    assert mode.notification() is not None


# ---------------------------------------------------------------------------
# The tool catalogue does not change
# ---------------------------------------------------------------------------


def test_exit_plan_mode_is_registered_regardless_of_the_mode():
    """The catalogue must not change with the mode.

    If it did, a tool call would fail for a reason unrelated to the tool, and the
    model would have no way to tell why.
    """
    session = make_session()
    mode = PlanMode(session)
    tool = plan_tools(mode)

    assert tool.name == "exit_plan_mode"
    # The definition is built once and does not consult the mode.
    assert "plan" in tool.definition.parameters["properties"]


def test_the_tool_refuses_when_plan_mode_is_not_active():
    """Present but refusing -- not absent."""
    mode = PlanMode(make_session())
    tool = plan_tools(mode)
    result = asyncio.run(invoke(tool, {"plan": "# A plan"}))

    assert result.is_error
    assert "not active" in result.content


def test_the_tool_requires_a_heading():
    """A plan that cannot be reviewed as a plan is not submittable."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    mode.set(True)
    tool = plan_tools(mode, approve=lambda _text: True)

    for bad in ("just prose", "", "  ", "#", "#x"):
        result = asyncio.run(invoke(tool, {"plan": bad}))
        assert result.is_error, f"{bad!r} should have been refused"
        assert "heading" in result.content


def test_the_tool_fails_closed_without_an_approval_channel():
    """A mode nobody can approve would leave the session stuck."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    mode.set(True)
    tool = plan_tools(mode, approve=None)

    result = asyncio.run(invoke(tool, {"plan": "# A plan"}))
    assert result.is_error
    assert "no approval channel" in result.content
    # And the mode was not exited by a failed submission.
    assert mode.state().active is True


def test_an_approved_plan_leaves_plan_mode():
    """Approval is the exit."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    mode.set(True)
    tool = plan_tools(mode, approve=lambda _text: True)

    result = asyncio.run(invoke(tool, {"plan": "# A plan\n\nDo the thing."}))
    assert not result.is_error
    assert result.concludes_turn, "an approved plan ends the step"
    assert mode.state().active is False


def test_a_rejected_plan_stays_in_plan_mode():
    """Rejection must not silently exit -- the user asked for a revision."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    mode.set(True)
    tool = plan_tools(mode, approve=lambda _text: False)

    result = asyncio.run(invoke(tool, {"plan": "# A plan"}))
    assert not result.is_error, "a rejection is an ordinary outcome"
    assert result.value == {"approved": False}
    assert mode.state().active is True, "a rejected plan must not exit the mode"


def test_a_broken_approval_channel_denies():
    """A channel that raises must not be read as consent."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    mode.set(True)

    def broken(_text: str) -> bool:
        raise RuntimeError("the approval UI crashed")

    tool = plan_tools(mode, approve=broken)
    result = asyncio.run(invoke(tool, {"plan": "# A plan"}))

    assert result.is_error
    assert "approval channel failed" in result.content
    assert mode.state().active is True


def test_an_async_approval_channel_works():
    """An approval may need to wait for a human, so it may be a coroutine."""
    session = make_session()
    open_turn(session)
    mode = PlanMode(session)
    mode.set(True)

    async def approve(_text: str) -> bool:
        return True

    tool = plan_tools(mode, approve=approve)
    result = asyncio.run(invoke(tool, {"plan": "# Approved plan"}))
    assert not result.is_error
    assert mode.state().active is False


# ---------------------------------------------------------------------------
# Integration with the loop
# ---------------------------------------------------------------------------


def build_loop(script):
    from arm.agent.loop import Runtime

    session = Session.create("plan-int")
    registry = ToolRegistry()
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    mode = PlanMode(session)
    registry.register(plan_tools(mode, approve=lambda _t: True))

    runtime = Runtime(
        session,
        MockAdapter(script),
        registry,
        assembler=assembler,
        plan=mode,
        model="mock",
    )
    return runtime, session, mode


def test_the_policy_reaches_the_system_prompt():
    """An active mode must actually change what the model is told."""
    runtime, session, mode = build_loop([text_response("ok")])
    mode.set(True)

    runtime.submit("plan something")
    asyncio.run(runtime.run_until_idle())

    system = next(
        e for e in session if e.type == "system/message"
    )
    text = system.data["message"]["content"][0]["text"]
    assert "# Plan mode" in text


def test_the_policy_is_absent_when_inactive():
    """An inactive mode must not add a policy section."""
    runtime, session, _mode = build_loop([text_response("ok")])
    runtime.submit("just do it")
    asyncio.run(runtime.run_until_idle())

    system = next(e for e in session if e.type == "system/message")
    text = system.data["message"]["content"][0]["text"]
    assert "# Plan mode" not in text


def test_entering_plan_mode_does_not_change_the_tool_catalogue():
    """The central claim: the mode is prompt-only."""
    runtime, session, mode = build_loop([text_response("ok")])

    runtime.submit("before")
    asyncio.run(runtime.run_until_idle())
    before = next(e for e in session if e.type == REQUEST_HEADER).data["tools"]

    mode.set(True)
    runtime.submit("during")
    asyncio.run(runtime.run_until_idle())

    headers = [e for e in session if e.type == REQUEST_HEADER]
    after = headers[-1].data["tools"]
    assert list(before) == list(after), (
        "plan mode changed the tool catalogue; a tool call would now fail for a "
        "reason unrelated to the tool"
    )


def test_a_pending_choice_commits_when_a_step_is_accepted():
    """The queue must drain through the real loop, not only in a unit test."""
    runtime, session, mode = build_loop([text_response("ok")])
    mode.set(True)
    assert mode.pending is True

    runtime.submit("go")
    asyncio.run(runtime.run_until_idle())

    assert mode.state().active is True, "the queued choice never committed"
    assert mode.pending is False


def test_a_mode_change_is_announced_in_the_transcript():
    """A toggle must be visible to a reader of the log."""
    runtime, session, mode = build_loop([text_response("ok")])
    mode.set(True)

    runtime.submit("go")
    asyncio.run(runtime.run_until_idle())

    notices = [
        e
        for e in session
        if e.type == "user/message" and e.data.get("source") == "plan-mode"
    ]
    assert notices, "the mode change was not announced"
    assert "Plan mode is active" in notices[0].data["message"]["content"][0]["text"]


def test_the_header_records_the_mode_for_each_request():
    """Every request says which mode it was made under."""
    runtime, session, mode = build_loop([text_response("ok")])
    mode.set(True)

    runtime.submit("go")
    asyncio.run(runtime.run_until_idle())

    header = next(e for e in session if e.type == REQUEST_HEADER)
    assert header.data["plan_active"] is True


def test_exiting_plan_mode_restores_the_plain_prompt():
    """Leaving the mode must remove the policy, not leave it behind."""
    runtime, session, mode = build_loop([text_response("first"), text_response("second")])
    mode.set(True)
    runtime.submit("one")
    asyncio.run(runtime.run_until_idle())

    mode.set(False)
    runtime.submit("two")
    asyncio.run(runtime.run_until_idle())

    # Two system messages were recorded: one with the policy, one without.
    systems = [e for e in session if e.type == "system/message"]
    texts = [e.data["message"]["content"][0]["text"] for e in systems]
    assert any("# Plan mode" in t for t in texts)
    assert any("# Plan mode" not in t for t in texts), (
        "exiting the mode did not change the prompt"
    )


def test_a_prompt_change_replaces_the_previous_prompt():
    """A changed prompt must be *replaced*, not appended.

    Appending a second ``system/message`` would leave the stale prompt on the
    surface ahead of the new one, so the model would read "plan mode is active"
    followed by a prompt that says nothing about plan mode -- a contradiction it
    would have to resolve on its own.
    """
    runtime, session, mode = build_loop([text_response("a"), text_response("b")])

    mode.set(True)
    runtime.submit("one")
    asyncio.run(runtime.run_until_idle())

    mode.set(False)
    runtime.submit("two")
    asyncio.run(runtime.run_until_idle())

    system_nodes = [
        seq
        for seq in session.surface_nodes()
        if session.event_at(seq).type == "system/message"
    ]
    assert len(system_nodes) == 1, (
        f"the surface holds {len(system_nodes)} system prompts; the stale one was "
        "not replaced"
    )
    derived_system = [m.text for m in session.derive() if m.role == "system"]
    assert len(derived_system) == 1
    assert "# Plan mode" not in derived_system[0], (
        "the model still sees the plan policy after leaving plan mode"
    )


def test_leaving_plan_mode_removes_the_policy_from_what_the_model_sees():
    """The policy must actually disappear, not merely stop being re-recorded."""
    runtime, session, mode = build_loop([text_response("a"), text_response("b")])
    mode.set(True)
    runtime.submit("one")
    asyncio.run(runtime.run_until_idle())
    assert any(
        "# Plan mode" in m.text for m in session.derive() if m.role == "system"
    )

    mode.set(False)
    runtime.submit("two")
    asyncio.run(runtime.run_until_idle())

    assert not any(
        "# Plan mode" in m.text for m in session.derive() if m.role == "system"
    ), "the plan policy survived leaving plan mode"


def test_the_prompt_is_the_first_surface_node():
    """The system prompt must head the transcript, or no model API accepts it."""
    runtime, session, mode = build_loop([text_response("ok")])
    mode.set(True)
    runtime.submit("go")
    asyncio.run(runtime.run_until_idle())

    nodes = session.surface_nodes()
    assert session.event_at(nodes[0]).type == "system/message"
    assert [m.role for m in session.derive()][0] == "system"


def test_a_plan_policy_change_is_visible_in_the_next_request():
    """End to end: the model is actually told about the mode."""
    runtime, session, mode = build_loop([text_response("a"), text_response("b")])
    mode.set(True)
    runtime.submit("one")
    asyncio.run(runtime.run_until_idle())

    first = runtime.adapter.requests[0]  # type: ignore[attr-defined]
    assert any("# Plan mode" in m.text for m in first.messages)

    mode.set(False)
    runtime.submit("two")
    asyncio.run(runtime.run_until_idle())

    # Each turn makes more than one request, so the last one is the request that
    # carries the *new* prompt rather than a later step of the previous turn.
    last = runtime.adapter.requests[-1]  # type: ignore[attr-defined]
    assert not any("# Plan mode" in m.text for m in last.messages), (
        "the final request still carries the plan policy"
    )
    # And the model is told about the change rather than left to infer it.
    assert any(
        "Plan mode is off" in m.text for m in last.messages
    ), "the mode change was never announced to the model"

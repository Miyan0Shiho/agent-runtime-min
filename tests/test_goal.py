"""Goal tests.

The design's central claim is a split: **``phase`` is durable, ``activation`` is
not**. So the tests that matter most are the ones about a restart -- a goal that
was active must come back *disarmed*, because an agent that keeps working toward
an objective nobody is watching is the failure this prevents.

The second theme is that the guardrails must be *refusals*, not silent state
changes. A model reporting a blocker once must not be able to mark a goal
blocked, and a stale revision must not be able to overwrite a decision it never
saw.
"""

from __future__ import annotations

import asyncio

from arm.goal.store import (
    ACTIVATION_DISARMED,
    BLOCK_ROUND_LIMIT,
    PHASE_ACTIVE,
    PHASE_BLOCKED,
    PHASE_COMPLETE,
    PHASE_PAUSED,
    GoalError,
    GoalRef,
    GoalStore,
)
from arm.session.events import GOAL_CHANGED, GOAL_ROUND
from arm.session.session import Session


def make_store(**kwargs) -> tuple[GoalStore, Session]:
    session = Session.create("goal")
    return GoalStore(session, **kwargs), session


# ---------------------------------------------------------------------------
# Creating
# ---------------------------------------------------------------------------


def test_create_starts_an_active_armed_goal():
    """A goal a human just asked for is active, and this process may drive it."""
    store, _ = make_store()
    goal = store.create("finish the parser")

    assert goal.phase == PHASE_ACTIVE
    assert goal.objective == "finish the parser"
    assert goal.revision == 1
    assert goal.armed, "a freshly created goal must be armed in this process"
    assert goal.may_continue


def test_create_refuses_an_empty_objective():
    """A goal with no objective cannot be worked toward."""
    store, _ = make_store()
    for empty in ("", "   ", "\n"):
        try:
            store.create(empty)
        except GoalError as exc:
            assert exc.code == "GOAL_EMPTY"
        else:
            raise AssertionError(f"{empty!r} should have been refused")


def test_create_refuses_to_overwrite_a_running_goal():
    """Replacing an objective would discard the work already done toward it."""
    store, _ = make_store()
    store.create("the first objective")

    try:
        store.create("a different objective")
    except GoalError as exc:
        assert exc.code == "GOAL_EXISTS"
    else:
        raise AssertionError("a running goal was overwritten")


def test_create_can_replace_a_completed_goal():
    """A finished goal is not in the way of the next one."""
    store, _ = make_store()
    first = store.create("first")
    store.complete(expected=first.ref)

    second = store.create("second")
    assert second.objective == "second"
    assert second.revision == 1, "a new goal starts at revision 1"


def test_create_requires_a_positive_round_limit():
    """A goal with no ceiling is a loop that runs until someone notices."""
    store, _ = make_store()
    try:
        store.create("x", max_rounds=0)
    except GoalError as exc:
        assert exc.code == "GOAL_INVALID"
    else:
        raise AssertionError("a zero round limit was accepted")


def test_the_goal_is_written_to_the_log():
    """Durability is the whole point, so the event must exist."""
    store, session = make_store()
    store.create("persisted")

    events = [e for e in session if e.type == GOAL_CHANGED]
    assert len(events) == 1
    assert events[0].data["goal"]["objective"] == "persisted"
    assert events[0].data["action"] == "create"


def test_two_goals_get_different_ids():
    """Ids must not collide across sessions or across time."""
    store, _ = make_store()
    first = store.create("one")
    store.complete(expected=first.ref)
    second = store.create("two")
    assert first.id != second.id


# ---------------------------------------------------------------------------
# The split that defines the design
# ---------------------------------------------------------------------------


def test_phase_survives_a_restart():
    """The durable half comes back, because it is in the log."""
    store, session = make_store()
    store.create("survives")

    restored = GoalStore(Session.from_events("goal", session.events()))
    goal = restored.current()
    assert goal is not None
    assert goal.phase == PHASE_ACTIVE
    assert goal.objective == "survives"


def test_activation_does_not_survive_a_restart():
    """The process-local half must NOT come back, and that is the point.

    An active goal that re-armed itself on load would keep working toward an
    objective nobody is watching. Because ``activation`` has no representation in
    the log, there is no code path that could restore it -- the guarantee is
    structural rather than a discipline someone has to remember.
    """
    store, session = make_store()
    goal = store.create("keeps running")
    assert goal.armed, "it is armed in the process that created it"

    restored = GoalStore(Session.from_events("goal", session.events()))
    after = restored.current()
    assert after is not None
    assert after.phase == PHASE_ACTIVE, "the phase is durable"
    assert after.activation == ACTIVATION_DISARMED, "the activation is not"
    assert not after.may_continue, "a restored goal must not continue by itself"


def test_a_restored_goal_resumes_only_when_a_human_asks():
    """The only route back to automatic continuation is an explicit resume."""
    store, session = make_store()
    store.create("resumable")

    restored = GoalStore(Session.from_events("goal", session.events()))
    assert not restored.require_current().may_continue

    resumed = restored.resume()
    assert resumed.armed
    assert resumed.may_continue


# ---------------------------------------------------------------------------
# Compare-and-set revisions
# ---------------------------------------------------------------------------


def test_a_stale_revision_is_refused():
    """A stale caller must be told, not silently obeyed.

    Last-write-wins would let a model that has been thinking for a while
    overwrite a decision a human made in between -- and the model would never
    learn that its view was wrong.
    """
    store, _ = make_store()
    goal = store.create("original")
    stale = goal.ref

    store.edit(expected=stale, objective="updated by someone else")

    try:
        store.edit(expected=stale, objective="updated by the stale caller")
    except GoalError as exc:
        assert exc.code == "GOAL_STALE"
        assert "revision" in str(exc).lower()
    else:
        raise AssertionError("a stale revision was accepted")


def test_a_revision_mismatch_names_both_values():
    """The error must be actionable: the caller has to know what to re-read."""
    store, _ = make_store()
    goal = store.create("x")
    store.edit(expected=goal.ref, objective="y")

    try:
        store.edit(expected=goal.ref, objective="z")
    except GoalError as exc:
        assert "1" in str(exc) and "2" in str(exc), (
            f"the error should name both revisions: {exc}"
        )


def test_an_edit_increments_the_revision():
    """Every accepted change moves the revision, or CAS cannot detect staleness."""
    store, _ = make_store()
    goal = store.create("x")
    assert goal.revision == 1

    edited = store.edit(expected=goal.ref, objective="y")
    assert edited.revision == 2


def test_editing_the_wrong_goal_is_refused():
    """A revision alone is not identity; the id must match too."""
    store, _ = make_store()
    goal = store.create("x")
    wrong = GoalRef(id="goal-someone-else", revision=goal.revision)

    try:
        store.edit(expected=wrong, objective="y")
    except GoalError as exc:
        assert exc.code == "GOAL_STALE"
    else:
        raise AssertionError("an edit aimed at another goal was accepted")


def test_an_edit_must_change_something():
    """A no-op edit would burn a revision for nothing."""
    store, _ = make_store()
    goal = store.create("x")
    try:
        store.edit(expected=goal.ref)
    except GoalError as exc:
        assert exc.code == "GOAL_EMPTY_EDIT"
    else:
        raise AssertionError("an empty edit was accepted")


def test_an_edit_cannot_empty_the_objective():
    """A goal with no objective cannot be worked toward."""
    store, _ = make_store()
    goal = store.create("x")
    try:
        store.edit(expected=goal.ref, objective="   ")
    except GoalError as exc:
        assert exc.code == "GOAL_EMPTY"
    else:
        raise AssertionError("an empty objective was accepted")


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------


def test_pause_stops_continuation_but_keeps_the_objective():
    """Pausing is not abandoning."""
    store, _ = make_store()
    goal = store.create("work")
    paused = store.pause(expected=goal.ref)

    assert paused.phase == PHASE_PAUSED
    assert paused.objective == "work"
    assert not paused.may_continue


def test_resume_returns_to_active():
    """Resuming undoes a pause."""
    store, _ = make_store()
    goal = store.create("work")
    store.pause(expected=goal.ref)
    resumed = store.resume()

    assert resumed.phase == PHASE_ACTIVE
    assert resumed.armed


def test_resume_refuses_a_goal_that_is_already_armed():
    """A duplicate resume is a caller mistake, not a state change."""
    store, _ = make_store()
    store.create("work")
    try:
        store.resume()
    except GoalError as exc:
        assert exc.code == "GOAL_INVALID_TRANSITION"
    else:
        raise AssertionError("a redundant resume was accepted")


def test_resume_refuses_a_completed_goal():
    """Finished is finished."""
    store, _ = make_store()
    goal = store.create("work")
    store.complete(expected=goal.ref)

    try:
        store.resume()
    except GoalError as exc:
        assert exc.code == "GOAL_INVALID_TRANSITION"
    else:
        raise AssertionError("a completed goal was resumed")


def test_complete_disarms():
    """A finished goal must not keep being driven."""
    store, _ = make_store()
    goal = store.create("work")
    done = store.complete(expected=goal.ref)

    assert done.phase == PHASE_COMPLETE
    assert not done.armed
    assert not done.may_continue


def test_completing_twice_is_refused():
    """The second call is a caller mistake."""
    store, _ = make_store()
    goal = store.create("work")
    store.complete(expected=goal.ref)
    try:
        store.complete()
    except GoalError as exc:
        assert exc.code == "GOAL_INVALID_TRANSITION"
    else:
        raise AssertionError("a double completion was accepted")


def test_pause_refuses_a_completed_goal():
    """A completed goal has no continuation to pause."""
    store, _ = make_store()
    goal = store.create("work")
    store.complete(expected=goal.ref)
    try:
        store.pause()
    except GoalError as exc:
        assert exc.code == "GOAL_INVALID_TRANSITION"
    else:
        raise AssertionError("a completed goal was paused")


# ---------------------------------------------------------------------------
# The blocked guardrail
# ---------------------------------------------------------------------------


def test_one_reported_blocker_is_not_enough():
    """A single bad round is a bad round, not a wall."""
    store, _ = make_store(blocked_after=3)
    goal = store.create("hard")

    try:
        store.block("the same error", expected=goal.ref)
    except GoalError as exc:
        assert exc.code == "GOAL_BLOCKED_TOO_EARLY"
    else:
        raise AssertionError("one report was enough to mark the goal blocked")


def test_the_same_blocker_persisting_does_block():
    """Three consecutive reports is evidence."""
    store, _ = make_store(blocked_after=3)
    goal = store.create("hard")

    for _ in range(2):
        try:
            store.block("the same error", expected=goal.ref)
        except GoalError:
            pass

    blocked = store.block("the same error", expected=goal.ref)
    assert blocked.phase == PHASE_BLOCKED
    assert blocked.snapshot.blocked_reason == "the same error"
    assert not blocked.armed


def test_a_forced_block_skips_the_guardrail():
    """A round limit is a fact, not a claim, so it needs no corroboration."""
    store, _ = make_store(blocked_after=3)
    store.create("exhausted")

    blocked = store.block(
        "the round limit was reached", code=BLOCK_ROUND_LIMIT, force=True
    )
    assert blocked.phase == PHASE_BLOCKED
    assert blocked.snapshot.blocked_code == BLOCK_ROUND_LIMIT


def test_a_block_needs_a_reason():
    """A blocked goal with no reason is unactionable."""
    store, _ = make_store()
    store.create("x")
    try:
        store.block("   ", force=True)
    except GoalError as exc:
        assert exc.code == "GOAL_BLOCKED_NO_REASON"
    else:
        raise AssertionError("a reasonless block was accepted")


def test_the_block_reason_survives_a_restart():
    """The reason is the whole value of the blocked state."""
    store, session = make_store(blocked_after=1)
    store.create("hard")
    store.block("the API key is missing", force=True)

    restored = GoalStore(Session.from_events("goal", session.events()))
    goal = restored.require_current()
    assert goal.phase == PHASE_BLOCKED
    assert goal.snapshot.blocked_reason == "the API key is missing"


def test_resuming_clears_the_block():
    """A resumed goal is not still described by the old blocker."""
    store, _ = make_store(blocked_after=1)
    store.create("hard")
    store.block("stuck", force=True)

    resumed = store.resume()
    assert resumed.phase == PHASE_ACTIVE
    assert resumed.snapshot.blocked_reason == ""
    assert resumed.snapshot.blocked_code == ""


def test_the_block_counter_resets_after_a_resume():
    """Otherwise two separate attempts would add up to one block."""
    store, _ = make_store(blocked_after=3)
    store.create("hard")

    try:
        store.block("same", force=False)
    except GoalError:
        pass
    store.pause()
    store.resume()

    # The streak should be back to zero, so one more report is still not enough.
    try:
        store.block("same")
    except GoalError as exc:
        assert exc.code == "GOAL_BLOCKED_TOO_EARLY"
    else:
        raise AssertionError("the block streak survived a resume")


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


def test_a_round_is_counted_from_the_events():
    """``rounds_started`` is counted, so it cannot drift from what happened."""
    store, session = make_store()
    store.create("work")

    assert store.start_round() == 1
    assert store.start_round() == 2
    assert store.require_current().rounds_started == 2

    restored = GoalStore(Session.from_events("goal", session.events()))
    assert restored.require_current().rounds_started == 2


def test_the_round_limit_is_enforced():
    """A goal with a ceiling must stop at it."""
    store, _ = make_store()
    store.create("bounded", max_rounds=2)

    store.start_round()
    store.start_round()
    assert store.require_current().exhausted

    try:
        store.start_round()
    except GoalError as exc:
        assert exc.code == "GOAL_NOT_RUNNABLE"
    else:
        raise AssertionError("the round limit was not enforced")


def test_a_disarmed_goal_cannot_start_a_round():
    """The process-local half gates continuation, as designed."""
    store, _ = make_store()
    store.create("work")
    store.disarm("the response was truncated")

    try:
        store.start_round()
    except GoalError as exc:
        assert exc.code == "GOAL_NOT_RUNNABLE"
    else:
        raise AssertionError("a disarmed goal started a round")


def test_disarm_does_not_change_the_phase():
    """The goal is still active; only this process stopped driving it."""
    store, _ = make_store()
    store.create("work")
    store.disarm()

    assert store.require_current().phase == PHASE_ACTIVE
    assert not store.require_current().armed


def test_a_paused_goal_cannot_start_a_round():
    """Pausing must actually stop the rounds."""
    store, _ = make_store()
    goal = store.create("work")
    store.pause(expected=goal.ref)

    try:
        store.start_round()
    except GoalError as exc:
        assert exc.code == "GOAL_NOT_RUNNABLE"
    else:
        raise AssertionError("a paused goal started a round")


def test_resume_refuses_an_exhausted_goal():
    """Resuming an exhausted goal would immediately hit the limit again."""
    store, _ = make_store()
    goal = store.create("bounded", max_rounds=1)
    store.start_round()
    store.pause(expected=GoalRef(id=goal.id, revision=goal.revision))

    try:
        store.resume()
    except GoalError as exc:
        assert exc.code == "GOAL_ROUND_LIMIT"
        assert "max_rounds" in str(exc)
    else:
        raise AssertionError("an exhausted goal was resumed")


def test_raising_the_limit_lets_an_exhausted_goal_continue():
    """The error message says to raise the limit, so that must work."""
    store, _ = make_store()
    goal = store.create("bounded", max_rounds=1)
    store.start_round()
    store.pause()

    current = store.require_current()
    store.edit(expected=current.ref, max_rounds=5)
    resumed = store.resume()

    assert resumed.may_continue
    assert store.start_round() == 2


def test_a_round_records_which_revision_it_belonged_to():
    """A round against a stale revision is detectable after the fact."""
    store, session = make_store()
    goal = store.create("work")
    store.start_round()

    rounds = [e for e in session if e.type == GOAL_ROUND]
    assert rounds[0].data["goal_id"] == goal.id
    assert rounds[0].data["revision"] == goal.revision


# ---------------------------------------------------------------------------
# Clearing
# ---------------------------------------------------------------------------


def test_clear_abandons_the_goal_without_deleting_history():
    """The log has no deletion, so clearing is a tombstone."""
    store, session = make_store()
    store.create("abandoned")
    store.clear()

    assert store.require_current().phase == PHASE_COMPLETE
    assert store.require_current().snapshot.blocked_code == "cleared"
    # The original creation is still in the log.
    assert any(e.type == GOAL_CHANGED for e in session)
    assert store.history()[0].objective == "abandoned"


def test_clear_refuses_when_there_is_no_goal():
    """Clearing nothing is a caller mistake."""
    store, _ = make_store()
    try:
        store.clear()
    except GoalError as exc:
        assert exc.code == "GOAL_MISSING"
    else:
        raise AssertionError("clearing with no goal was accepted")


def test_require_current_names_the_problem_when_there_is_no_goal():
    """The error must say what is missing."""
    store, _ = make_store()
    try:
        store.require_current()
    except GoalError as exc:
        assert exc.code == "GOAL_MISSING"
    else:
        raise AssertionError("require_current accepted an absent goal")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_records_every_snapshot_in_order():
    """A reader must be able to see how the goal evolved."""
    store, _ = make_store()
    goal = store.create("v1")
    store.edit(expected=goal.ref, objective="v2")
    store.pause()

    history = store.history()
    assert [h.objective for h in history] == ["v1", "v2", "v2"]
    assert [h.revision for h in history] == [1, 2, 3]
    assert [h.phase for h in history] == [PHASE_ACTIVE, PHASE_ACTIVE, PHASE_PAUSED]


def test_revisions_are_monotonic():
    """CAS depends on revisions only ever increasing."""
    store, _ = make_store()
    goal = store.create("x")
    revisions = [goal.revision]
    for index in range(5):
        goal = store.edit(expected=goal.ref, objective=f"v{index}")
        revisions.append(goal.revision)
    assert revisions == sorted(revisions)
    assert len(set(revisions)) == len(revisions)


# ---------------------------------------------------------------------------
# The store is per-session
# ---------------------------------------------------------------------------


def test_two_sessions_have_independent_goals():
    """One session's goal must not leak into another's."""
    first, _ = make_store()
    second, _ = make_store()
    first.create("first session's goal")

    assert second.current() is None


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


def make_tools(**kwargs):
    from arm.goal.tools import GoalTools

    store, session = make_store(**kwargs)
    tools = {t.name: t for t in GoalTools(store).tools()}
    return tools, store, session


async def invoke(tool, arguments: dict):
    return await tool.body(arguments, None)


def test_the_three_goal_tools_exist():
    """create, read, update -- and nothing else."""
    tools, _store, _session = make_tools()
    assert sorted(tools) == ["create_goal", "get_goal", "update_goal"]


def test_create_tool_returns_the_revision():
    """The model needs the revision to make its next call."""
    tools, _store, _session = make_tools()
    result = asyncio.run(invoke(tools["create_goal"], {"objective": "build it"}))

    assert not result.is_error
    assert "revision 1" in result.content
    assert result.value["revision"] == 1


def test_create_tool_reports_a_refusal_as_a_result():
    """A refusal is an ordinary outcome, so the step continues."""
    tools, _store, _session = make_tools()
    asyncio.run(invoke(tools["create_goal"], {"objective": "first"}))
    again = asyncio.run(invoke(tools["create_goal"], {"objective": "second"}))

    assert again.is_error
    assert again.error_code == "GOAL_EXISTS"


def test_get_tool_reports_the_absence_of_a_goal():
    """Reading nothing is a valid answer, not an error."""
    tools, _store, _session = make_tools()
    result = asyncio.run(invoke(tools["get_goal"], {}))

    assert not result.is_error
    assert "no goal" in result.content.lower()


def test_get_tool_reports_that_a_restored_goal_is_disarmed():
    """The model must be told, or it will wonder why nothing is happening."""
    tools, store, session = make_tools()
    asyncio.run(invoke(tools["create_goal"], {"objective": "work"}))

    # Simulate a restart: a fresh store over the same log is disarmed.
    from arm.goal.store import GoalStore as _GoalStore
    from arm.goal.tools import GoalTools as _GoalTools

    restored = _GoalTools(_GoalStore(Session.from_events("goal", session.events())))
    read = {t.name: t for t in restored.tools()}
    result = asyncio.run(invoke(read["get_goal"], {}))

    assert "not armed" in result.content
    assert "resume" in result.content


def test_update_requires_a_revision():
    """Without one, a stale caller could overwrite a decision it never saw."""
    tools, _store, _session = make_tools()
    asyncio.run(invoke(tools["create_goal"], {"objective": "work"}))

    result = asyncio.run(invoke(tools["update_goal"], {"action": "complete"}))
    assert result.is_error
    assert "revision" in result.content


def test_update_refuses_a_stale_revision():
    """The refusal must reach the model as a readable result."""
    tools, store, _session = make_tools()
    asyncio.run(invoke(tools["create_goal"], {"objective": "work"}))
    asyncio.run(
        invoke(tools["update_goal"], {"action": "edit", "revision": 1, "objective": "v2"})
    )

    stale = asyncio.run(
        invoke(tools["update_goal"], {"action": "complete", "revision": 1})
    )
    assert stale.is_error
    assert stale.error_code == "GOAL_STALE"


def test_update_rejects_an_unknown_action():
    """An unknown action must be named, not silently ignored."""
    tools, _store, _session = make_tools()
    asyncio.run(invoke(tools["create_goal"], {"objective": "work"}))

    result = asyncio.run(
        invoke(tools["update_goal"], {"action": "teleport", "revision": 1})
    )
    assert result.is_error
    assert "unknown action" in result.content
    assert "complete" in result.content, "the error should list the valid actions"


def test_update_can_complete_a_goal():
    """The ordinary success path."""
    tools, store, _session = make_tools()
    asyncio.run(invoke(tools["create_goal"], {"objective": "work"}))

    result = asyncio.run(
        invoke(tools["update_goal"], {"action": "complete", "revision": 1})
    )
    assert not result.is_error
    assert store.require_current().phase == PHASE_COMPLETE


def test_update_can_resume_a_disarmed_goal():
    """Resuming is how a restored goal comes back to life."""
    tools, store, session = make_tools()
    asyncio.run(invoke(tools["create_goal"], {"objective": "work"}))

    from arm.goal.store import GoalStore as _GoalStore
    from arm.goal.tools import GoalTools as _GoalTools

    restored_store = _GoalStore(Session.from_events("goal", session.events()))
    restored = {t.name: t for t in _GoalTools(restored_store).tools()}
    assert not restored_store.require_current().armed

    result = asyncio.run(
        invoke(restored["update_goal"], {"action": "resume", "revision": 1})
    )
    assert not result.is_error
    assert restored_store.require_current().armed


def test_blocked_through_the_tool_needs_corroboration():
    """The guardrail must be visible to the model as a readable refusal."""
    tools, _store, _session = make_tools(blocked_after=3)
    asyncio.run(invoke(tools["create_goal"], {"objective": "hard"}))

    first = asyncio.run(
        invoke(
            tools["update_goal"],
            {"action": "blocked", "revision": 1, "blocked_reason": "no API key"},
        )
    )
    assert first.is_error
    assert first.error_code == "GOAL_BLOCKED_TOO_EARLY"
    assert "consecutive" in first.content, (
        "the model must be told what the rule is, not just that it failed"
    )


def test_clear_through_the_tool_reports_abandonment():
    """Clearing is not completing, and the message must say so."""
    tools, store, _session = make_tools()
    asyncio.run(invoke(tools["create_goal"], {"objective": "work"}))

    result = asyncio.run(
        invoke(tools["update_goal"], {"action": "clear", "revision": 1})
    )
    assert not result.is_error
    assert "abandoned" in result.content
    assert store.require_current().phase == PHASE_COMPLETE


def test_every_goal_tool_schema_normalizes():
    """The schemas must survive the model-facing subset."""
    import json as _json

    tools, _store, _session = make_tools()
    for tool in tools.values():
        rendered = _json.dumps(dict(tool.definition.parameters))
        assert "title" not in rendered
        assert "$defs" not in rendered

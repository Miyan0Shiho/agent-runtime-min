"""Team journal tests.

Three themes, one per projection:

* **The roster** -- membership is recorded before the member exists, so a crash
  leaves something reconciliation can act on.
* **The mailbox** -- delivery is confirmed by the recipient, and the limits are
  enforced at queue time.
* **The task board** -- a DAG with compare-and-set revisions, where every way a
  board becomes unusable is refused before it is written.

The recurring check is that a *refusal* leaves the log unchanged. A validation
that appends before it validates is worse than no validation, because the bad
state is then durable.
"""

from __future__ import annotations

from typing import Any

import pytest

from arm.session.session import Session
from arm.team.journal import (
    LEAD_NAME,
    MAX_SUBJECT_CHARS,
    MEMBER_ACTIVE,
    MEMBER_FAILED,
    MEMBER_PROVISIONING,
    TASK_COMPLETED,
    TASK_IN_PROGRESS,
    TASK_PENDING,
    TeamError,
    TeamJournal,
    TeamMember,
    TeamMessage,
    TeamTask,
    assert_task_graph,
    validate_member_name,
)


def make_journal(**kwargs) -> tuple[TeamJournal, Session]:
    session = Session.create("lead")
    return TeamJournal(session, **kwargs), session


def add_member(journal: TeamJournal, name: str, phase: str = MEMBER_ACTIVE) -> None:
    """Add an active member, in two steps as the real path does."""
    journal.add_member(TeamMember(name=name))
    journal.set_member_phase(name, phase)


# ===========================================================================
# Member names
# ===========================================================================


def test_a_valid_name_is_accepted():
    validate_member_name("alice")
    validate_member_name("worker-1")
    validate_member_name("a1-b2-c3")


@pytest.mark.parametrize(
    "bad", ["Alice", "alice smith", "alice_smith", "-alice", "alice-", "a--b", "", "x" * 65]
)
def test_an_invalid_name_is_refused(bad: str):
    """The name is what a teammate types, so its shape is enforced."""
    with pytest.raises(TeamError) as info:
        validate_member_name(bad)
    assert info.value.code == "TEAM_INVALID_MEMBER_NAME"


def test_the_lead_name_is_reserved():
    """The lead owns the log, so a member cannot take its name."""
    with pytest.raises(TeamError) as info:
        validate_member_name(LEAD_NAME)
    assert info.value.code == "TEAM_INVALID_MEMBER_NAME"


# ===========================================================================
# The roster
# ===========================================================================


def test_a_member_starts_provisioning():
    """Recorded before its agent exists, so a crash leaves evidence."""
    journal, _ = make_journal()
    journal.add_member(TeamMember(name="alice"))

    member = journal.members()["alice"]
    assert member.phase == MEMBER_PROVISIONING
    assert not member.active


def test_a_member_becomes_active():
    """The provisioning phase is settled once the agent exists."""
    journal, _ = make_journal()
    add_member(journal, "alice")

    assert journal.members()["alice"].active
    assert set(journal.active_members()) == {"alice"}


def test_a_duplicate_name_is_refused():
    """Two members with one name would make every message ambiguous."""
    journal, _ = make_journal()
    add_member(journal, "alice")

    with pytest.raises(TeamError) as info:
        journal.add_member(TeamMember(name="alice"))
    assert info.value.code == "TEAM_MEMBER_NAME_TAKEN"


def test_a_failed_member_can_be_replaced():
    """A name freed by a failure is usable again; nothing was created."""
    journal, _ = make_journal()
    journal.add_member(TeamMember(name="alice"))
    journal.set_member_phase("alice", MEMBER_FAILED, error="the provider blew up")

    journal.add_member(TeamMember(name="alice"))
    assert journal.members()["alice"].phase == MEMBER_PROVISIONING


def test_the_member_limit_is_enforced():
    """An unbounded team is an unbounded amount of work in flight."""
    journal, _ = make_journal(max_members=2)
    add_member(journal, "a")
    add_member(journal, "b")

    with pytest.raises(TeamError) as info:
        journal.add_member(TeamMember(name="c"))
    assert info.value.code == "TEAM_MEMBER_LIMIT"


def test_a_failed_member_does_not_count_against_the_limit():
    """The limit bounds *live* members, not historical ones."""
    journal, _ = make_journal(max_members=1)
    journal.add_member(TeamMember(name="a"))
    journal.set_member_phase("a", MEMBER_FAILED)

    journal.add_member(TeamMember(name="b"))
    assert "b" in journal.members()


def test_changing_an_unknown_member_is_refused():
    journal, _ = make_journal()
    with pytest.raises(TeamError) as info:
        journal.set_member_phase("nobody", MEMBER_ACTIVE)
    assert info.value.code == "TEAM_MEMBER_NOT_FOUND"


def test_the_roster_survives_a_restart():
    """It is folded from the log, so recovery is free."""
    journal, session = make_journal()
    add_member(journal, "alice")

    restored = TeamJournal(Session.from_events("lead", session.events()))
    assert restored.members()["alice"].active


def test_the_child_session_is_recorded():
    """The link to the member's own log is on the member."""
    journal, _ = make_journal()
    journal.add_member(TeamMember(name="alice"))
    journal.set_member_phase("alice", MEMBER_ACTIVE, child_session="subagent-1")

    assert journal.members()["alice"].child_session == "subagent-1"


# ===========================================================================
# The mailbox
# ===========================================================================


def message(sender: str, recipient: str, body: str = "hello", mid: str = "m1") -> TeamMessage:
    return TeamMessage(message_id=mid, sender=sender, recipient=recipient, body=body)


def test_a_message_is_queued():
    """The ordinary path."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    journal.send(message("lead", "alice"))

    pending = journal.pending_for("alice")
    assert len(pending) == 1
    assert pending[0].body == "hello"
    assert not pending[0].delivered


def test_a_message_cannot_be_sent_to_itself():
    """A self-message is a mistake, not a conversation."""
    journal, _ = make_journal()
    add_member(journal, "alice")

    with pytest.raises(TeamError) as info:
        journal.send(message("alice", "alice"))
    assert info.value.code == "TEAM_SELF_MESSAGE"


def test_a_message_to_an_unknown_member_is_refused():
    journal, _ = make_journal()
    with pytest.raises(TeamError) as info:
        journal.send(message("lead", "nobody"))
    assert info.value.code == "TEAM_MEMBER_NOT_FOUND"


def test_a_message_to_a_provisioning_member_is_refused():
    """A member that does not exist yet cannot receive anything."""
    journal, _ = make_journal()
    journal.add_member(TeamMember(name="alice"))  # still provisioning

    with pytest.raises(TeamError) as info:
        journal.send(message("lead", "alice"))
    assert info.value.code == "TEAM_MEMBER_NOT_FOUND"


def test_an_empty_message_is_refused():
    journal, _ = make_journal()
    add_member(journal, "alice")

    with pytest.raises(TeamError) as info:
        journal.send(message("lead", "alice", body=""))
    assert info.value.code == "TEAM_INVALID_ARGUMENT"


def test_an_oversized_message_is_refused_by_bytes():
    """The limit is a transport limit, and a multi-byte character is not one byte."""
    journal, _ = make_journal(max_message_bytes=10)
    add_member(journal, "alice")

    # Five two-byte characters is ten bytes, which fits.
    journal.send(message("lead", "alice", body="ééééé", mid="fits"))

    with pytest.raises(TeamError) as info:
        journal.send(message("lead", "alice", body="éééééé", mid="too-big"))
    assert info.value.code == "TEAM_MESSAGE_TOO_LARGE"
    assert "12" in str(info.value), "the error should name the actual size"


def test_the_mailbox_limit_is_enforced():
    """An unbounded inbox would let one member drown another."""
    journal, _ = make_journal(max_pending_per_member=2)
    add_member(journal, "alice")

    journal.send(message("lead", "alice", mid="m1"))
    journal.send(message("lead", "alice", mid="m2"))

    with pytest.raises(TeamError) as info:
        journal.send(message("lead", "alice", mid="m3"))
    assert info.value.code == "TEAM_MAILBOX_FULL"


def test_a_duplicate_message_id_is_refused():
    """A repeated id would make delivery tracking ambiguous."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    journal.send(message("lead", "alice", mid="m1"))

    with pytest.raises(TeamError) as info:
        journal.send(message("lead", "alice", mid="m1"))
    assert info.value.code == "TEAM_DUPLICATE_MESSAGE"


def test_delivery_is_confirmed_by_the_recipient():
    """A sender that believed it delivered would lose the message silently."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    journal.send(message("lead", "alice"))

    assert journal.confirm_delivery("m1", "alice") is True
    assert journal.pending_for("alice") == ()
    assert journal.messages()[0].delivered


def test_confirming_delivery_twice_is_not_an_error():
    """A retry after a crash is expected, so it must be idempotent."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    journal.send(message("lead", "alice"))

    assert journal.confirm_delivery("m1", "alice") is True
    assert journal.confirm_delivery("m1", "alice") is False


def test_confirming_delivery_to_the_wrong_recipient_is_refused():
    """Delivery is per recipient, so confirming for another one is a bug."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    add_member(journal, "bob")
    journal.send(message("lead", "alice"))

    with pytest.raises(TeamError) as info:
        journal.confirm_delivery("m1", "bob")
    assert info.value.code == "TEAM_INVALID_ARGUMENT"


def test_confirming_an_unknown_message_is_refused():
    journal, _ = make_journal()
    add_member(journal, "alice")

    with pytest.raises(TeamError):
        journal.confirm_delivery("never-sent", "alice")


def test_the_mailbox_survives_a_restart():
    """Queued and delivered state are both folded from the log."""
    journal, session = make_journal()
    add_member(journal, "alice")
    journal.send(message("lead", "alice", mid="m1"))
    journal.send(message("lead", "alice", mid="m2"))
    journal.confirm_delivery("m1", "alice")

    restored = TeamJournal(Session.from_events("lead", session.events()))
    assert [m.message_id for m in restored.pending_for("alice")] == ["m2"]


def test_a_refused_send_leaves_no_event():
    """A validation that appends before it validates makes the bad state durable."""
    journal, session = make_journal()
    add_member(journal, "alice")
    before = len(session)

    with pytest.raises(TeamError):
        journal.send(message("lead", "alice", body=""))

    assert len(session) == before


# ===========================================================================
# The task board
# ===========================================================================


def test_a_task_is_created_pending():
    journal, _ = make_journal()
    task = journal.create_task(subject="do the thing", created_by="lead")

    assert task.id == "task-1"
    assert task.phase == TASK_PENDING
    assert task.revision == 1


def test_task_ids_are_sequential_and_not_reused():
    """A reused id would make history ambiguous."""
    journal, _ = make_journal()
    first = journal.create_task(subject="one")
    second = journal.create_task(subject="two")
    assert (first.id, second.id) == ("task-1", "task-2")


def test_a_task_needs_a_subject():
    journal, _ = make_journal()
    with pytest.raises(TeamError) as info:
        journal.create_task(subject="   ")
    assert info.value.code == "TEAM_INVALID_ARGUMENT"


def test_an_oversized_subject_is_refused():
    journal, _ = make_journal()
    with pytest.raises(TeamError):
        journal.create_task(subject="x" * (MAX_SUBJECT_CHARS + 1))


def test_the_task_limit_is_enforced():
    journal, _ = make_journal(max_tasks=2)
    journal.create_task(subject="one")
    journal.create_task(subject="two")

    with pytest.raises(TeamError) as info:
        journal.create_task(subject="three")
    assert info.value.code == "TEAM_TASK_LIMIT"


def test_write_scopes_are_deduplicated():
    """Scopes are advisory, but a duplicated scope is still a mistake."""
    journal, _ = make_journal()
    task = journal.create_task(
        subject="one", write_scopes=["a.py", "a.py", "b.py"]
    )
    assert task.write_scopes == ("a.py", "b.py")


def test_the_board_survives_a_restart():
    journal, session = make_journal()
    journal.create_task(subject="persisted")

    restored = TeamJournal(Session.from_events("lead", session.events()))
    assert restored.tasks()["task-1"].subject == "persisted"


# ---------------------------------------------------------------------------
# The dependency graph
# ---------------------------------------------------------------------------


def test_a_dependency_on_a_missing_task_is_refused():
    """A dependency that can never be satisfied would strand the task."""
    journal, _ = make_journal()
    with pytest.raises(TeamError) as info:
        journal.create_task(subject="one", blocked_by=["task-99"])
    assert info.value.code == "TEAM_TASK_NOT_FOUND"


def test_a_task_cannot_block_itself():
    journal, _ = make_journal()
    with pytest.raises(TeamError, match="cannot block itself"):
        assert_task_graph(
            TeamTask(id="task-1", revision=1, subject="x", blocked_by=("task-1",)),
            {},
        )


def test_a_duplicate_blocker_is_refused():
    journal, _ = make_journal()
    first = journal.create_task(subject="one")
    with pytest.raises(TeamError, match="twice"):
        journal.create_task(subject="two", blocked_by=[first.id, first.id])


def test_a_cycle_is_refused():
    """Every task in a cycle becomes permanently unclaimable.

    The cycle check runs over the *candidate* graph, so a change that would
    introduce one is refused before it is written.
    """
    journal, _ = make_journal()
    a = journal.create_task(subject="a")
    b = journal.create_task(subject="b", blocked_by=[a.id])

    with pytest.raises(TeamError) as info:
        journal.update_task(
            a.id,
            expected_revision=a.revision,
            actor="lead",
            is_lead=True,
            action="set_dependencies",
            blocked_by=[b.id],
        )
    assert info.value.code == "TEAM_TASK_DEPENDENCY_CYCLE"


def test_a_three_task_cycle_is_refused():
    """The check is a real DFS, not a two-node special case."""
    journal, _ = make_journal()
    a = journal.create_task(subject="a")
    b = journal.create_task(subject="b", blocked_by=[a.id])
    c = journal.create_task(subject="c", blocked_by=[b.id])

    with pytest.raises(TeamError) as info:
        journal.update_task(
            a.id,
            expected_revision=a.revision,
            actor="lead",
            is_lead=True,
            action="set_dependencies",
            blocked_by=[c.id],
        )
    assert info.value.code == "TEAM_TASK_DEPENDENCY_CYCLE"


def test_a_refused_cycle_leaves_the_board_unchanged():
    """The refusal must happen before the write."""
    journal, session = make_journal()
    a = journal.create_task(subject="a")
    b = journal.create_task(subject="b", blocked_by=[a.id])
    before = len(session)

    with pytest.raises(TeamError):
        journal.update_task(
            a.id, expected_revision=a.revision, actor="lead", is_lead=True,
            action="set_dependencies", blocked_by=[b.id],
        )

    assert len(session) == before
    assert journal.tasks()[a.id].blocked_by == ()


def test_readiness_follows_the_blockers():
    """A task is ready only when every blocker is complete."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    first = journal.create_task(subject="first")
    second = journal.create_task(subject="second", blocked_by=[first.id])

    assert journal.task_ready(first)
    assert not journal.task_ready(second)

    journal.update_task(
        first.id, expected_revision=first.revision, actor="alice", action="claim"
    )
    current = journal.tasks()[first.id]
    journal.update_task(
        first.id, expected_revision=current.revision, actor="alice", action="complete"
    )

    assert journal.task_ready(journal.tasks()[second.id])


# ---------------------------------------------------------------------------
# Actions and authorization
# ---------------------------------------------------------------------------


def test_claiming_marks_in_progress():
    journal, _ = make_journal()
    add_member(journal, "alice")
    task = journal.create_task(subject="work")

    claimed = journal.update_task(
        task.id, expected_revision=task.revision, actor="alice", action="claim"
    )
    assert claimed.phase == TASK_IN_PROGRESS
    assert claimed.owner == "alice"
    assert claimed.revision == 2


def test_a_blocked_task_cannot_be_claimed():
    """Claiming blocked work defeats the point of declaring the dependency."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    first = journal.create_task(subject="first")
    second = journal.create_task(subject="second", blocked_by=[first.id])

    with pytest.raises(TeamError) as info:
        journal.update_task(
            second.id, expected_revision=second.revision, actor="alice", action="claim"
        )
    assert info.value.code == "TEAM_TASK_BLOCKED"
    assert first.id in str(info.value), "the error should name the blocker"


def test_a_task_claimed_by_another_member_cannot_be_taken():
    journal, _ = make_journal()
    add_member(journal, "alice")
    add_member(journal, "bob")
    task = journal.create_task(subject="work")
    journal.update_task(
        task.id, expected_revision=task.revision, actor="alice", action="claim"
    )

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id,
            expected_revision=journal.tasks()[task.id].revision,
            actor="bob",
            action="claim",
        )
    assert info.value.code == "TEAM_TASK_ALREADY_CLAIMED"


def test_a_stale_revision_is_refused():
    """A stale caller must be told, not silently obeyed."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    task = journal.create_task(subject="work")
    stale = task.revision

    journal.update_task(
        task.id, expected_revision=stale, actor="alice", action="claim"
    )

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id, expected_revision=stale, actor="alice", action="complete"
        )
    assert info.value.code == "TEAM_TASK_STALE_REVISION"
    assert "revision" in str(info.value).lower()


def test_only_the_owner_or_lead_may_complete():
    journal, _ = make_journal()
    add_member(journal, "alice")
    add_member(journal, "bob")
    task = journal.create_task(subject="work")
    journal.update_task(
        task.id, expected_revision=task.revision, actor="alice", action="claim"
    )

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id,
            expected_revision=journal.tasks()[task.id].revision,
            actor="bob",
            action="complete",
        )
    assert info.value.code == "TEAM_TASK_UNAUTHORIZED"


def test_the_lead_may_complete_another_members_task():
    """The lead owns the log, so it can act on any task."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    task = journal.create_task(subject="work")
    journal.update_task(
        task.id, expected_revision=task.revision, actor="alice", action="claim"
    )

    done = journal.update_task(
        task.id,
        expected_revision=journal.tasks()[task.id].revision,
        actor="lead",
        is_lead=True,
        action="complete",
    )
    assert done.phase == TASK_COMPLETED


def test_only_the_lead_may_reassign():
    """Reassigning moves work between people, so it is the lead's alone."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    add_member(journal, "bob")
    task = journal.create_task(subject="work")

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id, expected_revision=task.revision, actor="alice",
            action="reassign", owner="bob",
        )
    assert info.value.code == "TEAM_LEAD_REQUIRED"


def test_the_lead_can_reassign_to_an_active_member():
    journal, _ = make_journal()
    add_member(journal, "alice")
    task = journal.create_task(subject="work")

    reassigned = journal.update_task(
        task.id, expected_revision=task.revision, actor="lead", is_lead=True,
        action="reassign", owner="alice",
    )
    assert reassigned.owner == "alice"
    assert reassigned.phase == TASK_IN_PROGRESS


def test_reassigning_to_an_unknown_member_is_refused():
    journal, _ = make_journal()
    task = journal.create_task(subject="work")

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id, expected_revision=task.revision, actor="lead", is_lead=True,
            action="reassign", owner="nobody",
        )
    assert info.value.code == "TEAM_MEMBER_NOT_FOUND"


def test_reassigning_to_nobody_returns_the_task_to_the_pool():
    """An empty owner is how work is handed back."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    task = journal.create_task(subject="work")
    journal.update_task(
        task.id, expected_revision=task.revision, actor="alice", action="claim"
    )

    released = journal.update_task(
        task.id,
        expected_revision=journal.tasks()[task.id].revision,
        actor="lead",
        is_lead=True,
        action="reassign",
        owner="",
    )
    assert released.phase == TASK_PENDING
    assert released.owner == ""


def test_a_deleted_task_cannot_be_edited():
    journal, _ = make_journal()
    task = journal.create_task(subject="work")
    journal.update_task(
        task.id, expected_revision=task.revision, actor="lead", is_lead=True,
        action="delete",
    )

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id, expected_revision=journal.tasks()[task.id].revision,
            actor="lead", is_lead=True, action="edit", subject="new",
        )
    assert info.value.code == "TEAM_TASK_DELETED"


def test_a_task_with_dependents_cannot_be_deleted():
    """A deleted task can never satisfy a dependency, so its dependents would
    be stranded permanently."""
    journal, _ = make_journal()
    first = journal.create_task(subject="first")
    journal.create_task(subject="second", blocked_by=[first.id])

    with pytest.raises(TeamError) as info:
        journal.update_task(
            first.id, expected_revision=first.revision, actor="lead",
            is_lead=True, action="delete",
        )
    assert info.value.code == "TEAM_TASK_HAS_DEPENDENTS"


def test_a_task_without_dependents_can_be_deleted():
    journal, _ = make_journal()
    task = journal.create_task(subject="work")
    deleted = journal.update_task(
        task.id, expected_revision=task.revision, actor="lead", is_lead=True,
        action="delete",
    )
    assert deleted.deleted
    assert task.id not in journal.live_tasks()


def test_completing_a_pending_task_is_refused():
    """Completion is a transition from in-progress, not from anywhere."""
    journal, _ = make_journal()
    task = journal.create_task(subject="work")

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id, expected_revision=task.revision, actor="lead", is_lead=True,
            action="complete",
        )
    assert info.value.code == "TEAM_INVALID_ARGUMENT"


def test_reopening_requires_a_completed_task():
    journal, _ = make_journal()
    task = journal.create_task(subject="work")

    with pytest.raises(TeamError):
        journal.update_task(
            task.id, expected_revision=task.revision, actor="lead", is_lead=True,
            action="reopen",
        )


def test_reopening_a_completed_task_works():
    journal, _ = make_journal()
    add_member(journal, "alice")
    task = journal.create_task(subject="work")
    journal.update_task(
        task.id, expected_revision=task.revision, actor="alice", action="claim"
    )
    journal.update_task(
        task.id,
        expected_revision=journal.tasks()[task.id].revision,
        actor="alice",
        action="complete",
    )

    reopened = journal.update_task(
        task.id,
        expected_revision=journal.tasks()[task.id].revision,
        actor="alice",
        action="reopen",
    )
    assert reopened.phase == TASK_IN_PROGRESS


def test_an_empty_edit_is_refused():
    """An edit that changes nothing would burn a revision for nothing."""
    journal, _ = make_journal()
    task = journal.create_task(subject="work")

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id, expected_revision=task.revision, actor="lead", is_lead=True,
            action="edit",
        )
    assert info.value.code == "TEAM_INVALID_ARGUMENT"


def test_an_edit_changes_the_subject():
    journal, _ = make_journal()
    task = journal.create_task(subject="old")

    edited = journal.update_task(
        task.id, expected_revision=task.revision, actor="lead", is_lead=True,
        action="edit", subject="new",
    )
    assert edited.subject == "new"
    assert edited.revision == 2


def test_set_dependencies_requires_the_list():
    journal, _ = make_journal()
    task = journal.create_task(subject="work")

    with pytest.raises(TeamError, match="blocked_by"):
        journal.update_task(
            task.id, expected_revision=task.revision, actor="lead", is_lead=True,
            action="set_dependencies",
        )


def test_an_unknown_action_is_refused_and_lists_the_valid_ones():
    """The error must be actionable."""
    journal, _ = make_journal()
    task = journal.create_task(subject="work")

    with pytest.raises(TeamError) as info:
        journal.update_task(
            task.id, expected_revision=task.revision, actor="lead", is_lead=True,
            action="teleport",
        )
    assert "claim" in str(info.value)
    assert "complete" in str(info.value)


def test_operating_on_an_unknown_task_is_refused():
    journal, _ = make_journal()
    with pytest.raises(TeamError) as info:
        journal.update_task(
            "task-99", expected_revision=1, actor="lead", is_lead=True, action="claim"
        )
    assert info.value.code == "TEAM_TASK_NOT_FOUND"


# ===========================================================================
# Summary
# ===========================================================================


def test_the_summary_counts_everything():
    """A compact view for a caller checking progress."""
    journal, _ = make_journal()
    add_member(journal, "alice")
    journal.add_member(TeamMember(name="bob"))  # still provisioning
    first = journal.create_task(subject="one")
    journal.create_task(subject="two", blocked_by=[first.id])
    journal.send(message("lead", "alice"))

    summary = journal.summary()
    assert summary["members"] == {
        "total": 2, "active": 1, "provisioning": 1, "failed": 0
    }
    assert summary["tasks"]["total"] == 2
    assert summary["tasks"]["ready"] == 1, "the blocked task is not ready"
    assert summary["mailbox"] == {"queued": 1, "delivered": 0}


# ===========================================================================
# Limits are validated
# ===========================================================================


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_members": 0},
        {"max_members": -1},
        {"max_tasks": 0},
        {"max_pending_per_member": 0},
        {"max_message_bytes": 0},
    ],
)
def test_a_non_positive_limit_is_refused_at_construction(kwargs):
    """A limit of zero means "nothing is allowed", which is never the intent."""
    session = Session.create("lead")
    with pytest.raises(TeamError, match="positive integer"):
        TeamJournal(session, **kwargs)


def test_the_journal_defaults_its_root_to_the_session():
    """A single-session team needs no explicit root."""
    journal, _ = make_journal()
    assert journal.root_id == "lead"


# ===========================================================================
# The tools
# ===========================================================================


async def invoke(tool, arguments: dict) -> Any:
    """Call a tool body and return its result."""
    return await tool.body(arguments, None)


def make_tools(*, with_delegate: bool = True, **kwargs):
    """Build the team tools.

    ``delegate`` is injected by default, because a runtime that cannot create
    teammates is the exception rather than the rule -- and the test for that case
    passes ``with_delegate=False`` explicitly.
    """
    from arm.team.tools import TeamTools

    journal, session = make_journal(**kwargs)
    delegate = (lambda name, task: f"run-{name}") if with_delegate else None
    tools = {
        t.name: t
        for t in TeamTools(journal, delegate=delegate).tools()
    }
    return tools, journal, session


def test_the_five_team_tools_exist():
    """Roster, mailbox, status, and the two task operations."""
    tools, _journal, _session = make_tools()
    assert sorted(tools) == [
        "team_send",
        "team_spawn",
        "team_status",
        "team_task_create",
        "team_task_update",
    ]


def test_spawn_adds_an_active_member():
    """The ordinary path: recorded, started, then activated."""
    import asyncio

    tools, journal, _session = make_tools()
    result = asyncio.run(
        invoke(tools["team_spawn"], {"name": "alice", "task": "do the thing"})
    )

    assert not result.is_error
    member = journal.members()["alice"]
    assert member.active
    assert member.child_session == "run-alice", (
        "the member was activated without recording what runs it"
    )


def test_spawn_without_a_delegate_refuses_rather_than_lying():
    """A teammate that exists only on the roster is worse than none.

    Reporting "now active" would leave every message to it undelivered forever,
    and nothing would report why.
    """
    import asyncio

    tools, journal, _session = make_tools(with_delegate=False)
    result = asyncio.run(
        invoke(tools["team_spawn"], {"name": "alice", "task": "do the thing"})
    )

    assert result.is_error
    assert result.error_code == "TEAM_NO_DELEGATION"
    assert journal.members()["alice"].phase == MEMBER_FAILED, (
        "the member was left provisioning instead of failed"
    )


def test_spawn_refuses_an_invalid_name():
    """And the error names the shape that is expected."""
    import asyncio

    tools, _journal, _session = make_tools()
    result = asyncio.run(
        invoke(tools["team_spawn"], {"name": "Alice Smith", "task": "x"})
    )
    assert result.is_error
    assert result.error_code == "TEAM_INVALID_MEMBER_NAME"


def test_spawn_refuses_an_empty_task():
    """A teammate with nothing to do is a resource leak."""
    import asyncio

    tools, _journal, _session = make_tools()
    result = asyncio.run(invoke(tools["team_spawn"], {"name": "alice", "task": "  "}))
    assert result.is_error
    assert "needs an initial task" in result.content


def test_send_queues_a_message():
    """The ordinary path, with no delivery channel attached."""
    import asyncio

    tools, journal, _session = make_tools()
    asyncio.run(invoke(tools["team_spawn"], {"name": "alice", "task": "x"}))
    result = asyncio.run(
        invoke(tools["team_send"], {"name": "alice", "message": "how is it going?"})
    )

    assert not result.is_error
    assert len(journal.pending_for("alice")) == 1


def test_send_delivers_and_confirms_when_a_channel_exists():
    """A queued message that never arrives is the failure this prevents."""
    import asyncio

    from arm.team.tools import TeamTools

    journal, _session = make_journal()
    journal.add_member(TeamMember(name="alice"))
    journal.set_member_phase("alice", MEMBER_ACTIVE, child_session="run-alice")

    delivered: list[tuple[str, str]] = []
    tools = {
        t.name: t
        for t in TeamTools(
            journal,
            delegate=lambda name, task: f"run-{name}",
            deliver=lambda name, body: (delivered.append((name, body)), True)[1],
        ).tools()
    }

    result = asyncio.run(
        invoke(tools["team_send"], {"name": "alice", "message": "hello"})
    )

    assert not result.is_error
    assert delivered == [("alice", "hello")]
    assert result.value["delivered"] is True
    assert journal.pending_for("alice") == (), (
        "the message was delivered but not marked delivered"
    )


def test_send_reports_an_undelivered_message_honestly():
    """The sender must be able to tell "delivered" from "queued"."""
    import asyncio

    from arm.team.tools import TeamTools

    journal, _session = make_journal()
    journal.add_member(TeamMember(name="alice"))
    journal.set_member_phase("alice", MEMBER_ACTIVE, child_session="run-alice")

    tools = {
        t.name: t
        for t in TeamTools(
            journal,
            delegate=lambda name, task: f"run-{name}",
            deliver=lambda name, body: False,
        ).tools()
    }

    result = asyncio.run(
        invoke(tools["team_send"], {"name": "alice", "message": "hello"})
    )

    assert not result.is_error
    assert result.value["delivered"] is False
    assert len(journal.pending_for("alice")) == 1, (
        "an undelivered message was marked delivered"
    )


def test_send_refuses_an_unknown_member():
    """And the refusal is a result, not an exception."""
    import asyncio

    tools, _journal, _session = make_tools()
    result = asyncio.run(invoke(tools["team_send"], {"name": "nobody", "message": "hi"}))

    assert result.is_error
    assert result.error_code == "TEAM_MEMBER_NOT_FOUND"


def test_status_renders_the_whole_team():
    """One call gives the lead everything it needs to decide what to do."""
    import asyncio

    tools, _journal, _session = make_tools()
    asyncio.run(invoke(tools["team_spawn"], {"name": "alice", "task": "x"}))
    asyncio.run(invoke(tools["team_task_create"], {"subject": "first"}))
    asyncio.run(invoke(tools["team_send"], {"name": "alice", "message": "hi"}))

    result = asyncio.run(invoke(tools["team_status"], {}))
    assert not result.is_error
    assert "alice" in result.content
    assert "task-1" in result.content
    assert "1 queued" in result.content


def test_status_shows_a_blocked_task_with_its_blockers():
    """A blocked task must say what is blocking it."""
    import asyncio

    tools, _journal, _session = make_tools()
    asyncio.run(invoke(tools["team_task_create"], {"subject": "first"}))
    asyncio.run(
        invoke(
            tools["team_task_create"],
            {"subject": "second", "blocked_by": ["task-1"]},
        )
    )

    result = asyncio.run(invoke(tools["team_status"], {}))
    assert "blocked by task-1" in result.content


def test_task_create_returns_the_revision():
    """The caller needs it for the next call."""
    import asyncio

    tools, _journal, _session = make_tools()
    result = asyncio.run(invoke(tools["team_task_create"], {"subject": "work"}))

    assert not result.is_error
    assert result.value == {"id": "task-1", "revision": 1}


def test_task_update_requires_a_revision():
    """Without one, a stale caller could overwrite a decision it never saw."""
    import asyncio

    tools, _journal, _session = make_tools()
    asyncio.run(invoke(tools["team_task_create"], {"subject": "work"}))

    result = asyncio.run(invoke(tools["team_task_update"], {"id": "task-1", "action": "claim"}))
    assert result.is_error
    assert "revision" in result.content


def test_task_update_claims():
    """The ordinary path."""
    import asyncio

    tools, journal, _session = make_tools()
    asyncio.run(invoke(tools["team_spawn"], {"name": "alice", "task": "x"}))
    asyncio.run(invoke(tools["team_task_create"], {"subject": "work"}))

    result = asyncio.run(
        invoke(
            tools["team_task_update"],
            {"id": "task-1", "revision": 1, "action": "claim"},
        )
    )
    assert not result.is_error
    assert journal.tasks()["task-1"].phase == TASK_IN_PROGRESS


def test_task_update_reports_a_stale_revision():
    """The refusal must reach the model as a readable result."""
    import asyncio

    tools, _journal, _session = make_tools()
    asyncio.run(invoke(tools["team_task_create"], {"subject": "work"}))
    asyncio.run(
        invoke(tools["team_task_update"], {"id": "task-1", "revision": 1, "action": "claim"})
    )

    stale = asyncio.run(
        invoke(
            tools["team_task_update"],
            {"id": "task-1", "revision": 1, "action": "complete"},
        )
    )
    assert stale.is_error
    assert stale.error_code == "TEAM_TASK_STALE_REVISION"


def test_task_update_rejects_an_unknown_action():
    """And lists the valid ones, so the model can correct itself."""
    import asyncio

    tools, _journal, _session = make_tools()
    asyncio.run(invoke(tools["team_task_create"], {"subject": "work"}))

    result = asyncio.run(
        invoke(
            tools["team_task_update"],
            {"id": "task-1", "revision": 1, "action": "teleport"},
        )
    )
    assert result.is_error
    assert "claim" in result.content


def test_a_non_lead_cannot_reassign_through_the_tool():
    """The role is bound to the tool, not passed as an argument.

    A tool that accepted an ``as_lead`` flag would let any member claim the
    privilege by passing a boolean.
    """
    import asyncio

    from arm.team.tools import TeamTools

    journal, _session = make_journal()
    add_member(journal, "alice")
    journal.create_task(subject="work")
    member_tools = {t.name: t for t in TeamTools(journal, actor="alice", is_lead=False).tools()}

    result = asyncio.run(
        invoke(
            member_tools["team_task_update"],
            {"id": "task-1", "revision": 1, "action": "reassign", "owner": "alice"},
        )
    )
    assert result.is_error
    assert result.error_code == "TEAM_LEAD_REQUIRED"


def test_every_team_tool_schema_normalizes():
    """The schemas must survive the model-facing subset."""
    import json

    tools, _journal, _session = make_tools()
    for tool in tools.values():
        rendered = json.dumps(dict(tool.definition.parameters))
        assert "title" not in rendered
        assert "$defs" not in rendered


def test_the_status_tool_is_concurrency_safe():
    """Reading the board changes nothing, so it may run alongside other calls."""
    tools, _journal, _session = make_tools()
    assert tools["team_status"].definition.allows_parallel({})


def test_the_mutating_tools_are_not_concurrency_safe():
    """Two concurrent board updates would race on the revision."""
    tools, _journal, _session = make_tools()
    for name in ("team_task_create", "team_task_update", "team_spawn", "team_send"):
        assert not tools[name].definition.allows_parallel({}), name


# ===========================================================================
# The delegate must be awaitable from inside a running loop
# ===========================================================================


def test_an_async_delegate_is_awaited():
    """Tool bodies are always async, so a delegate must be awaitable.

    Calling ``asyncio.run`` inside one raises "cannot be called from a running
    event loop" -- which made ``team_spawn`` fail on every real invocation while
    every test passed, because the tests used a synchronous delegate.
    """
    import asyncio

    from arm.team.tools import TeamTools

    journal, _session = make_journal()
    started: list[str] = []

    async def delegate(name: str, task: str) -> str:
        await asyncio.sleep(0)
        started.append(name)
        return f"run-{name}"

    tools = {t.name: t for t in TeamTools(journal, delegate=delegate).tools()}
    result = asyncio.run(
        invoke(tools["team_spawn"], {"name": "alice", "task": "do it"})
    )

    assert not result.is_error, result.content
    assert started == ["alice"]
    assert journal.members()["alice"].child_session == "run-alice"


def test_an_async_deliver_is_awaited():
    import asyncio

    from arm.team.tools import TeamTools

    journal, _session = make_journal()
    journal.add_member(TeamMember(name="alice"))
    journal.set_member_phase("alice", MEMBER_ACTIVE, child_session="run-alice")

    async def deliver(name: str, body: str) -> bool:
        await asyncio.sleep(0)
        return True

    tools = {
        t.name: t
        for t in TeamTools(
            journal, delegate=lambda n, t: f"run-{n}", deliver=deliver
        ).tools()
    }
    result = asyncio.run(
        invoke(tools["team_send"], {"name": "alice", "message": "hi"})
    )

    assert not result.is_error, result.content
    assert result.value["delivered"] is True


def test_a_sync_delegate_still_works():
    """Both shapes are accepted, so a caller is not forced into one."""
    import asyncio

    from arm.team.tools import TeamTools

    journal, _session = make_journal()
    tools = {
        t.name: t
        for t in TeamTools(journal, delegate=lambda n, t: f"sync-{n}").tools()
    }
    result = asyncio.run(
        invoke(tools["team_spawn"], {"name": "bob", "task": "x"})
    )

    assert not result.is_error
    assert journal.members()["bob"].child_session == "sync-bob"

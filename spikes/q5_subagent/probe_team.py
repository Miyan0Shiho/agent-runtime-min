"""Probe: team journal reachability and task-board invariants.

Run:  cd spikes && ../.venv/bin/python q5_subagent/probe_team.py
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

import _bootstrap  # noqa: F401

from arm.session.session import Session
from arm.team.journal import (
    MEMBER_ACTIVE,
    TASK_COMPLETED,
    TASK_IN_PROGRESS,
    TASK_PENDING,
    TeamError,
    TeamJournal,
    TeamMember,
    TeamMessage,
)
from arm.team.tools import TeamTools


def journal(**kw):
    s = Session.create("lead")
    return TeamJournal(s, **kw), s


def add(j, name):
    j.add_member(TeamMember(name=name))
    j.set_member_phase(name, MEMBER_ACTIVE)


print("=" * 72)
print("A. does team_spawn create an agent?")
print("=" * 72)
j, s = journal()
tools = {t.name: t for t in TeamTools(j).tools()}
spawn = tools["team_spawn"]
result = asyncio.run(spawn.body({"name": "alice", "task": "review the parser"}, None))
print("spawn result content:", result.content.replace("\n", " | "))
member = j.members()["alice"]
print("member.child_session:", repr(member.child_session))
print("member.phase        :", member.phase)
print(">>> no child session, no agent: the 'teammate' is a roster row")

print()
print("=" * 72)
print("B. is a message ever delivered? confirm_delivery callers")
print("=" * 72)
send = tools["team_send"]
r = asyncio.run(send.body({"name": "alice", "message": "please review"}, None))
print("send result:", r.content)
print("pending_for('alice'):", len(j.pending_for("alice")))
print("delivered          :", [m.delivered for m in j.messages()])
out = subprocess.run(
    ["grep", "-rn", "--include=*.py", "confirm_delivery", "src/"],
    capture_output=True, text=True, cwd="..",
).stdout.strip()
print("src callers of confirm_delivery:\n ", out or "(none)")
print(">>> the mailbox is a write-only queue in the real system")

print()
print("=" * 72)
print("C. root_id is written but never used to filter")
print("=" * 72)
s2 = Session.create("shared")
a = TeamJournal(s2, root_id="team-a")
b = TeamJournal(s2, root_id="team-b")
add(a, "alice")
add(b, "bob")
print("journal A members:", sorted(a.members()))
print("journal B members:", sorted(b.members()))
print(">>> both journals fold the same events; root_id filters nothing")

print()
print("=" * 72)
print("D. diamond dependency and reopen")
print("=" * 72)
j3, _ = journal()
add(j3, "alice")
t1 = j3.create_task(subject="root", created_by="alice")
t2 = j3.create_task(subject="left", blocked_by=[t1.id], created_by="alice")
t3 = j3.create_task(subject="right", blocked_by=[t1.id], created_by="alice")
t4 = j3.create_task(subject="join", blocked_by=[t2.id, t3.id], created_by="alice")
print("diamond created:", [t.id for t in (t1, t2, t3, t4)])
print("t4 ready (expect False):", j3.task_ready(j3.tasks()[t4.id]))
j3.update_task(t1.id, expected_revision=1, actor="alice", action="claim")
j3.update_task(t1.id, expected_revision=2, actor="alice", action="complete")
print("after t1 complete -> t2 ready:", j3.task_ready(j3.tasks()[t2.id]),
      " t4 ready:", j3.task_ready(j3.tasks()[t4.id]))
j3.update_task(t2.id, expected_revision=1, actor="alice", action="claim")
j3.update_task(t2.id, expected_revision=2, actor="alice", action="complete")
j3.update_task(t3.id, expected_revision=1, actor="alice", action="claim")
j3.update_task(t3.id, expected_revision=2, actor="alice", action="complete")
print("after t2,t3 complete -> t4 ready:", j3.task_ready(j3.tasks()[t4.id]))
j3.update_task(t4.id, expected_revision=1, actor="alice", action="claim")
print("t4 claimed; now REOPEN t2")
j3.update_task(t2.id, expected_revision=3, actor="alice", action="reopen")
print("t2 phase:", j3.tasks()[t2.id].phase)
print("t4 phase:", j3.tasks()[t4.id].phase, "(in progress despite a reopened blocker)")
print("t4 ready now:", j3.task_ready(j3.tasks()[t4.id]))
print(">>> an in-progress task is never re-checked against reopened blockers")

print()
print("=" * 72)
print("E. id allocation termination")
print("=" * 72)
j4, _ = journal(max_tasks=3)
ids = [j4.create_task(subject=f"t{i}", created_by="lead").id for i in range(3)]
print("ids:", ids)
try:
    j4.create_task(subject="overflow", created_by="lead")
except TeamError as exc:
    print("4th ->", exc.code)
# Delete the first; ids are never reused, so the board is still full.
j4.update_task("task-1", expected_revision=1, actor="lead", is_lead=True, action="delete")
print("after deleting task-1, live:", sorted(j4.live_tasks()))
try:
    print("new task:", j4.create_task(subject="reuse", created_by="lead").id)
except TeamError as exc:
    print("create after delete ->", exc.code)
print(">>> terminates; ids are never reused")

print()
print("=" * 72)
print("F. the edit action cannot clear a description through the tool")
print("=" * 72)
j5, _ = journal()
t = j5.create_task(subject="s", description="original", created_by="lead")
upd = {x.name: x for x in TeamTools(j5).tools()}["team_task_update"]
r = asyncio.run(upd.body(
    {"id": t.id, "revision": 1, "action": "edit", "subject": "new", "description": ""}, None
))
print("edit with description='' ->", r.content)
print("description now:", repr(j5.tasks()[t.id].description))
print(">>> an empty description is silently ignored (cannot be cleared)")

print()
print("=" * 72)
print("G. write_scopes: is anything enforced?")
print("=" * 72)
t = j5.create_task(subject="w", write_scopes=["src/a", "src/a", "src/b"], created_by="lead")
print("write_scopes stored:", t.write_scopes)
print(">>> advisory metadata only; tool description says so explicitly")

print()
print("=" * 72)
print("H. lead privileges are hard-coded in the registered tools")
print("=" * 72)
tt = TeamTools(j5)
print("actor:", tt.actor, " is_lead:", tt.is_lead)
print(">>> every registered team tool acts as the lead; no member-facing path")

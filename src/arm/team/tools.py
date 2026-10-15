"""The team tools: how the lead runs a team.

Five tools, and the split between them follows the split between the three
projections. A tool that did two things would make an error message ambiguous --
"the send failed" and "the task could not be claimed" need different responses.

The design decision worth stating: **the lead is a role, not a tool argument.**
Tools that only the lead may use say so in their description, and the journal
enforces it. A tool that accepted an ``as_lead`` flag would let any member claim
the privilege by passing a boolean.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .journal import (
    MEMBER_ACTIVE,
    MEMBER_FAILED,
    MEMBER_PROVISIONING,
    TeamError,
    TeamJournal,
    TeamMember,
    TeamMessage,
)


async def _maybe_await(value: Any) -> Any:
    """Await a value if it is awaitable, otherwise return it.

    A delegate is naturally async -- it runs an agent -- but a caller may also
    supply a plain function. Accepting both avoids forcing one shape, and it is
    the difference between working inside a running event loop and calling
    ``asyncio.run`` from one, which raises.
    """
    if inspect.isawaitable(value):
        return await value
    return value


class SpawnParams(BaseModel):
    name: str = Field(
        description=(
            "The teammate's name, kebab-case and unique in the team. It is what "
            "messages are addressed to."
        )
    )
    task: str = Field(
        description=(
            "The teammate's initial task. It starts with no context, so state "
            "everything it needs."
        )
    )
    label: str = Field(default="", description="A short display label.")


class SendParams(BaseModel):
    name: str = Field(description="The teammate to message.")
    message: str = Field(description="The message body.")


class TaskCreateParams(BaseModel):
    subject: str = Field(description="A one-line summary of the work.")
    description: str = Field(default="", description="The full detail.")
    blocked_by: list[str] = Field(
        default_factory=list,
        description="Task ids that must complete before this one can start.",
    )
    write_scopes: list[str] = Field(
        default_factory=list,
        description=(
            "Advisory file or directory prefixes this task expects to modify. "
            "They are recorded for coordination; they are not a lock."
        ),
    )


class TaskUpdateParams(BaseModel):
    id: str = Field(description="The task id, e.g. 'task-1'.")
    revision: int = Field(
        description=(
            "The revision you read. A mismatch means your view is stale and the "
            "update is refused; read the board again and retry."
        )
    )
    action: str = Field(
        description=(
            "One of: claim, release, complete, reopen, edit, set_dependencies, "
            "reassign, delete."
        )
    )
    subject: str = Field(default="", description="New subject, for 'edit'.")
    description: str = Field(default="", description="New description, for 'edit'.")
    blocked_by: list[str] = Field(
        default_factory=list, description="New blockers, for 'set_dependencies'."
    )
    owner: str = Field(
        default="",
        description="The new owner for 'reassign'; empty returns it to the pool.",
    )


class TeamTools:
    """The team tool set, bound to one team's journal.

    ``spawn`` and ``send`` need a way to reach a teammate's agent. That is
    injected as ``delegate``: a callable that runs a task on a fresh agent and
    returns its output. Without it, spawning would write a roster row and report
    success while creating nothing -- a teammate that exists only on paper.
    """

    def __init__(
        self,
        journal: TeamJournal,
        *,
        actor: str = "lead",
        is_lead: bool = True,
        delegate: Any = None,
        deliver: Any = None,
    ) -> None:
        self.journal = journal
        self.actor = actor
        self.is_lead = is_lead
        #: Runs a task on a new agent. Absent means this runtime cannot actually
        #: create teammates, and ``team_spawn`` says so rather than lying.
        self.delegate = delegate
        #: Delivers a message to a member's agent, returning whether it landed.
        self.deliver = deliver

    def tools(self) -> list[RegisteredTool]:
        return [
            self._spawn(),
            self._send(),
            self._status(),
            self._task_create(),
            self._task_update(),
        ]

    # -- roster ----------------------------------------------------------

    def _spawn(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            name = str(arguments.get("name", "")).strip()
            task = str(arguments.get("task", "")).strip()
            if not task:
                return ToolResult.fail("a teammate needs an initial task")

            try:
                # Recorded before the teammate exists, so a crash leaves a member
                # that reconciliation can resolve rather than an agent nothing
                # knows about.
                self.journal.add_member(
                    TeamMember(
                        name=name,
                        phase=MEMBER_PROVISIONING,
                        label=str(arguments.get("label", "")),
                    )
                )
            except TeamError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            if self.delegate is None:
                # No way to create an agent. Reporting "now active" would be a
                # lie: the roster would show a teammate that does not exist, and
                # every message to it would sit undelivered forever.
                self.journal.set_member_phase(
                    name, MEMBER_FAILED, error="this runtime cannot create teammates"
                )
                return ToolResult.fail(
                    "this runtime cannot create teammates: no subagent runtime is "
                    "attached, so the member would exist only on the roster",
                    code="TEAM_NO_DELEGATION",
                )

            try:
                run_id = await _maybe_await(self.delegate(name, task))
            except Exception as exc:  # noqa: BLE001 - a failed spawn is a state
                self.journal.set_member_phase(name, MEMBER_FAILED, error=str(exc))
                return ToolResult.fail(
                    f"recorded {name!r} but could not start it: {exc}",
                    code="TEAM_PROVISIONING_FAILED",
                )

            try:
                member = self.journal.set_member_phase(
                    name, MEMBER_ACTIVE, child_session=str(run_id or "")
                )
            except TeamError as exc:
                self.journal.set_member_phase(name, MEMBER_FAILED, error=str(exc))
                return ToolResult.fail(
                    f"recorded {name!r} but could not activate it: {exc}",
                    code=exc.code,
                )

            return ToolResult.ok(
                f"Added teammate {member.name} (now active).\n"
                f"Its task: {task}\n"
                f"Message it with team_send using the name {member.name!r}.",
                value={"name": member.name, "phase": member.phase},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="team_spawn",
                description=(
                    "Add a teammate to this team. Only the lead may do this. The "
                    "teammate starts with no context, so its initial task must be "
                    "self-contained."
                ),
                parameters=SpawnParams.model_json_schema(),
            ),
            body=body,
            validator=SpawnParams,
        )

    def _send(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            import uuid

            name = str(arguments.get("name", "")).strip()
            text = str(arguments.get("message", ""))
            if not text:
                return ToolResult.fail("a message needs a body")

            message_id = str(uuid.uuid4())
            try:
                self.journal.send(
                    TeamMessage(
                        message_id=message_id,
                        sender=self.actor,
                        recipient=name,
                        body=text,
                    )
                )
            except TeamError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            # Delivery is attempted immediately, and recorded only if the
            # recipient's own agent confirms it. A message that is merely queued
            # and never delivered is the failure this exists to prevent: the
            # sender would believe it had communicated.
            if self.deliver is not None:
                try:
                    landed = bool(await _maybe_await(self.deliver(name, text)))
                except Exception as exc:  # noqa: BLE001 - a delivery fault is a result
                    landed = False
                    note = f" (delivery failed: {exc})"
                else:
                    note = ""
                if landed:
                    self.journal.confirm_delivery(message_id, name)
                    return ToolResult.ok(
                        f"Delivered to {name}.",
                        value={"recipient": name, "delivered": True},
                    )
                return ToolResult.ok(
                    f"Queued for {name}; it was not delivered yet{note}. It will "
                    "be delivered on the member's next run.",
                    value={"recipient": name, "delivered": False},
                )

            return ToolResult.ok(
                f"Queued for {name}. It will be delivered when they next run.",
                value={"recipient": name, "delivered": False},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="team_send",
                description=(
                    "Send a message to a teammate. Messages are queued and "
                    "delivered in order; delivery is confirmed before it is "
                    "recorded as delivered."
                ),
                parameters=SendParams.model_json_schema(),
            ),
            body=body,
            validator=SendParams,
        )

    # -- status ----------------------------------------------------------

    def _status(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            summary = self.journal.summary()
            members = self.journal.members()
            tasks = self.journal.live_tasks()

            lines: list[str] = []
            m = summary["members"]
            lines.append(
                f"Members: {m['active']} active, {m['provisioning']} provisioning, "
                f"{m['failed']} failed"
            )
            for member in sorted(members.values(), key=lambda x: x.name):
                note = f" — {member.error}" if member.error else ""
                lines.append(f"  {member.name}: {member.phase}{note}")

            t = summary["tasks"]
            lines.append(
                f"Tasks: {t['total']} total, {t['pending']} pending "
                f"({t['ready']} ready), {t['in_progress']} in progress, "
                f"{t['completed']} completed"
            )
            for task in sorted(tasks.values(), key=lambda x: x.id):
                owner = f" [{task.owner}]" if task.owner else ""
                blocked = (
                    f" (blocked by {', '.join(task.blocked_by)})"
                    if task.blocked_by and not self.journal.task_ready(task)
                    else ""
                )
                lines.append(
                    f"  {task.id} rev{task.revision} {task.phase}{owner}"
                    f"{blocked}: {task.subject}"
                )

            mb = summary["mailbox"]
            lines.append(
                f"Mailbox: {mb['queued']} queued, {mb['delivered']} delivered"
            )

            return ToolResult.ok("\n".join(lines), value=summary)

        return RegisteredTool(
            definition=ToolDefinition(
                name="team_status",
                description=(
                    "Read the team's state: members, the task board, and the "
                    "mailbox. Read the task revisions here before updating a task."
                ),
                parameters={"type": "object", "properties": {}},
                concurrency_safe=lambda _args: True,
            ),
            body=body,
        )

    # -- task board ------------------------------------------------------

    def _task_create(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            try:
                task = self.journal.create_task(
                    subject=str(arguments.get("subject", "")),
                    description=str(arguments.get("description", "")),
                    created_by=self.actor,
                    blocked_by=[str(b) for b in (arguments.get("blocked_by") or ())],
                    write_scopes=[str(s) for s in (arguments.get("write_scopes") or ())],
                )
            except TeamError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            return ToolResult.ok(
                f"Created {task.id} (revision {task.revision}): {task.subject}",
                value={"id": task.id, "revision": task.revision},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="team_task_create",
                description=(
                    "Add a task to the team's board. Tasks may declare blockers, "
                    "which form a dependency graph; a blocked task cannot be "
                    "claimed until its blockers complete."
                ),
                parameters=TaskCreateParams.model_json_schema(),
            ),
            body=body,
            validator=TaskCreateParams,
        )

    def _task_update(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            revision = arguments.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool):
                return ToolResult.fail(
                    "team_task_update requires the integer 'revision' you read; "
                    "call team_status first"
                )

            action = str(arguments.get("action", "")).strip()
            changes: dict[str, Any] = {}
            if action == "edit":
                if arguments.get("subject"):
                    changes["subject"] = str(arguments["subject"])
                # ``description`` is passed through whenever the key is present,
                # including when it is empty. Testing for truthiness would make an
                # empty string mean "leave it alone", so a description could never
                # be cleared -- the caller would have no way to say "remove this".
                if "description" in arguments:
                    changes["description"] = str(arguments.get("description") or "")
            elif action == "set_dependencies":
                changes["blocked_by"] = [
                    str(b) for b in (arguments.get("blocked_by") or ())
                ]
            elif action == "reassign":
                changes["owner"] = str(arguments.get("owner", ""))

            try:
                task = self.journal.update_task(
                    str(arguments.get("id", "")),
                    expected_revision=revision,
                    actor=self.actor,
                    is_lead=self.is_lead,
                    action=action,
                    **changes,
                )
            except TeamError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            return ToolResult.ok(
                f"{task.id} is now {task.phase} at revision {task.revision}"
                + (f", owned by {task.owner}" if task.owner else ""),
                value={
                    "id": task.id,
                    "revision": task.revision,
                    "phase": task.phase,
                    "owner": task.owner,
                },
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="team_task_update",
                description=(
                    "Change a task. Actions: 'claim' (take it), 'release' (give it "
                    "back), 'complete', 'reopen', 'edit', 'set_dependencies', "
                    "'reassign' (lead only), 'delete'.\n"
                    "The revision must match the one you read; a mismatch means "
                    "someone changed it first, so read the board again."
                ),
                parameters=TaskUpdateParams.model_json_schema(),
            ),
            body=body,
            validator=TaskUpdateParams,
        )

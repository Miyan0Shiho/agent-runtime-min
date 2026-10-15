"""The schedule tools: creating, listing and cancelling reminders.

Three tools. The interesting one is ``schedule_create``, and what makes it
interesting is what it refuses to guess.

**The title is required and is never derived.** It would be easy to generate one
from the instruction, and it would be wrong: a derived title is a summary of
something the user wrote, and a bad summary misrepresents what the reminder is
for. A missing title is a question to ask, not a gap to fill.

**The time is required, and it must be unambiguous.** A naive timestamp is
refused rather than assumed local, because the assumption is invisible and it is
wrong exactly for the reminders that cross a daylight-saving boundary.

**The instruction is what the model will be told.** It is separate from the title
because a list needs to be scannable and a prompt needs to be complete, and
optimising one for the other makes both worse.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..session.envelope import utc_now_iso
from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .runtime import (
    MAX_INTERVAL_SECONDS,
    KIND_AFTER,
    KIND_AT,
    KIND_CRON,
    KIND_DAILY,
    KIND_EVERY,
    KIND_WEEKLY,
    STATUS_ACTIVE,
    ScheduleError,
    ScheduleRecord,
    ScheduleStore,
)


class ScheduleCreateParams(BaseModel):
    title: str = Field(
        description=(
            "A short name for this reminder, for a list. Required, and it is "
            "never derived from the instruction -- say what it is for."
        )
    )
    instruction: str = Field(
        description=(
            "What to tell yourself when this fires. It should stand alone: the "
            "reminder arrives in a session that may have no other context."
        )
    )
    kind: str = Field(
        description=(
            "One of: 'after' (once, N seconds from now), 'at' (once, at an "
            "instant), 'every' (repeating interval), 'daily', 'weekly', 'cron'."
        )
    )
    after_seconds: int = Field(default=0, description="For 'after': the delay.")
    at: str = Field(
        default="",
        description=(
            "For 'at': an ISO-8601 instant **with a timezone**, e.g. "
            "'2026-10-14T09:00:00+08:00'."
        ),
    )
    every_seconds: int = Field(default=0, description="For 'every': the interval.")
    at_time: str = Field(
        default="", description="For 'daily'/'weekly': the local time, HH:MM."
    )
    weekdays: list[int] = Field(
        default_factory=list,
        description="For 'weekly': ISO weekdays (Monday is 1, Sunday is 7).",
    )
    cron: str = Field(default="", description="For 'cron': a five-field expression.")
    timezone: str = Field(
        default="",
        description=(
            "For 'daily'/'weekly': the IANA timezone the time of day is in, e.g. "
            "'Asia/Shanghai'. Required in practice: without it the time is "
            "resolved in the system zone, which is rarely what was meant."
        ),
    )


class ScheduleListParams(BaseModel):
    """No arguments: listing is always safe."""


class ScheduleCancelParams(BaseModel):
    id: str = Field(description="The schedule id, e.g. 'sch-3'.")


class ScheduleTools:
    """The schedule tool set, bound to one session and one store."""

    def __init__(self, store: ScheduleStore, *, session_id: str) -> None:
        self.store = store
        self.session_id = session_id

    def tools(self) -> list[RegisteredTool]:
        return [self._create(), self._list(), self._cancel(), self._history()]

    def _create(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            kind = str(arguments.get("kind", "")).strip()
            title = str(arguments.get("title", ""))
            instruction = str(arguments.get("instruction", ""))
            now = datetime.now(timezone.utc)
            schedule_id = f"sch-{uuid.uuid4().hex[:8]}"

            fields: dict[str, Any] = {
                "id": schedule_id,
                "title": title,
                "instruction": instruction,
                "session_id": self.session_id,
                "kind": kind,
                "status": STATUS_ACTIVE,
                "created_at": utc_now_iso(),
            }

            if kind == KIND_AFTER:
                seconds = int(arguments.get("after_seconds") or 0)
                if seconds < 1:
                    return ToolResult.fail("'after' needs a positive after_seconds")
                # Bounded *before* the arithmetic: adding a huge number of
                # seconds to a datetime raises OverflowError, which would escape
                # as a crash rather than a refusal. The bound is applied here
                # because this is where the addition happens.
                if seconds > MAX_INTERVAL_SECONDS:
                    return ToolResult.fail(
                        f"after_seconds must be at most {MAX_INTERVAL_SECONDS}, "
                        f"got {seconds}"
                    )
                fields["after_seconds"] = seconds
                fields["at_instant"] = (now + timedelta(seconds=seconds)).isoformat()
                fields["scheduled_at"] = fields["at_instant"]
            elif kind == KIND_AT:
                instant = str(arguments.get("at", "")).strip()
                if not instant:
                    return ToolResult.fail(
                        "'at' needs an instant, with a timezone"
                    )
                fields["at_instant"] = instant
                fields["scheduled_at"] = instant
            elif kind == KIND_EVERY:
                seconds = int(arguments.get("every_seconds") or 0)
                if seconds < 1:
                    return ToolResult.fail("'every' needs a positive every_seconds")
                if seconds > MAX_INTERVAL_SECONDS:
                    return ToolResult.fail(
                        f"every_seconds must be at most {MAX_INTERVAL_SECONDS}, "
                        f"got {seconds}"
                    )
                fields["every_seconds"] = seconds
                fields["anchor"] = now.isoformat()
                fields["scheduled_at"] = (now + timedelta(seconds=seconds)).isoformat()
            elif kind in (KIND_DAILY, KIND_WEEKLY):
                at_time = str(arguments.get("at_time", "")).strip()
                if not at_time:
                    return ToolResult.fail(f"'{kind}' needs a time of day (HH:MM)")
                fields["at_time"] = at_time
                fields["timezone"] = str(arguments.get("timezone", "")).strip()
                if kind == KIND_WEEKLY:
                    weekdays = [int(d) for d in (arguments.get("weekdays") or ())]
                    if not weekdays:
                        return ToolResult.fail("'weekly' needs at least one weekday")
                    fields["weekdays"] = tuple(weekdays)
            elif kind == KIND_CRON:
                expression = str(arguments.get("cron", "")).strip()
                if not expression:
                    return ToolResult.fail("'cron' needs an expression")
                fields["cron"] = expression
                fields["timezone"] = str(arguments.get("timezone", "")).strip()
            else:
                return ToolResult.fail(
                    f"unknown kind {kind!r}; expected one of: after, at, every, "
                    "daily, weekly, cron"
                )

            try:
                record = ScheduleRecord(**fields)
                # Computed once at creation so the caller is told when it will
                # first fire -- a schedule whose first fire time is wrong is
                # discovered too late to be useful.
                first = record.due_at()
                if first is None and record.recurring:
                    from .runtime import next_occurrence

                    first = next_occurrence(record, now=now, fired_at=now)
                if first is not None:
                    record = _with_scheduled_at(record, first.isoformat())
            except ScheduleError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            try:
                self.store.put(record)
            except ScheduleError as exc:
                return ToolResult.fail(str(exc), code=exc.code)

            when = record.due_at()
            return ToolResult.ok(
                f"Scheduled {record.id}: {record.title}\n"
                f"First fires at {when.isoformat() if when else 'soon'}.",
                value={
                    "id": record.id,
                    "kind": record.kind,
                    "scheduled_at": when.isoformat() if when else "",
                },
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="schedule_create",
                description=(
                    "Schedule a reminder for this session. It fires even if the "
                    "session is not loaded, and after a restart.\n"
                    "A recurring reminder that was missed delivers only its most "
                    "recent occurrence, not every one it skipped.\n"
                    "The title is for a list and must be given; it is never "
                    "derived from the instruction."
                ),
                parameters=ScheduleCreateParams.model_json_schema(),
            ),
            body=body,
            validator=ScheduleCreateParams,
        )

    def _list(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            tasks = self.store.for_session(self.session_id)
            if not tasks:
                return ToolResult.ok("No reminders are scheduled for this session.")

            lines: list[str] = []
            for task in sorted(tasks, key=lambda t: t.id):
                when = task.due_at()
                lines.append(
                    f"{task.id} [{task.status}] {task.kind} "
                    f"{when.isoformat() if when else '(no time)'}: {task.title}"
                )
                lines.append(f"    {task.instruction}")
            return ToolResult.ok("\n".join(lines), value={"count": len(tasks)})

        return RegisteredTool(
            definition=ToolDefinition(
                name="schedule_list",
                description="List this session's scheduled reminders.",
                parameters={"type": "object", "properties": {}},
                concurrency_safe=lambda _args: True,
            ),
            body=body,
        )

    def _history(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            limit = int(arguments.get("limit", 20))
            receipts = self.store.history(limit=limit)
            if not receipts:
                return ToolResult.ok(
                    "No reminders have been delivered yet.",
                    value={"count": 0},
                )
            lines = ["Recent reminder deliveries (newest first):"]
            for receipt in receipts:
                title = ""
                task = self.store.get(receipt.schedule_id)
                if task is not None:
                    title = task.title
                lines.append(
                    f"  {receipt.delivered_at} {receipt.schedule_id} "
                    f"{title or '(deleted)'}"
                )
            return ToolResult.ok(
                "\n".join(lines), value={"count": len(receipts)}
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="schedule_history",
                description=(
                    "Show when reminders were actually delivered. A receipt means "
                    "the reminder reached this session's inbox -- it does not mean "
                    "the model read it or acted on it."
                ),
                parameters=ScheduleHistoryParams.model_json_schema(),
                concurrency_safe=lambda _args: True,
            ),
            body=body,
            validator=ScheduleHistoryParams,
        )

    def _cancel(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            schedule_id = str(arguments.get("id", "")).strip()
            task = self.store.get(schedule_id)
            if task is None:
                return ToolResult.fail(f"no reminder {schedule_id!r}")
            if task.session_id != self.session_id:
                # A reminder belongs to the session that created it; cancelling
                # another session's would be a cross-session side effect nobody
                # asked for.
                return ToolResult.fail(
                    f"reminder {schedule_id!r} belongs to another session"
                )
            try:
                self.store.delete(schedule_id)
            except ScheduleError as exc:
                return ToolResult.fail(str(exc), code=exc.code)
            return ToolResult.ok(f"Cancelled {schedule_id} ({task.title}).")

        return RegisteredTool(
            definition=ToolDefinition(
                name="schedule_cancel",
                description="Cancel a reminder created by this session.",
                parameters=ScheduleCancelParams.model_json_schema(),
            ),
            body=body,
            validator=ScheduleCancelParams,
        )


class ScheduleHistoryParams(BaseModel):
    limit: int = Field(
        default=20, description="How many recent deliveries to show."
    )


def _with_scheduled_at(record: ScheduleRecord, when: str) -> ScheduleRecord:
    """A copy of a record with its next fire time set."""
    from .runtime import _replace

    return _replace(record, scheduled_at=when)

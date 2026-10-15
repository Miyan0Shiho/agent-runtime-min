"""The ``todo_write`` tool.

One tool, one operation: replace the list. There is no add, no update, no
delete, and that is the point -- a single operation has a single code path, and
the log entry it writes is self-describing, so a reader needs no prior state to
know what the list became.

The result the model gets back is one line of counts. Not the rendered list:
the model just supplied the list, so echoing it would spend tokens restating
something it already has. What it needs to know is that the write succeeded and
what the totals are.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .store import STATUS_PENDING, TodoError, TodoItem, TodoStore


class TodoItemParams(BaseModel):
    content: str = Field(description="What the task is. Must be unique in the list.")
    status: str = Field(
        default=STATUS_PENDING,
        description="One of: pending, in_progress, completed.",
    )


class TodoWriteParams(BaseModel):
    todos: list[TodoItemParams] = Field(
        default_factory=list,
        description=(
            "The complete list, replacing whatever was there. At most one task "
            "may be in_progress."
        ),
    )


class TodoTools:
    """The todo tool, bound to one session's store."""

    def __init__(self, store: TodoStore) -> None:
        self.store = store

    def tools(self) -> list[RegisteredTool]:
        return [self._write()]

    def _write(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            raw = arguments.get("todos") or []
            items: list[TodoItem] = []
            for entry in raw:
                if isinstance(entry, Mapping):
                    items.append(
                        TodoItem(
                            content=str(entry.get("content", "")),
                            status=str(entry.get("status", STATUS_PENDING)),
                        )
                    )

            try:
                summary = self.store.write(items)
            except TodoError as exc:
                # A refusal is an ordinary result, so the step continues and the
                # model can fix the list it proposed.
                return ToolResult.fail(str(exc), code=exc.code)

            return ToolResult.ok(
                summary.render(),
                value={
                    "pending": summary.pending,
                    "in_progress": summary.in_progress,
                    "completed": summary.completed,
                },
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="todo_write",
                description=(
                    "Record or update your task list. Pass the complete list: it "
                    "replaces whatever was there. Use it to keep track of "
                    "multi-step work, and mark a task completed as soon as it is "
                    "done rather than batching the updates."
                ),
                parameters=TodoWriteParams.model_json_schema(),
            ),
            body=body,
            validator=TodoWriteParams,
        )

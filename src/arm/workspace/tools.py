"""Worktrees as tools: isolated checkouts the model can ask for.

Exposing worktrees to the model is what makes them useful. An agent that needs to
try an approach without disturbing the checkout it is working in can create a
worktree, work there, and remove it -- or leave it, if the work is worth keeping.

The tool surface is deliberately small and **never forces**. ``worktree_remove``
refuses a dirty worktree and reports why; a model that wants to discard work has
to ask for it explicitly. That asymmetry matters because a model cleaning up after
itself is a routine operation, and routine operations must not be the thing that
destroys work.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .worktree import WorktreeError, WorktreeManager


class WorktreeCreateParams(BaseModel):
    name: str = Field(description="Short name for the worktree, e.g. 'try-refactor'.")
    branch: str = Field(
        default="", description="Branch to create. Defaults to 'arm/<name>'."
    )
    from_revision: str = Field(
        default="HEAD", description="Revision to start from."
    )


class WorktreeRemoveParams(BaseModel):
    name: str = Field(description="Name of the worktree to remove.")
    force: bool = Field(
        default=False,
        description=(
            "Remove even with uncommitted changes. This discards work; only set "
            "it when you are certain the changes are not wanted."
        ),
    )
    delete_branch: bool = Field(
        default=False, description="Also delete the branch the worktree was on."
    )


class WorktreePathParams(BaseModel):
    name: str = Field(description="Name of the worktree.")


class WorktreeTools:
    """Worktree operations, wrapped as tools."""

    def __init__(self, manager: WorktreeManager) -> None:
        self.manager = manager

    def tools(self) -> list[RegisteredTool]:
        return [
            self._create(),
            self._list(),
            self._remove(),
        ]

    def _create(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            try:
                worktree = self.manager.create(
                    str(arguments["name"]),
                    branch=str(arguments.get("branch") or "") or None,
                    from_revision=str(arguments.get("from_revision") or "HEAD"),
                )
            except WorktreeError as exc:
                return ToolResult.fail(str(exc))
            return ToolResult.ok(
                f"Created worktree {worktree.name} at {worktree.path} "
                f"on branch {worktree.branch}.\n"
                "Use absolute paths under that directory to work in it.",
                value={"path": str(worktree.path), "branch": worktree.branch},
            )

        return RegisteredTool(
            definition=ToolDefinition(
                name="worktree_create",
                description=(
                    "Create an isolated checkout of this repository on its own "
                    "branch. Use it to try a change without disturbing the "
                    "current working tree."
                ),
                parameters=WorktreeCreateParams.model_json_schema(),
            ),
            body=body,
            validator=WorktreeCreateParams,
        )

    def _list(self) -> RegisteredTool:
        async def body(_arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            try:
                worktrees = self.manager.list()
            except WorktreeError as exc:
                return ToolResult.fail(str(exc))

            if not worktrees:
                return ToolResult.ok("No worktrees.")
            lines = []
            for worktree in worktrees:
                kind = "main" if worktree.is_main else "linked"
                clean, reason = self.manager.is_clean(worktree)
                status = "clean" if clean else reason
                lines.append(
                    f"{worktree.name}\t{worktree.path}\t"
                    f"{worktree.branch or '(detached)'}\t{kind}\t{status}"
                )
            return ToolResult.ok("\n".join(lines))

        return RegisteredTool(
            definition=ToolDefinition(
                name="worktree_list",
                description="List the repository's worktrees and whether each has uncommitted changes.",
                parameters={"type": "object", "properties": {}},
                concurrency_safe=lambda _args: True,
            ),
            body=body,
        )

    def _remove(self) -> RegisteredTool:
        async def body(arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
            try:
                worktree = self.manager.remove(
                    str(arguments["name"]),
                    force=bool(arguments.get("force", False)),
                    delete_branch=bool(arguments.get("delete_branch", False)),
                )
            except WorktreeError as exc:
                # A refusal is an ordinary result, and the message explains what
                # would have to change -- so the model can decide rather than
                # retry blindly.
                return ToolResult.fail(str(exc))
            return ToolResult.ok(f"Removed worktree {worktree.name}.")

        return RegisteredTool(
            definition=ToolDefinition(
                name="worktree_remove",
                description=(
                    "Remove a worktree. Refuses if it has uncommitted changes "
                    "unless force is set, because removing it discards them."
                ),
                parameters=WorktreeRemoveParams.model_json_schema(),
            ),
            body=body,
            validator=WorktreeRemoveParams,
        )

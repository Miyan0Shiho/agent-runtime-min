"""The ``skill`` tool: how the model loads a skill body.

One tool, one job. The model sees a catalogue of names and descriptions; this is
what turns a name into content.

The policy check happens **twice**: once against the candidate's summary, and
again after the body is loaded. The second check exists because a file can be
edited between discovery and load, and a skill that flipped itself to
non-model-invocable in that window must not be delivered anyway. Checking only
the summary would make the policy a race rather than a rule.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field

from ..tools.definition import RegisteredTool, ToolDefinition, ToolResult
from .catalog import render_skill_content
from .registry import SKILL_NAME, SkillRegistry


class SkillParams(BaseModel):
    """The tool's arguments."""

    name: str = Field(
        description="Name of the skill to load, exactly as it appears in the catalogue."
    )


class SkillTool:
    """Loads a skill by name and renders it for injection."""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry
        #: Names loaded in this session, so a repeat load can be reported rather
        #: than silently re-injected.
        self.loaded: list[str] = []

    def register(self) -> RegisteredTool:
        return RegisteredTool(
            definition=ToolDefinition(
                name="skill",
                description=(
                    "Load a skill's instructions. Skills are listed in the "
                    "available-skills catalogue with a name and a description; "
                    "call this with the name to read the full instructions before "
                    "following them."
                ),
                parameters=SkillParams.model_json_schema(),
                # A skill load is a pure read of a file, so it is safe to run
                # concurrently with anything else.
                concurrency_safe=lambda _args: True,
            ),
            body=self.run,
            validator=SkillParams,
        )

    async def run(self, arguments: Mapping[str, Any], _execution: Any) -> ToolResult:
        """Load a skill.

        Every failure path returns a result rather than raising, so a bad name or
        a missing skill does not interrupt the step -- consistent with the rest of
        the tool pipeline.
        """
        name = str(arguments.get("name", "")).strip()
        if not name:
            return ToolResult.fail("a skill name is required")

        # Validated before lookup so that a malformed name produces a clear
        # message instead of "not found", which would suggest a spelling problem
        # in the catalogue rather than in the call.
        if not SKILL_NAME.match(name):
            return ToolResult.fail(
                f"invalid skill name {name!r}: names are kebab-case "
                "(lowercase letters, digits and single hyphens)"
            )

        snapshot = self.registry.snapshot()
        summary = snapshot.by_name(name)
        if summary is None:
            available = ", ".join(s.name for s in snapshot.skills) or "(none)"
            return ToolResult.fail(
                f'unknown skill "{name}"; available skills are: {available}'
            )

        if not summary.model_invocable:
            return ToolResult.fail(
                f'skill "{name}" cannot be loaded by the model; it is available '
                "only when the user invokes it explicitly"
            )

        definition = self.registry.get(name)
        if definition is None:
            return ToolResult.fail(
                f'skill "{name}" is listed in the catalogue but its content could '
                "not be read"
            )

        # Re-checked after loading. A file edited between discovery and load could
        # have changed its own policy, and honoring the summary alone would make
        # the policy a race condition rather than a rule.
        if not definition.summary.model_invocable:
            return ToolResult.fail(
                f'skill "{name}" was changed to be non-model-invocable while '
                "loading; refusing to deliver it"
            )

        self.loaded.append(name)
        return ToolResult.ok(
            render_skill_content(definition),
            value={"name": name, "provider": definition.summary.source},
            additional_context=(f"Loaded skill {name}.",),
        )

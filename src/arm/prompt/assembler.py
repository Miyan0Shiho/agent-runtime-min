"""Prompt assembly: an ordered list of segments, rendered once per step.

The system prompt is assembled from segments rather than stored as one string, so
that a change in any contributor is visible as a change in *that* segment. That
matters because the prompt is recorded: when the prompt changes mid-session, the
log must say which segment moved, not just that the text differs.

Segments are ordered by an explicit priority, not by registration order.
Registration order would make the prompt depend on plugin load order, which is
not a design decision anyone made on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ..tools.definition import ToolDefinition
from ..tools.schema import to_wire


@dataclass(frozen=True, slots=True)
class Segment:
    """One contributor to the system prompt.

    ``priority`` sorts ascending: lower numbers appear earlier. Ties break by
    name so the output is stable across runs.
    """

    name: str
    text: str
    priority: int = 100


@dataclass
class PromptAssembler:
    """Collects prompt segments and renders the final system text."""

    def __init__(self) -> None:
        self._segments: dict[str, Segment] = {}

    def set(self, name: str, text: str, *, priority: int = 100) -> Callable[[], None]:
        """Add or replace a segment. Returns a disposer restoring the previous."""
        previous = self._segments.get(name)
        self._segments[name] = Segment(name=name, text=text, priority=priority)

        def _dispose() -> None:
            if previous is None:
                self._segments.pop(name, None)
            else:
                self._segments[name] = previous

        return _dispose

    def remove(self, name: str) -> None:
        self._segments.pop(name, None)

    def render(self) -> str:
        """Join the segments in priority order.

        Empty segments are skipped entirely rather than emitting a blank line: a
        segment that contributes nothing should not change the prompt's bytes,
        otherwise a plugin toggling itself on and off would look like a prompt
        change and force a new ``system/message`` event.
        """
        ordered = sorted(
            (s for s in self._segments.values() if s.text.strip()),
            key=lambda s: (s.priority, s.name),
        )
        return "\n\n".join(s.text.strip() for s in ordered)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._segments))


def render_tools(tools: Sequence[ToolDefinition]) -> list[dict]:
    """Render tool definitions for the wire.

    Only *visible* tools are rendered. A hidden tool remains callable but is not
    advertised, so it must not appear here.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": to_wire(tool.parameters),
        }
        for tool in tools
    ]


def developer_message_for(
    added: Iterable[str], removed: Iterable[str]
) -> str | None:
    """Describe a tool-set change for the model.

    Returns ``None`` when nothing changed, so the caller can skip appending an
    event. Emitting an empty developer message would add a node to the surface
    that carries no information -- and, because empty messages keep their
    position, it would shift every later index for no reason.
    """
    added = sorted(added)
    removed = sorted(removed)
    if not added and not removed:
        return None
    lines: list[str] = []
    if added:
        lines.append("Tools now available: " + ", ".join(added))
    if removed:
        lines.append("Tools no longer available: " + ", ".join(removed))
    return "\n".join(lines)


@dataclass
class ToolAnnouncement:
    """Tracks the tool set last announced to the model.

    The model is told about tool changes incrementally rather than being re-sent
    the whole catalogue, which keeps the transcript small. The cost is that the
    runtime must remember what it already said -- and it remembers by reading the
    log, so a restart re-derives the same answer.
    """

    announced: tuple[str, ...] = ()

    def diff(self, current: Sequence[str]) -> tuple[list[str], list[str]]:
        before = set(self.announced)
        after = set(current)
        return sorted(after - before), sorted(before - after)

    def update(self, current: Sequence[str]) -> None:
        self.announced = tuple(sorted(current))

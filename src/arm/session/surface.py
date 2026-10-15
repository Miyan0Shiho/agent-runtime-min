"""The surface: a fold over the log producing the model-visible node sequence.

The surface is *not* a second copy of the data. It is a list of ``seq`` numbers
plus two generation counters, recomputed from the log. Everything the model can
see is reachable from here, and everything here is reachable from the log.

Why two generations:

* ``content_generation`` -- bumped by anything that changes the *derived
  messages*: a ``replace``, or a projection. Appending does not bump it, which
  is what lets :meth:`Session.derive` extend its cache by slicing the tail
  instead of rebuilding the whole table on every append.
* ``replace_generation`` -- bumped only by ``replace``. Compaction uses it to
  answer "did this attempt actually shrink anything?", which is the difference
  between retrying productively and looping forever.

Using the wrong one in either place is a silent performance cliff or a silent
correctness bug, so they are kept apart and named for their consumers.

Validation is two-phase on purpose: :meth:`plan` computes what an event *would*
do without touching committed state, and :meth:`apply` commits it. A rejected
event therefore cannot leave a half-applied surface behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..kernel.errors import (
    ProjectionRemovedError,
    SeqDiscontinuityError,
    SourceReferenceError,
    SurfaceMetadataError,
    SurfaceRangeError,
    SystemHeadRewriteError,
    ToolResultRewriteError,
)
from ..kernel.freeze import thaw
from .envelope import Envelope
from .events import (
    ASSISTANT_MESSAGE,
    SYSTEM_MESSAGE,
    TOOL_RESULT,
    is_surface_eligible,
)
from .messages import Message

#: The exact key set of a well-formed replace operation.
_REPLACE_KEYS = frozenset({"op", "start_seq", "end_seq"})


def _json_equal(left: Any, right: Any) -> bool:
    """Deep equality over frozen structures, ignoring tuple/list differences."""
    return thaw(left) == thaw(right)


def is_replace_op(value: Any) -> bool:
    """Whether ``value`` has the exact shape of a replace operation.

    "Exactly three keys" is checked rather than "at least these keys": an extra
    key means the writer believed something extra would be honored, and ignoring
    it silently would make the log lie about its own semantics.
    """
    if not isinstance(value, Mapping):
        return False
    if set(value.keys()) != _REPLACE_KEYS:
        return False
    if value.get("op") != "replace":
        return False
    start, end = value.get("start_seq"), value.get("end_seq")
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
    )


def surface_op_of(event: Envelope) -> Any:
    """Read and validate the ``surface_op`` field of ``event``.

    Surface events must declare one; non-surface events must not carry one. The
    second half matters as much as the first: an event that carries a
    ``surface_op`` but is not surface-eligible is a writer that believes it can
    rewrite history, and honoring that belief would turn shadowing into deletion.
    """
    op = event.data.get("surface_op")
    eligible = is_surface_eligible(event.type)

    if not eligible:
        if op is not None:
            raise SurfaceMetadataError(
                f"{event.type} at seq {event.seq} is not surface-eligible and must "
                "not carry surface_op"
            )
        if "source_event_seqs" in event.data:
            raise SurfaceMetadataError(
                f"{event.type} at seq {event.seq} is not surface-eligible and must "
                "not carry source_event_seqs"
            )
        return None

    if op is None:
        raise SurfaceMetadataError(
            f"surface event {event.type} at seq {event.seq} must declare surface_op"
        )
    if op == "append":
        return "append"
    if is_replace_op(op):
        return op
    raise SurfaceMetadataError(
        f"invalid surface_op on {event.type} at seq {event.seq}: {op!r}"
    )


@dataclass(frozen=True, slots=True)
class Replacement:
    """A recorded rewrite: what it declared, and what it actually shadowed."""

    seq: int
    start_seq: int
    end_seq: int
    shadowed: tuple[int, ...]
    source_event_seqs: tuple[int, ...]


@dataclass(slots=True)
class Plan:
    """The computed effect of one event, before it is committed."""

    kind: str  # "append" | "replace" | "project" | "log-only"
    seq: int
    event_type: str
    start_index: int = 0
    end_index: int = 0
    shadowed: tuple[int, ...] = ()
    source_event_seqs: tuple[int, ...] = ()
    projected: tuple[Message, ...] = ()


@dataclass(slots=True)
class Surface:
    """The node sequence plus the generations that gate cache invalidation."""

    nodes: list[int] = field(default_factory=list)
    content_generation: int = 0
    replace_generation: int = 0
    replacements: list[Replacement] = field(default_factory=list)
    #: seq -> messages supplied by a registered projection.
    projected_messages: dict[int, tuple[Message, ...]] = field(default_factory=dict)

    def copy(self) -> "Surface":
        return Surface(
            nodes=list(self.nodes),
            content_generation=self.content_generation,
            replace_generation=self.replace_generation,
            replacements=list(self.replacements),
            projected_messages=dict(self.projected_messages),
        )

    def index_of(self, seq: int) -> int:
        """Position of ``seq`` in the *current* surface.

        Deliberately the surface position, not the log position: a replacement's
        endpoints are resolved against what the model currently sees.
        """
        try:
            return self.nodes.index(seq)
        except ValueError:
            raise SurfaceRangeError(
                f"seq {seq} is not present in the current surface"
            ) from None


class SurfaceFold:
    """Fold log events into a :class:`Surface`.

    The fold is pure: :meth:`plan` never mutates, and :meth:`apply` is the only
    writer. Keeping them apart is what makes "a rejected event changes nothing"
    true by construction rather than by careful ordering.
    """

    def __init__(
        self,
        projections: Mapping[str, Any] | None = None,
        *,
        events: Sequence[Envelope] | None = None,
    ) -> None:
        self._projections: dict[str, Any] = dict(projections or {})
        self._events: Sequence[Envelope] = events if events is not None else ()
        #: Projection names actually consulted while folding this log. If one is
        #: unregistered later, reads must fail rather than quietly return the
        #: value it used to produce.
        self._used_projections: set[str] = set()

    # -- projection registry --------------------------------------------

    def register_projection(self, event_type: str, definition: Any) -> None:
        self._projections[event_type] = definition

    def unregister_projection(self, event_type: str) -> None:
        self._projections.pop(event_type, None)

    def _assert_projections_intact(self) -> None:
        for name in self._used_projections:
            if name not in self._projections:
                raise ProjectionRemovedError(
                    f"message projection for {name!r} was removed or replaced after "
                    "it had already been used to derive this log"
                )

    # -- planning --------------------------------------------------------

    def plan(self, event: Envelope, surface: Surface, *, expected_seq: int) -> Plan:
        """Compute the effect of ``event`` without changing anything."""
        if event.seq != expected_seq:
            raise SeqDiscontinuityError(
                f"expected seq {expected_seq} but event carries seq {event.seq}"
            )

        op = surface_op_of(event)

        # A projection definition takes precedence: the event is surface-eligible
        # but its content comes from a registered projection, not its own body.
        if event.type in self._projections:
            definition = self._projections[event.type]
            self._used_projections.add(event.type)
            messages = definition.project(event, self._events)
            return Plan(
                kind="project",
                seq=event.seq,
                event_type=event.type,
                projected=tuple(messages),
            )

        if op is None:
            return Plan(kind="log-only", seq=event.seq, event_type=event.type)

        if op == "append":
            return Plan(kind="append", seq=event.seq, event_type=event.type)

        return self._plan_replace(event, surface, op)

    def _plan_replace(
        self, event: Envelope, surface: Surface, op: Mapping[str, Any]
    ) -> Plan:
        start_seq = int(op["start_seq"])
        end_seq = int(op["end_seq"])

        # A replacement may only point backwards. Referencing the future would
        # mean the log's meaning depends on events not yet written.
        if start_seq >= event.seq or end_seq >= event.seq:
            raise SurfaceRangeError(
                f"replace endpoints must be earlier than the event itself "
                f"(event seq {event.seq}, range [{start_seq}, {end_seq}])"
            )

        start_index = surface.index_of(start_seq)
        end_index = surface.index_of(end_seq)
        if start_index > end_index:
            raise SurfaceRangeError(
                f"replace range is reversed: start_seq {start_seq} sits after "
                f"end_seq {end_seq} on the current surface"
            )

        shadowed = tuple(surface.nodes[start_index : end_index + 1])

        # assistant/message embeds its own stream, so it can never shadow others.
        if event.type == ASSISTANT_MESSAGE and "source_event_seqs" in event.data:
            raise SourceReferenceError(
                f"assistant/message at seq {event.seq} must not carry "
                "source_event_seqs"
            )

        raw_refs = event.data.get("source_event_seqs", ())
        if isinstance(raw_refs, (str, bytes)) or not isinstance(raw_refs, Sequence):
            raise SourceReferenceError(
                "source_event_seqs must be a list of sequence numbers"
            )
        refs = tuple(int(r) for r in raw_refs)

        if not refs:
            raise SourceReferenceError(
                f"replace at seq {event.seq} must declare source_event_seqs"
            )
        if len(set(refs)) != len(refs):
            raise SourceReferenceError(
                f"replace at seq {event.seq} repeats entries in source_event_seqs"
            )
        if any(r >= event.seq for r in refs):
            raise SourceReferenceError(
                f"replace at seq {event.seq} references an event at or after itself"
            )
        missing = [s for s in shadowed if s not in refs]
        if missing:
            raise SourceReferenceError(
                f"replace at seq {event.seq} fails to cover shadowed nodes {missing}; "
                "every shadowed node must be declared"
            )
        extra = [r for r in refs if r not in shadowed]
        if extra:
            raise SourceReferenceError(
                f"replace at seq {event.seq} declares {extra} which are not being "
                "shadowed by this range"
            )

        self._assert_tool_result_rewrite(event, surface, shadowed)
        self._assert_system_head_rewrite(event, surface, start_index, shadowed)

        return Plan(
            kind="replace",
            seq=event.seq,
            event_type=event.type,
            start_index=start_index,
            end_index=end_index,
            shadowed=shadowed,
            source_event_seqs=refs,
        )

    def _assert_tool_result_rewrite(
        self, event: Envelope, surface: Surface, shadowed: tuple[int, ...]
    ) -> None:
        """A ``tool/result`` may only be rewritten, never swapped for another.

        Pruning a large result is legitimate. Replacing a result with a
        *different tool's* result is not, and the two look identical unless the
        non-content fields are compared.

        The model-visible text lives at ``message.content``, so that is the one
        path allowed to differ. Everything else -- including the rest of the
        ``message`` object, such as ``tool_call_id`` -- must be deeply equal.
        """
        if event.type != TOOL_RESULT:
            return
        if len(shadowed) != 1:
            raise ToolResultRewriteError(
                f"tool/result replacement at seq {event.seq} must shadow exactly one "
                f"node, got {len(shadowed)}"
            )
        original = self._event_at(shadowed[0])
        if original.type != TOOL_RESULT:
            raise ToolResultRewriteError(
                f"tool/result replacement at seq {event.seq} targets "
                f"{original.type} at seq {original.seq}"
            )

        def _outside_content(value: Any) -> Any:
            """The payload with the model-visible content removed."""
            if not isinstance(value, Mapping):
                return thaw(value)
            body = {k: thaw(v) for k, v in value.items() if k != "content"}
            return body

        for key in set(original.data) | set(event.data):
            if key in {"surface_op", "source_event_seqs"}:
                continue
            before = original.data.get(key)
            after = event.data.get(key)
            if key == "message":
                before, after = _outside_content(before), _outside_content(after)
            if before != after:
                raise ToolResultRewriteError(
                    f"tool/result replacement at seq {event.seq} changed {key!r}; "
                    "only the message content may differ"
                )

    def _assert_system_head_rewrite(
        self,
        event: Envelope,
        surface: Surface,
        start_index: int,
        shadowed: tuple[int, ...],
    ) -> None:
        """The system prompt at node 0 is protected.

        It may only be replaced by another ``system/message`` covering exactly
        that one node. Any wider range would let a compaction sweep the system
        prompt away as a side effect of shrinking history.
        """
        if not surface.nodes or surface.nodes[0] != shadowed[0]:
            return
        head = self._event_at(surface.nodes[0])
        if head.type != SYSTEM_MESSAGE:
            return
        if event.type != SYSTEM_MESSAGE:
            raise SystemHeadRewriteError(
                f"the system prompt at seq {head.seq} may only be rewritten by a "
                f"system/message, not {event.type}"
            )
        if len(shadowed) != 1 or start_index != 0:
            raise SystemHeadRewriteError(
                "the system prompt may only be rewritten by a replacement that "
                "covers exactly that node"
            )

    def _event_at(self, seq: int) -> Envelope:
        if seq >= len(self._events):
            raise SurfaceRangeError(
                f"seq {seq} is beyond the events available to this fold"
            )
        return self._events[seq]

    # -- applying --------------------------------------------------------

    def apply(self, surface: Surface, plan: Plan) -> None:
        """Commit ``plan`` to ``surface``. The only mutation point."""
        if plan.kind == "log-only":
            return
        if plan.kind == "append":
            # Appending must not bump content_generation: doing so would force a
            # full rebuild of the derived table on every single append.
            surface.nodes.append(plan.seq)
            return
        if plan.kind == "project":
            surface.projected_messages[plan.seq] = plan.projected
            surface.nodes.append(plan.seq)
            surface.content_generation += 1
            return

        # replace
        surface.nodes[plan.start_index : plan.end_index + 1] = [plan.seq]
        surface.content_generation += 1
        surface.replace_generation += 1
        surface.replacements.append(
            Replacement(
                seq=plan.seq,
                start_seq=int(plan.source_event_seqs[0]) if plan.source_event_seqs else 0,
                end_seq=int(plan.source_event_seqs[-1]) if plan.source_event_seqs else 0,
                shadowed=plan.shadowed,
                source_event_seqs=plan.source_event_seqs,
            )
        )


def fold_surface(
    events: Sequence[Envelope], projections: Mapping[str, Any] | None = None
) -> Surface:
    """Fold a whole log into a surface. Used by cold reads and tests."""
    fold = SurfaceFold(projections, events=events)
    surface = Surface()
    for event in events:
        plan = fold.plan(event, surface, expected_seq=event.seq)
        fold.apply(surface, plan)
    return surface

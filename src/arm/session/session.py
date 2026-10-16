"""The session: one append-only log, and everything derived from it.

This is the only object that owns durable state. Phase, context, token counts
and tool progress are all *derived* -- if any of them were stored independently,
two sources of truth would exist and they would drift.

The commit path has exactly one point, and it is ``log.append``:

1. validate the payload is JSON-lossless and the event type is known;
2. plan the surface effect (which may reject the event);
3. append to the log;
4. apply the surface plan;
5. notify listeners, whose failures are contained.

Steps 1-2 happen before anything changes, so a rejected event leaves no trace.
Listeners run *after* the commit, so an observer that raises cannot unwind a
fact that has already happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..kernel.errors import (
    FormatVersionError,
    ProjectionRequiredError,
    ReentrantAppendError,
    SessionContractError,
    UnknownEventTypeError,
)
from ..kernel.freeze import assert_json_lossless
from .envelope import Envelope, utc_now_iso
from .events import (
    FORMAT_VERSION,
    SESSION_END_SEED,
    SESSION_START,
    TOOL_RESULT,
    is_known,
    needs_projection,
)
from .messages import Message
from .surface import Surface, SurfaceFold

log = logging.getLogger("arm.session")

#: Called after each committed event, with ``(session, event)``.
Listener = Callable[["Session", Envelope], None]


def _derive_event_message(
    event: Envelope, projected: Mapping[int, tuple[Message, ...]]
) -> Message | None:
    """Map one surface event to the message the model sees, or ``None``.

    ``None`` is a real answer, not a failure: an empty ``system``/``developer``
    /``assistant`` message keeps its surface position but must not be sent. That
    separation is what keeps message ordering intact while allowing placeholder
    nodes.
    """
    supplied = projected.get(event.seq)
    if supplied is not None:
        return supplied[0] if supplied else None

    payload = event.data.get("message")
    if payload is None:
        return None
    message = Message.from_record(payload)

    # An empty message holds a surface position but is never sent. Dropping the
    # *position* instead would shift tool pairing.
    #
    # "Empty" means no content AND no tool calls. An assistant message whose only
    # payload is a tool call has no text but is emphatically not empty: it is the
    # message the following tool result answers, and omitting it would leave the
    # transcript with a tool result whose call is missing.
    if (
        message.role in ("system", "developer", "assistant")
        and not message.content
        and not message.tool_calls
    ):
        return None
    return message


@dataclass
class SessionStats:
    """Cheap counters for diagnostics. Never authoritative."""

    appends: int = 0
    replaces: int = 0
    derivations: int = 0
    full_rebuilds: int = 0


class Session:
    """An append-only event log with a derived model-visible surface.

    Not thread-safe by design: a session is driven by one loop, and pretending
    otherwise would hide the ordering assumptions the log depends on.
    """

    def __init__(
        self,
        session_id: str,
        *,
        seed: Sequence[Envelope] = (),
        inherited_event_count: int = 0,
        projections: Mapping[str, Any] | None = None,
        header: Mapping[str, Any] | None = None,
    ) -> None:
        if inherited_event_count > len(seed):
            raise SessionContractError(
                "inherited_event_count cannot exceed the seed length"
            )
        if inherited_event_count and not seed:
            raise SessionContractError("a seeded session must provide seed events")

        # Seed continuity is checked here rather than trusted: a gap at the head
        # would be indistinguishable from a truncation later.
        for index, event in enumerate(seed):
            if event.seq != index:
                raise SessionContractError(
                    f"seed must be contiguous from 0; found seq {event.seq} at "
                    f"index {index}"
                )

        self.session_id = session_id
        self.header: dict[str, Any] = {
            "session_id": session_id,
            "format_version": FORMAT_VERSION,
            "created_at": utc_now_iso(),
            **(header or {}),
        }
        self.inherited_event_count = inherited_event_count
        self.log: list[Envelope] = list(seed)
        self.stats = SessionStats()

        self._fold = SurfaceFold(projections, events=self.log)
        self._surface = Surface()
        self._rebuild_surface()

        # Derived-message cache. Invalidated by ``content_generation`` only:
        # appends extend the cache, replaces and projections rebuild it.
        self._derived: list[Message] = []
        self._derived_nodes = 0
        self._derived_generation = self._surface.content_generation

        self._listeners: list[Listener] = []
        self._appending = False

    # -- construction helpers -------------------------------------------

    @classmethod
    def create(
        cls, session_id: str, *, header: Mapping[str, Any] | None = None
    ) -> "Session":
        """Start a fresh session, writing the two session-scope events."""
        session = cls(session_id, header=header)
        # The header goes into the event, not just onto the object.
        #
        # Found by reading back a subagent's log to check the lineage and finding
        # it empty: `header` was stored on the Session and `session/start` was
        # written with only the id, so a value set here was never recorded. The
        # session knew its parent; the log -- which is supposed to be the record --
        # did not.
        #
        # It matters for children specifically, because the parent link is what
        # makes a delegation tree walkable from the log alone. Reconstructing it
        # from anything else would mean the log was not sufficient.
        start: dict[str, Any] = {"session_id": session_id}
        if header:
            start["header"] = dict(header)
        session.append(SESSION_START, start)
        session.append(SESSION_END_SEED, {})
        return session

    @classmethod
    def from_events(
        cls,
        session_id: str,
        events: Sequence[Envelope],
        *,
        header: Mapping[str, Any] | None = None,
        inherited_event_count: int = 0,
        projections: Mapping[str, Any] | None = None,
    ) -> "Session":
        """Rehydrate from a stored log."""
        return cls(
            session_id,
            seed=events,
            inherited_event_count=inherited_event_count,
            projections=projections,
            header=header,
        )

    # -- log access ------------------------------------------------------

    @property
    def seq(self) -> int:
        """The seq the next appended event will receive."""
        return len(self.log)

    def event_at(self, seq: int) -> Envelope:
        return self.log[seq]

    def events(self) -> tuple[Envelope, ...]:
        return tuple(self.log)

    def __len__(self) -> int:
        return len(self.log)

    def __iter__(self) -> Iterator[Envelope]:
        return iter(self.log)

    # -- surface ---------------------------------------------------------

    @property
    def surface(self) -> Surface:
        return self._surface

    def _rebuild_surface(self) -> None:
        """Fold the whole log. Used at construction and by cold reads."""
        self._fold = SurfaceFold(self._fold._projections, events=self.log)
        surface = Surface()
        for event in self.log:
            plan = self._fold.plan(event, surface, expected_seq=event.seq)
            self._fold.apply(surface, plan)
        self._surface = surface
        self._derived = []
        self._derived_nodes = 0
        self._derived_generation = surface.content_generation

    def surface_nodes(self) -> tuple[int, ...]:
        return tuple(self._surface.nodes)

    # -- projection registry --------------------------------------------

    def register_projection(self, event_type: str, definition: Any) -> None:
        """Register a message projection for ``event_type``.

        Registering changes how the log *reads*, so the derived cache is
        invalidated. It is not a rewrite of history -- the events are untouched.
        """
        self._fold.register_projection(event_type, definition)
        self._invalidate_derived()

    def unregister_projection(self, event_type: str) -> None:
        self._fold.unregister_projection(event_type)
        self._invalidate_derived()

    def _invalidate_derived(self) -> None:
        self._derived = []
        self._derived_nodes = 0
        self._derived_generation = -1

    # -- the commit path -------------------------------------------------

    def append(self, event_type: str, data: Mapping[str, Any] | None = None, *,
               ts: str | None = None, ignorable: bool = False) -> Envelope:
        """Validate, commit, then notify. The only writer.

        Re-entrancy is refused outright: a listener that appends would recurse
        inside the commit path, and the resulting log order would depend on
        listener registration order rather than on program order.
        """
        if self._appending:
            raise ReentrantAppendError(
                "cannot append from inside a commit; a listener is trying to write "
                "while the log is being committed"
            )

        body = dict(data or {})
        event = Envelope(
            seq=self.seq,
            type=event_type,
            data=body,
            ts=ts or utc_now_iso(),
            format_version=FORMAT_VERSION,
            ignorable=ignorable,
        )

        # -- validate (nothing has changed yet) --------------------------
        self._validate_event(event)

        # -- plan the surface effect (may still reject) ------------------
        plan = self._fold.plan(event, self._surface, expected_seq=event.seq)

        # -- commit ------------------------------------------------------
        # The re-entrancy guard covers the notification phase too. A listener
        # runs *inside* the commit, so a listener that appends is refused rather
        # than allowed to recurse (M1.3/M1.4).
        self._appending = True
        try:
            self.log.append(event)
            self._fold.apply(self._surface, plan)
            self.stats.appends += 1
            if plan.kind == "replace":
                self.stats.replaces += 1
            self._notify(event)
        finally:
            self._appending = False

        return event

    def _validate_event(self, event: Envelope) -> None:
        if event.format_version > FORMAT_VERSION:
            raise FormatVersionError(
                f"event declares format version {event.format_version} but this "
                f"build writes {FORMAT_VERSION}; upgrade the runtime"
            )

        if not is_known(event.type):
            if event.ignorable:
                return
            if event.type.startswith(
                ("subagent/", "team/", "feedback/", "hook/", "telemetry/",
                 "image/", "session-log-")
            ):
                raise UnknownEventTypeError(
                    f"event type {event.type!r} is reserved for a later format "
                    "generation; this build cannot represent it"
                )
            raise UnknownEventTypeError(
                f"unknown event type {event.type!r} at seq {event.seq}; mark it "
                "ignorable:true if it is safe to skip"
            )

        # A projection-eligible surface event with no definition would silently
        # lose content, so it is refused rather than skipped.
        if needs_projection(event.type) and event.type not in self._fold._projections:
            if not _is_self_describing(event):
                raise ProjectionRequiredError(
                    f"{event.type} at seq {event.seq} requires a message projection"
                )

        assert_json_lossless(dict(event.data), what=f"{event.type} data")

    # -- listeners -------------------------------------------------------

    def listen(self, listener: Listener) -> Callable[[], None]:
        """Subscribe to committed events. Returns an unsubscribe callable."""
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _unsubscribe

    def _notify(self, event: Envelope) -> None:
        for listener in tuple(self._listeners):
            try:
                listener(self, event)
            except Exception:  # noqa: BLE001 - containment is deliberate
                log.exception(
                    "session listener failed for %s seq %d; the event is already "
                    "committed",
                    event.type,
                    event.seq,
                )

    # -- derivation ------------------------------------------------------

    def derive(self) -> list[Message]:
        """The model-visible message list, derived from the log.

        Correctness criterion: given the log, this reproduces byte-for-byte what
        was sent. The cache is keyed on ``content_generation`` so appends extend
        it incrementally while replaces and projections rebuild it.
        """
        self.stats.derivations += 1
        # Reading through a projection that has since been unregistered would
        # silently hand back a value its plugin no longer owns (M1.11).
        self._fold._assert_projections_intact()

        surface = self._surface
        nodes = surface.nodes
        generation = surface.content_generation

        if generation != self._derived_generation:
            self._derived = []
            self._derived_nodes = 0
            self._derived_generation = generation
            self.stats.full_rebuilds += 1

        for seq in nodes[self._derived_nodes :]:
            message = _derive_event_message(self.log[seq], surface.projected_messages)
            if message is not None:
                self._derived.append(message)
        self._derived_nodes = len(nodes)

        # A fresh list over shared frozen messages: callers may reorder their
        # copy but cannot rewrite history through it.
        return list(self._derived)

    # -- convenience constructors for surface events ---------------------

    def append_message(self, role: str, content: Sequence[Mapping[str, Any]], **extra: Any) -> Envelope:
        """Append a message-bearing surface event.

        Every message event stores a complete ``message`` object rather than a
        bare content array. One shape for all five roles means the deriver has
        one path, and a projection plugin can replace that path wholesale
        without changing what the log stores.
        """
        if role not in {"system", "user", "developer", "assistant"}:
            raise ValueError(f"unsupported message role {role!r}")
        message: dict[str, Any] = {
            "role": role,
            "content": [dict(block) for block in content],
        }
        for key in ("tool_calls", "tool_call_id", "name", "interrupted"):
            if key in extra:
                message[key] = extra.pop(key)
        data: dict[str, Any] = {"message": message, "surface_op": "append", **extra}
        return self.append(f"{role}/message", data)

    def append_tool_result(
        self,
        *,
        call_seq: int,
        call_id: str,
        message: Mapping[str, Any],
        is_error: bool = False,
        info: Mapping[str, Any] | None = None,
        replace: Mapping[str, Any] | None = None,
        source_event_seqs: Sequence[int] | None = None,
    ) -> Envelope:
        """Append a ``tool/result`` paired to its call.

        ``call_seq`` is recorded so pairing is checkable by inspection rather
        than by scanning for a matching id. When ``replace`` is given, the
        caller must also declare ``source_event_seqs`` -- the shadowed nodes are
        what licenses the rewrite.
        """
        data: dict[str, Any] = {
            "call_seq": call_seq,
            "call_id": call_id,
            "message": dict(message),
            "is_error": bool(is_error),
            "surface_op": dict(replace) if replace else "append",
        }
        if source_event_seqs is not None:
            data["source_event_seqs"] = list(source_event_seqs)
        if info is not None:
            data["info"] = dict(info)
        return self.append(TOOL_RESULT, data)

    def append_tool_call(
        self,
        *,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        step: int,
        index: int,
    ) -> Envelope:
        """Append a ``tool/call``. Not a surface event: only results occupy
        surface positions, which keeps pairing checkable."""
        return self.append(
            "tool/call",
            {
                "call_id": call_id,
                "name": name,
                "arguments": dict(arguments),
                "step": step,
                "index": index,
            },
        )


def _is_self_describing(event: Envelope) -> bool:
    """Whether a projection-eligible event carries its own message payload.

    Every message-bearing event stores a complete ``message`` object, so a
    registered projection is an *override* rather than a requirement. Events
    without one genuinely need a plugin, and are refused instead of dropped.
    """
    return "message" in event.data

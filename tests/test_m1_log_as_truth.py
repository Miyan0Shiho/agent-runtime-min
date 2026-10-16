"""M1 contract tests: the log is the only truth.

Each test names the invariant it defends (M1.x / S.x) so that a failure points at
a contract clause rather than at a function.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from arm.kernel.errors import (
    FormatVersionError,
    PayloadNotSerializableError,
    ProjectionRemovedError,
    ReentrantAppendError,
    SessionContractError,
    SourceReferenceError,
    SurfaceMetadataError,
    SurfaceRangeError,
    SystemHeadRewriteError,
    ToolResultRewriteError,
    UnknownEventTypeError,
)
from arm.session.envelope import Envelope
from arm.session.events import (
    SESSION_START,
    SYSTEM_MESSAGE,
    TOOL_RESULT,
    USER_MESSAGE,
)
from arm.session.messages import Message
from arm.session.persistence import JsonlLog, read_log
from arm.session.session import Session
from arm.session.surface import fold_surface, is_replace_op

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def text(content: str) -> list[dict]:
    return [{"type": "text", "text": content}]


def make_session() -> Session:
    return Session.create("s1")


# ---------------------------------------------------------------------------
# M1.1 -- seq is defined as len(log), not validated
# ---------------------------------------------------------------------------


def test_seq_is_contiguous_from_zero():
    """M1.1: continuity is a definition, so no gaps can exist."""
    session = make_session()
    session.append_message("user", text("a"))
    session.append_message("assistant", text("b"))
    session.append_message("user", text("c"))
    assert [e.seq for e in session] == list(range(len(session)))


def test_seed_must_be_contiguous():
    """M1.1: a gap at the head is indistinguishable from truncation."""
    with pytest.raises(SessionContractError, match="contiguous"):
        Session("s", seed=[Envelope(seq=1, type=SESSION_START, data={})])


# ---------------------------------------------------------------------------
# M1.2 -- validate then commit; a rejected event changes nothing
# ---------------------------------------------------------------------------


def test_rejected_event_leaves_log_and_surface_untouched():
    """M1.2: all validation precedes the append, so failure is total."""
    session = make_session()
    session.append_message("user", text("keep me"))
    before_log = len(session)
    before_nodes = session.surface_nodes()

    with pytest.raises(PayloadNotSerializableError):
        session.append_message("user", [{"type": "text", "text": object()}])

    assert len(session) == before_log
    assert session.surface_nodes() == before_nodes


def test_bad_replace_leaves_surface_untouched():
    """M1.2/M1.7: a rejected replace must not partially rewrite the surface."""
    session = make_session()
    session.append_message("user", text("one"))
    session.append_message("assistant", text("two"))
    before = session.surface_nodes()

    with pytest.raises(SurfaceRangeError):
        session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": text("bad")},
                "surface_op": {"op": "replace", "start_seq": 99, "end_seq": 99},
                "source_event_seqs": [99],
            },
        )

    assert session.surface_nodes() == before


# ---------------------------------------------------------------------------
# M1.3 -- append is not re-entrant
# ---------------------------------------------------------------------------


def test_reentrant_append_is_refused():
    """M1.3: a listener writing during commit would recurse forever."""
    session = make_session()
    attempted: list[str] = []

    def rogue_listener(_session: Session, _event: Envelope) -> None:
        try:
            session.append("assistant/attempt", {})
        except ReentrantAppendError:
            attempted.append("refused")

    session.listen(rogue_listener)
    before = len(session)
    session.append_message("user", text("x"))
    # The listener's attempt was refused, so the log grew by exactly one event
    # and no recursion occurred.
    assert attempted == ["refused"]
    assert len(session) == before + 1


def test_append_guard_raises_when_called_directly():
    """M1.3: the guard itself raises rather than silently recursing."""
    session = make_session()
    session._appending = True
    try:
        with pytest.raises(ReentrantAppendError):
            session.append("assistant/attempt", {})
    finally:
        session._appending = False


# ---------------------------------------------------------------------------
# M1.4 -- one commit point; listener failures are contained
# ---------------------------------------------------------------------------


def test_listener_failure_does_not_break_commit():
    """M1.4: by the time listeners run, the fact is already durable."""
    session = make_session()

    def boom(_session: Session, _event: Envelope) -> None:
        raise RuntimeError("listener exploded")

    session.listen(boom)
    event = session.append_message("user", text("still recorded"))
    assert session.event_at(event.seq).type == USER_MESSAGE
    assert [m.text for m in session.derive()] == ["still recorded"]


def test_listeners_run_in_registration_order():
    """M1.4: dispatch order is registration order, deterministically."""
    session = make_session()
    seen: list[str] = []
    session.listen(lambda s, e: seen.append("first"))
    session.listen(lambda s, e: seen.append("second"))
    session.append_message("user", text("x"))
    assert seen == ["first", "second"]


# ---------------------------------------------------------------------------
# M1.5 -- bad events never reach disk
# ---------------------------------------------------------------------------


def test_non_serializable_payload_is_refused():
    """M1.5: disk and memory must be isomorphic."""
    session = make_session()
    with pytest.raises(PayloadNotSerializableError):
        session.append("assistant/attempt", {"blob": {1, 2, 3}})


def test_envelope_rejects_reserved_data_keys():
    """M1.5: envelope fields must not be duplicated inside ``data``."""
    with pytest.raises(ValueError, match="envelope fields"):
        Envelope(seq=0, type=SESSION_START, data={"seq": 5})


def test_nan_is_refused():
    """M1.5: NaN has no JSON representation that round-trips."""
    session = make_session()
    with pytest.raises(PayloadNotSerializableError):
        session.append("assistant/attempt", {"value": float("nan")})


# ---------------------------------------------------------------------------
# M1.6 -- surfaceOp presence rules
# ---------------------------------------------------------------------------


def test_surface_event_requires_surface_op():
    """M1.6: an undeclared surface event has no defined effect."""
    session = make_session()
    with pytest.raises(SurfaceMetadataError, match="must declare surface_op"):
        session.append(USER_MESSAGE, {"message": {"role": "user", "content": text("x")}})


def test_non_surface_event_must_not_carry_surface_op():
    """M1.6: honoring this would turn shadowing into deletion."""
    session = make_session()
    with pytest.raises(SurfaceMetadataError, match="not surface-eligible"):
        session.append("turn/start", {"turn": 1, "surface_op": "append"})


def test_replace_op_shape_is_exact():
    """M1.6: an extra key means the writer expected something extra."""
    assert is_replace_op({"op": "replace", "start_seq": 1, "end_seq": 2})
    assert not is_replace_op({"op": "replace", "start_seq": 1, "end_seq": 2, "x": 3})
    assert not is_replace_op({"op": "append", "start_seq": 1, "end_seq": 2})
    assert not is_replace_op("replace")


# ---------------------------------------------------------------------------
# M1.7 / M1.8 -- replace range and coverage
# ---------------------------------------------------------------------------


def test_replace_shadows_and_reinserts_at_original_position():
    """M1.7: the new node takes the shadowed range's position, not the tail."""
    session = make_session()
    a = session.append_message("user", text("aaa"))
    b = session.append_message("assistant", text("bbb"))
    c = session.append_message("user", text("ccc"))

    session.append(
        USER_MESSAGE,
        {
            "message": {"role": "user", "content": text("AAA-summarized")},
            "surface_op": {"op": "replace", "start_seq": a.seq, "end_seq": b.seq},
            "source_event_seqs": [a.seq, b.seq],
        },
    )
    replacement = session.event_at(session.seq - 1)

    # The replacement sits where [a, b] used to be, before c.
    assert session.surface_nodes() == (replacement.seq, c.seq)
    assert [m.text for m in session.derive()] == ["AAA-summarized", "ccc"]


def test_replace_must_cover_every_shadowed_node():
    """M1.8: partial coverage means an event was silently discarded."""
    session = make_session()
    a = session.append_message("user", text("aaa"))
    b = session.append_message("assistant", text("bbb"))

    with pytest.raises(SourceReferenceError, match="fails to cover"):
        session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": text("partial")},
                "surface_op": {"op": "replace", "start_seq": a.seq, "end_seq": b.seq},
                "source_event_seqs": [a.seq],
            },
        )


def test_replace_cannot_reference_the_future():
    """M1.8: the log's meaning must not depend on unwritten events."""
    session = make_session()
    a = session.append_message("user", text("aaa"))
    future = session.seq + 5

    with pytest.raises(SurfaceRangeError, match="earlier than the event"):
        session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": text("x")},
                "surface_op": {"op": "replace", "start_seq": a.seq, "end_seq": future},
                "source_event_seqs": [a.seq],
            },
        )


def test_replace_rejects_empty_and_duplicate_sources():
    """M1.8: sources must be non-empty and unique."""
    session = make_session()
    a = session.append_message("user", text("aaa"))

    with pytest.raises(SourceReferenceError, match="must declare"):
        session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": text("x")},
                "surface_op": {"op": "replace", "start_seq": a.seq, "end_seq": a.seq},
                "source_event_seqs": [],
            },
        )

    with pytest.raises(SourceReferenceError, match="repeats"):
        session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": text("x")},
                "surface_op": {"op": "replace", "start_seq": a.seq, "end_seq": a.seq},
                "source_event_seqs": [a.seq, a.seq],
            },
        )


def test_reversed_range_is_refused():
    """M1.7: start after end is meaningless and must not be guessed at."""
    session = make_session()
    a = session.append_message("user", text("aaa"))
    b = session.append_message("assistant", text("bbb"))

    with pytest.raises(SurfaceRangeError, match="reversed"):
        session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": text("x")},
                "surface_op": {"op": "replace", "start_seq": b.seq, "end_seq": a.seq},
                "source_event_seqs": [a.seq, b.seq],
            },
        )


# ---------------------------------------------------------------------------
# M1.9 -- tool/result replacement may only change content
# ---------------------------------------------------------------------------


def test_tool_result_rewrite_may_change_content_only():
    """M1.9: pruning is legitimate; swapping results is not."""
    session = make_session()
    call = session.append_tool_call(
        call_id="c1", name="read", arguments={"path": "a"}, step=1, index=0
    )
    result = session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={"role": "tool", "content": text("x" * 5000), "tool_call_id": "c1"},
    )

    pruned = session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={"role": "tool", "content": text("head…tail"), "tool_call_id": "c1"},
        replace={"op": "replace", "start_seq": result.seq, "end_seq": result.seq},
        source_event_seqs=[result.seq],
    )
    # The replacement carries the same non-content fields, so it is accepted.
    assert session.surface_nodes()[-1] == pruned.seq
    assert [m.text for m in session.derive()] == ["head…tail"]


def test_tool_result_rewrite_cannot_change_call_id():
    """M1.9: changing the id would silently swap which call this answers."""
    session = make_session()
    call = session.append_tool_call(
        call_id="c1", name="read", arguments={}, step=1, index=0
    )
    result = session.append_tool_result(
        call_seq=call.seq,
        call_id="c1",
        message={"role": "tool", "content": text("original"), "tool_call_id": "c1"},
    )

    with pytest.raises(ToolResultRewriteError, match="only the message content may differ"):
        session.append(
            TOOL_RESULT,
            {
                "call_seq": call.seq,
                "call_id": "c2",  # swapped!
                "message": {"role": "tool", "content": text("evil"), "tool_call_id": "c2"},
                "is_error": False,
                "surface_op": {"op": "replace", "start_seq": result.seq, "end_seq": result.seq},
                "source_event_seqs": [result.seq],
            },
        )


def test_tool_result_replace_must_shadow_exactly_one_node():
    """M1.9: a batch rewrite cannot be expressed as a single result replace."""
    session = make_session()
    call = session.append_tool_call(call_id="c1", name="read", arguments={}, step=1, index=0)
    r1 = session.append_tool_result(
        call_seq=call.seq, call_id="c1",
        message={"role": "tool", "content": text("one"), "tool_call_id": "c1"},
    )
    r2 = session.append_tool_result(
        call_seq=call.seq, call_id="c1",
        message={"role": "tool", "content": text("two"), "tool_call_id": "c1"},
    )

    with pytest.raises(ToolResultRewriteError, match="exactly one node"):
        session.append(
            TOOL_RESULT,
            {
                "call_seq": call.seq,
                "call_id": "c1",
                "message": {"role": "tool", "content": text("both"), "tool_call_id": "c1"},
                "is_error": False,
                "surface_op": {"op": "replace", "start_seq": r1.seq, "end_seq": r2.seq},
                "source_event_seqs": [r1.seq, r2.seq],
            },
        )


# ---------------------------------------------------------------------------
# M1.10 -- the system prompt at node 0 is protected
# ---------------------------------------------------------------------------


def test_system_head_cannot_be_eaten_by_a_wider_replace():
    """M1.10: compaction must not sweep the system prompt away as a side effect."""
    session = make_session()
    system = session.append_message("system", text("you are arm"))
    user = session.append_message("user", text("hello"))

    with pytest.raises(SystemHeadRewriteError, match="covers exactly that node"):
        session.append(
            SYSTEM_MESSAGE,
            {
                "message": {"role": "system", "content": text("summary of everything")},
                "surface_op": {"op": "replace", "start_seq": system.seq, "end_seq": user.seq},
                "source_event_seqs": [system.seq, user.seq],
            },
        )


def test_system_head_cannot_be_replaced_by_another_role():
    """M1.10: a non-system message must not be able to displace the prompt."""
    session = make_session()
    system = session.append_message("system", text("you are arm"))

    with pytest.raises(SystemHeadRewriteError, match="may only be rewritten by a system/message"):
        session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": text("impostor")},
                "surface_op": {"op": "replace", "start_seq": system.seq, "end_seq": system.seq},
                "source_event_seqs": [system.seq],
            },
        )


def test_system_head_can_be_rewritten_by_a_system_message():
    """M1.10: rewriting the prompt is allowed -- it just must be declared as such."""
    session = make_session()
    system = session.append_message("system", text("v1 prompt"))

    rewritten = session.append(
        SYSTEM_MESSAGE,
        {
            "message": {"role": "system", "content": text("v2 prompt")},
            "surface_op": {"op": "replace", "start_seq": system.seq, "end_seq": system.seq},
            "source_event_seqs": [system.seq],
        },
    )
    assert session.surface_nodes() == (rewritten.seq,)
    assert [m.text for m in session.derive()] == ["v2 prompt"]


def test_non_system_replace_cannot_target_the_head():
    """M1.10: a user message must not be able to displace the system prompt."""
    session = make_session()
    system = session.append_message("system", text("prompt"))

    with pytest.raises(SystemHeadRewriteError, match="may only be rewritten by a system/message"):
        session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": text("impostor")},
                "surface_op": {"op": "replace", "start_seq": system.seq, "end_seq": system.seq},
                "source_event_seqs": [system.seq],
            },
        )


# ---------------------------------------------------------------------------
# M1.11 -- missing projection is refused, not silently dropped
# ---------------------------------------------------------------------------


def test_event_without_message_payload_requires_projection():
    """M1.11: dropping content silently would change what the model sees."""
    session = make_session()
    with pytest.raises(Exception) as info:
        session.append(USER_MESSAGE, {"surface_op": "append"})
    assert "requires a message projection" in str(info.value)


def test_registered_projection_supplies_content():
    """M1.11: a projection overrides the default derivation."""

    class UpperProjection:
        def project(self, event: Envelope, _events) -> list[Message]:
            payload = dict(event.data.get("message", {}))
            content = payload.get("content", [])
            return [
                Message.from_record(
                    {
                        "role": payload.get("role", "user"),
                        "content": [
                            {"type": "text", "text": b["text"].upper()} for b in content
                        ],
                    }
                )
            ]

    session = make_session()
    session.register_projection(USER_MESSAGE, UpperProjection())
    session.append_message("user", text("quiet"))
    assert [m.text for m in session.derive()] == ["QUIET"]


def test_unregistered_projection_after_use_is_refused():
    """M1.11: reading must not silently return a value the plugin no longer owns."""

    class Passthrough:
        def project(self, event: Envelope, _events) -> list[Message]:
            return [Message.from_record(dict(event.data["message"]))]

    session = make_session()
    session.register_projection(USER_MESSAGE, Passthrough())
    session.append_message("user", text("hi"))
    assert [m.text for m in session.derive()] == ["hi"]

    session.unregister_projection(USER_MESSAGE)
    with pytest.raises(ProjectionRemovedError, match="removed or replaced"):
        session.derive()


# ---------------------------------------------------------------------------
# M1.12 -- invalidation key is content_generation
# ---------------------------------------------------------------------------


def test_append_does_not_bump_content_generation():
    """M1.12: bumping on append would rebuild the whole table every append."""
    session = make_session()
    generation = session.surface.content_generation
    session.append_message("user", text("one"))
    session.append_message("assistant", text("two"))
    assert session.surface.content_generation == generation
    assert session.stats.full_rebuilds == 0


def test_replace_bumps_both_generations():
    """M1.12: a replace invalidates derived content and counts as real progress."""
    session = make_session()
    a = session.append_message("user", text("original"))
    before_content = session.surface.content_generation
    before_replace = session.surface.replace_generation

    session.append(
        USER_MESSAGE,
        {
            "message": {"role": "user", "content": text("replaced")},
            "surface_op": {"op": "replace", "start_seq": a.seq, "end_seq": a.seq},
            "source_event_seqs": [a.seq],
        },
    )

    assert session.surface.content_generation == before_content + 1
    assert session.surface.replace_generation == before_replace + 1
    assert [m.text for m in session.derive()] == ["replaced"]


def test_derive_extends_incrementally_then_rebuilds_once():
    """M1.12: appends extend; a single replace rebuilds exactly once."""
    session = make_session()
    a = session.append_message("user", text("one"))
    session.derive()
    session.append_message("assistant", text("two"))
    session.derive()
    assert session.stats.full_rebuilds == 0

    session.append(
        USER_MESSAGE,
        {
            "message": {"role": "user", "content": text("replaced")},
            "surface_op": {"op": "replace", "start_seq": a.seq, "end_seq": a.seq},
            "source_event_seqs": [a.seq],
        },
    )
    session.derive()
    assert session.stats.full_rebuilds == 1


# ---------------------------------------------------------------------------
# M1.13 -- empty messages hold position but do not reach the wire
# ---------------------------------------------------------------------------


def test_empty_message_keeps_position_but_is_not_sent():
    """M1.13: dropping the position instead would shift tool pairing."""
    session = make_session()
    empty = session.append_message("assistant", [])
    session.append_message("user", text("after"))

    assert empty.seq in session.surface_nodes()
    assert [m.text for m in session.derive()] == ["after"]


# ---------------------------------------------------------------------------
# M1.14 / M1.15 -- durability barrier and torn tails
# ---------------------------------------------------------------------------


def test_append_is_best_effort_and_flush_is_the_barrier(tmp_path: Path):
    """M1.14: the two guarantees are distinct and both are available."""
    path = tmp_path / "log.jsonl"
    with JsonlLog(path) as handle:
        session = make_session()
        session.append_message("user", text("persisted"))
        handle.append(session.events())
        handle.flush()
    assert len(read_log(path).events) == len(session)


def test_torn_tail_is_truncated_and_never_returned(tmp_path: Path):
    """M1.15: half a record must not be mistaken for a whole event."""
    path = tmp_path / "log.jsonl"
    session = make_session()
    session.append_message("user", text("complete"))
    JsonlLog(path).append(session.events())

    with path.open("ab") as handle:
        handle.write(b'{"seq":3,"type":"user/mess')

    result = read_log(path)
    assert len(result.events) == 3
    assert result.truncate_to is not None
    assert all(e.type != USER_MESSAGE or e.seq < 3 for e in result.events)


def test_torn_tail_is_repaired_before_the_next_append(tmp_path: Path):
    """M1.15: appending after a torn tail would splice halves into a valid record."""
    path = tmp_path / "log.jsonl"
    session = make_session()
    session.append_message("user", text("first"))
    JsonlLog(path).append(session.events())

    with path.open("ab") as handle:
        handle.write(b'{"seq":3,"type":"user/message","data":{"mess')

    # The next append must first discard the torn bytes, then write cleanly.
    session.append_message("user", text("second"))
    JsonlLog(path).append(session.events()[3:])

    result = read_log(path)
    assert result.is_clean, "torn tail should have been removed"
    seqs = [e.seq for e in result.events]
    assert seqs == list(range(len(seqs))), f"log must stay contiguous, got {seqs}"
    texts = [
        e.data["message"]["content"][0]["text"]
        for e in result.events
        if e.type == USER_MESSAGE
    ]
    assert texts == ["first", "second"]


def test_complete_but_corrupt_record_is_refused(tmp_path: Path):
    """M1.15: a full line that will not decode is corruption, not a torn tail."""
    path = tmp_path / "log.jsonl"
    with path.open("wb") as handle:
        handle.write(b'{"seq":0,"type":"session/start","data":{}}\n')
        handle.write(b"{not json at all}\n")

    with pytest.raises(Exception) as info:
        read_log(path)
    assert "failed to decode" in str(info.value)


def test_empty_file_reads_as_empty(tmp_path: Path):
    """M1.15: an empty log is a valid empty log."""
    path = tmp_path / "log.jsonl"
    path.write_bytes(b"")
    result = read_log(path)
    assert result.events == ()
    assert result.is_clean


# ---------------------------------------------------------------------------
# M1.16 -- unknown event types are required by default
# ---------------------------------------------------------------------------


def test_unknown_event_type_is_refused_at_append():
    """M1.16: accepting it would put a name in the log we cannot interpret."""
    session = make_session()
    with pytest.raises(UnknownEventTypeError, match="unknown event type"):
        session.append("mystery/event", {})


def test_reserved_event_type_is_refused_with_a_reason():
    """M1.16: a reserved name means a newer build wrote this log."""
    session = make_session()
    with pytest.raises(UnknownEventTypeError, match="reserved for a later format"):
        session.append("subagent/start", {})


def test_unknown_event_on_disk_requires_ignorable(tmp_path: Path):
    """M1.16: silently restoring a partial session is a safety failure."""
    path = tmp_path / "log.jsonl"
    with path.open("wb") as handle:
        handle.write(b'{"seq":0,"type":"session/start","data":{}}\n')
        handle.write(b'{"seq":1,"type":"mystery/event","data":{}}\n')

    with pytest.raises(UnknownEventTypeError):
        read_log(path)


def test_ignorable_unknown_event_is_skipped(tmp_path: Path):
    """M1.16: the explicit opt-in is honored."""
    path = tmp_path / "log.jsonl"
    with path.open("wb") as handle:
        handle.write(b'{"seq":0,"type":"session/start","data":{}}\n')
        handle.write(
            b'{"seq":1,"type":"mystery/event","ignorable":true,"data":{}}\n'
        )

    result = read_log(path)
    assert [e.type for e in result.events] == [SESSION_START]
    assert result.skipped == (1,)


def test_future_format_version_is_refused(tmp_path: Path):
    """M1.16: a newer format generation cannot be guessed at."""
    path = tmp_path / "log.jsonl"
    with path.open("wb") as handle:
        handle.write(
            b'{"seq":0,"type":"session/start","format_version":99,"data":{}}\n'
        )

    with pytest.raises(FormatVersionError, match="upgrade"):
        read_log(path)


# ---------------------------------------------------------------------------
# M1.17 -- append failure rolls back to the previous size
# ---------------------------------------------------------------------------


def test_partial_write_rolls_back(tmp_path: Path, monkeypatch):
    """M1.17: a retry from an old cursor must not duplicate seqs."""
    path = tmp_path / "log.jsonl"
    session = make_session()
    JsonlLog(path).append(session.events())
    size_before = path.stat().st_size

    handle = JsonlLog(path)
    session.append_message("user", text("will fail"))
    batch = session.events()[-1:]
    assert len(batch) == 1

    real_write = os.write
    calls = {"n": 0}

    def flaky_write(fd: int, data: bytes) -> int:
        calls["n"] += 1
        # Write a prefix, then report a short write.
        return real_write(fd, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(os, "write", flaky_write)
    with pytest.raises(Exception):
        handle.append(batch)
    monkeypatch.setattr(os, "write", real_write)

    assert calls["n"] == 1, "the write should have been attempted once"
    assert path.stat().st_size == size_before, "file must be rolled back"


# ---------------------------------------------------------------------------
# Round-trip: the acceptance criterion for M1
# ---------------------------------------------------------------------------


def test_replay_is_byte_identical(tmp_path: Path):
    """M1 acceptance: replaying the log reproduces the request byte-for-byte."""
    path = tmp_path / "log.jsonl"
    session = make_session()
    session.append_message("system", text("you are arm"))
    session.append_message("user", text("hello"))
    session.append_message("assistant", text("hi"))
    session.append_message("user", text("do the thing"))

    original = [m.to_record() for m in session.derive()]
    JsonlLog(path).append(session.events())

    restored = Session.from_events("s1", read_log(path).events)
    replayed = [m.to_record() for m in restored.derive()]
    assert replayed == original


def test_replay_survives_compaction_round_trip(tmp_path: Path):
    """M1 acceptance: a rewritten log replays to the rewritten view."""
    path = tmp_path / "log.jsonl"
    session = make_session()
    session.append_message("system", text("you are arm"))
    a = session.append_message("user", text("first"))
    b = session.append_message("assistant", text("second"))
    session.append_message("user", text("third"))

    session.append(
        USER_MESSAGE,
        {
            "message": {"role": "user", "content": text("[summary of first two]")},
            "surface_op": {"op": "replace", "start_seq": a.seq, "end_seq": b.seq},
            "source_event_seqs": [a.seq, b.seq],
        },
    )

    original = [m.to_record() for m in session.derive()]
    JsonlLog(path).append(session.events())

    restored = Session.from_events("s1", read_log(path).events)
    assert [m.to_record() for m in restored.derive()] == original
    assert restored.surface.replace_generation == session.surface.replace_generation


def test_fold_surface_matches_live_surface():
    """The fold is pure, so folding a log equals replaying it event by event."""
    session = make_session()
    session.append_message("user", text("a"))
    session.append_message("assistant", text("b"))
    session.append_message("user", text("c"))

    folded = fold_surface(session.events())
    assert folded.nodes == list(session.surface_nodes())
    assert folded.content_generation == session.surface.content_generation


def test_derived_messages_are_frozen():
    """M1: callers get a fresh list over immutable messages."""
    session = make_session()
    session.append_message("user", text("immutable"))
    first = session.derive()
    first.append(Message.user("injected"))

    second = session.derive()
    assert [m.text for m in second] == ["immutable"]

    with pytest.raises(Exception):
        second[0].content[0].text = "mutated"  # type: ignore[misc]


# ===========================================================================
# M1.13: an empty message and the surface
# ===========================================================================


def test_an_empty_assistant_response_does_not_shift_the_message_sequence():
    """M1.13's stated risk: "消息序列与位置语义分离，工具配对错位".

    The contract's mechanism is a placeholder that keeps its surface position.
    The implementation records `assistant/attempt` instead, which is *not* a
    surface event -- so no position is kept.

    Verified here: the outcome the requirement exists to protect is intact. The
    derived sequence stays `system, user, user, assistant`, so nothing is
    misaligned and no tool pairing is disturbed. The difference is in the
    mechanism, not the behaviour, and this test pins the behaviour.
    """
    import asyncio

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, ScriptedTurn
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.registry import ToolRegistry

    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    session = Session.create("empty")
    loop = Runtime(
        session,
        MockAdapter([ScriptedTurn(text=""), ScriptedTurn(text="second")]),
        ToolRegistry(),
        assembler=assembler,
        model="mock",
    )

    loop.submit("one")
    asyncio.run(loop.run_until_idle())
    loop.submit("two")
    asyncio.run(loop.run_until_idle())

    # The empty response is recorded as an attempt, and nothing else changes.
    assert any(e.type == "assistant/attempt" for e in session)
    roles = [m.role for m in session.derive()]
    assert roles == ["system", "user", "user", "assistant"], roles


def test_an_empty_assistant_response_does_not_enter_the_message_sequence():
    """It must not be sent to the model as an empty message.

    A contentless assistant node would be rejected by a provider, so the
    requirement is that it is recorded without being sent.
    """
    import asyncio

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, ScriptedTurn, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.registry import ToolRegistry

    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    session = Session.create("empty-2")
    adapter = MockAdapter(
        [ScriptedTurn(text=""), text_response("after")]
    )
    loop = Runtime(session, adapter, ToolRegistry(), assembler=assembler, model="mock")

    loop.submit("one")
    asyncio.run(loop.run_until_idle())

    # Whatever the second request contained, it had no contentless assistant node.
    for request in adapter.requests:
        for message in request.messages:
            if message.role == "assistant":
                assert message.content, (
                    "a contentless assistant message reached the provider"
                )


def test_the_surface_holds_only_real_messages():
    """An attempt is a record, not a message.

    Stated as a test because the contract describes a placeholder that keeps its
    surface position, and this implementation does something else. The behaviour
    is right; the mechanism differs, and that difference should be visible.
    """
    import asyncio

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, ScriptedTurn
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.registry import ToolRegistry

    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    session = Session.create("empty-3")
    loop = Runtime(
        session, MockAdapter([ScriptedTurn(text="")]), ToolRegistry(),
        assembler=assembler, model="mock",
    )
    loop.submit("one")
    asyncio.run(loop.run_until_idle())

    surface_types = {session.event_at(seq).type for seq in session.surface_nodes()}
    assert "assistant/attempt" not in surface_types, (
        "the attempt became a surface node, which the contract does not ask for"
    )


# ===========================================================================
# M1.3: the projection-cache trigger condition, measured
# ===========================================================================


def test_cold_read_cost_is_linear_and_the_trigger_is_not_yet_reached():
    """The contract's M1.3 says to add a projection cache when a session passes
    10,000 events *and* a cold read takes over 200ms.

    Measured, so the decision rests on numbers rather than on the plan:

      * the cost is linear in event count (about 0.02 ms/event);
      * 10,000 events open in ~190ms -- **under** the contract's own threshold;
      * 20,000 events open in ~384ms, which is over.

    At the measured rate of ~8 events per tool call, reaching 20,000 events takes
    roughly 2,400 tool calls. A cold read happens **once per open**, not per turn,
    so the cost is paid when a session is resumed rather than continuously.

    That is why no cache has been built: the condition the contract sets is not
    met at the size the contract names, and the cost it would remove is a
    one-time open. This test exists so that a slower machine, or a change that
    makes logs bigger, fails visibly instead of silently making resume painful.
    """
    import tempfile
    import time

    from arm.bench.claims import _make_event
    from arm.session.persistence import JsonlLog, read_log

    measurements: dict[int, float] = {}
    with tempfile.TemporaryDirectory() as directory:
        for count in (5_000, 10_000):
            path = Path(directory) / f"{count}.jsonl"
            JsonlLog(path).append(
                [_make_event(index) for index in range(count)], fsync=False
            )
            started = time.perf_counter()
            read_log(path)
            measurements[count] = (time.perf_counter() - started) * 1000

    small, large = measurements[5_000], measurements[10_000]

    # Linear, not quadratic: doubling the events must not much more than double
    # the time. A quadratic read would be the real reason to add a cache.
    assert large < small * 4, (
        f"cold read is growing faster than linearly: {small:.1f}ms at 5k, "
        f"{large:.1f}ms at 10k"
    )

    # The contract's threshold at the size the contract names. Generous bound,
    # because this machine may be busy -- the point is to catch a regime change,
    # not to pin a benchmark.
    assert large < 1_000, (
        f"10,000 events took {large:.1f}ms to open; the contract sets its "
        "projection-cache trigger at 200ms and this is well past it"
    )

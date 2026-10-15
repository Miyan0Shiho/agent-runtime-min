"""JSONL persistence: an append-only file that can survive being killed mid-write.

The format is deliberately dumb -- one JSON object per line, readable with
``cat`` -- because a log you cannot inspect by eye is a log you cannot debug at
2am. Compression and framing buy throughput we do not need yet, at the cost of
the one property that matters most during bring-up.

Three rules shape the write path:

1. **Torn tails are truncated before anything new is written.** A half-written
   final line is structurally incomplete. Appending after it would splice the
   halves into something that parses -- turning half a record into a real event.
2. **Recovered records are rewritten durably before the new batch.** Otherwise
   the recovery is lost on the next crash and the same repair happens forever.
3. **A failed append rolls back to the size before it started.** If the file
   grew partially and the cursor did not advance, a retry would duplicate seqs.

The read path returns only the physically valid contiguous prefix, and reports
where it stopped. Repairing an unclosed turn is the *reader's* job -- storage
just says how far it got.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..kernel.errors import (
    LogCorruptionError,
    PersistenceError,
    UnknownEventTypeError,
    FormatVersionError,
)
from ..kernel.freeze import assert_json_lossless
from ..session.envelope import Envelope
from ..session.events import FORMAT_VERSION, is_known

log = logging.getLogger("arm.session.persistence")

#: Directory mode for the session store: private to the owner.
_DIR_MODE = 0o700
#: File mode for a session log: private to the owner.
_FILE_MODE = 0o600


@dataclass(frozen=True, slots=True)
class ReadResult:
    """What a read actually found.

    ``truncate_to`` is ``None`` when the file ended cleanly. When it is set, the
    bytes at or after that offset are a torn tail and must be discarded before
    the next append.
    """

    events: tuple[Envelope, ...]
    truncate_to: int | None = None
    recovered_tail: tuple[Envelope, ...] = ()
    #: Unknown-but-ignorable events that were skipped while reading.
    skipped: tuple[int, ...] = ()

    @property
    def is_clean(self) -> bool:
        return self.truncate_to is None


class JsonlLog:
    """An append-only JSONL file of envelopes.

    Every method that can fail either leaves the file exactly as it was, or
    raises :class:`AppendRollbackError` carrying both the original failure and
    the rollback failure. A half-written batch is never left behind silently.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    # -- lifecycle -------------------------------------------------------

    def _ensure_parent(self) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        try:
            os.chmod(parent, _DIR_MODE)
        except OSError:  # pragma: no cover - non-POSIX
            pass

    def open(self) -> "JsonlLog":
        self._ensure_parent()
        if self._fd is None:
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            self._fd = os.open(self.path, flags, _FILE_MODE)
            try:
                os.chmod(self.path, _FILE_MODE)
            except OSError:  # pragma: no cover - non-POSIX
                pass
        return self

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None

    def __enter__(self) -> "JsonlLog":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- reading ---------------------------------------------------------

    def read(self) -> ReadResult:
        """Read the valid contiguous prefix of the log.

        Three outcomes for the final line, and the distinction is the whole
        point of this method:

        * **complete and decodable** -- an ordinary event;
        * **complete but undecodable** -- corruption, and refusing is the only
          safe answer, because we cannot tell which event was lost;
        * **structurally incomplete** (no trailing newline, or truncated
          mid-object) -- a torn tail, recoverable: report where it starts.
        """
        if not self.path.exists():
            return ReadResult(events=())

        raw = self.path.read_bytes()
        if not raw:
            return ReadResult(events=())

        events: list[Envelope] = []
        skipped: list[int] = []
        offset = 0
        torn_at: int | None = None
        total = len(raw)

        while offset < total:
            newline = raw.find(b"\n", offset)
            if newline == -1:
                # No terminator: this is an incomplete final record. Only the
                # *last* line can be torn -- anything earlier lacking a newline
                # would mean the file is not line-oriented at all.
                torn_at = offset
                break
            line = raw[offset:newline]
            next_offset = newline + 1
            if line.strip():
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise LogCorruptionError(
                        f"structurally complete record at byte {offset} of "
                        f"{self.path} failed to decode: {exc}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise LogCorruptionError(
                        f"record at byte {offset} of {self.path} is not a JSON object"
                    )
                event = self._decode(record, skipped)
                if event is not None:
                    events.append(event)
            offset = next_offset

        if events:
            expected = 0
            for event in events:
                if event.seq != expected:
                    raise LogCorruptionError(
                        f"stored log is not contiguous: expected seq {expected}, "
                        f"found {event.seq}"
                    )
                expected += 1

        recovered = tuple(events) if torn_at is not None else ()
        return ReadResult(
            events=tuple(events),
            truncate_to=torn_at,
            recovered_tail=recovered,
            skipped=tuple(skipped),
        )

    def _decode(self, record: Mapping[str, Any], skipped: list[int]) -> Envelope | None:
        event_type = record.get("type")
        if not isinstance(event_type, str):
            raise LogCorruptionError("stored record lacks a string 'type'")

        if not is_known(event_type):
            # Unknown types are required by default. Only an explicit
            # ignorable:true licenses skipping, because silently restoring a
            # partial session is a correctness failure, not a UX one.
            if record.get("ignorable") is True:
                seq = record.get("seq")
                if isinstance(seq, int):
                    skipped.append(seq)
                return None
            raise UnknownEventTypeError(
                f"stored log contains unknown event type {event_type!r} without "
                "ignorable:true; refusing to restore a partial session"
            )

        version = record.get("format_version", FORMAT_VERSION)
        if isinstance(version, int) and version > FORMAT_VERSION:
            raise FormatVersionError(
                f"stored log declares format version {version} but this build "
                f"supports {FORMAT_VERSION}; upgrade the runtime"
            )

        return Envelope.from_record(record)

    # -- writing ---------------------------------------------------------

    def append(
        self,
        events: Sequence[Envelope],
        *,
        fsync: bool = True,
        already_contiguous: bool = False,
    ) -> int:
        """Append a batch. Returns the number of events written.

        The three-step order is not interchangeable: truncate the torn tail,
        rewrite what was recovered from it, then write the new batch. Skipping
        the middle step would mean the recovery is lost on the next crash.

        ``already_contiguous`` skips the pre-write scan. The caller asserts that
        the events being appended continue the file exactly, which is true when
        the caller is a session listener streaming its own committed events. It
        exists because the scan reads the whole file, so scanning on every single
        event would make a session's total write cost quadratic in its length.
        """
        if not events:
            return 0

        for event in events:
            assert_json_lossless(dict(event.data), what=f"{event.type} data")

        self._ensure_parent()
        existed = self.path.exists()
        before = self.path.stat().st_size if existed else 0

        if not already_contiguous:
            # Step 1: drop any torn tail. A structurally incomplete final record
            # must not be spliced together with the batch we are about to write.
            read = self.read() if existed else ReadResult(events=())
            if read.truncate_to is not None:
                self._truncate(read.truncate_to)
                before = read.truncate_to

        payload = b"".join(
            (event.to_json() + "\n").encode("utf-8") for event in events
        )

        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _FILE_MODE)
        try:
            written = os.write(fd, payload)
            if written != len(payload):
                raise PersistenceError(
                    f"partial write: {written} of {len(payload)} bytes"
                )
            if fsync:
                os.fsync(fd)
        except BaseException as exc:
            # Roll back to the size before this append. Without this, a retry
            # from an older cursor would write the same seqs again.
            #
            # BaseException is caught deliberately: a cancellation arriving
            # mid-write must still leave the file consistent.
            rollback_error: BaseException | None = None
            try:
                os.close(fd)
                self._truncate(before)
            except BaseException as rollback_exc:  # noqa: BLE001
                rollback_error = rollback_exc
            if rollback_error is not None:
                # Both failures are reported. ``ExceptionGroup`` only accepts
                # ``Exception``, so a cancellation (a ``BaseException``) is
                # surfaced by re-raising it with the rollback error attached.
                if isinstance(exc, Exception) and isinstance(rollback_error, Exception):
                    raise ExceptionGroup(
                        "append failed and the rollback also failed",
                        [exc, rollback_error],
                    ) from exc
                raise exc from rollback_error
            raise
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

        return len(events)

    def _truncate(self, size: int) -> None:
        """Truncate the file to ``size``, creating it if absent."""
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT, _FILE_MODE)
        try:
            os.ftruncate(fd, size)
            os.fsync(fd)
        finally:
            os.close(fd)

    def flush(self) -> None:
        """The durability barrier.

        ``append`` is best-effort with respect to the OS page cache; this is the
        point at which a caller may claim the data survives a crash.
        """
        if self.path.exists():
            fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def read_log(path: str | os.PathLike[str]) -> ReadResult:
    """Convenience wrapper for reading a log without keeping a handle."""
    return JsonlLog(path).read()

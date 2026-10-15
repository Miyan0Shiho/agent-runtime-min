"""The output ring: one buffer, two kinds of reader, no interference.

A background job produces output that two different parties want to read:

* **the model**, which consumes it -- once read, it is delivered and the model
  should not see it again;
* **an observer** (a UI, a log tail, an evaluation), which reads the same bytes at
  its own pace without affecting anyone else.

The naive designs both fail. "Read once and delete" breaks the observer. "Keep
everything forever" grows without bound. The ring solves it with **absolute byte
offsets**: writers only ever append and advance a high-water mark; a reader keeps
its own offset and reads from there. Nobody's position moves anyone else's.

Three details make it correct rather than merely plausible:

**Trimming never cuts bytes an unread cursor still needs.** When the buffer is
over its limit, the head is discarded and ``earliest`` advances -- but only up to
the lowest outstanding pin. A model that has not yet read the output must not
have it silently dropped; if the limit genuinely cannot be respected without
doing so, the limit is what gives way. An unbounded buffer is a resource problem;
silently dropping output is a correctness problem.

**A gap is announced, not hidden.** When a single chunk is larger than the whole
budget, or when a reader's offset has fallen behind the retained window, the
reader is told how many bytes it missed. A reader that silently receives a
truncated stream will reason about it as though it were complete.

**Offsets are bytes, not characters.** The limit is a memory limit, and a
multi-byte character is not one byte. Slicing happens on bytes and decoding only
at the edges, so a reader never receives half a character.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of output, addressed by its absolute start offset."""

    offset: int
    text: str

    @property
    def size(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True, slots=True)
class ReadResult:
    """What a reader got, and what it might have missed."""

    text: str
    #: The offset the next read should start from.
    next_offset: int
    #: Whether the reader has caught up with the writer.
    at_end: bool = True
    #: Bytes that were dropped before this read could see them.
    gap_bytes: int = 0

    @property
    def has_gap(self) -> bool:
        return self.gap_bytes > 0

    def render_gap_note(self) -> str:
        """A line to prepend when output was missed.

        Rendered by the ring rather than by each caller so the wording is
        consistent -- and so a caller cannot forget to mention it.
        """
        if not self.has_gap:
            return ""
        return f"… [{self.gap_bytes} bytes of output were dropped before this point] …\n"


@dataclass(frozen=True, slots=True)
class _Gap:
    """A range of offsets that no longer exists."""

    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


class OutputRing:
    """An append-only byte buffer with independent read cursors."""

    def __init__(self, *, limit_bytes: int = 262_144) -> None:
        if limit_bytes < 1:
            raise ValueError("limit_bytes must be positive")
        self.limit_bytes = limit_bytes
        self._chunks: list[Chunk] = []
        #: Total bytes ever written, including what has been discarded.
        self._written = 0
        #: The lowest offset still available.
        self._earliest = 0
        #: Named offsets that must not be trimmed past.
        self._pins: dict[str, int] = {}
        #: Ranges that were dropped, so a reader can be told how much it missed.
        self._gaps: list[_Gap] = []

    # -- writing ---------------------------------------------------------

    @property
    def written(self) -> int:
        """Total bytes ever appended: the writer's high-water mark."""
        return self._written

    @property
    def earliest(self) -> int:
        """The lowest offset a reader can still obtain."""
        return self._earliest

    def append(self, text: str) -> None:
        """Append text, trimming the head if the limit is exceeded."""
        if not text:
            return
        data = text.encode("utf-8")

        if len(data) > self.limit_bytes:
            # A single chunk larger than the whole budget. Keep a UTF-8-safe tail
            # and record the rest as a gap.
            keep = self._tail_on_char_boundary(data, self.limit_bytes)
            dropped = len(data) - len(keep)
            start = self._written
            self._written += dropped
            self._record_gap(start, self._written)
            self._chunks.clear()
            self._earliest = self._written
            self._append_bytes(keep)
            self._trim()
            return

        self._append_bytes(data)
        self._trim()

    def _append_bytes(self, data: bytes) -> None:
        offset = self._written
        self._written += len(data)
        self._chunks.append(Chunk(offset=offset, text=data.decode("utf-8", "replace")))

    @staticmethod
    def _tail_on_char_boundary(data: bytes, limit: int) -> bytes:
        """The last ``limit`` bytes, moved forward to a character boundary.

        Cutting mid-character would put a replacement character at the start of
        what a reader receives -- a corruption it cannot distinguish from real
        output.
        """
        tail = data[-limit:]
        for index in range(min(4, len(tail))):
            if tail[index] & 0xC0 != 0x80:
                return tail[index:]
        return tail

    def _record_gap(self, start: int, end: int) -> None:
        if end <= start:
            return
        self._gaps.append(_Gap(start, end))
        # Keep the gap list bounded: only the most recent few can matter to a
        # reader that is anywhere near the live end.
        if len(self._gaps) > 64:
            self._gaps = self._gaps[-64:]

    def _trim(self) -> None:
        """Discard the head until the limit is respected, respecting pins.

        Never trims past a pin. A cursor that has not read the output yet is
        entitled to it, and the memory limit gives way rather than the data.
        """
        pinned = min(self._pins.values()) if self._pins else self._written

        while self._chunks:
            if self._written - self._earliest <= self.limit_bytes:
                break
            head = self._chunks[0]
            if head.end > pinned:
                break
            self._chunks.pop(0)
            self._earliest = head.end
            self._record_gap(head.offset, head.end)

        # Drop anything that starts before the earliest readable offset.
        while self._chunks and self._chunks[0].end <= self._earliest:
            self._chunks.pop(0)

    # -- reading ---------------------------------------------------------

    def pin(self, name: str, offset: int) -> None:
        """Protect everything from ``offset`` onward from trimming."""
        self._pins[name] = offset

    def unpin(self, name: str) -> None:
        self._pins.pop(name, None)

    def read(self, offset: int) -> ReadResult:
        """Read everything from ``offset`` onward, without consuming it.

        Non-consuming by design: this is the observer's read. The model's
        consuming read is a *cursor* the caller advances, and that cursor also
        acts as a pin while it lags.
        """
        gap = 0
        if offset < self._earliest:
            gap = self._earliest - offset
            offset = self._earliest
        else:
            # The offset may still fall inside a recorded gap.
            for recorded in self._gaps:
                if recorded.start <= offset < recorded.end:
                    gap += recorded.end - offset
                    offset = recorded.end
                    break

        parts: list[str] = []
        for chunk in self._chunks:
            if chunk.end <= offset:
                continue
            if chunk.offset >= offset:
                parts.append(chunk.text)
                continue
            # The offset falls inside this chunk. Slice by bytes so the
            # arithmetic matches the offsets, then decode.
            data = chunk.text.encode("utf-8")
            parts.append(data[offset - chunk.offset :].decode("utf-8", "replace"))

        return ReadResult(
            text="".join(parts),
            next_offset=self._written,
            at_end=True,
            gap_bytes=gap,
        )

    def size(self) -> int:
        """Bytes currently retained."""
        return self._written - self._earliest

    def __len__(self) -> int:
        return len(self._chunks)

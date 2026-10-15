"""The durable envelope: one event, as it exists in memory and on disk.

An :class:`Envelope` is the unit of truth. Its ``seq`` is not validated against
anything -- it *is* ``len(log)``, which makes continuity a definition rather than
a check. There is no way to construct a log with a hole in it, so there is no
code path that has to detect one.

The envelope is frozen and JSON-lossless by construction. Both properties are
enforced at build time, because an event that can mutate after being observed
would make "the log is the truth" false, and an event that cannot round-trip
would make replay and migration disagree about what happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from ..kernel.freeze import FrozenDict, assert_json_lossless, deep_freeze, thaw

#: Fields that may never appear inside ``data``: they belong to the envelope.
RESERVED_DATA_KEYS = frozenset({"seq", "type", "ts", "format_version", "ignorable"})


def utc_now_iso() -> str:
    """Timestamp in ISO-8601 with a trailing ``Z``.

    Seconds resolution is deliberate: sub-second precision buys nothing here and
    makes logs harder to diff by eye.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class Envelope:
    """One durable event.

    ``ignorable`` is the forward-compatibility escape hatch: an unknown event
    type is *required* by default, and only a writer that explicitly marked it
    ignorable may be skipped. The asymmetry is intentional -- refusing to open a
    log is an inconvenience, whereas silently restoring half a session is a
    correctness failure.
    """

    seq: int
    type: str
    data: Mapping[str, Any]
    ts: str = field(default_factory=utc_now_iso)
    format_version: int = 1
    ignorable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.seq, int) or self.seq < 0:
            raise ValueError(f"seq must be a non-negative int, got {self.seq!r}")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("type must be a non-empty string")
        if not isinstance(self.format_version, int) or self.format_version < 0:
            raise ValueError("format_version must be a non-negative int")

        # Normalize ``data`` into a frozen mapping without copying one that is
        # already frozen, so repeated freeze passes stay cheap.
        raw = self.data
        if not isinstance(raw, FrozenDict):
            if not isinstance(raw, Mapping):
                raise TypeError(f"data must be a mapping, got {type(raw).__name__}")
            leaked = RESERVED_DATA_KEYS.intersection(raw)
            if leaked:
                raise ValueError(
                    f"data must not carry envelope fields: {sorted(leaked)}"
                )
            assert_json_lossless(dict(raw), what=f"{self.type} data")
            object.__setattr__(self, "data", deep_freeze(dict(raw)))
        else:
            leaked = RESERVED_DATA_KEYS.intersection(raw)
            if leaked:
                raise ValueError(
                    f"data must not carry envelope fields: {sorted(leaked)}"
                )

    # -- accessors -------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def require(self, key: str) -> Any:
        """Fetch a field that the contract says must exist."""
        if key not in self.data:
            raise KeyError(f"{self.type} event at seq {self.seq} lacks {key!r}")
        return self.data[key]

    def surface_op(self) -> Any:
        return self.data.get("surface_op")

    # -- serialization ---------------------------------------------------

    def to_record(self) -> dict[str, Any]:
        """The on-disk form: a flat object with envelope fields inline."""
        record: dict[str, Any] = {
            "seq": self.seq,
            "type": self.type,
            "ts": self.ts,
            "format_version": self.format_version,
        }
        if self.ignorable:
            record["ignorable"] = True
        record["data"] = thaw(self.data)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Envelope":
        """Rebuild from disk. Missing required fields are a hard error."""
        missing = [k for k in ("seq", "type") if k not in record]
        if missing:
            raise ValueError(f"stored event lacks required fields: {missing}")
        data = record.get("data", {})
        if not isinstance(data, Mapping):
            raise TypeError(f"stored event data must be an object, got {type(data)}")
        return cls(
            seq=int(record["seq"]),
            type=str(record["type"]),
            data=dict(data),
            ts=str(record.get("ts") or utc_now_iso()),
            format_version=int(record.get("format_version", 1)),
            ignorable=bool(record.get("ignorable", False)),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_record(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )

    def with_data(self, **changes: Any) -> "Envelope":
        """A copy carrying a new ``data`` body, same identity fields."""
        merged = dict(self.data)
        merged.update(changes)
        return Envelope(
            seq=self.seq,
            type=self.type,
            data=merged,
            ts=self.ts,
            format_version=self.format_version,
            ignorable=self.ignorable,
        )

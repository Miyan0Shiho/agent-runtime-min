"""Deep freezing and JSON-losslessness.

Two guarantees live here, and both exist for the same reason: the log is the
only truth, so nothing that enters it may change afterwards, and nothing that
enters it may mean something different once written to disk than it did in
memory.

* :func:`deep_freeze` turns a payload into an immutable, hashable structure.
  Python's ``frozen=True`` dataclasses are shallow -- a nested dict stays
  mutable -- so the freezing is done explicitly and recursively here.

* :func:`assert_json_lossless` refuses any payload that does not survive a JSON
  round-trip. The disk format and the in-memory log must be isomorphic;
  otherwise replay and migration stop sharing one source of truth.

Tuples are the frozen form of lists. They compare equal to the *same* tuple and
serialize back to a JSON array, so the round-trip property is preserved.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .errors import PayloadNotSerializableError

_JSON_SCALARS = (str, bool, int, float, type(None))


class FrozenDict(dict):
    """A dict that refuses mutation.

    Subclasses ``dict`` so that structural equality, ``json.dumps`` and
    ``isinstance(x, dict)`` all keep working for downstream consumers.
    """

    __slots__ = ("_hash",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._hash: int | None = None

    # -- mutation is always an error -------------------------------------
    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable")

    def __hash__(self) -> int:  # type: ignore[override]
        if self._hash is None:
            self._hash = hash(tuple(sorted((k, v) for k, v in self.items())))
        return self._hash

    def __reduce__(self) -> Any:
        return (FrozenDict, (dict(self),))


def deep_freeze(value: Any) -> Any:
    """Recursively freeze ``value`` into an immutable structure.

    dict -> :class:`FrozenDict`, list -> tuple, set -> frozenset. Scalars pass
    through untouched.
    """
    if isinstance(value, (FrozenDict, tuple, frozenset, str, bytes, bool, int, float)):
        if isinstance(value, FrozenDict):
            return value
        return value
    if isinstance(value, dict):
        return FrozenDict({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Unknown object: give it a chance to be JSON-serializable, else refuse.
    assert_json_lossless(value, what=type(value).__name__)
    return value


def thaw(value: Any) -> Any:
    """Convert a frozen structure back to plain JSON-shaped containers."""
    if isinstance(value, FrozenDict) or isinstance(value, dict):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    if isinstance(value, frozenset):
        return [thaw(v) for v in sorted(value, key=repr)]
    return value


def _is_json_lossless(value: Any, path: str) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the losslessness of ``value``."""
    if value is None or isinstance(value, (str, bool, int)):
        return True, ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return False, f"{path}: non-finite float is not JSON"
        return True, ""
    if isinstance(value, (FrozenDict, dict)):
        for key, item in value.items():
            if not isinstance(key, str):
                return False, f"{path}: object key {key!r} is not a string"
            ok, reason = _is_json_lossless(item, f"{path}.{key}")
            if not ok:
                return False, reason
        return True, ""
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ok, reason = _is_json_lossless(item, f"{path}[{index}]")
            if not ok:
                return False, reason
        return True, ""
    if isinstance(value, (bytes, bytearray)):
        return False, f"{path}: bytes must be encoded to str before logging"
    return False, f"{path}: {type(value).__name__} is not JSON-serializable"


def assert_json_lossless(value: Any, *, what: str) -> None:
    """Raise unless ``value`` survives a JSON round-trip unchanged.

    The check is a real round-trip rather than a type whitelist: a payload can
    be made of JSON-ish types and still lose information (a tuple becoming a
    list is fine and expected; a dict with integer keys silently becoming
    strings is not).
    """
    ok, reason = _is_json_lossless(value, what)
    if not ok:
        raise PayloadNotSerializableError(
            f"{what} must be losslessly JSON-serializable ({reason})"
        )
    try:
        encoded = json.dumps(thaw(value), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PayloadNotSerializableError(
            f"{what} must be losslessly JSON-serializable ({exc})"
        ) from exc
    decoded = json.loads(encoded)
    original = thaw(value)
    if decoded != original:
        raise PayloadNotSerializableError(
            f"{what} does not survive a JSON round-trip unchanged"
        )

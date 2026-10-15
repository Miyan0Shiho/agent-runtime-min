"""Representative pydantic models and payload corpora for the differential test.

Each :class:`Case` pairs a pydantic model with a hand-curated payload corpus
(valid, invalid, boundary, wrong-type, null, extra-field) plus programmatically
generated mutations of the valid payloads. The differential test then asks the
same question of every payload:

    does the RAW pydantic schema accept it?  ==  does the NORMALIZED schema?

A disagreement in either direction is a finding:

* normalized accepts, raw rejects  -> the normalizer **widened** (a constraint
  was dropped: invalid arguments reach the tool body);
* raw accepts, normalized rejects  -> the normalizer **narrowed** (a valid call
  is refused, or the normalizer crashed on a legal schema).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------


class Flat(BaseModel):
    """Required scalars only -- the baseline pydantic shape."""

    a: int
    b: str
    c: float
    d: bool


class OptionalDefault(BaseModel):
    """``Optional[X]`` *with* a default -> ``anyOf`` + ``default``."""

    x: int = 3
    y: Optional[int] = None


class OptionalNoDefault(BaseModel):
    """``Optional[X]`` without a default -> still required, but nullable."""

    x: Optional[int]


class Inner(BaseModel):
    value: int


class Outer(BaseModel):
    """Nested model -> ``$defs`` + ``$ref``."""

    inner: Inner
    name: str = "default"


class WithLiteral(BaseModel):
    """``Literal`` -> ``const`` (single) and ``enum`` (multiple)."""

    one: Literal["a"]
    two: Literal["a", "b"]
    three: Literal[1, "x", None]


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class WithEnum(BaseModel):
    """``str``-Enum -> ``$defs`` entry + ``$ref``."""

    color: Color
    color2: Color = Color.RED


class ListOfModel(BaseModel):
    items: list[Inner]
    matrix: list[list[int]]


class DictOfModel(BaseModel):
    mapping: dict[str, Inner]
    plain: dict[str, int]


class A(BaseModel):
    kind: Literal["a"]
    av: int


class B(BaseModel):
    kind: Literal["b"]
    bv: str


class PlainUnion(BaseModel):
    """``Union`` -> ``anyOf`` of ``$ref``s."""

    u: Union[int, str]
    m: Union[A, B]


class Discriminated(BaseModel):
    """``Field(discriminator=...)`` -> ``oneOf`` + ``discriminator``."""

    du: Annotated[Union[A, B], Field(discriminator="kind")]


class Constrained(BaseModel):
    """``Annotated`` + ``Field`` constraints -> min/max/pattern."""

    s: Annotated[str, Field(min_length=2, max_length=5, pattern=r"^[a-z]+$")]
    n: Annotated[int, Field(ge=0, le=10, multiple_of=2)]
    l: Annotated[list[int], Field(min_length=1, max_length=3)]


class Recursive(BaseModel):
    """Self-referencing model -> genuinely recursive ``$ref``."""

    name: str
    child: Optional["Recursive"] = None


class TupleModel(BaseModel):
    """Fixed tuple -> ``prefixItems`` + ``minItems``/``maxItems``."""

    t: tuple[int, str]
    t2: tuple[int, ...]


class Scalars(BaseModel):
    """``datetime``/``date``/``UUID``/``Decimal``/``bytes``."""

    dt: datetime
    d: date
    uid: UUID
    dec: Decimal
    raw: bytes


class ExtraForbid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: int


class ExtraAllow(BaseModel):
    model_config = ConfigDict(extra="allow")

    a: int


class RefSibling(BaseModel):
    """A ``$ref`` carrying a *validation* sibling via ``json_schema_extra``.

    Legal JSON Schema, legal pydantic, and the case where the normalizer's
    "siblings win" rule is not equivalent to "siblings conjoin".
    """

    inner: Annotated[Inner, Field(json_schema_extra={"required": ["zzz"]})]


class SchemaExtraValidation(BaseModel):
    """``json_schema_extra`` injecting real assertions next to ``type``."""

    v: Annotated[
        str,
        Field(json_schema_extra={"minProperties": 1, "properties": {"q": {"type": "integer"}}}),
    ]


class SetField(BaseModel):
    s: set[str]
    fs: frozenset[int]


class DeepChain13(BaseModel):
    """A non-recursive nesting chain deep enough to trip the depth counter."""

    class L0(BaseModel):
        v: int

    class L1(BaseModel):
        child: "DeepChain13.L0"

    class L2(BaseModel):
        child: "DeepChain13.L1"

    class L3(BaseModel):
        child: "DeepChain13.L2"

    class L4(BaseModel):
        child: "DeepChain13.L3"

    class L5(BaseModel):
        child: "DeepChain13.L4"

    class L6(BaseModel):
        child: "DeepChain13.L5"

    class L7(BaseModel):
        child: "DeepChain13.L6"

    class L8(BaseModel):
        child: "DeepChain13.L7"

    class L9(BaseModel):
        child: "DeepChain13.L8"

    class L10(BaseModel):
        child: "DeepChain13.L9"

    class L11(BaseModel):
        child: "DeepChain13.L10"

    class L12(BaseModel):
        child: "DeepChain13.L11"

    child: "DeepChain13.L12"


class AliasedRef(BaseModel):
    """A field whose **alias** is literally ``$ref``.

    Legal: ``$ref`` inside ``properties`` is a property *name*, not a reference.
    """

    inner: Inner
    ref: str = Field(alias="$ref")


class UnionWithEnumAndNone(BaseModel):
    u: Union[Inner, Color, None]


# ---------------------------------------------------------------------------
# Payload corpora
# ---------------------------------------------------------------------------

_MISSING = object()


def mutate(payload: Any, *, depth: int = 0) -> list[Any]:
    """Generate a spread of mutations of a valid payload.

    Deterministic (no RNG) so a failure is always reproducible: drop a key, null
    a value, swap a value's type, add an unknown key, empty a container, go
    out of range.
    """
    out: list[Any] = []
    if isinstance(payload, dict) and depth < 3:
        for key in list(payload):
            dropped = {k: v for k, v in payload.items() if k != key}
            out.append(dropped)
            out.append({**payload, key: None})
            out.append({**payload, key: "wrong-type"})
            out.append({**payload, key: 12345})
            out.append({**payload, key: []})
            out.append({**payload, key: {}})
            out.append({**payload, key: ""})
            out.append({**payload, key: -1})
            out.append({**payload, key: 10**9})
            out.extend(
                {**payload, key: m} for m in mutate(payload[key], depth=depth + 1)
            )
        out.append({**payload, "__extra__": 1})
        out.append({**payload, "extra": None})
    elif isinstance(payload, (list, tuple)) and depth < 3:
        seq = list(payload)
        out.append([])
        out.append(seq + [None])
        out.append(seq[:-1])
        for index in range(len(seq)):
            out.extend(
                seq[:index] + [m] + seq[index + 1 :]
                for m in mutate(seq[index], depth=depth + 1)
            )
    elif isinstance(payload, str):
        out.extend(["", " ", payload.upper(), payload + "x", payload * 3])
    elif isinstance(payload, bool):
        out.append(not payload)
    elif isinstance(payload, (int, float)):
        out.extend([0, 1, -1, payload + 1, payload - 1, 10**9, 0.5])
    return out


@dataclass
class Case:
    name: str
    model: type[BaseModel]
    #: Payloads used to *seed* the corpus; mutations are generated from them.
    valid: list[Any]
    #: Hand-written payloads aimed at the interesting boundary on either side.
    #: Their actual verdict is decided by the RAW schema, not by this label --
    #: see :func:`partition`.
    extra: list[Any] = field(default_factory=list)
    note: str = ""

    def payloads(self) -> list[tuple[Any, str]]:
        """(payload, origin) pairs: curated + generated mutations."""
        out: list[tuple[Any, str]] = []
        for payload in self.valid:
            out.append((payload, "curated-seed"))
            out.extend((m, "mutated") for m in mutate(payload))
        out.extend((p, "curated-boundary") for p in self.extra)
        return out


def partition(case: "Case", validator_factory) -> tuple[list[Any], list[Any]]:
    """Split a case's payloads by what the RAW pydantic schema says.

    The corpus is *defined* by the raw schema rather than by a hand-written
    valid/invalid label: a payload is "accepted" iff the raw schema accepts it.
    That is the only classification the differential test needs, and it removes
    the reviewer's own judgement as a source of error.
    """
    raw = case.model.model_json_schema()
    accepted, rejected = [], []
    for payload, _origin in case.payloads():
        (accepted if validator_factory(raw).is_valid(payload) else rejected).append(payload)
    return accepted, rejected


def _cases() -> list[Case]:
    return [
        Case(
            "flat",
            Flat,
            valid=[{"a": 1, "b": "x", "c": 1.5, "d": True}],
            extra=[
                {},
                {"a": 1, "b": "x", "c": 1.5},
                {"a": 1, "b": "x", "c": 1.5, "d": True, "e": 1},
                {"a": True, "b": "x", "c": 1.5, "d": True},
                {"a": 1.0, "b": "x", "c": 1.5, "d": True},
                {"a": 1, "b": 2, "c": 1.5, "d": True},
                {"a": 1, "b": "x", "c": "nope", "d": True},
                {"a": 1, "b": "x", "c": 1.5, "d": "true"},
                {"a": 1, "b": "x", "c": 1.5, "d": None},
            ],
            note="baseline required scalars",
        ),
        Case(
            "optional_with_default",
            OptionalDefault,
            valid=[{}, {"x": 1}, {"y": 2}, {"x": 1, "y": 2}, {"y": None}],
            extra=[{"x": None}, {"x": "s"}, {"y": "s"}, {"x": 1, "y": 2, "z": 3}],
            note="Optional[X] with default -> anyOf[int,null] + default",
        ),
        Case(
            "optional_no_default",
            OptionalNoDefault,
            valid=[{"x": 1}, {"x": None}],
            extra=[{}, {"x": "s"}],
            note="Optional[X] without default is required-but-nullable",
        ),
        Case(
            "nested_model",
            Outer,
            valid=[{"inner": {"value": 1}}, {"inner": {"value": 1}, "name": "n"}],
            extra=[
                {},
                {"inner": {}},
                {"inner": {"value": "s"}},
                {"inner": None},
                {"inner": {"value": 1, "extra": 2}},
                {"inner": {"value": 1}, "name": None},
            ],
            note="$defs + $ref",
        ),
        Case(
            "literal",
            WithLiteral,
            valid=[{"one": "a", "two": "a", "three": 1}, {"one": "a", "two": "b", "three": None}],
            extra=[
                {"one": "b", "two": "a", "three": 1},
                {"one": "a", "two": "c", "three": 1},
                {"one": "a", "two": "a", "three": 2},
                {"one": "a", "two": "a", "three": True},
                {"one": "a", "two": "a", "three": "x"},
            ],
            note="const + enum",
        ),
        Case(
            "enum",
            WithEnum,
            valid=[{"color": "red"}, {"color": "blue", "color2": "red"}],
            extra=[
                {"color": "green"},
                {"color": None},
                {},
                {"color": "red", "color2": "green"},
            ],
            note="Enum -> $defs + $ref (+ default sibling)",
        ),
        Case(
            "list_of_model",
            ListOfModel,
            valid=[{"items": [{"value": 1}], "matrix": [[1, 2]]}, {"items": [], "matrix": []}],
            extra=[
                {"items": [{}], "matrix": []},
                {"items": [{"value": "s"}], "matrix": []},
                {"items": {}, "matrix": []},
                {"items": [{"value": 1}], "matrix": [["s"]]},
                {"items": [{"value": 1, "extra": 1}], "matrix": []},
            ],
            note="$ref inside items",
        ),
        Case(
            "dict_of_model",
            DictOfModel,
            valid=[{"mapping": {"k": {"value": 1}}, "plain": {"k": 1}}, {"mapping": {}, "plain": {}}],
            extra=[
                {"mapping": {"k": {}}, "plain": {}},
                {"mapping": {"k": {"value": "s"}}, "plain": {}},
                {"mapping": {"k": {"value": 1}}, "plain": {"k": "s"}},
                {"mapping": [], "plain": {}},
            ],
            note="$ref inside additionalProperties",
        ),
        Case(
            "union_plain",
            PlainUnion,
            valid=[{"u": 1, "m": {"kind": "a", "av": 1}}, {"u": "s", "m": {"kind": "b", "bv": "x"}}],
            extra=[
                {"u": 1.5, "m": {"kind": "a", "av": 1}},
                {"u": True, "m": {"kind": "a", "av": 1}},
                {"u": 1, "m": {"kind": "c"}},
                {"u": 1, "m": {"kind": "a"}},
                {"u": 1, "m": {"kind": "a", "av": 1, "bv": "x"}},
            ],
            note="anyOf of $refs",
        ),
        Case(
            "union_discriminated",
            Discriminated,
            valid=[{"du": {"kind": "a", "av": 1}}, {"du": {"kind": "b", "bv": "x"}}],
            extra=[
                {"du": {"kind": "c", "av": 1}},
                {"du": {"av": 1}},
                {"du": {"kind": "a"}},
                {"du": {"kind": "a", "av": 1, "bv": "x"}},
                {"du": None},
            ],
            note="oneOf + discriminator (discriminator is stripped)",
        ),
        Case(
            "constrained",
            Constrained,
            valid=[
                {"s": "ab", "n": 0, "l": [1]},
                {"s": "abcde", "n": 10, "l": [1, 2, 3]},
                {"s": "zzz", "n": 4, "l": []},
            ],
            extra=[
                {"s": "a", "n": 0, "l": [1]},
                {"s": "abcdef", "n": 0, "l": [1]},
                {"s": "ABC", "n": 0, "l": [1]},
                {"s": "ab", "n": -1, "l": [1]},
                {"s": "ab", "n": 11, "l": [1]},
                {"s": "ab", "n": 3, "l": [1]},
                {"s": "ab", "n": 0, "l": []},
                {"s": "ab", "n": 0, "l": [1, 2, 3, 4]},
            ],
            note="min/max/pattern/multipleOf",
        ),
        Case(
            "recursive",
            Recursive,
            valid=[{"name": "n"}, {"name": "n", "child": {"name": "c"}}],
            extra=[{"name": "n", "child": {"name": 1}}, {"child": {"name": "c"}}],
            note="genuinely recursive: the normalizer refuses this on purpose",
        ),
        Case(
            "tuple",
            TupleModel,
            valid=[{"t": [1, "s"], "t2": []}, {"t": [2, "x"], "t2": [1, 2, 3]}],
            extra=[
                {"t": [1], "t2": []},
                {"t": [1, "s", 3], "t2": []},
                {"t": ["s", 1], "t2": []},
                {"t": [1, "s"], "t2": ["s"]},
            ],
            note="prefixItems",
        ),
        Case(
            "scalars",
            Scalars,
            valid=[
                {
                    "dt": "2026-01-02T03:04:05Z",
                    "d": "2026-01-02",
                    "uid": "12345678-1234-5678-1234-567812345678",
                    "dec": "1.25",
                    "raw": "aGk=",
                },
                {
                    "dt": "2026-01-02T03:04:05Z",
                    "d": "2026-01-02",
                    "uid": "12345678-1234-5678-1234-567812345678",
                    "dec": 1.25,
                    "raw": "",
                },
            ],
            extra=[
                {
                    "dt": "not-a-date",
                    "d": "2026-01-02",
                    "uid": "12345678-1234-5678-1234-567812345678",
                    "dec": 1,
                    "raw": "",
                },
                {
                    "dt": "2026-01-02T03:04:05Z",
                    "d": "2026-01-02",
                    "uid": "nope",
                    "dec": 1,
                    "raw": "",
                },
                {
                    "dt": "2026-01-02T03:04:05Z",
                    "d": "2026-01-02",
                    "uid": "12345678-1234-5678-1234-567812345678",
                    "dec": True,
                    "raw": "",
                },
            ],
            note="format is annotation-only; Decimal is anyOf[number,string]",
        ),
        Case(
            "extra_forbid",
            ExtraForbid,
            valid=[{"a": 1}],
            extra=[{"a": 1, "b": 2}, {}],
            note="additionalProperties: false must survive",
        ),
        Case(
            "extra_allow",
            ExtraAllow,
            valid=[{"a": 1}, {"a": 1, "b": "anything"}],
            extra=[{}],
            note="additionalProperties absent (true)",
        ),
        Case(
            "ref_sibling_validation",
            RefSibling,
            valid=[{"inner": {"value": 1}}],
            extra=[{"inner": {"value": 1, "zzz": 2}}],
            note="THE ref-with-siblings case: siblings must conjoin, not overwrite",
        ),
        Case(
            "schema_extra_validation",
            SchemaExtraValidation,
            valid=[{"v": "anything"}],
            extra=[{"v": 1}],
            note="json_schema_extra assertions pass through untouched",
        ),
        Case(
            "set_field",
            SetField,
            valid=[{"s": ["a", "b"], "fs": [1, 2]}],
            extra=[{"s": ["a", "a"], "fs": []}, {"s": "a", "fs": []}],
            note="set -> array + uniqueItems",
        ),
        Case(
            "deep_chain_13",
            DeepChain13,
            valid=[
                {
                    "child": {
                        "child": {
                            "child": {
                                "child": {
                                    "child": {
                                        "child": {
                                            "child": {
                                                "child": {
                                                    "child": {
                                                        "child": {
                                                            "child": {
                                                                "child": {"v": 1}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            ],
            note="non-recursive, but 13 models deep -> depth-limit false positive",
        ),
        Case(
            "aliased_dollar_ref",
            AliasedRef,
            valid=[{"inner": {"value": 1}, "$ref": "s"}],
            extra=[{"inner": {"value": 1}, "$ref": 1}, {"inner": {"value": 1}}],
            note="a property literally NAMED $ref: leaves a dangling $ref",
        ),
        Case(
            "union_enum_none",
            UnionWithEnumAndNone,
            valid=[{"u": {"value": 1}}, {"u": "red"}, {"u": None}],
            extra=[{"u": "green"}, {"u": {}}],
            note="anyOf[$ref, $ref(enum), null]",
        ),
    ]


CASES: list[Case] = _cases()

CASES_BY_NAME = {case.name: case for case in CASES}


def raw_schema(case: Case) -> dict[str, Any]:
    return copy.deepcopy(case.model.model_json_schema())

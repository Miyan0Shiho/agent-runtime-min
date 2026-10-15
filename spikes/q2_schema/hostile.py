"""Hostile inputs aimed at the normalizer.

Every entry is a schema a caller could legitimately construct (or that pydantic
could be made to emit via ``json_schema_extra`` / ``RootModel`` / ``TypeAdapter``)
and which the normalizer either mishandles or does not handle at all.

Each attack declares:

* ``schema``     -- the input;
* ``probes``     -- payloads to differentially test, when the normalizer
  succeeds. Each probe carries the verdict the *raw* schema gives it, computed
  by :mod:`jsonschema_subset` rather than asserted by hand;
* ``expect``     -- what a correct normalizer must do, in one line;
* ``invariant``  -- which property of the module docstring is at stake
  (``lossless`` / ``total`` / ``no-dangling`` / ``subset``).

The runner reports, per attack: the normalizer's outcome, whether the result
still contains keywords that cannot be expressed in the function-calling subset,
whether it contains a ``$ref`` that no longer resolves, and every payload where
raw and normalized disagree.

Run directly::

    cd spikes && ../.venv/bin/python q2_schema/hostile.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import _bootstrap  # noqa: F401
from arm.tools.schema import SchemaNormalizationError, normalize_schema, to_wire
from jsonschema_subset import SubsetValidator, UnresolvableRef

#: Keywords that a function-calling API does NOT accept, per vendor documentation
#: (see FINDINGS.md Q5). Anything here surviving into the output is a wire risk.
#:
#: NOTE: ``$ref``/``$defs`` are deliberately ABSENT. OpenAI documents both as
#: supported -- its own recursive-schema example ships ``$ref``/``$defs`` -- so
#: treating them as wire risks is not supported by the docs. ``anyOf`` is also
#: supported. The keywords below are the ones the docs actually reject.
BANNED_ON_WIRE = ("allOf", "not", "if", "then", "else", "dependentSchemas",
                  "dependentRequired", "patternProperties")

#: Keywords that are unsupported by SOME providers but fine for others. Reported
#: separately so the two risks are not conflated.
PROVIDER_SPECIFIC = ("oneOf", "$ref", "$defs", "prefixItems", "uniqueItems", "const")


@dataclass
class Attack:
    name: str
    schema: Any
    probes: list[Any] = field(default_factory=list)
    expect: str = ""
    invariant: str = "lossless"
    note: str = ""


@dataclass
class AttackResult:
    attack: Attack
    outcome: str = ""
    normalizer_error: str | None = None
    output: Any = None
    banned_present: list[str] = field(default_factory=list)
    provider_specific_present: list[str] = field(default_factory=list)
    dangling_refs: list[str] = field(default_factory=list)
    divergences: list[str] = field(default_factory=list)
    probe_verdicts: list[tuple[Any, bool, bool]] = field(default_factory=list)

    @property
    def wrong(self) -> bool:
        return bool(self.divergences or self.dangling_refs or self.banned_present)


def _collect_refs(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            out.append(ref)
        for value in node.values():
            _collect_refs(value, out)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_refs(item, out)


def _resolve(root: Any, ref: str) -> Any:
    """Resolve a local ref the way :mod:`jsonschema_subset` does."""
    if not ref.startswith("#"):
        raise UnresolvableRef(ref)
    pointer = ref[1:]
    if pointer in ("", "/"):
        return root
    node = root
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, (list, tuple)):
            node = node[int(token)]
        else:
            raise UnresolvableRef(ref)
    return node


def run_attack(attack: Attack) -> AttackResult:
    result = AttackResult(attack=attack)
    try:
        normalized = normalize_schema(attack.schema)
        result.output = to_wire(normalized)
        result.outcome = "normalized"
    except SchemaNormalizationError as exc:
        result.outcome = "refused (SchemaNormalizationError)"
        result.normalizer_error = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001 - an unplanned crash is the finding
        result.outcome = f"CRASHED ({type(exc).__name__})"
        result.normalizer_error = str(exc)
        return result

    blob = json.dumps(result.output)
    result.banned_present = [k for k in BANNED_ON_WIRE if f'"{k}"' in blob]
    result.provider_specific_present = [k for k in PROVIDER_SPECIFIC if f'"{k}"' in blob]

    refs: list[str] = []
    _collect_refs(result.output, refs)
    for ref in refs:
        try:
            _resolve(attack.schema, ref)
        except (UnresolvableRef, Exception):  # noqa: BLE001
            result.dangling_refs.append(ref)

    raw_validator = SubsetValidator(attack.schema)
    norm_validator = SubsetValidator(result.output)
    for payload in attack.probes:
        raw_ok = raw_validator.is_valid(payload)
        norm_ok = norm_validator.is_valid(payload)
        result.probe_verdicts.append((payload, raw_ok, norm_ok))
        if raw_ok != norm_ok:
            direction = "WIDENED" if norm_ok else "NARROWED"
            result.divergences.append(
                f"{direction}: {json.dumps(payload, default=str)} "
                f"(raw={raw_ok} norm={norm_ok})"
            )
    return result


# ---------------------------------------------------------------------------
# The attacks
# ---------------------------------------------------------------------------

_INNER = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
}


def attacks() -> list[Attack]:
    out: list[Attack] = []

    # -- 1. $ref hygiene ---------------------------------------------------
    out.append(
        Attack(
            name="ref_to_missing_def",
            schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Missing"}},
                "$defs": {},
            },
            expect="pass through unchanged (never silently drop a constraint)",
            invariant="lossless",
            note="unresolvable $ref is preserved, which is correct for a validator "
            "but ships a $ref no model server can resolve",
        )
    )

    out.append(
        Attack(
            name="ref_to_boolean_false_def",
            schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Nope"}},
                "$defs": {"Nope": False},
            },
            expect="preserve (or refuse loudly); False rejects every value",
            invariant="total",
        )
    )

    out.append(
        Attack(
            name="ref_to_boolean_true_def",
            schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Yes"}},
                "$defs": {"Yes": True},
            },
            expect="preserve (or refuse loudly); True accepts every value",
            invariant="total",
        )
    )

    out.append(
        Attack(
            name="ref_chain_a_b_c",
            schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/A"}},
                "$defs": {
                    "A": {"$ref": "#/$defs/B"},
                    "B": {"$ref": "#/$defs/C"},
                    "C": {"type": "integer"},
                },
            },
            probes=[{"x": 1}, {"x": "s"}, {"x": None}, {}],
            expect="inline transitively to {type: integer}",
            invariant="lossless",
        )
    )

    out.append(
        Attack(
            name="mutual_recursion_a_b_a",
            schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/A"}},
                "$defs": {
                    "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
                    "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
                },
            },
            expect="refuse loudly (cannot be inlined)",
            invariant="total",
        )
    )

    out.append(
        Attack(
            name="ref_with_escaped_pointer_token",
            schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/A~1B"}},
                "$defs": {"A/B": {"type": "integer"}},
            },
            probes=[{"x": 1}, {"x": "s"}],
            expect="decode ~1 to / and inline the target",
            invariant="lossless",
            note="RFC 6901 escaping; a def name containing '/' is legal",
        )
    )

    out.append(
        Attack(
            name="ref_with_escaped_tilde_token",
            schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/A~0B"}},
                "$defs": {"A~B": {"type": "integer"}},
            },
            probes=[{"x": 1}, {"x": "s"}],
            expect="decode ~0 to ~ and inline the target",
            invariant="lossless",
        )
    )

    # -- 2. $ref with siblings --------------------------------------------
    out.append(
        Attack(
            name="ref_sibling_adds_property_key",
            schema={
                "type": "object",
                "properties": {
                    "inner": {
                        "$ref": "#/$defs/Inner",
                        "properties": {"sneaky": {"type": "integer"}},
                    }
                },
                "required": ["inner"],
                "$defs": {"Inner": _INNER},
            },
            probes=[
                {"inner": {"value": 1}},
                {"inner": {"value": 1, "sneaky": 2}},
                {"inner": {"value": "not-an-integer"}},
                {"inner": {"value": "not-an-integer", "sneaky": 2}},
                {"inner": {"sneaky": 2}},
                {"inner": {}},
            ],
            expect="conjoin: object with value:int AND sneaky:int",
            invariant="lossless",
            note="THE widening: the sibling 'properties' key overwrites the target's",
        )
    )

    out.append(
        Attack(
            name="ref_sibling_required_collision",
            schema={
                "type": "object",
                "properties": {"inner": {"$ref": "#/$defs/Inner", "required": ["zzz"]}},
                "required": ["inner"],
                "$defs": {"Inner": _INNER},
            },
            probes=[
                {"inner": {"value": 1, "zzz": 2}},
                {"inner": {"zzz": 2}},
                {"inner": {"value": 1}},
            ],
            expect="conjoin: required == [value, zzz]",
            invariant="lossless",
            note="THE narrowing sibling: 'required' overwrites the target's",
        )
    )

    out.append(
        Attack(
            name="ref_sibling_additional_properties",
            schema={
                "type": "object",
                "properties": {"inner": {"$ref": "#/$defs/Inner", "additionalProperties": False}},
                "required": ["inner"],
                "$defs": {"Inner": _INNER},
            },
            probes=[{"inner": {"value": 1}}, {"inner": {"value": 1, "extra": 2}}],
            expect="conjoin: no extra properties",
            invariant="lossless",
        )
    )

    out.append(
        Attack(
            name="ref_sibling_description_only",
            schema={
                "type": "object",
                "properties": {
                    "inner": {"$ref": "#/$defs/Inner", "description": "the inner thing"}
                },
                "$defs": {"Inner": dict(_INNER, description="Inner model")},
            },
            expect="keep both descriptions, or at least do not lose the field's",
            invariant="lossless",
            note="exactly what pydantic emits for Annotated[Model, Field(description=...)]",
        )
    )

    # -- 3. single-branch combinator with siblings ------------------------
    out.append(
        Attack(
            name="single_branch_anyof_with_sibling",
            schema={
                "type": "object",
                "properties": {
                    "s": {
                        "anyOf": [{"type": "string", "pattern": "^a"}],
                        "pattern": "^b",
                    }
                },
            },
            probes=[
                {"s": "abc"},
                {"s": "bcd"},
                {"s": "xyz"},
                {"s": 1},
            ],
            expect="conjoin the branch with the sibling (both patterns apply)",
            invariant="lossless",
            note="THE second widening: the sibling 'pattern' overwrites the branch's",
        )
    )

    out.append(
        Attack(
            name="single_branch_oneof_with_sibling_type",
            schema={
                "type": "object",
                "properties": {"x": {"oneOf": [{"type": "integer", "minimum": 5}], "maximum": 3}},
            },
            probes=[{"x": 1}, {"x": 4}, {"x": 7}],
            expect="conjoin: integer, >= 5 and <= 3 (nothing satisfies it)",
            invariant="lossless",
        )
    )

    # -- 4. allOf ----------------------------------------------------------
    out.append(
        Attack(
            name="allof_two_objects",
            schema={
                "allOf": [
                    {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                    {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
                ]
            },
            probes=[
                {"a": 1, "b": "x"},
                {"a": 1},
                {"b": "x"},
                {"a": "s", "b": "x"},
            ],
            expect="merge into one object schema, or refuse loudly",
            invariant="subset",
            note="allOf is in _SCHEMA_LIST_KEYWORDS but _collapse only handles "
            "anyOf/oneOf, so allOf reaches the wire untouched",
        )
    )

    out.append(
        Attack(
            name="allof_with_ref_branches",
            schema={
                "type": "object",
                "properties": {
                    "m": {"allOf": [{"$ref": "#/$defs/Inner"}]},
                },
                "$defs": {"Inner": _INNER},
            },
            probes=[{"m": {"value": 1}}, {"m": {}}, {"m": {"value": "s"}}],
            expect="a single-branch allOf is pure indirection; inline it",
            invariant="subset",
        )
    )

    # -- 5. $defs lifetime -------------------------------------------------
    out.append(
        Attack(
            name="unreferenced_defs_at_root",
            schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "$defs": {"Unused": {"type": "string"}},
            },
            probes=[{"a": 1}, {"a": "s"}, {}],
            expect="drop the unused $defs (they are not part of this schema)",
            invariant="subset",
        )
    )

    out.append(
        Attack(
            name="dangling_defs_from_property_named_dollar_ref",
            schema={
                "type": "object",
                "properties": {
                    "inner": {"$ref": "#/$defs/Inner"},
                    "$ref": {"type": "string"},
                },
                "$defs": {"Inner": _INNER},
            },
            probes=[
                {"inner": {"value": 1}, "$ref": "s"},
                {"inner": {"value": 1}},
                {"inner": {"value": 1}, "$ref": 1},
            ],
            expect="drop $defs once the real reference is inlined; a property "
            "*named* $ref is not a reference",
            invariant="no-dangling",
            note="pydantic emits exactly this for Field(alias='$ref')",
        )
    )

    out.append(
        Attack(
            name="defs_not_a_mapping",
            schema={"type": "object", "properties": {"a": {"type": "integer"}}, "$defs": ["x"]},
            expect="refuse loudly rather than crash",
            invariant="total",
        )
    )

    # -- 6. keyword coverage ----------------------------------------------
    out.append(
        Attack(
            name="additional_properties_false",
            schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            probes=[{"a": 1}, {"a": 1, "b": 2}, {}],
            expect="preserve additionalProperties: false verbatim",
            invariant="lossless",
        )
    )

    out.append(
        Attack(
            name="property_named_type",
            schema={
                "type": "object",
                "properties": {"type": {"type": "string"}, "properties": {"type": "integer"}},
                "required": ["type"],
            },
            probes=[{"type": "s"}, {"type": "s", "properties": 1}, {"type": 1}],
            expect="property names are opaque; only schema positions are rewritten",
            invariant="lossless",
        )
    )

    out.append(
        Attack(
            name="deep_anyof_in_items_in_properties",
            schema={
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cell": {
                                    "anyOf": [
                                        {"$ref": "#/$defs/Inner"},
                                        {"type": "null"},
                                    ]
                                }
                            },
                            "required": ["cell"],
                        },
                    }
                },
                "$defs": {"Inner": _INNER},
            },
            probes=[
                {"rows": [{"cell": {"value": 1}}]},
                {"rows": [{"cell": None}]},
                {"rows": [{"cell": {}}]},
                {"rows": [{"cell": {"value": "s"}}]},
                {"rows": [{}]},
            ],
            expect="inline the nested $ref, keep anyOf[object, null]",
            invariant="lossless",
        )
    )

    out.append(
        Attack(
            name="ref_inside_unlisted_schema_keyword",
            schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "$defs": {"Inner": _INNER},
                # dependentSchemas / unevaluatedProperties / contentSchema are
                # schema positions that _SCHEMA_MAP_KEYWORDS does not list.
                "dependentSchemas": {"a": {"properties": {"b": {"$ref": "#/$defs/Inner"}}}},
            },
            expect="refuse loudly: dependentSchemas cannot be expressed in the subset",
            invariant="no-dangling",
            note="a schema position the normalizer does not descend into, while "
            "_contains_ref still sees the $ref and keeps $defs",
        )
    )

    out.append(
        Attack(
            name="recursive_ref_through_allof",
            schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/A"}},
                "$defs": {
                    "A": {"allOf": [{"$ref": "#/$defs/A"}]},
                },
            },
            expect="refuse loudly (recursive)",
            invariant="total",
        )
    )

    out.append(
        Attack(
            name="root_level_ref_non_recursive",
            schema={"$defs": {"Inner": _INNER}, "$ref": "#/$defs/Inner"},
            probes=[{"value": 1}, {}, {"value": "s"}],
            expect="inline to the object schema (this is what RootModel emits)",
            invariant="lossless",
        )
    )

    out.append(
        Attack(
            name="boolean_root_schema",
            schema=True,
            probes=[1, "s", None, {}],
            expect="preserve or refuse; True accepts everything",
            invariant="total",
        )
    )

    out.append(
        Attack(
            name="empty_root_schema",
            schema={},
            probes=[1, "s", None, {}],
            expect="preserve; {} accepts everything",
            invariant="total",
        )
    )

    out.append(
        Attack(
            name="not_keyword",
            schema={"type": "object", "properties": {"x": {"not": {"type": "string"}}}},
            probes=[{"x": 1}, {"x": "s"}],
            expect="refuse loudly: 'not' cannot be expressed in the subset, and "
            "dropping it would widen",
            invariant="lossless",
        )
    )

    out.append(
        Attack(
            name="if_then_else",
            schema={
                "type": "object",
                "properties": {"kind": {"type": "string"}},
                "if": {"properties": {"kind": {"const": "a"}}, "required": ["kind"]},
                "then": {"required": ["av"]},
                "else": {"required": ["bv"]},
            },
            probes=[
                {"kind": "a", "av": 1},
                {"kind": "a"},
                {"kind": "b", "bv": 1},
                {"kind": "b"},
            ],
            expect="refuse loudly: if/then/else cannot be expressed in the subset",
            invariant="lossless",
        )
    )

    out.append(
        Attack(
            name="defs_only_referenced_from_dropped_keyword",
            schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "$defs": {"Inner": _INNER},
                # title is stripped; if a $ref only lived under a stripped key
                # the def would become unreferenced.
                "title": {"$ref": "#/$defs/Inner"},
            },
            expect="strip the presentation keyword and then the unused $defs",
            invariant="no-dangling",
        )
    )

    return out


def main() -> None:
    results = [run_attack(a) for a in attacks()]
    print("=" * 78)
    print("Q2 hostile inputs")
    print("=" * 78)
    for result in results:
        attack = result.attack
        print(f"\n### {attack.name}  [{attack.invariant}]")
        print(f"  expect    : {attack.expect}")
        if attack.note:
            print(f"  note      : {attack.note}")
        print(f"  outcome   : {result.outcome}")
        if result.normalizer_error:
            print(f"  error     : {result.normalizer_error}")
        if result.output is not None:
            blob = json.dumps(result.output, sort_keys=True)
            print(f"  output    : {blob[:300]}{'...' if len(blob) > 300 else ''}")
        if result.banned_present:
            print(f"  ON WIRE   : {result.banned_present}  <- rejected by OpenAI strict mode")
        if result.provider_specific_present:
            print(f"  PROVIDER  : {result.provider_specific_present}  <- provider-specific")
        if result.dangling_refs:
            print(f"  DANGLING  : {result.dangling_refs}")
        for verdict in result.divergences:
            print(f"  DIVERGENCE: {verdict}")
        if result.probe_verdicts and not result.divergences:
            shown = ", ".join(
                f"{json.dumps(p, default=str)}->{'ok' if r == n else '??'}"
                for p, r, n in result.probe_verdicts[:6]
            )
            print(f"  probes    : {shown}")

    print("\n" + "=" * 78)
    crashes = [r for r in results if r.outcome.startswith("CRASHED")]
    refused = [r for r in results if r.outcome.startswith("refused")]
    widened = [r for r in results if r.divergences]
    dangling = [r for r in results if r.dangling_refs]
    onwire = [r for r in results if r.banned_present]
    print(f"attacks={len(results)}")
    print(f"  crashed            : {len(crashes)} {[r.attack.name for r in crashes]}")
    print(f"  refused            : {len(refused)} {[r.attack.name for r in refused]}")
    print(f"  payload divergence : {len(widened)} {[r.attack.name for r in widened]}")
    print(f"  dangling $ref      : {len(dangling)} {[r.attack.name for r in dangling]}")
    print(f"  banned kw on wire  : {len(onwire)} {[r.attack.name for r in onwire]}")
    print("=" * 78)


if __name__ == "__main__":
    main()

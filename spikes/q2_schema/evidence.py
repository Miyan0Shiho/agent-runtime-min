"""Print the raw observations quoted in FINDINGS.md.

Run from ``spikes/``::

    cd spikes && ../.venv/bin/python q2_schema/evidence.py

This is not a test; it is the transcript generator, so FINDINGS.md quotes
observed output rather than a paraphrase.

The sections are numbered to match FINDINGS.md.
"""

from __future__ import annotations

import json
import sys

import _bootstrap  # noqa: F401
from arm.tools.schema import SchemaNormalizationError, normalize_schema, to_wire
from catalogue import CASES
from differential import run_all
from hostile import attacks, run_attack
from jsonschema_subset import SubsetValidator
from schema_lint import lint_schema, schema_keywords


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def j(value, limit: int = 400) -> str:
    text = json.dumps(value, indent=1, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 20] + "\n  ... (trimmed)"


# ---------------------------------------------------------------------------
def q1_catalogue() -> None:
    hdr("Q1  What pydantic actually emits (pydantic 2.14.1)")
    union: set[str] = set()
    for case in CASES:
        raw = case.model.model_json_schema()
        keys = sorted(schema_keywords(raw))
        union |= set(keys)
        print(f"\n{case.name}")
        print(f"  keywords: {keys}")
    print(f"\nUNION over all {len(CASES)} models ({len(union)} keywords):")
    print(f"  {sorted(union)}")
    print("\nEmitted but NOT in OpenAI's documented supported set:")
    supported = {
        "type", "properties", "required", "additionalProperties", "enum", "anyOf",
        "pattern", "format", "multipleOf", "maximum", "exclusiveMaximum", "minimum",
        "exclusiveMinimum", "minItems", "maxItems", "description", "items", "$ref", "$defs",
    }
    print(f"  {sorted(union - supported)}")


# ---------------------------------------------------------------------------
def q2_differential() -> None:
    hdr("Q2  Does normalization change the accepted payload set?")
    results = run_all()
    total = sum(r.payload_count for r in results)
    for r in results:
        if r.divergences:
            print(f"\n[{r.case}] {len(r.divergences)} divergences")
            for d in r.divergences[:4]:
                print(d.render())
    print(f"\ncases={len(results)} payloads={total} "
          f"divergences={sum(len(r.divergences) for r in results)} "
          f"refused={sum(1 for r in results if r.normalizer_error)}")
    print("\nPer-case payload counts:")
    for r in results:
        print(f"  {r.case:24s} {r.payload_count:5d} payloads  "
              f"{'REFUSED' if r.normalizer_error else 'ok'}")


# ---------------------------------------------------------------------------
def q3_hostile() -> None:
    hdr("Q3  Hostile inputs")
    results = [run_attack(a) for a in attacks()]
    for r in results:
        print(f"\n### {r.attack.name}")
        print(f"  outcome : {r.outcome}")
        if r.normalizer_error:
            print(f"  error   : {r.normalizer_error[:150]}")
        if r.output is not None:
            print(f"  output  : {json.dumps(r.output, sort_keys=True)[:220]}")
        if r.banned_present:
            print(f"  ON WIRE : {r.banned_present}")
        if r.dangling_refs:
            print(f"  DANGLING: {r.dangling_refs}")
        for d in r.divergences:
            print(f"  DIVERGE : {d}")
    print(f"\nattacks={len(results)}")
    print(f"  crashed           : {[r.attack.name for r in results if r.outcome.startswith('CRASHED')]}")
    print(f"  refused           : {[r.attack.name for r in results if r.outcome.startswith('refused')]}")
    print(f"  payload divergence: {[r.attack.name for r in results if r.divergences]}")
    print(f"  dangling $ref     : {[r.attack.name for r in results if r.dangling_refs]}")


# ---------------------------------------------------------------------------
def q3_live_bugs() -> None:
    hdr("Q3b  The three live defects, reproduced")

    print("\n--- BUG 10: `properties` merge is shallow ---")
    schema = {
        "type": "object",
        "properties": {"m": {"$ref": "#/$defs/T", "properties": {"x": {"type": "string"}}}},
        "$defs": {"T": {"type": "object", "properties": {"x": {"type": "integer"}}}},
    }
    print(f"RAW : {json.dumps(schema['properties']['m'])}")
    out = to_wire(normalize_schema(schema))
    print(f"NORM: {json.dumps(out['properties']['m'])}")
    for payload in ({"m": {"x": 1}}, {"m": {"x": "s"}}):
        raw_ok = SubsetValidator(schema).is_valid(payload)
        norm_ok = SubsetValidator(out).is_valid(payload)
        print(f"  raw={raw_ok!s:5s} norm={norm_ok!s:5s} "
              f"{'<-- WIDENED' if raw_ok != norm_ok else ''}  {json.dumps(payload)}")

    print("\n--- BUG 11: `definitions` dropped, `#/definitions/` ref survives ---")
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/Node"}},
        "definitions": {
            "Node": {"type": "object", "properties": {"child": {"$ref": "#/definitions/Node"}}}
        },
    }
    out = to_wire(normalize_schema(schema))
    print(f"NORM: {json.dumps(out)}")
    print(f"  refs in output : {[v for v in json.dumps(out).split(chr(34)) if v.startswith('#/')]}")
    print(f"  keys in output : {sorted(out)}")
    print("  => the ref says '#/definitions/Node' but the definition was re-emitted under '$defs'")

    print("\n--- BUG 12: `additionalProperties: false` sibling is merged ---")
    schema = {
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner", "additionalProperties": False}},
        "required": ["inner"],
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
    }
    out = to_wire(normalize_schema(schema))
    print(f"RAW  sibling: {json.dumps(schema['properties']['inner'])}")
    print(f"NORM sibling: {json.dumps(out['properties']['inner'])}")
    for payload in ({"inner": {"value": 1}}, {"inner": {}}):
        raw_ok = SubsetValidator(schema).is_valid(payload)
        norm_ok = SubsetValidator(out).is_valid(payload)
        print(f"  raw={raw_ok!s:5s} norm={norm_ok!s:5s} "
              f"{'<-- WIDENED' if raw_ok != norm_ok else ''}  {json.dumps(payload)}")


# ---------------------------------------------------------------------------
def q4_defs_liveness() -> None:
    hdr("Q4  `$defs` removal logic")
    probes = {
        "unreferenced defs": {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "$defs": {"Unused": {"type": "string"}},
        },
        "property literally named $ref": {
            "type": "object",
            "properties": {"inner": {"$ref": "#/$defs/I"}, "$ref": {"type": "string"}},
            "$defs": {"I": {"type": "object", "properties": {"v": {"type": "integer"}}}},
        },
        "ref inside an unsupported keyword": {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "$defs": {"Inner": {"type": "object"}},
            "dependentSchemas": {"a": {"properties": {"b": {"$ref": "#/$defs/Inner"}}}},
        },
        "unresolvable ref": {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Missing"}},
            "$defs": {},
        },
        "recursive def, referenced": {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/Node"}},
            "$defs": {"Node": {"type": "object", "properties": {"c": {"$ref": "#/$defs/Node"}}}},
        },
    }
    for name, schema in probes.items():
        try:
            out = to_wire(normalize_schema(schema))
            print(f"\n{name}:")
            print(f"  -> {json.dumps(out)[:260]}")
            print(f"  $defs present: {'$defs' in out}")
        except SchemaNormalizationError as exc:
            print(f"\n{name}:\n  REFUSED: {str(exc)[:140]}")


# ---------------------------------------------------------------------------
def q5_wire_shape() -> None:
    hdr("Q5  Wire risk: which emitted keywords are documented as unsupported")
    supported = {
        "type", "properties", "required", "additionalProperties", "enum", "anyOf",
        "pattern", "format", "multipleOf", "maximum", "exclusiveMaximum", "minimum",
        "exclusiveMinimum", "minItems", "maxItems", "description", "items", "$ref", "$defs",
    }
    explicitly_unsupported = {
        "allOf", "not", "dependentRequired", "dependentSchemas", "if", "then", "else",
        "patternProperties", "oneOf", "const", "prefixItems", "uniqueItems",
        "minLength", "maxLength", "minProperties",
    }
    emitted = {k for case in CASES for k in schema_keywords(case.model.model_json_schema())}
    print("Emitted by pydantic AND documented as supported:")
    print(f"  {sorted(emitted & supported)}")
    print("\nEmitted by pydantic and NOT in the documented supported set:")
    print(f"  {sorted(emitted - supported)}")
    print("\nDocumented as explicitly NOT supported (would error under strict: true):")
    print(f"  {sorted(explicitly_unsupported)}")
    print("\n=> the normalizer strips `title`/`default` (harmless annotations) and inlines")
    print("   `$ref`/`$defs` (documented as supported), while `allOf`/`not`/`if`/`then`/`else`")
    print("   (documented as REJECTED) reach the wire.")
    print("\nNormalized output for the builtin tool set -- which keywords remain:")
    from arm.tools.builtin import builtin_tools

    for tool in builtin_tools():
        keys = sorted(schema_keywords(dict(tool.definition.parameters)))
        print(f"  {tool.name:8s} {keys}")


# ---------------------------------------------------------------------------
def main() -> None:
    which = sys.argv[1:] or ["1", "2", "3", "3b", "4", "5"]
    if "1" in which:
        q1_catalogue()
    if "2" in which:
        q2_differential()
    if "3" in which:
        q3_hostile()
    if "3b" in which:
        q3_live_bugs()
    if "4" in which:
        q4_defs_liveness()
    if "5" in which:
        q5_wire_shape()


if __name__ == "__main__":
    main()

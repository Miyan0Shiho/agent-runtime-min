"""PASSING tests: what the normalizer does correctly, and what it must never do.

These are the guard rails around the bug demonstrations in
``test_findings_bugs.py``. They fall into four groups:

1. **Catalogue** -- every pydantic construct, with the exact keyword set pydantic
   emits, frozen so a pydantic upgrade that changes emission is noticed.
2. **Differential** -- raw vs normalized over ~1600 payloads per full run. The
   claim under test ("normalization must not change the accepted set") holds for
   the shapes pydantic actually emits; this pins that down.
3. **Properties** -- idempotence, well-formedness, no new keywords invented.
4. **Cross-check** -- if the real ``jsonschema`` library is importable it is used
   as an oracle for :mod:`jsonschema_subset`, so a bug in the mini-validator
   cannot quietly manufacture or hide a finding.
"""

from __future__ import annotations

import json
import sys

import pytest

import _bootstrap  # noqa: F401
from arm.tools.schema import (
    PRESENTATION_KEYWORDS,
    SchemaNormalizationError,
    normalize_schema,
    to_wire,
)
from catalogue import CASES, CASES_BY_NAME, partition
from differential import run_all, run_case
from jsonschema_subset import SubsetValidator
from schema_lint import lint_schema, schema_keywords

# ---------------------------------------------------------------------------
# 1. Catalogue: what pydantic actually emits
# ---------------------------------------------------------------------------

#: Frozen observation of ``Model.model_json_schema()`` under pydantic 2.14.1.
#: Only schema keywords are listed (property names are filtered out by the
#: ``_SCHEMA_KEYWORD_ONLY`` set below).
EXPECTED_KEYWORDS: dict[str, list[str]] = {
    "flat": ["description", "properties", "required", "title", "type"],
    "optional_with_default": ["anyOf", "default", "description", "properties", "title", "type"],
    "optional_no_default": ["anyOf", "description", "properties", "required", "title", "type"],
    "nested_model": ["$defs", "$ref", "default", "description", "properties", "required", "title", "type"],
    "literal": ["const", "description", "enum", "properties", "required", "title", "type"],
    "enum": ["$defs", "$ref", "default", "description", "enum", "properties", "required", "title", "type"],
    "list_of_model": ["$defs", "$ref", "items", "properties", "required", "title", "type"],
    "dict_of_model": ["$defs", "$ref", "additionalProperties", "properties", "required", "title", "type"],
    "union_plain": ["$defs", "$ref", "anyOf", "const", "description", "properties", "required", "title", "type"],
    "union_discriminated": ["$defs", "$ref", "const", "description", "discriminator", "oneOf", "properties", "required", "title", "type"],
    "constrained": ["description", "items", "maxItems", "maxLength", "maximum", "minItems", "minLength", "minimum", "multipleOf", "pattern", "properties", "required", "title", "type"],
    "recursive": ["$defs", "$ref", "anyOf", "default", "description", "properties", "required", "title", "type"],
    "tuple": ["description", "items", "maxItems", "minItems", "prefixItems", "properties", "required", "title", "type"],
    "scalars": ["anyOf", "description", "format", "properties", "required", "title", "type"],
    "extra_forbid": ["additionalProperties", "properties", "required", "title", "type"],
    "extra_allow": ["additionalProperties", "properties", "required", "title", "type"],
    "ref_sibling_validation": ["$defs", "$ref", "description", "properties", "required", "title", "type"],
    "schema_extra_validation": ["description", "minProperties", "properties", "required", "title", "type"],
    "set_field": ["items", "properties", "required", "title", "type", "uniqueItems"],
    "deep_chain_13": ["$defs", "$ref", "description", "properties", "required", "title", "type"],
    "aliased_dollar_ref": ["$defs", "$ref", "description", "properties", "required", "title", "type"],
    "union_enum_none": ["$defs", "$ref", "anyOf", "enum", "properties", "required", "title", "type"],
}

#: Every keyword pydantic 2.14.1 emitted anywhere in this catalogue.
OBSERVED_UNION = frozenset(
    {
        "$defs", "$ref", "additionalProperties", "anyOf", "const", "default",
        "description", "discriminator", "enum", "format", "items",
        "maxItems", "maxLength", "maximum", "minItems", "minLength",
        "minProperties", "minimum", "multipleOf", "oneOf", "pattern",
        "prefixItems", "properties", "required", "title", "type", "uniqueItems",
    }
)

#: Keywords pydantic is commonly assumed to emit but does NOT, for these models.
NOT_EMITTED = frozenset({"allOf", "not", "if", "then", "else", "patternProperties", "propertyNames", "contains", "examples", "$schema", "$id"})


def test_catalogue_matches_frozen_pydantic_emission():
    """The catalogue records exactly what pydantic 2.14.1 emits, keyword for keyword.

    If this fails after a pydantic upgrade, the catalogue in FINDINGS.md needs
    re-checking -- the point is to notice emission changes rather than assume
    they did not happen.
    """
    mismatches = {}
    for case in CASES:
        observed = sorted(schema_keywords(case.model.model_json_schema()))
        expected = EXPECTED_KEYWORDS[case.name]
        if observed != sorted(expected):
            mismatches[case.name] = {"expected": sorted(expected), "observed": observed}
    assert not mismatches, f"pydantic emission changed: {json.dumps(mismatches, indent=2)}"


def test_the_observed_keyword_union_is_exactly_what_was_recorded():
    """No pydantic keyword appears in this catalogue that FINDINGS.md does not list."""
    union = {k for case in CASES for k in schema_keywords(case.model.model_json_schema())}
    assert union == set(OBSERVED_UNION), (
        f"unexpected: {sorted(union - set(OBSERVED_UNION))}; "
        f"missing: {sorted(set(OBSERVED_UNION) - union)}"
    )


def test_pydantic_does_not_emit_allof_for_these_models():
    """``allOf`` is not emitted by any model here -- so BUG 7 needs a hand-built schema.

    This is recorded explicitly because it bounds the severity of BUG 7: for the
    constructs in this catalogue, the normalizer's failure to handle ``allOf`` is
    not reachable from ``model_json_schema()``. It *is* reachable from
    ``tool(parameters={...})``, which accepts a raw mapping.
    """
    offenders = [
        case.name
        for case in CASES
        if "allOf" in schema_keywords(case.model.model_json_schema())
    ]
    assert not offenders, f"allOf unexpectedly emitted by: {offenders}"


def test_discriminator_is_stripped_but_oneof_survives():
    """``discriminator`` is presentation for a strict-subset API; ``oneOf`` is not.

    Recorded as an observation: stripping ``discriminator`` is safe (it is a
    validation hint, not a constraint), but it leaves ``oneOf`` on the wire,
    which OpenAI-compatible function calling does not accept.
    """
    case = CASES_BY_NAME["union_discriminated"]
    normalized = to_wire(normalize_schema(case.model.model_json_schema()))
    assert "discriminator" not in schema_keywords(normalized)
    assert "oneOf" in schema_keywords(normalized), "oneOf survives and is the wire risk"


# ---------------------------------------------------------------------------
# 2. Differential: normalization must not change the accepted payload set
# ---------------------------------------------------------------------------


def test_normalization_is_lossless_for_every_normalizable_case():
    """The headline claim, over every payload in the corpus.

    Cases where the normalizer *refuses* are excluded here and asserted
    separately in :func:`test_normalizer_refusals_are_exactly_the_known_two`, so
    a refusal can never be mistaken for a pass.
    """
    failures = []
    checked = 0
    for result in run_all():
        if result.normalizer_error:
            continue
        checked += result.payload_count
        if result.divergences:
            failures.append(
                {
                    "case": result.case,
                    "divergences": [
                        {
                            "kind": d.kind,
                            "payload": json.dumps(d.payload, default=str),
                            "raw_accepts": d.raw_accepts,
                            "norm_accepts": d.norm_accepts,
                        }
                        for d in result.divergences
                    ],
                }
            )
    assert checked > 1400, f"corpus unexpectedly small: {checked} payloads"
    assert not failures, f"normalization changed the accepted set: {json.dumps(failures, indent=2)}"


def test_no_catalogue_case_is_refused():
    """Every catalogue case now normalizes -- including the two that used to be refused.

    History: ``recursive`` was refused by design and ``deep_chain_13`` was
    refused by a depth-limit false positive (BUG 4). The rewrite handles both:
    a cyclic definition is kept as ``$ref``/``$defs`` rather than inlined, and
    the depth ceiling was raised well above realistic nesting.

    Pinned so that a change which starts refusing schemas is visible.
    """
    refused = {r.case: r.normalizer_error for r in run_all() if r.normalizer_error}
    assert not refused, f"cases unexpectedly refused: {json.dumps(refused, indent=2)}"


def test_recursive_schema_keeps_a_resolvable_reference():
    """A genuinely recursive model must produce a schema whose ``$ref`` still resolves.

    Refusing outright (the old behaviour) made a recursive tool unregisterable;
    inlining is impossible; the correct answer is to keep the reference and its
    definition together.
    """
    from catalogue import Recursive

    normalized = to_wire(normalize_schema(Recursive.model_json_schema()))
    refs = [value for value in _walk_refs(normalized)]
    assert refs, "the recursive reference must survive"
    for ref in refs:
        _resolve_pointer(normalized, ref)


def _walk_refs(node):
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield ref
        for value in node.values():
            yield from _walk_refs(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_refs(item)


def _resolve_pointer(root, ref):
    if not ref.startswith("#"):
        return root
    node = root
    for raw in ref[1:].split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        else:
            raise AssertionError(f"dangling $ref {ref!r}: {token!r} not found")
    return node


def test_every_normalizable_output_is_well_formed():
    """A normalized schema must still *be* a JSON Schema.

    This is the check that catches BUG 2 without any payload reasoning: the
    output is structurally invalid, independent of what it accepts.
    """
    problems = {}
    for case in CASES:
        raw = case.model.model_json_schema()
        try:
            normalized = to_wire(normalize_schema(raw))
        except SchemaNormalizationError:
            continue
        report = lint_schema(normalized)
        if not report.ok:
            problems[case.name] = report.render()
    assert not problems, f"malformed normalized schemas: {json.dumps(problems, indent=2)}"


def test_raw_pydantic_schemas_are_all_well_formed():
    """The lint itself is not over-firing: every raw pydantic schema passes it."""
    problems = {
        case.name: lint_schema(case.model.model_json_schema()).render()
        for case in CASES
        if not lint_schema(case.model.model_json_schema()).ok
    }
    assert not problems, f"the lint rejects valid pydantic output: {problems}"


# ---------------------------------------------------------------------------
# 3. Properties
# ---------------------------------------------------------------------------


def test_normalization_is_idempotent():
    """Normalizing an already-normalized schema changes nothing.

    A non-idempotent normalizer makes the wire schema depend on how many times
    it was rendered, which breaks the "what the model saw is what was recorded"
    invariant.
    """
    offenders = []
    for case in CASES:
        raw = case.model.model_json_schema()
        try:
            once = to_wire(normalize_schema(raw))
        except SchemaNormalizationError:
            continue
        twice = to_wire(normalize_schema(once))
        if once != twice:
            offenders.append(case.name)
    assert not offenders, f"not idempotent: {offenders}"


def test_presentation_keywords_are_actually_removed():
    """Every keyword in ``PRESENTATION_KEYWORDS`` that pydantic emits is gone."""
    for case in CASES:
        raw = case.model.model_json_schema()
        try:
            normalized = to_wire(normalize_schema(raw))
        except SchemaNormalizationError:
            continue
        present = {k for k in schema_keywords(normalized) if k in PRESENTATION_KEYWORDS}
        assert not present, f"{case.name}: presentation keywords survived: {present}"


def test_description_is_kept_because_it_is_prompt_content():
    """``description`` is explicitly *not* presentation: the model needs it."""
    from pydantic import BaseModel, Field

    class Documented(BaseModel):
        a: int = Field(description="the a value")

    normalized = to_wire(normalize_schema(Documented.model_json_schema()))
    assert normalized["properties"]["a"]["description"] == "the a value"


def test_normalizer_never_invents_keywords():
    """Normalization removes and inlines; it must not add a keyword pydantic never emitted.

    (Except ``$defs``/``definitions`` retention, which is an existing key.)
    """
    for case in CASES:
        raw = case.model.model_json_schema()
        try:
            normalized = to_wire(normalize_schema(raw))
        except SchemaNormalizationError:
            continue
        raw_keys = schema_keywords(raw)
        new_keys = schema_keywords(normalized) - raw_keys
        assert not new_keys, f"{case.name}: normalizer invented keywords {new_keys}"


def test_normalized_schema_is_frozen():
    """A tool schema is part of the model-facing contract and must not be mutable."""
    from arm.kernel.freeze import FrozenDict

    normalized = normalize_schema(CASES_BY_NAME["flat"].model.model_json_schema())
    assert isinstance(normalized, FrozenDict)
    with pytest.raises(TypeError):
        normalized["type"] = "string"  # type: ignore[index]


def test_to_wire_round_trips_to_plain_json():
    """``to_wire`` must produce something ``json.dumps`` accepts unchanged."""
    for case in CASES:
        raw = case.model.model_json_schema()
        try:
            normalized = to_wire(normalize_schema(raw))
        except SchemaNormalizationError:
            continue
        assert json.loads(json.dumps(normalized)) == normalized


# ---------------------------------------------------------------------------
# 4. Cross-check the mini-validator against the real jsonschema library
# ---------------------------------------------------------------------------

def _load_reference_jsonschema():
    """Try to import the real ``jsonschema`` library, wherever it may live.

    It is deliberately NOT a dependency of this spike: the whole point of
    :mod:`jsonschema_subset` is to be independent of it. When it happens to be
    importable the cross-check runs; otherwise those tests skip and the finding
    stands on the mini-validator alone.

    Set ``Q2_JSONSCHEMA_PATH`` to a site-packages directory to point at a copy
    that is not on the default path::

        Q2_JSONSCHEMA_PATH=/path/to/site-packages ../.venv/bin/python -m pytest q2_schema
    """
    import os

    extra = os.environ.get("Q2_JSONSCHEMA_PATH")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        import jsonschema as module

        return module
    except Exception:  # noqa: BLE001
        return None


_jsonschema = _load_reference_jsonschema()
_HAVE_JSONSCHEMA = _jsonschema is not None


@pytest.mark.skipif(not _HAVE_JSONSCHEMA, reason="reference jsonschema library not installed")
def test_mini_validator_agrees_with_reference_library():
    """If the real ``jsonschema`` package is present, the two must agree exactly.

    Without this, a bug in :mod:`jsonschema_subset` could invent divergences (or
    hide them). Every payload of every case is checked against both validators,
    on the raw schema.

    The reference library needs ``$defs`` support, so it is run in
    ``Draft202012Validator`` mode.
    """
    from jsonschema import Draft202012Validator

    disagreements = []
    checked = 0
    for case in CASES:
        raw = case.model.model_json_schema()
        reference = Draft202012Validator(raw)
        for payload, origin in case.payloads():
            checked += 1
            mine = SubsetValidator(raw).is_valid(payload)
            theirs = reference.is_valid(payload)
            if mine != theirs:
                disagreements.append(
                    {
                        "case": case.name,
                        "origin": origin,
                        "payload": json.dumps(payload, default=str),
                        "mini": mine,
                        "reference": theirs,
                    }
                )
    assert checked > 1400
    assert not disagreements, (
        f"mini-validator disagrees with jsonschema on {len(disagreements)} payloads: "
        f"{json.dumps(disagreements[:10], indent=2)}"
    )


@pytest.mark.skipif(not _HAVE_JSONSCHEMA, reason="reference jsonschema library not installed")
def test_mini_validator_agrees_on_the_bug_reproductions():
    """The same cross-check, on the hand-built hostile schemas.

    The hostile schemas are where the findings live, so the validator that
    reports them is cross-checked there specifically.
    """
    from jsonschema import Draft202012Validator

    from hostile import attacks

    disagreements = []
    for attack in attacks():
        if not isinstance(attack.schema, dict):
            continue
        try:
            normalized = to_wire(normalize_schema(attack.schema))
        except Exception:  # noqa: BLE001 - refusals/crashes are covered elsewhere
            continue
        for label, schema in (("raw", attack.schema), ("normalized", normalized)):
            try:
                reference = Draft202012Validator(schema)
            except Exception:  # noqa: BLE001 - a malformed normalized schema
                continue
            for payload in attack.probes:
                mine = SubsetValidator(schema).is_valid(payload)
                theirs = reference.is_valid(payload)
                if mine != theirs:
                    disagreements.append(
                        {
                            "attack": attack.name,
                            "schema": label,
                            "payload": json.dumps(payload, default=str),
                            "mini": mine,
                            "reference": theirs,
                        }
                    )
    assert not disagreements, (
        f"mini-validator disagrees with jsonschema: {json.dumps(disagreements[:10], indent=2)}"
    )

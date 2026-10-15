"""PASSING regression guards for defects that were found and then FIXED.

These started life as ``xfail(strict=True)`` demonstrations in
``test_findings_bugs.py``. The Lead applied fixes while this review was running,
so they were promoted here: they now pass, and they stay here to stop the bug
from coming back.

Kept separate from ``test_findings_bugs.py`` so that "still broken" and "was
broken, now guarded" never blur together.
"""

from __future__ import annotations

import json

import pytest

import _bootstrap  # noqa: F401
from arm.tools.schema import SchemaNormalizationError, normalize_schema, to_wire
from jsonschema_subset import SubsetValidator
from schema_lint import lint_schema

# ---------------------------------------------------------------------------
# FIXED: `$ref` siblings are now conjoined instead of overwriting
# ---------------------------------------------------------------------------


def test_fixed_ref_sibling_properties_conjoins_instead_of_overwriting():
    """A sibling ``properties`` must ADD to the target's, not replace them.

    Before the fix the normalized schema was::

        {"inner": {"properties": {"sneaky": {...}}, "required": ["value"]}}

    -- ``value``'s type constraint was gone, so ``{"inner": {"value": "x"}}``
    was accepted where the raw schema rejected it. Now::

        {"inner": {"properties": {"value": {...}, "sneaky": {...}},
                   "required": ["value"]}}
    """
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "$ref": "#/$defs/Inner",
                "properties": {"sneaky": {"type": "integer"}},
            }
        },
        "required": ["inner"],
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))

    inner = normalized["properties"]["inner"]
    assert set(inner["properties"]) == {"value", "sneaky"}, (
        f"the target's `value` must survive alongside the sibling's `sneaky`: {inner}"
    )

    divergences = [
        (payload, SubsetValidator(schema).is_valid(payload), SubsetValidator(normalized).is_valid(payload))
        for payload in (
            {"inner": {"value": "not-an-integer"}},
            {"inner": {"value": "not-an-integer", "sneaky": 1}},
            {"inner": {"value": None, "sneaky": 1}},
            {"inner": {"value": 1, "sneaky": 1}},
            {"inner": {"sneaky": 1}},
        )
    ]
    assert not [d for d in divergences if d[1] != d[2]], f"divergences: {divergences}"


def test_fixed_ref_sibling_required_is_a_union():
    """A sibling ``required`` must UNION with the target's, not replace it.

    Before: ``required`` became ``["zzz"]`` and ``value`` was lost, so
    ``{"inner": {"zzz": 2}}`` was accepted. Now it is ``["value", "zzz"]``.
    """
    schema = {
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner", "required": ["zzz"]}},
        "required": ["inner"],
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))

    required = normalized["properties"]["inner"]["required"]
    assert sorted(required) == ["value", "zzz"], f"required must be the union, got {required}"

    divergences = [
        (payload, SubsetValidator(schema).is_valid(payload), SubsetValidator(normalized).is_valid(payload))
        for payload in (
            {"inner": {"value": 1, "zzz": 2}},
            {"inner": {"zzz": 2}},
            {"inner": {"value": 1}},
            {"inner": {}},
        )
    ]
    assert not [d for d in divergences if d[1] != d[2]], f"divergences: {divergences}"


def test_fixed_ref_sibling_description_that_agrees_is_kept():
    """A sibling that merely repeats the target's value conjoins cleanly."""
    schema = {
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner", "description": "same"}},
        "$defs": {"Inner": {"type": "object", "description": "same", "properties": {}}},
    }
    normalized = to_wire(normalize_schema(schema))
    assert normalized["properties"]["inner"]["description"] == "same"


# ---------------------------------------------------------------------------
# FIXED: a single-branch `anyOf` no longer emits `anyOf: {...}`
# ---------------------------------------------------------------------------


def test_fixed_bare_single_branch_anyof_collapses_to_the_branch():
    """A one-branch ``anyOf`` collapses to the branch itself, not to a malformed ``anyOf``.

    Before: ``{"anyOf": [{"type": "string"}]}`` -> ``{"anyOf": {"type": "string"}}``
    -- an object where JSON Schema requires an array. Now: ``{"type": "string"}``.
    """
    schema = {"type": "object", "properties": {"s": {"anyOf": [{"type": "string"}]}}}
    normalized = to_wire(normalize_schema(schema))

    assert lint_schema(normalized).ok, lint_schema(normalized).render()
    assert normalized["properties"]["s"] == {"type": "string"}


def test_fixed_single_branch_oneof_also_collapses():
    """The same fix applies to ``oneOf``: a single branch is pure indirection."""
    schema = {"type": "object", "properties": {"s": {"oneOf": [{"type": "integer"}]}}}
    normalized = to_wire(normalize_schema(schema))

    assert lint_schema(normalized).ok, lint_schema(normalized).render()
    assert normalized["properties"]["s"] == {"type": "integer"}


def test_fixed_no_catalogue_case_produces_malformed_output():
    """No model in the catalogue yields a malformed normalized schema any more."""
    from catalogue import CASES

    problems = {}
    for case in CASES:
        try:
            normalized = to_wire(normalize_schema(case.model.model_json_schema()))
        except SchemaNormalizationError:
            continue
        report = lint_schema(normalized)
        if not report.ok:
            problems[case.name] = report.render()
    assert not problems, json.dumps(problems, indent=2)


# ---------------------------------------------------------------------------
# ACCEPTED (fail-closed): a single-branch combinator whose siblings conflict
# ---------------------------------------------------------------------------


def test_conflicting_single_branch_siblings_refuse_rather_than_widen():
    """A conflicting sibling now REFUSES; it must never silently prefer one side.

    ``{"anyOf": [{"pattern": "^a"}], "pattern": "^b"}`` is satisfiable in JSON
    Schema (both patterns apply) but the normalizer cannot express the
    conjunction without ``allOf``. Refusing is the fail-closed choice and is
    strictly better than the old behaviour, which kept only ``^b`` and widened.

    This test accepts either a correct conjunction OR a loud refusal. It fails
    only on the two bad outcomes: silent widening, or malformed output.
    """
    schema = {
        "type": "object",
        "properties": {
            "s": {"anyOf": [{"type": "string", "pattern": "^a"}], "pattern": "^b"}
        },
    }
    try:
        normalized = to_wire(normalize_schema(schema))
    except SchemaNormalizationError:
        return  # fail-closed: acceptable

    assert lint_schema(normalized).ok, lint_schema(normalized).render()
    divergences = [
        (payload, SubsetValidator(schema).is_valid(payload), SubsetValidator(normalized).is_valid(payload))
        for payload in ({"s": "abc"}, {"s": "bcd"}, {"s": "xyz"}, {"s": 1})
    ]
    assert not [d for d in divergences if d[1] != d[2]], (
        f"silently widened: {divergences}; normalized={json.dumps(normalized)}"
    )


# ---------------------------------------------------------------------------
# FIXED: crash-safety (BUG 3) -- every refusal is SchemaNormalizationError
# ---------------------------------------------------------------------------


def test_fixed_boolean_refs_are_handled():
    """A ``$ref`` to a boolean definition is handled, not crashed on.

    Before: ``TypeError: 'bool' object is not iterable``.
    Now: ``$ref`` -> ``true`` resolves to the siblings (``true`` accepts
    everything); ``$ref`` -> ``false`` raises ``SchemaNormalizationError``
    because the conjunction is unsatisfiable.
    """
    # true: accepts everything, so the siblings are the whole story.
    normalized = to_wire(
        normalize_schema(
            {
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Yes", "type": "string"}},
                "$defs": {"Yes": True},
            }
        )
    )
    assert normalized["properties"]["x"] == {"type": "string"}

    # false: nothing can satisfy it.
    with pytest.raises(SchemaNormalizationError):
        normalize_schema(
            {
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Nope"}},
                "$defs": {"Nope": False},
            }
        )


def test_fixed_boolean_root_schema_raises_the_documented_error():
    """A boolean root schema raises ``SchemaNormalizationError``, not ``AttributeError``.

    Before: ``AttributeError: 'bool' object has no attribute 'get'``.
    """
    with pytest.raises(SchemaNormalizationError):
        normalize_schema(True)


def test_fixed_malformed_defs_raises_the_documented_error():
    """A non-mapping ``$defs`` raises ``SchemaNormalizationError``, not ``ValueError``.

    Before: ``ValueError: dictionary update sequence element #0 has length 1``.
    """
    with pytest.raises(SchemaNormalizationError):
        normalize_schema(
            {"type": "object", "properties": {"a": {"type": "integer"}}, "$defs": ["x"]}
        )


def test_fixed_non_mapping_root_raises_the_documented_error():
    """A non-mapping, non-boolean root raises ``SchemaNormalizationError``."""
    with pytest.raises(SchemaNormalizationError):
        normalize_schema(["not", "a", "schema"])


# ---------------------------------------------------------------------------
# FIXED: the depth ceiling no longer refuses ordinary nesting (BUG 4)
# ---------------------------------------------------------------------------


def test_fixed_deep_but_finite_nesting_is_accepted():
    """Non-recursive nesting is accepted far beyond any realistic tool schema.

    Before: a chain of 13 nested models raised
    ``SchemaNormalizationError: ... a recursive schema cannot be expressed ...``
    -- wrong on the facts (it was not recursive) and fatal (the tool could not be
    registered).
    """
    from catalogue import DeepChain13

    normalized = to_wire(normalize_schema(DeepChain13.model_json_schema()))
    assert normalized["type"] == "object"
    assert "properties" in normalized


# ---------------------------------------------------------------------------
# FIXED: unreachable recursion no longer refuses the schema (BUG 5)
# ---------------------------------------------------------------------------


def test_fixed_unused_recursive_def_does_not_refuse():
    """An unreferenced recursive ``$defs`` entry is dropped, not treated as a cycle.

    Before: ``SchemaNormalizationError: recursive $ref '#/$defs/Unused' ...``
    even though the root never referenced ``Unused``.
    """
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "$defs": {
            "Unused": {
                "type": "object",
                "properties": {"self": {"$ref": "#/$defs/Unused"}},
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))
    assert "$defs" not in normalized
    assert normalized["properties"] == {"a": {"type": "integer"}}


# ---------------------------------------------------------------------------
# FIXED: a property named `$ref` no longer defeats `$defs` liveness (BUG 6)
# ---------------------------------------------------------------------------


def test_fixed_property_named_dollar_ref_does_not_pin_defs():
    """``$defs`` liveness is decided by real references, not by a key-name scan.

    Before: ``$defs`` survived because ``_contains_ref`` saw
    ``properties.$ref`` -- a property *named* ``$ref`` -- and mistook it for a
    reference. Reachable from pydantic via ``Field(alias="$ref")``.
    """
    schema = {
        "type": "object",
        "properties": {
            "inner": {"$ref": "#/$defs/Inner"},
            "$ref": {"type": "string"},
        },
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))
    assert "$defs" not in normalized, json.dumps(normalized)
    assert normalized["properties"]["$ref"] == {"type": "string"}


# ---------------------------------------------------------------------------
# FIXED: `allOf` is collapsed and inlined (BUG 7)
# ---------------------------------------------------------------------------


def test_fixed_single_branch_allof_is_inlined():
    """A one-branch ``allOf`` is pure indirection and is inlined.

    Before it reached the model-facing schema verbatim, and ``allOf`` is
    explicitly rejected by OpenAI strict mode.
    """
    schema = {
        "type": "object",
        "properties": {"m": {"allOf": [{"$ref": "#/$defs/Inner"}]}},
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))
    assert "allOf" not in json.dumps(normalized), json.dumps(normalized)
    assert normalized["properties"]["m"]["properties"]["value"] == {"type": "integer"}


def test_fixed_multi_branch_allof_is_merged_or_refused():
    """A multi-branch ``allOf`` over objects is merged, or refused -- never passed through.

    ``allOf`` is not in the function-calling subset, so passing it through is not
    an option. The acceptable outcomes are a merged conjunction or a loud
    refusal; this test rejects only the silent pass-through.
    """
    schema = {
        "allOf": [
            {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
        ]
    }
    try:
        normalized = to_wire(normalize_schema(schema))
    except SchemaNormalizationError:
        return
    assert "allOf" not in json.dumps(normalized), json.dumps(normalized)

    divergences = [
        (payload, SubsetValidator(schema).is_valid(payload), SubsetValidator(normalized).is_valid(payload))
        for payload in ({"a": 1, "b": "x"}, {"a": 1}, {"b": "x"}, {"a": "s", "b": "x"})
    ]
    assert not [d for d in divergences if d[1] != d[2]], f"divergences: {divergences}"


# ---------------------------------------------------------------------------
# FIXED: RFC 6901 escapes in `$ref` are decoded (BUG 8)
# ---------------------------------------------------------------------------


def test_fixed_escaped_pointer_tokens_are_inlined():
    """``~1`` and ``~0`` in a ``$ref`` are decoded so the target is found.

    Before, ``#/$defs/A~1B`` was never resolved: the ``$ref`` and the whole
    ``$defs`` block survived, so the normalizer silently did nothing.
    """
    schema = {
        "type": "object",
        "properties": {
            "slash": {"$ref": "#/$defs/A~1B"},
            "tilde": {"$ref": "#/$defs/A~0B"},
        },
        "$defs": {"A/B": {"type": "integer"}, "A~B": {"type": "string"}},
    }
    normalized = to_wire(normalize_schema(schema))
    assert "$ref" not in json.dumps(normalized), json.dumps(normalized)
    assert normalized["properties"]["slash"] == {"type": "integer"}
    assert normalized["properties"]["tilde"] == {"type": "string"}


def test_fixed_escape_decoding_order_matters():
    """``~01`` must decode to ``~1``, not to ``/``.

    RFC 6901 requires decoding ``~1`` before ``~0``; the reverse order turns
    ``~01`` into ``/``. This pins the order.
    """
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/A~01B"}},
        "$defs": {"A~1B": {"type": "integer"}},
    }
    normalized = to_wire(normalize_schema(schema))
    assert normalized["properties"]["x"] == {"type": "integer"}, json.dumps(normalized)


def test_fixed_the_depth_ceiling_is_now_unreachable_from_pydantic():
    """The normalizer's nesting ceiling is above what pydantic can even generate.

    After BUG 4 the ceiling is ``_MAX_NESTING = 400``, raised deliberately above
    any realistic schema. Measuring where each side actually gives up:

        n=40   pydantic ok,      normalize_schema ok
        n=55   pydantic RecursionError  (the normalizer is never reached)
        n=60   pydantic RecursionError

    So the ceiling can no longer be hit by a schema pydantic produced: pydantic's
    own recursion limit fires first. The ceiling survives only as a guard against
    a hand-written or hostile document.

    This test records the ordering so that lowering the ceiling back into the
    reachable range is a visible change.
    """
    from pydantic import BaseModel

    from arm.tools.schema import _MAX_NESTING

    def chain(n):
        prev = type("L0", (BaseModel,), {"__annotations__": {"v": int}})
        for i in range(1, n):
            prev = type(f"L{i}", (BaseModel,), {"__annotations__": {"child": prev}})
        return prev

    assert _MAX_NESTING >= 400, f"the ceiling was lowered to {_MAX_NESTING}"

    # pydantic itself gives up well before the ceiling.
    with pytest.raises(RecursionError):
        chain(60).model_json_schema()

"""FAILING tests: the two defects that survive in the current normalizer.

HISTORY. This file originally demonstrated nine defects (BUG 1 - BUG 9). The
Lead fixed all nine while this review was running, including a regression (BUG 9)
that the BUG 1 fix had introduced. Those nine now live in
``test_fixed_regressions.py``, where they pass and guard against re-introduction.

What is left here is what a second adversarial pass found in the *rewritten*
implementation:

* **BUG 10** -- the ``properties`` merge is one level deep, so a same-named
  sub-property is silently overwritten (a widening).
* **BUG 11** -- ``definitions`` is deleted and re-emitted as ``$defs`` while
  ``#/definitions/...`` references survive verbatim, producing a dangling
  pointer.
* **BUG 12** -- an ``additionalProperties: false`` sibling is merged into the
  target, where it suddenly sees the target's ``properties`` and stops
  forbidding them (a widening).

Both are ``xfail(strict=True)``:

* while the bug is present the test **fails**, which ``strict=True`` reports as
  ``xfail`` -- the suite stays green while the defect stays documented;
* the moment the fix lands the test **passes**, and ``strict=True`` turns that
  into ``XPASS(strict)`` = a *failure*. A fixed bug therefore cannot be silently
  un-fixed, and the marker cannot be left behind by accident.

Each test carries its minimal reproduction, the observed output, and the exact
minimal fix.

Nothing here modifies ``src/``.
"""

from __future__ import annotations

import json

import pytest

import _bootstrap  # noqa: F401
from arm.tools.schema import SchemaNormalizationError, normalize_schema, to_wire
from jsonschema_subset import SubsetValidator


def _walk_refs(node):
    """Yield every ``$ref`` string appearing anywhere in ``node``."""
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
    """Resolve a local JSON Pointer, raising if it dangles."""
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


# ---------------------------------------------------------------------------
# BUG 10 -- the `properties` merge is SHALLOW: a colliding name is overwritten
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG 10: `_conjoin` merges `properties` one level deep, so a colliding name is lost",
)
def test_bug10_conjoin_properties_name_collision_widens():
    """Merging ``properties`` with ``{**existing, **value}`` loses a constraint on collision.

    Repro::

        {"type": "object",
         "properties": {"m": {"$ref": "#/$defs/T",
                              "properties": {"x": {"type": "string"}}}},
         "$defs": {"T": {"type": "object",
                         "properties": {"x": {"type": "integer"}}}}}

    The raw schema requires ``m.x`` to be **both** an integer and a string --
    unsatisfiable, so nothing is accepted. Observed::

        {"properties": {"m": {"type": "object",
                              "properties": {"x": {"type": "string"}}}}}

    The target's ``x: integer`` was replaced by the sibling's ``x: string``, so
    ``{"m": {"x": "s"}}`` -- which the raw schema rejects -- is now accepted.

    The merge is correct when the names *differ*: ``{"x": integer, "y":
    string}``. The defect is specifically a same-named sub-property, because the
    merge is one level deep -- ``properties`` is merged by key, but the schemas
    *under* those keys are not.

    Reachable from pydantic::

        class M(BaseModel):
            inner: Annotated[
                Inner,
                Field(json_schema_extra={"properties": {"value": {"type": "string"}}}),
            ]

    Fix: the merge must be a genuine conjunction at every depth, or it must
    refuse. Sketch::

        if key in _SCHEMA_MAP_KEYWORDS and isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_schema_maps(existing, value, ref=ref)
            continue

    where ``_merge_schema_maps`` calls ``_conjoin`` on names present in both maps
    and concatenates the rest. ``_conjoin`` cannot always express the
    conjunction -- two different ``type`` values have no single-schema form here
    -- so for a genuine scalar collision the correct outcome is to raise
    ``SchemaNormalizationError``. Refusing is strictly better than silently
    dropping a constraint. **At minimum, a collision must never be resolved by
    preferring one side.**
    """
    schema = {
        "type": "object",
        "properties": {
            "m": {"$ref": "#/$defs/T", "properties": {"x": {"type": "string"}}}
        },
        "$defs": {"T": {"type": "object", "properties": {"x": {"type": "integer"}}}},
    }
    try:
        normalized = to_wire(normalize_schema(schema))
    except SchemaNormalizationError:
        return  # fail-closed: acceptable

    divergences = [
        (
            payload,
            SubsetValidator(schema).is_valid(payload),
            SubsetValidator(normalized).is_valid(payload),
        )
        for payload in ({"m": {"x": 1}}, {"m": {"x": "s"}})
    ]
    assert not [d for d in divergences if d[1] != d[2]], (
        f"WIDENED by a shallow `properties` merge: {divergences}; "
        f"normalized={json.dumps(normalized)}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG 10: reachable from pydantic via Field(json_schema_extra={'properties': ...})",
)
def test_bug10_reachable_from_pydantic():
    """The same shallow merge, reached through a documented pydantic feature.

    ``Field(json_schema_extra=...)`` is the supported way to shape the emitted
    schema, and it can place a ``properties`` sibling next to a ``$ref``. So this
    is not a hand-written-schema-only concern.
    """
    from typing import Annotated

    from pydantic import BaseModel, Field

    class Inner(BaseModel):
        value: int

    class M(BaseModel):
        inner: Annotated[
            Inner,
            Field(json_schema_extra={"properties": {"value": {"type": "string"}}}),
        ]

    raw = M.model_json_schema()
    try:
        normalized = to_wire(normalize_schema(raw))
    except SchemaNormalizationError:
        return

    divergences = [
        (
            payload,
            SubsetValidator(raw).is_valid(payload),
            SubsetValidator(normalized).is_valid(payload),
        )
        for payload in ({"inner": {"value": 1}}, {"inner": {"value": "s"}})
    ]
    assert not [d for d in divergences if d[1] != d[2]], (
        f"WIDENED: {divergences}; normalized={json.dumps(normalized)}"
    )


# ---------------------------------------------------------------------------
# BUG 11 -- `definitions` is deleted while `#/definitions/...` refs survive
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG 11: `definitions` is dropped and re-emitted as `$defs`, dangling every `#/definitions/` ref",
)
def test_bug11_definitions_spelling_produces_a_dangling_pointer():
    """A surviving ``#/definitions/X`` ref is left pointing at a deleted location.

    ``normalize_schema`` accepts both ``$defs`` and ``definitions``, and
    ``_pointer_name`` decodes both, but the output always re-emits under
    ``$defs`` while the reference *text* is preserved verbatim.

    Repro (draft-07 style -- still common in hand-written and MCP schemas)::

        {"type": "object",
         "properties": {"a": {"$ref": "#/definitions/Node"}},
         "definitions": {"Node": {"type": "object",
                                  "properties": {"child": {"$ref": "#/definitions/Node"}}}}}

    ``Node`` is cyclic, so the reference is correctly *not* inlined. Observed::

        {"properties": {"a": {"$ref": "#/definitions/Node"}},
         "$defs": {"Node": {"properties": {"child": {"$ref": "#/definitions/Node"}},
                            "type": "object"}}}

    The definition moved from ``definitions`` to ``$defs``, but the reference
    still says ``#/definitions/Node``. Dereferencing fails::

        jsonschema._WrappedReferencingError: PointerToNowhere:
        '/definitions/Node' does not exist within {...}

    The schema is therefore unusable for any payload that actually contains
    ``a`` -- and it fails *at validation time*, in the provider, not here.

    Reachable via ``tool(parameters={...})``, which takes a raw mapping, and
    ``#/definitions/`` is what draft-07 emitters and several MCP servers produce.

    Fix: keep the two spellings consistent. Either

    (a) write kept definitions back into the keyword the surviving refs use --
        emit them under ``definitions`` when a ``#/definitions/...`` ref
        survives; or

    (b) rewrite the reference text in the same pass that moves the definition,
        mapping ``#/definitions/X`` to ``#/$defs/X``.

    (b) is the smaller change and keeps one output spelling, which is what a
    model-facing subset wants. Whichever is chosen, the invariant to hold is:
    **every ``$ref`` in the output must resolve within the output.**
    """
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/Node"}},
        "definitions": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/definitions/Node"}},
            }
        },
    }
    normalized = to_wire(normalize_schema(schema))

    refs = list(_walk_refs(normalized))
    assert refs, "the cyclic reference should have been preserved"
    for ref in refs:
        _resolve_pointer(normalized, ref)  # raises if dangling


@pytest.mark.xfail(
    strict=True,
    reason="BUG 11: `definitions` is popped unconditionally even when a ref survives",
)
def test_bug11_mixed_defs_and_definitions_drops_one():
    """``$defs`` and ``definitions`` in one schema: one is silently deleted.

    Repro::

        {"type": "object",
         "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"$ref": "#/definitions/B"}},
         "$defs": {"A": {"type": "integer"}},
         "definitions": {"B": {"type": "string"}}}

    ``A`` is inlined (fine). ``B`` survives as a reference, but ``definitions``
    is popped while the ``#/definitions/B`` reference remains::

        {"properties": {"a": {"type": "integer"}, "b": {"$ref": "#/definitions/B"}}}

    Dereferencing ``b`` raises ``PointerToNowhere``.

    Fix: see :func:`test_bug11_definitions_spelling_produces_a_dangling_pointer`.
    """
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"$ref": "#/definitions/B"}},
        "$defs": {"A": {"type": "integer"}},
        "definitions": {"B": {"type": "string"}},
    }
    normalized = to_wire(normalize_schema(schema))
    for ref in _walk_refs(normalized):
        _resolve_pointer(normalized, ref)


# ---------------------------------------------------------------------------
# BUG 12 -- `additionalProperties: false` as a `$ref` sibling changes meaning
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG 12: an `additionalProperties` sibling is merged into the target and stops forbidding the target's properties",
)
def test_bug12_additional_properties_sibling_widens():
    """``additionalProperties: false`` next to ``$ref`` forbids EVERYTHING; after merging it forbids nothing.

    This one is subtle and is a direct consequence of how JSON Schema scopes
    ``additionalProperties``: it applies to the ``properties``/``patternProperties``
    of **its own subschema**, not of the schemas it is combined with.

    Repro::

        {"type": "object",
         "properties": {"inner": {"$ref": "#/$defs/Inner", "additionalProperties": false}},
         "required": ["inner"],
         "$defs": {"Inner": {"type": "object",
                             "properties": {"value": {"type": "integer"}},
                             "required": ["value"]}}}

    The sibling subschema is ``{"$ref": ..., "additionalProperties": false}``. It
    declares **no** ``properties`` of its own, so ``additionalProperties: false``
    there means "no property is allowed at all" -- the conjunction with the
    target's ``required: ["value"]`` is unsatisfiable. Nothing is accepted.

    Observed::

        {"properties": {"inner": {"type": "object",
                                  "properties": {"value": {"type": "integer"}},
                                  "required": ["value"],
                                  "additionalProperties": false}}}

    The keyword was moved into the target, where it now *does* see
    ``properties: {value}`` and therefore permits ``value``. So
    ``{"inner": {"value": 1}}`` -- rejected by the raw schema -- is accepted.

    Verified against the reference ``jsonschema`` library::

        Draft202012Validator({"$ref": "#/$defs/Inner", "additionalProperties": false,
                              "$defs": {...}}).is_valid({"value": 1})   # False

    Fix: do not merge a sibling ``additionalProperties`` into the target's node.
    The two are not conjunctive in the naive way -- moving the keyword changes
    which ``properties`` it applies to. Either

    (a) refuse when a sibling carries ``additionalProperties`` and the target has
        ``properties``/``patternProperties`` (fail-closed, smallest change); or

    (b) express the conjunction structurally, e.g. keep the sibling's own
        ``properties: {}`` alongside it so the scoping is preserved::

            merged["additionalProperties"] = value
            merged.setdefault("patternProperties", {})
            # and do NOT let it see the target's properties -- which in general
            # means emitting {"allOf": [target, {"additionalProperties": false}]}

    (a) is the honest minimal fix: this is a case the normalizer cannot express
    in the subset, so it should say so rather than change the accepted set.
    """
    schema = {
        "type": "object",
        "properties": {
            "inner": {"$ref": "#/$defs/Inner", "additionalProperties": False}
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
    try:
        normalized = to_wire(normalize_schema(schema))
    except SchemaNormalizationError:
        return  # fail-closed: acceptable

    divergences = [
        (
            payload,
            SubsetValidator(schema).is_valid(payload),
            SubsetValidator(normalized).is_valid(payload),
        )
        for payload in ({"inner": {"value": 1}}, {"inner": {}})
    ]
    assert not [d for d in divergences if d[1] != d[2]], (
        f"WIDENED by merging an `additionalProperties` sibling: {divergences}; "
        f"normalized={json.dumps(normalized)}"
    )

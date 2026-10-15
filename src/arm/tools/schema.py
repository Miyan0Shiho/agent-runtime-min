"""Turning pydantic's JSON Schema output into something a model can consume.

pydantic v2 emits a faithful, fully general JSON Schema: ``$defs``, ``$ref``,
``anyOf``, ``title``, ``default``, and more. That is the right output for
validation and the wrong output for a function-calling API, which accepts a
narrow subset and may silently ignore -- or reject -- what it does not
understand.

The normalizer is deliberately **conservative and lossless with respect to
validation**: it removes presentation-only keywords and inlines references, but
it never weakens a constraint. If a payload validated against the original
schema, it must still validate against the normalized one, and vice versa. That
equivalence is what the tests check, because a normalizer that quietly loosens a
constraint would let invalid arguments reach a tool body.

What it does:

* inlines ``$ref`` into ``$defs`` **where that is possible**, conjoining the
  reference with its siblings (JSON Schema says ``$ref`` is conjunctive, so a
  sibling may add constraints but may never replace the target's). A reference
  that takes part in a cycle cannot be inlined, and is kept as a reference --
  ``$ref``/``$defs`` are themselves part of the accepted subset, so keeping one
  is a correct outcome rather than a fallback;
* drops ``$defs`` once nothing references it, and keeps only the definitions that
  are still referenced;
* **refuses** keywords the subset does not accept and that cannot be rewritten
  (``not``, ``if``/``then``/``else``, ...). Passing them through would produce a
  schema the provider rejects -- or, worse, silently ignores, which would let the
  model send payloads the tool author excluded;
* strips presentation-only keywords (``title``, ``default``, ...) while keeping
  ``description``, which is prompt content;
* collapses a single-branch ``anyOf``/``oneOf``/``allOf`` into its branch,
  conjoining rather than overwriting.

Three subtleties, each of which was a real defect found by adversarial review:

* **Annotations never conflict.** ``description`` is not an assertion -- two
  descriptions do not have to be reconciled, so a sibling's description simply
  wins. Treating them as constraints made the idiomatic
  ``Annotated[Model, Field(description=...)]`` pattern refuse to register at all.
* **``$defs`` is walked lazily.** Normalizing every definition eagerly meant an
  *unreferenced* recursive definition refused the whole schema, even though
  nothing in the model-facing output touched it.
* **Boolean schemas are schemas.** ``true`` and ``false`` are valid JSON Schema
  subschemas and must pass through rather than crash.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..kernel.freeze import deep_freeze, thaw

#: Keywords that carry no validation meaning. Stripped because they inflate the
#: schema the model must read without telling it anything actionable.
PRESENTATION_KEYWORDS = frozenset(
    {
        "title",
        "default",
        "examples",
        "$schema",
        "$id",
        "$comment",
        "discriminator",
        "readOnly",
        "writeOnly",
        "deprecated",
    }
)

#: Keywords that *annotate* rather than assert. Two different values do not form
#: an unsatisfiable conjunction -- JSON Schema collects annotations rather than
#: intersecting them -- so a sibling's value simply wins.
#:
#: ``description`` is deliberately here and deliberately not in
#: ``PRESENTATION_KEYWORDS``: it is prompt content and must survive into the
#: output, but it must never be treated as a constraint to reconcile.
ANNOTATION_KEYWORDS = frozenset(
    {"description", "title", "examples", "$comment", "deprecated", "readOnly", "writeOnly"}
)

#: Keywords whose values are themselves schemas (or lists of schemas).
_SCHEMA_KEYWORDS = frozenset(
    {
        "items",
        "additionalProperties",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
    }
)

_SCHEMA_LIST_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})

_SCHEMA_MAP_KEYWORDS = frozenset({"properties", "patternProperties", "$defs", "definitions"})

#: Combinators that may be collapsed when they have exactly one branch.
_COLLAPSIBLE = ("anyOf", "oneOf", "allOf")

#: Keywords that a function-calling JSON Schema subset does not accept and that
#: cannot be rewritten into an equivalent accepted form.
#:
#: These are refused rather than passed through. Passing one through would send a
#: schema the provider either rejects outright or silently ignores -- and a
#: silently ignored constraint is the worse of the two, because the model is then
#: allowed to send payloads the tool author explicitly excluded.
UNSUPPORTED_KEYWORDS = frozenset(
    {
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "dependentRequired",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "contains",
        "patternProperties",
        "$dynamicRef",
        "$dynamicAnchor",
        "contentSchema",
    }
)

#: A sanity ceiling on nesting. Cycles are caught by reference tracking, so this
#: only guards against a pathologically deep document; it is set well above any
#: realistic tool schema so that ordinary nesting is never refused.
_MAX_NESTING = 400


class SchemaNormalizationError(ValueError):
    """The schema could not be normalized without changing its meaning."""


def normalize_schema(schema: Mapping[str, Any] | bool) -> Mapping[str, Any]:
    """Return a model-safe view of ``schema``.

    The result is frozen, because a tool schema is part of the model-facing
    contract and must not be edited after registration.
    """
    if isinstance(schema, bool):
        # A boolean schema is valid JSON Schema (``true`` accepts anything,
        # ``false`` accepts nothing). There is nothing to normalize.
        raise SchemaNormalizationError(
            "a boolean root schema has no object form to normalize"
        )
    if not isinstance(schema, Mapping):
        raise SchemaNormalizationError(
            f"a schema must be an object, got {type(schema).__name__}"
        )

    raw_defs = schema.get("$defs")
    raw_legacy = schema.get("definitions")
    for block, label in ((raw_defs, "$defs"), (raw_legacy, "definitions")):
        if block is not None and not isinstance(block, Mapping):
            raise SchemaNormalizationError(
                f"{label} must be an object, got {type(block).__name__}"
            )
    # Both spellings are read into one table, and only ``$defs`` is emitted. A
    # schema that used the draft-07 spelling therefore keeps working, instead of
    # losing its definitions and leaving dangling references behind.
    defs: dict[str, Any] = {}
    if isinstance(raw_legacy, Mapping):
        defs.update(raw_legacy)
    if isinstance(raw_defs, Mapping):
        defs.update(raw_defs)

    # Definitions that take part in a reference cycle are never inlined. Deciding
    # this up front -- rather than discovering the cycle mid-walk -- is what makes
    # normalization idempotent: a first pass that inlined one level of a cycle
    # would leave a document the second pass inlines further.
    cyclic = _cyclic_definitions(defs)

    #: Definitions still referenced after inlining. Populated during the walk, so
    #: ``$defs`` liveness is decided by what actually survived rather than by
    #: re-scanning the output for the ``$ref`` key -- a scan cannot tell a
    #: reference from a *property* that happens to be named ``$ref``.
    still_referenced: set[str] = set()

    normalized = _normalize(
        dict(schema),
        defs,
        seen=(),
        referenced=still_referenced,
        cyclic=cyclic,
        depth=0,
    )

    # Drop ``$defs`` unless something genuinely still points into it. When
    # something does, keep only the definitions that are reachable, normalized
    # lazily -- an unreferenced definition (recursive or not) must never affect
    # the result.
    if isinstance(normalized, dict) and still_referenced:
        kept: dict[str, Any] = {}
        pending = list(still_referenced)
        while pending:
            name = pending.pop()
            if name in kept or name not in defs:
                continue
            nested: set[str] = set()
            kept[name] = _normalize(
                defs[name],
                defs,
                seen=(name,),
                referenced=nested,
                cyclic=cyclic,
                depth=0,
            )
            pending.extend(nested - set(kept))
        if kept:
            normalized["$defs"] = kept
    if isinstance(normalized, dict):
        normalized.pop("definitions", None)
        if not normalized.get("$defs"):
            normalized.pop("$defs", None)

    return deep_freeze(normalized)


def _normalize(
    node: Any,
    defs: Mapping[str, Any],
    *,
    seen: tuple[str, ...],
    referenced: set[str],
    cyclic: frozenset[str],
    depth: int,
) -> Any:
    if depth > _MAX_NESTING:
        raise SchemaNormalizationError(
            "schema nesting exceeded the sanity limit; refusing rather than "
            "risking unbounded recursion"
        )

    # -- boolean schemas are valid and pass through ----------------------
    if isinstance(node, bool):
        return node

    if isinstance(node, list):
        return [
            _normalize(
                item, defs, seen=seen, referenced=referenced, cyclic=cyclic, depth=depth + 1
            )
            for item in node
        ]

    if not isinstance(node, Mapping):
        return node

    # -- resolve a reference, conjoining it with its siblings ------------
    if "$ref" in node:
        ref = _canonical_ref(str(node["$ref"]))
        target = _resolve_ref(ref, defs)
        if target is None:
            # An unresolvable reference is passed through rather than dropped:
            # silently removing a constraint would widen the schema. It also
            # means ``$defs`` must survive, so the reference stays meaningful.
            name = _pointer_name(ref)
            if name is not None:
                referenced.add(name)
            return {
                key: _normalize(
                    value,
                        defs,
                        seen=seen,
                        referenced=referenced,
                        cyclic=cyclic,
                        depth=depth + 1,
                )
                for key, value in node.items()
            }

        name = _pointer_name(ref)
        if name is not None:
            # A definition in a reference cycle is never inlined. The reference
            # is kept intact and its definition preserved, so the schema stays
            # usable: ``$ref``/``$defs`` are part of the model-facing subset.
            #
            # Deciding this from a precomputed cycle set, rather than from a
            # running ``seen`` chain, is what makes normalization idempotent: a
            # chain-based check would inline one level of a cycle on the first
            # pass and leave a document the second pass inlines further.
            if name in cyclic or name in seen:
                referenced.add(name)
                return {
                    ("$ref" if key == "$ref" else key): (
                        ref
                        if key == "$ref"
                        else _normalize(
                            value,
                            defs,
                            seen=seen,
                            referenced=referenced,
                            cyclic=cyclic,
                            depth=depth + 1,
                        )
                    )
                    for key, value in node.items()
                }
            seen = (*seen, name)

        siblings = {k: v for k, v in node.items() if k != "$ref"}
        if isinstance(target, bool):
            # ``true`` accepts everything, so the siblings are the whole story;
            # ``false`` accepts nothing, so the conjunction is unsatisfiable.
            if target is True:
                return _normalize(
                    dict(siblings),
                    defs,
                    seen=seen,
                    referenced=referenced,
                    cyclic=cyclic,
                    depth=depth + 1,
                )
            raise SchemaNormalizationError(
                f"$ref {ref!r} points at the ``false`` schema, which accepts "
                "nothing; no sibling can make that satisfiable"
            )
        if not isinstance(target, Mapping):
            raise SchemaNormalizationError(
                f"$ref {ref!r} does not point at a schema object"
            )
        return _normalize(
            _conjoin(target, siblings, ref=ref),
            defs,
            seen=seen,
            referenced=referenced,
            cyclic=cyclic,
            depth=depth + 1,
        )

    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in PRESENTATION_KEYWORDS:
            continue
        if key in UNSUPPORTED_KEYWORDS:
            # Refusing here is the loud option, and loud is right: the alternative
            # is a tool that registers fine and then fails -- or worse, quietly
            # loses a constraint -- the first time the model calls it.
            raise SchemaNormalizationError(
                f"{key!r} is not supported by the model-facing JSON Schema "
                "subset and cannot be rewritten into an equivalent form; remove "
                "it, or express the constraint another way"
            )
        if key in _SCHEMA_KEYWORDS:
            result[key] = _normalize(
                value,
                        defs,
                        seen=seen,
                        referenced=referenced,
                        cyclic=cyclic,
                        depth=depth + 1,
            )
        elif key in _SCHEMA_LIST_KEYWORDS:
            # The list is normalized but *not* collapsed here. Collapsing is left
            # entirely to ``_collapse_combinator``, which needs to see the
            # original array: collapsing first would replace the list with a
            # single object, and the later pass would then emit
            # ``"anyOf": {...}`` -- an object where JSON Schema requires an array.
            result[key] = [
                _normalize(
                item, defs, seen=seen, referenced=referenced, cyclic=cyclic, depth=depth + 1
            )
                for item in (value or [])
            ]
        elif key in _SCHEMA_MAP_KEYWORDS:
            # ``$defs`` is *not* walked here: definitions are normalized lazily,
            # so an unreferenced (possibly recursive) definition cannot refuse a
            # schema that never uses it.
            if key in ("$defs", "definitions"):
                continue
            if not isinstance(value, Mapping):
                raise SchemaNormalizationError(
                    f"{key!r} must be an object, got {type(value).__name__}"
                )
            result[key] = {
                entry: _normalize(
                    sub,
                        defs,
                        seen=seen,
                        referenced=referenced,
                        cyclic=cyclic,
                        depth=depth + 1,
                )
                for entry, sub in value.items()
            }
        else:
            result[key] = value

    return _collapse_combinator(result)


def _conjoin(
    target: Mapping[str, Any], siblings: Mapping[str, Any], *, ref: str
) -> dict[str, Any]:
    """Combine a ``$ref`` target with its sibling keywords.

    JSON Schema treats ``$ref`` as **conjunctive** with its siblings: the
    instance must satisfy both. A plain ``dict.update`` would instead let a
    sibling *replace* a target keyword, which changes the accepted payload set in
    both directions -- a sibling ``properties`` would drop the target's
    constraints (widening), and a sibling ``required`` would erase the target's
    (narrowing). Either is a silent change to what the model is allowed to send.

    Where two sides genuinely cannot be merged, this refuses rather than picking
    a winner.
    """
    merged: dict[str, Any] = dict(target)
    for key, value in siblings.items():
        # Annotations are not assertions: there is no conjunction to satisfy, so
        # the sibling simply wins. Treating a description mismatch as a conflict
        # made the idiomatic ``Annotated[Model, Field(description=...)]`` pattern
        # impossible to register.
        if key in ANNOTATION_KEYWORDS:
            merged[key] = value
            continue

        # A keyword that is stripped anyway can never conflict in the output.
        if key in PRESENTATION_KEYWORDS:
            continue

        if key not in merged:
            merged[key] = value
            continue

        existing = merged[key]

        # Object-shaped keywords merge by key: both sets of constraints apply.
        #
        # The merge recurses rather than taking the later value wholesale. A
        # one-level ``{**existing, **value}`` would let a sibling redefine a
        # same-named *sub-property* and silently discard the target's constraint
        # on it -- a widening that looks like an ordinary override.
        if key in _SCHEMA_MAP_KEYWORDS and isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _conjoin_maps(existing, value, ref=ref, path=key)
            continue

        # ``additionalProperties`` interacts with ``properties``, so merging the
        # two can invert its meaning: ``false`` alongside a sibling's ``properties``
        # starts forbidding *those* names, when the author wrote it to forbid
        # everything the sibling had not listed.
        if key == "additionalProperties" and existing != value:
            raise SchemaNormalizationError(
                f"cannot conjoin conflicting 'additionalProperties' from a sibling "
                f"of $ref {ref!r}: the target says {existing!r} and the sibling "
                f"says {value!r}; merging them would change which properties are "
                "forbidden"
            )

        # ``required`` is a set union: every name from either side is required.
        if key == "required" and isinstance(existing, list) and isinstance(value, list):
            combined = list(existing)
            for name in value:
                if name not in combined:
                    combined.append(name)
            merged[key] = combined
            continue

        # These combinators accumulate: both sets of branches apply.
        if key in ("allOf", "anyOf") and isinstance(existing, list) and isinstance(value, list):
            merged[key] = [*existing, *value]
            continue

        # A scalar constraint on both sides must be identical, or the
        # conjunction is unsatisfiable and cannot be expressed here.
        if existing != value:
            raise SchemaNormalizationError(
                f"cannot conjoin {key!r} from a sibling of $ref {ref!r}: the "
                f"target says {existing!r} and the sibling says {value!r}; "
                "refusing rather than silently preferring one"
            )
    return merged


def _conjoin_maps(
    left: Mapping[str, Any], right: Mapping[str, Any], *, ref: str, path: str
) -> dict[str, Any]:
    """Conjoin two maps of schemas, recursing where both sides define a key.

    ``properties`` is a map from name to *schema*, so a naive ``update`` treats a
    same-named property as a replacement. The two schemas for that name must
    instead be conjoined -- otherwise a sibling silently discards the target's
    constraint, which widens what the model may send.
    """
    result: dict[str, Any] = dict(left)
    for name, value in right.items():
        if name not in result:
            result[name] = value
            continue
        existing = result[name]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[name] = _conjoin(
                existing, value, ref=f"{ref} (at {path}.{name})"
            )
            continue
        if existing != value:
            raise SchemaNormalizationError(
                f"cannot conjoin {path}.{name!r} from a sibling of $ref {ref!r}: "
                f"the target says {existing!r} and the sibling says {value!r}"
            )
    return result


def _collapse_combinator(node: dict[str, Any]) -> dict[str, Any]:
    """Inline a single-branch combinator into its parent node.

    pydantic expresses ``Optional[X]`` as ``anyOf: [X, {type: null}]`` and a
    ``Literal`` as ``anyOf`` of ``const`` nodes. A *one-branch* combinator is pure
    indirection, and some function-calling APIs do not accept ``anyOf`` at all,
    so inlining it is a real simplification.

    The branch and the surrounding keywords are **conjoined**, not overwritten.
    Overwriting would let the surviving branch's keywords replace the node's own
    -- changing which payloads are accepted, silently.
    """
    for keyword in _COLLAPSIBLE:
        branches = node.get(keyword)
        if not isinstance(branches, list):
            continue

        if len(branches) == 1:
            single = branches[0]
            rest = {k: v for k, v in node.items() if k != keyword}
            if not rest:
                # Nothing to conjoin. A non-object branch is left wrapped,
                # because emitting a bare boolean or list in its place would not
                # be a schema in that position.
                return single if isinstance(single, dict) else node
            if isinstance(single, dict):
                return _conjoin(single, rest, ref=keyword)
            continue

        if keyword == "allOf":
            # ``allOf`` is *not* interchangeable with ``anyOf``: its branches
            # conjoin, so it cannot be left on the wire either. Unlike the
            # combinators, every function-calling subset that rejects ``anyOf``
            # also rejects ``allOf`` -- so a multi-branch ``allOf`` must be
            # merged into one object schema, or refused.
            return _merge_all_of(branches, node)
    return node


def _merge_all_of(branches: list[Any], node: dict[str, Any]) -> dict[str, Any]:
    """Fold an ``allOf`` into a single schema, or refuse.

    Every branch must be an object schema with no conflicting scalar constraint;
    the merge is the same conjunction used for ``$ref`` siblings. A branch that
    cannot be merged -- a boolean, a non-object, or a conflicting scalar -- is
    refused rather than dropped, because dropping it would silently widen the
    accepted payload set.
    """
    rest = {k: v for k, v in node.items() if k != "allOf"}
    merged: dict[str, Any] = dict(rest)

    for index, branch in enumerate(branches):
        if isinstance(branch, bool):
            if branch is False:
                raise SchemaNormalizationError(
                    "an allOf branch is the ``false`` schema, which accepts "
                    "nothing; the conjunction is unsatisfiable"
                )
            continue  # ``true`` adds no constraint
        if not isinstance(branch, dict):
            raise SchemaNormalizationError(
                f"an allOf branch is {type(branch).__name__}, not a schema object; "
                "refusing rather than dropping a constraint"
            )
        merged = _conjoin(merged, branch, ref="allOf")

    return _collapse_combinator(merged) if len(merged) != len(rest) else merged


def _cyclic_definitions(defs: Mapping[str, Any]) -> frozenset[str]:
    """Names of definitions that participate in a reference cycle.

    A definition is cyclic when it can reach itself through ``$ref``. Such a
    definition can never be inlined -- the expansion would not terminate -- so
    identifying them up front lets the walk leave their references intact and
    stay idempotent.

    The graph is built by scanning for ``$ref`` *keyword values* only. A property
    that happens to be named ``$ref`` is not a reference, and treating it as one
    would make an ordinary schema look recursive.
    """
    edges: dict[str, set[str]] = {}
    for name, definition in defs.items():
        found: set[str] = set()
        _collect_refs(definition, defs, found)
        edges[name] = found

    cyclic: set[str] = set()
    for start in edges:
        # Depth-first search for a path back to ``start``.
        stack = list(edges.get(start, ()))
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current == start:
                cyclic.add(start)
                break
            if current in visited:
                continue
            visited.add(current)
            stack.extend(edges.get(current, ()))
    return frozenset(cyclic)


def _collect_refs(node: Any, defs: Mapping[str, Any], found: set[str]) -> None:
    """Collect the names of definitions referenced anywhere inside ``node``."""
    if isinstance(node, bool) or not isinstance(node, (Mapping, list)):
        return
    if isinstance(node, list):
        for item in node:
            _collect_refs(item, defs, found)
        return
    for key, value in node.items():
        if key == "$ref" and isinstance(value, str):
            name = _pointer_name(value)
            if name is not None and name in defs:
                found.add(name)
            continue
        # ``$defs`` nested inside a definition is not part of this definition's
        # own references; it is a separate scope.
        if key in ("$defs", "definitions"):
            continue
        _collect_refs(value, defs, found)


def _canonical_ref(ref: str) -> str:
    """Rewrite a draft-07 ``#/definitions/`` reference to the ``$defs`` spelling.

    Both spellings name the same thing, and the output only ever carries
    ``$defs``. Leaving a ``#/definitions/X`` reference intact while deleting the
    ``definitions`` block produces a dangling pointer -- a schema that no
    validator and no provider can resolve.
    """
    if ref.startswith("#/definitions/"):
        return "#/$defs/" + ref[len("#/definitions/") :]
    return ref


def _pointer_name(ref: str) -> str | None:
    """The ``$defs`` key a local reference points at, if any."""
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            return _decode_pointer(ref[len(prefix) :])
    return None


def _decode_pointer(token: str) -> str:
    """Decode an RFC 6901 JSON Pointer token.

    ``~1`` means ``/`` and ``~0`` means ``~``, and the order matters: decoding
    ``~0`` first would turn ``~01`` into ``/`` instead of ``~1``.
    """
    return token.replace("~1", "/").replace("~0", "~")


def _resolve_ref(ref: str, defs: Mapping[str, Any]) -> Any | None:
    """Resolve a local ``#/$defs/Name`` reference. Remote refs are not resolved."""
    name = _pointer_name(ref)
    if name is None:
        return None
    return defs.get(name)


def to_wire(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Thaw a normalized schema for sending over the wire."""
    return dict(thaw(schema))

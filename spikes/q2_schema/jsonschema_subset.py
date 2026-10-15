"""An independent JSON Schema 2020-12 subset validator.

Why this file exists
--------------------
The question under test is *"does normalizing change the set of payloads a schema
accepts?"*. The only honest way to answer it is to apply **one** validator to
**both** the raw pydantic schema and the normalized schema. If two different
validators were used (say, pydantic for one side and a JSON Schema library for
the other), every difference they reported would be confounded by the difference
between the validators.

The repo venv has no ``jsonschema`` package, so the subset is implemented here.
It is deliberately independent of ``arm.tools.schema``: it shares no code with the
thing it is testing.

Specification used: JSON Schema 2020-12.

* ``$ref`` is resolved against the *root* document (``#/$defs/...``,
  ``#/definitions/...``, and arbitrary JSON Pointers).
* ``$ref`` **is conjunctive with its siblings** (2020-12 removed the draft-07
  "siblings are ignored" rule). This is the single most important line in the
  file: it is what makes the normalizer's ``merged[key] = value`` overwrite
  observable as a widening.
* Boolean schemas (``true`` / ``false``) are valid schemas.
* Unknown keywords are annotations and are ignored, exactly as the spec says.
* ``format`` is an annotation, not an assertion (same default as the
  ``jsonschema`` library and as every function-calling API).

Assertions implemented
    type, enum, const, properties, required, additionalProperties,
    patternProperties, propertyNames, minProperties, maxProperties, items,
    prefixItems, minItems, maxItems, uniqueItems, contains, minContains,
    maxContains, minLength, maxLength, pattern, minimum, maximum,
    exclusiveMinimum, exclusiveMaximum, multipleOf, anyOf, oneOf, allOf, not,
    if/then/else, $ref.

Accepted-but-never-asserted
    title, description, default, examples, deprecated, readOnly, writeOnly,
    $comment, $id, $schema, $anchor, format, discriminator, contentEncoding,
    contentMediaType, x-*.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

#: Keywords that carry no assertion. Kept here so a test can assert that a
#: keyword it cares about is genuinely being *checked* rather than ignored.
ANNOTATION_KEYWORDS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
        "$id",
        "$schema",
        "$anchor",
        "$dynamicAnchor",
        "format",
        "discriminator",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
    }
)


class UnresolvableRef(Exception):
    """A ``$ref`` that this validator cannot follow."""


def json_equal(left: Any, right: Any) -> bool:
    """JSON equality.

    Python's ``==`` is wrong for JSON in two places that matter here:
    ``True == 1`` and ``1 == 1.0``. JSON says ``true``, ``1`` and ``1.0`` are
    three distinct values for ``const``/``enum``/``uniqueItems``.
    """
    left_bool = isinstance(left, bool)
    right_bool = isinstance(right, bool)
    if left_bool or right_bool:
        return left_bool and right_bool and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) and (math.isnan(left) or math.isinf(left)):
            return False
        if isinstance(right, float) and (math.isnan(right) or math.isinf(right)):
            return False
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            json_equal(left[k], right[k]) for k in left
        )
    return False


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def _is_array(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value) and float(value).is_integer()
    return False


_TYPE_CHECKS = {
    "null": lambda v: v is None,
    "boolean": lambda v: isinstance(v, bool),
    "object": _is_object,
    "array": _is_array,
    "number": _is_number,
    "integer": _is_integer,
    "string": lambda v: isinstance(v, str),
}


class SubsetValidator:
    """Validates a payload against a JSON Schema, collecting every error."""

    def __init__(self, root: Any) -> None:
        self.root = root
        #: (id(schema node), id(payload node)) pairs currently on the stack.
        #: Only ever populated while descending, so it detects genuine cycles
        #: (a recursive schema against a recursive payload) without misfiring on
        #: the same schema node being applied to two sibling values.
        self._active: set[tuple[int, int]] = set()
        self.unresolved_refs: list[str] = []

    # -- public API --------------------------------------------------------
    def is_valid(self, payload: Any) -> bool:
        return not self.errors(payload)

    def errors(self, payload: Any, schema: Any = None) -> list[str]:
        out: list[str] = []
        self._check(schema if schema is not None else self.root, payload, "", out)
        return out

    # -- core walk ---------------------------------------------------------
    def _check(self, schema: Any, value: Any, path: str, out: list[str]) -> None:
        if schema is True or schema == {}:
            return
        if schema is False:
            out.append(f"{path or '<root>'}: false schema rejects everything")
            return
        if not isinstance(schema, dict):
            out.append(f"{path or '<root>'}: malformed schema {schema!r}")
            return

        key = (id(schema), id(value))
        if key in self._active:
            # Already checking this exact (schema, value) pair higher up the
            # stack: this is a cycle, and the coinductive answer is "valid".
            return
        self._active.add(key)
        try:
            self._check_inner(schema, value, path, out)
        finally:
            self._active.discard(key)

    def _check_inner(self, schema: dict, value: Any, path: str, out: list[str]) -> None:
        # -- $ref is conjunctive with its siblings (2020-12) ----------------
        ref = schema.get("$ref")
        if ref is not None:
            try:
                target = self._resolve(ref)
            except UnresolvableRef:
                self.unresolved_refs.append(str(ref))
                target = None
            if target is not None:
                self._check(target, value, path, out)
            siblings = {k: v for k, v in schema.items() if k != "$ref"}
            if siblings:
                self._check(siblings, value, path, out)
            return

        if "type" in schema:
            declared = schema["type"]
            names = declared if isinstance(declared, list) else [declared]
            if not any(_TYPE_CHECKS.get(n, lambda _v: True)(value) for n in names):
                out.append(
                    f"{path or '<root>'}: {_short(value)} is not of type {declared!r}"
                )

        if "enum" in schema and not any(json_equal(value, e) for e in schema["enum"]):
            out.append(f"{path or '<root>'}: {_short(value)} not in enum")

        if "const" in schema and not json_equal(value, schema["const"]):
            out.append(f"{path or '<root>'}: {_short(value)} != const {schema['const']!r}")

        if isinstance(value, dict):
            self._check_object(schema, value, path, out)

        if _is_array(value):
            self._check_array(schema, value, path, out)

        if isinstance(value, str):
            self._check_string(schema, value, path, out)

        if _is_number(value):
            self._check_number(schema, value, path, out)

        self._check_combinators(schema, value, path, out)

    def _check_object(self, schema: dict, value: dict, path: str, out: list[str]) -> None:
        properties = schema.get("properties") or {}
        pattern_properties = schema.get("patternProperties") or {}
        required = schema.get("required") or []

        for name in required:
            if name not in value:
                out.append(f"{path or '<root>'}: missing required property {name!r}")

        if "minProperties" in schema and len(value) < schema["minProperties"]:
            out.append(f"{path or '<root>'}: fewer than {schema['minProperties']} properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            out.append(f"{path or '<root>'}: more than {schema['maxProperties']} properties")

        for name, sub in properties.items():
            if name in value:
                self._check(sub, value[name], f"{path}.{name}", out)

        for pattern, sub in pattern_properties.items():
            try:
                regex = re.compile(pattern)
            except re.error:
                continue
            for name, item in value.items():
                if regex.search(name):
                    self._check(sub, item, f"{path}.{name}", out)

        if "propertyNames" in schema:
            for name in value:
                self._check(schema["propertyNames"], name, f"{path}.<name:{name}>", out)

        if "additionalProperties" in schema:
            extra = [
                n
                for n in value
                if n not in properties
                and not any(
                    _safe_search(p, n) for p in pattern_properties
                )
            ]
            if schema["additionalProperties"] is False and extra:
                out.append(
                    f"{path or '<root>'}: additional properties not allowed: {sorted(extra)!r}"
                )
            elif isinstance(schema["additionalProperties"], dict):
                for name in extra:
                    self._check(
                        schema["additionalProperties"],
                        value[name],
                        f"{path}.{name}",
                        out,
                    )

    def _check_array(self, schema: dict, value: Any, path: str, out: list[str]) -> None:
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(f"{path or '<root>'}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(f"{path or '<root>'}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            for i in range(len(value)):
                for j in range(i + 1, len(value)):
                    if json_equal(value[i], value[j]):
                        out.append(f"{path or '<root>'}: items {i} and {j} are not unique")
                        break

        prefix = schema.get("prefixItems") or []
        for index, sub in enumerate(prefix):
            if index < len(value):
                self._check(sub, value[index], f"{path}[{index}]", out)

        items = schema.get("items")
        if items is not None:
            if isinstance(items, list):  # draft-07 tuple form; tolerated
                for index, sub in enumerate(items):
                    if index < len(value):
                        self._check(sub, value[index], f"{path}[{index}]", out)
            else:
                for index in range(len(prefix), len(value)):
                    self._check(items, value[index], f"{path}[{index}]", out)

        if "contains" in schema:
            hits = 0
            for index, item in enumerate(value):
                sub_out: list[str] = []
                self._check(schema["contains"], item, f"{path}[{index}]", sub_out)
                if not sub_out:
                    hits += 1
            low = schema.get("minContains", 1)
            high = schema.get("maxContains")
            if hits < low or (high is not None and hits > high):
                out.append(f"{path or '<root>'}: contains matched {hits} items")

    def _check_string(self, schema: dict, value: str, path: str, out: list[str]) -> None:
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(f"{path or '<root>'}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{path or '<root>'}: longer than {schema['maxLength']}")
        if "pattern" in schema:
            try:
                if re.search(schema["pattern"], value) is None:
                    out.append(
                        f"{path or '<root>'}: {value!r} does not match {schema['pattern']!r}"
                    )
            except re.error:
                pass

    def _check_number(self, schema: dict, value: Any, path: str, out: list[str]) -> None:
        def num(x: Any) -> Decimal:
            return Decimal(str(x))

        try:
            v = num(value)
        except (InvalidOperation, ValueError):
            return
        if "minimum" in schema and v < num(schema["minimum"]):
            out.append(f"{path or '<root>'}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and v > num(schema["maximum"]):
            out.append(f"{path or '<root>'}: {value} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and v <= num(schema["exclusiveMinimum"]):
            out.append(f"{path or '<root>'}: {value} <= exclusiveMinimum")
        if "exclusiveMaximum" in schema and v >= num(schema["exclusiveMaximum"]):
            out.append(f"{path or '<root>'}: {value} >= exclusiveMaximum")
        if "multipleOf" in schema:
            divisor = num(schema["multipleOf"])
            if divisor != 0:
                quotient = v / divisor
                if quotient != quotient.to_integral_value():
                    out.append(f"{path or '<root>'}: {value} is not a multiple of {schema['multipleOf']}")

    def _check_combinators(
        self, schema: dict, value: Any, path: str, out: list[str]
    ) -> None:
        for keyword in ("allOf", "anyOf", "oneOf"):
            branches = schema.get(keyword)
            if branches is None:
                continue
            if not isinstance(branches, list):
                out.append(
                    f"{path or '<root>'}: {keyword} must be an array, got "
                    f"{type(branches).__name__} ({branches!r})"
                )
                continue
            results = []
            for branch in branches:
                sub: list[str] = []
                self._check(branch, value, path, sub)
                results.append(sub)
            passing = sum(1 for r in results if not r)
            if keyword == "allOf" and passing != len(results):
                out.append(f"{path or '<root>'}: allOf failed ({len(results) - passing} branches)")
            elif keyword == "anyOf" and passing == 0:
                out.append(f"{path or '<root>'}: anyOf matched no branch")
            elif keyword == "oneOf" and passing != 1:
                out.append(
                    f"{path or '<root>'}: oneOf matched {passing} branches, expected exactly 1"
                )

        if "not" in schema:
            sub = []
            self._check(schema["not"], value, path, sub)
            if not sub:
                out.append(f"{path or '<root>'}: 'not' schema matched")

        if "if" in schema:
            cond: list[str] = []
            self._check(schema["if"], value, path, cond)
            branch = "then" if not cond else "else"
            if branch in schema:
                self._check(schema[branch], value, path, out)

    # -- refs --------------------------------------------------------------
    def _resolve(self, ref: Any) -> Any:
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise UnresolvableRef(ref)
        pointer = ref[1:]
        if pointer in ("", "/"):
            return self.root
        if not pointer.startswith("/"):
            raise UnresolvableRef(ref)
        node = self.root
        for raw in pointer.split("/")[1:]:
            token = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict):
                if token not in node:
                    raise UnresolvableRef(ref)
                node = node[token]
            elif isinstance(node, (list, tuple)):
                try:
                    node = node[int(token)]
                except (ValueError, IndexError) as exc:
                    raise UnresolvableRef(ref) from exc
            else:
                raise UnresolvableRef(ref)
        return node


def _safe_search(pattern: str, name: str) -> bool:
    try:
        return re.search(pattern, name) is not None
    except re.error:
        return False


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def accepts(schema: Any, payload: Any) -> bool:
    """Convenience wrapper: does ``schema`` accept ``payload``?"""
    return SubsetValidator(schema).is_valid(payload)


def errors_of(schema: Any, payload: Any) -> list[str]:
    return SubsetValidator(schema).errors(payload)

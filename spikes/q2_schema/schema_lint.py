"""Structural well-formedness checks for a JSON Schema.

Independent of :mod:`arm.tools.schema` and of :mod:`jsonschema_subset`: this
module only asks "is this document a *well-formed* JSON Schema?", never "what
does it accept?". The two questions are different, and the normalizer can fail
the first while passing the second.

The check that matters most here: ``anyOf``/``oneOf``/``allOf`` must be
**arrays** of schemas. A normalizer that "collapses" a single-branch ``anyOf``
by replacing the array with the branch produces a document that is not a JSON
Schema at all -- it is a shape no validator, and no model server, can interpret.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: keyword -> the JSON type its value must have
_SCHEMA_OR_BOOL = {"additionalProperties", "propertyNames", "not", "if", "then", "else", "contains", "items", "unevaluatedItems", "unevaluatedProperties", "additionalItems"}
_SCHEMA_ARRAY = {"anyOf", "oneOf", "allOf", "prefixItems"}
_SCHEMA_MAP = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_STRING = {"$ref", "$schema", "$id", "$anchor", "title", "description", "format", "pattern", "contentEncoding", "contentMediaType", "$comment"}
_NUMBER = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
_NON_NEG_INT = {"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties", "minContains", "maxContains"}


@dataclass
class LintProblem:
    path: str
    keyword: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '<root>'}: {self.keyword}: {self.message}"


@dataclass
class LintReport:
    problems: list[LintProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self, limit: int = 6) -> str:
        shown = [str(p) for p in self.problems[:limit]]
        extra = len(self.problems) - len(shown)
        text = "; ".join(shown)
        return text + (f" (+{extra} more)" if extra > 0 else "")


def _is_schema(value: Any) -> bool:
    return isinstance(value, (dict, bool))


def lint_schema(schema: Any, *, _path: str = "", _report: LintReport | None = None) -> LintReport:
    """Return every structural problem in ``schema``."""
    report = _report if _report is not None else LintReport()
    _lint(schema, _path, report)
    return report


def _lint(node: Any, path: str, report: LintReport) -> None:
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        report.problems.append(
            LintProblem(path, "<schema>", f"must be an object or boolean, got {type(node).__name__}")
        )
        return

    for keyword, value in node.items():
        where = f"{path}.{keyword}"

        if keyword in _SCHEMA_ARRAY:
            if not isinstance(value, list):
                report.problems.append(
                    LintProblem(path, keyword, f"must be an ARRAY of schemas, got {type(value).__name__}")
                )
                continue
            if not value:
                report.problems.append(LintProblem(path, keyword, "must not be empty"))
            for index, branch in enumerate(value):
                if not _is_schema(branch):
                    report.problems.append(
                        LintProblem(path, keyword, f"branch {index} is not a schema")
                    )
                else:
                    _lint(branch, f"{where}[{index}]", report)

        elif keyword in _SCHEMA_MAP:
            if not isinstance(value, dict):
                report.problems.append(
                    LintProblem(path, keyword, f"must be an object of schemas, got {type(value).__name__}")
                )
                continue
            for name, sub in value.items():
                if not _is_schema(sub):
                    report.problems.append(
                        LintProblem(path, keyword, f"entry {name!r} is not a schema")
                    )
                else:
                    _lint(sub, f"{where}.{name}", report)

        elif keyword in _SCHEMA_OR_BOOL:
            if not _is_schema(value):
                report.problems.append(
                    LintProblem(path, keyword, f"must be a schema or boolean, got {type(value).__name__}")
                )
            elif keyword == "items" and isinstance(value, list):
                for index, sub in enumerate(value):
                    _lint(sub, f"{where}[{index}]", report)
            else:
                _lint(value, where, report)

        elif keyword == "type":
            names = value if isinstance(value, list) else [value]
            if not names or not all(isinstance(n, str) for n in names):
                report.problems.append(
                    LintProblem(path, "type", f"must be a string or array of strings, got {value!r}")
                )

        elif keyword == "required":
            if not isinstance(value, list) or not all(isinstance(n, str) for n in value):
                report.problems.append(
                    LintProblem(path, "required", f"must be an array of strings, got {value!r}")
                )

        elif keyword == "enum":
            if not isinstance(value, list) or not value:
                report.problems.append(
                    LintProblem(path, "enum", f"must be a non-empty array, got {value!r}")
                )

        elif keyword in _STRING:
            if not isinstance(value, str):
                report.problems.append(
                    LintProblem(path, keyword, f"must be a string, got {type(value).__name__}")
                )

        elif keyword in _NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                report.problems.append(
                    LintProblem(path, keyword, f"must be a number, got {type(value).__name__}")
                )

        elif keyword in _NON_NEG_INT:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                report.problems.append(
                    LintProblem(path, keyword, f"must be a non-negative integer, got {value!r}")
                )

        elif keyword == "discriminator":
            if not isinstance(value, dict) or "propertyName" not in value:
                report.problems.append(
                    LintProblem(path, keyword, "must be an object with a propertyName")
                )


def lint_ok(schema: Any) -> bool:
    return lint_schema(schema).ok


# ---------------------------------------------------------------------------
# Keyword discovery
# ---------------------------------------------------------------------------

#: Keywords whose *values* are schemas or collections of schemas.
_SCHEMA_VALUED = _SCHEMA_OR_BOOL | _SCHEMA_ARRAY | _SCHEMA_MAP


def schema_keywords(schema: Any) -> set[str]:
    """Every keyword used in ``schema``, walking **schema positions only**.

    This is deliberately not "every dict key anywhere": ``properties`` maps
    property *names* to schemas, so a model with a field called ``pattern``
    would otherwise be reported as using the ``pattern`` keyword. Property names
    are opaque data; only the values under them are schemas.

    Used by the catalogue tests to freeze what pydantic emits.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool) or not isinstance(node, dict):
            return
        for keyword, value in node.items():
            found.add(keyword)
            if keyword in _SCHEMA_MAP and isinstance(value, dict):
                for sub in value.values():
                    walk(sub)
            elif keyword in _SCHEMA_ARRAY and isinstance(value, list):
                for sub in value:
                    walk(sub)
            elif keyword in _SCHEMA_OR_BOOL:
                walk(value)
            elif keyword == "items" and isinstance(value, list):
                for sub in value:
                    walk(sub)

    walk(schema)
    return found

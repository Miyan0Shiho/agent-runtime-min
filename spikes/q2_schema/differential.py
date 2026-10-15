"""The differential test: does normalization change the accepted payload set?

For every model in :mod:`catalogue`, and every payload in its corpus, this
module applies **one** validator (:mod:`jsonschema_subset`) to **two** schemas:

* ``raw``  -- ``Model.model_json_schema()`` exactly as pydantic emits it;
* ``norm`` -- ``to_wire(normalize_schema(raw))``, what the runtime would ship.

It reports every payload where the two disagree. That disagreement is the whole
deliverable: it is either a **widening** (a constraint was dropped, so invalid
arguments reach a tool body) or a **narrowing** (a legal call is refused, or the
normalizer could not produce a schema at all).

Run directly for a human-readable transcript::

    cd spikes && ../.venv/bin/python q2_schema/differential.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import _bootstrap  # noqa: F401  (puts ``src/`` on sys.path)
from arm.tools.schema import SchemaNormalizationError, normalize_schema, to_wire
from catalogue import CASES, Case
from jsonschema_subset import SubsetValidator


def _new_validator(schema: Any) -> SubsetValidator:
    return SubsetValidator(schema)


@dataclass
class Divergence:
    case: str
    payload: Any
    origin: str
    raw_accepts: bool
    norm_accepts: bool
    raw_errors: list[str] = field(default_factory=list)
    norm_errors: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        if self.norm_accepts and not self.raw_accepts:
            return "WIDENED"
        return "NARROWED"

    def render(self, limit: int = 160) -> str:
        text = json.dumps(self.payload, default=str)
        if len(text) > limit:
            text = text[: limit - 3] + "..."
        lines = [
            f"  [{self.kind}] {self.case} ({self.origin})",
            f"    payload      : {text}",
            f"    raw  accepts : {self.raw_accepts}",
            f"    norm accepts : {self.norm_accepts}",
        ]
        if self.raw_errors:
            lines.append(f"    raw  errors  : {self.raw_errors[0]}")
        if self.norm_errors:
            lines.append(f"    norm errors  : {self.norm_errors[0]}")
        return "\n".join(lines)


@dataclass
class CaseResult:
    case: str
    payload_count: int
    divergences: list[Divergence] = field(default_factory=list)
    normalizer_error: str | None = None
    raw_schema: dict[str, Any] = field(default_factory=dict)
    norm_schema: dict[str, Any] | None = None
    keywords_raw: list[str] = field(default_factory=list)
    keywords_norm: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.divergences and self.normalizer_error is None


def all_keywords(schema: Any) -> list[str]:
    """Every dict key appearing anywhere in a schema, sorted."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            found.update(node.keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(schema)
    return sorted(found)


def run_case(
    case: Case,
    *,
    validator_factory: Callable[[Any], SubsetValidator] = _new_validator,
) -> CaseResult:
    raw = case.model.model_json_schema()
    result = CaseResult(case=case.name, payload_count=0, raw_schema=raw)
    result.keywords_raw = all_keywords(raw)

    try:
        norm = to_wire(normalize_schema(raw))
    except SchemaNormalizationError as exc:
        result.normalizer_error = f"SchemaNormalizationError: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001 - a crash is itself the finding
        result.normalizer_error = f"{type(exc).__name__}: {exc}"
        return result

    result.norm_schema = norm
    result.keywords_norm = all_keywords(norm)

    payloads = case.payloads()
    result.payload_count = len(payloads)
    for payload, origin in payloads:
        raw_v = validator_factory(raw)
        norm_v = validator_factory(norm)
        raw_ok = raw_v.is_valid(payload)
        norm_ok = norm_v.is_valid(payload)
        if raw_ok != norm_ok:
            result.divergences.append(
                Divergence(
                    case=case.name,
                    payload=payload,
                    origin=origin,
                    raw_accepts=raw_ok,
                    norm_accepts=norm_ok,
                    raw_errors=raw_v.errors(payload),
                    norm_errors=norm_v.errors(payload),
                )
            )
    return result


def run_all() -> list[CaseResult]:
    return [run_case(case) for case in CASES]


def main() -> None:
    results = run_all()
    total_payloads = 0
    total_divergences = 0

    print("=" * 78)
    print("Q2 differential: raw pydantic schema vs normalized schema")
    print("=" * 78)
    for result in results:
        total_payloads += result.payload_count
        total_divergences += len(result.divergences)
        status = "ok " if result.ok else "DIFF"
        print(
            f"\n[{status}] {result.case:24s} payloads={result.payload_count:4d} "
            f"divergences={len(result.divergences)}"
        )
        if result.normalizer_error:
            print(f"    normalizer refused: {result.normalizer_error}")
        for divergence in result.divergences:
            print(divergence.render())

    print("\n" + "=" * 78)
    print(f"cases={len(results)} payloads={total_payloads} divergences={total_divergences}")
    widened = sum(
        1 for r in results for d in r.divergences if d.kind == "WIDENED"
    )
    narrowed = sum(
        1 for r in results for d in r.divergences if d.kind == "NARROWED"
    )
    refused = sum(1 for r in results if r.normalizer_error)
    print(f"  widened={widened} narrowed={narrowed} refused={refused}")
    print("=" * 78)


if __name__ == "__main__":
    main()

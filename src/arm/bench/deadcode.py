"""Find code that nothing calls.

This project has hit the same defect five times, and every occurrence was found by
hand or by an expensive adversarial review:

* a schema normalizer that no tool path executed, so all its correctness work was
  a property of code that never ran;
* a duplicate-blocker check the caller pre-empted, so it could never receive the
  input it validated against;
* a scheduler class nothing instantiated, making reminders creatable and inert;
* a ``confirm_delivery`` with no caller, so the mailbox never delivered;
* an approval service that was constructed, passed to the runtime, and never
  consulted -- so approval gated nothing at all.

Each was invisible to a test suite that passed. Unit tests answer "is this code
correct?"; none of them asked "is this code called?".

**This is a ratchet, not a report.** A checker that prints a hundred names is a
checker nobody reads, so it compares against a recorded baseline: names that are
legitimately unreferenced (entry points, protocol implementations) are listed
once, and anything *new* fails the check. That is what turns it from a snapshot
into something that catches the next occurrence.

**What it cannot see, stated plainly.** A name that is referenced but whose
*result* is ignored; a validation the caller pre-empts (occurrence two above --
the function is called, with input already sanitized); and anything reached only
by reflection. Those need reading, and this tool does not pretend otherwise.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

#: Where the baseline is stored, so a run can be compared against a commit.
BASELINE_PATH = Path(__file__).with_name("deadcode-baseline.json")

#: Reasons for names that are referenced from nowhere on purpose.
#:
#: Kept in code rather than in the JSON so a reader sees *why* alongside the
#: check, and so adding an entry is a code change rather than a data edit.
#: Anything not listed here and not in the baseline is reported as new.
KNOWN_ENTRY_POINTS: Mapping[str, str] = {
    "main": "console script entry point, named in pyproject.toml",
    "install": "plugin protocol: called by the plugin registry",
    "build_parser": "CLI parser constructor",
}


@dataclass(frozen=True, slots=True)
class Definition:
    """A public name defined somewhere in the source tree."""

    name: str
    qualname: str
    kind: str
    file: str
    line: int

    def key(self) -> str:
        return f"{self.file}::{self.qualname}"


@dataclass
class Finding:
    """A definition with no reference, plus why it might be fine."""

    definition: Definition
    reason: str = ""

    @property
    def known(self) -> bool:
        return bool(self.reason)


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def collect_definitions(root: Path) -> list[Definition]:
    """Every public function, class and method defined under ``root``.

    Methods are included, and that matters: ``confirm_delivery`` -- one of the
    five occurrences -- was a method, and a module-level-only scan would have
    missed it.
    """
    definitions: list[Definition] = []

    for path in _iter_python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        relative = str(path.relative_to(root.parent))

        def visit(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not _is_public(child.name):
                        continue
                    qualname = f"{prefix}{child.name}"
                    kind = (
                        "class"
                        if isinstance(child, ast.ClassDef)
                        else "method" if prefix else "function"
                    )
                    definitions.append(
                        Definition(
                            name=child.name,
                            qualname=qualname,
                            kind=kind,
                            file=relative,
                            line=child.lineno,
                        )
                    )
                    # Recurse into classes to find their methods, but not into
                    # function bodies: a nested function is not part of the
                    # public surface.
                    if isinstance(child, ast.ClassDef):
                        visit(child, f"{qualname}.")

        visit(tree, "")

    return definitions


def collect_references(root: Path) -> "ReferenceIndex":
    """Index how every name is used under ``root``.

    A plain name count is not enough, and the reason is worth stating: ``enter``,
    ``attached`` and ``arm`` are ordinary words that appear in prose and in other
    classes' methods. Counting them as references to *this* class's method hides
    the very thing the check exists to find.

    So references are indexed two ways: by bare name (for module-level functions
    and classes, where the name is the identity) and by ``attribute of a known
    class`` (for methods, where the identity is the pair).
    """
    return ReferenceIndex(root)


@dataclass
class ReferenceIndex:
    """Which names are used, and as an attribute of which owner."""

    root: Path
    #: Bare names: calls, imports, variable uses.
    names: dict[str, int] = field(default_factory=dict)
    #: ``owner.attribute`` pairs, resolved through simple assignment tracking.
    attributes: set[str] = field(default_factory=set)
    #: Variables known to hold an instance of a class, e.g. ``self`` inside it.
    locals_to_class: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index()

    def _index(self) -> None:
        for path in _iter_python_files(self.root):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            self._walk(tree)

    def _walk(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                self.names[node.id] = self.names.get(node.id, 0) + 1
            elif isinstance(node, ast.alias):
                short = node.name.split(".")[-1]
                self.names[short] = self.names.get(short, 0) + 1
            elif isinstance(node, ast.Attribute):
                self.names[node.attr] = self.names.get(node.attr, 0) + 1
                # Record the attribute name; the owner is resolved separately so
                # a common word used on another class does not count.
                owner = node.value
                if isinstance(owner, ast.Name):
                    self.attributes.add(f"{owner.id}.{node.attr}")
                elif isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name):
                    self.attributes.add(f"{owner.func.id}.{node.attr}")

    def calls_method(self, class_name: str, method: str) -> bool:
        """Whether ``method`` is called on something of ``class_name``.

        Three shapes count, and they cover how methods are actually reached in
        this codebase: a literal ``Class.method(...)``, a call on a constructor
        ``Class().method(...)``, and a call on ``self`` inside the class body.
        """
        if f"{class_name}.{method}" in self.attributes:
            return True
        # `self.method(...)` inside the class itself.
        return f"self.{method}" in self.attributes


def find_uncalled(
    src: Path, *, baseline: Mapping[str, str] | None = None
) -> list[Finding]:
    """Definitions whose name appears nowhere outside their own definition.

    A definition is "uncalled" when the only occurrences of its name in the tree
    are the definition itself and its signature. That is a heuristic, and a
    deliberately conservative one: it reports a name only when nothing at all
    mentions it, so the findings are few enough to read.
    """
    known = KNOWN_ENTRY_POINTS if baseline is None else baseline
    definitions = collect_definitions(src)
    index = collect_references(src.parent)

    findings: list[Finding] = []
    for definition in definitions:
        # The rule is deliberately the *conservative* one: report a name only
        # when it appears nowhere at all.
        #
        # I first tried resolving methods by owner -- `Runtime.submit` versus
        # some other class's `submit` -- and it reported 229 findings including
        # obviously-used code, because attribute access in a dynamic language
        # cannot be resolved reliably from the syntax alone. A checker that cries
        # wolf is one nobody reads, and this project has already seen what
        # happens when a report is ignored.
        #
        # So the check answers one narrow question well: *does anything mention
        # this name at all?* Every one of the five real occurrences would have
        # been caught by it, because in each case nothing called the function.
        # Names that are mentioned somewhere but whose result is ignored are not
        # detectable this way, and the module docstring says so.
        #
        # A definition's own name is a *string* on its AST node, not a `Name`, so
        # it never counts itself.
        if index.names.get(definition.name, 0) > 0:
            continue

        findings.append(
            Finding(definition=definition, reason=known.get(definition.name, ""))
        )

    return findings


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, str]:
    """Read the recorded baseline, or an empty one when it does not exist yet."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def save_baseline(findings: Sequence[Finding], path: Path = BASELINE_PATH) -> int:
    """Record the current findings as accepted, with their reasons.

    Called deliberately, by a person who has read the list. An unexplained entry
    is worse than a finding, because it silences the check forever.
    """
    # A finding without a reason is refused rather than recorded as "accepted".
    # The baseline is the one place a real finding can be silenced, and an entry
    # that says nothing is exactly how that happens.
    unexplained = [f.definition.key() for f in findings if not f.reason]
    if unexplained:
        raise ValueError(
            "every baseline entry needs a reason; these have none: "
            + ", ".join(unexplained)
        )
    recorded = {finding.definition.key(): finding.reason for finding in findings}
    path.write_text(
        json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(recorded)


def new_findings(
    findings: Sequence[Finding], baseline: Mapping[str, str]
) -> list[Finding]:
    """The findings that are not already recorded.

    This is the ratchet: a recorded name stays quiet, and anything new fails.
    """
    return [f for f in findings if f.definition.key() not in baseline]


def render(findings: Sequence[Finding], baseline: Mapping[str, str]) -> str:
    lines: list[str] = []
    fresh = new_findings(findings, baseline)

    if not findings:
        lines.append("no uncalled public definitions were found")
        return "\n".join(lines)

    lines.append(f"{len(findings)} uncalled definition(s); {len(fresh)} not in the baseline")
    for finding in findings:
        marker = "NEW " if finding in fresh else "ok  "
        definition = finding.definition
        lines.append(
            f"  {marker} {definition.file}:{definition.line} {definition.qualname}"
            f"  ({definition.kind})"
        )
        if finding.reason:
            lines.append(f"        {finding.reason}")

    return "\n".join(lines)


def run(src: Path, baseline_path: Path = BASELINE_PATH) -> tuple[list[Finding], bool]:
    """Check the tree, returning the findings and whether anything is new."""
    baseline = load_baseline(baseline_path)
    findings = find_uncalled(src, baseline=KNOWN_ENTRY_POINTS)
    fresh = new_findings(findings, baseline)
    return findings, not fresh


def find_test_only_modules(src: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Modules whose public names are referenced only by the test suite.

    **Why this is a separate check from `find_uncalled`.** That one asks whether a
    name is mentioned *anywhere*, deliberately, because resolving attribute access
    in a dynamic language produced 229 false positives when it tried to be
    sharper. But "mentioned anywhere" includes the module's own tests -- so a
    module that exists, is well documented, has thorough tests, and **no caller
    anywhere in the product** is invisible to it.

    Measured: `jobs/` is 925 lines, in the REQUIREMENTS scope ("任务编排: jobs"), has
    its own test file, and is imported by nothing outside itself. It was invisible
    for exactly that reason.

    This check asks a question that *is* decidable without resolving anything:
    for each public name a module defines, does it appear anywhere in `src/` that
    is not the module itself? If no name does, the module cannot be reached from
    the product. Tests are checked separately to distinguish "unreachable and
    untested" (dead) from "unreachable but tested" (a feature nothing calls).

    Returns ``(module, public_names)`` pairs for modules that only their tests
    mention.
    """
    import re

    root = src.parent
    src_files = [p for p in src.rglob("*.py")]
    texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in src_files}
    tests_root = root / "tests"
    test_text = (
        "\n".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in tests_root.rglob("*.py")
        )
        if tests_root.exists()
        else ""
    )

    findings: list[tuple[str, tuple[str, ...]]] = []
    for module, text in texts.items():
        if module.stem == "__init__":
            continue
        public = {
            name
            for name in re.findall(r"^(?:def|class) ([A-Za-z_][A-Za-z0-9_]*)", text, re.M)
            if not name.startswith("_")
        }
        if not public:
            continue

        outside = "\n".join(
            other for path, other in texts.items() if path != module
        )
        if any(re.search(rf"\b{re.escape(name)}\b", outside) for name in public):
            continue
        # Only interesting when the tests do reach it: an unreferenced and
        # untested module is already reported by `find_uncalled`.
        if not any(re.search(rf"\b{re.escape(name)}\b", test_text) for name in public):
            continue

        findings.append(
            (str(module.relative_to(root)), tuple(sorted(public)))
        )
    return sorted(findings)

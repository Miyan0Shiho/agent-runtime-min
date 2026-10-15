"""Throwaway probes: run me, look at output, decide what is a real bug."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from arm.skills.catalog import CatalogPublisher, render_catalog
from arm.skills.registry import (
    PROJECT_RANK,
    CatalogSnapshot,
    FilesystemSkillProvider,
    SkillRegistry,
    SkillSummary,
    _parse_simple_frontmatter,
    _parse_yaml,
    _split_frontmatter,
)


def show(label, fn):
    try:
        print(f"{label}: {fn()!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: RAISED {type(exc).__name__}: {exc}")


print("=" * 70)
print("A. frontmatter parsing asymmetry")
print("=" * 70)

cases = {
    "duplicate keys": "name: a\ndescription: one\nname: b\ndescription: two",
    "name as list": "name: [a, b]\ndescription: ddd",
    "name as int": "name: 42\ndescription: ddd",
    "name as null": "name:\ndescription: ddd",
    "name as bool": "name: true\ndescription: ddd",
    "nested yaml": "name: a\ndescription: ddd\nmeta:\n  a: 1",
    "block scalar |": "name: a\ndescription: |\n  line one\n  line two",
    "folded scalar >": "name: a\ndescription: >\n  line one\n  line two",
    "anchors": "x: &d dd\nname: a\ndescription: *d",
    "anchor + merge": "base: &b\n  name: a\ndescription: dd\n<<: *b",
    "tab indent": "name: a\ndescription:\n\tddd",
    "unclosed quote": 'name: a\ndescription: "unclosed',
    "python tag": "name: !!python/object/apply:os.system ['echo hi']\ndescription: dd",
    "nul in value": "name: a\ndescription: d\x00d",
    "bom inside": "name: a\ndescription: \ufeffdd",
}
for label, raw in cases.items():
    show(f"  yaml   {label:20}", lambda raw=raw: _parse_yaml(raw))
    show(f"  simple {label:20}", lambda raw=raw: _parse_simple_frontmatter(raw))

print()
print("=" * 70)
print("B. _split_frontmatter shapes")
print("=" * 70)
show("BOM file", lambda: _split_frontmatter("\ufeff---\nname: a\n---\nbody"))
show("CRLF", lambda: _split_frontmatter("---\r\nname: a\r\n---\r\nbody"))
show("--- in body", lambda: _split_frontmatter("---\nname: a\n---\nbody\n---\nmore"))
show("empty frontmatter", lambda: _split_frontmatter("---\n---\nbody"))
show("no body", lambda: _split_frontmatter("---\nname: a\n---"))
show("frontmatter only+newline", lambda: _split_frontmatter("---\nname: a\n---\n"))
show("leading blank line", lambda: _split_frontmatter("\n---\nname: a\n---\nbody"))
show("spaces before ---", lambda: _split_frontmatter("   ---\nname: a\n---\nbody"))
show("---- fence", lambda: _split_frontmatter("----\nname: a\n----\nbody"))
show("nested fence", lambda: _split_frontmatter("---\nname: a\n  ---\nbody"))
show("trailing spaces fence", lambda: _split_frontmatter("---\nname: a\n---   \nbody"))

print()
print("=" * 70)
print("C. digest collisions")
print("=" * 70)
reg = SkillRegistry()


def s(name, desc):
    return SkillSummary(name=name, description=desc)


a = [s("a", "b\x1fc"), s("d", "e")]
b = [s("a", "b"), s("c", "d"), s("e", "")]
print("  A:", reg.catalog_digest(CatalogSnapshot(a)))
print("  B:", reg.catalog_digest(CatalogSnapshot(b)))
print("  equal?", reg.catalog_digest(CatalogSnapshot(a)) == reg.catalog_digest(CatalogSnapshot(b)))

print()
print("=" * 70)
print("D. publisher: same digest, different entries")
print("=" * 70)
pub = CatalogPublisher()
entries1 = [s("alpha", "one")]
r1 = pub.update(CatalogSnapshot(entries1), entries1, digest="D")
entries2 = [s("alpha", "two")]
r2 = pub.update(CatalogSnapshot(entries2), entries2, digest="D")
print("  first publish:", bool(r1))
print("  changed entries, same digest -> publish:", r2)
print("  published_entries now:", pub.published_entries)

print()
print("=" * 70)
print("E. simple parser accepted where yaml rejects / vice versa")
print("=" * 70)
raw = "name: a\ndescription: ddd\nwhenToUse: yes"
show("  yaml whenToUse yes", lambda: _parse_yaml(raw))
show("  simple whenToUse yes", lambda: _parse_simple_frontmatter(raw))

raw2 = "name: a\ndescription: >-\n  folded text here"
show("  yaml folded", lambda: _parse_yaml(raw2))
show("  simple folded", lambda: _parse_simple_frontmatter(raw2))

raw3 = "name: a\ndescription: ddd\ndisable-model-invocation: true"
show("  yaml bool", lambda: _parse_yaml(raw3))
show("  simple bool", lambda: _parse_simple_frontmatter(raw3))

print()
print("=" * 70)
print("F. full provider behaviour on hostile files")
print("=" * 70)
tmp = Path(tempfile.mkdtemp())
root = tmp / "skills"
root.mkdir()


def write(name, text):
    d = root / name
    d.mkdir(exist_ok=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


write("dupkeys", "---\nname: dupkeys\ndescription: first desc\nname: other\ndescription: second desc\n---\nbody\n")
write("bom", "\ufeff---\nname: bom\ndescription: a fine description\n---\nbody\n")
write("crlf", "---\r\nname: crlf\r\ndescription: a fine description\r\n---\r\nbody\r\n")
write("no-body", "---\nname: no-body\ndescription: a fine description\n---\n")
write("tagged", "---\nname: tagged\ndescription: !!python/object/apply:os.system ['echo pwned']\n---\nbody\n")
write("nested", "---\nname: nested\ndescription: a fine description\nextra:\n  a: 1\n---\nbody\n")
write("anchors", "---\nname: anchors\ndescription: &d a fine description\nwhenToUse: *d\n---\nbody\n")
write("fence-body", "---\nname: fence-body\ndescription: a fine description\n---\nbody\n---\nmore\n")

provider = FilesystemSkillProvider([(PROJECT_RANK, "project", root)])
for c in provider.list():
    print(f"  {c.summary.name:12} desc={c.summary.description!r} when={c.summary.when_to_use!r}")
print("  --- loads ---")
for c in provider.list():
    d = provider.load(c)
    print(f"  {c.summary.name:12} content={d.content!r}" if d else f"  {c.summary.name:12} None")

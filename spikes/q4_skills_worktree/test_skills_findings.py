"""FAILING tests: confirmed defects in the skills module.

S1, S1b, S2, S3 and S3b were fixed by the Lead while this review was running;
they are now ordinary passing tests and act as regression guards. The remainder
are still-open ``xfail(strict=True)`` findings.

Each test carries its minimal reproduction, the observed output, and the exact
minimal fix. Open findings are ``xfail(strict=True)``:

* while the bug is present the test **fails**, which ``strict=True`` reports as
  ``xfail`` -- the suite stays green while the defect stays documented;
* the moment the fix lands the test **passes**, and ``strict=True`` turns that
  into ``XPASS(strict)`` = a *failure*. A fixed bug therefore cannot be silently
  un-fixed, and the marker cannot be left behind by accident.

Nothing here modifies ``src/``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

import _bootstrap  # noqa: F401
from arm.skills.catalog import CatalogPublisher, render_catalog
from arm.skills.registry import (
    PROJECT_RANK,
    CatalogSnapshot,
    FilesystemSkillProvider,
    SkillError,
    SkillRegistry,
    SkillSummary,
    _parse_simple_frontmatter,
    _parse_yaml,
    _split_frontmatter,
)
from arm.skills.tool import SkillTool


def write_skill(root: Path, name: str, text: str) -> Path:
    """Create a ``<name>/SKILL.md`` bundle with raw text."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


def provider_for(root: Path) -> FilesystemSkillProvider:
    return FilesystemSkillProvider([(PROJECT_RANK, "project", root)])


def registry_for(root: Path) -> SkillRegistry:
    registry = SkillRegistry()
    registry.register_provider(provider_for(root))
    return registry


# ---------------------------------------------------------------------------
# BUG S1 -- a YAML syntax error escapes as a parser exception, not SkillError,
#           and takes the whole catalogue down with it
# ---------------------------------------------------------------------------


def test_one_yaml_syntax_error_does_not_hide_every_other_skill(tmp_path):
    """The module's stated contract: "failure is localized".

    ``list()`` catches only ``SkillError`` around ``_parse``. PyYAML raises
    ``yaml.scanner.ScannerError`` (a ``YAMLError``/``Exception``, *not* a
    ``SkillError``), so it propagates out of ``list()`` entirely.

    Repro: three skills, one with a tab-indented value (YAML forbids a tab as
    indentation). PyYAML raises; the other two skills become invisible.

    Observed::

        list() raised ScannerError; snapshot=[]

    and through the loop the model is told there are *no* skills::

        skill tool deploy  -> Error: unknown skill "deploy"; available skills are: (none)
        skill tool review  -> Error: unknown skill "review"; available skills are: (none)

    Fix: in ``_parse_yaml``, translate parser failures into the documented type,
    next to the existing ``frontmatter must be a mapping`` raise::

        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise SkillError(f"frontmatter is not valid YAML: {exc}") from exc

    ``list()`` then skips the one bad file and keeps the other two.
    """
    root = tmp_path / "skills"
    write_skill(root, "deploy", "---\nname: deploy\ndescription: deploy the service\n---\nbody\n")
    write_skill(root, "review", "---\nname: review\ndescription: review a diff\n---\nbody\n")
    # A tab cannot start a YAML token; PyYAML raises ScannerError.
    write_skill(root, "tabbed", "---\nname: tabbed\ndescription:\n\ttab indented\n---\nbody\n")

    provider = provider_for(root)
    try:
        names = [c.summary.name for c in provider.list()]
    except Exception as exc:  # noqa: BLE001 - the point is that a non-SkillError escapes
        raise AssertionError(
            f"{type(exc).__name__} escaped list() instead of the file being "
            f"skipped: {exc}"
        ) from exc

    assert names == ["deploy", "review"], (
        "one malformed file hid every other skill; failure was not localized"
    )


def test_a_yaml_syntax_error_does_not_empty_the_model_visible_catalogue(tmp_path):
    """The end-to-end consequence of BUG S1.

    ``SkillRegistry.snapshot`` catches ``Exception`` per provider, so it does not
    crash -- but it marks the snapshot ``complete=False`` and returns *nothing*.
    ``loop._publish_catalog`` refuses to publish an incomplete snapshot, so the
    model is never told about any skill and cannot load one.

    Observed::

        snapshot.skills == []        complete=False
        skill tool "deploy" -> Error: unknown skill "deploy"; available skills are: (none)

    Fix: the same ``SkillError`` translation as BUG S1. Localizing the failure at
    the file is what keeps the snapshot both complete and non-empty.
    """
    root = tmp_path / "skills"
    write_skill(root, "deploy", "---\nname: deploy\ndescription: deploy the service\n---\nbody\n")
    write_skill(root, "tabbed", "---\nname: tabbed\ndescription:\n\ttab indented\n---\nbody\n")

    registry = registry_for(root)
    snapshot = registry.snapshot()

    assert [s.name for s in snapshot.skills] == ["deploy"], (
        "the healthy skill disappeared because a sibling failed to parse"
    )
    assert snapshot.complete is True

    result = asyncio.run(SkillTool(registry).run({"name": "deploy"}, None))
    assert not result.is_error, f"the model cannot load a healthy skill: {result.content}"


# ---------------------------------------------------------------------------
# BUG S2 -- invalid UTF-8 escapes as UnicodeDecodeError (not an OSError)
# ---------------------------------------------------------------------------


def test_a_non_utf8_skill_file_is_skipped_not_fatal(tmp_path):
    """``read_text(encoding="utf-8")`` raises ``UnicodeDecodeError``, which is a
    ``ValueError`` -- **not** an ``OSError`` -- so the ``except OSError`` in
    ``_parse`` does not catch it and it escapes ``list()``.

    This is the same "one dirty file hides every skill" failure as BUG S1, via a
    different exception. It is realistic: a skill file saved as Latin-1 by an
    editor, or a stray binary byte.

    Observed::

        provider.list() raised builtins.UnicodeDecodeError
        isinstance(UnicodeDecodeError(), OSError) = False

    Fix: widen the read guard in ``_parse`` (and ``load``) to cover both::

        except (OSError, UnicodeDecodeError) as exc:
            raise SkillError(f"could not read {path}: {exc}") from exc
    """
    root = tmp_path / "skills"
    write_skill(root, "good", "---\nname: good\ndescription: a fine skill\n---\nbody\n")
    (root / "latin").mkdir(parents=True)
    # Valid frontmatter, then a byte that is not valid UTF-8.
    (root / "latin" / "SKILL.md").write_bytes(
        b"---\nname: latin\ndescription: a fine skill\n---\nbody \xe9\xff\n"
    )

    provider = provider_for(root)
    try:
        names = [c.summary.name for c in provider.list()]
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{type(exc).__name__} escaped list(): {exc}") from exc

    assert names == ["good"], "a non-UTF-8 file hid every other skill"


# ---------------------------------------------------------------------------
# BUG S3 -- an indented '---' inside a YAML block scalar ends the frontmatter
# ---------------------------------------------------------------------------


def test_an_indented_fence_inside_a_block_scalar_does_not_truncate_frontmatter(tmp_path):
    """``_split_frontmatter`` looks for any line whose ``strip()`` is ``---``.
    Inside a YAML block scalar (``description: |``) an indented ``---`` is
    ordinary prose -- a Markdown horizontal rule in the description -- but the
    splitter treats it as the closing fence.

    The frontmatter is truncated at that line, so every key *after* it is
    silently lost. A ``disable-model-invocation: true`` written below the rule is
    dropped, the skill stays model-invocable, and the leftover YAML is delivered
    to the model as part of the body.

    Observed::

        description parsed : 'Deploy it.'
        model_invocable    : True     <-- author wrote disable-model-invocation: true
        body delivered     : 'Then stop.\\ndisable-model-invocation: true\\n---\\n\\nReal body.'

    Fix: only treat a ``---`` line as the closing fence when it is **not
    indented** -- a block scalar's content is always indented under its key::

        if lines[index].rstrip() == "---":   # instead of lines[index].strip() == "---"

    (The opening fence keeps its ``strip()`` test so a BOM or stray leading
    whitespace is still tolerated.)
    """
    root = tmp_path / "skills"
    write_skill(
        root,
        "doc",
        "---\n"
        "name: doc\n"
        "description: |\n"
        "  Deploy it.\n"
        "  ---\n"
        "  Then stop.\n"
        "disable-model-invocation: true\n"
        "---\n"
        "\n"
        "Real body.\n",
    )

    candidates = provider_for(root).list()
    assert len(candidates) == 1, "the skill should still be discovered"
    candidate = candidates[0]

    assert candidate.summary.model_invocable is False, (
        "`disable-model-invocation: true` was silently dropped because the "
        "frontmatter was truncated at an indented '---'"
    )
    assert "Then stop." in candidate.summary.description, (
        "the description was truncated at the indented fence: "
        f"{candidate.summary.description!r}"
    )


def test_frontmatter_never_leaks_into_the_delivered_body(tmp_path):
    """The body must not leak YAML into the model's context.

    ``tests/test_skills.py::test_frontmatter_is_not_part_of_the_body`` asserts
    this for a well-formed file. It does not hold when a block scalar contains an
    indented ``---``: everything after the false fence -- including the rest of
    the frontmatter -- becomes the body.

    Observed::

        body delivered: 'Then stop.\\ndisable-model-invocation: true\\n---\\n\\nReal body.'

    Fix: the same non-indented fence test as BUG S3.
    """
    root = tmp_path / "skills"
    write_skill(
        root,
        "doc",
        "---\n"
        "name: doc\n"
        "description: |\n"
        "  Deploy it.\n"
        "  ---\n"
        "  Then stop.\n"
        "disable-model-invocation: true\n"
        "---\n"
        "\n"
        "Real body.\n",
    )

    registry = registry_for(root)
    definition = registry.get("doc")
    assert definition is not None
    assert "disable-model-invocation" not in definition.content, (
        f"frontmatter leaked into the body: {definition.content!r}"
    )


# ---------------------------------------------------------------------------
# BUG S4 -- the digest separator is forgeable, so a real change goes unpublished
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG S4: the digest joins entries with a raw \\x1f, so one skill's description can forge a boundary and collide with two skills",
)
def test_two_different_catalogues_cannot_share_a_digest(tmp_path):
    """``catalog_digest`` builds its payload as ``name\\x1fdescription`` joined by
    newlines. Both separators are ordinary characters that a **description may
    contain**, so two genuinely different catalogues can hash identically.

    ``\\x1f`` is reachable from a real file: YAML's double-quoted style supports
    ``\\x1f`` as an escape, and a literal ``\\n`` in a description supplies the
    other separator.

    Observed -- a one-skill catalogue and a two-skill catalogue, same digest::

        A skills: [('alpha', 'one\\ntwo\\x1fthree')]
        B skills: [('alpha', 'one'), ('two', 'three')]
        A digest: 923fb569e1b09cdb
        B digest: 923fb569e1b09cdb
        COLLIDE: True

    ``CatalogPublisher.update`` then suppresses the replacement, so the model
    keeps a catalogue that does not match the registry -- a real change goes
    unpublished, which is exactly what the digest exists to prevent.

    Fix: make the payload unambiguous instead of relying on separator characters
    that can appear in the data -- hash a canonical encoding of the entries::

        import json
        payload = json.dumps(
            [[s.name, s.description] for s in snapshot.skills],
            ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    JSON escapes both ``\\x1f`` and newlines inside strings, so no description can
    forge an entry boundary.
    """
    registry = SkillRegistry()

    one_skill = CatalogSnapshot(
        [SkillSummary(name="alpha", description="one\ntwo\x1fthree")]
    )
    two_skills = CatalogSnapshot(
        [
            SkillSummary(name="alpha", description="one"),
            SkillSummary(name="two", description="three"),
        ]
    )

    assert render_catalog(list(one_skill.skills), first=True) != render_catalog(
        list(two_skills.skills), first=True
    ), "the two catalogues must genuinely differ"

    assert registry.catalog_digest(one_skill) != registry.catalog_digest(two_skills), (
        "two different catalogues produced the same digest; a real change can go "
        "unpublished"
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG S4b: the collision is reachable from a real skill file, and it suppresses a republish",
)
def test_a_forged_separator_in_a_description_cannot_suppress_a_republish(tmp_path):
    """BUG S4 reached through the real filesystem provider.

    Observed::

        parsed description: 'line1\\nbeta\\x1fdesc'
        digest from a REAL file:  59e936e7edf9b90b
        digest of an innocent 2-skill catalogue: 59e936e7edf9b90b
        COLLIDE? True
        publisher.update(B, same digest) -> None   (nothing published)

    Fix: the canonical JSON encoding from BUG S4.
    """
    root = tmp_path / "skills"
    # YAML double-quoted style decodes \n and \x1f into real characters.
    write_skill(root, "alpha", '---\nname: alpha\ndescription: "line1\\nbeta\\x1fdesc"\n---\nbody\n')

    registry = registry_for(root)
    from_file = registry.snapshot()
    assert [s.name for s in from_file.skills] == ["alpha"]
    assert from_file.skills[0].description == "line1\nbeta\x1fdesc", (
        "the YAML escapes should decode to real separator characters"
    )

    innocent = CatalogSnapshot(
        [
            SkillSummary(name="alpha", description="line1"),
            SkillSummary(name="beta", description="desc"),
        ]
    )

    assert registry.catalog_digest(from_file) != registry.catalog_digest(innocent), (
        "a skill file forged an entry boundary in the digest"
    )


# ---------------------------------------------------------------------------
# BUG S5 -- flipping a skill to non-model-invocable does not change the digest
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG S5: the digest covers every skill, but publishing compares only the model-visible subset, so a visibility change is invisible to the digest",
)
def test_making_a_skill_non_model_invocable_republishes_the_catalogue(tmp_path):
    """``loop._publish_catalog`` computes ``digest = catalog_digest(snapshot)``
    over **all** skills and hands ``visible = model_visible(snapshot)`` to
    ``CatalogPublisher.update``. The publisher compares that digest and, when it
    matches, returns ``None``.

    Adding ``disable-model-invocation: true`` to a skill changes ``visible`` but
    **not** the digest, so the model is never told the skill disappeared. It
    keeps reading a catalogue that offers a skill the tool will now refuse.

    Observed::

        before flip: visible=['flip']  digest=d9c889e7ef85c26d  published=True
        after  flip: visible=[]        digest=d9c889e7ef85c26d  published=False
        skill tool "flip" -> Error: skill "flip" cannot be loaded by the model

    Fix: make the digest reflect what is actually published. Either digest the
    *visible* subset in ``_publish_catalog``::

        visible = self.skills.model_visible(snapshot)
        digest = self.skills.catalog_digest(CatalogSnapshot(visible))

    or have ``catalog_digest`` include each entry's ``model_invocable`` flag. The
    first is smaller and matches the publisher's own contract.
    """
    root = tmp_path / "skills"
    write_skill(root, "flip", "---\nname: flip\ndescription: a flippable skill\n---\nbody\n")

    registry = registry_for(root)
    publisher = CatalogPublisher()

    before = registry.snapshot()
    visible_before = registry.model_visible(before)
    assert [s.name for s in visible_before] == ["flip"]
    assert publisher.update(before, visible_before, registry.catalog_digest(before)) is not None

    # The author disables model invocation. The entry set is unchanged...
    write_skill(
        root,
        "flip",
        "---\nname: flip\ndescription: a flippable skill\n"
        "disable-model-invocation: true\n---\nbody\n",
    )
    after = registry.snapshot()
    visible_after = registry.model_visible(after)
    assert visible_after == (), "the skill should no longer be model-visible"

    # ...but the model must still be told that the catalogue changed.
    republished = publisher.update(after, visible_after, registry.catalog_digest(after))
    assert republished is not None, (
        "the skill silently vanished from the model's catalogue: the digest did "
        "not change, so no replacement was published"
    )


# ---------------------------------------------------------------------------
# BUG S6 -- the documented cross-provider tie-break is inert
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG S6: `(candidate.rank, -order) < (existing.rank, -order)` uses the same `order` on both sides, so the tie-break term cancels and the FIRST provider always wins",
)
def test_a_later_provider_overrides_an_equal_rank_earlier_one(tmp_path):
    """``SkillRegistry.snapshot``'s docstring says "Across providers, the later
    registration wins -- which is how a project provider overrides a user one".

    The comparison is ``(candidate.rank, -order) < (existing.rank, -order)``.
    ``order`` is the loop variable for the *current* provider, and the tuple on
    the right was built with the *same* value, so ``-order`` cancels: the test
    reduces to ``candidate.rank < existing.rank``. On an equal rank that is
    ``False``, so the **first** provider always wins.

    Observed (register A, then B, same rank)::

        catalogue says 'from provider A'   (docstring: later = B wins)

    Nothing in ``src/`` registers two providers today, so this is reachable only
    from a hand-constructed registry -- but it is a public extension point, and
    the docstring is the contract.

    Fix: compare against the order the *existing* candidate was registered with.
    Carry it alongside the candidate::

        chosen: dict[str, tuple[int, SkillCandidate]] = {}   # (order, candidate)
        ...
        if (candidate.rank, -order) < (existing.rank, -existing_order):
            chosen[name] = (order, candidate)
    """
    root = tmp_path
    a_dir = root / "A"
    b_dir = root / "B"
    write_skill(a_dir, "shared", "---\nname: shared\ndescription: from provider A\n---\nbody\n")
    write_skill(b_dir, "shared", "---\nname: shared\ndescription: from provider B\n---\nbody\n")

    registry = SkillRegistry()
    registry.register_provider(
        FilesystemSkillProvider([(PROJECT_RANK, "project", a_dir)], name="A")
    )
    registry.register_provider(
        FilesystemSkillProvider([(PROJECT_RANK, "project", b_dir)], name="B")
    )

    snapshot = registry.snapshot()
    assert snapshot.skills[0].description == "from provider B", (
        "the later provider did not win the rank tie; the documented tie-break "
        "is inert"
    )


# ---------------------------------------------------------------------------
# BUG S7 -- load() returns the candidate's stale summary
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG S7: `load()` re-parses only the body and returns `candidate.summary`, so tool.py's post-load policy check compares an object with itself",
)
def test_load_reports_the_policy_the_file_currently_declares(tmp_path):
    """``skills/tool.py`` documents its second policy check as the defence against
    a file edited between discovery and load:

        "Re-checked after loading. A file edited between discovery and load could
         have changed its own policy, and honoring the summary alone would make
         the policy a race condition rather than a rule."

    ``FilesystemSkillProvider.load`` calls ``_split_frontmatter`` and then
    **discards** the parsed frontmatter, returning ``candidate.summary``
    unchanged. ``tool.py`` then tests ``definition.summary.model_invocable`` --
    the same object it already tested. The check is a self-comparison and can
    never fire.

    Observed::

        load(candidate).summary is candidate.summary: True
        file now says disable-model-invocation:true
        load(candidate).summary.model_invocable: True    <-- should be False

    Fix: re-derive the summary from the freshly parsed frontmatter in ``load``
    (reuse ``_parse`` on the same path, or re-parse the policy and rebuild the
    summary), so the returned definition reflects the file as it is now.
    """
    root = tmp_path / "skills"
    path = write_skill(
        root, "alpha", "---\nname: alpha\ndescription: the alpha skill\n---\nALPHA BODY\n"
    )

    provider = provider_for(root)
    candidate = provider.list()[0]
    assert candidate.summary.model_invocable is True

    # The file flips itself to non-model-invocable before it is loaded.
    path.write_text(
        "---\nname: alpha\ndescription: the alpha skill\n"
        "disable-model-invocation: true\n---\nBETA BODY\n",
        encoding="utf-8",
    )

    definition = provider.load(candidate)
    assert definition is not None
    assert definition.summary.model_invocable is False, (
        "load() returned the stale summary, so the post-load policy check cannot "
        "detect a file that changed its own policy"
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG S7b: `get()`'s post-load name check and policy check both compare the stale candidate summary with itself, so neither can fire",
)
def test_get_refuses_a_body_that_renamed_itself_between_list_and_load(tmp_path):
    """The narrow window inside ``SkillRegistry.get``.

    NOTE ON SCOPE -- this is deliberately *not* the "the skill tool delivers a
    forbidden body" scenario. ``SkillTool.run`` calls ``snapshot()`` and then
    ``get()``, and ``get()`` re-lists, so a file edited between those two calls is
    seen by the fresh listing and the tool refuses correctly. I verified that
    before writing this test; the tool-level claim would have been a false
    positive.

    The reachable defect is the window *inside* ``get``: between its own
    ``provider.list()`` and its ``provider.load(best)``. ``load`` returns
    ``candidate.summary`` verbatim, so both of ``get``'s post-load guards are
    self-comparisons:

    * ``definition.summary.name != name`` -- the summary was selected *because*
      its name equals ``name``, so this is always ``False``;
    * ``definition.summary.model_invocable`` -- the same stale object the caller
      already checked.

    Observed (provider flips the file between ``list`` and ``load``)::

        get('alpha') -> summary.name='alpha', model_invocable=True, content='BETA'
        file now says name: beta, disable-model-invocation: true

    The body of a file that now declares a different name, and forbids model
    invocation, is served under the old name.

    Fix: the same re-derivation as BUG S7 -- ``load`` must rebuild the summary
    from the frontmatter it has already parsed.
    """
    root = tmp_path / "skills"
    path = write_skill(
        root, "alpha", "---\nname: alpha\ndescription: the alpha skill\n---\nALPHA\n"
    )

    class Straddling(FilesystemSkillProvider):
        """Flips the file in the window between ``list`` and ``load``."""

        def list(self):
            candidates = super().list()
            path.write_text(
                "---\nname: beta\ndescription: the beta skill\n"
                "disable-model-invocation: true\n---\nBETA\n",
                encoding="utf-8",
            )
            return candidates

    registry = SkillRegistry()
    registry.register_provider(Straddling([(PROJECT_RANK, "project", root)]))

    definition = registry.get("alpha")
    assert definition is None or definition.summary.name != "alpha", (
        "get() served a body whose file now declares a different name and forbids "
        "model invocation; both post-load guards compared the stale summary with "
        "itself"
    )


# ---------------------------------------------------------------------------
# BUG S8 -- the PyYAML fallback parser disagrees with the real parser
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG S8: `_parse_simple_frontmatter` is only reachable when PyYAML is absent, so its divergence from `safe_load` is never exercised",
)
def test_the_fallback_parser_agrees_with_pyyaml_on_block_scalars(tmp_path):
    """``_parse_yaml`` uses ``yaml.safe_load`` when PyYAML is importable and
    ``_parse_simple_frontmatter`` otherwise. The fallback is therefore **dead in
    any environment with PyYAML installed** (it is a hard dependency in
    ``pyproject.toml``), so no test exercises it.

    The two disagree on constructs a real skill may use. Observed::

        yaml   block scalar | : {'description': 'line one\\nline two'}
        simple block scalar | : {'description': '|'}
        yaml   folded scalar >: {'description': 'line one line two'}
        simple folded scalar >: {'description': '>'}
        yaml   nested yaml    : {'name': 'a', 'description': 'ddd', 'meta': {'a': 1}}
        simple nested yaml    : {'name': 'a', 'description': 'ddd', 'meta': '', 'a': '1'}

    With the fallback, ``description: |`` becomes the literal string ``"|"`` --
    which is shorter than ``MIN_DESCRIPTION_CHARS`` for a single-line block, so a
    valid skill is **rejected**; for a longer literal it becomes a nonsense
    description that misroutes the model.

    Fix: either make the fallback reject what it cannot faithfully parse (raise
    ``SkillError`` on a value that is exactly ``|`` or ``>``, or on any indented
    continuation line), or drop the fallback and require PyYAML -- it is already
    a declared dependency, so "PyYAML is missing" is not a supported state.
    """
    raw = "name: a\ndescription: |\n  line one\n  line two"
    assert _parse_yaml(raw) == _parse_simple_frontmatter(raw), (
        "the fallback parser disagrees with PyYAML on a block scalar"
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG S8b: a block scalar read by the fallback produces a description that is not the author's text",
)
def test_the_fallback_parser_does_not_invent_a_description(tmp_path):
    """The user-visible consequence of BUG S8.

    Observed::

        simple block scalar | : {'description': '|'}

    ``"|"`` is one character, below ``MIN_DESCRIPTION_CHARS``, so the skill is
    silently dropped; a multi-line description would instead be truncated to its
    first line.

    Fix: as BUG S8.
    """
    fallback = _parse_simple_frontmatter("name: a\ndescription: |\n  a real description")
    assert fallback.get("description") != "|", (
        "the fallback parser turned a block scalar into the literal '|'"
    )


# ---------------------------------------------------------------------------
# BUG S9 -- duplicate frontmatter keys are silently resolved
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG S9: duplicate frontmatter keys are silently last-wins, so a skill can declare one description and be catalogued under another",
)
def test_duplicate_frontmatter_keys_are_refused(tmp_path):
    """``yaml.safe_load`` accepts duplicate mapping keys and keeps the last. The
    frontmatter contract is strict everywhere else (an unknown legacy key is a
    hard ``SkillError``), so a duplicated key silently overriding its sibling is
    inconsistent -- and the author has no way to see which value won.

    Observed::

        yaml duplicate keys: {'name': 'b', 'description': 'two'}

    Fix: load with a mapping constructor that rejects duplicates, e.g.::

        class _StrictLoader(yaml.SafeLoader):
            pass

        def _no_duplicates(loader, node, deep=False):
            seen = set()
            for key_node, _value in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in seen:
                    raise SkillError(f"duplicate frontmatter key {key!r}")
                seen.add(key)
            return yaml.SafeLoader.construct_mapping(loader, node, deep)

        _StrictLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
        )
    """
    raw = "name: alpha\ndescription: the first description\nname: beta\ndescription: the second"
    try:
        parsed = _parse_yaml(raw)
    except SkillError:
        return  # refused, which is the desired behaviour
    pytest.fail(
        f"duplicate keys were silently resolved to {parsed!r} instead of refused"
    )


# ---------------------------------------------------------------------------
# BUG S10 -- `_split_frontmatter` accepts a document with no closing fence
#            in a way that loses the body, and mis-handles a lone `---`
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG S10: `_split_frontmatter` returns `body=''` for a file with frontmatter and no body, and `load()` then returns an empty body rather than refusing",
)
def test_a_skill_with_no_body_is_reported_rather_than_silently_empty(tmp_path):
    """``_split_frontmatter`` returns ``(frontmatter, "")`` when the closing fence
    is the last line. ``load()`` accepts that, so the tool injects a skill whose
    content is empty -- the model is told to follow instructions that are not
    there, with no indication anything is wrong.

    Observed::

        _split_frontmatter("---\\nname: a\\n---") -> ({'name': 'a'}, '')
        load(...).content -> ''

    Fix: treat an empty body as a contract violation in ``load`` (or ``_parse``),
    e.g. ``if body is None or not body.strip(): return None`` so the tool reports
    "listed but its content could not be read" instead of injecting nothing.
    """
    root = tmp_path / "skills"
    write_skill(root, "empty", "---\nname: empty\ndescription: has no body at all\n---\n")

    registry = registry_for(root)
    definition = registry.get("empty")
    assert definition is not None, "the skill should be discoverable"
    assert definition.content.strip() != "", (
        "an empty skill body was accepted and would be injected as instructions"
    )


# ---------------------------------------------------------------------------
# BUG S11 -- the resource_base for a bundle points at the skill directory,
#            which `render_skill_content` emits unescaped
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG S11: `render_skill_content` escapes the skill name but interpolates the resource path raw, so a path containing markup breaks the envelope",
)
def test_the_resource_base_is_escaped_in_the_rendered_skill(tmp_path):
    """``render_skill_content`` escapes ``definition.name`` with ``escape_xml``
    but interpolates ``definition.resource_base`` verbatim. The base is a
    filesystem path, and a directory name may legally contain ``<`` or ``>``.

    Observed: a skill under a directory named ``a<b>c`` renders a header block
    containing raw angle brackets inside ``<skill>``.

    Fix: ``escape_xml(definition.resource_base)``.
    """
    from arm.skills.catalog import render_skill_content
    from arm.skills.registry import SkillDefinition

    definition = SkillDefinition(
        summary=SkillSummary(name="s", description="d"),
        content="body",
        resource_base="/tmp/skills/a<b>c",
    )
    rendered = render_skill_content(definition)
    assert "<b>" not in rendered, (
        f"the resource base was not escaped and can inject markup: {rendered[:160]!r}"
    )

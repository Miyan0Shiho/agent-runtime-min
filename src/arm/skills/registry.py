"""Skills: procedural knowledge the model loads on demand.

A skill is a Markdown file with frontmatter, discovered from a set of roots, that
the model can load by name. The design is built around one idea:

    **progressive disclosure.** The catalogue is cheap; the body is on demand.

The catalogue gives the model a name and a description per skill -- enough to
decide whether to load one. The body is only fetched when the model actually
asks. This is what keeps a hundred installed skills from costing a hundred
bodies' worth of context.

Two consequences follow, and both are enforced here rather than left to
convention:

* **The catalogue carries name and description only.** Any field added to it is
  multiplied by the number of skills, and routing decisions barely use more than
  these two. A skill's path, source, provider and body never appear.
* **The body is not cached.** Editing a skill file takes effect on the next load.
  That matters because the author of a skill is usually the person the runtime is
  helping, and "edit, then immediately try" is the loop they are in.

Failure is localized deliberately: an unreadable root or one malformed file
degrades to a warning and a partial catalogue, because a dirty workspace must not
stop the agent from working. A candidate that violates the provider contract --
a malformed name, a missing description -- fails fast instead, because that is a
bug in the skill rather than a transient condition.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..kernel.errors import ArmError

log = logging.getLogger("arm.skills")

#: Skill names are kebab-case. Restrictive on purpose: the name is what the model
#: types to invoke the skill, and a name with spaces or capitals invites typos
#: that fail in ways the model cannot diagnose.
SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: How much of a description reaches the catalogue. Descriptions are the only
#: per-skill cost that scales with the number of installed skills, so they are
#: bounded rather than trusted to be short.
MAX_DESCRIPTION_CHARS = 500
MIN_DESCRIPTION_CHARS = 3

#: The gesture a user types to invoke a skill directly.
SKILL_GESTURE = re.compile(r"(^|\s)/([a-z0-9]+(?:-[a-z0-9]+)*)(?=\s|$)")


class SkillError(ArmError):
    """A skill violates the provider contract. Fails fast."""


@dataclass(frozen=True, slots=True)
class InvocationPolicy:
    """Who may invoke a skill.

    Two independent positive booleans rather than a passthrough of frontmatter.
    The four combinations are all meaningful:

    * ``{model: True,  user: True}``  -- the default;
    * ``{model: True,  user: False}`` -- the model may load it, the user has no
      gesture for it;
    * ``{model: False, user: True}``  -- only reachable by explicit ``/name``,
      which is how a skill that should not be auto-selected is expressed;
    * ``{model: False, user: False}`` -- unreachable by either; only a trusted
      programmatic caller can load it.
    """

    model: bool = True
    user: bool = True

    @staticmethod
    def parse(raw: Mapping[str, Any], *, name: str) -> "InvocationPolicy":
        """Read the policy from frontmatter.

        The legacy spellings are rejected rather than ignored. A user who writes
        ``disableModelInvocation`` and is silently ignored believes they have
        disabled something they have not -- which is a security-relevant
        misunderstanding, not a typo.
        """
        legacy = {
            "disableModelInvocation": "disable-model-invocation",
            "modelInvocable": "disable-model-invocation (inverted)",
            "userInvocable": "user-invocable",
        }
        for old, new in legacy.items():
            if old in raw:
                raise SkillError(
                    f"skill {name!r}: frontmatter key {old!r} is not recognized; "
                    f"use {new!r} instead"
                )

        model = not _as_bool(raw.get("disable-model-invocation"), default=False, name=name)
        user = _as_bool(raw.get("user-invocable"), default=True, name=name)
        return InvocationPolicy(model=model, user=user)


def _as_bool(value: Any, *, default: bool, name: str) -> bool:
    """Parse a frontmatter boolean, accepting the spellings YAML produces."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    raise SkillError(
        f"skill {name!r}: expected a boolean, got {value!r}"
    )


@dataclass(frozen=True, slots=True)
class SkillSummary:
    """What the catalogue carries: a name and a description.

    Deliberately small. ``when_to_use`` is kept out of the catalogue rendering
    because it is not needed to decide *whether* to load a skill -- the
    description is -- and every field here is paid for once per skill.
    """

    name: str
    description: str
    when_to_use: str = ""
    policy: InvocationPolicy = field(default_factory=InvocationPolicy)
    source: str = ""

    @property
    def model_invocable(self) -> bool:
        return self.policy.model

    @property
    def user_invocable(self) -> bool:
        return self.policy.user


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """A loaded skill: its summary plus its body."""

    summary: SkillSummary
    content: str
    #: Where relative resource paths in the body resolve from.
    resource_base: str = ""

    @property
    def name(self) -> str:
        return self.summary.name


@dataclass(frozen=True, slots=True)
class SkillCandidate:
    """A summary plus the information needed to load its body later.

    The locator is opaque to everything except the provider that produced it, so
    how skills are stored never becomes part of the registry's contract.
    """

    summary: SkillSummary
    locator: str
    rank: int = 500


class SkillProvider:
    """A source of skills.

    Providers are the extension point: the filesystem is one, but a bundled set,
    a network source or a database would be others. The registry only resolves
    conflicts and serves consumers; it never scans anything itself.
    """

    name: str = "provider"

    def list(self) -> Sequence[SkillCandidate]:  # pragma: no cover - interface
        raise NotImplementedError

    def load(self, candidate: SkillCandidate) -> SkillDefinition | None:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Filesystem provider
# ---------------------------------------------------------------------------

#: Roots, in priority order. Lower rank wins within a layer, so a project skill
#: overrides a user skill of the same name -- which is what "project-local
#: configuration beats global configuration" means in practice.
PROJECT_RANK = 100
USER_RANK = 400
CUSTOM_RANK = 300
BUNDLED_RANK = 600


def find_project_root(start: Path) -> Path:
    """The nearest ancestor containing ``.git``, or ``start`` itself.

    Anchoring on the VCS root rather than the process's working directory means a
    skill found from a subdirectory is the same skill found from the top.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


class FilesystemSkillProvider(SkillProvider):
    """Discovers skills from directories on disk.

    Two layouts are supported, and only one level is scanned:

    * ``<root>/<name>/SKILL.md`` -- a directory bundle, so the skill can ship
      resources beside it;
    * ``<root>/<name>.md`` -- a single file.

    Nested discovery is deliberately absent. ``**/SKILL.md`` would let an
    unrelated file deep in a repository become an invocable instruction source,
    which is a surprising thing for a search to do.
    """

    def __init__(
        self,
        roots: Sequence[tuple[int, str, Path]],
        *,
        name: str = "filesystem",
    ) -> None:
        #: ``(rank, source_label, path)`` triples.
        self.roots = list(roots)
        self.name = name

    @staticmethod
    def default_roots(
        *,
        cwd: Path | None = None,
        home: Path | None = None,
        extra: Sequence[Path] = (),
        bundled: Path | None = None,
    ) -> list[tuple[int, str, Path]]:
        """The standard root set, in priority order."""
        roots: list[tuple[int, str, Path]] = []
        base = Path(cwd) if cwd else Path.cwd()
        project = find_project_root(base)

        roots.append((PROJECT_RANK, "project", project / ".arm" / "skills"))
        roots.append((PROJECT_RANK + 1, "project-agents", project / ".agents" / "skills"))
        for index, path in enumerate(extra):
            roots.append((CUSTOM_RANK + index, "custom", Path(path)))

        home_dir = Path(home) if home else Path.home()
        roots.append((USER_RANK, "user", home_dir / ".arm" / "skills"))

        if bundled is not None:
            roots.append((BUNDLED_RANK, "bundled", Path(bundled)))
        return roots

    def list(self) -> Sequence[SkillCandidate]:
        """Discover candidates, skipping anything malformed.

        A broken skill is skipped with a warning rather than failing the scan:
        one bad file in a dirty workspace must not hide every other skill. The
        *contract* violations (a bad name, an empty description) are what raise
        inside :meth:`_parse`, and those are caught here per file.
        """
        candidates: list[SkillCandidate] = []
        for rank, source, root in self.roots:
            if not root.is_dir():
                continue
            try:
                entries = sorted(root.iterdir(), key=lambda p: p.name)
            except OSError as exc:
                # An unreadable root is a partial observation, not a failure.
                log.warning("skill root %s is not readable: %s", root, exc)
                continue

            for entry in entries:
                try:
                    candidate = self._candidate_for(entry, rank, source)
                except SkillError as exc:
                    log.warning("skipping skill %s: %s", entry, exc)
                    continue
                except Exception as exc:  # noqa: BLE001 - one file must not hide the rest
                    # A broader net than SkillError, deliberately. Discovery walks
                    # a directory a user controls, so an unexpected exception type
                    # is a real possibility -- and letting one escape would abort
                    # the whole scan, hiding every other skill in the workspace.
                    #
                    # A single malformed file is a local problem; it must stay
                    # local. The alternative is that one stray file makes the
                    # runtime appear to have no skills at all.
                    log.warning(
                        "skipping skill %s: unexpected %s: %s",
                        entry,
                        type(exc).__name__,
                        exc,
                    )
                    continue
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def _candidate_for(
        self, entry: Path, rank: int, source: str
    ) -> SkillCandidate | None:
        """Build a candidate from one directory entry, or ``None`` to skip it."""
        if entry.is_dir():
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                # A directory without a SKILL.md is not a skill bundle.
                return None
            return self._parse(skill_file, entry.name, rank, source, str(entry))

        if entry.suffix == ".md" and entry.is_file():
            return self._parse(entry, entry.stem, rank, source, str(entry.parent))

        return None

    def _parse(
        self, path: Path, expected_name: str, rank: int, source: str, base: str
    ) -> SkillCandidate:
        """Parse one skill file. Raises :class:`SkillError` on a contract breach."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillError(f"could not read {path}: {exc}") from exc

        front, body = _split_frontmatter(text)
        if front is None:
            raise SkillError(f"{path} has no frontmatter block")

        name = front.get("name")
        if not isinstance(name, str) or not name:
            raise SkillError(f"{path} is missing a 'name'")
        if not SKILL_NAME.match(name):
            raise SkillError(
                f"{path}: name {name!r} is not kebab-case "
                "(lowercase letters, digits and single hyphens)"
            )
        if name != expected_name and not path.name == "SKILL.md":
            # For a flat file the filename is the name; a mismatch would make the
            # skill unfindable by the name it claims.
            raise SkillError(
                f"{path}: name {name!r} does not match the filename "
                f"{expected_name!r}"
            )

        description = front.get("description")
        if not isinstance(description, str) or not description.strip():
            raise SkillError(f"{path}: 'description' must be a non-empty string")
        description = description.strip()
        if len(description) < MIN_DESCRIPTION_CHARS:
            raise SkillError(
                f"{path}: 'description' is too short to route on "
                f"(minimum {MIN_DESCRIPTION_CHARS} characters)"
            )
        if len(description) > MAX_DESCRIPTION_CHARS:
            description = description[: MAX_DESCRIPTION_CHARS - 1] + "…"

        policy = InvocationPolicy.parse(front, name=name)
        when = front.get("whenToUse")
        if when is not None and not isinstance(when, str):
            raise SkillError(f"{path}: 'whenToUse' must be a string")

        return SkillCandidate(
            summary=SkillSummary(
                name=name,
                description=description,
                when_to_use=(when or "").strip(),
                policy=policy,
                source=source,
            ),
            locator=str(path),
            rank=rank,
        )

    def load(self, candidate: SkillCandidate) -> SkillDefinition | None:
        """Read a skill's body. Never cached, so an edit takes effect at once."""
        path = Path(candidate.locator)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("could not load skill %s: %s", path, exc)
            return None

        _front, body = _split_frontmatter(text)
        if body is None:
            return None
        # The name must still match: a file edited between discovery and load
        # would otherwise be returned under the old name.
        return SkillDefinition(
            summary=candidate.summary,
            content=body.strip(),
            resource_base=str(path.parent),
        )


def _split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Split a Markdown document into frontmatter and body.

    Only a leading ``---`` block counts. Frontmatter is YAML, parsed with the
    standard library's parser via PyYAML when available; a document with no
    frontmatter returns ``(None, None)`` so the caller can report it as invalid
    rather than silently treating the whole file as a body.
    """
    # A byte-order mark is an encoding artefact, not content. Editors on Windows
    # and some generators emit one, and without stripping it the file appears to
    # start with something other than ``---`` -- so a perfectly valid skill would
    # be skipped with a message about missing frontmatter, which is both wrong and
    # unhelpful.
    text = text.lstrip("\ufeff")

    if not text.startswith("---"):
        return None, None
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, None

    # The closing fence is the first line that is *exactly* ``---`` with no
    # leading whitespace.
    #
    # An indented ``---`` is content, not a fence: inside a block scalar it is
    # just text. Treating it as the closing fence ends the frontmatter early,
    # which silently drops every key below it -- including
    # ``disable-model-invocation``. A skill the author deliberately protected
    # would become model-invocable, and the leftover YAML would leak into the
    # model's context as part of the body.
    for index in range(1, len(lines)):
        if lines[index].rstrip() == "---":
            raw = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            return _parse_yaml(raw), body
    return None, None


def _parse_yaml(raw: str) -> dict[str, Any]:
    """Parse the frontmatter block.

    PyYAML is used when present. A fallback parser handles the flat
    ``key: value`` shape, so a missing dependency degrades rather than disabling
    the feature entirely.
    """
    try:
        import yaml
    except ImportError:
        return _parse_simple_frontmatter(raw)

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        # PyYAML raises its own exception hierarchy (ScannerError, ParserError,
        # ConstructorError, ...). None of those is a ``SkillError``, so letting
        # one escape would abort the whole discovery pass -- and a single
        # malformed file would hide every other skill in the workspace.
        #
        # Normalizing it here is what keeps "failure is localized" true: the bad
        # file is skipped with a message, and the rest of the catalogue is
        # unaffected.
        raise SkillError(f"frontmatter is not valid YAML: {exc}") from exc

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise SkillError("frontmatter must be a mapping")
    return {str(k): v for k, v in parsed.items()}


def _parse_simple_frontmatter(raw: str) -> dict[str, Any]:
    """A minimal ``key: value`` parser, used only when PyYAML is unavailable.

    This is a **degraded** path, not an equivalent one, and it is important to be
    honest about that. It handles the flat ``key: value`` shape that skills
    overwhelmingly use, and it **refuses** the constructs it cannot parse
    correctly rather than guessing:

    * block scalars (``|`` and ``>``), which it would store as the literal
      characters ``"|"``;
    * anchors, aliases and merge keys, which it would store as text;
    * anything indented, which is a nested structure it cannot represent.

    Refusing matters because the alternative is a skill whose description is the
    string ``"|"`` -- which would be accepted, published to the catalogue, and
    silently wrong. A missing dependency should degrade the *feature*, not the
    *data*.
    """
    result: dict[str, Any] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1] in (" ", "\t"):
            raise SkillError(
                "nested frontmatter is not supported without PyYAML; install "
                "pyyaml, or keep the frontmatter flat"
            )
        if ":" not in stripped:
            raise SkillError(
                f"frontmatter line {stripped!r} is not a 'key: value' pair; "
                "install pyyaml for full support"
            )
        key, _, value = stripped.partition(":")
        text = value.strip()
        if text in ("|", ">", "|-", ">-", "|+", ">+"):
            raise SkillError(
                "block scalars are not supported without PyYAML; install pyyaml, "
                "or write the value on one line"
            )
        if text.startswith("&") or text.startswith("*") or text.startswith("<<"):
            raise SkillError(
                "YAML anchors and aliases are not supported without PyYAML; "
                "install pyyaml"
            )
        text = text.strip('"').strip("'")
        if text.lower() in ("true", "yes", "on"):
            result[key.strip()] = True
        elif text.lower() in ("false", "no", "off"):
            result[key.strip()] = False
        else:
            result[key.strip()] = text
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """The visible skill set, with a flag for whether it is authoritative.

    ``complete`` is what a caller checks before caching. A snapshot assembled
    while a provider was failing is not wrong, but it is *incomplete*, and
    publishing it would freeze a partial catalogue into a session's history.
    """

    skills: tuple[SkillSummary, ...] | Sequence[SkillSummary] = ()

    def __post_init__(self) -> None:
        # Accept any sequence and normalize to a tuple, so callers do not have to
        # remember which shape this particular field wants.
        if not isinstance(self.skills, tuple):
            object.__setattr__(self, "skills", tuple(self.skills))

    complete: bool = True

    def by_name(self, name: str) -> SkillSummary | None:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None


class SkillRegistry:
    """Resolves candidates from several providers into one visible set.

    Within a provider, lower ``rank`` wins. Across providers, the later
    registration wins -- which is how a project provider overrides a user one
    without either needing to know about the other.
    """

    def __init__(self) -> None:
        self._providers: list[SkillProvider] = []
        self._runtime: dict[str, SkillDefinition] = {}

    def register_provider(self, provider: SkillProvider) -> Any:
        """Add a provider. Returns a disposer, so registration is reversible."""
        self._providers.append(provider)

        def _dispose() -> None:
            if provider in self._providers:
                self._providers.remove(provider)

        return _dispose

    def register_skill(self, definition: SkillDefinition) -> Any:
        """Register a skill defined in code rather than on disk.

        Runtime skills outrank every provider: a skill the host explicitly
        installed is more specific than one discovered by scanning.
        """
        name = definition.summary.name
        self._runtime[name] = definition

        def _dispose() -> None:
            self._runtime.pop(name, None)

        return _dispose

    def snapshot(self) -> CatalogSnapshot:
        """Build the visible skill set.

        Resolution: runtime skills win outright; otherwise the candidate with the
        lowest rank wins, and a later provider breaks a rank tie.
        """
        chosen: dict[str, SkillCandidate] = {}
        complete = True

        for order, provider in enumerate(self._providers):
            try:
                candidates = list(provider.list())
            except Exception as exc:  # noqa: BLE001 - a provider must not break the set
                # A provider that fails yields a partial observation. Reporting
                # it as complete would let a caller cache half a catalogue.
                log.warning("skill provider %r failed: %s", provider.name, exc)
                complete = False
                continue

            for candidate in candidates:
                name = candidate.summary.name
                existing = chosen.get(name)
                if existing is None:
                    chosen[name] = candidate
                    continue
                if (candidate.rank, -order) < (existing.rank, -order):
                    chosen[name] = candidate

        summaries: dict[str, SkillSummary] = {
            name: candidate.summary for name, candidate in chosen.items()
        }
        for name, definition in self._runtime.items():
            summaries[name] = definition.summary

        ordered = tuple(sorted(summaries.values(), key=lambda s: s.name))
        return CatalogSnapshot(skills=ordered, complete=complete)

    def get(self, name: str) -> SkillDefinition | None:
        """Load a skill by name.

        Deliberately uncached. Reading the file on every load is what makes
        "edit the skill, try again" work, and skill loads are rare enough that
        the read costs nothing measurable.
        """
        if name in self._runtime:
            return self._runtime[name]

        best: SkillCandidate | None = None
        for provider in self._providers:
            try:
                candidates = list(provider.list())
            except Exception:  # noqa: BLE001
                continue
            for candidate in candidates:
                if candidate.summary.name != name:
                    continue
                if best is None or candidate.rank < best.rank:
                    best = candidate

        if best is None:
            return None

        definition: SkillDefinition | None = None
        for provider in self._providers:
            try:
                definition = provider.load(best)
            except Exception:  # noqa: BLE001
                continue
            if definition is not None:
                break

        if definition is None:
            return None

        # The loaded body must still carry the name that was asked for. A file
        # edited between discovery and load could otherwise be served under a
        # name it no longer claims.
        if definition.summary.name != name:
            log.warning(
                "skill %s loaded under a different name (%s); refusing",
                name,
                definition.summary.name,
            )
            return None
        return definition

    def catalog_digest(self, snapshot: CatalogSnapshot) -> str:
        """A stable fingerprint of the catalogue's *entries*.

        Computed from the structured entries, not from the rendered text. If the
        digest were taken over the rendered Markdown, changing one word of
        wrapper prose would make every session resend its whole catalogue and
        destroy prompt-prefix caching.
        """
        # A structured encoding, not a hand-rolled one.
        #
        # Concatenating fields with a separator is forgeable: a description
        # containing the separator and a newline can produce exactly the same
        # string as two separate entries, so a real change would hash identically
        # and never be published. ``json.dumps`` escapes the characters that
        # would otherwise be ambiguous, which removes the class of collision
        # rather than making it harder to hit.
        payload = json.dumps(
            [[skill.name, skill.description] for skill in snapshot.skills],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def model_visible(self, snapshot: CatalogSnapshot) -> tuple[SkillSummary, ...]:
        """The subset the model may load.

        The catalogue and the loader consult the same predicate, so the model can
        never see a skill it cannot load, and never fail to see one it can.
        """
        return tuple(s for s in snapshot.skills if s.model_invocable)

    def find_gesture(self, text: str, snapshot: CatalogSnapshot) -> tuple[SkillSummary, ...]:
        """Skills named by a ``/name`` gesture in ``text``.

        This is the only route to a skill that is not model-invocable, which is
        what makes ``disable-model-invocation`` meaningful: the skill exists, the
        model cannot select it, and the user can still ask for it.
        """
        found: list[SkillSummary] = []
        for _prefix, name in SKILL_GESTURE.findall(text):
            summary = snapshot.by_name(name)
            if summary is not None and summary.user_invocable and summary not in found:
                found.append(summary)
        return tuple(found)

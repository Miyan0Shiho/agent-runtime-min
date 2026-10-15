"""Rendering the skill catalogue, and the ``skill`` tool that loads one.

The catalogue is a *message*, not a system-prompt segment. That choice matters
for two reasons:

* The system prompt is a single string that is re-recorded only when it changes.
  A catalogue that changes as skills are installed would therefore either
  invalidate the prompt constantly or silently go stale.
* A message carries its own provenance in the log. The catalogue's entries are
  recorded with it, so a non-model consumer can read what was published without
  parsing the rendered text.

The catalogue is also **incremental**: it is published when it first becomes
non-empty, and replaced when its entries change. "Changed" is decided by a digest
over the structured entries, never over the rendered prose -- otherwise editing
one word of the wrapper would make every session resend the whole catalogue and
destroy prompt-prefix caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .registry import CatalogSnapshot, SkillDefinition, SkillSummary

#: The wrapper the catalogue is delivered in. A ``system-reminder`` reads as
#: runtime guidance rather than as something the user said.
_ENVELOPE_OPEN = "<system-reminder>"
_ENVELOPE_CLOSE = "</system-reminder>"

#: The catalogue's own element name, so a reader can find it structurally.
CATALOG_TAG = "available_skills"

#: The first catalogue of a session.
FIRST_CATALOG_PREAMBLE = (
    "The following skills are available in this session. This catalogue lists "
    "summaries only: do not infer or follow a skill's instructions until you "
    "have loaded it."
)

#: A catalogue that supersedes an earlier one.
REPLACEMENT_PREAMBLE = (
    "This is the complete list of available skills. It replaces every previous "
    "available-skills list; use only the names in this catalogue."
)

#: An explicitly empty catalogue, which must say so rather than be omitted.
EMPTY_PREAMBLE = (
    "No skills are currently available through the skill tool. Do not use names "
    "from an earlier catalogue."
)

#: Reminds the model not to reload what the user already invoked directly.
NO_RELOAD_NOTE = (
    "A skill the user invoked explicitly with /name is already loaded; do not "
    "load it again with the skill tool."
)


def escape_xml(text: str) -> str:
    """Escape text for inclusion in the catalogue envelope.

    A description is author-supplied and may contain angle brackets. Without
    escaping, a description could close the envelope early and inject text that
    reads as runtime guidance rather than as a skill summary.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_catalog(
    skills: Sequence[SkillSummary],
    *,
    first: bool,
    user_invoked: Sequence[str] = (),
) -> str:
    """Render a catalogue message.

    An empty catalogue still renders, and says so explicitly. Omitting it would
    leave the model holding a stale list from earlier in the conversation, which
    is worse than telling it there is nothing.
    """
    if not skills:
        return f"{_ENVELOPE_OPEN}\n{EMPTY_PREAMBLE}\n{_ENVELOPE_CLOSE}"

    preamble = FIRST_CATALOG_PREAMBLE if first else REPLACEMENT_PREAMBLE
    lines = [f"{_ENVELOPE_OPEN}", preamble, f"<{CATALOG_TAG}>"]

    for skill in skills:
        lines.append(f"- {escape_xml(skill.name)}: {escape_xml(skill.description)}")

    lines.append(f"</{CATALOG_TAG}>")
    if user_invoked:
        lines.append(NO_RELOAD_NOTE)
    lines.append(_ENVELOPE_CLOSE)
    return "\n".join(lines)


def render_skill_content(definition: SkillDefinition) -> str:
    """Render a loaded skill for injection.

    The resource base is included so relative paths in the body resolve. The
    instruction to load resources only as needed is deliberate: a skill that
    enumerates its own directory would defeat the point of loading it on demand.
    """
    header = f'<skill name="{escape_xml(definition.name)}">'
    parts = [header]

    if definition.resource_base:
        # Escaped, because a directory name is not trusted input.
        #
        # The path is chosen by whoever created the directory. Unescaped, a name
        # containing ``</skill>`` closes the envelope early and everything after
        # it reads as top-level content rather than as part of the skill -- which
        # is precisely the "text that looks like an instruction" that the
        # envelope exists to delimit.
        parts.append(
            f"Base directory for this skill: {escape_xml(definition.resource_base)}\n"
            "Resolve any relative paths in this skill against that directory."
        )
        parts.append("Load only the resources this skill actually references.")

    parts.append(definition.content)
    parts.append("</skill>")
    return "\n\n".join(parts)


@dataclass
class CatalogPublisher:
    """Tracks what catalogue the model has been shown.

    The baseline is the last published digest, so the decision to publish is made
    against *entries* rather than against rendered text. That is what lets the
    wrapper prose evolve without every session resending its catalogue.
    """

    published_digest: str | None = None
    published_entries: tuple[tuple[str, str], ...] = ()

    @property
    def has_published(self) -> bool:
        return self.published_digest is not None

    def update(
        self, snapshot: CatalogSnapshot, visible: Sequence[SkillSummary], digest: str
    ) -> str | None:
        """Return the message to publish, or ``None`` if nothing changed.

        Three cases:

        * nothing published yet and nothing visible -> publish nothing (there is
          no news, and an empty first catalogue is pure noise);
        * the digest is unchanged -> publish nothing (a re-render would be a
          no-op that still costs context);
        * otherwise -> publish a first or replacement catalogue.
        """
        entries = tuple((s.name, s.description) for s in visible)

        if not self.has_published:
            if not entries:
                return None
            self.published_digest = digest
            self.published_entries = entries
            return render_catalog(visible, first=True)

        if digest == self.published_digest:
            return None

        self.published_digest = digest
        self.published_entries = entries
        return render_catalog(visible, first=False)

    def reset(self) -> None:
        """Forget what was published.

        Used when compaction hides the earlier catalogue: the next snapshot must
        be published as a first catalogue again, because the model can no longer
        see the old one.
        """
        self.published_digest = None
        self.published_entries = ()

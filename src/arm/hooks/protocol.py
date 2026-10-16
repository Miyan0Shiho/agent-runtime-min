"""The hook protocol: configuration, matching, decisions, and how they merge.

A hook is an **external command** run at a point in the runtime's own flow. It is
not a second event system: the runtime already has waterfall interception points,
and a hook is a way to attach a subprocess to one of them. The kernel does not
know hooks exist, and a hook does not know the kernel exists; this module is the
translation.

Three parts are worth understanding, because each is where a bridge like this
usually goes wrong:

**Matching is dialect-specific, and the dialects genuinely differ.** Claude Code
treats a bare alphanumeric pattern as an exact alternation and everything else as
a regex; Codex treats every pattern as an unanchored regex. Unifying them would
silently change which hooks fire for one of the two -- a hook that stops firing
is nearly impossible to notice, because nothing reports a hook that did not run.

**Decisions merge most-restrictive-first.** When several hooks match, the
strictest outcome wins: ``deny > ask > allow``. Merging by "first one wins" or
"last one wins" would make the outcome depend on configuration order, so adding a
hook could *loosen* a restriction -- which is the opposite of what anyone
expects when they add a safety hook.

**An invalid regex does not match, rather than raising.** A hook with a broken
pattern must not take down the turn; it simply never fires, and the configuration
parse reports the problem where it can be seen.
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

log = logging.getLogger("arm.hooks")

#: The interception points. Named after what they are rather than after either
#: dialect, so the runtime's vocabulary does not inherit one vendor's spelling.
POINT_AGENT_CREATED = "agent/created"
POINT_PRE_STEP = "agent/pre-step"
POINT_PRE_TOOL = "tools/pre-execute"
POINT_POST_TOOL = "tools/post-execute"
POINT_TURN_STOPPING = "agent/turn-stopping"

POINTS = frozenset(
    {
        POINT_AGENT_CREATED,
        POINT_PRE_STEP,
        POINT_PRE_TOOL,
        POINT_POST_TOOL,
        POINT_TURN_STOPPING,
    }
)

#: The dialects whose configuration format is understood.
DIALECT_CLAUDE = "claude-code"
DIALECT_CODEX = "codex"

DIALECTS = frozenset({DIALECT_CLAUDE, DIALECT_CODEX})

#: Claude Code's hook names, mapped onto the runtime's points.
CLAUDE_POINTS: Mapping[str, str] = {
    "SessionStart": POINT_AGENT_CREATED,
    "UserPromptSubmit": POINT_PRE_STEP,
    "PreToolUse": POINT_PRE_TOOL,
    "PostToolUse": POINT_POST_TOOL,
    "Stop": POINT_TURN_STOPPING,
}

#: Decisions, ordered by severity. The order *is* the merge rule.
DECISION_ALLOW = "allow"
DECISION_ASK = "ask"
DECISION_DENY = "deny"

#: Most restrictive first. A merge that does not follow this order lets a newly
#: added hook loosen an existing restriction.
_SEVERITY = (DECISION_DENY, DECISION_ASK, DECISION_ALLOW)

#: The default timeout. Ten minutes, matching both dialects' documented default:
#: a hook is often a formatter or a test run, and a shorter default would abort
#: legitimate work.
DEFAULT_TIMEOUT_MS = 600_000

#: How much of a hook's stderr to keep in the log record.
STDERR_SUMMARY_CHARS = 500


class HookError(ArmError):
    """A hook configuration or execution problem."""

    def __init__(self, message: str, *, code: str = "HOOK_INVALID") -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Matcher:
    """A pattern deciding whether a hook applies to a subject.

    The dialect determines the interpretation, and the difference is real:

    * **Claude Code**: a pattern of only ``[A-Za-z0-9_|]`` is a literal
      alternation -- ``Read|Write`` matches exactly those two names. Anything
      else is a regex.
    * **Codex**: every pattern is an unanchored regex, so ``Read`` matches
      ``Read`` and also ``ReadFile``.

    A missing, empty or ``*`` pattern matches everything, in both dialects.
    """

    pattern: str
    dialect: str = DIALECT_CLAUDE

    def matches(self, subject: str) -> bool:
        """Whether this matcher applies to a subject.

        An invalid regex **does not match** rather than raising: a broken pattern
        must not take down the turn. The configuration parse reports it where it
        can be seen.
        """
        if self.pattern in ("", "*"):
            return True

        if self.dialect == DIALECT_CLAUDE and _LITERAL_PATTERN.fullmatch(self.pattern):
            # A literal alternation: exact names separated by '|'.
            return subject in self.pattern.split("|")

        compiled = compile_pattern(self.pattern)
        if compiled is None:
            return False
        return compiled.search(subject) is not None


#: A pattern made only of these characters is a literal alternation in Claude
#: Code's dialect. The pipe is included, which is what makes it an alternation
#: rather than a single literal.
_LITERAL_PATTERN = re.compile(r"[A-Za-z0-9_|]+")

#: Compiled patterns are cached because a matcher is consulted on every tool call
#: and a hook configuration does not change at runtime.
_PATTERN_CACHE: dict[str, re.Pattern[str] | None] = {}


def compile_pattern(pattern: str) -> re.Pattern[str] | None:
    """Compile a regex, returning ``None`` for an invalid one.

    Returning ``None`` rather than raising is deliberate: the caller is a match
    test on a hot path, and a configuration error must not become a turn failure.
    """
    if pattern in _PATTERN_CACHE:
        return _PATTERN_CACHE[pattern]
    try:
        compiled: re.Pattern[str] | None = re.compile(pattern)
    except re.error as exc:
        log.warning("ignoring an invalid hook matcher %r: %s", pattern, exc)
        compiled = None
    _PATTERN_CACHE[pattern] = compiled
    return compiled


def matcher_diagnostic(pattern: str, dialect: str) -> str | None:
    """Report a pattern that will never match, at parse time.

    A pattern that is silently dead is the worst kind of configuration bug: the
    hook looks installed and never runs.
    """
    if pattern in ("", "*"):
        return None
    if dialect == DIALECT_CLAUDE and _LITERAL_PATTERN.fullmatch(pattern):
        return None
    if compile_pattern(pattern) is None:
        return f"the matcher {pattern!r} is not a valid regular expression"
    return None


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    """What a hook decided.

    ``reason`` is kept only for the winning decision: an explanation for a
    restriction that did not apply would be confusing rather than helpful.
    """

    kind: str = DECISION_ALLOW
    reason: str = ""
    #: Text the hook wants added to the conversation.
    context: tuple[str, ...] = ()
    #: Whether the hook asked the run to stop. Sticky: the first one wins.
    stop: bool = False
    #: Feedback to show the model when a post-tool hook blocks.
    feedback: str = ""

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.reason:
            payload["reason"] = self.reason
        if self.context:
            payload["context"] = list(self.context)
        if self.stop:
            payload["stop"] = True
        if self.feedback:
            payload["feedback"] = self.feedback
        return payload


def merge_decisions(decisions: Sequence[Decision]) -> Decision:
    """Fold several decisions into one, most restrictive first.

    The rule is ``deny > ask > allow``. Merging by configuration order instead
    would let adding a hook *loosen* an existing restriction, which is the
    opposite of what someone adding a safety hook expects.

    Context accumulates across all of them regardless of severity: a hook that
    was overruled on the decision may still have useful text to contribute, and
    discarding it would lose information for no benefit.
    """
    if not decisions:
        return Decision()

    context: list[str] = []
    stop = False
    feedback = ""
    for decision in decisions:
        context.extend(decision.context)
        # `stop` is sticky on first set: once a hook asks to stop, a later hook
        # cannot un-ask it.
        stop = stop or decision.stop
        if decision.feedback and not feedback:
            feedback = decision.feedback

    # The winner is the most restrictive kind present, and its reason is the
    # *first* reason given at that severity. First rather than last because the
    # explanation a reader wants is the one from the hook that was checked first,
    # and a later hook agreeing should not overwrite it.
    winner = DECISION_ALLOW
    reason = ""
    for severity in _SEVERITY:
        matching = [d for d in decisions if d.kind == severity]
        if matching:
            winner = severity
            reason = matching[0].reason
            break

    return Decision(
        kind=winner, reason=reason, context=tuple(context), stop=stop,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HookHandler:
    """One command to run at one point."""

    point: str
    command: str
    args: tuple[str, ...] = ()
    matcher: Matcher = field(default_factory=lambda: Matcher(""))
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    #: A stable identifier, used to pair the invoked/result log records.
    handler_id: str = ""

    def __post_init__(self) -> None:
        if self.point not in POINTS:
            raise HookError(f"unknown hook point {self.point!r}")
        if not self.command:
            raise HookError("a hook needs a command")
        if self.timeout_ms <= 0:
            raise HookError("a hook timeout must be positive")

        # The identifier is derived when the caller did not supply one, so every
        # handler has an identity regardless of how it was constructed.
        #
        # The config loader assigns `dialect:point:n`, but a handler built by
        # hand -- in a test, or by an embedding application -- got the empty
        # string. That showed up when a deny needed to record *which* hook
        # refused: the one construction path that did not come from the loader
        # was also the one path with nothing to report.
        #
        # Derived from what the handler *is* rather than from where it came from,
        # so two identical handlers are indistinguishable on purpose and two
        # different ones never collide.
        if not self.handler_id:
            digest = hashlib.sha256(
                "\x00".join((self.point, self.command, *self.args)).encode("utf-8")
            ).hexdigest()[:8]
            object.__setattr__(self, "handler_id", f"{self.point}:{digest}")


def parse_config(
    raw: Mapping[str, Any], *, dialect: str = DIALECT_CLAUDE
) -> tuple[list[HookHandler], list[str]]:
    """Parse a hooks configuration.

    Returns the handlers and a list of diagnostics. A malformed entry produces a
    diagnostic and is skipped rather than aborting the parse: one bad hook must
    not disable every other hook, and a startup that refuses to run because of a
    typo is worse than one that reports the typo.
    """
    if dialect not in DIALECTS:
        raise HookError(f"unknown hook dialect {dialect!r}")

    diagnostics: list[str] = []
    handlers: list[HookHandler] = []
    counter = 0

    hooks = raw.get("hooks")
    if hooks is None:
        return [], ["the configuration has no 'hooks' key"]
    if not isinstance(hooks, Mapping):
        return [], ["'hooks' must be an object"]

    for name, entries in hooks.items():
        point = _resolve_point(str(name), dialect)
        if point is None:
            diagnostics.append(f"unknown hook point {name!r}; it will never fire")
            continue
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            diagnostics.append(f"the entries for {name!r} must be a list")
            continue

        for entry in entries:
            if not isinstance(entry, Mapping):
                diagnostics.append(f"an entry for {name!r} must be an object")
                continue
            matcher_pattern = str(entry.get("matcher", ""))
            problem = matcher_diagnostic(matcher_pattern, dialect)
            if problem is not None:
                diagnostics.append(f"{name}: {problem}")

            inner = entry.get("hooks")
            if not isinstance(inner, Sequence) or isinstance(inner, (str, bytes)):
                diagnostics.append(f"{name}: an entry needs a 'hooks' list")
                continue

            for item in inner:
                if not isinstance(item, Mapping):
                    continue
                kind = str(item.get("type", "command"))
                if kind != "command":
                    # Claude Code also defines `prompt`, `agent` and `http`.
                    # They are recognised and skipped rather than treated as
                    # errors: refusing to start because a config uses a feature
                    # this bridge does not implement would be hostile.
                    diagnostics.append(
                        f"{name}: hook type {kind!r} is not supported and was skipped"
                    )
                    continue
                command = str(item.get("command", ""))
                if not command:
                    diagnostics.append(f"{name}: a command hook needs a 'command'")
                    continue
                counter += 1
                handlers.append(
                    HookHandler(
                        point=point,
                        command=command,
                        args=tuple(str(a) for a in (item.get("args") or ())),
                        matcher=Matcher(matcher_pattern, dialect),
                        timeout_ms=int(item.get("timeout", DEFAULT_TIMEOUT_MS)),
                        handler_id=f"{dialect}:{name}:{counter}",
                    )
                )

    return handlers, diagnostics


def _resolve_point(name: str, dialect: str) -> str | None:
    """Map a configuration's point name onto the runtime's vocabulary."""
    if name in POINTS:
        return name
    if dialect == DIALECT_CLAUDE:
        return CLAUDE_POINTS.get(name)
    # Codex spells its points in the runtime's own form.
    return CLAUDE_POINTS.get(name)


def load_config(
    path: Path, *, dialect: str = DIALECT_CLAUDE
) -> tuple[list[HookHandler], list[str]]:
    """Read a hooks file.

    A missing or malformed file yields **zero hooks and a diagnostic**, not an
    error. A broken hook configuration must not prevent the agent from starting:
    the agent is still useful, and refusing to run would make a typo look like a
    broken installation.
    """
    if not path.exists():
        return [], []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"could not read {path}: {exc}"]
    if not isinstance(raw, Mapping):
        return [], [f"{path} must contain an object"]
    # A settings file may nest hooks under a key; accept both shapes.
    if "hooks" not in raw and isinstance(raw.get("hooks_config"), Mapping):
        raw = {"hooks": raw["hooks_config"]}
    return parse_config(raw, dialect=dialect)

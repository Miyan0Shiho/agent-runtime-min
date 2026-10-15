"""Delegation contracts: depth, capabilities, and what a subagent may do.

Three ideas, each of which prevents a specific way delegation goes wrong.

**Depth is bounded, and the bound is checked before anything is created.** A
subagent that can spawn subagents without a limit is a fork bomb with extra
steps. The limit is enforced at the entry point, not discovered when a chain
finally gets too deep.

**Capabilities are asserted one by one, in a fixed order.** A provider that
cannot honour ``output_schema`` must say so *before* a run starts, and the error
must name the specific capability that is missing. A single "unsupported" flag
would leave the caller guessing which part of its request to drop.

**The child is isolated by construction.** It gets its own session, its own tool
view and its own sandbox policy. Sharing any of those would make a delegated task
able to disturb the work that delegated it -- and the parent's transcript would
then contain changes it never made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..kernel.errors import ArmError

log = logging.getLogger("arm.subagent")

#: How deep a delegation chain may go. Depth 0 is the top-level agent.
DEFAULT_MAX_DEPTH = 3

#: The stop reasons a run can report. Deliberately small: each one implies a
#: different action for the caller, and a longer list would be a list of
#: distinctions nobody acts on.
STOP_COMPLETED = "completed"
STOP_MAX_TOKENS = "max-tokens"
STOP_ABORTED = "aborted"
STOP_REFUSAL = "refusal"
STOP_ERROR = "error"

STOP_REASONS = frozenset(
    {STOP_COMPLETED, STOP_MAX_TOKENS, STOP_ABORTED, STOP_REFUSAL, STOP_ERROR}
)

#: The capabilities a caller may require of a provider. Asserted in this order,
#: so the error names the *first* thing that is missing rather than an arbitrary
#: one -- which is what makes it actionable.
CAPABILITY_ORDER = (
    "agent_options",
    "output_schema",
    "depth_limit",
    "tool_filter",
    "persona",
)


class SubagentError(ArmError):
    """A delegation was refused."""

    def __init__(self, message: str, *, code: str = "SUBAGENT_INVALID") -> None:
        super().__init__(message)
        self.code = code


class NoProviderError(SubagentError):
    """No provider is registered under the requested name."""

    def __init__(self, name: str, available: Sequence[str] = ()) -> None:
        listed = ", ".join(sorted(available)) or "(none)"
        super().__init__(
            f"no subagent provider named {name!r}; registered providers are: {listed}",
            code="NO_PROVIDER",
        )


class UnsupportedCapabilityError(SubagentError):
    """A provider cannot honour part of the request."""

    def __init__(self, provider: str, capability: str) -> None:
        super().__init__(
            f"provider {provider!r} does not support {capability!r}, which this "
            "request requires",
            code="UNSUPPORTED_CAPABILITY",
        )
        self.capability = capability


class DepthLimitError(SubagentError):
    """The delegation would exceed the depth limit."""

    def __init__(self, requested: int, limit: int) -> None:
        super().__init__(
            f"delegation depth {requested} exceeds the limit of {limit}; a "
            "subagent cannot spawn a chain deeper than this",
            code="DEPTH_LIMIT",
        )


def assert_max_depth(depth: int, *, limit: int = DEFAULT_MAX_DEPTH) -> None:
    """Refuse a depth beyond the limit.

    Checked at the entry point so a chain is stopped before a child exists,
    rather than after several children have been created and must be torn down.
    """
    if depth < 0:
        raise SubagentError(f"depth must be non-negative, got {depth}")
    if depth >= limit:
        raise DepthLimitError(depth + 1, limit)


def assert_capabilities(
    provider: str, supported: frozenset[str], request: "DelegationRequest"
) -> None:
    """Refuse the request if the provider cannot honour it.

    Asserted one capability at a time, in a fixed order, so the error names the
    first thing that is missing. Reporting "unsupported" without saying *what*
    leaves the caller guessing which part of its request to drop.
    """
    for capability in CAPABILITY_ORDER:
        if not request.requires(capability):
            continue
        if capability not in supported:
            raise UnsupportedCapabilityError(provider, capability)


@dataclass(frozen=True, slots=True)
class DelegationRequest:
    """What the caller is asking a subagent to do.

    Every optional field is a *requirement*: if it is set, the provider must
    support it or the delegation is refused. That direction is deliberate --
    silently dropping a requirement would produce a run that appears to succeed
    while doing something other than what was asked.
    """

    prompt: str
    provider: str = "in-process"
    label: str = ""
    #: How deep the child will be. The top-level agent is depth 0.
    depth: int = 0
    max_depth: int = DEFAULT_MAX_DEPTH
    #: Tool names the child may use. Empty means "inherit the parent's view".
    tools: tuple[str, ...] = ()
    #: A persona appended to the child's prompt.
    persona: str = ""
    #: A JSON Schema the child's final answer must satisfy.
    output_schema: Mapping[str, Any] | None = None
    #: Model overrides for the child, if the provider can apply them.
    model: str = ""

    def requires(self, capability: str) -> bool:
        """Whether this request needs a capability.

        The single place that decides, so the assertion loop and the provider
        cannot disagree about what a request needs.
        """
        if capability == "agent_options":
            return bool(self.model)
        if capability == "output_schema":
            return self.output_schema is not None
        if capability == "depth_limit":
            return self.max_depth != DEFAULT_MAX_DEPTH
        if capability == "tool_filter":
            return bool(self.tools)
        if capability == "persona":
            return bool(self.persona)
        return False

    def child_depth(self) -> int:
        return self.depth + 1


@dataclass(frozen=True, slots=True)
class DelegationResult:
    """What a delegation produced.

    ``stop_reason`` is the field a caller must branch on. ``output`` is the
    child's final message, which may be empty for a run that stopped for a
    reason other than completion -- and is deliberately *not* defaulted to a
    placeholder, so "the child said nothing" stays distinguishable from "the
    child said something empty".
    """

    run_id: str
    provider: str
    stop_reason: str
    output: str = ""
    #: The validated structured value, when an output schema was required.
    structured: Mapping[str, Any] | None = None
    error: str = ""
    #: The child's session id, so the caller can inspect or resume it.
    child_session: str = ""
    events: int = 0

    @property
    def ok(self) -> bool:
        return self.stop_reason == STOP_COMPLETED

    def to_stop_reason(self, raw: str) -> str:
        """Map a raw termination reason onto the vocabulary."""
        return map_stop_reason(raw)


def map_stop_reason(raw: str) -> str:
    """Translate a turn's termination reason into a delegation stop reason.

    The mapping is not the identity, and the differences matter:

    * ``blocked`` becomes ``refusal`` -- the child declined, which is a result
      rather than a failure;
    * an unknown reason becomes ``error`` -- the conservative direction, because
      reporting an unknown outcome as success would hide it.
    """
    if raw == "completed":
        return STOP_COMPLETED
    if raw == "max-tokens":
        return STOP_MAX_TOKENS
    if raw == "aborted":
        return STOP_ABORTED
    if raw == "blocked":
        return STOP_REFUSAL
    return STOP_ERROR


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """What a provider can do.

    Declared rather than probed: a provider that could only discover its own
    limits by failing would turn every unsupported request into a partially
    executed run.
    """

    name: str
    supported: frozenset[str] = field(default_factory=frozenset)

    def can(self, capability: str) -> bool:
        return capability in self.supported


#: What every provider must offer. The three shared capabilities are the ones
#: every implementation can do; the rest are opt-in.
BASE_CAPABILITIES = frozenset({"depth_limit"})


def capabilities_for(*extra: str) -> ProviderCapabilities:
    """Build a capability set from the base plus whatever is claimed."""
    return ProviderCapabilities(
        name="", supported=BASE_CAPABILITIES | frozenset(extra)
    )

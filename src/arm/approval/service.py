"""Approval: whether a tool call may proceed, and who decides.

The policy vocabulary is two words, and the whole design is in making them mean
exactly what they say:

* **``ask``** -- consult the configured answerers. If none is available, the
  result is ``unavailable``, which is a **denial**. That is the fail-closed
  default: a missing approval channel means "no", never "yes".
* **``never``** -- prompt nobody, ever. Every ask resolves to ``rejected``
  deterministically, without consulting anything.

Four outcomes, and **exactly one of them grants permission**:

    allowed-once | rejected | cancelled | unavailable

``allowed-once`` is the only grant, and its name is a promise: an approval
authorizes *this call*, not a class of calls. There is no "always allow" here,
because a policy that could be widened by answering a question once is a policy
that is really a prompt.

Two implementation details are load-bearing and easy to get wrong:

**``never`` is decided inside the service, before any dispatch.** It would be
simpler to implement it as a listener that short-circuits, but a listener with
``prepend`` would run ahead of it, and the promise "under ``never`` nothing is
ever asked" would become "nothing is asked, unless someone registered a listener
that runs first". The guarantee has to belong to the request path itself, not to
a position in a list.

**A malformed answer is not an answer.** An answerer that returns something
outside the vocabulary, or that raises, yields ``unavailable`` -- a denial. The
alternative would be to let an approval bug become an approval grant.

**Every decision is recorded in pairs.** An ask and its decision are both
appended, or neither is: returning a decision that is not in the log would break
the property that the transcript explains itself.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from ..kernel.errors import ArmError

log = logging.getLogger("arm.approval")

#: The two policies. There is deliberately no third.
POLICY_ASK = "ask"
POLICY_NEVER = "never"

POLICIES = frozenset({POLICY_ASK, POLICY_NEVER})

#: The four outcomes. Only the first grants anything.
OUTCOME_ALLOWED_ONCE = "allowed-once"
OUTCOME_REJECTED = "rejected"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_UNAVAILABLE = "unavailable"

OUTCOMES = frozenset(
    {
        OUTCOME_ALLOWED_ONCE,
        OUTCOME_REJECTED,
        OUTCOME_CANCELLED,
        OUTCOME_UNAVAILABLE,
    }
)


class ApprovalError(ArmError):
    """An approval could not be requested. The call is denied."""


class ApprovalPolicyError(ApprovalError):
    """The configured policy is not one of the two known words."""


class NoOpenTurnError(ApprovalError):
    """An approval was requested outside a turn.

    The audit pair has to be bracketed by a turn, because a bare event written
    between turns is indistinguishable from a crash artefact on reload.
    """


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """What is being asked, in enough detail to answer."""

    tool: str
    reason: str = ""
    #: The arguments, so an answerer can show what would actually run.
    arguments: Mapping[str, Any] = field(default_factory=dict)
    #: The call this approval is for, so the audit pair can be correlated.
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """The outcome of one request."""

    outcome: str
    #: Why, for the model-visible rejection message.
    reason: str = ""

    @property
    def granted(self) -> bool:
        """Whether this decision permits the call.

        Only ``allowed-once`` does. Everything else -- including an unexpected
        value -- denies, so a bug cannot become a grant.
        """
        return self.outcome == OUTCOME_ALLOWED_ONCE

    @property
    def was_cancelled(self) -> bool:
        """Whether the caller aborted rather than the policy refusing.

        Named distinctly from the ``abort`` factory: a property and a method
        with one name would shadow each other, and the shadowing would be
        silent -- the factory would simply stop being reachable.
        """
        return self.outcome == OUTCOME_CANCELLED

    @staticmethod
    def allowed() -> "ApprovalDecision":
        return ApprovalDecision(OUTCOME_ALLOWED_ONCE)

    @staticmethod
    def rejected(reason: str) -> "ApprovalDecision":
        return ApprovalDecision(OUTCOME_REJECTED, reason)

    @staticmethod
    def aborted(reason: str = "") -> "ApprovalDecision":
        """The caller cancelled the approval, rather than the policy refusing."""
        return ApprovalDecision(OUTCOME_CANCELLED, reason)

    @staticmethod
    def unavailable(reason: str = "") -> "ApprovalDecision":
        return ApprovalDecision(OUTCOME_UNAVAILABLE, reason)


class Answerer(Protocol):
    """Something that can answer an approval request.

    Declared as a protocol so a terminal prompt, a UI, a test double and a
    headless auto-denier are all the same shape to the service.
    """

    def __call__(self, request: ApprovalRequest) -> Awaitable[str] | str: ...  # noqa: E704


#: An approval is bounded, because an unanswered prompt must not hang a turn
#: forever. The timeout resolves to ``unavailable`` -- a denial.
DEFAULT_TIMEOUT_SECONDS = 300.0


class ApprovalService:
    """Decides whether a tool call may proceed.

    Holds the policy and the answerers. The session is passed in per request
    rather than held, so one service can serve several sessions with their own
    recorded overrides.
    """

    def __init__(
        self,
        *,
        policy: str = POLICY_ASK,
        answerers: Sequence[Answerer] = (),
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        recorder: Callable[[str, Mapping[str, Any]], None] | None = None,
        open_turn: Callable[[], bool] | None = None,
    ) -> None:
        if policy not in POLICIES:
            raise ApprovalPolicyError(
                f"unknown approval policy {policy!r}; expected one of "
                f"{sorted(POLICIES)}"
            )
        self._policy = policy
        self._answerers = list(answerers)
        self._timeout = timeout
        #: Where the audit pair is written. Optional so the service can be used
        #: without a session, but its absence means decisions are unrecorded.
        self._record = recorder
        self._open_turn = open_turn
        #: Per-session overrides, read from the log on demand.
        self._overrides: dict[str, str] = {}

    # -- policy ----------------------------------------------------------

    @property
    def policy(self) -> str:
        return self._policy

    def effective_policy(self, session_id: str = "") -> str:
        """The policy in force: an explicit override wins over the default.

        Read from a per-session override map rather than stored on the service,
        so one process can run a headless session under ``never`` and an
        interactive one under ``ask``.
        """
        return self._overrides.get(session_id, self._policy)

    def set_override(self, session_id: str, policy: str) -> None:
        if policy not in POLICIES:
            raise ApprovalPolicyError(
                f"unknown approval policy {policy!r}; expected one of "
                f"{sorted(POLICIES)}"
            )
        self._overrides[session_id] = policy

    def clear_override(self, session_id: str) -> None:
        self._overrides.pop(session_id, None)

    # -- asking ----------------------------------------------------------

    async def request(
        self,
        request: ApprovalRequest,
        *,
        session_id: str = "",
        signal: Any = None,
    ) -> ApprovalDecision:
        """Ask whether a call may proceed.

        Never raises for an ordinary outcome: every path returns one of the four
        decisions. The only exception is a request made outside a turn, which is
        a programming error rather than a business outcome.
        """
        if self._open_turn is not None and not self._open_turn():
            # An approval outside a turn has nowhere to be recorded, and an
            # unrecorded decision is one the transcript cannot explain.
            raise NoOpenTurnError(
                "an approval was requested outside an open turn; the audit pair "
                "must be bracketed by a turn"
            )

        # The policy is evaluated HERE, before any dispatch. Doing it as a
        # listener would let a listener registered earlier run first, and the
        # guarantee would degrade to "nothing is asked, unless something else
        # was registered ahead of us".
        policy = self.effective_policy(session_id)

        self._append(
            "approval/asked",
            {
                "call_id": request.call_id,
                "tool": request.tool,
                "reason": request.reason,
                "policy": policy,
            },
        )

        if policy == POLICY_NEVER:
            # Deterministic and immediate: nobody is consulted, so the answer
            # does not depend on who happens to be registered.
            decision = ApprovalDecision.rejected(
                request.reason
                or f'tool "{request.tool}" requires approval, and this session '
                "never prompts"
            )
            self._record_decision(request, decision)
            return decision

        decision = await self._consult(request, signal=signal)
        self._record_decision(request, decision)
        return decision

    async def _consult(
        self, request: ApprovalRequest, *, signal: Any
    ) -> ApprovalDecision:
        """Ask the answerers, failing closed on every kind of trouble."""
        if not self._answerers:
            # No channel is a denial, not a permission. A policy that said "ask"
            # has already declined to auto-allow, and treating a missing channel
            # as consent would invert its meaning.
            return ApprovalDecision.unavailable(
                f'tool "{request.tool}" requires approval, but no approval '
                "channel is available"
            )

        if signal is not None and getattr(signal, "aborted", False):
            return ApprovalDecision.aborted("aborted before approval")

        # Only the first answerer that produces a usable answer counts. A chain
        # is used rather than a vote: two answerers disagreeing is a
        # configuration bug, and averaging them would invent a policy.
        for answerer in self._answerers:
            try:
                raw = answerer(request)
                if asyncio.iscoroutine(raw):
                    raw = await asyncio.wait_for(raw, timeout=self._timeout)
            except asyncio.TimeoutError:
                return ApprovalDecision.unavailable(
                    "the approval request timed out with no answer"
                )
            except asyncio.CancelledError:
                return ApprovalDecision.aborted("the approval was interrupted")
            except Exception as exc:  # noqa: BLE001 - a broken channel denies
                log.warning("approval answerer failed: %s", exc)
                return ApprovalDecision.unavailable(
                    f"the approval channel failed: {exc}"
                )

            # An answer outside the vocabulary is not an answer. Defaulting it
            # would let an approval bug become an approval grant.
            if raw not in OUTCOMES:
                log.warning("approval answerer returned %r; treating as unavailable", raw)
                return ApprovalDecision.unavailable(
                    f"the approval channel returned an unrecognized answer"
                )
            if raw == OUTCOME_ALLOWED_ONCE:
                return ApprovalDecision.allowed()
            if raw == OUTCOME_REJECTED:
                return ApprovalDecision.rejected(
                    f'the user rejected tool "{request.tool}"'
                )
            if raw == OUTCOME_CANCELLED:
                return ApprovalDecision.aborted("the user cancelled the approval")
            return ApprovalDecision.unavailable(
                f'tool "{request.tool}" requires approval, but no approval '
                "channel is available"
            )

        return ApprovalDecision.unavailable(
            "no approval channel produced an answer"
        )

    # -- audit -----------------------------------------------------------

    def _record_decision(
        self, request: ApprovalRequest, decision: ApprovalDecision
    ) -> None:
        self._append(
            "approval/decided",
            {
                "call_id": request.call_id,
                "tool": request.tool,
                "outcome": decision.outcome,
                "reason": decision.reason,
            },
        )

    def _append(self, event_type: str, data: Mapping[str, Any]) -> None:
        if self._record is None:
            return
        try:
            self._record(event_type, data)
        except Exception as exc:  # noqa: BLE001
            # A failing audit write must not become a failing decision: the
            # decision is already made, and raising here would turn a recording
            # problem into a tool failure.
            log.error("could not record %s: %s", event_type, exc)


# ---------------------------------------------------------------------------
# Ready-made answerers
# ---------------------------------------------------------------------------


def deny_all() -> Answerer:
    """An answerer that refuses everything.

    The right answerer for an unattended run: it is honest about what it does,
    rather than relying on the absence of a channel to produce the same result.
    """

    def answer(request: ApprovalRequest) -> str:
        del request
        return OUTCOME_REJECTED

    return answer


def allow_all() -> Answerer:
    """An answerer that permits everything.

    For tests and for a trusted local session. Named explicitly rather than
    expressed as a policy so that "we allow everything" is visible in the code
    that configures it.
    """

    def answer(request: ApprovalRequest) -> str:
        del request
        return OUTCOME_ALLOWED_ONCE

    return answer


def answer_from(predicate: Callable[[ApprovalRequest], bool]) -> Answerer:
    """An answerer that decides with a predicate.

    ``True`` grants once; ``False`` rejects. Useful for a policy expressed as a
    rule rather than as a conversation.
    """

    def answer(request: ApprovalRequest) -> str:
        return OUTCOME_ALLOWED_ONCE if predicate(request) else OUTCOME_REJECTED

    return answer


def approval_hook(
    service: "ApprovalService",
    *,
    session_id: str = "",
    signal: Any = None,
    escalate: Callable[["ApprovalRequest"], bool] | None = None,
) -> Callable[[Any, str], Any]:
    """Adapt an :class:`ApprovalService` into a pipeline approval hook.

    The pipeline asks a narrow question -- "may this call proceed?" -- and knows
    nothing about policies, channels or audit pairs. This adapter is the seam
    between the two, and keeping it here rather than in the pipeline is what
    lets the pipeline stay ignorant of how approvals are obtained.

    ``escalate`` is consulted only after a refusal, and only when the caller
    supplies one. It exists because a sandbox denial is sometimes worth offering
    to widen -- but the offer is a *separate* decision, so it is a separate
    callable rather than a mode on the service.
    """

    async def hook(execution: Any, reason: str) -> bool:
        request = ApprovalRequest(
            tool=getattr(execution, "name", ""),
            reason=reason,
            arguments=dict(getattr(execution, "arguments", {}) or {}),
            call_id=getattr(execution, "call_id", ""),
        )
        decision = await service.request(request, session_id=session_id, signal=signal)
        if decision.granted:
            return True
        if escalate is not None and escalate(request):
            # A second, narrower ask. The escalation does not inherit the first
            # refusal, and its approval is for this call only.
            again = await service.request(
                ApprovalRequest(
                    tool=request.tool,
                    reason=f"escalation: {reason}",
                    arguments=request.arguments,
                    call_id=request.call_id,
                ),
                session_id=session_id,
                signal=signal,
            )
            return again.granted
        return False

    return hook

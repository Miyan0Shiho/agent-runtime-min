"""Error taxonomy for the runtime.

One rule shapes this module: **business failures are results, not exceptions.**
A tool that rejects its arguments, a model that returns garbage, a permission
check that denies -- none of those propagate out of the loop. They become a
recorded terminal state.

What *does* raise is a contract violation: the caller asked for something the
log cannot represent. Those errors are loud, typed, and never swallowed, because
silently accepting them is how a log stops being the truth.
"""

from __future__ import annotations

from typing import Any


class ArmError(Exception):
    """Base class for every error this runtime raises deliberately."""


# --------------------------------------------------------------------------
# Log / session contract violations
# --------------------------------------------------------------------------


class SessionContractError(ArmError):
    """The requested operation would make the log unable to describe itself."""


class SeqDiscontinuityError(SessionContractError):
    """Events must be contiguous from 0. Continuity is defined, not checked."""


class ReentrantAppendError(SessionContractError):
    """A listener tried to append while a commit was in flight.

    Letting this through would recurse forever inside the commit path.
    """


class SurfaceMetadataError(SessionContractError):
    """A surface event lacked ``surfaceOp``, or a non-surface event carried one."""


class SurfaceRangeError(SessionContractError):
    """A ``replace`` pointed at a range that is not currently on the surface."""


class SourceReferenceError(SessionContractError):
    """``source_event_seqs`` did not exactly cover the shadowed nodes."""


class ToolResultRewriteError(SessionContractError):
    """A ``tool/result`` replacement tried to change something other than content."""


class SystemHeadRewriteError(SessionContractError):
    """Node 0 is the system prompt and may only be rewritten by a same-type,
    exactly-covering replacement."""


class ProjectionRequiredError(SessionContractError):
    """An event needs a message projection that is not registered.

    Dropping the content instead would silently change what the model sees.
    """


class ProjectionRemovedError(SessionContractError):
    """A projection used earlier in this log has since been unregistered."""


class PayloadNotSerializableError(SessionContractError):
    """The payload would not survive a JSON round-trip byte-for-byte.

    Disk and memory must be isomorphic, so a payload that cannot round-trip
    must never reach either.
    """


class UnknownEventTypeError(SessionContractError):
    """An unknown event type arrived without ``ignorable: true``.

    Defaulting to "skip it" would silently restore a partial session, which is
    a safety problem rather than a UX problem.
    """


class FormatVersionError(SessionContractError):
    """The log was written by a newer format generation than this build knows."""


class LogCorruptionError(SessionContractError):
    """A structurally complete record failed to decode. Not recoverable."""


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


class PersistenceError(ArmError):
    """Base class for storage-layer failures."""


class AppendRollbackError(PersistenceError):
    """An append failed *and* the rollback failed. Both causes are reported."""

    def __init__(self, message: str, *, causes: list[BaseException]) -> None:
        super().__init__(message)
        self.causes = causes


# --------------------------------------------------------------------------
# Waterfall / hooks
# --------------------------------------------------------------------------


class WaterfallError(ArmError):
    """Base class for middleware-chain misuse."""


class DoubleContinuationError(WaterfallError):
    """A hook called ``next()`` more than once."""


class MissingDecisionError(WaterfallError):
    """A hook neither continued the chain nor returned a decision."""


class WaterfallExhaustedError(WaterfallError):
    """The chain ended without producing a decision."""


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------


class ServiceNotProvidedError(ArmError):
    """A service was requested before anything provided it."""


class ServiceAlreadyProvidedError(ArmError):
    """A second provider tried to claim a name that is already taken."""


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class ToolDefinitionError(ArmError):
    """A tool declaration is malformed."""


class ReservedToolNameError(ToolDefinitionError):
    """A tool tried to take a name the runtime reserves.

    Checked at *definition* time as well as registration time, so the failure
    names the offending tool rather than surfacing later as a mysterious
    registry conflict.
    """


class ToolContractError(ArmError):
    """A policy hook broke its contract.

    Distinct from a business denial: a hook that neither continues the chain nor
    returns a decision is a bug in the plugin, and the runtime says so instead of
    guessing which behaviour was intended.
    """


def aggregate(message: str, causes: list[BaseException]) -> AppendRollbackError:
    """Build a rollback error carrying every cause, mirroring AggregateError."""
    return AppendRollbackError(message, causes=causes)


def describe(value: Any) -> str:
    """Render a thrown value for a model-visible error string.

    Tool bodies may raise anything, including non-Exception objects. The model
    still needs a readable message, and the runtime must not crash formatting it.
    """
    if isinstance(value, BaseException):
        text = str(value)
        return text if text else type(value).__name__
    try:
        return str(value)
    except Exception:  # pragma: no cover - pathological __str__
        return repr(type(value))

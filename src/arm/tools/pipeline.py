"""The tool execution pipeline: seven stages, and no stage can throw away the step.

The pipeline's central rule is **failure is a result**. Invalid arguments, an
unknown tool, a permission denial, a body that raises, output that fails
validation -- every one of them becomes an ``isError`` result. Not one of them
propagates into the loop. The single exception is a failure of the *scheduler
itself* (the log cannot be written), which is an infrastructure fault rather than
a business outcome.

The second rule is **arguments are frozen once**. Policies may deny, and may
replace a result, but nothing may rewrite the arguments. That is not a
limitation -- a guard that can rewrite arguments is a guard that can bypass
argument validation.

Stage order and who may act:

    1. materialize + freeze arguments   create the execution, snapshot args
    2. pre-execute (waterfall)          policy: allow / deny / cancel / ask
    3. guard                            monotonic: may only deny
    4. execute (wrapper)                the only stage that may swap the signal
    5. body                             the tool itself
    6. project content                  render what the model sees
    7. post-execute (waterfall)         may replace content/value, or block

Then ``finalize`` runs exactly once and must be total.

Whether stage 7 runs depends on *where* the failure happened, and the
distinction is load-bearing: a failure inside the execute wrapper skips
post-execute (the wrapper's own contract is broken), while a body error, an
argument error or an output-validation error still runs it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from ..kernel.errors import WaterfallError, describe
from ..kernel.freeze import FrozenDict, assert_json_lossless, deep_freeze
from ..kernel.waterfall import compose
from .definition import ERROR_PREFIX, RegisteredTool, ToolDefinition, ToolResult
from .drain import BodyCancelled, DrainScope
from .registry import ToolRegistry, ToolView

log = logging.getLogger("arm.tools.pipeline")

#: Error codes carried on results. These persist for diagnosis but never reach
#: the model's content.
CODE_INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
CODE_UNKNOWN_TOOL = "UNKNOWN_TOOL"
CODE_DENIED = "DENIED"
CODE_APPROVAL_REJECTED = "APPROVAL_REJECTED"
CODE_NO_APPROVAL_CHANNEL = "NO_APPROVAL_CHANNEL"
CODE_BODY_ERROR = "BODY_ERROR"
CODE_INVALID_OUTPUT = "INVALID_OUTPUT"
CODE_TIMEOUT = "TOOL_TIMEOUT"
CODE_ABORTED = "ABORTED"
CODE_ABORTED_BEFORE_DISPATCH = "ABORTED_BEFORE_DISPATCH"
CODE_WRAPPER_ERROR = "WRAPPER_ERROR"
CODE_SCHEDULER_FAILURE = "SCHEDULER_FAILURE"

#: Synthetic result text for a call that was cancelled before it was dispatched.
#: Without this, the next request would contain a tool_call with no matching
#: tool result, and the model would reject it.
ABORTED_BEFORE_DISPATCH_TEXT = f"{ERROR_PREFIX}tool call aborted before dispatch"


class PipelineAbort(Exception):
    """Internal signal that a policy denied or cancelled the call.

    Never escapes the pipeline: it is caught and converted into a result.
    """

    def __init__(self, reason: str, code: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


@dataclass(frozen=True, slots=True)
class Decision:
    """What a policy stage decided.

    ``allow`` continues the chain. ``deny`` and ``cancel`` short-circuit.
    ``ask`` means an approval is required before proceeding.
    """

    kind: str  # "allow" | "deny" | "cancel" | "ask"
    reason: str = ""

    @staticmethod
    def allow() -> "Decision":
        return Decision(kind="allow")

    @staticmethod
    def deny(reason: str) -> "Decision":
        return Decision(kind="deny", reason=reason)

    @staticmethod
    def cancel(reason: str = "") -> "Decision":
        return Decision(kind="cancel", reason=reason)

    @staticmethod
    def ask(reason: str = "") -> "Decision":
        return Decision(kind="ask", reason=reason)


#: Signature of a pre-execute hook: ``(ctx, next) -> Decision | None``.
PreExecuteHook = Callable[[Any, Any], Awaitable[Any]]


@dataclass
class Execution:
    """One tool call in flight.

    ``arguments`` is frozen at materialization and never reassigned. The ``signal``
    is the one mutable field, and only stage 4 may change it -- the registry
    re-fuses the caller's signal before the body runs, so "swap the signal" can
    never mean "detach the cancellation".
    """

    call_id: str
    name: str
    arguments: Mapping[str, Any]
    tool: RegisteredTool | None
    definition: ToolDefinition | None
    #: The caller's cancellation signal, as an asyncio.Event-like flag.
    signal: "AbortSignal"
    #: Set by a policy stage to override the final result.
    override: ToolResult | None = None
    #: Result of the body, before projection.
    raw_value: Any = None
    started: bool = False
    finished: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0
    #: Free-form context for policies (session, step, permission mode...).
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.monotonic()
        return (end - self.started_at) * 1000.0


class AbortSignal:
    """A cancellation flag shared by the caller and the pipeline.

    Deliberately not an ``asyncio.CancelledError``: the pipeline needs to *ask*
    whether cancellation was requested, and to keep running a body that already
    started. A task cancellation would inject an exception at the next await,
    which is precisely the "abandon started work" behaviour the contract forbids.

    ``shield`` provides the scope in which already-started work is allowed to
    finish: inside it, cancellation requests are recorded but not raised.
    """

    __slots__ = ("_event", "_cause")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._cause: str | None = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def cause(self) -> str | None:
        """The typed cancellation cause, or ``None`` if not cancelled."""
        return self._cause

    def abort(self, cause: str) -> None:
        """Request cancellation. The first cause wins; later ones are ignored."""
        if not self._event.is_set():
            self._cause = cause
            self._event.set()

    def raise_if_aborted(self) -> None:
        """Raise if cancellation has been requested.

        Called at *dispatch* boundaries -- never in the middle of a body that has
        already begun.
        """
        if self._event.is_set():
            raise PipelineAbort(self._cause or "aborted", CODE_ABORTED)

    async def wait(self) -> None:
        await self._event.wait()

    def fused_with(self, other: "AbortSignal") -> "AbortSignal":
        """A signal that aborts when either input does.

        Used to re-attach the caller's signal after stage 4, so a wrapper cannot
        drop the cancellation by substituting its own signal.
        """
        fused = AbortSignal()
        if self.aborted:
            fused.abort(self.cause or "aborted")
        if other.aborted:
            fused.abort(other.cause or "aborted")
        return fused


@dataclass(frozen=True, slots=True)
class PipelineHooks:
    """The policy seams. All optional; absent means "no opinion"."""

    pre_execute: tuple[PreExecuteHook, ...] = ()
    post_execute: tuple[Any, ...] = ()
    #: Monotonic guard: receives the execution and may only deny.
    guard: Callable[[Execution], Decision] | None = None
    #: Wraps the body. The only place a signal may be replaced.
    execute_wrapper: Callable[[Execution, Callable[[], Awaitable[ToolResult]]], Awaitable[ToolResult]] | None = None
    #: Renders the model-visible content from the raw value.
    project_content: Callable[[Execution, Any], str] | None = None
    #: Approval channel. Absent means any ``ask`` decision fails closed.
    approve: Callable[[Execution, str], Awaitable[bool]] | None = None


class ToolPipeline:
    """Runs one tool call through the seven stages.

    Stateless with respect to individual calls: everything about a call lives in
    its :class:`Execution`, so a cancelled or failed call leaves nothing behind
    in the pipeline itself.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        hooks: PipelineHooks | None = None,
        *,
        view: ToolView | None = None,
    ) -> None:
        self.registry = registry
        self.hooks = hooks or PipelineHooks()
        self._view = view

    def _current_view(self) -> ToolView:
        return self._view if self._view is not None else self.registry.view()

    # -- stage 1: materialize -------------------------------------------

    def materialize(
        self,
        *,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        signal: AbortSignal | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Execution:
        """Freeze the arguments and bind the tool (M3.4).

        A payload that cannot be losslessly snapshotted is refused *before* any
        policy runs, because a policy that cannot trust the arguments it is
        inspecting cannot make a sound decision about them.
        """
        try:
            assert_json_lossless(dict(arguments), what="tool execution arguments")
        except Exception as exc:  # noqa: BLE001 - becomes a result, not a raise
            definition = None
            execution = Execution(
                call_id=call_id,
                name=name,
                arguments=FrozenDict(),
                tool=None,
                definition=None,
                signal=signal or AbortSignal(),
                context=dict(context or {}),
            )
            execution.override = ToolResult.fail(
                "tool execution arguments must be losslessly JSON-serializable"
                f" ({describe(exc)})",
                code=CODE_INVALID_ARGUMENTS,
            )
            return execution

        view = self._current_view()
        definition = view.get(name)
        tool = self.registry.get(name) if definition is not None else None

        return Execution(
            call_id=call_id,
            name=name,
            arguments=deep_freeze(dict(arguments)),
            tool=tool,
            definition=definition,
            signal=signal or AbortSignal(),
            context=dict(context or {}),
        )

    # -- the pipeline ----------------------------------------------------

    async def run(self, execution: Execution) -> ToolResult:
        """Run all seven stages, always returning a result (M3.7).

        ``finalize`` is guaranteed to run exactly once for every *business*
        outcome. Even an unexpected exception is converted into an error result,
        because the caller has already committed to pairing this call with a
        result.

        One class of failure deliberately escapes: a **contract violation** by a
        policy hook (a hook that neither continues the chain nor decides, or one
        that continues twice). That is a bug in a plugin, not a business outcome,
        and converting it into an ordinary-looking tool result would hide it
        behind a plausible error message. It propagates so the loop can record a
        turn failure and the bug is visible.
        """
        try:
            return await self._run_stages(execution)
        except WaterfallError:
            # A broken hook is a programming error: loud, and never disguised as
            # a normal tool failure.
            raise
        except PipelineAbort as abort:
            return ToolResult.fail(abort.reason, code=abort.code)
        except asyncio.CancelledError:
            # The body already started, so its side effects may have happened.
            # Recording "aborted" is honest; discarding the call would not be.
            if execution.started and execution.override is not None:
                return execution.override
            raise
        except Exception as exc:  # noqa: BLE001 - scheduler must still produce a result
            log.exception("tool pipeline failed unexpectedly for %s", execution.name)
            return ToolResult.fail(
                f"tool {execution.name!r} failed: {describe(exc)}",
                code=CODE_SCHEDULER_FAILURE,
            )

    async def _run_stages(self, execution: Execution) -> ToolResult:
        # A materialization-time refusal short-circuits, but still runs the
        # post-execute stage: policies may want to observe the failure.
        if execution.override is not None:
            execution.finished = True
            return self._finalize(execution, execution.override)

        # -- stage 2: pre-execute (waterfall) ---------------------------
        decision = await self._pre_execute(execution)
        if decision.kind == "deny":
            return self._finalize(
                execution,
                ToolResult.fail(decision.reason or "denied", code=CODE_DENIED),
            )
        if decision.kind == "cancel":
            return self._finalize(
                execution,
                ToolResult.fail(
                    decision.reason or "tool call aborted before dispatch",
                    code=CODE_ABORTED_BEFORE_DISPATCH,
                ),
            )
        if decision.kind == "ask":
            approval = await self._request_approval(execution, decision.reason)
            if approval is not None:
                return self._finalize(execution, approval)

        # -- stage 3: guard (monotonic) ---------------------------------
        if self.hooks.guard is not None:
            guard_decision = self.hooks.guard(execution)
            if guard_decision.kind != "allow":
                return self._finalize(
                    execution,
                    ToolResult.fail(
                        guard_decision.reason or "denied by guard", code=CODE_DENIED
                    ),
                )

        # An unknown tool still went through pre-execute (M3: the contract says
        # so explicitly), and only now becomes a result.
        if execution.definition is None or execution.tool is None:
            return self._finalize(
                execution,
                ToolResult.fail(f'unknown tool "{execution.name}"', code=CODE_UNKNOWN_TOOL),
            )

        # -- argument validation ----------------------------------------
        validated, violation = execution.tool.validate_arguments(execution.arguments)
        if violation is not None:
            return self._finalize(
                execution,
                ToolResult.fail(
                    f"invalid arguments: {violation}", code=CODE_INVALID_ARGUMENTS
                ),
            )
        # The validated form replaces the snapshot for the body only; the frozen
        # snapshot remains what the log and any observer sees (M3.4).
        body_arguments = deep_freeze(validated)

        # -- stages 4-5: execute wrapper + body -------------------------
        self._dispatch_guard(execution)
        result = await self._execute(execution, body_arguments)

        # A failure inside the execute wrapper skips post-execute entirely: the
        # wrapper's own contract is what broke, so downstream observers must not
        # be invited to interpret its output. Signalled by the execution already
        # being finalized.
        if execution.finished and execution.override is not None:
            return execution.override

        # -- stage 6: project content -----------------------------------
        if self.hooks.project_content is not None and not result.is_error:
            try:
                projected = self.hooks.project_content(execution, result.value)
            except Exception as exc:  # noqa: BLE001
                return self._finalize(
                    execution,
                    ToolResult.fail(
                        f"tool {execution.name!r} failed to project content: "
                        f"{describe(exc)}",
                        code=CODE_BODY_ERROR,
                    ),
                )
            if isinstance(projected, str):
                result = ToolResult(
                    content=projected,
                    is_error=result.is_error,
                    value=result.value,
                    error_code=result.error_code,
                    error_info=result.error_info,
                    additional_context=result.additional_context,
                    concludes_turn=result.concludes_turn,
                )

        # -- stage 7: post-execute --------------------------------------
        return self._finalize(execution, result)

    # -- individual stages ----------------------------------------------

    async def _pre_execute(self, execution: Execution) -> Decision:
        """Run the pre-execute waterfall.

        Semantics (defined here because they were never verified upstream): a hook
        that does not call ``next()`` short-circuits the chain. Subsequent hooks
        do not run. A hook that neither continues nor decides is an error rather
        than a silent pass.
        """
        if not self.hooks.pre_execute:
            return Decision.allow()

        async def _terminal(_ctx: Execution) -> Decision:
            return Decision.allow()

        chain = compose(self.hooks.pre_execute, _terminal, name="pre-execute")
        result = await chain.run(execution)
        if isinstance(result, Decision):
            return result
        if result is None:
            return Decision.allow()
        return Decision.allow()

    async def _request_approval(
        self, execution: Execution, reason: str
    ) -> ToolResult | None:
        """Ask for approval, failing closed when no channel exists.

        No channel means denial, not permission: a policy that said "ask" has
        already declined to auto-allow, and treating a missing channel as consent
        would invert its meaning.
        """
        if self.hooks.approve is None:
            return ToolResult.fail(
                f'tool "{execution.name}" requires approval, but no approval '
                "channel is available",
                code=CODE_NO_APPROVAL_CHANNEL,
            )
        try:
            granted = await self.hooks.approve(execution, reason)
        except Exception as exc:  # noqa: BLE001 - a broken channel denies
            return ToolResult.fail(
                f"approval for tool \"{execution.name}\" failed: {describe(exc)}",
                code=CODE_APPROVAL_REJECTED,
            )
        if granted:
            return None
        return ToolResult.fail(
            f'the user rejected tool "{execution.name}"', code=CODE_APPROVAL_REJECTED
        )

    def _dispatch_guard(self, execution: Execution) -> None:
        """Check for cancellation *before* the body starts.

        This is the last safe moment to refuse: once the body runs, its side
        effects are real and the result must be recorded.
        """
        if execution.signal.aborted:
            raise PipelineAbort(
                execution.signal.cause or "tool call aborted before dispatch",
                CODE_ABORTED_BEFORE_DISPATCH,
            )

    async def _execute(
        self, execution: Execution, arguments: Mapping[str, Any]
    ) -> ToolResult:
        """Run the body, optionally inside a wrapper.

        Two guarantees are enforced here, and both were the subject of a
        dedicated spike because both are easy to get wrong:

        **The body is owned by a task.** A started body is registered
        synchronously and drained under a shield, so a cancellation arriving
        while it runs cannot destroy work whose side effects have already
        happened (M3.8). Awaiting the coroutine inline would let cancellation be
        injected at the body's next ``await`` and silently discard everything
        after it.

        **The caller's signal is re-fused.** A wrapper may substitute the signal,
        so the original is restored before the body observes it. Without that,
        "wrap the call to add a timeout" would silently also mean "remove
        cancellation" (M3.5).
        """
        assert execution.tool is not None
        tool = execution.tool

        async def _body() -> ToolResult:
            # The last check before the side effect. A wrapper runs before this,
            # so a signal it substituted -- or an abort it raised -- is seen here
            # rather than being outrun. Without this the guard upstream is the
            # only one, and a wrapper sits between it and the body.
            if execution.signal.aborted:
                raise PipelineAbort(
                    execution.signal.cause or "tool call aborted before dispatch",
                    CODE_ABORTED_BEFORE_DISPATCH,
                )

            execution.started = True
            execution.started_at = time.monotonic()
            try:
                result = await tool.body(arguments, execution)
            except asyncio.CancelledError:
                # A tool body must never convert cancellation into a result.
                # Re-raising is mandatory: swallowing it would report an abort as
                # a success, and the caller would never learn it was cancelled.
                raise
            except Exception as exc:  # noqa: BLE001 - body failure is a result
                return ToolResult.fail(describe(exc), code=CODE_BODY_ERROR)
            finally:
                execution.finished = True
                execution.finished_at = time.monotonic()
            if not isinstance(result, ToolResult):
                return ToolResult.fail(
                    f"tool \"{execution.name}\" returned an invalid result object "
                    f"({type(result).__name__})",
                    code=CODE_INVALID_OUTPUT,
                )
            return result

        async def _run_body() -> ToolResult:
            """Start the body as an owned task and wait for it under a shield.

            The scope drains on the way out, so a cancellation that arrives while
            the body is running still leaves the body finished and its outcome
            recorded rather than discarded.
            """
            async with DrainScope(name=f"tool:{execution.name}") as scope:
                scope.start(execution.call_id, _body())
                try:
                    result = await scope.wait(execution.call_id)
                except BodyCancelled:
                    # The body cancelled itself. That is an outcome to record,
                    # not the caller's abort, and the two must not be confused.
                    return ToolResult.fail("tool call aborted", code=CODE_ABORTED)
                return result

        # The caller's signal is captured before the wrapper runs, and re-fused
        # onto whatever the wrapper substituted when the body is entered.
        caller_signal = execution.signal

        def _fused_run_body() -> Any:
            """Enter the body with cancellation that cannot have been removed.

            The wrapper's signal may add cancellation, never drop it: the fused
            signal aborts when either the caller's or the wrapper's does.
            """
            if execution.signal is not caller_signal:
                execution.signal = caller_signal.fused_with(execution.signal)
            return _run_body()

        if self.hooks.execute_wrapper is None:
            try:
                return await _run_body()
            except asyncio.CancelledError:
                self._preserve_body_outcome_on_cancel(execution)
                raise

        try:
            # The caller's signal is re-fused after the wrapper has run its setup.
            #
            # A wrapper may legitimately substitute a signal to add cancellation
            # (a timeout, a scope of its own). It must not be able to *remove*
            # it, and simply restoring the original afterwards is not enough:
            # the body runs inside the wrapper, so by then the substitution has
            # already taken effect and an aborted call would execute anyway.
            #
            # Fusing means the effective signal aborts when *either* does, so a
            # wrapper can only ever narrow what is permitted.
            original = execution.signal
            try:
                return await self.hooks.execute_wrapper(execution, _fused_run_body)
            finally:
                execution.signal = original
        except asyncio.CancelledError:
            self._preserve_body_outcome_on_cancel(execution)
            raise
        except PipelineAbort:
            raise
        except Exception as exc:  # noqa: BLE001
            # A failure in the wrapper itself skips post-execute: the wrapper's
            # contract is what broke, so downstream observers must not run.
            execution.finished = True
            result = ToolResult.fail(describe(exc), code=CODE_WRAPPER_ERROR)
            return self._finalize_no_post(execution, result)

    @staticmethod
    def _preserve_body_outcome_on_cancel(execution: Execution) -> None:
        """Record a started body's outcome before the cancellation propagates.

        By the time this runs the drain has already waited for the body, so the
        side effects are real and a result exists. Storing it means the dispatcher
        can still write a ``tool/result`` for the call instead of leaving a
        dangling ``tool_call`` -- which is what a cancellation would otherwise
        produce.
        """
        if execution.override is not None:
            return
        if execution.started and execution.finished:
            execution.override = ToolResult.fail(
                "tool call aborted", code=CODE_ABORTED
            )

    # -- finalize --------------------------------------------------------

    def _finalize(self, execution: Execution, result: ToolResult) -> ToolResult:
        """Run post-execute, then commit the result exactly once (M3.7)."""
        result = self._post_execute(execution, result)
        execution.override = result
        execution.finished = True
        return result

    def _finalize_no_post(self, execution: Execution, result: ToolResult) -> ToolResult:
        """Commit without post-execute. Used only when the wrapper itself failed."""
        execution.override = result
        execution.finished = True
        return result

    def _post_execute(self, execution: Execution, result: ToolResult) -> ToolResult:
        """Apply post-execute hooks.

        A hook may replace ``content`` or ``value``, but not both -- the ambiguity
        would make the outcome depend on hook order. It may also block, which
        converts the result into feedback.
        """
        for hook in self.hooks.post_execute:
            try:
                outcome = hook(execution, result)
            except Exception as exc:  # noqa: BLE001 - a broken hook denies, not allows
                return ToolResult.fail(
                    f"post-execute hook failed: {describe(exc)}",
                    code=CODE_BODY_ERROR,
                )
            if outcome is None:
                continue
            if isinstance(outcome, ToolResult):
                result = outcome
                continue
            blocked = _blocked_text(outcome)
            if blocked is not None:
                result = ToolResult.fail(
                    blocked or "the result was withheld by a policy",
                    code=CODE_DENIED,
                )
        return result


def _blocked_text(outcome: Any) -> str | None:
    """Extract block feedback from a post-execute outcome, if it blocks."""
    kind = getattr(outcome, "kind", None)
    if kind == "block":
        return str(getattr(outcome, "feedback", "") or "")
    if isinstance(outcome, Mapping) and outcome.get("block"):
        return str(outcome.get("feedback", "") or "")
    return None


def aborted_before_dispatch(call_id: str, name: str) -> ToolResult:
    """The synthetic result for a call that never started (M3.9).

    Every dispatched-but-unstarted call must still produce a result, or the next
    request carries a dangling tool_call and the model rejects it.
    """
    return ToolResult.fail(ABORTED_BEFORE_DISPATCH_TEXT, code=CODE_ABORTED_BEFORE_DISPATCH)

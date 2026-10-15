"""The agent loop: a driver that appends to a log.

The whole runtime is this: a loop that claims input, appends the events of a
turn, and derives what to send from what it appended. Phase, context, token
counts and tool progress are all derived; if any of them were stored separately,
there would be two sources of truth and they would drift.

The shape of a turn is fixed and the order is load-bearing:

    turn/start          BEFORE claiming, so a blocked turn is a complete turn
      step/start        AFTER pre-step, so a rejected batch leaves no empty step
        system/message  only when the prompt actually changed
        user/message    only on the first attempt of the step
        request/header  the model-visible contract of this call
        developer/message  only when the tool set changed
        request/context the derived context snapshot
        -> derive() -> request hooks -> model
        assistant/message
        tool/call ... tool/result   (committed in model order)
      step/end
    turn/end            ALWAYS, on every exit path

Every exit writes ``turn/end``. A turn without an end is indistinguishable from a
crash, and recovery would have to guess.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

from ..kernel.errors import ArmError, WaterfallError, describe
from ..llm.seam import (
    ModelAdapter,
    Request,
    ToolCallRequest,
    ToolSchema,
    Usage,
    collect,
)
from ..prompt.assembler import PromptAssembler, ToolAnnouncement, developer_message_for
from ..session.events import (
    ASSISTANT_ATTEMPT,
    ASSISTANT_MESSAGE,
    CANCEL_USER,
    DEVELOPER_MESSAGE,
    ERROR_TURN_FAILURE,
    REQUEST_CONTEXT,
    REQUEST_HEADER,
    STEP_END,
    STEP_START,
    SYSTEM_MESSAGE,
    TURN_ABORTED,
    TURN_BLOCKED,
    TURN_COMPLETED,
    TURN_MAX_TOKENS,
    TURN_END,
    TURN_ERROR,
    TURN_START,
    USER_MESSAGE,
)
from ..session.inbox import Claim, Inbox, PendingInput
from ..session.session import Session
from ..plan.mode import PlanMode
from ..skills.catalog import CatalogPublisher, render_skill_content
from ..skills.registry import CatalogSnapshot, SkillRegistry
from ..tools.dispatch import CallRequest, DispatchReport, Dispatcher

if TYPE_CHECKING:  # pragma: no cover - avoids a cycle at import time
    from ..approval.service import ApprovalService
    from ..context.manager import ContextManager
    from ..sandbox.policy import SandboxPolicy

from ..tools.pipeline import AbortSignal
from ..tools.registry import ToolRegistry

log = logging.getLogger("arm.loop")

#: The runtime's phases. Exactly one is active at any moment (S1).
PHASE_IDLE = "idle"
PHASE_RUNNING = "running"
PHASE_MAINTENANCE = "maintenance"

#: Where a step-level rejection sends the turn.
STEP_ENTER = "enter"
STEP_REJECT = "reject"


class LoopError(ArmError):
    """The loop cannot proceed. Distinct from a business failure."""


@dataclass(frozen=True, slots=True)
class StepDecision:
    """What a pre-step hook decided."""

    kind: str  # "enter" | "reject"
    reason: str = ""

    @staticmethod
    def enter() -> "StepDecision":
        return StepDecision(kind=STEP_ENTER)

    @staticmethod
    def reject(reason: str) -> "StepDecision":
        return StepDecision(kind=STEP_REJECT, reason=reason)


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """The result of one turn, mirroring what was written to the log."""

    reason: str
    steps: int = 0
    appended: int = 0
    #: Whether the driver should run another turn.
    continue_loop: bool = False
    cancel_cause: str | None = None
    error: str | None = None


@dataclass
class LoopHooks:
    """The extension seams of the loop.

    ``pre_step`` may reject the whole batch. ``request`` may rewrite the prepared
    request (the last chance to influence what is sent, and therefore the last
    chance to make the log disagree with the wire -- so it is recorded).
    """

    pre_step: Callable[[Session, Claim], Awaitable[StepDecision]] | None = None
    request: Callable[[Session, Request], Awaitable[Request]] | None = None
    #: Called after each step, for progress reporting.
    on_step: Callable[[Session, int], None] | None = None


class Runtime:
    """The driver: ``while await turn()``.

    A single instance owns one session. It is not re-entrant: two concurrent
    turns on one session would interleave their appends and produce a log whose
    order depends on scheduling.
    """

    def __init__(
        self,
        session: Session,
        adapter: ModelAdapter,
        registry: ToolRegistry,
        *,
        assembler: PromptAssembler | None = None,
        hooks: LoopHooks | None = None,
        dispatcher: Dispatcher | None = None,
        context: "ContextManager | None" = None,
        skills: SkillRegistry | None = None,
        plan: PlanMode | None = None,
        approval: "ApprovalService | None" = None,
        sandbox: "SandboxPolicy | None" = None,
        scheduler: Any = None,
        subagents: Any = None,
        hook_bridge: Any = None,
        model: str = "",
        max_steps: int = 64,
    ) -> None:
        from ..tools.pipeline import ToolPipeline

        self.session = session
        self.adapter = adapter
        self.registry = registry
        self.assembler = assembler or PromptAssembler()
        self.hooks = hooks or LoopHooks()
        self.model = model or getattr(adapter, "model", "") or "unknown"
        self.max_steps = max_steps
        self.inbox = Inbox(session)
        # External hooks attach to the pipeline's existing pre-execute waterfall
        # rather than to a mechanism of their own: the runtime already has the
        # interception point, and a second one would be a second place for policy
        # to live.
        self.hook_bridge = hook_bridge
        pipeline_hooks = None
        if hook_bridge is not None or sandbox is not None or approval is not None:
            from ..tools.pipeline import PipelineHooks

            pre_execute: tuple[Any, ...] = ()
            # Approval runs before the external hook bridge: a hook is an
            # addition the user installed, and it should not be able to approve
            # on the user's behalf by running before the gate.
            if approval is not None:
                from ..approval.service import gating_hook

                pre_execute = (
                    gating_hook(approval, session_id=session.session_id),
                )
            if hook_bridge is not None:
                from ..hooks.bridge import pipeline_hook

                pre_execute = (*pre_execute, pipeline_hook(hook_bridge))

            # The sandbox is enforced by an execute wrapper rather than by each
            # tool: the wrapper is on the mandatory path, so a tool cannot be
            # added that forgets to check. A policy that lives in the tools is a
            # policy enforced in five places and forgotten in the sixth.
            execute_wrapper = None
            if sandbox is not None:
                from ..sandbox.policy import enforcement_for

                execute_wrapper = enforcement_for(sandbox)

            pipeline_hooks = PipelineHooks(
                pre_execute=pre_execute, execute_wrapper=execute_wrapper
            )
        self.dispatcher = dispatcher or Dispatcher(
            ToolPipeline(registry, hooks=pipeline_hooks)
        )
        # Context management is optional but, when present, runs before every
        # request: the loop must not send a request it knows is over budget.
        self.context = context
        # Skills are optional too. When present, the catalogue is published as a
        # message rather than as a system-prompt segment: the prompt is a single
        # string that is only re-recorded when it changes, so a catalogue that
        # changes as skills are installed would either invalidate the prompt
        # constantly or silently go stale.
        self.skills = skills
        self._catalog = CatalogPublisher()
        # Plan mode contributes a prompt section and nothing else: it never
        # changes the tool catalogue, so a tool call can never fail because of
        # the mode the session happens to be in.
        self.plan = plan or PlanMode(session)
        # Approval and sandbox are held rather than injected into the pipeline:
        # the pipeline asks a narrow question and the policy answers it, so the
        # loop needs a reference only to report the policy in the prompt and to
        # let a caller change it.
        self.approval = approval
        self.sandbox = sandbox
        # The scheduler is held rather than driven from here. Firing a reminder
        # means persisting it into a session, and the loop is already the thing
        # driving one -- a scheduler inside the loop would have to reach back out
        # for the state it is in the middle of mutating.
        self.scheduler = scheduler
        # Held so ``shutdown`` can stop children that were started and never
        # finished. A subagent started through ``start`` and abandoned by its
        # caller would otherwise keep running with nothing left to stop it.
        self.subagents = subagents

        self.phase = PHASE_IDLE
        self._announced = ToolAnnouncement()
        self._last_prompt: str | None = None
        self._turn_counter = self._count_turns()
        self._abort = AbortSignal()
        self._running = False

        # A restored session already recorded a system prompt and a tool
        # announcement. Starting from scratch would re-record both, duplicating
        # them in the transcript: the model would read its instructions twice and
        # see the tool list announced again for no reason.
        self._restore_announcement_state()

    def _restore_announcement_state(self) -> None:
        """Seed prompt/announcement state from what the log already says.

        Derived from the log rather than passed in, so a session resumed in a
        fresh process behaves the same as one that never stopped.
        """
        for event in self.session:
            if event.type == SYSTEM_MESSAGE:
                message = event.data.get("message") or {}
                content = message.get("content") or ()
                text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
                if text:
                    self._last_prompt = text
            elif event.type == REQUEST_HEADER:
                announced = event.data.get("tools")
                if announced is not None:
                    self._announced.update([str(name) for name in announced])

    # -- state derived from the log -------------------------------------

    def _count_turns(self) -> int:
        """Number of turns already started, read from the log (S3)."""
        return sum(1 for e in self.session if e.type == TURN_START)

    @property
    def current_turn(self) -> int:
        return self._turn_counter

    # -- input -----------------------------------------------------------

    def submit(self, text: str, *, kind: str = "next-turn") -> int:
        """Record a user input and return its seq.

        The input becomes an event immediately. It is not queued in memory, so a
        crash between submission and processing loses nothing: the input is
        already durable and the next driver will claim it.

        The system prompt is recorded *before* the first input, because the
        prompt is the head of the message sequence. Recording it lazily inside
        the step would place it after the user's message and put the transcript
        in an order no model API accepts.
        """
        # The prompt is recorded here, before the input, because the system
        # prompt is the head of the surface: recording it inside the step would
        # place it after the user's message and put the transcript in an order
        # no model API accepts.
        #
        # This is still correct for plan mode, because the *effective* mode
        # already reflects a pending choice -- so a user who selects a mode and
        # then submits sees the policy in the prompt of the turn that follows.
        self._ensure_prompt_recorded()
        event = self.session.append(
            USER_MESSAGE,
            {
                "message": {"role": "user", "content": [{"type": "text", "text": text}]},
                "kind": kind,
                "surface_op": "append",
            },
        )

        # A `/name` gesture loads that skill for the user.
        #
        # This is the *only* route to a skill marked ``disable-model-invocation``.
        # Without it that flag would make a skill unreachable by the model and by
        # the user both -- the opposite of what it is for, and the flag's whole
        # purpose would not exist.
        self._apply_skill_gestures(text)
        return event.seq

    def _apply_skill_gestures(self, text: str) -> None:
        """Load any skills the user invoked by name.

        The body is injected as a context message rather than a user message: it
        is material for the model to read, not work the user is asking for. That
        distinction is what keeps a loaded skill from being claimed as new input
        on the next turn.
        """
        if self.skills is None or "/" not in text:
            return

        snapshot = self.skills.snapshot()
        for summary in self.skills.find_gesture(text, snapshot):
            definition = self.skills.get(summary.name)
            if definition is None:
                continue
            self.session.append(
                USER_MESSAGE,
                {
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": render_skill_content(definition),
                            }
                        ],
                    },
                    "surface_op": "append",
                    "source": "skill-invocation",
                    "context_only": True,
                    "skill": summary.name,
                },
            )

    def _ensure_prompt_recorded(self) -> None:
        """Record the system prompt, replacing the previous one when it changed.

        The replacement matters and is easy to get wrong. Appending a second
        ``system/message`` would leave the *first* prompt on the surface, so the
        model would see both -- and, worse, the stale one would come first. A
        prompt that says "plan mode is active" followed by one that says nothing
        about plan mode is a contradiction the model has to resolve on its own.

        The surface contract already provides the mechanism: node 0 may be
        rewritten by a same-type replacement that covers exactly that node
        (M1.10). That rule exists precisely so the prompt can be updated without
        either duplicating it or being swept away by an unrelated rewrite.
        """
        prompt = self._render_prompt()
        if not prompt or prompt == self._last_prompt:
            return

        existing = self._system_node_seq()
        if existing is None:
            self.session.append(
                SYSTEM_MESSAGE,
                {
                    "message": {
                        "role": "system",
                        "content": [{"type": "text", "text": prompt}],
                    },
                    "surface_op": "append",
                },
            )
        else:
            self.session.append(
                SYSTEM_MESSAGE,
                {
                    "message": {
                        "role": "system",
                        "content": [{"type": "text", "text": prompt}],
                    },
                    "surface_op": {
                        "op": "replace",
                        "start_seq": existing,
                        "end_seq": existing,
                    },
                    "source_event_seqs": [existing],
                },
            )
        self._last_prompt = prompt

    def _system_node_seq(self) -> int | None:
        """The seq of the system message currently at the head, if any."""
        nodes = self.session.surface_nodes()
        if not nodes:
            return None
        first = self.session.event_at(nodes[0])
        return first.seq if first.type == SYSTEM_MESSAGE else None

    def _render_prompt(self) -> str:
        """The system prompt for the step about to run.

        The plan policy is a dynamic segment: it depends on the mode, which can
        change between steps. Folding it in here rather than registering it as a
        static segment means the mode is read at the moment the prompt is built.
        """
        base = self.assembler.render()
        policy = self.plan.prompt_section()
        if not policy:
            return base
        if not base:
            return policy
        # The policy goes last so it reads as the most specific instruction.
        return f"{base}\n\n{policy}"

    def request_abort(self, cause: str = CANCEL_USER) -> None:
        """Request cancellation. The in-flight step finishes what it started."""
        self._abort.abort(cause)

    def clear_abort(self) -> None:
        self._abort = AbortSignal()

    # -- the driver ------------------------------------------------------

    async def run_until_idle(self, *, max_turns: int = 100) -> list[TurnOutcome]:
        """Drive turns until there is no claimable input (S2)."""
        if self._running:
            raise LoopError("the runtime is already driving turns")

        self._running = True
        outcomes: list[TurnOutcome] = []
        try:
            turns = 0
            while turns < max_turns:
                outcome = await self._turn_locked()
                outcomes.append(outcome)
                turns += 1
                if not outcome.continue_loop:
                    break
            else:
                raise LoopError(
                    f"exceeded {max_turns} turns without going idle; the loop is "
                    "not making progress"
                )
        finally:
            self._running = False
            self.phase = PHASE_IDLE
        return outcomes

    async def turn(self) -> TurnOutcome:
        """Run exactly one turn. Always writes ``turn/end`` (S6)."""
        if self._running:
            raise LoopError("turn() is not re-entrant")
        self._running = True
        try:
            return await self._turn_locked()
        finally:
            self._running = False

    async def _turn_locked(self) -> TurnOutcome:
        """The body of a turn, called with the re-entrancy guard already held."""
        self.phase = PHASE_RUNNING

        # Due reminders are delivered before the turn reads its input, so a
        # reminder that arrived while the session was idle is part of the work
        # this turn sees. Delivering after claiming would mean the reminder waits
        # a whole turn -- and the reminder exists precisely to not wait.
        self._fire_due_reminders()
        appended_before = len(self.session)

        claim = self.inbox.claim()

        # turn/start precedes the claim, so a rejected or empty turn is still a
        # structurally complete turn in the log (S5).
        turn_index = self._turn_counter
        self._turn_counter += 1
        self.session.append(
            TURN_START,
            {
                "turn": turn_index,
                "claimed_seqs": [item.seq for item in claim.all_claimed()],
                "kind": "turn" if claim.turn is not None else "step-only",
            },
        )

        if claim.empty:
            return self._end_turn(
                turn_index, TURN_COMPLETED, appended_before, steps=0, continue_loop=False
            )

        # A step-only claim (no turn starter) still runs: the input asked to join
        # the current work, and there is no reason to refuse it.
        if self._abort.aborted:
            return self._end_turn(
                turn_index,
                TURN_ABORTED,
                appended_before,
                steps=0,
                cancel_cause=self._abort.cause,
                continue_loop=False,
            )

        # -- pre-step: may reject the entire batch ----------------------
        if self.hooks.pre_step is not None:
            try:
                decision = await self.hooks.pre_step(self.session, claim)
            except Exception as exc:  # noqa: BLE001 - a broken hook blocks, not runs
                return self._end_turn(
                    turn_index,
                    TURN_BLOCKED,
                    appended_before,
                    steps=0,
                    error=f"pre-step hook failed: {describe(exc)}",
                    continue_loop=False,
                )
            if decision.kind == STEP_REJECT:
                # No step is written, so no model call happens (S7).
                return self._end_turn(
                    turn_index,
                    TURN_BLOCKED,
                    appended_before,
                    steps=0,
                    error=decision.reason,
                    continue_loop=False,
                )

        # -- the step loop ----------------------------------------------
        steps = 0
        pending: tuple[PendingInput, ...] = claim.all_claimed()

        while steps < self.max_steps:
            steps += 1
            step_index = steps - 1

            self.session.append(
                STEP_START,
                {"turn": turn_index, "step": step_index, "inputs": [i.seq for i in pending]},
            )

            try:
                outcome = await self._run_step(
                    turn_index, step_index, pending, first=(steps == 1)
                )
            except WaterfallError:
                # A broken hook is a bug, not a turn outcome: record it as a turn
                # failure and re-raise so it is not mistaken for a normal error.
                self.session.append(
                    ERROR_TURN_FAILURE,
                    {"turn": turn_index, "step": step_index, "message": "hook contract violated"},
                )
                self.session.append(STEP_END, {"turn": turn_index, "step": step_index})
                self._end_turn(
                    turn_index, TURN_ERROR, appended_before, steps=steps, continue_loop=False
                )
                raise
            except asyncio.CancelledError:
                self.session.append(STEP_END, {"turn": turn_index, "step": step_index})
                self._end_turn(
                    turn_index,
                    TURN_ABORTED,
                    appended_before,
                    steps=steps,
                    cancel_cause="cancelled",
                    continue_loop=False,
                )
                raise

            self.session.append(STEP_END, {"turn": turn_index, "step": step_index})
            if self.hooks.on_step is not None:
                self.hooks.on_step(self.session, step_index)

            if outcome is None:
                # The step produced no model call (should not happen for a real
                # step); treat it as the end of this turn's work.
                break

            if outcome.kind == "error":
                self.session.append(
                    ERROR_TURN_FAILURE,
                    {
                        "turn": turn_index,
                        "step": step_index,
                        "message": outcome.message or "model error",
                        "code": outcome.error_code,
                    },
                )
                return self._end_turn(
                    turn_index,
                    TURN_ERROR,
                    appended_before,
                    steps=steps,
                    error=outcome.message,
                    continue_loop=False,
                )

            if outcome.kind == "aborted":
                return self._end_turn(
                    turn_index,
                    TURN_ABORTED,
                    appended_before,
                    steps=steps,
                    cancel_cause=outcome.cancel_cause or CANCEL_USER,
                    continue_loop=False,
                )

            if outcome.concludes_turn:
                # A truncated answer is not a completed turn. The distinction is
                # what lets a caller -- or a subagent's caller -- tell a finished
                # result from half of one.
                reason = (
                    TURN_MAX_TOKENS
                    if outcome.finish_reason == "length"
                    else TURN_COMPLETED
                )

                # The model says it is finished; a hook may disagree. This is the
                # `Stop` point in both hook dialects, and without it a hook that
                # decided "the task is not actually done" had no way to say so --
                # the point existed in the vocabulary and nothing consulted it.
                if await self._hook_wants_another_step(reason, steps):
                    # The hook asked for another step. Bounded by max_steps, so a
                    # hook that always disagrees cannot loop forever.
                    continue

                return self._end_turn(
                    turn_index, reason, appended_before, steps=steps,
                    continue_loop=False,
                )

            # Tool results were recorded this step, so the model must be given a
            # chance to read them. This is the normal multi-step turn: call a
            # tool, then act on its output.
            if outcome.needs_another_step:
                continue

            # More input may have arrived while this step ran; it joins the same
            # turn rather than starting a new one (S2).
            next_claim = self.inbox.claim()
            if next_claim.empty:
                # This is the ordinary way a turn ends -- a text answer with no
                # tool call -- so two things belong here that were first wired
                # into only the other path:
                #
                #   * the turn-stopping hook, without which a `Stop` hook did
                #     nothing at all;
                #   * the truncation reason, without which a cut-off answer was
                #     reported as a completed one.
                #
                # Both bugs were the same mistake: handling one of two paths.
                reason = (
                    TURN_MAX_TOKENS
                    if outcome.finish_reason == "length"
                    else TURN_COMPLETED
                )
                if await self._hook_wants_another_step(reason, steps):
                    continue
                return self._end_turn(
                    turn_index,
                    reason,
                    appended_before,
                    steps=steps,
                    continue_loop=False,
                )
            pending = next_claim.all_claimed()

        return self._end_turn(
            turn_index,
            TURN_COMPLETED,
            appended_before,
            steps=steps,
            continue_loop=False,
        )

    async def _hook_wants_another_step(self, reason: str, steps: int) -> bool:
        """Ask the hooks whether the turn is really finished.

        The `Stop` point in both hook dialects. A hook that decides the task is
        not done can keep the turn going, bounded by ``max_steps`` so a hook that
        always disagrees cannot loop forever.

        A session with no hooks pays nothing: the check short-circuits before any
        work.
        """
        if self.hook_bridge is None:
            return False
        try:
            decision = await self.hook_bridge.turn_stopping(reason, steps)
        except Exception as exc:  # noqa: BLE001 - a hook fault is not a turn fault
            log.warning("the turn-stopping hook failed: %s", exc)
            return False
        return bool(decision.stop)

    async def shutdown(self) -> list[str]:
        """Stop everything this runtime started.

        Called when a session ends. Without it, a subagent that was started and
        whose caller never awaited it keeps running -- and the process exits with
        it still holding whatever it holds.

        Returns the names of children that would not settle, so a caller can
        report them rather than assume a clean exit.
        """
        orphans: list[str] = []
        if self.subagents is not None:
            try:
                orphans = await self.subagents.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                log.warning("shutting down subagents failed: %s", exc)
                orphans = ["<subagent shutdown failed>"]
        if self.scheduler is not None:
            try:
                self.scheduler.store._flush()
            except Exception as exc:  # noqa: BLE001
                log.warning("flushing the schedule failed: %s", exc)
        return orphans

    def _fire_due_reminders(self) -> None:
        """Deliver any reminder whose time has arrived.

        Failures are swallowed deliberately: a scheduler problem must not stop a
        turn. The reminder stays in the store, so the next turn or the next
        process retries it -- which is the whole reason the store is durable and
        separate.
        """
        if self.scheduler is None:
            return
        try:
            receipts = self.scheduler.drive()
        except Exception as exc:  # noqa: BLE001 - a scheduler fault is not a turn fault
            log.warning("could not deliver due reminders: %s", exc)
            return
        if receipts:
            log.info("delivered %d reminder(s)", len(receipts))

    def _publish_catalog(self, turn_index: int, step_index: int) -> None:
        """Publish the skill catalogue as a message, when its entries changed.

        Publishing is driven by a digest over the *entries*, so editing the
        wrapper prose does not make every session resend its whole catalogue.
        """
        if self.skills is None:
            return

        snapshot = self.skills.snapshot()
        visible = self.skills.model_visible(snapshot)
        # The digest is taken over exactly what gets published -- the *visible*
        # subset, not the whole snapshot.
        #
        # Digesting everything and publishing a subset means a change that only
        # affects visibility is invisible to the digest: flipping a skill to
        # non-model-invocable leaves the catalogue offering a skill the tool then
        # refuses. Hashing what is published is the only way for "the catalogue
        # changed" and "the digest changed" to mean the same thing.
        digest = self.skills.catalog_digest(CatalogSnapshot(skills=visible))

        # An incomplete snapshot is not published. It is not wrong, but it is
        # partial, and writing it into the transcript would make a transient
        # provider failure look like a permanent catalogue.
        if not snapshot.complete:
            return

        message = self._catalog.update(snapshot, visible, digest)
        if message is None:
            return

        self.session.append(
            USER_MESSAGE,
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": message}],
                },
                "surface_op": "append",
                "source": "skill-catalog",
                # Marks this as context the model should read, not work waiting
                # to be done. The inbox skips it, so a republished catalogue
                # cannot become new pending input.
                "context_only": True,
                "turn": turn_index,
                "step": step_index,
                "entries": [[s.name, s.description] for s in visible],
            },
        )

    def _catalogue_on_surface(self) -> bool:
        """Whether the published catalogue is still visible to the model.

        Asked of the surface rather than inferred from "did compaction run",
        because those are different questions: compaction may leave the catalogue
        untouched, and republishing in that case is pure noise.
        """
        for seq in self.session.surface_nodes():
            event = self.session.event_at(seq)
            if event.data.get("source") == "skill-catalog":
                return True
        return False

    def _end_turn(
        self,
        turn_index: int,
        reason: str,
        appended_before: int,
        *,
        steps: int,
        continue_loop: bool,
        cancel_cause: str | None = None,
        error: str | None = None,
    ) -> TurnOutcome:
        """Write ``turn/end`` -- the last durable event of every turn (S6)."""
        data: dict[str, Any] = {"turn": turn_index, "reason": reason, "steps": steps}
        if cancel_cause is not None:
            data["cancel_cause"] = cancel_cause
        if error is not None:
            data["error"] = error
        self.session.append(TURN_END, data)
        self.phase = PHASE_IDLE

        # Another turn is warranted if more input arrived.
        more = not self.inbox.is_empty()
        return TurnOutcome(
            reason=reason,
            steps=steps,
            appended=len(self.session) - appended_before,
            continue_loop=continue_loop or (more and reason == TURN_COMPLETED),
            cancel_cause=cancel_cause,
            error=error,
        )

    # -- one step --------------------------------------------------------

    async def _run_step(
        self,
        turn_index: int,
        step_index: int,
        pending: Sequence[PendingInput],
        *,
        first: bool,
    ) -> "_StepOutcome | None":
        """Append the step's events, call the model, dispatch its tool calls."""
        view = self.registry.view()

        # A prompt that changed since the last input is recorded here, at the
        # point it becomes relevant -- still ahead of the request it describes.
        self._ensure_prompt_recorded()

        # The user input was already recorded by ``submit``. Appending it again
        # here would put the same question in the transcript twice, so the loop
        # only tracks which inputs this step is acting on (recorded in
        # ``step/start`` and ``turn/start``).
        del first, pending  # the parameters document intent; nothing to append

        # -- request/header: the contract of this call -------------------
        tool_names = [d.name for d in view.visible]
        added, removed = self._announced.diff(tool_names)
        # A pending mode choice commits here, after the step was accepted -- a
        # rejected step must not consume the user's choice.
        self.plan.commit_pending()

        # The change notice is computed *before* the header is written, because
        # the header is what records "the model has been told". Reading it after
        # the write would always find the new value already in place, and the
        # notice would never fire.
        notice = self.plan.notification()

        self.session.append(
            REQUEST_HEADER,
            {
                "turn": turn_index,
                "step": step_index,
                "model": self.model,
                "tools": tool_names,
                **self.plan.header_fields(),
            },
        )

        # The mode change is announced as a message rather than folded into the
        # prompt, because the prompt is re-recorded only when it changes and a
        # mode toggle would otherwise be invisible in the transcript.
        if notice is not None:
            self.session.append(
                USER_MESSAGE,
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": notice}],
                    },
                    "surface_op": "append",
                    "source": "plan-mode",
                    "context_only": True,
                },
            )

        # -- developer/message: only when the tool set actually changed --
        announcement = developer_message_for(added, removed)
        if announcement is not None:
            self.session.append(
                DEVELOPER_MESSAGE,
                {
                    "message": {
                        "role": "developer",
                        "content": [{"type": "text", "text": announcement}],
                    },
                    "surface_op": "append",
                },
            )
            self._announced.update(tool_names)

        # -- skill catalogue: published before the request it describes ----
        self._publish_catalog(turn_index, step_index)

        # -- context pressure: relieve it before deriving the request -----
        # This runs *before* the context snapshot is recorded, so the log's
        # record of what was sent reflects the reduction that was actually
        # applied rather than the state that preceded it.
        if self.context is not None:
            report = await self.context.relieve()
            if report.acted:
                self.session.append(
                    REQUEST_CONTEXT,
                    {
                        "turn": turn_index,
                        "step": step_index,
                        "context_management": {
                            "spilled": report.spilled,
                            "pruned": report.pruned,
                            "compacted": report.compacted,
                            "measured": report.measured,
                            "threshold": report.threshold,
                            "reason": report.reason,
                        },
                    },
                )

            # Compaction can shadow the catalogue message, taking the skill list
            # out of what the model can see. The publisher still believes it
            # published, so it would never republish -- and the model would be
            # left unable to name a single skill for the rest of the session.
            #
            # The reset is conditional on the catalogue having *actually left the
            # surface*, not merely on compaction having run. Resetting whenever
            # compaction applied would republish the catalogue on every step of a
            # session that stays over budget -- and since the catalogue is a
            # surface message, each republication would be a new node the next
            # step has to account for. Checking the fact rather than the trigger
            # is what keeps the two in step.
            if report.compacted and not self._catalogue_on_surface():
                self._catalog.reset()

        # -- request/context: the derived snapshot -----------------------
        messages = self.session.derive()
        self.session.append(
            REQUEST_CONTEXT,
            {
                "turn": turn_index,
                "step": step_index,
                "message_count": len(messages),
                "content_generation": self.session.surface.content_generation,
            },
        )

        request = Request(
            messages=tuple(messages),
            tools=tuple(
                ToolSchema(
                    name=d.name,
                    description=d.description,
                    parameters=d.parameters,
                )
                for d in view.visible
            ),
            model=self.model,
        )

        # The request hook is the last chance to change what is sent. Because the
        # log records the derived context *before* this hook runs, a hook that
        # rewrites the request makes the log disagree with the wire -- so the
        # rewrite is itself recorded.
        if self.hooks.request is not None:
            rewritten = await self.hooks.request(self.session, request)
            if rewritten is not request and rewritten.messages != request.messages:
                self.session.append(
                    REQUEST_CONTEXT,
                    {
                        "turn": turn_index,
                        "step": step_index,
                        "rewritten": True,
                        "message_count": len(rewritten.messages),
                    },
                )
            request = rewritten

        # -- call the model ----------------------------------------------
        result = await collect(self.adapter, request)

        if result.finish_reason == "error":
            return _StepOutcome(
                kind="error",
                message=result.error or "model call failed",
                error_code=result.error_code,
            )

        # -- record the assistant turn -----------------------------------
        tool_calls = tuple(result.tool_calls)
        interrupted = result.finish_reason == "length"

        if result.text or tool_calls:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": (
                    [{"type": "text", "text": result.text}] if result.text else []
                ),
            }
            if tool_calls:
                message["tool_calls"] = [
                    {"id": c.id, "name": c.name, "arguments": dict(c.arguments)}
                    for c in tool_calls
                ]
            if interrupted:
                message["interrupted"] = True
            # The usage rides on the message event, because the attempt event is
            # only written when there is no content. Recording usage only there
            # meant every answer that actually said something discarded its token
            # counts -- which is most of them.
            self.session.append(
                ASSISTANT_MESSAGE,
                {
                    "message": message,
                    "surface_op": "append",
                    "usage": _usage_record(result.usage),
                },
            )
        else:
            # No content at all: record the attempt rather than an empty message,
            # so the transcript never carries a contentless assistant node.
            self.session.append(
                ASSISTANT_ATTEMPT,
                {
                    "turn": turn_index,
                    "step": step_index,
                    "finish_reason": result.finish_reason,
                    "usage": _usage_record(result.usage),
                },
            )

        if not tool_calls:
            return _StepOutcome(
                kind="ok",
                concludes_turn=False,
                finish_reason=result.finish_reason,
            )

        # -- dispatch tool calls -----------------------------------------
        report = await self._dispatch(tool_calls, turn_index, step_index)

        if report.aborted:
            return _StepOutcome(
                kind="aborted", cancel_cause=report.abort_cause or CANCEL_USER
            )

        # Tool results were just recorded, so the model has not yet had a chance
        # to react to them. The step loop must run again -- ending the turn here
        # would leave the tool output unread and the task unfinished.
        return _StepOutcome(
            kind="ok",
            concludes_turn=report.concludes_turn,
            needs_another_step=True,
            finish_reason=result.finish_reason,
        )

    async def _dispatch(
        self,
        tool_calls: Sequence[ToolCallRequest],
        turn_index: int,
        step_index: int,
    ) -> DispatchReport:
        """Record the calls, run them, and commit results in model order (M3.9)."""
        # tool/call events are appended up front, in model order, so that the
        # calls exist in the log before any body can run and fail.
        call_seqs: list[int] = []
        for index, call in enumerate(tool_calls):
            event = self.session.append_tool_call(
                call_id=call.id,
                name=call.name,
                arguments=call.arguments,
                step=step_index,
                index=index,
            )
            call_seqs.append(event.seq)

        requests = [
            CallRequest(call_id=c.id, name=c.name, arguments=c.arguments)
            for c in tool_calls
        ]
        report = await self.dispatcher.dispatch(
            requests,
            signal=self._abort,
            context={"turn": turn_index, "step": step_index},
        )

        # Results are committed in the order the model asked for them, each
        # referencing the call event it answers.
        for index, outcome in enumerate(report.outcomes):
            call_seq = call_seqs[index] if index < len(call_seqs) else -1
            self.session.append_tool_result(
                call_seq=call_seq,
                call_id=outcome.call_id,
                message=outcome.result.to_message(
                    call_id=outcome.call_id, name=outcome.name
                ).to_record(),
                is_error=outcome.result.is_error,
                info=(
                    {"code": outcome.result.error_code}
                    if outcome.result.error_code
                    else None
                ),
            )

        # Spill oversized results the moment they land, before the next request
        # is assembled. Doing it here rather than only at the pressure check is
        # what stops pressure from building in the first place.
        if self.context is not None:
            self.context.spill_now()

        return report


@dataclass(frozen=True, slots=True)
class _StepOutcome:
    """Internal: how a step ended, before it becomes a turn reason."""

    kind: str  # "ok" | "error" | "aborted"
    concludes_turn: bool = False
    message: str | None = None
    error_code: str | None = None
    cancel_cause: str | None = None
    #: Whether the step loop should run again within this turn. Set when tool
    #: results were recorded and the model has not yet seen them.
    needs_another_step: bool = False
    #: The provider's own termination reason, when it gave one. Carried through so
    #: a truncated answer can be recorded as ``max-tokens`` rather than as a
    #: completion -- the two need different handling, and only the provider knows
    #: which happened.
    finish_reason: str = ""


def _usage_record(usage: Usage | None) -> dict[str, Any]:
    """Render usage for the log, preserving "not reported" as absent."""
    if usage is None:
        return {}
    record: dict[str, Any] = {}
    if usage.prompt_tokens is not None:
        record["prompt_tokens"] = usage.prompt_tokens
    if usage.completion_tokens is not None:
        record["completion_tokens"] = usage.completion_tokens
    if usage.total_tokens is not None:
        record["total_tokens"] = usage.total_tokens
    return record

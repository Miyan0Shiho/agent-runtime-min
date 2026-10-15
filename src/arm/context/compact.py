"""Compaction: replace a span of history with a summary, as a logged transaction.

The important idea is that compaction is **not a deletion**. It is an ordinary
``replace`` operation, validated by exactly the same surface rules as any other
rewrite. The original events stay in the log; they simply stop being reachable
from the surface. That is what makes "compact, then audit, then replay" possible
at all.

The transaction is ordered so that a crash at any point leaves a *recognisable*
state rather than a half-applied one:

    compaction/start     the lock. A marker in the log, not a memory lock, so a
                         crash leaves detectable evidence of an orphan.
      (await the model)  the only long suspension in the whole sequence
    compaction/summary   the summary text, recorded before it is applied
    user/message{replace} the replacement itself
    compaction/end       the commit

Between ``summary``, the replace, and ``end`` there is no ``await``. If a crash
landed between them, the log would describe a summary that was computed but never
applied -- recoverable, but only if the three are known to be adjacent.

Two invariants do most of the work:

* **A summary must be smaller than what it replaces.** Compacting and ending up
  larger is not a rare edge case, it is a failure, and it must not be written.
* **A compaction may not cut a tool call from its result.** Truncating the middle
  of a pairing produces a transcript the model rejects outright, so the span
  boundaries are chosen to be pairing-balanced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from ..kernel.errors import ArmError, describe
from ..session.events import (
    COMPACTION_END,
    COMPACTION_START,
    COMPACTION_SUMMARY,
    TOOL_CALL,
    TOOL_RESULT,
    USER_MESSAGE,
)
from ..session.messages import Block, Message
from ..session.session import Session
from ..session.surface import Surface
from .meter import Meter, Thresholds

log = logging.getLogger("arm.context.compact")

#: The instruction given to the model when summarizing.
#:
#: The section structure is kept because it is the part that was earned through
#: use: a summary that always answers the same questions is one a reader can
#: trust, whereas a free-form one may omit exactly the detail that mattered. The
#: wording is ours; the shape is not.
SUMMARY_INSTRUCTION = """Summarize the conversation above so that work can continue without it.

Preserve, in this order, and omit anything that does not apply:

1. The user's original request, in their own terms.
2. What has been established as fact (decisions, constraints, findings).
3. Files and paths that were created or modified, and what changed.
4. Commands that were run and their outcomes.
5. What is still in progress, and the immediate next step.
6. Anything that was tried and failed, so it is not retried blindly.

Rules:
- Be specific. Keep names, paths and identifiers exactly as written.
- Do not invent details that are not in the conversation.
- Do not address the user. This is a note to your future self.
- Prefer concrete facts over narrative.
"""


class CompactionError(ArmError):
    """A compaction could not be performed. The session is unchanged."""


class SurfaceChangedError(CompactionError):
    """The surface moved during the summary call.

    Compacting against a stale range would shadow content the caller never
    inspected, so the transaction is abandoned instead.
    """


@dataclass
class CompactionResult:
    """What a compaction actually did."""

    performed: bool
    start_seq: int | None = None
    end_seq: int | None = None
    shadowed: tuple[int, ...] = ()
    summary: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    reason: str = ""

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


#: Produces a summary for the given messages. Injected so compaction can be
#: tested without a model, and so the summarizer can be swapped independently.
Summarizer = Callable[[Sequence[Message]], Awaitable[str]]


@dataclass
class Compactor:
    """Performs summary compaction against a session."""

    session: Session
    meter: Meter
    thresholds: Thresholds
    summarize: Summarizer
    #: How many summarization attempts are allowed before giving up. One attempt
    #: is often not enough: the first summary may still leave the session over
    #: budget.
    max_attempts: int = 3

    # -- deciding --------------------------------------------------------

    def needed(self) -> bool:
        """Whether the route price has crossed the threshold.

        The **route** price is used here, not the heuristic one: the threshold is
        about what the current model will actually charge, and the same history
        costs different amounts under different tokenizers.
        """
        try:
            threshold = self.thresholds.threshold_tokens
        except ValueError:
            # Capacity not declared (E1). The documented degradation: warn once
            # and continue, because "configuration incomplete" must not block
            # work -- the provider's own overflow error can still be recovered.
            return False
        return self.meter.route_total >= threshold

    # -- span selection --------------------------------------------------

    def select_span(self) -> tuple[int, int, tuple[int, ...]] | None:
        """Choose the surface range to replace.

        The span runs from just after the protected head to wherever the retained
        tail begins. It is then *adjusted* so that:

        * node 0 (the system prompt) is never included, and
        * the span is tool-pairing balanced at both ends.

        An unbalanced boundary is the one thing that reliably breaks the next
        request, so a span that cannot be balanced is rejected outright rather
        than written and hoped for.
        """
        surface = self.session.surface
        nodes = list(surface.nodes)
        if len(nodes) < 3:
            return None

        retain = self.thresholds.retain_tokens_value

        # Walk backwards accumulating price until the retain budget is filled.
        # ``cut`` is the index of the first node that must be *kept*; everything
        # before it is a compaction candidate.
        accumulated = 0
        cut: int | None = None
        for index in range(len(nodes) - 1, -1, -1):
            accumulated += self.meter.heuristic_of(nodes[index])
            if accumulated >= retain:
                cut = index
                break

        if cut is None:
            # Even the whole history fits inside the retain budget, so there is
            # nothing that compaction could usefully remove.
            return None

        start_index = 1  # never include the system prompt at node 0
        end_index = cut - 1
        if end_index <= start_index:
            return None

        balanced = self._balance(nodes, start_index, end_index)
        if balanced is None:
            return None
        start_index, end_index = balanced
        if end_index <= start_index:
            return None

        shadowed = tuple(nodes[start_index : end_index + 1])
        return nodes[start_index], nodes[end_index], shadowed

    def _balance(
        self, nodes: Sequence[int], start_index: int, end_index: int
    ) -> tuple[int, int] | None:
        """Move the span boundaries outward until tool pairs are whole.

        Moving *outward* rather than inward is deliberate: shrinking the span
        would leave a tool call inside the retained region whose result is inside
        the compacted region (or the reverse), and that is the transcript the
        model rejects.
        """
        # A call at or after the start with its result before it would be split.
        for index in range(start_index, end_index + 1):
            event = self.session.event_at(nodes[index])
            if event.type == TOOL_RESULT:
                call_seq = event.data.get("call_seq")
                if isinstance(call_seq, int) and call_seq >= 0:
                    try:
                        call_index = nodes.index(call_seq)
                    except ValueError:
                        return None  # the call was already shadowed
                    if call_index > end_index:
                        return None
                    if call_index < start_index:
                        # The call sits before the span; extend to include it.
                        start_index = call_index

        # A call inside the span whose result is outside it must be included too.
        for index in range(start_index, end_index + 1):
            event = self.session.event_at(nodes[index])
            if event.type != TOOL_CALL:
                continue
            call_id = event.data.get("call_id")
            result_index = self._find_result_index(nodes, call_id)
            if result_index is None:
                # A call with no result is an incomplete pairing (it may be the
                # call currently in flight). Refuse rather than guess.
                return None
            if result_index > end_index:
                end_index = result_index

        return start_index, end_index

    def _find_result_index(self, nodes: Sequence[int], call_id: Any) -> int | None:
        for index, seq in enumerate(nodes):
            event = self.session.event_at(seq)
            if event.type == TOOL_RESULT and event.data.get("call_id") == call_id:
                return index
        return None

    # -- the transaction -------------------------------------------------

    async def compact(self, *, manual: bool = False) -> CompactionResult:
        """Run one compaction transaction.

        The lock is ``compaction/start`` in the log. It is written *before* the
        model call and before any yield, so a crash during summarization leaves a
        marker that a later reader can find and reason about.
        """
        if not manual and not self.needed():
            return CompactionResult(performed=False, reason="below threshold")

        span = self.select_span()
        if span is None:
            return CompactionResult(performed=False, reason="no compactable span")

        start_seq, end_seq, shadowed = span
        before = self.meter.route_total
        heuristics = self.meter.heuristic_of_range(shadowed)

        # -- step 5: the lock -------------------------------------------
        lock = self.session.append(
            COMPACTION_START,
            {
                "start_seq": start_seq,
                "end_seq": end_seq,
                "shadowed": list(shadowed),
                "heuristic_tokens": heuristics,
                "manual": manual,
            },
        )

        # -- step 6: prepare --------------------------------------------
        messages = self._messages_for(shadowed)
        if not messages:
            self.session.append(
                COMPACTION_END,
                {"start": lock.seq, "applied": False, "reason": "empty span"},
            )
            return CompactionResult(performed=False, reason="empty span")

        # -- step 7: the only long suspension ---------------------------
        try:
            summary = await self.summarize(messages)
        except Exception as exc:  # noqa: BLE001 - a failed summary is not fatal
            self.session.append(
                COMPACTION_END,
                {
                    "start": lock.seq,
                    "applied": False,
                    "reason": f"summarization failed: {describe(exc)}",
                },
            )
            return CompactionResult(
                performed=False, reason=f"summarization failed: {describe(exc)}"
            )

        # -- step 8: the range must still be intact ---------------------
        current = self.session.surface
        if not self._range_intact(current, shadowed):
            self.session.append(
                COMPACTION_END,
                {"start": lock.seq, "applied": False, "reason": "surface changed"},
            )
            raise SurfaceChangedError(
                "the surface changed while the summary was being produced; "
                "compaction abandoned rather than shadowing the wrong content"
            )

        if not summary.strip():
            self.session.append(
                COMPACTION_END,
                {"start": lock.seq, "applied": False, "reason": "empty summary"},
            )
            return CompactionResult(performed=False, reason="empty summary")

        # -- the summary must actually be smaller -----------------------
        from .meter import price_message

        summary_price = price_message(
            Message(role="user", content=(Block.text_block(summary),))
        ).total
        if summary_price >= heuristics:
            # Writing it would make the session bigger, which is the opposite of
            # the point. Refusing is the correct outcome, not a silent no-op.
            self.session.append(
                COMPACTION_END,
                {
                    "start": lock.seq,
                    "applied": False,
                    "reason": "summary is not smaller than the shadowed span",
                    "summary_tokens": summary_price,
                    "shadowed_tokens": heuristics,
                },
            )
            return CompactionResult(
                performed=False,
                reason="summary is not smaller than the shadowed span",
                tokens_before=before,
                tokens_after=before,
            )

        # -- steps 9-11: summary, replace, end -- no await between -------
        self.session.append(
            COMPACTION_SUMMARY,
            {
                "start": lock.seq,
                "text": summary,
                "tokens": summary_price,
                "shadowed": list(shadowed),
            },
        )

        replacement = self.session.append(
            USER_MESSAGE,
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": summary}],
                },
                "surface_op": {
                    "op": "replace",
                    "start_seq": start_seq,
                    "end_seq": end_seq,
                },
                "source_event_seqs": list(shadowed),
                "compaction": lock.seq,
            },
        )

        self.session.append(
            COMPACTION_END,
            {
                "start": lock.seq,
                "applied": True,
                "replacement_seq": replacement.seq,
                "shadowed": list(shadowed),
                "tokens_before": before,
            },
        )

        after = self.meter.route_total - heuristics + summary_price
        return CompactionResult(
            performed=True,
            start_seq=start_seq,
            end_seq=end_seq,
            shadowed=shadowed,
            summary=summary,
            tokens_before=before,
            tokens_after=after,
        )

    async def compact_until_under_budget(self, *, manual: bool = False) -> list[CompactionResult]:
        """Compact repeatedly until the session is under the threshold.

        One pass is frequently not enough: a single summary of a long history can
        still leave the session over budget. Each pass is a full transaction, so a
        failure partway leaves the earlier passes applied and the session
        consistent.
        """
        results: list[CompactionResult] = []
        for _ in range(self.max_attempts):
            result = await self.compact(manual=manual)
            results.append(result)
            if not result.performed:
                break
            if not self.needed():
                break
        return results

    # -- helpers ---------------------------------------------------------

    def _messages_for(self, shadowed: Sequence[int]) -> list[Message]:
        """The messages the summarizer is shown.

        Derived from the same events the replacement will shadow, so the summary
        describes exactly what it removes.
        """
        from ..session.session import _derive_event_message

        projected = self.session.surface.projected_messages
        out: list[Message] = []
        for seq in shadowed:
            message = _derive_event_message(self.session.event_at(seq), projected)
            if message is not None:
                out.append(message)
        return out

    @staticmethod
    def _range_intact(surface: Surface, shadowed: Sequence[int]) -> bool:
        """Whether every shadowed node is still present and contiguous."""
        nodes = surface.nodes
        try:
            positions = [nodes.index(seq) for seq in shadowed]
        except ValueError:
            return False
        return positions == list(range(positions[0], positions[0] + len(positions)))

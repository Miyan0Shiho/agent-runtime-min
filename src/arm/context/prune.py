"""Pruning: shrink one oversized tool result, without spending a model call.

Pruning is the middle of the three reduction strategies, and the ordering is the
point:

    spill (free, at tool time) -> prune (free, no model call) -> compact (costs a
    model call)

Everything free is attempted first, so a session that can be brought under budget
by trimming a single oversized result never pays for a summary.

Pruning is expressed as a ``replace`` on exactly one node. That constraint is not
an implementation convenience -- it is what keeps the change auditable: a pruner
that could rewrite several nodes at once would be a compaction wearing a
pruner's name, and its shadowed set would be hard to verify by inspection.

The algorithm keeps the head and tail of the text and drops the middle, because
the beginning of a tool result usually says what it is and the end usually says
how it went.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..session.envelope import Envelope
from ..session.messages import Block, Message
from ..session.session import Session
from ..session.surface import Surface
from ..session.events import TOOL_RESULT
from .meter import Meter

log = logging.getLogger("arm.context.prune")

#: Characters kept at the head of a pruned result.
DEFAULT_HEAD_CHARS = 4096
#: Characters kept at the tail.
DEFAULT_TAIL_CHARS = 1024

#: The marker inserted where content was removed. Deliberately explicit: the
#: model must be able to tell that something is missing, otherwise it will
#: reason about a partial result as though it were complete.
PRUNE_MARKER = "\n\n… [{removed} characters pruned; use the locator to retrieve if needed] …\n\n"


@dataclass(frozen=True, slots=True)
class PrunePlan:
    """One planned replacement, ready to be appended."""

    seq: int
    call_id: str
    shadowed: tuple[int, ...]
    #: The replacement content blocks, preserving structure. Stored as blocks
    #: rather than as a flat string so non-text blocks survive the rewrite.
    new_blocks: tuple[dict[str, Any], ...]
    removed_chars: int
    heuristic_before: int
    heuristic_after: int

    @property
    def new_content(self) -> str:
        """The replacement's text, for reporting."""
        return "".join(
            str(b.get("text", "")) for b in self.new_blocks if b.get("type") == "text"
        )


class Pruner:
    """Trims oversized tool results by replacing a single node.

    Holds no state about the session: every decision is made from the snapshot it
    is given, so a prune either happens completely or not at all.
    """

    def __init__(
        self,
        *,
        head_chars: int = DEFAULT_HEAD_CHARS,
        tail_chars: int = DEFAULT_TAIL_CHARS,
        threshold_chars: int | None = None,
    ) -> None:
        if head_chars < 0 or tail_chars < 0:
            raise ValueError("head_chars and tail_chars must be non-negative")
        self.head_chars = head_chars
        self.tail_chars = tail_chars

        # Only results longer than this are considered.
        #
        # The default is computed from the *worst case* marker width, because the
        # marker embeds the number of removed characters and a narrow estimate
        # would make the budget unsatisfiable for large results -- pruning would
        # then silently never happen. The bound is generous: a removed count needs
        # at most ~20 digits for any real string.
        worst_case_marker = len(PRUNE_MARKER.format(removed=10**20))
        self.threshold_chars = (
            threshold_chars
            if threshold_chars is not None
            else head_chars + tail_chars + worst_case_marker
        )

        # M2.12: the budget must be self-consistent at construction. A head and
        # tail that cannot fit inside the threshold means the pruner can never
        # satisfy its own guarantee, and discovering that per-call would look like
        # "pruning just doesn't work" rather than a configuration error.
        if head_chars + tail_chars + worst_case_marker > self.threshold_chars:
            raise ValueError(
                f"prune budget is unsatisfiable: head({head_chars}) + "
                f"tail({tail_chars}) + marker({worst_case_marker}) exceeds "
                f"threshold_chars({self.threshold_chars}); the pruned result could "
                "never fit inside the threshold"
            )

    # -- selection -------------------------------------------------------

    def candidates(
        self, session: Session, *, limit: int | None = None
    ) -> list[int]:
        """Seqs of tool results worth pruning, largest first.

        Candidates come from the *current surface*, so an already-pruned node is
        not selected again -- its content no longer exceeds the threshold.
        """
        surface = session.surface
        found: list[tuple[int, int]] = []
        for seq in surface.nodes:
            event = session.event_at(seq)
            if event.type != TOOL_RESULT:
                continue
            total = self._text_length(event)
            if total > self.threshold_chars:
                found.append((total, seq))
        found.sort(reverse=True)
        selected = [seq for _, seq in found]
        return selected[:limit] if limit is not None else selected

    def _text_length(self, event: Envelope) -> int:
        """Total text length of a tool result.

        Non-text blocks are skipped rather than measured: they cannot be sliced,
        so they are never a reason to prune.
        """
        message = event.data.get("message") or {}
        content = message.get("content") or ()
        total = 0
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                total += len(str(block.get("text", "")))
        return total

    # -- planning --------------------------------------------------------

    def plan(self, session: Session, seq: int) -> PrunePlan | None:
        """Compute a replacement for one node, or ``None`` if it is not worth it.

        The plan is computed from the *original event*, not from the current
        surface: candidates are taken from a snapshot, so several replacements in
        one pass cannot affect one another (and cannot prune a node twice).
        """
        event = session.event_at(seq)
        if event.type != TOOL_RESULT:
            return None

        message = event.data.get("message") or {}
        content = list(message.get("content") or ())
        total = sum(
            len(str(b.get("text", "")))
            for b in content
            if isinstance(b, Mapping) and b.get("type") == "text"
        )
        if total <= self.threshold_chars:
            return None

        # Strictly greater, not greater-or-equal: pruning a result that is
        # exactly at the threshold would produce a longer string.
        removed_start = self.head_chars
        removed_end = total - self.tail_chars
        if removed_end <= removed_start:
            return None

        new_blocks, removed = self._rebuild(content, removed_start, removed_end)
        if removed <= 0:
            return None

        # The structural guarantee: the replacement must be both smaller and
        # within budget, or it is not a reduction at all.
        after = sum(
            len(str(b.get("text", "")))
            for b in new_blocks
            if isinstance(b, Mapping) and b.get("type") == "text"
        )
        if not (after < total and after <= self.threshold_chars):
            return None

        return PrunePlan(
            seq=seq,
            call_id=str(event.data.get("call_id", "")),
            shadowed=(seq,),
            new_blocks=tuple(new_blocks),
            removed_chars=removed,
            heuristic_before=_heuristic_total(content),
            heuristic_after=_heuristic_total(new_blocks),
        )

    def _rebuild(
        self, content: Sequence[Any], removed_start: int, removed_end: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Splice the marker into the text blocks, counting characters globally.

        The head/tail budgets are measured across the whole result rather than per
        block, because the model reads the result as one thing. Cutting each block
        independently would keep the first 4 KiB of *every* block and blow the
        budget.

        Non-text blocks are preserved in place and count as zero length, so a
        result containing an image keeps its structure.
        """
        out: list[dict[str, Any]] = []
        cursor = 0
        removed = 0
        inserted = False

        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") != "text":
                # Preserved verbatim: it has no character length to spend.
                out.append(dict(block))
                continue

            text = str(block.get("text", ""))
            start = cursor
            end = cursor + len(text)
            cursor = end

            head_end = min(end, max(start, removed_start))
            tail_start = max(head_end, min(end, removed_end))

            kept_head = text[: head_end - start]
            kept_tail = text[tail_start - start :]
            removed += max(0, tail_start - head_end)

            if not inserted and removed_start <= end:
                merged = kept_head + PRUNE_MARKER.format(removed=0) + kept_tail
                inserted = True
            else:
                merged = kept_head + kept_tail

            if merged:
                out.append({"type": "text", "text": merged})

        if removed:
            # Rewrite the marker now that the true count is known. Doing it in one
            # pass avoids a second traversal and keeps the block order untouched.
            out = [
                {
                    **b,
                    "text": str(b.get("text", "")).replace(
                        PRUNE_MARKER.format(removed=0),
                        PRUNE_MARKER.format(removed=removed),
                    ),
                }
                if b.get("type") == "text"
                else b
                for b in out
            ]

        return out, removed

    # -- execution -------------------------------------------------------

    def apply(self, session: Session, plan: PrunePlan) -> Envelope | None:
        """Append the replacement event for ``plan``.

        Returns the new event, or ``None`` if the surface moved underneath us --
        in which case nothing is written, because replacing a range that is no
        longer where we thought would shadow the wrong content.
        """
        surface: Surface = session.surface
        if plan.seq not in surface.nodes:
            log.warning(
                "prune target seq %d is no longer on the surface; skipping", plan.seq
            )
            return None

        original = session.event_at(plan.seq)
        message = dict(original.data.get("message") or {})
        message["content"] = [dict(b) for b in plan.new_blocks]

        return session.append_tool_result(
            call_seq=int(original.data.get("call_seq", -1)),
            call_id=str(original.data.get("call_id", "")),
            message=message,
            is_error=bool(original.data.get("is_error", False)),
            info=(
                dict(original.data["info"])
                if "info" in original.data
                else None
            ),
            replace={"op": "replace", "start_seq": plan.seq, "end_seq": plan.seq},
            source_event_seqs=[plan.seq],
        )

    def prune_to_budget(
        self, session: Session, meter: Meter, target_tokens: int, *, limit: int = 20
    ) -> list[PrunePlan]:
        """Prune the largest results until the heuristic total is under target.

        Stops as soon as the target is met, so the cheapest sufficient set of
        replacements is applied. The meter is updated as plans are applied, so the
        loop sees the effect of each one.
        """
        applied: list[PrunePlan] = []
        for seq in self.candidates(session, limit=limit):
            if meter.heuristic_total <= target_tokens:
                break
            plan = self.plan(session, seq)
            if plan is None:
                continue
            if self.apply(session, plan) is None:
                continue
            meter.observe(seq, None)
            meter.observe(session.seq - 1, _message_from_content(plan.new_blocks))
            applied.append(plan)
        return applied


def _render(blocks: Sequence[Mapping[str, Any]]) -> str:
    """Flatten text blocks for reporting."""
    return "".join(
        str(b.get("text", "")) for b in blocks if b.get("type") == "text"
    )


def _heuristic_total(content: Sequence[Any]) -> int:
    from .meter import price_blocks

    blocks = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        if block.get("type") == "text":
            blocks.append(Block.text_block(str(block.get("text", ""))))
        else:
            blocks.append(Block.image({}))
    return price_blocks(blocks).total


def _message_from_content(blocks: Sequence[Mapping[str, Any]]) -> Message:
    """Build the model-visible message a pruned result will derive to.

    Non-text blocks are preserved so the meter prices the same thing the model
    will actually see.
    """
    out: list[Block] = []
    for block in blocks:
        if block.get("type") == "text":
            out.append(Block.text_block(str(block.get("text", ""))))
        else:
            out.append(Block.image({}))
    return Message(role="tool", content=tuple(out))

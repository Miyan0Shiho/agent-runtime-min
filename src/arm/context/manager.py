"""Context management as a loop concern: measure, then reduce.

This module is the seam between the agent loop and the three reduction
strategies. It exists so the loop does not have to know *how* pressure is
relieved -- only that it should ask before each request.

The order of attempts is the design:

    spill  (free, at tool time)  ->  prune (free, no model call)  ->  compact
    (costs a model call)

Everything free is tried first. A session that can be brought under budget by
trimming one oversized result never pays for a summary, and the summary is only
reached when trimming genuinely cannot help.

The manager also owns the meter's upkeep. Recomputing the whole surface after
every event would be O(surface) per append; instead the meter is advanced
incrementally from the log, and a replacement resets only what it shadowed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..session.events import TOOL_RESULT
from ..session.session import Session
from .compact import Compactor, CompactionResult, Summarizer
from .meter import Meter, Thresholds
from .prune import Pruner
from .spill import SpillError, SpillStore

log = logging.getLogger("arm.context")


@dataclass
class ContextReport:
    """What the manager did about pressure, for the log and for tests."""

    measured: int = 0
    threshold: int | None = None
    pruned: int = 0
    compacted: int = 0
    spilled: int = 0
    reason: str = ""

    @property
    def acted(self) -> bool:
        return bool(self.pruned or self.compacted or self.spilled)


class ContextManager:
    """Measures pressure and relieves it, cheapest strategy first.

    Deliberately not a plugin: context management is part of the loop's contract
    (the model must not be sent an over-budget request), not an optional add-on.
    """

    def __init__(
        self,
        session: Session,
        *,
        thresholds: Thresholds | None = None,
        summarize: Summarizer | None = None,
        pruner: Pruner | None = None,
        spill: SpillStore | None = None,
        spill_threshold: int = 24_000,
        auto: bool = True,
    ) -> None:
        self.session = session
        self.thresholds = thresholds or Thresholds(context_window=None)
        self.meter = Meter()
        self.pruner = pruner or Pruner()
        self.spill = spill
        self.spill_threshold = spill_threshold
        self.auto = auto
        self._summarize = summarize
        self._last_seq_seen = 0
        self._compactor: Compactor | None = None

    # -- measurement -----------------------------------------------------

    def refresh(self) -> None:
        """Bring the meter up to date with the log.

        Only new events are priced, so this is O(new) rather than O(surface). A
        replacement invalidates the nodes it shadowed, which are then forgotten.
        """
        events = self.session.events()
        if len(events) < self._last_seq_seen:
            # The log shrank, which should be impossible; rebuild rather than
            # reason about a state that cannot happen.
            self.meter = Meter()
            self._last_seq_seen = 0

        # Forget nodes that are no longer on the surface (they were replaced).
        on_surface = set(self.session.surface.nodes)
        stale = [
            seq
            for seq in self.meter.snapshot()
            if seq not in on_surface and seq < len(events)
        ]
        if stale:
            self.meter.forget(stale)

        from ..session.session import _derive_event_message

        projected = self.session.surface.projected_messages
        for seq in range(self._last_seq_seen, len(events)):
            if seq not in on_surface:
                # A log-only event (turn/start, tool/call, ...) occupies no
                # surface position and therefore costs nothing on the wire.
                continue
            message = _derive_event_message(events[seq], projected)
            self.meter.observe(seq, message)

        self._last_seq_seen = len(events)
        self.meter.set_order(self.session.surface.nodes)

    @property
    def measured(self) -> int:
        """The route price: what the current model would charge."""
        self.refresh()
        return self.meter.route_total

    @property
    def over_threshold(self) -> bool:
        try:
            return self.meter.route_total >= self.thresholds.threshold_tokens
        except ValueError:
            # Capacity undeclared (E1). The documented degradation is to warn
            # once and continue: incomplete configuration must not block work,
            # and the provider's own overflow error remains recoverable.
            return False

    # -- relief ----------------------------------------------------------

    def spill_large_results(self) -> int:
        """Offload oversized tool results to disk, in place.

        Spilling *appends* rather than replacing, so it does not invalidate the
        derived-message cache. That makes it the cheapest possible reduction.

        This runs **at tool time**, independent of the pressure threshold -- not
        as part of the measure-then-reduce sequence. The distinction matters: a
        result that is oversized the moment it lands is oversized regardless of
        whether the session happens to be under budget yet, and offloading it
        early means the pressure never builds in the first place.
        """
        if self.spill is None:
            return 0

        spilled = 0
        for seq in list(self.session.surface.nodes):
            event = self.session.event_at(seq)
            if event.type != TOOL_RESULT:
                continue
            message = event.data.get("message") or {}
            content = message.get("content") or ()
            text = "".join(
                str(b.get("text", ""))
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if len(text) <= self.spill_threshold:
                continue

            try:
                locator = self.spill.put(text, session_id=self.session.session_id)
            except SpillError as exc:
                # Contained: the original content stays exactly as it was, and
                # the caller keeps working with less headroom rather than losing
                # the result entirely.
                log.warning("spill failed for seq %d: %s", seq, exc)
                continue

            head = text[: self.pruner.head_chars]
            tail = text[-self.pruner.tail_chars :]
            replacement = (
                f"{head}\n\n… [{len(text) - len(head) - len(tail)} characters "
                f"moved to disk; locator {locator} ] …\n\n{tail}"
            )
            # The locator goes in the *content*, not in ``info``.
            #
            # Two reasons, and the second is the important one. M1.9 permits a
            # tool/result replacement to change only its content, so putting the
            # locator in a sibling field would be rejected as a rewrite of
            # something other than content. And the model is the party that needs
            # to retrieve the bytes, so the handle belongs where the model can
            # read it -- a locator recorded only for the runtime's benefit would
            # be unreachable by the one caller that can act on it.
            self.session.append_tool_result(
                call_seq=int(event.data.get("call_seq", -1)),
                call_id=str(event.data.get("call_id", "")),
                message={
                    **dict(message),
                    "content": [{"type": "text", "text": replacement}],
                },
                is_error=bool(event.data.get("is_error", False)),
                replace={"op": "replace", "start_seq": seq, "end_seq": seq},
                source_event_seqs=[seq],
            )
            spilled += 1

        return spilled

    async def relieve(self) -> ContextReport:
        """Bring the session back under budget, cheapest strategy first.

        Returns a report of what was done. A failure at any stage leaves the
        session consistent: each strategy either completes or writes nothing.
        """
        self.refresh()
        report = ContextReport(
            measured=self.meter.route_total,
            threshold=(
                self.thresholds.threshold_tokens
                if self.thresholds.context_window is not None
                else None
            ),
        )

        if not self.auto:
            report.reason = "automatic management disabled"
            return report

        # 1. Spill: free, and it does not disturb the surface order. It also runs
        #    at tool time, so by now most oversized results are already offloaded;
        #    repeating it here catches anything that arrived by another route.
        report.spilled = self.spill_large_results()
        if report.spilled:
            self.refresh()

        if not self.over_threshold:
            report.reason = (
                "spill was sufficient" if report.spilled else "below threshold"
            )
            return report

        # 2. Prune: free, replaces single nodes.
        target = self.thresholds.threshold_tokens
        applied = self.pruner.prune_to_budget(self.session, self.meter, target)
        report.pruned = len(applied)
        if applied:
            self.refresh()

        if not self.over_threshold:
            report.reason = "prune was sufficient"
            return report

        # 3. Compact: costs a model call, so it is last.
        if self._summarize is None:
            report.reason = "over threshold but no summarizer is configured"
            return report

        compactor = Compactor(
            session=self.session,
            meter=self.meter,
            thresholds=self.thresholds,
            summarize=self._summarize,
        )
        results: list[CompactionResult] = await compactor.compact_until_under_budget()
        report.compacted = sum(1 for r in results if r.performed)
        self.refresh()

        if self.over_threshold:
            report.reason = "still above threshold after all strategies"
        else:
            report.reason = "compaction was sufficient"
        return report

    def spill_now(self) -> int:
        """Offload any oversized result immediately, ignoring the threshold.

        Called right after tool results land, so pressure never accumulates.
        """
        if self.spill is None:
            return 0
        count = self.spill_large_results()
        if count:
            self.refresh()
        return count

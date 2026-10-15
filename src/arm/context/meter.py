"""Token metering: two prices, kept deliberately separate.

This is the part of context management most likely to be implemented wrongly,
because one price *looks* sufficient. Two are required, and they answer
different questions:

* **The heuristic price** answers "how much did that replacement save?" It must
  be **additive** -- the sum over a range must equal the price of the range --
  because a replacement needs to compute the cost of what it shadowed without
  re-reading the whole surface. Additivity is what keeps pressure accounting
  O(1) instead of O(surface).
* **The route price** answers "are we actually over budget?" It depends on which
  model is serving the request, because the same text costs different token
  counts under different tokenizers.

Conflating them breaks one of the two questions. A real tokenizer is not
additive across arbitrary boundaries, so using it for the first question would
make the shadow accounting drift; using the heuristic for the second would make
the threshold meaningless.

The heuristic is deliberately crude. Its job is to be *proportional and
additive*, not accurate: a 4-characters-per-token estimate with a small
per-block and per-role tax tracks real token counts closely enough to make
threshold decisions, and it never surprises you by changing between runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..session.messages import Block, Message

#: Characters per token for the heuristic. Deliberately conservative: over-
#: estimating triggers compaction slightly early, which is the safe direction.
CHARS_PER_TOKEN = 4.0

#: Flat tax per content block. Real tokenizers emit separators and role markers
#: that a pure character count misses, and the error compounds over many small
#: blocks -- a message of 50 one-word blocks is not the same cost as one 50-word
#: block.
BLOCK_TAX = 4

#: Flat tax per message, for the role and framing tokens.
MESSAGE_TAX = 3


@dataclass(frozen=True, slots=True)
class Price:
    """A price, with its parts visible.

    Keeping the components separate makes a surprising number debuggable: "why
    did this message cost 4000?" is answerable only if the content cost and the
    overhead are distinguishable.
    """

    content: int = 0
    overhead: int = 0

    @property
    def total(self) -> int:
        return self.content + self.overhead

    def __add__(self, other: "Price") -> "Price":
        return Price(self.content + other.content, self.overhead + other.overhead)

    def __radd__(self, other: int) -> "Price":
        # Support ``sum(prices)``, which starts from the integer 0.
        if other == 0:
            return self
        raise TypeError("Price can only be added to a Price or 0")


def heuristic_price(text: str) -> int:
    """Additive token estimate for a string.

    Rounded up, so any non-empty string costs at least one token -- an empty
    price for real content would let a caller believe a range was free.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def price_blocks(blocks: Sequence[Block]) -> Price:
    """Price a sequence of content blocks.

    Non-text blocks are counted as a flat cost rather than measured: an image has
    no character length, and pretending it is free would under-count exactly the
    messages most likely to blow a budget.
    """
    content = 0
    overhead = 0
    for block in blocks:
        if block.kind == "text":
            content += heuristic_price(block.text)
        else:
            # A non-text block is not free; it just cannot be measured by length.
            content += 256
        overhead += BLOCK_TAX
    return Price(content=content, overhead=overhead)


def price_message(message: Message) -> Price:
    """Price one message."""
    price = price_blocks(message.content)
    overhead = price.overhead + MESSAGE_TAX
    # Tool calls are part of what the model reads, so they are part of the cost.
    for call in message.tool_calls:
        overhead += heuristic_price(str(call.get("name", "")))
        overhead += heuristic_price(str(call.get("arguments", "")))
    return Price(content=price.content, overhead=overhead)


def price_messages(messages: Iterable[Message]) -> int:
    """Total heuristic price of a message list. Additive by construction."""
    return sum(price_message(m).total for m in messages)


class Meter:
    """Tracks prices for a session's surface, incrementally.

    The meter answers two questions:

    * ``heuristic_total`` -- additive, used for shadow pricing a replacement.
    * ``route_total`` -- what the current model would charge, used for the
      threshold decision.

    Both are maintained per node so that replacing a range is arithmetic rather
    than a re-scan.
    """

    def __init__(self) -> None:
        self._heuristic: dict[int, int] = {}
        self._route: dict[int, int] = {}
        #: Surface order, so a range can be summed without touching the log.
        self._order: list[int] = []

    # -- maintenance -----------------------------------------------------

    def observe(
        self, seq: int, message: Message | None = None, *, route_price: int | None = None
    ) -> None:
        """Record the price of one surface node.

        ``message is None`` means the node holds a position but contributes no
        wire content -- an empty placeholder. Its price is zero, and it must still
        be tracked so a later range sum can include it.
        """
        if message is None:
            self._heuristic[seq] = 0
        else:
            self._heuristic[seq] = price_message(message).total
        self._route[seq] = (
            route_price if route_price is not None else self._heuristic[seq]
        )
        if seq not in self._order:
            self._order.append(seq)

    def observe_session(self, session: Any) -> None:
        """Price every node on a session's surface.

        The meter derives each node's message itself rather than accepting a
        price from the caller. A caller that passed ``None`` for every node -- the
        obvious thing to write, and the mistake this method exists to prevent --
        would produce a meter that reports zero pressure forever, and compaction
        would never trigger.

        A node whose derivation yields nothing costs zero, which is correct: an
        empty placeholder is genuinely not sent.
        """
        from ..session.session import _derive_event_message

        projected = session.surface.projected_messages
        for seq in session.surface.nodes:
            message = _derive_event_message(session.event_at(seq), projected)
            self.observe(seq, message)
        self.set_order(session.surface.nodes)

    def forget(self, seqs: Iterable[int]) -> None:
        """Drop nodes that are no longer on the surface."""
        for seq in seqs:
            self._heuristic.pop(seq, None)
            self._route.pop(seq, None)
        gone = set(seqs)
        self._order = [s for s in self._order if s not in gone]

    def set_order(self, seqs: Sequence[int]) -> None:
        self._order = list(seqs)

    # -- queries ---------------------------------------------------------

    @property
    def heuristic_total(self) -> int:
        return sum(self._heuristic.values())

    @property
    def route_total(self) -> int:
        return sum(self._route.values())

    def heuristic_of(self, seq: int) -> int:
        return self._heuristic.get(seq, 0)

    def route_of(self, seq: int) -> int:
        return self._route.get(seq, 0)

    def heuristic_of_range(self, seqs: Iterable[int]) -> int:
        """Additive sum over a range. This is what a replacement shadows."""
        return sum(self._heuristic.get(seq, 0) for seq in seqs)

    def route_of_range(self, seqs: Iterable[int]) -> int:
        return sum(self._route.get(seq, 0) for seq in seqs)

    def snapshot(self) -> dict[int, int]:
        """A copy of the heuristic prices, keyed by seq."""
        return dict(self._heuristic)


@dataclass(frozen=True, slots=True)
class Thresholds:
    """The compaction thresholds, validated at construction.

    All five documented error conditions are checked here rather than at the
    point of use, so a misconfiguration surfaces when it is configured rather
    than in the middle of a long session.
    """

    context_window: int | None
    reserved_completion: int = 0
    headroom: int = 0
    threshold_ratio: float = 0.8
    retain_ratio: float | None = 0.16
    retain_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.context_window is not None:
            if not isinstance(self.context_window, int) or self.context_window <= 0:
                raise ValueError("E1: context_window must be a positive integer")
        if self.reserved_completion < 0:
            raise ValueError("E2: reserved_completion must be non-negative")
        if self.headroom < 0:
            raise ValueError("headroom must be non-negative")
        if not (0 < self.threshold_ratio <= 1):
            raise ValueError("threshold_ratio must be in (0, 1]")
        if self.retain_ratio is not None and not (0 < self.retain_ratio <= 1):
            raise ValueError("retain_ratio must be in (0, 1]")
        if self.retain_ratio is not None and self.retain_tokens is not None:
            raise ValueError(
                "retain_ratio and retain_tokens are mutually exclusive"
            )
        if self.context_window is not None:
            if self.context_window - self.reserved_completion <= 0:
                raise ValueError(
                    "E3: reserved_completion consumes the whole context window, "
                    "leaving no message budget"
                )
            if (
                self.context_window - self.reserved_completion - self.headroom
                <= 0
            ):
                raise ValueError(
                    "E4: there is a message budget but no pressure budget"
                )
        if self.retain_ratio is not None and self.context_window is not None:
            if self.retain_ratio >= self.effective_threshold_ratio:
                raise ValueError(
                    "E5: retain_ratio must be below the effective threshold_ratio, "
                    "otherwise compaction retains more than it removes"
                )

    @property
    def message_budget(self) -> int:
        if self.context_window is None:
            raise ValueError("context_window is not declared (E1)")
        return self.context_window - self.reserved_completion

    @property
    def pressure_budget(self) -> int:
        return self.message_budget - self.headroom

    @property
    def effective_threshold_ratio(self) -> float:
        """The ratio actually in force.

        ``headroom`` is a *subtractive* term, so whenever it is large relative to
        the window it silently becomes the dominant one and ``threshold_ratio``
        stops mattering. Reporting the effective value makes that visible instead
        of leaving a configuration knob that does nothing.
        """
        if self.context_window is None:
            return self.threshold_ratio
        by_ratio = self.context_window * self.threshold_ratio
        if by_ratio <= 0:
            return self.threshold_ratio
        return min(self.threshold_ratio, self.pressure_budget / self.context_window)

    @property
    def threshold_tokens(self) -> int:
        """The pressure threshold: ``floor(min(W*ratio, W-O-headroom))``."""
        if self.context_window is None:
            raise ValueError("context_window is not declared (E1)")
        return math.floor(
            min(
                self.context_window * self.threshold_ratio,
                self.message_budget - self.headroom,
            )
        )

    @property
    def retain_tokens_value(self) -> int:
        """How much context to keep after a compaction."""
        if self.retain_tokens is not None:
            return self.retain_tokens
        if self.retain_ratio is None:
            raise ValueError("neither retain_ratio nor retain_tokens was given")
        return math.floor(self.message_budget * self.retain_ratio)

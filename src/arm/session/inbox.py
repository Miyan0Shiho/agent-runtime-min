"""The inbox: pending input, projected from the log rather than queued in memory.

An in-memory queue would make the session's state depend on which process is
looking at it: a restart, a second reader, or a UI would each see a different
set of pending inputs. Deriving the inbox from the log means there is exactly
one answer, and it is the one written down.

Claiming is deletion. An input that has been claimed does not come back, and an
input that was rejected is not restored -- re-delivering a rejected input would
process it twice, which is worse than losing it because the duplicate is silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .envelope import Envelope
from .events import STEP_START, TURN_START, USER_MESSAGE


class SessionReader(Protocol):
    """The only thing the inbox needs from a session: its events.

    Declared as a protocol so the inbox does not depend on the concrete
    ``Session`` class -- it reads history and nothing else.
    """

    def events(self) -> tuple[Envelope, ...]: ...

#: How an input participates in the next turn.
#:
#: ``next-step`` joins the turn already in progress; ``next-turn`` waits for the
#: current turn to finish. The distinction is what lets a user interject mid-turn
#: without tearing down the step being executed.
InputKind = Literal["next-step", "next-turn"]


@dataclass(frozen=True, slots=True)
class PendingInput:
    """One unclaimed user input, with its provenance in the log."""

    seq: int
    text: str
    kind: InputKind
    message: dict
    #: The turn during which this input arrived, if any.
    arrived_in_turn: int | None = None


@dataclass
class Claim:
    """The result of claiming: what joins this step, and what waits for the next.

    ``turn`` is the input that starts the turn. Exactly one input may start a
    turn; everything else either joins it or waits.
    """

    turn: PendingInput | None
    steps: tuple[PendingInput, ...] = ()
    deferred: tuple[PendingInput, ...] = ()

    @property
    def empty(self) -> bool:
        return self.turn is None and not self.steps

    def all_claimed(self) -> tuple[PendingInput, ...]:
        items = list(self.steps)
        if self.turn is not None:
            items.append(self.turn)
        return tuple(items)


class Inbox:
    """Pending inputs, derived from the session log.

    The inbox is a *projection*, not a queue: every call re-derives from the log,
    so a restart cannot lose or duplicate an input.
    """

    def __init__(self, session: SessionReader) -> None:
        self._session = session

    def pending(self) -> tuple[PendingInput, ...]:
        """Inputs that arrived but have not been claimed by a turn or a step.

        A claim is durable: it is recorded in ``turn/start.claimed_seqs`` or in
        ``step/start.inputs``. Deriving claims from the log rather than from
        memory is what makes the inbox survive a restart, and it is also what
        stops mid-turn input from being claimed over and over.
        """
        events = self._session.events()
        claimed: set[int] = set()
        for event in events:
            if event.type == TURN_START:
                for seq in event.data.get("claimed_seqs", ()):
                    claimed.add(int(seq))
            elif event.type == STEP_START:
                # Inputs that joined a step mid-turn are claimed by that step.
                for seq in event.data.get("inputs", ()):
                    claimed.add(int(seq))

        pending: list[PendingInput] = []
        for event in events:
            if event.type != USER_MESSAGE:
                continue
            if event.seq in claimed:
                continue
            # Runtime-generated context is not user input.
            #
            # The runtime writes ``user/message`` events for things the model
            # should *read* -- the skill catalogue, a plan-mode notice -- because
            # a user-role message is how context is delivered. Those are not work
            # waiting to be done, and treating them as input is not a cosmetic
            # bug: each one would be claimed by the next turn, and a context
            # message that is republished whenever the session is over budget
            # would therefore generate new input on every step, forever.
            if event.data.get("context_only"):
                continue
            message = event.data.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content") or []
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            kind: InputKind = event.data.get("kind") or "next-turn"  # type: ignore[assignment]
            pending.append(
                PendingInput(
                    seq=event.seq,
                    text=text,
                    kind=kind,
                    message=dict(message),
                )
            )
        return tuple(pending)

    def claim(self) -> Claim:
        """Take everything claimable for the next turn.

        The rule: all ``next-step`` inputs plus exactly one ``next-turn`` input
        (the oldest). Taking more than one turn-starter would begin two turns at
        once; taking none when a turn-starter exists would stall forever.
        """
        pending = self.pending()
        steps = tuple(item for item in pending if item.kind == "next-step")
        turn_starters = [item for item in pending if item.kind != "next-step"]
        turn = turn_starters[0] if turn_starters else None
        deferred = tuple(turn_starters[1:])
        return Claim(turn=turn, steps=steps, deferred=deferred)

    def is_empty(self) -> bool:
        return not self.pending()

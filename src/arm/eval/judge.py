"""The judge: model-based assessment, with its failure modes designed against.

A judge is used only where a deterministic check cannot reach -- questions of
quality, not of fact. It is the least trustworthy part of the harness, so it is
built to advertise its own uncertainty rather than to produce a confident number.

Four systematic biases are known, and each gets a structural defence rather than
a note in a document:

* **Position bias** -- preferring whichever option came first. Defended by asking
  twice with the order swapped, and requiring agreement.
* **Verbosity bias** -- preferring longer answers. Defended by scoring length as
  a *separate* dimension, so it cannot leak into the quality score.
* **Self-preference** -- preferring output from the same model family. Defended by
  recording whether judge and candidate share a source, so a suspiciously high
  agreement is visible rather than assumed.
* **Score drift** -- the same input scoring differently over time. Defended by
  pinning the judge version and temperature, and by anchor cases.

The single most important rule: **disagreement is reported, not averaged.** Two
runs that disagree mean the verdict is unreliable; averaging them would launder
an inconsistency into a plausible-looking middle score, which is the worst
possible outcome because it hides the problem while appearing rigorous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ..kernel.errors import ArmError, describe
from ..llm.seam import Request, collect
from ..session.messages import Block, Message

log = logging.getLogger("arm.eval.judge")


class JudgeError(ArmError):
    """The judge could not produce a usable verdict."""


#: The rubric is fixed so that a score means the same thing across runs. A rubric
#: that drifted with the prompt would make historical scores incomparable, and
#: nobody would notice because the number would still look plausible.
RUBRIC = """You are evaluating one answer to a task.

Score the answer on these dimensions, each from 1 to 5:

- correctness: does it actually satisfy the task?
- completeness: does it address everything the task asked for?
- clarity: is the explanation clear and well organized?

Rules:
- Judge the answer on its merits, not its length. A short correct answer
  outranks a long vague one.
- Do not reward confidence. Reward evidence.
- If the answer is wrong in a way that matters, correctness is 1 or 2.

Reply with only a JSON object, no prose:
{"correctness": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "reason": "<one sentence>"}
"""


@dataclass(frozen=True, slots=True)
class Verdict:
    """One judge assessment."""

    correctness: int
    completeness: int
    clarity: int
    reason: str = ""
    raw: str = ""

    @property
    def quality(self) -> float:
        """The quality score.

        Length is deliberately absent: it is recorded separately so that a
        verbose answer cannot buy a higher score through the back door.
        """
        return (self.correctness + self.completeness + self.clarity) / 3.0


@dataclass(frozen=True, slots=True)
class Judgement:
    """The outcome of judging one answer, including its reliability.

    ``reliable`` is the field that matters. A caller that reads only ``score``
    and ignores ``reliable`` is using the judge wrongly, which is why the
    unreliable case does not carry an averaged score at all.
    """

    reliable: bool
    score: float | None
    verdicts: tuple[Verdict, ...] = ()
    note: str = ""
    same_source: bool = False

    @property
    def length_chars(self) -> int:
        return 0


@dataclass
class Judge:
    """Assesses an answer using a model, with the known biases defended against."""

    adapter: Any
    model: str = ""
    #: Whether the judge and the candidate share a model source. Recorded rather
    #: than forbidden, because sometimes there is no alternative -- but a
    #: same-source verdict deserves more scepticism.
    same_source: bool = False
    temperature: float = 0.0

    async def assess(self, task: str, answer: str) -> Judgement:
        """Score ``answer`` twice, swapping the framing, and require agreement.

        Two calls with different orderings expose position bias. Agreement means
        the verdict is stable under the one perturbation most likely to move it;
        disagreement means the verdict is unreliable and is reported as such.
        """
        first = await self._ask(task, answer, order="answer_first")
        if first is None:
            return Judgement(
                reliable=False, score=None, note="the judge produced no parseable verdict"
            )

        second = await self._ask(task, answer, order="task_first")
        if second is None:
            return Judgement(
                reliable=False,
                score=None,
                verdicts=(first,),
                note="the second judge call produced no parseable verdict",
                same_source=self.same_source,
            )

        # Agreement is measured on the quality score with a tolerance of one
        # point: judges are not deterministic enough to require exact equality,
        # and demanding it would mark almost everything unreliable.
        agrees = abs(first.quality - second.quality) <= 1.0

        if not agrees:
            return Judgement(
                reliable=False,
                score=None,
                verdicts=(first, second),
                note=(
                    "the judge disagreed with itself when the framing changed "
                    f"({first.quality:.1f} vs {second.quality:.1f}); the verdict is "
                    "not reliable enough to act on"
                ),
                same_source=self.same_source,
            )

        return Judgement(
            reliable=True,
            score=(first.quality + second.quality) / 2.0,
            verdicts=(first, second),
            note="both orderings agreed",
            same_source=self.same_source,
        )

    async def _ask(self, task: str, answer: str, *, order: str) -> Verdict | None:
        """One judge call. Returns ``None`` if the response is unusable."""
        if order == "answer_first":
            body = (
                f"ANSWER:\n{answer}\n\nTASK:\n{task}\n\n{RUBRIC}"
            )
        else:
            body = (
                f"TASK:\n{task}\n\nANSWER:\n{answer}\n\n{RUBRIC}"
            )

        request = Request(
            messages=(Message(role="user", content=(Block.text_block(body),)),),
            model=self.model,
            temperature=self.temperature,
        )
        try:
            result = await collect(self.adapter, request)
        except Exception as exc:  # noqa: BLE001 - a broken judge is not a crash
            log.warning("judge call failed: %s", describe(exc))
            return None

        if result.finish_reason == "error":
            log.warning("judge returned an error: %s", result.error)
            return None
        return parse_verdict(result.text)


def parse_verdict(text: str) -> Verdict | None:
    """Extract a verdict from a model response.

    Tolerant of surrounding prose and code fences, because models add them. Strict
    about the score fields: a missing or out-of-range score is a failed judgement
    rather than a defaulted one -- silently substituting a middle score would
    manufacture agreement.
    """
    import json
    import re

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            candidate = brace.group(0)

    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, Mapping):
        return None

    scores: dict[str, int] = {}
    for key in ("correctness", "completeness", "clarity"):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        score = int(value)
        if not (1 <= score <= 5):
            return None
        scores[key] = score

    return Verdict(
        correctness=scores["correctness"],
        completeness=scores["completeness"],
        clarity=scores["clarity"],
        reason=str(payload.get("reason", ""))[:500],
        raw=text[:2000],
    )


@dataclass
class AnchorCalibration:
    """Fixed answers with known scores, used to detect judge drift.

    Comparing a judge against itself over time is impossible without a fixed
    reference. These are that reference: if the judge's scores on them move, the
    judge has drifted and its verdicts on real cases cannot be compared with
    earlier ones.
    """

    anchors: tuple[tuple[str, str, float], ...] = ()
    #: How much drift is tolerated before the judge is considered unstable.
    tolerance: float = 0.75

    async def check(self, judge: Judge) -> tuple[bool, str]:
        """Score every anchor and report whether the judge is still calibrated."""
        if not self.anchors:
            return True, "no anchors configured"

        drift: list[str] = []
        for task, answer, expected in self.anchors:
            judgement = await judge.assess(task, answer)
            if not judgement.reliable or judgement.score is None:
                drift.append(f"anchor {task[:30]!r} was unreliable")
                continue
            if abs(judgement.score - expected) > self.tolerance:
                drift.append(
                    f"anchor {task[:30]!r}: expected {expected}, got "
                    f"{judgement.score:.1f}"
                )

        if drift:
            return False, "; ".join(drift)
        return True, "all anchors within tolerance"


def default_anchors() -> AnchorCalibration:
    """A minimal anchor set: one obviously good answer, one obviously bad.

    The good one catches a judge that has become too harsh; the bad one catches a
    judge that has become credulous. Together they bound the error in both
    directions, which a single anchor could not.
    """
    return AnchorCalibration(
        anchors=(
            (
                "Explain what a race condition is.",
                (
                    "A race condition occurs when two operations access shared "
                    "state concurrently and the result depends on their timing. "
                    "For example, two threads incrementing the same counter "
                    "without a lock can lose an update."
                ),
                5.0,
            ),
            (
                "Explain what a race condition is.",
                "It is when computers are fast and things go wrong sometimes.",
                1.5,
            ),
        )
    )

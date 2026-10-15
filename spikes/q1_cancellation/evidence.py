"""Print the raw observations quoted in FINDINGS.md.

Run from ``spikes/``::

    cd spikes && ../.venv/bin/python q1_cancellation/evidence.py

This is not a test; it is the transcript generator so that FINDINGS.md quotes
observed output rather than a paraphrase.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    Cause,
    DrainScope,
    ToolRequest,
    dispatch_and_drain,
    naive_create_task,
    naive_inline,
    naive_shield_only,
)


def hdr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def e1_naive() -> None:
    hdr("E1  naive cancellation: what survives?")
    for name, strategy in (
        ("naive_inline", naive_inline),
        ("naive_create_task", naive_create_task),
        ("naive_shield_only", naive_shield_only),
    ):
        log: list[str] = []

        async def body():
            log.append("body_start")
            await asyncio.sleep(0.05)
            log.append("side_effect")
            return "RESULT"

        caller = asyncio.create_task(strategy(body))
        await asyncio.sleep(0.01)
        caller.cancel()
        outcome = ""
        try:
            r = await caller
            outcome = f"returned {r!r}"
        except asyncio.CancelledError:
            outcome = "CancelledError"
        await asyncio.sleep(0.1)
        print(f"  {name:20s} outer={outcome:20s} ledger={log}")


async def e2_shield_strands() -> None:
    hdr("E2  shield alone: body survives, result stranded")
    log: list[str] = []
    inner: list[asyncio.Task] = []

    async def body():
        log.append("body_start")
        await asyncio.sleep(0.05)
        log.append("side_effect")
        return "RESULT"

    async def caller():
        t = asyncio.create_task(body())
        inner.append(t)
        return await asyncio.shield(t)

    c = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    c.cancel()
    try:
        await c
    except asyncio.CancelledError:
        pass
    print(f"  immediately: inner.done()={inner[0].done()} inner.cancelled()={inner[0].cancelled()}")
    await asyncio.sleep(0.1)
    print(f"  ledger={log}  -> body finished, but nothing collected the value")


async def e3_loop_teardown() -> None:
    hdr("E3  loop teardown: shield-only loses the side effect, composite keeps it")

    def scenario(kind: str) -> list[str]:
        import threading

        events: list[str] = []

        async def body():
            events.append("body_start")
            await asyncio.sleep(0.05)
            events.append("side_effect")
            return "RESULT"

        async def shield_only():
            t = asyncio.create_task(body())
            return await asyncio.shield(t)

        async def composite():
            t = asyncio.create_task(body())
            try:
                return await asyncio.shield(t)
            finally:
                await asyncio.shield(t)

        async def main():
            c = asyncio.create_task(shield_only() if kind == "shield" else composite())
            await asyncio.sleep(0.01)
            c.cancel()
            try:
                await c
            except asyncio.CancelledError:
                pass
            events.append("turn_ended")

        th = threading.Thread(target=lambda: asyncio.run(main()))
        th.start()
        th.join(timeout=10)
        return events

    print(f"  shield_only : {scenario('shield')}")
    print(f"  composite   : {scenario('composite')}")


async def e4_repeat_cancel() -> None:
    hdr("E4  repeat cancel during the drain")
    log: list[str] = []

    async def body():
        log.append("body_start")
        await asyncio.sleep(0.15)
        log.append("side_effect")
        return "RESULT"

    # unshielded drain
    async def unshielded():
        t = asyncio.create_task(body())
        try:
            return await asyncio.shield(t)
        finally:
            await t

    c = asyncio.create_task(unshielded())
    await asyncio.sleep(0.01)
    c.cancel()
    await asyncio.sleep(0.02)
    c.cancel()
    try:
        await c
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.2)
    print(f"  unshielded drain : ledger={log}")
    log.clear()

    scope = DrainScope()
    req = ToolRequest("c1", "t", body)
    scope.declare(req)

    async def composite():
        call = scope.dispatch(req)
        try:
            return await asyncio.shield(call.task)
        finally:
            await scope.drain(already_pending=True)

    c2 = asyncio.create_task(composite())
    await asyncio.sleep(0.01)
    c2.cancel()
    await asyncio.sleep(0.02)
    c2.cancel()
    await asyncio.sleep(0.02)
    c2.cancel()
    try:
        await c2
    except asyncio.CancelledError:
        pass
    print(f"  composite drain  : ledger={log} absorbed={scope.drain_absorbed_cancels}")
    print(f"  recovered result : {scope.results()[0]}")


async def e5_baseexception() -> None:
    hdr("E5  the BaseException trap (cancellation delivered into the body)")
    log: list[str] = []

    async def bad_body():
        log.append("body_start")
        try:
            await asyncio.sleep(0.05)
        except BaseException as e:
            log.append(f"swallowed_{type(e).__name__}")
            return "FAKE_SUCCESS"
        return "REAL_RESULT"

    async def caller():
        return await bad_body()

    c = asyncio.create_task(caller())
    await asyncio.sleep(0.01)
    c.cancel()
    r = await c
    print(f"  outer RETURNED {r!r}  cancelled()={c.cancelled()}  ledger={log}")


async def e6_multi() -> None:
    hdr("E6  multiple in-flight calls + synthetic aborted-before-dispatch")
    log: list[str] = []
    holder: dict[str, DrainScope] = {}

    def make(i: int, d: float):
        async def body():
            log.append(f"body{i}_start")
            await asyncio.sleep(d)
            log.append(f"body{i}_side_effect")
            return f"RESULT{i}"

        return body

    async def step():
        scope = DrainScope(cause=Cause.USER_ABORT)
        holder["s"] = scope
        started = [ToolRequest(f"c{i}", f"tool{i}", make(i, 0.05 + i * 0.02)) for i in range(3)]
        never = [ToolRequest("c3", "tool3", make(3, 0.0)), ToolRequest("c4", "tool4", make(4, 0.0))]
        for r in started + never:
            scope.declare(r)
        for r in started:
            scope.dispatch(r)
        try:
            await asyncio.gather(*(asyncio.shield(c.task) for c in scope.started))
        finally:
            await scope.drain(already_pending=True)

    c = asyncio.create_task(step())
    await asyncio.sleep(0.01)
    c.cancel()
    try:
        await c
    except asyncio.CancelledError:
        print("  step: CancelledError propagated")
    print(f"  ledger={log}")
    for r in holder["s"].results():
        print(f"    {r.call_id} {r.tool_name:7s} {r.status.value:24s} value={r.value!r}")


async def e7_timeout() -> None:
    hdr("E7  asyncio.timeout: kills the body unless shielded")
    log: list[str] = []

    async def body():
        log.append("body_start")
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            log.append("BODY_KILLED")
            raise
        log.append("side_effect")
        return "RESULT"

    async def bare():
        try:
            async with asyncio.timeout(0.05):
                return await body()
        except TimeoutError:
            return "TIMEOUT"

    print(f"  bare timeout      : {await bare()!r}  ledger={log}")
    log.clear()

    async def shielded():
        inner = asyncio.create_task(body())
        try:
            async with asyncio.timeout(0.05):
                return await asyncio.shield(inner)
        except TimeoutError:
            return await asyncio.shield(inner)

    print(f"  timeout + shield  : {await shielded()!r}  ledger={log}")


async def e8_normal_path() -> None:
    hdr("E8  normal path is unaffected")
    log: list[str] = []

    def make(i: int):
        async def body():
            await asyncio.sleep(0.01)
            log.append(f"body{i}_ok")
            return i * 10

        return body

    reqs = [ToolRequest(f"c{i}", f"tool{i}", make(i)) for i in range(3)]
    results, scope = await dispatch_and_drain(reqs)
    print(f"  results={[(r.call_id, r.value) for r in results]}")
    print(f"  absorbed={scope.drain_absorbed_cancels}  ledger={log}")


async def main() -> None:
    print(f"python {sys.version.split()[0]}")
    await e1_naive()
    await e2_shield_strands()
    await e3_loop_teardown()
    await e4_repeat_cancel()
    await e5_baseexception()
    await e6_multi()
    await e7_timeout()
    await e8_normal_path()


if __name__ == "__main__":
    asyncio.run(main())

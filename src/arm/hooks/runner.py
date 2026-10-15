"""The hook runner: executing commands and translating their output.

A hook is a subprocess. Its contract is: read a JSON payload on stdin, write a
JSON decision on stdout, and use stderr for anything a human should see.

Three rules govern what happens when a hook misbehaves, and all three exist
because the alternative is a broken hook breaking the agent:

**A hook failure is not a turn failure.** A command that cannot start, times out,
or prints garbage produces no decision -- the run continues as if the hook had
said "allow". A hook is an addition to the runtime, so it must not be able to
remove the runtime's ability to work.

**A timeout kills the process group.** Killing only the direct child would leave
its children running, and a hook that spawned a build would keep consuming the
machine after the runtime gave up on it.

**The payload is the trusted, complete description of what is happening.** The
hook receives a structured event rather than a set of environment variables,
because a payload can be versioned and validated, and an environment cannot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..kernel.errors import describe
from .protocol import (
    DECISION_ALLOW,
    DECISION_ASK,
    DECISION_DENY,
    STDERR_SUMMARY_CHARS,
    Decision,
    HookHandler,
    merge_decisions,
)

log = logging.getLogger("arm.hooks.runner")


@dataclass(frozen=True, slots=True)
class HookOutcome:
    """What running one hook produced, for the log record."""

    handler_id: str
    point: str
    decision: Decision
    exit_code: int | None = None
    stderr_summary: str = ""
    duration_ms: int = 0
    #: Why there was no decision, when that is the case.
    failure: str = ""


def _decision_from_output(
    handler: HookHandler, stdout: bytes, stderr: bytes, exit_code: int
) -> tuple[Decision, str]:
    """Translate a hook's output into a decision.

    Two conventions are honoured, because both dialects use them:

    * a **non-zero exit code** means "deny", with stderr as the reason. This is
      the older convention and the one a shell script naturally produces;
    * a **JSON object on stdout** with a ``decision`` field is the explicit form,
      and takes precedence when present.
    """
    text = stdout.decode("utf-8", "replace").strip()
    stderr_text = stderr.decode("utf-8", "replace").strip()

    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Not JSON: fall back to the exit-code convention rather than
            # treating unparseable output as an error. A hook that prints a
            # human-readable line and exits 0 is saying "allow".
            payload = None

        if isinstance(payload, Mapping):
            decision = str(payload.get("decision", "") or "").strip().lower()
            if decision in ("allow", "approve", "continue", ""):
                kind = DECISION_ALLOW
            elif decision in ("ask", "confirm"):
                kind = DECISION_ASK
            elif decision in ("deny", "block", "reject"):
                kind = DECISION_DENY
            else:
                # An unknown decision is not silently treated as allow: the hook
                # tried to say something and we did not understand it.
                return (
                    Decision(
                        kind=DECISION_ALLOW,
                        reason=(
                            f"the hook returned an unrecognised decision "
                            f"{decision!r}; treating it as allow"
                        ),
                    ),
                    "",
                )

            context: list[str] = []
            for key in ("additionalContext", "context", "systemMessage"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    context.append(value.strip())
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    context.extend(str(v) for v in value if str(v).strip())

            feedback = str(payload.get("feedback", "") or "")
            reason = str(payload.get("reason", "") or "")

            # A non-zero exit still denies, even if the JSON said allow: the
            # process failing is a stronger signal than its last words.
            if exit_code != 0 and kind == DECISION_ALLOW:
                kind = DECISION_DENY
                reason = reason or stderr_text or f"the hook exited {exit_code}"

            return (
                Decision(
                    kind=kind,
                    reason=reason,
                    context=tuple(context),
                    stop=bool(payload.get("continue") is False),
                    feedback=feedback,
                ),
                stderr_text,
            )

    if exit_code != 0:
        return (
            Decision(
                kind=DECISION_DENY,
                reason=stderr_text or f"the hook exited with code {exit_code}",
            ),
            stderr_text,
        )

    return Decision(kind=DECISION_ALLOW), stderr_text


async def run_hook(
    handler: HookHandler,
    payload: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str = "",
) -> HookOutcome:
    """Run one hook and return its outcome.

    **Never raises.** Every failure mode produces an ``allow`` decision with the
    reason recorded. A hook is an addition to the runtime; it must not be able to
    remove the runtime's ability to work.
    """
    started = asyncio.get_running_loop().time()

    # The payload is the complete description of the event. A hook reads it from
    # stdin rather than from environment variables, because a payload can be
    # versioned and validated and an environment cannot.
    stdin_payload = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")

    child_env = dict(os.environ if env is None else env)
    # The dialect and point are passed as environment as well, because a shell
    # script is far more likely to branch on an env var than to parse stdin.
    child_env["ARM_HOOK_POINT"] = handler.point
    child_env["ARM_HOOK_ID"] = handler.handler_id

    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            handler.command,
            *handler.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
            cwd=cwd or None,
            # A new process group, so a timeout can kill the whole tree rather
            # than just the direct child.
            start_new_session=True,
        )
    except FileNotFoundError:
        return _failed(handler, started, f"the hook command {handler.command!r} was not found")
    except OSError as exc:
        return _failed(handler, started, f"could not start the hook: {describe(exc)}")

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin_payload), timeout=handler.timeout_ms / 1000.0
        )
    except asyncio.TimeoutError:
        await _kill_group(process)
        return _failed(
            handler,
            started,
            f"the hook timed out after {handler.timeout_ms}ms",
        )
    except asyncio.CancelledError:
        await _kill_group(process)
        raise
    except Exception as exc:  # noqa: BLE001 - a hook fault is not a turn fault
        await _kill_group(process)
        return _failed(handler, started, f"the hook failed: {describe(exc)}")

    exit_code = process.returncode if process.returncode is not None else -1
    decision, stderr_text = _decision_from_output(handler, stdout, stderr, exit_code)

    return HookOutcome(
        handler_id=handler.handler_id,
        point=handler.point,
        decision=decision,
        exit_code=exit_code,
        stderr_summary=stderr_text[:STDERR_SUMMARY_CHARS],
        duration_ms=int((asyncio.get_running_loop().time() - started) * 1000),
    )


def _failed(handler: HookHandler, started: float, reason: str) -> HookOutcome:
    """An outcome for a hook that could not produce a decision.

    The decision is ``allow``, which is the permissive direction -- and that is
    the right choice here. A hook that cannot run has told us nothing, and
    treating "told us nothing" as "deny" would mean a missing formatter binary
    could block every tool call.
    """
    log.warning("hook %s: %s", handler.handler_id, reason)
    return HookOutcome(
        handler_id=handler.handler_id,
        point=handler.point,
        decision=Decision(kind=DECISION_ALLOW, reason=reason),
        exit_code=None,
        failure=reason,
        duration_ms=int((asyncio.get_running_loop().time() - started) * 1000),
    )


async def _kill_group(process: asyncio.subprocess.Process) -> None:
    """Kill a hook's whole process group.

    Killing only the direct child would leave its children running: a hook that
    spawned a build would keep consuming the machine long after the runtime gave
    up on it.
    """
    if process.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except ProcessLookupError:
            return
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except (asyncio.TimeoutError, ProcessLookupError):
        pass


class HookRunner:
    """Runs the hooks registered for each point."""

    def __init__(
        self,
        handlers: Sequence[HookHandler] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str = "",
    ) -> None:
        self.handlers = tuple(handlers)
        self.env = env
        self.cwd = cwd
        #: Every outcome, in order. Read by a caller that wants to log them.
        self.outcomes: list[HookOutcome] = []

    def for_point(self, point: str, subject: str) -> tuple[HookHandler, ...]:
        """The handlers that apply to one event.

        Matching happens per event, not per configuration: a hook with a tool
        matcher fires only for the tools it names.
        """
        return tuple(
            handler
            for handler in self.handlers
            if handler.point == point and handler.matcher.matches(subject)
        )

    async def fire(
        self, point: str, subject: str, payload: Mapping[str, Any]
    ) -> Decision:
        """Run every matching hook and merge their decisions.

        Hooks run **concurrently**. They are independent processes with no shared
        state, so serialising them would add latency for no benefit -- and a hook
        that blocks on a network call would delay every other hook.
        """
        handlers = self.for_point(point, subject)
        if not handlers:
            return Decision(kind=DECISION_ALLOW)

        outcomes = await asyncio.gather(
            *(run_hook(h, payload, env=self.env, cwd=self.cwd) for h in handlers)
        )
        self.outcomes.extend(outcomes)
        return merge_decisions([o.decision for o in outcomes])

    async def fire_parallel(
        self, point: str, subject: str, payload: Mapping[str, Any]
    ) -> tuple[Decision, tuple[HookOutcome, ...]]:
        """Run hooks at a point that only observes, returning the outcomes.

        The distinction matters: a parallel point cannot change what happens, so
        its decisions are reported rather than applied. Pretending otherwise would
        let a "stop" hook silently have no effect, and the user would believe a
        guard was in place.
        """
        handlers = self.for_point(point, subject)
        if not handlers:
            return Decision(kind=DECISION_ALLOW), ()
        outcomes = tuple(
            await asyncio.gather(
                *(run_hook(h, payload, env=self.env, cwd=self.cwd) for h in handlers)
            )
        )
        self.outcomes.extend(outcomes)
        return merge_decisions([o.decision for o in outcomes]), outcomes

"""Background jobs: work that outlives the step that started it.

Two properties shape this module, and both are about not leaving a mess.

**Start is atomic.** Everything that can refuse -- an unknown owner, an invalid
label, a full concurrency budget -- is checked *before* any resource is created.
A rejected start leaves no job id and no process behind. The alternative is worse
than it sounds: a half-started job is invisible to the caller (which got an
error) and invisible to the registry (which never recorded it), so nothing will
ever clean it up.

**Authorization is by owner, not by secrecy.** Every operation takes the calling
session's id. A caller sees its own jobs; a caller with no id sees only unowned
ones. Knowing a job id is not permission to read it -- an id is a name, not a
capability.

The lifecycle is small on purpose:

    running -> stopping -> completed | killed | failed

``stopping`` exists because stopping is not instantaneous: a cancellation has
been *requested*, and the job has not yet confirmed it stopped. Collapsing that
into "killed" would claim the work is over while it is still running, and the
claim would be indistinguishable from a lie.

Output lives in a ring per job, so the model can consume it while an observer
tails the same bytes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from ..kernel.errors import ArmError
from .ring import OutputRing, ReadResult

log = logging.getLogger("arm.jobs")

#: Lifecycle states. ``stopping`` is a real state, not a formality: a
#: cancellation has been requested and not yet confirmed.
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_COMPLETED = "completed"
STATE_KILLED = "killed"
STATE_FAILED = "failed"

TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_KILLED, STATE_FAILED})

#: Why a job settled.
CAUSE_COMPLETION = "completion"
CAUSE_KILL = "kill"
CAUSE_TEARDOWN = "teardown"
CAUSE_FAILURE = "failure"

#: Per-owner ceiling on jobs that are running or stopping.
DEFAULT_MAX_CONCURRENT = 10

#: Retained output while running, and after settling.
DEFAULT_RETAIN_BYTES = 262_144
DEFAULT_SETTLED_RETAIN_BYTES = 16_384


class JobError(ArmError):
    """A job operation was refused."""

    def __init__(self, message: str, *, code: str = "JOB_INVALID") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class JobStatus:
    """A snapshot of one job, safe to hand to a caller."""

    id: str
    kind: str
    label: str
    state: str
    owner: str
    started_at: float
    finished_at: float | None = None
    settle_cause: str = ""
    detail: str = ""
    #: Producer-supplied result, delivered exactly once to the model.
    result: str = ""
    output_bytes: int = 0

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at


@dataclass
class Job:
    """One job's live state. Never handed out directly."""

    id: str
    kind: str
    label: str
    owner: str
    ring: OutputRing
    started_at: float
    state: str = STATE_RUNNING
    finished_at: float | None = None
    settle_cause: str = ""
    detail: str = ""
    result: str = ""
    #: Set by the producer so a kill can be requested.
    cancel: Callable[[], Awaitable[None]] | None = None
    #: The handle the producer holds, so a kill can signal its cancellation
    #: event. Without this the producer's own ``wait_cancelled`` never returns,
    #: and the job would sit in ``stopping`` until the task was torn down.
    handle: "JobHandle | None" = None
    #: The task driving the producer, if it runs in this process.
    task: "asyncio.Task[Any] | None" = None
    #: The model's consuming cursor. Pinned while it lags, so trimming cannot
    #: discard output the model has not read.
    model_cursor: int = 0
    #: Whether the producer's result has been delivered.
    result_delivered: bool = False
    #: Waiters released when the job settles.
    waiters: list[asyncio.Future[Any]] = field(default_factory=list)
    settled: bool = False

    def status(self) -> JobStatus:
        return JobStatus(
            id=self.id,
            kind=self.kind,
            label=self.label,
            state=self.state,
            owner=self.owner,
            started_at=self.started_at,
            finished_at=self.finished_at,
            settle_cause=self.settle_cause,
            detail=self.detail,
            result=self.result,
            output_bytes=self.ring.written,
        )


#: What a producer is: something that runs and can be cancelled.
Producer = Callable[["JobHandle"], Awaitable[str | None]]


class JobHandle:
    """What a producer receives: a way to emit output and learn it was cancelled.

    Deliberately narrow. A producer that could inspect the registry could
    interfere with other jobs; a producer that could settle its own job could
    report success before finishing.
    """

    def __init__(self, job: Job) -> None:
        self._job = job
        self._cancelled = asyncio.Event()

    @property
    def id(self) -> str:
        return self._job.id

    @property
    def label(self) -> str:
        return self._job.label

    def emit(self, text: str) -> None:
        """Append output. Safe to call from the producer at any time."""
        self._job.ring.append(text)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    async def wait_cancelled(self) -> None:
        """Suspend until cancellation is requested.

        A producer that wants to shut down gracefully awaits this; one that does
        not care simply finishes. Either way the registry stops it.
        """
        await self._cancelled.wait()

    def _request_cancel(self) -> None:
        self._cancelled.set()


class JobRegistry:
    """Tracks background jobs for a set of owners.

    Holds no reference to any agent or session object -- only to opaque owner
    ids. That keeps the registry usable from a test, from a CLI and from the
    loop without any of them having to construct an agent.
    """

    def __init__(
        self,
        *,
        max_concurrent_per_owner: int = DEFAULT_MAX_CONCURRENT,
        retain_bytes: int = DEFAULT_RETAIN_BYTES,
        settled_retain_bytes: int = DEFAULT_SETTLED_RETAIN_BYTES,
    ) -> None:
        self.max_concurrent_per_owner = max_concurrent_per_owner
        self.retain_bytes = retain_bytes
        self.settled_retain_bytes = settled_retain_bytes
        self._jobs: dict[str, Job] = {}
        #: Per-kind counters, so ids are ``<kind>-N`` and each kind counts
        #: independently. A global counter would leak how many jobs of other
        #: kinds exist.
        self._counters: dict[str, int] = {}
        #: Owners that may not start jobs, because nothing would collect them.
        self._served_owners: set[str] | None = None

    # -- ownership -------------------------------------------------------

    def serve_owners(self, owners: Sequence[str]) -> None:
        """Restrict job ownership to these owners.

        An owner that nothing serves is refused at start rather than accepted and
        forgotten: a job whose owner cannot receive it is work that will run to
        completion with nobody listening.
        """
        self._served_owners = set(owners)

    def _check_owner(self, owner: str) -> None:
        if self._served_owners is not None and owner and owner not in self._served_owners:
            raise JobError(
                f"no job controller serves owner {owner!r}; a job started for it "
                "could not be collected or stopped",
                code="JOB_OWNER_UNSERVED",
            )

    def _visible(self, job: Job, caller: str) -> bool:
        """Whether ``caller`` may see ``job``.

        Authorization is by owner, not by knowing the id: an id is a name, not a
        capability.
        """
        if not job.owner:
            return True
        return job.owner == caller

    # -- starting --------------------------------------------------------

    def start(
        self,
        *,
        kind: str,
        producer: Producer,
        label: str = "",
        owner: str = "",
        run: bool = True,
    ) -> str:
        """Start a job, returning its id.

        Every check happens before anything is created, so a refusal leaves no
        trace. The id is allocated before the producer runs but registered only
        after it is successfully launched, so a producer that throws immediately
        consumes an id and nothing else -- which is why ids are allowed to have
        gaps.
        """
        if not kind or not kind.strip():
            raise JobError("a job needs a kind", code="JOB_NO_KIND")
        if kind in TERMINAL_STATES:
            raise JobError(
                f"job kind {kind!r} collides with a lifecycle state name",
                code="JOB_BAD_KIND",
            )

        # -- prechecks, in order. Nothing is allocated before the last one.
        self._check_owner(owner)

        if label and "\n" in label:
            raise JobError(
                "a job label must be a single line", code="JOB_BAD_LABEL"
            )

        active = sum(
            1
            for job in self._jobs.values()
            if job.owner == owner and job.state in (STATE_RUNNING, STATE_STOPPING)
        )
        if active >= self.max_concurrent_per_owner:
            raise JobError(
                f"owner {owner!r} already has {active} running jobs "
                f"(limit {self.max_concurrent_per_owner}); wait for one to finish",
                code="JOB_TOO_MANY",
            )

        job_id = self._allocate_id(kind)
        ring = OutputRing(limit_bytes=self.retain_bytes)
        job = Job(
            id=job_id,
            kind=kind,
            label=label or kind,
            owner=owner,
            ring=ring,
            started_at=time.monotonic(),
        )
        # The model's cursor starts at zero and pins the whole buffer, so nothing
        # is trimmed before the model has had a chance to read it.
        ring.pin("model", 0)
        job.model_cursor = 0

        handle = JobHandle(job)
        job.handle = handle

        if run:
            # A task needs a running loop. Saying so clearly beats an obscure
            # failure later: creating the task lazily would mean the job appears
            # registered but never actually runs, which looks like a hang.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                raise JobError(
                    "jobs must be started from inside a running event loop; "
                    "call start() from async code",
                    code="JOB_NO_LOOP",
                ) from None
            # The task is created before registration, so a failure to create it
            # is still a clean refusal.
            job.task = asyncio.ensure_future(self._drive(job, handle, producer))

            # A done-callback is the only reliable way to settle a cancelled job.
            #
            # A task cancelled *before it first runs* never enters its coroutine
            # at all: no ``except`` and no ``finally`` inside the body executes,
            # because the body is never started. Without this callback, killing a
            # just-started job leaves it in ``stopping`` forever and a caller
            # waiting on it waits until it gives up.
            #
            # A callback is attached to the *task*, so it runs on every
            # completion path -- including that one. Settling is idempotent, so
            # the paths that already settled are unaffected.
            job.task.add_done_callback(lambda _t: self._on_task_done(job))

        # The commit point: after this the job exists and the caller owns it.
        self._jobs[job_id] = job
        return job_id

    def _allocate_id(self, kind: str) -> str:
        self._counters[kind] = self._counters.get(kind, 0) + 1
        return f"{kind}-{self._counters[kind]}"

    async def _drive(self, job: Job, handle: JobHandle, producer: Producer) -> None:
        """Run a producer and settle the job, whatever happens.

        The cancellation path needs care. When this task is cancelled, any
        ``await`` inside the handler can be cancelled too -- including the await
        that would settle the job. That is how a killed job ends up stuck in
        ``stopping`` forever, with a caller waiting on a state change that never
        comes.

        So settling on the cancellation path is done **synchronously**, with the
        cancellation swallowed first so it cannot interrupt the settle itself.
        """
        try:
            result = await producer(handle)
            self._settle_sync(job, STATE_COMPLETED, CAUSE_COMPLETION, result=result or "")
        except asyncio.CancelledError:
            # Swallow the cancellation so the settle below cannot be interrupted,
            # then re-raise so the task still reports as cancelled.
            self._settle_sync(job, STATE_KILLED, CAUSE_KILL, detail="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - a failing producer is a state
            self._settle_sync(job, STATE_FAILED, CAUSE_FAILURE, detail=str(exc)[:500])


    def _on_task_done(self, job: Job) -> None:
        """Settle a job whose driving task has finished, whatever the reason.

        Runs for every outcome, including a task cancelled before its coroutine
        was ever entered -- which is the case no in-body handler can catch.
        """
        if job.settled:
            return
        task = job.task
        if task is not None and task.cancelled():
            self._settle_sync(
                job, STATE_KILLED, CAUSE_KILL, detail="cancelled before it started"
            )
            return
        # The body already settled, or the task ended some other way. Recording a
        # failure is better than leaving the job in a non-terminal state forever.
        detail = ""
        if task is not None and task.exception() is not None:
            detail = str(task.exception())[:500]
        self._settle_sync(job, STATE_FAILED, CAUSE_FAILURE, detail=detail)

    # -- settling --------------------------------------------------------

    def _settle_sync(
        self,
        job: Job,
        state: str,
        cause: str,
        *,
        detail: str = "",
        result: str = "",
    ) -> None:
        """Move a job to a terminal state. First settle wins.

        Synchronous by design: it is called from the cancellation path, where an
        ``await`` could itself be cancelled and leave the job stuck.
        """
        if job.settled:
            return
        job.settled = True
        job.state = state
        job.settle_cause = cause
        job.finished_at = time.monotonic()
        if detail:
            job.detail = detail
        if result:
            job.result = result

        # The model pin is deliberately NOT released here.
        #
        # Releasing it would let the settled trim discard output the model has
        # not read yet. The pin stays at the model's cursor, so the trim below can
        # only reclaim what has actually been delivered -- and the ring's own
        # limit is what bounds the buffer until then.

        # Waiters are released before any trimming, so a waiter that wakes and
        # reads cannot find the output it was waiting for already discarded.
        for waiter in job.waiters:
            if not waiter.done():
                waiter.set_result(job.status())
        job.waiters.clear()

        # No trimming here on purpose. The model has not read the output yet, so
        # the only correct budget is the one that was in force while running --
        # and applying the smaller settled budget now would discard exactly the
        # bytes the model is about to ask for. Trimming happens on read, once the
        # delivered portion is known.

    def _trim_settled(self, job: Job) -> None:
        """Reduce a settled job's retained output.

        Never trims past the model's cursor: the model has not read that output,
        and discarding it would silently lose the job's result. The settled limit
        is therefore a *target*, not a guarantee -- correctness wins over it.

        The budget covers only what is still unread, so a settled job whose output
        the model has already consumed shrinks to the settled limit. Setting the
        budget to the whole buffer instead would keep every byte alive until the
        read, which is the opposite of the intent.
        """
        unread = max(0, job.ring.written - job.model_cursor)
        job.ring.limit_bytes = max(self.settled_retain_bytes, unread)
        job.ring._trim()

    # -- reading ---------------------------------------------------------

    def get(self, job_id: str, *, caller: str = "") -> JobStatus | None:
        """One job's status, or ``None`` if it is invisible to the caller."""
        job = self._jobs.get(job_id)
        if job is None or not self._visible(job, caller):
            return None
        return job.status()

    def list(self, *, caller: str = "", owner: str | None = None) -> list[JobStatus]:
        """Jobs visible to the caller, newest first."""
        out = [
            job.status()
            for job in self._jobs.values()
            if self._visible(job, caller) and (owner is None or job.owner == owner)
        ]
        out.sort(key=lambda s: s.started_at, reverse=True)
        return out

    def read(self, job_id: str, *, caller: str = "") -> ReadResult | None:
        """Consume output: advances the model's cursor.

        This is the model's read. It also moves the pin forward, which is what
        allows the ring to trim the part that has now been delivered.
        """
        job = self._jobs.get(job_id)
        if job is None or not self._visible(job, caller):
            return None

        result = job.ring.read(job.model_cursor)
        job.model_cursor = result.next_offset
        # The cursor now pins only what has not been read, so the bytes just
        # delivered become eligible for trimming.
        job.ring.pin("model", job.model_cursor)
        # Trimming is attempted on every read, not only after settling: a long
        # running job whose model keeps up must not grow without bound just
        # because it has not finished yet.
        job.ring._trim()
        if job.settled:
            self._trim_settled(job)
        return result

    def read_at(self, job_id: str, offset: int, *, caller: str = "") -> ReadResult | None:
        """Read without consuming, at a caller-supplied offset.

        This is the observer's read. It cannot interfere with the model's cursor,
        which is the entire reason there are two.
        """
        job = self._jobs.get(job_id)
        if job is None or not self._visible(job, caller):
            return None
        return job.ring.read(offset)

    def take_result(self, job_id: str, *, caller: str = "") -> str:
        """Deliver the producer's result exactly once.

        Exactly once, because a result re-delivered on every read would be
        appended to the transcript repeatedly, and the model would see the same
        conclusion as though it were new information each time.
        """
        job = self._jobs.get(job_id)
        if job is None or not self._visible(job, caller):
            return ""
        if job.result_delivered or not job.result:
            return ""
        job.result_delivered = True
        return job.result

    # -- stopping --------------------------------------------------------

    async def kill(self, job_id: str, *, caller: str = "") -> str:
        """Request cancellation.

        Returns ``"requested"`` or ``"already-finished"``. The state becomes
        ``stopping``, not ``killed``: the cancellation has been asked for and not
        yet confirmed, and claiming otherwise would report work as finished while
        it is still running.
        """
        job = self._jobs.get(job_id)
        if job is None or not self._visible(job, caller):
            raise JobError(f"no visible job {job_id!r}", code="JOB_NOT_FOUND")
        if job.settled:
            return "already-finished"

        # Signal cooperative cancellation FIRST, so a producer that wants to
        # clean up has the chance. The task is cancelled immediately afterwards,
        # so a producer that ignores the signal is still stopped -- cooperation
        # is preferred, not required.
        if job.handle is not None:
            job.handle._request_cancel()

        # A producer-supplied cancel callback runs before the state changes. If
        # it raises, the state is untouched and the caller can retry -- changing
        # the state first would leave a job marked stopping with nothing stopping
        # it.
        if job.cancel is not None:
            try:
                await job.cancel()
            except Exception as exc:  # noqa: BLE001
                log.warning("cancel callback for %s failed: %s", job_id, exc)

        job.state = STATE_STOPPING

        # The task is given a brief chance to finish its own cleanup before it is
        # cancelled outright. A producer that honours the cancellation signal may
        # need to close a file or kill a child process, and cancelling instantly
        # would interrupt that cleanup halfway -- which is worse than either
        # letting it finish or killing it outright.
        #
        # The wait is bounded, so a producer that ignores the signal is still
        # stopped promptly.
        if job.task is not None and not job.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(job.task), timeout=0.05)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception:  # noqa: BLE001 - the producer failed; settling handles it
                pass

        if job.task is not None and not job.task.done():
            job.task.cancel()
        return "requested"

    async def wait(
        self, job_id: str, *, timeout: float | None = None, caller: str = ""
    ) -> JobStatus:
        """Wait for a job to settle, or for the timeout.

        A timeout is an error rather than a silent return: a caller that asked to
        wait and got nothing back cannot tell "not finished" from "finished with
        no output".
        """
        job = self._jobs.get(job_id)
        if job is None or not self._visible(job, caller):
            raise JobError(f"no visible job {job_id!r}", code="JOB_NOT_FOUND")
        if job.settled:
            return job.status()

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[Any] = loop.create_future()
        job.waiters.append(waiter)

        try:
            if timeout is None:
                return await waiter
            return await asyncio.wait_for(waiter, timeout=timeout)
        except asyncio.TimeoutError:
            if waiter in job.waiters:
                job.waiters.remove(waiter)
            raise JobError(
                f"job {job_id!r} did not finish within {timeout}s",
                code="JOB_WAIT_TIMEOUT",
            ) from None
        except asyncio.CancelledError:
            if waiter in job.waiters:
                job.waiters.remove(waiter)
            raise

    def remove(self, job_id: str, *, caller: str = "") -> bool:
        """Forget a settled job.

        Only settled jobs can be removed. Removing a running job would orphan the
        work: nothing would be left to stop it and nothing would be left to
        report its outcome.
        """
        job = self._jobs.get(job_id)
        if job is None or not self._visible(job, caller):
            return False
        if not job.settled:
            raise JobError(
                f"job {job_id!r} is still {job.state}; kill it before removing it",
                code="JOB_NOT_SETTLED",
            )
        del self._jobs[job_id]
        return True

    # -- teardown --------------------------------------------------------

    async def shutdown(self, *, owner: str = "") -> list[str]:
        """Stop every job, waiting for each to settle.

        Used when an owner goes away. Cancellation is attempted, then the job is
        awaited -- and a job that refuses to stop is reported rather than
        claimed as stopped. Saying "stopped" about work that is still running is
        the one outcome that would make the registry untrustworthy.
        """
        targets = [
            job
            for job in self._jobs.values()
            if not job.settled and (not owner or job.owner == owner)
        ]
        orphans: list[str] = []

        for job in targets:
            try:
                await self.kill(job.id, caller=job.owner)
            except JobError:
                pass

        for job in targets:
            try:
                # Waited as the job's *own* owner, because that is who is
                # authorized to see it. Passing a different caller would make
                # every job look invisible and the shutdown would report
                # orphans that are actually fine.
                await self.wait(job.id, timeout=5.0, caller=job.owner)
            except (JobError, asyncio.CancelledError):
                # The job did not settle. It is reported as a possible orphan
                # rather than silently dropped.
                orphans.append(job.id)
                job.settle_cause = CAUSE_TEARDOWN
                continue
            job.settle_cause = CAUSE_TEARDOWN

        return orphans

    @property
    def size(self) -> int:
        return len(self._jobs)

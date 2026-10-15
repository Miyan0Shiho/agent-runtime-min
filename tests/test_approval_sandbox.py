"""Approval and sandbox tests.

The theme is that **denial is the default and permission is explicit**. So the
tests that matter most are the negative ones: a missing channel must deny, a
malformed answer must deny, and a policy check that is easy to skip must instead
raise.

The sandbox tests use a real temporary directory, because the boundary being
tested *is* a filesystem boundary -- a test with a mocked filesystem would verify
the mock rather than the boundary.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from arm.llm.mock import tool_response
from arm.approval.service import (
    OUTCOME_ALLOWED_ONCE,
    OUTCOME_CANCELLED,
    OUTCOME_REJECTED,
    OUTCOME_UNAVAILABLE,
    POLICY_ASK,
    POLICY_NEVER,
    ApprovalDecision,
    ApprovalPolicyError,
    ApprovalRequest,
    ApprovalService,
    NoOpenTurnError,
    allow_all,
    answer_from,
    deny_all,
)
from arm.sandbox.policy import (
    CODE_EXEC_DISABLED,
    CODE_NO_ACCESS,
    CODE_OUTSIDE_WORKSPACE,
    CODE_READ_ONLY,
    CODE_SENSITIVE_PATH,
    MODE_NONE,
    MODE_READ_ONLY,
    MODE_WORKSPACE_WRITE,
    OP_EXECUTE,
    OP_READ,
    OP_WRITE,
    SandboxConfigurationError,
    SandboxDenied,
    SandboxError,
    SandboxedFS,
    SandboxPolicy,
    escalation_for,
    resolve_policy,
)


def request(tool: str = "bash", reason: str = "") -> ApprovalRequest:
    return ApprovalRequest(tool=tool, reason=reason, call_id="c1")


def service(**kwargs) -> ApprovalService:
    recorded: list[tuple[str, dict]] = []
    kwargs.setdefault("recorder", lambda t, d: recorded.append((t, dict(d))))
    svc = ApprovalService(**kwargs)
    svc.recorded = recorded  # type: ignore[attr-defined]
    return svc


# ---------------------------------------------------------------------------
# The policy vocabulary
# ---------------------------------------------------------------------------


def test_only_two_policies_exist():
    """A third policy would be a third set of semantics to reason about."""
    from arm.approval.service import POLICIES

    assert POLICIES == {"ask", "never"}


def test_an_unknown_policy_is_refused_at_construction():
    """A typo in a policy must fail when it is configured, not when it is used."""
    with pytest.raises(ApprovalPolicyError, match="unknown approval policy"):
        ApprovalService(policy="maybe")


def test_the_default_policy_is_ask():
    """Asking is the default; never is the deliberate choice."""
    assert ApprovalService().policy == POLICY_ASK


# ---------------------------------------------------------------------------
# never: decided before any dispatch
# ---------------------------------------------------------------------------


def test_never_rejects_without_consulting_anything():
    """The guarantee must not depend on what happens to be registered."""
    consulted: list[str] = []

    def spy(req: ApprovalRequest) -> str:
        consulted.append(req.tool)
        return OUTCOME_ALLOWED_ONCE

    svc = service(policy=POLICY_NEVER, answerers=[spy])
    decision = asyncio.run(svc.request(request()))

    assert decision.outcome == OUTCOME_REJECTED
    assert consulted == [], "an answerer was consulted under policy 'never'"


def test_never_rejects_even_with_an_allow_all_answerer():
    """The policy outranks the channel.

    Implementing ``never`` as a listener would let anything registered ahead of
    it run first, and the promise would degrade to "nothing is asked, unless
    something else was registered earlier".
    """
    svc = service(policy=POLICY_NEVER, answerers=[allow_all()])
    assert not asyncio.run(svc.request(request())).granted


def test_never_is_deterministic_across_calls():
    """The same input gives the same answer, every time."""
    svc = service(policy=POLICY_NEVER, answerers=[allow_all()])
    outcomes = {asyncio.run(svc.request(request())).outcome for _ in range(5)}
    assert outcomes == {OUTCOME_REJECTED}


# ---------------------------------------------------------------------------
# ask: fail closed on every kind of trouble
# ---------------------------------------------------------------------------


def test_a_missing_channel_denies():
    """No channel is a denial, not a permission.

    A policy that said "ask" has already declined to auto-allow; reading a
    missing channel as consent would invert its meaning.
    """
    svc = service(policy=POLICY_ASK, answerers=[])
    decision = asyncio.run(svc.request(request()))

    assert decision.outcome == OUTCOME_UNAVAILABLE
    assert not decision.granted
    assert "no approval channel" in decision.reason


def test_an_allowing_channel_grants():
    """The one path that permits."""
    svc = service(answerers=[allow_all()])
    assert asyncio.run(svc.request(request())).granted


def test_a_denying_channel_rejects():
    """An explicit refusal."""
    svc = service(answerers=[deny_all()])
    decision = asyncio.run(svc.request(request()))

    assert decision.outcome == OUTCOME_REJECTED
    assert "rejected" in decision.reason


def test_a_malformed_answer_is_treated_as_unavailable():
    """An answer outside the vocabulary is not an answer.

    Defaulting it would let an approval bug become an approval grant.
    """
    svc = service(answerers=[lambda _r: "sure, go ahead"])
    decision = asyncio.run(svc.request(request()))

    assert decision.outcome == OUTCOME_UNAVAILABLE
    assert not decision.granted


def test_an_answerer_that_raises_denies():
    """A broken channel must not become an open door."""
    def broken(_request: ApprovalRequest) -> str:
        raise RuntimeError("the approval UI crashed")

    svc = service(answerers=[broken])
    decision = asyncio.run(svc.request(request()))

    assert decision.outcome == OUTCOME_UNAVAILABLE
    assert not decision.granted


def test_an_answerer_that_raises_synchronously_denies():
    """A synchronous throw must take the same path as an asynchronous one.

    The distinction matters because a naive implementation awaits first, and a
    synchronous throw would escape to the caller instead of being normalized.
    """
    def broken(_request: ApprovalRequest) -> str:
        raise ValueError("immediate failure")

    svc = service(answerers=[broken])
    decision = asyncio.run(svc.request(request()))
    assert decision.outcome == OUTCOME_UNAVAILABLE


def test_an_async_answerer_works():
    """An approval may need to wait for a human."""
    async def answer(_request: ApprovalRequest) -> str:
        return OUTCOME_ALLOWED_ONCE

    svc = service(answerers=[answer])
    assert asyncio.run(svc.request(request())).granted


def test_a_timeout_denies():
    """An unanswered prompt must not hang a turn forever."""
    async def slow(_request: ApprovalRequest) -> str:
        await asyncio.sleep(5)
        return OUTCOME_ALLOWED_ONCE

    svc = service(answerers=[slow], timeout=0.02)
    decision = asyncio.run(svc.request(request()))

    assert decision.outcome == OUTCOME_UNAVAILABLE
    assert "timed out" in decision.reason


def test_an_aborted_signal_cancels_rather_than_rejects():
    """Cancellation is not a policy refusal, and the two route differently."""
    class AbortedSignal:
        aborted = True

    svc = service(answerers=[allow_all()])
    decision = asyncio.run(svc.request(request(), signal=AbortedSignal()))

    assert decision.outcome == OUTCOME_CANCELLED
    assert decision.was_cancelled
    assert not decision.granted


def test_cancellation_is_distinguishable_from_rejection():
    """The caller needs to tell "the user said no" from "the user stopped it"."""
    rejected = ApprovalDecision.rejected("no")
    cancelled = ApprovalDecision.aborted("stopped")

    assert not rejected.was_cancelled
    assert cancelled.was_cancelled
    assert not rejected.granted and not cancelled.granted


def test_only_allowed_once_grants():
    """Everything else denies, including an unrecognized outcome."""
    assert ApprovalDecision.allowed().granted
    for outcome in (OUTCOME_REJECTED, OUTCOME_CANCELLED, OUTCOME_UNAVAILABLE):
        assert not ApprovalDecision(outcome).granted
    assert not ApprovalDecision("something-else").granted


def test_the_first_usable_answerer_wins():
    """A chain, not a vote: two disagreeing answerers is a config bug."""
    svc = service(answerers=[deny_all(), allow_all()])
    assert not asyncio.run(svc.request(request())).granted


# ---------------------------------------------------------------------------
# Audit pairing
# ---------------------------------------------------------------------------


def test_every_decision_is_recorded_in_pairs():
    """An ask and its decision, or the transcript cannot explain itself."""
    svc = service(answerers=[allow_all()])
    asyncio.run(svc.request(request(tool="read")))

    kinds = [kind for kind, _data in svc.recorded]  # type: ignore[attr-defined]
    assert kinds == ["approval/asked", "approval/decided"]


def test_a_denied_decision_is_recorded_too():
    """The refusals are the ones an audit is most interested in."""
    svc = service(answerers=[deny_all()])
    asyncio.run(svc.request(request()))

    decided = svc.recorded[-1]  # type: ignore[attr-defined]
    assert decided[1]["outcome"] == OUTCOME_REJECTED


def test_the_decision_records_the_policy_in_force():
    """An audit that cannot say which policy applied is incomplete."""
    svc = service(policy=POLICY_NEVER)
    asyncio.run(svc.request(request()))

    asked = svc.recorded[0]  # type: ignore[attr-defined]
    assert asked[1]["policy"] == POLICY_NEVER


def test_a_failing_recorder_does_not_break_the_decision():
    """The decision is already made; a recording problem is not a tool failure."""
    def broken(_kind: str, _data: object) -> None:
        raise RuntimeError("the log is full")

    svc = ApprovalService(answerers=[allow_all()], recorder=broken)
    decision = asyncio.run(svc.request(request()))
    assert decision.granted


def test_an_approval_outside_a_turn_is_refused():
    """A bare event between turns is indistinguishable from a crash artefact."""
    svc = service(
        answerers=[allow_all()], open_turn=lambda: False
    )
    with pytest.raises(NoOpenTurnError, match="outside an open turn"):
        asyncio.run(svc.request(request()))


def test_an_approval_inside_a_turn_proceeds():
    """The normal case."""
    svc = service(answerers=[allow_all()], open_turn=lambda: True)
    assert asyncio.run(svc.request(request())).granted


# ---------------------------------------------------------------------------
# Per-session overrides
# ---------------------------------------------------------------------------


def test_a_session_can_override_the_policy():
    """A headless session and an interactive one can differ in one process."""
    svc = service(policy=POLICY_ASK, answerers=[allow_all()])
    svc.set_override("ci", POLICY_NEVER)

    assert asyncio.run(svc.request(request(), session_id="ci")).outcome == (
        OUTCOME_REJECTED
    )
    assert asyncio.run(svc.request(request(), session_id="local")).granted


def test_an_override_can_be_cleared():
    """The override is not permanent."""
    svc = service(answerers=[allow_all()])
    svc.set_override("s", POLICY_NEVER)
    svc.clear_override("s")
    assert asyncio.run(svc.request(request(), session_id="s")).granted


def test_an_invalid_override_is_refused():
    """A typo must fail at the point it is written."""
    svc = service()
    with pytest.raises(ApprovalPolicyError):
        svc.set_override("s", "sometimes")


# ---------------------------------------------------------------------------
# Ready-made answerers
# ---------------------------------------------------------------------------


def test_answer_from_grants_only_where_the_predicate_holds():
    """A rule expressed as a predicate."""
    svc = service(answerers=[answer_from(lambda r: r.tool == "read")])

    assert asyncio.run(svc.request(request(tool="read"))).granted
    assert not asyncio.run(svc.request(request(tool="bash"))).granted


# ===========================================================================
# Sandbox
# ===========================================================================


def test_workspace_write_allows_a_write_inside(tmp_path: Path):
    """The ordinary case."""
    policy = SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path))
    policy.check(OP_WRITE, str(tmp_path / "file.txt"))


def test_workspace_write_refuses_a_write_outside(tmp_path: Path):
    """The boundary is the point."""
    policy = SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path))
    with pytest.raises(SandboxDenied) as info:
        policy.check(OP_WRITE, "/tmp/definitely-outside.txt")

    assert info.value.code == CODE_OUTSIDE_WORKSPACE
    assert info.value.path == "/tmp/definitely-outside.txt"


def test_the_boundary_cannot_be_spelled_around(tmp_path: Path):
    """A boundary that can be written around is not a boundary.

    Resolution happens before the comparison, so ``..`` does not escape.
    """
    policy = SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path / "inside"))
    (tmp_path / "inside").mkdir()

    escape = str(tmp_path / "inside" / ".." / "outside.txt")
    with pytest.raises(SandboxDenied) as info:
        policy.check(OP_WRITE, escape)
    assert info.value.code == CODE_OUTSIDE_WORKSPACE


def test_read_only_refuses_every_write(tmp_path: Path):
    """Even inside the workspace."""
    policy = SandboxPolicy(mode=MODE_READ_ONLY, workspace=str(tmp_path))
    with pytest.raises(SandboxDenied) as info:
        policy.check(OP_WRITE, str(tmp_path / "file.txt"))
    assert info.value.code == CODE_READ_ONLY


def test_read_only_permits_reading(tmp_path: Path):
    """A read-only session can still see the workspace."""
    policy = SandboxPolicy(mode=MODE_READ_ONLY, workspace=str(tmp_path))
    policy.check(OP_READ, str(tmp_path / "file.txt"))
    policy.check(OP_READ, "/etc/hosts")


def test_mode_none_refuses_reading_too(tmp_path: Path):
    """The strictest mode refuses everything."""
    policy = SandboxPolicy(mode=MODE_NONE, workspace=str(tmp_path))
    with pytest.raises(SandboxDenied) as info:
        policy.check(OP_READ, str(tmp_path / "file.txt"))
    assert info.value.code == CODE_NO_ACCESS


def test_credentials_are_protected_even_inside_the_workspace(tmp_path: Path):
    """A protected path is protected in every mode that allows writing.

    Not a security boundary against a determined agent -- the agent runs as the
    user -- but a mistake here damages the machine, and the check costs nothing.
    """
    policy = SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path))
    secret = tmp_path / ".ssh" / "id_rsa"
    secret.parent.mkdir(parents=True)

    with pytest.raises(SandboxDenied) as info:
        policy.check(OP_WRITE, str(secret))
    assert info.value.code == CODE_SENSITIVE_PATH


def test_a_pem_file_is_protected(tmp_path: Path):
    """Key material is recognised by extension too."""
    policy = SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path))
    with pytest.raises(SandboxDenied) as info:
        policy.check(OP_WRITE, str(tmp_path / "server.pem"))
    assert info.value.code == CODE_SENSITIVE_PATH


def test_a_protected_path_cannot_be_escalated(tmp_path: Path):
    """Offering to widen for it would be offering what the policy cannot give."""
    policy = SandboxPolicy(mode=MODE_READ_ONLY, workspace=str(tmp_path))
    try:
        policy.check(OP_WRITE, str(tmp_path / ".ssh" / "id_rsa"))
    except SandboxDenied:
        pass
    # In read-only mode the refusal is READ_ONLY, which is escalatable.
    denial = SandboxDenied("x", code=CODE_SENSITIVE_PATH, path="/x", op=OP_WRITE)
    with pytest.raises(SandboxError, match="protected in every mode"):
        escalation_for(denial, policy)


def test_execute_can_be_disabled_independently(tmp_path: Path):
    """A mode that permits writing may still forbid running commands."""
    policy = SandboxPolicy(
        mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path), allow_execute=False
    )
    with pytest.raises(SandboxDenied) as info:
        policy.check(OP_EXECUTE)
    assert info.value.code == CODE_EXEC_DISABLED


def test_mode_none_forbids_execution(tmp_path: Path):
    """The strictest mode refuses commands too."""
    policy = SandboxPolicy(mode=MODE_NONE, workspace=str(tmp_path))
    with pytest.raises(SandboxDenied) as info:
        policy.check(OP_EXECUTE)
    assert info.value.code == CODE_EXEC_DISABLED


def test_workspace_write_requires_a_workspace():
    """Without one it would mean "write anywhere", the opposite of the name."""
    with pytest.raises(SandboxConfigurationError, match="requires a workspace"):
        SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace="")


def test_an_unknown_mode_is_refused():
    """A typo must fail at construction."""
    with pytest.raises(SandboxConfigurationError, match="unknown sandbox mode"):
        SandboxPolicy(mode="everything", workspace="/tmp")


def test_an_unknown_operation_is_refused(tmp_path: Path):
    """A typo in the operation must not silently pass."""
    policy = SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path))
    with pytest.raises(SandboxConfigurationError, match="unknown operation"):
        policy.check("delete", str(tmp_path / "x"))


def test_extra_roots_widen_the_boundary(tmp_path: Path):
    """A caller can grant a second root without changing the mode."""
    extra = tmp_path / "extra"
    extra.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    policy = SandboxPolicy(
        mode=MODE_WORKSPACE_WRITE,
        workspace=str(workspace),
        extra_roots=(str(extra),),
    )
    policy.check(OP_WRITE, str(extra / "file.txt"))
    with pytest.raises(SandboxDenied):
        policy.check(OP_WRITE, str(tmp_path / "elsewhere.txt"))


def test_allows_is_a_non_raising_form(tmp_path: Path):
    """Some callers genuinely want to ask rather than act."""
    policy = SandboxPolicy(mode=MODE_READ_ONLY, workspace=str(tmp_path))
    assert policy.allows(OP_READ, str(tmp_path / "x"))
    assert not policy.allows(OP_WRITE, str(tmp_path / "x"))


def test_describe_explains_the_policy(tmp_path: Path):
    """A policy nobody can read is a policy nobody reviews."""
    assert "read-only" in SandboxPolicy(mode=MODE_READ_ONLY, workspace=str(tmp_path)).describe()
    assert "no filesystem" in SandboxPolicy(mode=MODE_NONE).describe()
    described = SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path)).describe()
    assert str(tmp_path) in described


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_the_default_preset_permits_workspace_writes(tmp_path: Path):
    """The everyday stance."""
    policy = resolve_policy(preset="default", workspace=str(tmp_path))
    assert policy.mode == MODE_WORKSPACE_WRITE
    assert policy.allow_execute


def test_the_unattended_preset_is_restrictive(tmp_path: Path):
    """An unattended run should not be able to change anything."""
    policy = resolve_policy(preset="unattended", workspace=str(tmp_path))
    assert policy.mode == MODE_READ_ONLY
    assert not policy.allow_execute


def test_the_locked_preset_permits_nothing(tmp_path: Path):
    """The strictest named stance."""
    policy = resolve_policy(preset="locked", workspace=str(tmp_path))
    assert policy.mode == MODE_NONE


def test_an_explicit_mode_overrides_a_preset(tmp_path: Path):
    """Start from a named stance and adjust one dimension."""
    policy = resolve_policy(
        preset="unattended", mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path)
    )
    assert policy.mode == MODE_WORKSPACE_WRITE


def test_an_unknown_preset_is_refused(tmp_path: Path):
    """A typo must fail when configured."""
    with pytest.raises(SandboxConfigurationError, match="unknown preset"):
        resolve_policy(preset="yolo", workspace=str(tmp_path))


# ---------------------------------------------------------------------------
# The filesystem seam
# ---------------------------------------------------------------------------


def test_the_fs_seam_reads_and_writes_inside_the_workspace(tmp_path: Path):
    """The seam is thin: a policy check plus the operation."""
    fs = SandboxedFS(SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path)))
    written = fs.write_text(str(tmp_path / "a.txt"), "hello")

    assert written == 5
    assert fs.read_text(str(tmp_path / "a.txt")) == "hello"
    assert fs.exists(str(tmp_path / "a.txt"))
    assert "a.txt" in fs.listdir(str(tmp_path))


def test_the_fs_seam_refuses_a_write_outside(tmp_path: Path):
    """The seam must not be a way around the policy."""
    fs = SandboxedFS(SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path)))
    with pytest.raises(SandboxDenied):
        fs.write_text("/tmp/outside-the-workspace.txt", "nope")


def test_the_fs_seam_creates_parent_directories(tmp_path: Path):
    """Writing a nested path should not require a separate mkdir step."""
    fs = SandboxedFS(SandboxPolicy(mode=MODE_WORKSPACE_WRITE, workspace=str(tmp_path)))
    fs.write_text(str(tmp_path / "a" / "b" / "c.txt"), "deep")
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


def test_the_fs_seam_reports_a_missing_file(tmp_path: Path):
    """A missing file is an error the caller can act on."""
    fs = SandboxedFS(SandboxPolicy(mode=MODE_READ_ONLY, workspace=str(tmp_path)))
    with pytest.raises(SandboxError, match="could not read"):
        fs.read_text(str(tmp_path / "absent.txt"))


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_an_escalation_carries_the_operation_it_is_for(tmp_path: Path):
    """An approval for "wider access in general" would be inherited by the next
    operation, which is exactly what must not happen."""
    policy = SandboxPolicy(mode=MODE_READ_ONLY, workspace=str(tmp_path))
    denial = SandboxDenied("no", code=CODE_READ_ONLY, path=str(tmp_path / "x"), op=OP_WRITE)

    escalation = escalation_for(denial, policy, reason="needs to write the fix")
    assert escalation.op == OP_WRITE
    assert escalation.path == str(tmp_path / "x")
    assert "write the fix" in escalation.reason


# ---------------------------------------------------------------------------
# The pipeline adapter
# ---------------------------------------------------------------------------


def test_the_hook_grants_when_the_service_grants():
    """The adapter must carry the decision through."""
    from arm.approval.service import approval_hook

    class Execution:
        name = "bash"
        arguments = {"command": "ls"}
        call_id = "c1"

    svc = service(answerers=[allow_all()])
    hook = approval_hook(svc)
    assert asyncio.run(hook(Execution(), "needs to run a command"))


def test_the_hook_denies_when_the_service_denies():
    """And a refusal must not become a grant."""
    from arm.approval.service import approval_hook

    class Execution:
        name = "bash"
        arguments = {}
        call_id = "c1"

    svc = service(answerers=[deny_all()])
    hook = approval_hook(svc)
    assert not asyncio.run(hook(Execution(), "needs to run a command"))


def test_the_hook_denies_when_there_is_no_channel():
    """Fail-closed must survive the adapter."""
    from arm.approval.service import approval_hook

    class Execution:
        name = "bash"
        arguments = {}
        call_id = "c1"

    svc = service(answerers=[])
    hook = approval_hook(svc)
    assert not asyncio.run(hook(Execution(), "needs approval"))


def test_escalation_is_a_second_separate_ask():
    """An escalation must not inherit the first refusal -- or the first grant."""
    from arm.approval.service import approval_hook

    class Execution:
        name = "write"
        arguments = {"path": "/etc/hosts"}
        call_id = "c1"

    asked: list[str] = []

    def answerer(request: ApprovalRequest) -> str:
        asked.append(request.reason)
        # Refuse the first ask, allow the escalation.
        return OUTCOME_ALLOWED_ONCE if "escalation" in request.reason else OUTCOME_REJECTED

    svc = service(answerers=[answerer])
    hook = approval_hook(svc, escalate=lambda _r: True)
    assert asyncio.run(hook(Execution(), "outside the workspace"))

    assert len(asked) == 2, "the escalation should have been asked separately"
    assert "escalation" in asked[1]
    # Both asks and both decisions are recorded.
    kinds = [k for k, _ in svc.recorded]  # type: ignore[attr-defined]
    assert kinds.count("approval/asked") == 2
    assert kinds.count("approval/decided") == 2


def test_without_an_escalation_callback_a_refusal_stands():
    """The escalation is opt-in, not automatic."""
    from arm.approval.service import approval_hook

    class Execution:
        name = "write"
        arguments = {}
        call_id = "c1"

    svc = service(answerers=[deny_all()])
    hook = approval_hook(svc)  # no escalate
    assert not asyncio.run(hook(Execution(), "denied"))


def test_the_hook_reports_the_tool_and_arguments_to_the_answerer():
    """An answerer must be able to show what would actually run."""
    from arm.approval.service import approval_hook

    class Execution:
        name = "bash"
        arguments = {"command": "rm -rf /"}
        call_id = "c42"

    seen: list[ApprovalRequest] = []

    def answerer(request: ApprovalRequest) -> str:
        seen.append(request)
        return OUTCOME_REJECTED

    svc = service(answerers=[answerer])
    asyncio.run(approval_hook(svc)(Execution(), "dangerous"))

    assert seen[0].tool == "bash"
    assert seen[0].arguments == {"command": "rm -rf /"}
    assert seen[0].call_id == "c42"


# ===========================================================================
# The sandbox must be enforced on the mandatory path
# ===========================================================================


def test_a_read_only_sandbox_blocks_a_write_through_the_pipeline():
    """A policy that is not enforced is not a policy.

    The sandbox was resolved, passed to the runtime, and consulted by nothing:
    the tools read and wrote the filesystem directly. Every sandbox test passed,
    because they tested the policy object rather than whether anything used it.
    """
    import asyncio
    import tempfile
    from pathlib import Path

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.builtin import builtin_tools
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many(builtin_tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)

    target = Path(tempfile.mkdtemp()) / "forbidden.txt"
    session = Session.create("sandboxed")
    loop = Runtime(
        session,
        MockAdapter(
            [
                tool_response(
                    ("c1", "write", {"path": str(target), "content": "nope"})
                ),
                text_response("ok"),
            ]
        ),
        registry,
        assembler=assembler,
        sandbox=SandboxPolicy(mode=MODE_READ_ONLY),
        model="mock",
    )

    loop.submit("write a file")
    asyncio.run(loop.run_until_idle())

    results = [e for e in session if e.type == "tool/result"]
    assert results, "the tool call was never recorded"
    assert results[0].data["is_error"] is True, (
        "a read-only sandbox allowed a write"
    )
    assert not target.exists(), "the file was created despite the refusal"


def test_a_read_only_sandbox_still_allows_a_read():
    """The refusal must be specific, not blanket."""
    import asyncio
    import tempfile
    from pathlib import Path

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.builtin import builtin_tools
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many(builtin_tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)

    readable = Path(tempfile.mkdtemp()) / "readable.txt"
    readable.write_text("hello there")

    session = Session.create("sandboxed-read")
    loop = Runtime(
        session,
        MockAdapter(
            [
                tool_response(("c1", "read", {"path": str(readable)})),
                text_response("ok"),
            ]
        ),
        registry,
        assembler=assembler,
        sandbox=SandboxPolicy(mode=MODE_READ_ONLY),
        model="mock",
    )

    loop.submit("read it")
    asyncio.run(loop.run_until_idle())

    results = [e for e in session if e.type == "tool/result"]
    assert results
    assert results[0].data["is_error"] is False, (
        "a read-only sandbox refused a read"
    )
    assert "hello there" in results[0].data["message"]["content"][0]["text"]


def test_a_sandbox_that_disallows_execution_blocks_bash():
    """``allow_execute`` must reach the tool, not just the policy object."""
    import asyncio

    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.builtin import builtin_tools
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many(builtin_tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)

    session = Session.create("no-exec")
    loop = Runtime(
        session,
        MockAdapter(
            [
                tool_response(("c1", "bash", {"command": "echo hi"})),
                text_response("ok"),
            ]
        ),
        registry,
        assembler=assembler,
        sandbox=SandboxPolicy(mode=MODE_READ_ONLY, allow_execute=False),
        model="mock",
    )

    loop.submit("run it")
    asyncio.run(loop.run_until_idle())

    results = [e for e in session if e.type == "tool/result"]
    assert results
    assert results[0].data["is_error"] is True, (
        "execution was permitted in a mode that disallows it"
    )
    assert "execution" in results[0].data["message"]["content"][0]["text"].lower()


def test_every_filesystem_tool_is_covered_by_the_table():
    """The table is the complete answer to "what does the sandbox cover".

    A tool that touches the filesystem and is absent from the table is a tool
    that bypasses the policy -- and the failure is silent, because the tool
    simply works when it should not.
    """
    from arm.sandbox.policy import TOOL_OPERATIONS, TOOL_PATH_LISTS

    covered = set(TOOL_OPERATIONS) | set(TOOL_PATH_LISTS)
    for name in ("read", "write", "edit", "bash"):
        assert name in covered, f"{name!r} bypasses the sandbox"

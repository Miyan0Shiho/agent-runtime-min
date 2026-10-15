"""Hook tests.

Three claims are under test:

* **Matching is dialect-specific, and the dialects differ.** Unifying them would
  silently change which hooks fire for one of the two, and a hook that stops
  firing is nearly impossible to notice.
* **Merging is most-restrictive-first.** Merging by configuration order would let
  adding a hook *loosen* an existing restriction.
* **A broken hook cannot break the agent.** A command that is missing, times out,
  or prints garbage produces no decision, and the run continues.

The last one is the reason hooks are safe to add at all: a hook is an addition to
the runtime, so it must not be able to remove the runtime's ability to work.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from arm.hooks.bridge import HookBridge, pipeline_hook
from arm.hooks.protocol import (
    CLAUDE_POINTS,
    DECISION_ALLOW,
    DECISION_ASK,
    DECISION_DENY,
    DIALECT_CLAUDE,
    DIALECT_CODEX,
    POINT_AGENT_CREATED,
    POINT_POST_TOOL,
    POINT_PRE_STEP,
    POINT_PRE_TOOL,
    POINT_TURN_STOPPING,
    Decision,
    HookError,
    HookHandler,
    Matcher,
    compile_pattern,
    load_config,
    matcher_diagnostic,
    merge_decisions,
    parse_config,
)
from arm.hooks.runner import HookRunner, run_hook


def run(coro):
    return asyncio.run(coro)


def script(tmp_path: Path, body: str, name: str = "hook.py") -> tuple[str, list[str]]:
    """Write a hook script and return the command to run it."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return sys.executable, [str(path)]


# ===========================================================================
# Matching
# ===========================================================================


def test_an_empty_matcher_matches_everything():
    """Both dialects agree here."""
    assert Matcher("").matches("anything")
    assert Matcher("*").matches("anything")


def test_a_literal_alternation_matches_exactly():
    """Claude Code's dialect: a bare alphanumeric pattern is an exact choice."""
    matcher = Matcher("Read|Write", DIALECT_CLAUDE)
    assert matcher.matches("Read")
    assert matcher.matches("Write")
    assert not matcher.matches("ReadFile"), (
        "a literal alternation must not match a prefix"
    )


def test_codex_treats_every_pattern_as_an_unanchored_regex():
    """The dialects genuinely differ, and this is the difference.

    Unifying them would silently change which hooks fire for one of the two, and
    a hook that stops firing is nearly impossible to notice.
    """
    claude = Matcher("Read", DIALECT_CLAUDE)
    codex = Matcher("Read", DIALECT_CODEX)

    assert claude.matches("Read")
    assert not claude.matches("ReadFile")
    assert codex.matches("Read")
    assert codex.matches("ReadFile"), "Codex patterns are unanchored regexes"


def test_a_regex_pattern_matches_as_a_regex():
    matcher = Matcher("^mcp__.*", DIALECT_CLAUDE)
    assert matcher.matches("mcp__server__tool")
    assert not matcher.matches("bash")


def test_an_invalid_regex_does_not_match_rather_than_raising():
    """A broken pattern must not take down the turn."""
    matcher = Matcher("[unclosed", DIALECT_CLAUDE)
    assert not matcher.matches("anything")


def test_an_invalid_regex_is_reported_at_parse_time():
    """A pattern that silently never fires is the worst configuration bug."""
    problem = matcher_diagnostic("[unclosed", DIALECT_CLAUDE)
    assert problem is not None
    assert "not a valid regular expression" in problem


def test_a_valid_pattern_has_no_diagnostic():
    assert matcher_diagnostic("Read|Write", DIALECT_CLAUDE) is None
    assert matcher_diagnostic("*", DIALECT_CLAUDE) is None
    assert matcher_diagnostic("^mcp__.*", DIALECT_CODEX) is None


def test_compiling_a_pattern_is_cached():
    """A matcher is consulted on every tool call."""
    first = compile_pattern("^cached$")
    second = compile_pattern("^cached$")
    assert first is second


# ===========================================================================
# Merging
# ===========================================================================


def test_an_empty_merge_allows():
    assert merge_decisions([]).kind == DECISION_ALLOW


def test_deny_wins():
    assert merge_decisions(
        [Decision(kind=DECISION_ALLOW), Decision(kind=DECISION_DENY)]
    ).kind == DECISION_DENY


def test_ask_beats_allow():
    assert merge_decisions(
        [Decision(kind=DECISION_ALLOW), Decision(kind=DECISION_ASK)]
    ).kind == DECISION_ASK


def test_deny_beats_ask():
    assert merge_decisions(
        [Decision(kind=DECISION_ASK), Decision(kind=DECISION_DENY)]
    ).kind == DECISION_DENY


@pytest.mark.parametrize(
    "order",
    [
        (DECISION_ALLOW, DECISION_ASK, DECISION_DENY),
        (DECISION_DENY, DECISION_ALLOW, DECISION_ASK),
        (DECISION_ASK, DECISION_DENY, DECISION_ALLOW),
        (DECISION_DENY, DECISION_ASK, DECISION_ALLOW),
    ],
)
def test_the_merge_does_not_depend_on_order(order):
    """Merging by configuration order would let adding a hook loosen a rule.

    Someone adding a safety hook expects the result to be at least as strict, not
    to depend on where in a list they put it.
    """
    decisions = [Decision(kind=kind, reason=f"from-{kind}") for kind in order]
    assert merge_decisions(decisions).kind == DECISION_DENY


def test_the_winning_reason_is_kept():
    """An explanation for a restriction that did not apply is confusing."""
    merged = merge_decisions(
        [
            Decision(kind=DECISION_ALLOW, reason="allowed by the first hook"),
            Decision(kind=DECISION_DENY, reason="denied by the second hook"),
        ]
    )
    assert merged.reason == "denied by the second hook"


def test_context_accumulates_from_every_hook():
    """A hook overruled on the decision may still have useful text.

    Discarding it would lose information for no benefit.
    """
    merged = merge_decisions(
        [
            Decision(kind=DECISION_ALLOW, context=("from the permissive hook",)),
            Decision(kind=DECISION_DENY, context=("from the strict hook",)),
        ]
    )
    assert merged.context == ("from the permissive hook", "from the strict hook")


def test_stop_is_sticky():
    """Once a hook asks to stop, a later hook cannot un-ask it."""
    merged = merge_decisions(
        [Decision(kind=DECISION_ALLOW, stop=True), Decision(kind=DECISION_DENY)]
    )
    assert merged.stop


# ===========================================================================
# Configuration parsing
# ===========================================================================


def test_a_claude_config_parses():
    raw = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "bash",
                    "hooks": [{"type": "command", "command": "/bin/true"}],
                }
            ]
        }
    }
    handlers, diagnostics = parse_config(raw, dialect=DIALECT_CLAUDE)

    assert len(handlers) == 1
    assert handlers[0].point == POINT_PRE_TOOL
    assert handlers[0].command == "/bin/true"
    assert diagnostics == []


def test_every_claude_point_maps_to_a_runtime_point():
    """The bridge's whole job is this translation."""
    for name, point in CLAUDE_POINTS.items():
        raw = {
            "hooks": {
                name: [{"hooks": [{"type": "command", "command": "/bin/true"}]}]
            }
        }
        handlers, _diagnostics = parse_config(raw, dialect=DIALECT_CLAUDE)
        assert handlers and handlers[0].point == point, name


def test_an_unknown_point_is_reported():
    """A hook that can never fire must be visible, not silently dropped."""
    raw = {"hooks": {"NotARealPoint": [{"hooks": [{"command": "/bin/true"}]}]}}
    handlers, diagnostics = parse_config(raw)

    assert handlers == []
    assert any("never fire" in d for d in diagnostics)


def test_a_non_command_hook_type_is_skipped_with_a_note():
    """Refusing to start because a config uses an unimplemented feature is hostile."""
    raw = {
        "hooks": {
            "PreToolUse": [
                {"hooks": [{"type": "prompt", "prompt": "is this ok?"}]}
            ]
        }
    }
    handlers, diagnostics = parse_config(raw)

    assert handlers == []
    assert any("prompt" in d and "not supported" in d for d in diagnostics)


def test_a_command_hook_without_a_command_is_reported():
    raw = {"hooks": {"PreToolUse": [{"hooks": [{"type": "command"}]}]}}
    handlers, diagnostics = parse_config(raw)

    assert handlers == []
    assert any("needs a 'command'" in d for d in diagnostics)


def test_a_malformed_entry_does_not_disable_the_others():
    """One bad hook must not disable every other hook."""
    raw = {
        "hooks": {
            "PreToolUse": [
                {"hooks": [{"type": "command", "command": "/bin/true"}]},
                "not an object",
                {"hooks": [{"type": "command", "command": "/bin/false"}]},
            ]
        }
    }
    handlers, diagnostics = parse_config(raw)

    assert len(handlers) == 2, "a malformed entry removed the good ones"
    assert diagnostics


def test_a_config_without_a_hooks_key_is_reported():
    handlers, diagnostics = parse_config({})
    assert handlers == []
    assert any("no 'hooks' key" in d for d in diagnostics)


def test_an_unknown_dialect_is_refused():
    with pytest.raises(HookError, match="unknown hook dialect"):
        parse_config({"hooks": {}}, dialect="something-else")


def test_a_missing_file_yields_no_hooks_and_no_error(tmp_path: Path):
    """A configuration that is not there is not a problem."""
    handlers, diagnostics = load_config(tmp_path / "nope.json")
    assert handlers == []
    assert diagnostics == []


def test_a_malformed_file_is_reported_not_fatal(tmp_path: Path):
    """A typo must not look like a broken installation."""
    path = tmp_path / "hooks.json"
    path.write_text("{ not json")
    handlers, diagnostics = load_config(path)

    assert handlers == []
    assert diagnostics and "could not read" in diagnostics[0]


def test_a_valid_file_loads(tmp_path: Path):
    path = tmp_path / "hooks.json"
    path.write_text(
        json.dumps(
            {"hooks": {"PreToolUse": [{"hooks": [{"command": "/bin/true"}]}]}}
        )
    )
    handlers, diagnostics = load_config(path)

    assert len(handlers) == 1
    assert diagnostics == []


def test_an_unknown_point_in_a_handler_is_refused():
    """A handler constructed directly with a bad point is a programming error."""
    with pytest.raises(HookError, match="unknown hook point"):
        HookHandler(point="not/a/point", command="/bin/true")


def test_a_non_positive_timeout_is_refused():
    with pytest.raises(HookError, match="timeout"):
        HookHandler(point=POINT_PRE_TOOL, command="/bin/true", timeout_ms=0)


# ===========================================================================
# Running hooks
# ===========================================================================


def test_a_hook_can_allow(tmp_path: Path):
    command, args = script(tmp_path, "print('{}')\n")
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {"tool": "bash"}))

    assert outcome.decision.kind == DECISION_ALLOW
    assert outcome.exit_code == 0


def test_a_hook_can_deny_via_json(tmp_path: Path):
    command, args = script(
        tmp_path,
        """
        import json
        print(json.dumps({"decision": "deny", "reason": "not on my watch"}))
        """,
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {"tool": "bash"}))

    assert outcome.decision.kind == DECISION_DENY
    assert outcome.decision.reason == "not on my watch"


def test_a_hook_can_deny_via_exit_code(tmp_path: Path):
    """The older convention, and the one a shell script naturally produces."""
    command, args = script(
        tmp_path,
        """
        import sys
        sys.stderr.write("this is not allowed")
        sys.exit(1)
        """,
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {"tool": "bash"}))

    assert outcome.decision.kind == DECISION_DENY
    assert "not allowed" in outcome.decision.reason


def test_a_hook_can_ask(tmp_path: Path):
    command, args = script(
        tmp_path, 'import json\nprint(json.dumps({"decision": "ask"}))\n'
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    assert run(run_hook(handler, {})).decision.kind == DECISION_ASK


def test_a_hook_receives_the_payload_on_stdin(tmp_path: Path):
    """The payload is the complete description of the event."""
    command, args = script(
        tmp_path,
        """
        import json, sys
        payload = json.load(sys.stdin)
        print(json.dumps({"decision": "deny", "reason": payload["tool"]}))
        """,
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {"tool": "the-tool-name"}))
    assert outcome.decision.reason == "the-tool-name"


def test_the_point_is_passed_as_environment(tmp_path: Path):
    """A shell script is more likely to branch on an env var than parse stdin."""
    command, args = script(
        tmp_path,
        """
        import json, os
        print(json.dumps({"decision": "deny",
                          "reason": os.environ.get("ARM_HOOK_POINT", "none")}))
        """,
    )
    handler = HookHandler(point=POINT_PRE_STEP, command=command, args=tuple(args))

    assert run(run_hook(handler, {})).decision.reason == POINT_PRE_STEP


def test_a_hook_can_add_context(tmp_path: Path):
    command, args = script(
        tmp_path,
        'import json\nprint(json.dumps({"additionalContext": "remember the style guide"}))\n',
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {}))
    assert outcome.decision.context == ("remember the style guide",)


def test_a_hook_can_ask_the_turn_to_continue(tmp_path: Path):
    """The dialects' "stop" hook, expressed as ``continue: false``."""
    command, args = script(
        tmp_path, 'import json\nprint(json.dumps({"continue": False}))\n'
    )
    handler = HookHandler(point=POINT_TURN_STOPPING, command=command, args=tuple(args))

    assert run(run_hook(handler, {})).decision.stop


def test_an_unknown_decision_is_reported_not_silently_allowed(tmp_path: Path):
    """The hook tried to say something and we did not understand it."""
    command, args = script(
        tmp_path, 'import json\nprint(json.dumps({"decision": "maybe"}))\n'
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {}))
    assert outcome.decision.kind == DECISION_ALLOW
    assert "unrecognised decision" in outcome.decision.reason


def test_a_nonzero_exit_overrides_a_json_allow(tmp_path: Path):
    """The process failing is a stronger signal than its last words."""
    command, args = script(
        tmp_path,
        """
        import json, sys
        print(json.dumps({"decision": "allow"}))
        sys.stderr.write("but I crashed")
        sys.exit(2)
        """,
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {}))
    assert outcome.decision.kind == DECISION_DENY


# ===========================================================================
# A broken hook cannot break the agent
# ===========================================================================


def test_a_missing_command_allows():
    """A missing formatter binary must not block every tool call."""
    handler = HookHandler(
        point=POINT_PRE_TOOL, command="/nonexistent/definitely-not-here"
    )
    outcome = run(run_hook(handler, {}))

    assert outcome.decision.kind == DECISION_ALLOW
    assert outcome.failure
    assert "was not found" in outcome.failure


def test_a_timeout_allows_and_is_recorded(tmp_path: Path):
    """A wedged hook must not hang the turn."""
    command, args = script(tmp_path, "import time\ntime.sleep(30)\n")
    handler = HookHandler(
        point=POINT_PRE_TOOL, command=command, args=tuple(args), timeout_ms=200
    )

    outcome = run(run_hook(handler, {}))

    assert outcome.decision.kind == DECISION_ALLOW
    assert "timed out" in outcome.failure


def test_a_hook_that_prints_garbage_does_not_crash(tmp_path: Path):
    """Unparseable output falls back to the exit-code convention."""
    command, args = script(tmp_path, "print('this is not json at all')\n")
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {}))
    assert outcome.decision.kind == DECISION_ALLOW


def test_a_hook_that_exits_nonzero_with_garbage_still_denies(tmp_path: Path):
    command, args = script(
        tmp_path, "import sys\nprint('not json')\nsys.exit(3)\n"
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {}))
    assert outcome.decision.kind == DECISION_DENY


def test_the_stderr_summary_is_bounded(tmp_path: Path):
    """A hook that crashes in a loop produces unbounded output."""
    command, args = script(
        tmp_path, "import sys\nsys.stderr.write('x' * 5000)\nsys.exit(1)\n"
    )
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    outcome = run(run_hook(handler, {}))
    assert len(outcome.stderr_summary) <= 500


def test_a_hook_can_print_and_exit_zero(tmp_path: Path):
    """A hook that prints a human-readable line and succeeds is saying allow."""
    command, args = script(tmp_path, "print('all good')\n")
    handler = HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))

    assert run(run_hook(handler, {})).decision.kind == DECISION_ALLOW


# ===========================================================================
# The runner
# ===========================================================================


def test_only_matching_handlers_fire(tmp_path: Path):
    command, args = script(tmp_path, 'import json\nprint(json.dumps({"decision": "deny"}))\n')
    bash_hook = HookHandler(
        point=POINT_PRE_TOOL, command=command, args=tuple(args),
        matcher=Matcher("bash"), handler_id="h1",
    )
    runner = HookRunner([bash_hook])

    assert len(runner.for_point(POINT_PRE_TOOL, "bash")) == 1
    assert len(runner.for_point(POINT_PRE_TOOL, "read")) == 0
    assert len(runner.for_point(POINT_PRE_STEP, "bash")) == 0


def test_no_matching_handlers_allows_without_running_anything():
    runner = HookRunner([])
    decision = run(runner.fire(POINT_PRE_TOOL, "bash", {}))

    assert decision.kind == DECISION_ALLOW
    assert runner.outcomes == []


def test_several_hooks_merge_most_restrictively(tmp_path: Path):
    """Two hooks, one allowing and one denying: the result is a deny."""
    allow_cmd, allow_args = script(
        tmp_path, 'import json\nprint(json.dumps({"decision": "allow"}))\n',
        name="allow.py",
    )
    deny_cmd, deny_args = script(
        tmp_path, 'import json\nprint(json.dumps({"decision": "deny", "reason": "no"}))\n',
        name="deny.py",
    )
    runner = HookRunner(
        [
            HookHandler(point=POINT_PRE_TOOL, command=allow_cmd, args=tuple(allow_args), handler_id="a"),
            HookHandler(point=POINT_PRE_TOOL, command=deny_cmd, args=tuple(deny_args), handler_id="b"),
        ]
    )

    decision = run(runner.fire(POINT_PRE_TOOL, "bash", {}))
    assert decision.kind == DECISION_DENY


def test_a_failing_hook_does_not_stop_the_others(tmp_path: Path):
    """One broken hook must not disable the rest."""
    good_cmd, good_args = script(
        tmp_path, 'import json\nprint(json.dumps({"decision": "deny", "reason": "yes"}))\n',
        name="good.py",
    )
    runner = HookRunner(
        [
            HookHandler(point=POINT_PRE_TOOL, command="/nonexistent", handler_id="bad"),
            HookHandler(point=POINT_PRE_TOOL, command=good_cmd, args=tuple(good_args), handler_id="good"),
        ]
    )

    decision = run(runner.fire(POINT_PRE_TOOL, "bash", {}))
    assert decision.kind == DECISION_DENY, "the working hook's decision was lost"


def test_every_outcome_is_recorded(tmp_path: Path):
    """Observability: the runtime can log what each hook decided."""
    command, args = script(tmp_path, "print('{}')\n")
    runner = HookRunner(
        [HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args), handler_id="h1")]
    )
    run(runner.fire(POINT_PRE_TOOL, "bash", {}))

    assert len(runner.outcomes) == 1
    assert runner.outcomes[0].handler_id == "h1"
    assert runner.outcomes[0].exit_code == 0


def test_the_handler_id_is_stable_and_unique():
    """It pairs the invoked and result log records."""
    raw = {
        "hooks": {
            "PreToolUse": [
                {"hooks": [{"type": "command", "command": "/bin/true"}]},
                {"hooks": [{"type": "command", "command": "/bin/false"}]},
            ]
        }
    }
    handlers, _diagnostics = parse_config(raw)
    ids = [h.handler_id for h in handlers]

    assert len(set(ids)) == 2, "handler ids are not unique"
    assert all(h.handler_id.startswith("claude-code:PreToolUse:") for h in handlers)


# ===========================================================================
# The bridge
# ===========================================================================


def test_the_bridge_translates_a_deny_into_a_pipeline_deny(tmp_path: Path):
    """This is what makes a hook able to prevent something."""
    command, args = script(
        tmp_path, 'import json\nprint(json.dumps({"decision": "deny", "reason": "no"}))\n'
    )
    bridge = HookBridge(
        HookRunner([HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))])
    )

    decision = run(bridge.pre_tool("bash", {"command": "rm -rf /"}))
    assert decision.kind == DECISION_DENY
    assert decision.reason == "no"


def test_the_bridge_translates_an_ask(tmp_path: Path):
    command, args = script(
        tmp_path, 'import json\nprint(json.dumps({"decision": "ask", "reason": "confirm"}))\n'
    )
    bridge = HookBridge(
        HookRunner([HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))])
    )

    assert run(bridge.pre_tool("bash", {})).kind == DECISION_ASK


def test_a_disabled_bridge_short_circuits():
    """A disabled bridge must cost nothing, not run an empty runner."""
    runner = HookRunner([HookHandler(point=POINT_PRE_TOOL, command="/nonexistent")])
    bridge = HookBridge(runner, enabled=False)

    assert run(bridge.pre_tool("bash", {})).kind == DECISION_ALLOW
    assert runner.outcomes == []


def test_the_pipeline_hook_denies_without_calling_next(tmp_path: Path):
    """Short-circuiting is what a waterfall means."""
    command, args = script(
        tmp_path, 'import json\nprint(json.dumps({"decision": "deny", "reason": "blocked"}))\n'
    )
    bridge = HookBridge(
        HookRunner([HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))])
    )
    hook = pipeline_hook(bridge)

    called: list[bool] = []

    async def next_():
        called.append(True)
        return "proceeded"

    class Call:
        name = "bash"
        arguments = {"command": "rm -rf /"}

    result = run(hook(Call(), next_))

    assert result.kind == "deny"
    assert called == [], "next() was called despite a deny"


def test_the_pipeline_hook_continues_on_allow(tmp_path: Path):
    command, args = script(tmp_path, "print('{}')\n")
    bridge = HookBridge(
        HookRunner([HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))])
    )
    hook = pipeline_hook(bridge)

    class Call:
        name = "read"
        arguments = {}

    assert run(hook(Call(), lambda: _return("proceeded"))) == "proceeded"


async def _return(value):
    return value


def test_unsupported_fields_are_reported_not_ignored(tmp_path: Path):
    """A configuration relying on ``updatedInput`` believes it rewrites arguments.

    Applying it would let a hook change what a tool does after the arguments were
    frozen, which breaks the pipeline's central guarantee. A silent no-op would
    leave the configuration believing it works forever.
    """
    bridge = HookBridge(HookRunner([]))
    notes = bridge.unsupported_fields({"updatedInput": {"command": "ls"}})

    assert notes
    assert "frozen" in notes[0]


def test_the_bridge_reports_the_post_tool_result(tmp_path: Path):
    """A post-tool hook sees what the tool produced."""
    command, args = script(
        tmp_path,
        """
        import json, sys
        payload = json.load(sys.stdin)
        ok = "PASS" in payload["result"]
        print(json.dumps({"decision": "allow" if ok else "deny",
                          "feedback": "" if ok else "the tests failed"}))
        """,
    )
    bridge = HookBridge(
        HookRunner([HookHandler(point="tools/post-execute", command=command, args=tuple(args))])
    )

    assert run(bridge.post_tool("bash", {}, "all PASS", is_error=False)).kind == DECISION_ALLOW
    blocked = run(bridge.post_tool("bash", {}, "1 failed", is_error=False))
    assert blocked.kind == DECISION_DENY
    assert blocked.feedback == "the tests failed"


# ===========================================================================
# End to end: a hook that blocks a real tool call
# ===========================================================================


def test_a_hook_blocks_a_real_tool_call_through_the_pipeline(tmp_path: Path):
    """The whole path: configuration, subprocess, decision, pipeline, model.

    This is the test that proves hooks are wired rather than merely implemented.
    Everything above tests a piece; this one tests that the pieces are connected,
    which is exactly the class of defect that has appeared repeatedly in this
    project -- correct code that nothing calls.
    """
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response, tool_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.builtin import builtin_tools
    from arm.tools.registry import ToolRegistry

    command, args = script(
        tmp_path,
        """
        import json, sys
        payload = json.load(sys.stdin)
        if payload.get("tool") == "bash":
            print(json.dumps({"decision": "deny",
                              "reason": "bash is blocked by a hook"}))
        else:
            print(json.dumps({"decision": "allow"}))
        """,
    )
    bridge = HookBridge(
        HookRunner([HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))])
    )

    registry = ToolRegistry()
    registry.register_many(builtin_tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)

    from arm.llm.mock import text_response, tool_response

    session = Session.create("hooked")
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
        hook_bridge=bridge,
        model="mock",
    )

    loop.submit("run echo hi")
    asyncio.run(loop.run_until_idle())

    results = [e for e in session if e.type == "tool/result"]
    assert results, "the tool never ran or was never recorded"
    assert results[0].data["is_error"] is True, (
        "the hook denied the call but the tool ran anyway"
    )
    text = results[0].data["message"]["content"][0]["text"]
    assert "blocked by a hook" in text, f"the reason did not reach the model: {text}"


def test_a_hook_that_allows_lets_the_tool_run(tmp_path: Path):
    """The complement: a hook must not block everything."""
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response, tool_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.builtin import builtin_tools
    from arm.tools.registry import ToolRegistry

    command, args = script(tmp_path, "print('{}')\n")
    bridge = HookBridge(
        HookRunner([HookHandler(point=POINT_PRE_TOOL, command=command, args=tuple(args))])
    )

    registry = ToolRegistry()
    registry.register_many(builtin_tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)

    session = Session.create("allowed")
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
        hook_bridge=bridge,
        model="mock",
    )

    loop.submit("run echo hi")
    asyncio.run(loop.run_until_idle())

    results = [e for e in session if e.type == "tool/result"]
    assert results
    assert results[0].data["is_error"] is False
    assert "hi" in results[0].data["message"]["content"][0]["text"]


def test_a_missing_hook_command_does_not_break_the_turn(tmp_path: Path):
    """The central safety property: a broken hook cannot break the agent."""
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response, tool_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.builtin import builtin_tools
    from arm.tools.registry import ToolRegistry

    bridge = HookBridge(
        HookRunner(
            [
                HookHandler(
                    point=POINT_PRE_TOOL, command="/nonexistent/definitely-not-a-command"
                )
            ]
        )
    )

    registry = ToolRegistry()
    registry.register_many(builtin_tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)

    session = Session.create("broken-hook")
    loop = Runtime(
        session,
        MockAdapter(
            [
                tool_response(("c1", "bash", {"command": "echo still works"})),
                text_response("ok"),
            ]
        ),
        registry,
        assembler=assembler,
        hook_bridge=bridge,
        model="mock",
    )

    loop.submit("run it")
    outcomes = asyncio.run(loop.run_until_idle())

    assert outcomes[0].reason == "completed"
    results = [e for e in session if e.type == "tool/result"]
    assert results and results[0].data["is_error"] is False, (
        "a missing hook command blocked the tool"
    )


# ===========================================================================
# The turn-stopping point must actually be consulted
# ===========================================================================


def _stopping_bridge(tmp_path: Path, *, keep_going: bool):
    """A bridge whose Stop hook either insists on another step or agrees."""
    command, args = script(
        tmp_path,
        f'import json, sys\njson.load(sys.stdin)\n'
        f'print(json.dumps({{"continue": {not keep_going}}}))\n',
        name=f"stop_{keep_going}.py",
    )
    return HookBridge(
        HookRunner(
            [HookHandler(point=POINT_TURN_STOPPING, command=command, args=tuple(args))]
        )
    )


def _run_turn(bridge, *, max_steps: int = 3):
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.registry import ToolRegistry

    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    session = Session.create("stop")
    loop = Runtime(
        session,
        MockAdapter([text_response("done")] * 10),
        ToolRegistry(),
        assembler=assembler,
        hook_bridge=bridge,
        model="mock",
        max_steps=max_steps,
    )
    loop.submit("x")
    return asyncio.run(loop.run_until_idle())


def test_a_turn_stopping_hook_can_ask_for_another_step(tmp_path: Path):
    """The `Stop` point in both hook dialects.

    A hook that decides the task is not actually done must be able to say so.
    Without this the point existed in the vocabulary and nothing consulted it, so
    a guard the user installed silently did nothing.
    """
    outcomes = _run_turn(_stopping_bridge(tmp_path, keep_going=True))
    assert outcomes[0].steps > 1, (
        "the hook asked for another step and the turn ended anyway"
    )


def test_a_turn_stopping_hook_that_agrees_lets_the_turn_end(tmp_path: Path):
    """The complement: the hook must not block every turn."""
    outcomes = _run_turn(_stopping_bridge(tmp_path, keep_going=False))
    assert outcomes[0].steps == 1
    assert outcomes[0].reason == "completed"


def test_an_always_disagreeing_hook_is_bounded_by_max_steps(tmp_path: Path):
    """A hook that never agrees must not loop forever.

    The bound is the loop's own ``max_steps``, so a broken hook degrades into a
    long turn rather than a hang.
    """
    outcomes = _run_turn(_stopping_bridge(tmp_path, keep_going=True), max_steps=3)
    assert outcomes[0].steps <= 3, (
        f"the turn ran {outcomes[0].steps} steps despite max_steps=3"
    )


def test_a_session_with_no_hooks_pays_nothing():
    """The check must short-circuit, or every turn would pay for a feature it
    does not use."""
    outcomes = _run_turn(None)
    assert outcomes[0].steps == 1


def test_a_failing_stop_hook_does_not_break_the_turn():
    """A hook fault is not a turn fault, and the turn should still end."""
    bridge = HookBridge(
        HookRunner(
            [
                HookHandler(
                    point=POINT_TURN_STOPPING, command="/nonexistent/not-a-command"
                )
            ]
        )
    )
    outcomes = _run_turn(bridge)
    assert outcomes[0].reason == "completed"
    assert outcomes[0].steps == 1


# ===========================================================================
# SessionStart must actually fire
# ===========================================================================


def test_the_bridge_fires_session_start_hooks(tmp_path: Path):
    """`SessionStart` is the hook point people reach for first.

    It was mapped, loaded and never consulted, so installing one produced silence
    -- indistinguishable from a hook that ran and decided nothing.
    """
    marker = tmp_path / "ran.txt"
    command, args = script(
        tmp_path,
        f"""
        import json, pathlib, sys
        json.load(sys.stdin)
        pathlib.Path({str(marker)!r}).write_text("fired")
        print(json.dumps({{"additionalContext": "remember the style guide"}}))
        """,
    )
    bridge = HookBridge(
        HookRunner(
            [
                HookHandler(
                    point=POINT_AGENT_CREATED, command=command, args=tuple(args)
                )
            ]
        )
    )

    context = asyncio.run(bridge.agent_created("sess-1"))

    assert marker.exists(), "the SessionStart hook did not run"
    assert context == ("remember the style guide",), context


def test_session_start_context_reaches_the_session(tmp_path: Path, monkeypatch):
    """End to end: a configured hook's context lands in the log."""
    import json as jsonlib

    from arm.cli.main import build_runtime

    hook = tmp_path / "start.py"
    hook.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'additionalContext': 'from the session-start hook'}))\n"
    )
    (tmp_path / "hooks.json").write_text(
        jsonlib.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": sys.executable,
                                    "args": [str(hook)],
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )

    # Imported by path: `tests/` is not a package on sys.path in every runner,
    # and this helper is the project's one real HTTP model server.
    sys.path.insert(0, str(Path(__file__).parent))
    from test_end_to_end import FakeServer

    with FakeServer([{"text": "ok"}]) as server:
        runtime, session, handle, _scheduler, _mcp = build_runtime(
            session_id="hooked",
            model="m",
            api_key="k",
            base_url=server.base_url,
            home=str(tmp_path),
        )
        assert handle is not None
        handle.close()

    notes = [
        e for e in session if e.data.get("source") == "hook:session-start"
    ]
    assert notes, "the hook's context never reached the session"
    text = notes[0].data["message"]["content"][0]["text"]
    assert "from the session-start hook" in text


def test_session_start_context_is_not_user_input(tmp_path: Path):
    """It is context, not a message: the inbox must not claim it.

    A claimed message would make the first turn run twice -- once for the user's
    question and once for the hook's note.
    """
    import json as jsonlib

    from arm.cli.main import build_runtime

    hook = tmp_path / "start.py"
    hook.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'additionalContext': 'a note'}))\n"
    )
    (tmp_path / "hooks.json").write_text(
        jsonlib.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": sys.executable, "args": [str(hook)]}]}
                    ]
                }
            }
        )
    )

    # Imported by path: `tests/` is not a package on sys.path in every runner,
    # and this helper is the project's one real HTTP model server.
    sys.path.insert(0, str(Path(__file__).parent))
    from test_end_to_end import FakeServer

    with FakeServer([{"text": "ok"}]) as server:
        runtime, session, handle, _scheduler, _mcp = build_runtime(
            session_id="hooked",
            model="m",
            api_key="k",
            base_url=server.base_url,
            home=str(tmp_path),
        )
        assert handle is not None
        handle.close()

    for event in session:
        if event.data.get("source") == "hook:session-start":
            assert event.data.get("context_only") is True, (
                "the hook's context would be claimed as user input"
            )


def test_a_failing_session_start_hook_does_not_prevent_startup(tmp_path: Path):
    """A hook fault is not a startup fault."""
    bridge = HookBridge(
        HookRunner(
            [
                HookHandler(
                    point=POINT_AGENT_CREATED, command="/nonexistent/not-a-command"
                )
            ]
        )
    )
    assert asyncio.run(bridge.agent_created("sess")) == ()


# ===========================================================================
# Every hook point must actually fire
# ===========================================================================


def _marker_hook(tmp_path: Path, point: str, *, decision: str = "allow"):
    """A hook that records that it ran, and optionally decides."""
    marker = tmp_path / f"{point.replace('/', '_')}.txt"
    command, args = script(
        tmp_path,
        f"""
        import json, pathlib, sys
        json.load(sys.stdin)
        pathlib.Path({str(marker)!r}).write_text("fired")
        print(json.dumps({{"decision": {decision!r}}}))
        """,
        name=f"hook_{point.replace('/', '_')}.py",
    )
    return marker, HookBridge(
        HookRunner(
            [HookHandler(point=point, command=command, args=tuple(args))]
        )
    )


def _run_with_bridge(bridge, script_, *, tmp_path: Path):
    """Run one turn through a real loop with the given bridge."""
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter
    from arm.prompt.assembler import PromptAssembler
    from arm.session.session import Session
    from arm.tools.builtin import builtin_tools
    from arm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many(builtin_tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    session = Session.create("hooks")
    loop = Runtime(
        session,
        MockAdapter(script_),
        registry,
        assembler=assembler,
        hook_bridge=bridge,
        model="mock",
    )
    loop.submit("go")
    outcomes = asyncio.run(loop.run_until_idle())
    return session, outcomes


def test_user_prompt_submit_fires(tmp_path: Path):
    """`UserPromptSubmit` was mapped, loaded, and never called.

    A hook that refuses work silently permitted it -- the failure mode is
    indistinguishable from a hook that ran and approved.
    """
    from arm.llm.mock import text_response

    marker, bridge = _marker_hook(tmp_path, POINT_PRE_STEP)
    _run_with_bridge(bridge, [text_response("ok")], tmp_path=tmp_path)

    assert marker.exists(), "UserPromptSubmit never fired"


def test_user_prompt_submit_can_reject_the_message(tmp_path: Path):
    """A denial must short-circuit before the model is called."""
    from arm.llm.mock import text_response

    marker, bridge = _marker_hook(tmp_path, POINT_PRE_STEP, decision="deny")
    session, outcomes = _run_with_bridge(
        bridge, [text_response("should not happen")], tmp_path=tmp_path
    )

    assert marker.exists()
    assert outcomes[0].reason == "blocked", (
        f"a rejected message still ran the turn: {outcomes[0].reason}"
    )
    # No step means no model call, so no assistant message exists.
    assert not [e for e in session if e.type == "assistant/message"], (
        "the model was called despite the rejection"
    )


def test_post_tool_use_fires(tmp_path: Path):
    """`PostToolUse` was mapped, loaded, and never called either."""
    from arm.llm.mock import text_response, tool_response

    marker, bridge = _marker_hook(tmp_path, POINT_POST_TOOL)
    _run_with_bridge(
        bridge,
        [tool_response(("c1", "read", {"path": "pyproject.toml"})), text_response("ok")],
        tmp_path=tmp_path,
    )

    assert marker.exists(), "PostToolUse never fired"


def test_post_tool_use_can_block_a_result(tmp_path: Path):
    """The tool already ran, so a denial replaces the result rather than stopping it."""
    from arm.llm.mock import text_response, tool_response

    _marker, bridge = _marker_hook(tmp_path, POINT_POST_TOOL, decision="deny")
    session, _outcomes = _run_with_bridge(
        bridge,
        [tool_response(("c1", "read", {"path": "pyproject.toml"})), text_response("ok")],
        tmp_path=tmp_path,
    )

    results = [e for e in session if e.type == "tool/result"]
    assert results, "no tool result was recorded"
    assert any(e.data.get("is_error") for e in results), (
        "the hook denied the result and it was accepted anyway"
    )


def test_every_hook_point_is_consulted_somewhere():
    """Every point must be referenced outside its own definition.

    Three of the five points were mapped and loaded but never called, so a user
    installing a hook got silence.

    **What this catches and what it does not**, stated because I checked: it
    catches a point that no code mentions at all, which is how a point gets
    forgotten while being added. It does **not** catch a point whose bridge
    method exists but is never called from the loop -- I verified that by
    removing the post-tool wiring, and this check still passed while the
    behavioural test above failed. The per-point behaviour tests are what cover
    that, and this is a cheap first line rather than the whole guard.
    """
    from pathlib import Path

    import arm.hooks.protocol as protocol

    # Each point has a module-level constant. The check looks for the *constant*
    # rather than the literal string, because that is how the code refers to a
    # point -- grepping for the string reports every point as unwired, which is a
    # false alarm rather than a finding.
    constants = {
        name: value
        for name, value in vars(protocol).items()
        if name.startswith("POINT_") and isinstance(value, str)
    }
    assert len(constants) == len(protocol.POINTS), (
        "a hook point has no POINT_ constant, so nothing can refer to it"
    )

    root = Path("src/arm")
    missing: list[str] = []
    for name, point in sorted(constants.items()):
        references = [
            path
            for path in root.rglob("*.py")
            if path.name != "protocol.py"
            and name in path.read_text(encoding="utf-8")
        ]
        if not references:
            missing.append(f"{point} ({name})")

    assert not missing, (
        "these hook points are defined but never consulted, so a hook installed "
        f"for them would silently do nothing: {missing}"
    )

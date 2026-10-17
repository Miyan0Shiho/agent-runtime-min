"""The command-line entry point.

A runtime that cannot be started is not a runtime. This module is deliberately
small: it wires the pieces together and gets out of the way. Anything that grows
here is a sign it belonged in a package.

Commands:

    arm run "task"      run one task to completion
    arm chat            interactive loop
    arm replay FILE     re-derive a stored session and print what the model saw
    arm sessions        list stored sessions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from ..agent.loop import LoopHooks, Runtime
from ..llm.openai_adapter import OpenAICompatibleAdapter
from ..approval.service import POLICY_ASK, ApprovalService, ask_in_terminal
from ..goal.store import GoalStore
from ..goal.tools import GoalTools
from ..plan.mode import PlanMode, plan_tools
from ..prompt.assembler import PromptAssembler
from ..sandbox.policy import resolve_policy
from ..session.events import USER_MESSAGE
from ..session.persistence import JsonlLog, read_log
from ..session.session import Session
from ..skills.registry import FilesystemSkillProvider, SkillRegistry
from ..subagent.contracts import DEFAULT_MAX_DEPTH
from ..subagent.inprocess import ChildFactory, InProcessProvider
from ..subagent.runtime import SubagentRuntime
from ..subagent.contracts import DelegationRequest
from ..subagent.tool import SubagentTool
from ..hooks.bridge import HookBridge
from ..hooks.protocol import DIALECT_CLAUDE, load_config as load_hook_config
from ..hooks.runner import HookRunner
from ..browser.cdp import DEFAULT_ENDPOINT as BROWSER_DEFAULT_ENDPOINT
from ..browser.cdp import Browser
from ..browser.tools import BrowserTools
from ..computer.desktop import Desktop
from ..computer.tools import ComputerTools
from ..mcp.client import McpManager, McpServerConfig
from ..plugins.registry import PluginRegistry, load_all as load_plugins
from ..plugins.registry import parse_plugin_config
from ..mcp.stdio import StdioServerConfig, connect_stdio
from ..schedule.runtime import ScheduleRuntime, ScheduleStore, ScheduleTimer
from ..schedule.tools import ScheduleTools
from ..team.journal import TeamJournal
from ..team.tools import TeamTools
from ..skills.tool import SkillTool
from ..todo.store import TodoStore
from ..todo.tools import TodoTools
from ..tools.builtin import builtin_tools
from ..tools.registry import ToolRegistry
from ..workspace.tools import WorktreeTools
from ..workspace.worktree import WorktreeManager

#: Where sessions live. Overridable so tests and the benchmark never touch a
#: user's real session store.
DEFAULT_HOME = os.environ.get("ARM_HOME") or str(Path.home() / ".arm")

TEAM_MEMBER_TIMEOUT = 600.0

SUBAGENT_PROMPT = """You are a subagent working on one delegated task.

You have no access to the conversation that delegated this task, and you cannot
ask questions. Work from the task description alone.

When you have finished, state your answer plainly. If the task cannot be
completed, say what blocked it rather than producing something that looks like a
result."""


SYSTEM_PROMPT = """You are arm, a coding agent running in a terminal.

You have tools for reading and writing files, running shell commands, and
searching the filesystem. Use them to inspect the real state of the world rather
than assuming what it contains.

Work in small steps. After each tool call, read the result before deciding what
to do next. When you have finished, state what you did and what you verified.

If a tool fails, the failure is information: read the error and adapt. Do not
retry the same call unchanged."""


def team_enabled() -> bool:
    """Whether the team tools are registered.

    On by default. ``ARM_TEAM=0`` disables them, which is the case a caller who
    wants a smaller tool surface actually asks for.
    """
    return os.environ.get("ARM_TEAM", "1") not in ("0", "", "false", "no")


def _load_mcp_servers(path: Path) -> list[McpServerConfig]:
    """Read MCP server definitions from a JSON file.

    A malformed file is reported and treated as empty rather than crashing: a
    broken configuration must not make the agent unusable, and the file is left
    alone so it can be inspected.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(raw, dict):
        print(f"warning: {path} must contain an object of server definitions")
        return []

    # Both shapes are accepted, and the wrapped one is the ecosystem standard.
    #
    #   {"mcpServers": {"adder": {...}}}   <- Claude Desktop, Cursor, VS Code
    #   {"adder": {...}}                   <- the bare form
    #
    # Found by writing a config the way every other MCP client documents it: the
    # loader read `mcpServers` as a *server named* "mcpServers", found no
    # `command` under it, and warned "'mcpServers' needs a 'command'". A config
    # that works in every other tool was silently ignored, and the warning
    # pointed at the wrong problem.
    #
    # Accepting both costs one unwrap. Requiring only the bare form would mean
    # every user's existing config has to be rewritten, and the error message
    # would not say so.
    if isinstance(raw.get("mcpServers"), dict):
        raw = raw["mcpServers"]

    servers: list[McpServerConfig] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict) or not entry.get("command"):
            print(f"warning: MCP server {name!r} needs a 'command'", file=sys.stderr)
            continue
        stdio = StdioServerConfig(
            command=str(entry["command"]),
            args=[str(a) for a in (entry.get("args") or ())],
            env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
            cwd=str(entry.get("cwd", "")),
        )
        servers.append(
            McpServerConfig(
                name=str(name),
                connect=lambda s=stdio: connect_stdio(s),
                required=bool(entry.get("required", False)),
                call_timeout=float(entry.get("call_timeout", 60.0)),
            )
        )
    return servers


def _load_schedule(path: Path) -> ScheduleStore:
    """Load the schedule from disk, with a sink that writes it back.

    A malformed schedule file is reported and started empty rather than crashing
    the runtime: a broken reminder store must not make the agent unusable, and
    the file is preserved so it can be inspected.
    """
    source: list[dict] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                source = [r for r in raw if isinstance(r, dict)]
            else:
                print(f"warning: {path} is not a list; starting with no reminders")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: could not read {path}: {exc}")

    def sink(records) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written atomically: a crash mid-write must not leave a truncated
        # schedule, which would lose every reminder in it.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(list(records), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)

    try:
        return ScheduleStore(sink=sink, source=source)
    except Exception as exc:  # noqa: BLE001 - a bad record must not be fatal
        print(f"warning: the schedule could not be loaded ({exc}); starting empty")
        return ScheduleStore(sink=sink)


def session_dir(home: str | None = None) -> Path:
    return Path(home or DEFAULT_HOME) / "sessions"


def session_path(session_id: str, home: str | None = None) -> Path:
    return session_dir(home) / f"{session_id}.jsonl"


def build_runtime(
    *,
    session_id: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    home: str | None = None,
    persist: bool = True,
    hooks: LoopHooks | None = None,
    hook_bridge: Any = None,
    quiet: bool = False,
    approval_policy: str | None = None,
    sandbox_preset: str | None = None,
) -> tuple[Runtime, Session, JsonlLog | None, ScheduleRuntime, McpManager]:
    """Wire a runtime from environment configuration.

    Persistence is attached as a session listener rather than driven by the loop:
    the loop should not know whether a session is being written to disk, and a
    listener that fails must not be able to unwind a committed event.
    """
    path = session_path(session_id, home)

    existing = None
    if persist and path.exists():
        result = read_log(path)
        if result.events:
            existing = result.events
            if result.truncate_to is not None:
                # A torn tail was found; the write path repairs it on next append.
                JsonlLog(path)._truncate(result.truncate_to)

    log_handle: JsonlLog | None = None
    if persist:
        log_handle = JsonlLog(path).open()

    if existing:
        session = Session.from_events(session_id, existing)
    else:
        session = Session.create(session_id)

    if persist and log_handle is not None and not existing:
        # The seed events were written before any listener existed, so they must
        # be flushed explicitly. Attaching the listener first and relying on it to
        # capture them is not possible: `Session.create` writes the seed during
        # construction, and a listener attached afterwards never sees them.
        #
        # Getting this wrong is not a cosmetic problem. A log whose first stored
        # record is seq 2 is not contiguous from 0, so it cannot be restored at
        # all -- the session would silently fail to resume.
        log_handle.append(session.events())

    if persist and log_handle is not None:
        # Write every subsequently committed event. Batching is not needed at
        # this scale, and an fsync per event keeps the durability story simple.
        #
        # The listener returns None deliberately: a listener's return value is
        # discarded by design, and returning the append count would imply
        # otherwise.
        def _persist(_session: Session, event: Any) -> None:
            assert log_handle is not None
            log_handle.append([event], already_contiguous=True)

        session.listen(_persist)

    registry = ToolRegistry()
    registry.register_many(builtin_tools())

    # Skills come from the standard roots plus any configured extras. A skill
    # root that does not exist contributes nothing, so this is safe on a machine
    # that has never installed one.
    skills = SkillRegistry()
    skills.register_provider(
        FilesystemSkillProvider(
            FilesystemSkillProvider.default_roots(
                cwd=Path.cwd(),
                home=Path.home(),
                extra=[Path(p) for p in (os.environ.get("ARM_SKILL_DIRS") or "").split(":") if p],
            )
        )
    )
    registry.register(SkillTool(skills).register())

    # Plan mode is always registered, whether or not it is active: the tool
    # catalogue must not change with the mode, so the tool is present and refuses
    # when there is nothing to exit.
    plan_mode = PlanMode(session)
    # Plan mode needs an approval channel, and the CLI passed none -- so
    # `exit_plan_mode` failed closed every time and plan mode was a trap: a mode
    # you could enter and never leave. Measured by asking a real model to do it;
    # it entered, planned, and reported that exiting was "an infrastructure
    # problem, not the plan's".
    #
    # Failing closed is the right default for a missing channel. What was missing
    # was the channel: under `ask` the person is present and can answer.
    def approve_plan(plan_text: str) -> bool:
        if policy != POLICY_ASK:
            return False
        print("\n--- proposed plan ---\n")
        print(plan_text)
        try:
            answer = input("\napprove this plan? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes"}

    registry.register(plan_tools(plan_mode, approve=approve_plan))
    # Plan mode had no way in: only the exit tool existed, and `PlanMode.set` was
    # called with False by it and nowhere else. A REQUIREMENTS item described a
    # mode the runtime could leave and never enter.
    from ..plan.mode import enter_tool as _enter_plan_tool

    registry.register(_enter_plan_tool(plan_mode))

    # Goals persist in the session log, so a restored session finds its goal.
    # Activation does not persist, which is why a resumed goal does not keep
    # running on its own until someone asks it to.
    goal_store = GoalStore(session)
    registry.register_many(GoalTools(goal_store).tools())

    # The todo list is the model's own scratchpad. It is recorded but never
    # injected into the prompt: re-feeding it would restate what the model just
    # wrote and present its draft as though it were an instruction.
    todo_store = TodoStore(session)
    registry.register_many(TodoTools(todo_store).tools())

    # Worktree tools are only registered inside a repository: offering them
    # elsewhere would advertise an operation that can only fail, and the model
    # would waste a step discovering that.
    worktree_manager = WorktreeManager(Path.cwd())
    if worktree_manager.is_repository():
        registry.register_many(WorktreeTools(worktree_manager).tools())

    assembler = PromptAssembler()
    assembler.set("core", SYSTEM_PROMPT, priority=10)

    if api_key is None:
        api_key = (
            os.environ.get("ARM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
    if base_url is None:
        base_url = os.environ.get("ARM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    adapter_kwargs: dict[str, Any] = {"api_key": api_key, "model": model}
    if base_url:
        adapter_kwargs["base_url"] = base_url
    adapter = OpenAICompatibleAdapter(**adapter_kwargs)

    # Approval policy: 'ask' by default, 'never' for unattended runs. Under
    # 'never' nothing is ever consulted, so an unattended session cannot block on
    # a prompt that nobody will answer -- and cannot accidentally be granted one
    # either.
    policy = approval_policy or os.environ.get("ARM_APPROVAL") or POLICY_ASK
    # Under 'ask' the user is actually asked. The previous code installed
    # deny_all here, so 'ask' meant 'always refuse' -- and because the service was
    # not wired into the pipeline, nothing surfaced the difference.
    #
    # Under 'never' there are no answerers at all: the policy rejects inside the
    # service, before any channel is consulted, so an unattended run cannot block
    # on a prompt nobody will answer.
    answerers = [ask_in_terminal()] if policy == POLICY_ASK else []
    approval = ApprovalService(policy=policy, answerers=answerers)

    # The sandbox decides what the session may touch. It is resolved once, so a
    # later change to the process's working directory cannot widen it.
    sandbox = resolve_policy(
        preset=sandbox_preset or os.environ.get("ARM_SANDBOX") or "default",
        workspace=os.getcwd(),
    )

    # The Docker sandbox replaces the `bash` body rather than wrapping the
    # pipeline. A wrapper would have to recognise the tool by name, and a name is
    # something a caller can change; swapping the body means there is no path in
    # which the host shell runs. See `sandbox/docker.py` for why the boundary has
    # to be structural: the policy checks tool names, so in `workspace-write` a
    # shell command could -- and, measured, did -- write outside the workspace.
    if os.environ.get("ARM_DOCKER_SANDBOX", "") not in ("", "0", "false", "no"):
        from ..sandbox.docker import DockerSandbox, containerized_bash
        from ..tools.builtin import BashParams
        from ..tools.definition import RegisteredTool, ToolDefinition

        available, why = DockerSandbox.available()
        if not available:
            print(f"warning: the Docker sandbox is unavailable: {why}", file=sys.stderr)
        else:
            container = DockerSandbox(
                sandbox,
                network=os.environ.get("ARM_DOCKER_NETWORK", "") not in ("", "0", "false", "no"),
            )
            registry.replace(
                RegisteredTool(
                    definition=ToolDefinition(
                        name="bash",
                        description=(
                            "Run a shell command inside a sandboxed container. The "
                            "workspace is the only writable location and the "
                            "network is unavailable; the container is discarded "
                            "after the call."
                        ),
                        parameters=BashParams.model_json_schema(),
                        requires_approval=True,
                    ),
                    body=containerized_bash(container),
                    validator=BashParams,
                )
            )



    # Delegation. The child is a real loop over the same adapter, with its own
    # session and a restricted tool view; the runtime enforces the depth limit
    # before any child exists.
    def build_child(child_session, child_registry, request):
        child_assembler = PromptAssembler()
        child_assembler.set("core", SUBAGENT_PROMPT, priority=10)
        return Runtime(
            child_session,
            adapter,
            child_registry,
            assembler=child_assembler,
            approval=approval,
            sandbox=sandbox,
            model=model,
        )

    subagents = SubagentRuntime(
        registry=registry, max_depth=DEFAULT_MAX_DEPTH
    )

    def delegation_tool(child_session, depth):
        """Build a ``subagent`` tool that knows the depth it is at.

        Each child gets its own, so the depth limit sees a number that actually
        increments. Inheriting the parent's tool would bake in depth 0 and make
        the limit unreachable.
        """
        return SubagentTool(
            subagents, parent_session=child_session, depth=depth
        ).register()

    subagents.register_provider(
        InProcessProvider(
            factory=ChildFactory(build=build_child),
            parent_registry=registry,
            approval=approval,
            sandbox=sandbox,
            delegation_tool_factory=delegation_tool,
            # Children are persisted alongside the root, so the session id the
            # model is handed actually resolves. Measured before this: a teammate
            # created a file, the lead's log recorded the child's id, and no log
            # for that id existed -- `arm sessions` listed only the root.
            session_dir=str(session_dir(home)) if persist else None,
        )
    )
    registry.register(delegation_tool(session, 0))

    # Reminders live in their own store, because they must fire whether or not
    # this session is loaded -- which is the situation a reminder exists to
    # escape. The file is per-home so two sessions do not share a schedule.
    schedule_path = Path(home or DEFAULT_HOME) / "schedule.json"
    schedule_store = _load_schedule(schedule_path)
    registry.register_many(
        ScheduleTools(schedule_store, session_id=session_id).tools()
    )

    # The team's state lives in this session's log -- there is no second store.
    # The tools are registered only when a team is asked for, so a session that
    # never uses one is not offered five tools it will not call.
    # Teams are always available. They were briefly gated behind ARM_TEAM, which
    # meant the feature could not be exercised at all in a default run -- so a
    # defect in it was invisible to every ordinary invocation, and the only way
    # to find one was to know the variable existed. A capability that has to be
    # switched on to be tested is a capability that is not really tested.
    if team_enabled():
        team = TeamJournal(session)

        async def team_delegate(name: str, task: str) -> str:
            """Start a teammate.

            The member is recorded before this runs, so a crash here leaves a
            roster row that reconciliation can resolve rather than an agent
            nothing knows about. The task is run to completion and the child's
            session id returned, so the member's ``child_session`` names something
            real.
            """
            result = await subagents.run(
                DelegationRequest(prompt=task, label=name),
                parent=session,
                timeout=TEAM_MEMBER_TIMEOUT,
            )
            return result.child_session

        async def team_deliver(name: str, body: str) -> bool:
            """Deliver a message to a teammate by starting a run for it.

            A teammate has no live agent between messages -- it is a task that
            runs when there is work -- so delivery means "run it with this
            message". A message that cannot be run is reported as undelivered
            rather than assumed delivered.
            """
            member = team.members().get(name)
            if member is None or not member.active:
                return False
            result = await subagents.run(
                DelegationRequest(prompt=body, label=f"{name}-message"),
                parent=session,
                timeout=TEAM_MEMBER_TIMEOUT,
            )
            return result.stop_reason in ("completed", "refusal")

        registry.register_many(
            TeamTools(team, delegate=team_delegate, deliver=team_deliver).tools()
        )

    # Plugins: one configuration file, one list, no layering. Loaded after the
    # built-in tools so a plugin can override or extend them, and before the
    # runtime is constructed so its registrations are present from the first turn.
    plugin_specs: list[str] = []
    config_path = Path(home or DEFAULT_HOME) / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                plugin_specs, plugin_diagnostics = parse_plugin_config(config)
                for note in plugin_diagnostics:
                    print(f"warning: config: {note}", file=sys.stderr)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: could not read {config_path}: {exc}", file=sys.stderr)

    plugin_registry = PluginRegistry(tools=registry, prompt=assembler)
    if plugin_specs:
        plugin_failures = load_plugins(plugin_registry, plugin_specs)
        for spec, reason in plugin_failures.items():
            print(f"warning: plugin {spec} did not load: {reason}", file=sys.stderr)

    # The browser seam is registered only when an endpoint is configured.
    #
    # Nothing is discovered by scanning: a runtime that found a browser on its own
    # would attach to whatever it found, and what it would find is the user's
    # logged-in session. Requiring ARM_BROWSER makes attaching a deliberate act.
    browser_endpoint = os.environ.get("ARM_BROWSER", "")
    if browser_endpoint:
        browser = Browser(browser_endpoint or BROWSER_DEFAULT_ENDPOINT)
        registry.register_many(BrowserTools(browser).tools())

    # Computer use is off unless explicitly enabled: the desktop is not a
    # sandbox, so a runtime that could drive the machine without being asked
    # would be reaching into the user's mail, their bank and their terminal.
    #
    # Two switches, because the two capabilities have different consequences:
    #
    #   ARM_COMPUTER=observe  look at the screen, do not touch it
    #   ARM_COMPUTER=1        observe and drive (the action tools still require
    #                         approval on top of this)
    #
    # The split is not cosmetic. Before it, granting screenshots also granted
    # typing -- so a runtime that only wanted to see what was on screen had to
    # allow itself to type into a password prompt. The tool layer already made
    # the distinction (`screen_capture` carries no approval requirement, the
    # acting tools do); the capability flag lumped them back together.
    computer_mode = os.environ.get("ARM_COMPUTER", "").strip().lower()
    if computer_mode not in ("", "0", "false", "no"):
        observe_only = computer_mode in ("observe", "read", "readonly", "read-only")
        desktop = Desktop(enabled=not observe_only, observe=True)
        available, why = Desktop.available()
        if available:
            registry.register_many(ComputerTools(desktop).tools())
        else:
            print(f"warning: computer use is unavailable: {why}", file=sys.stderr)

    # External hooks are loaded from a file and attached to the pipeline's
    # existing pre-execute waterfall. A broken configuration produces diagnostics
    # and no hooks, rather than preventing the agent from starting: the agent is
    # still useful, and refusing to run would make a typo look like a broken
    # installation.
    if hook_bridge is None:
        hook_handlers, hook_diagnostics = load_hook_config(
            Path(home or DEFAULT_HOME) / "hooks.json", dialect=DIALECT_CLAUDE
        )
        for note in hook_diagnostics:
            print(f"warning: hooks: {note}", file=sys.stderr)
        if hook_handlers:
            hook_bridge = HookBridge(HookRunner(hook_handlers))
            # Session-start hooks fire once, here, before the first turn. The
            # point was mapped and loaded but never consulted, so a SessionStart
            # hook -- the one people reach for first -- silently did nothing.
            context = asyncio.run(hook_bridge.agent_created(session.session_id))
            for note in context:
                session.append(
                    USER_MESSAGE,
                    {
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": note}],
                        },
                        "surface_op": "append",
                        "source": "hook:session-start",
                        # Context, not input: the inbox must not claim it, or the
                        # first turn would run twice.
                        "context_only": True,
                    },
                )

    # MCP servers are configured in a file rather than on the command line: a
    # server definition is a command line and an environment, which is not
    # something to retype. The file is read from the session home.
    mcp_manager = McpManager(registry)
    for server in _load_mcp_servers(Path(home or DEFAULT_HOME) / "mcp.json"):
        try:
            mcp_manager.register(server)
        except Exception as exc:  # noqa: BLE001 - one bad entry is not fatal
            print(f"warning: ignoring MCP server {server.name!r}: {exc}", file=sys.stderr)
    # MCP servers are connected by the caller, inside the event loop that will
    # use them. Connecting here with `asyncio.run` created a loop, opened the
    # servers' pipes on it, and then closed it -- so every later tool call failed:
    #
    #     got Future <Future pending> attached to a different loop
    #
    # Measured through the CLI: the server started, its tool was registered, and
    # every invocation of that tool failed. The registration made it look healthy.
    #
    # The connection is a resource bound to a loop, so it belongs to whatever loop
    # runs the session, not to configuration building.
        # A server's instructions become a prompt segment, attributed per server.
        # The assembler is consulted at prompt-render time, so registering the
        # segment after the servers connected is what makes the text appear in the
        # next request rather than never.
        section = mcp_manager.prompt_section()
        if section:
            assembler.set("mcp", section, priority=50)

    # MCP servers are child processes, so they must be closed or every session
    # leaks one per server. The manager is returned with the rest so the caller
    # that owns the process lifetime can close it.
    #
    # The scheduler is wired here, not in the runtime, because firing a reminder
    # means *persisting it into a session* -- and the session is what this
    # function owns. A scheduler inside the loop would have to reach back out for
    # the very thing the loop is driving.
    #
    # `resolve` returns the live session for this process. A reminder for a
    # session that is not loaded cannot be delivered here; the runtime records it
    # as failed rather than dropping it, so it is retried once that session is
    # opened.
    def _resolve(session_id: str) -> Any:
        return session if session_id == session_id_ else None

    def _deliver(target_session_id: str, body: str, message_id: str) -> bool:
        """Persist a reminder into the session, and confirm it landed.

        Confirmation is read back from the session's own log rather than assumed
        from the append succeeding: the receipt must describe a fact, and the
        only place that fact exists is the log.
        """
        try:
            before = len(session)
            session.append(
                USER_MESSAGE,
                {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": body}],
                    },
                    "surface_op": "append",
                    "source": "schedule",
                    "message_id": message_id,
                    # Context, not a new task: the inbox must not claim it, or a
                    # reminder would generate a turn every time it fired.
                    "context_only": True,
                },
            )
            if log_handle is not None:
                log_handle.flush()
        except Exception as exc:  # noqa: BLE001 - a failed delivery is not fatal
            print(f"warning: could not deliver a reminder: {exc}", file=sys.stderr)
            return False
        # Read back: the append is only a fact once it is in the log.
        return any(
            event.type == USER_MESSAGE
            and event.data.get("message_id") == message_id
            for event in session.events()[before:]
        )

    session_id_ = session_id
    scheduler = ScheduleRuntime(
        store=schedule_store, resolve=_resolve, deliver=_deliver
    )

    # Constructed here, after the scheduler exists, so the runtime can hold it.
    # The loop fires due reminders at the start of each turn; without this the
    # scheduler would be reachable only by tests -- reminders would be creatable
    # and never delivered.
    runtime = Runtime(
        session,
        adapter,
        registry,
        assembler=assembler,
        hooks=hooks,
        skills=skills,
        plan=plan_mode,
        approval=approval,
        sandbox=sandbox,
        scheduler=scheduler,
        subagents=subagents,
        hook_bridge=hook_bridge,
        model=model,
    )

    return runtime, session, log_handle, scheduler, mcp_manager


def _print_events(session: Session, *, verbose: bool) -> None:
    """Show the turn as it happens, using the log as the source."""
    for event in session.events():
        if event.type == "assistant/message":
            text = event.data["message"].get("content") or ()
            for block in text:
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block["text"], end="", flush=True)
        elif event.type == "tool/call" and verbose:
            print(
                f"\n[tool] {event.data['name']}({json.dumps(dict(event.data['arguments']), ensure_ascii=False)})",
                file=sys.stderr,
            )
        elif event.type == "tool/result" and verbose:
            payload = event.data["message"]["content"]
            text = payload[0]["text"] if payload else ""
            flag = "error" if event.data.get("is_error") else "ok"
            print(f"[{flag}] {text[:400]}", file=sys.stderr)
        elif event.type == "turn/end" and verbose:
            print(
                f"\n[turn {event.data['turn']} {event.data['reason']}, "
                f"{event.data['steps']} steps]",
                file=sys.stderr,
            )


async def _connect_mcp(mcp_manager: McpManager) -> None:
    """Start the MCP servers on *this* loop, before anything uses their tools.

    Called from inside the loop that runs the session, because the transport
    holds loop-bound resources: the subprocess and its pipes. A connection made on
    another loop is not usable from this one.
    """
    if not mcp_manager.servers:
        return
    failures = await mcp_manager.connect_all()
    required = mcp_manager.required_failures(failures)
    if required:
        raise RuntimeError(f"required MCP servers could not start: {required}")
    for name in failures:
        print(f"warning: MCP server {name!r} is unavailable", file=sys.stderr)


async def _run_task(
    runtime: Runtime, session: Session, task: str, *, verbose: bool,
    scheduler: ScheduleRuntime | None = None,
) -> int:
    # The timer runs for the duration of the task, so a reminder that comes due
    # while the agent is working is delivered into the same session rather than
    # waiting for the next turn.
    timer = ScheduleTimer(scheduler, interval=5.0) if scheduler is not None else None
    if timer is not None:
        timer.start()
    try:
        runtime.submit(task)
        before = len(session)
        outcomes = await runtime.run_until_idle()
    finally:
        if timer is not None:
            await timer.stop()
    _print_events(session, verbose=verbose)

    if not outcomes:
        print("no turn ran", file=sys.stderr)
        return 1
    reason = outcomes[-1].reason
    if verbose:
        print(f"\n[{len(session) - before} events appended]", file=sys.stderr)
    return 0 if reason == "completed" else 1


def _cmd_run(args: argparse.Namespace) -> int:
    runtime, session, handle, _scheduler, mcp_manager = build_runtime(
        session_id=args.session,
        model=args.model,
        home=args.home,
        hooks=None,
    )
    try:
        async def _with_mcp() -> int:
            await _connect_mcp(mcp_manager)
            return await _run_task(
                runtime, session, args.task,
                verbose=not args.quiet, scheduler=_scheduler,
            )

        return asyncio.run(_with_mcp())
    finally:
        if handle is not None:
            handle.close()


def _cmd_chat(args: argparse.Namespace) -> int:
    runtime, session, handle, scheduler, mcp_manager = build_runtime(
        session_id=args.session, model=args.model, home=args.home
    )
    try:
        print(f"arm · session {args.session} · model {args.model}")
        print("type a task, or 'exit' to leave\n")
        # The timer runs for the whole session, not just while a task is being
        # processed. A reminder set for three hours from now must fire while the
        # user is sitting at the prompt, which is the idle gap the timer exists
        # for -- covering only the running task would leave exactly that gap
        # uncovered, and the reminder would wait for the next unrelated task.
        async def _with_mcp() -> int:
            await _connect_mcp(mcp_manager)
            return await _chat_loop(runtime, session, scheduler, args)

        asyncio.run(_with_mcp())
    finally:
        # Children are stopped before the log is closed: a subagent still writing
        # to a closed handle would fail in a way nobody could attribute.
        orphans = asyncio.run(runtime.shutdown())
        if orphans:
            print(f"warning: {len(orphans)} subagent(s) did not stop", file=sys.stderr)
        # MCP servers are child processes; leaving them running would leak one per
        # session, and a server holding a lock or a port would block the next run.
        asyncio.run(mcp_manager.close_all())
        if handle is not None:
            handle.close()
    return 0


async def _chat_loop(
    runtime: Runtime, session: Session, scheduler: Any, args: argparse.Namespace
) -> int:
    """Read lines and run tasks, with the scheduler alive throughout.

    Reading a line is blocking, so it runs in an executor: awaiting it directly
    would stop the event loop, and a stopped loop cannot deliver a reminder. That
    is the whole difficulty here, and it is why the loop is async at all.
    """
    timer = ScheduleTimer(scheduler, interval=5.0) if scheduler is not None else None
    if timer is not None:
        timer.start()
    try:
        while True:
            try:
                line = (
                    await asyncio.get_running_loop().run_in_executor(None, input, "> ")
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in {"exit", "quit", ":q"}:
                return 0
            if line == ":replay":
                _cmd_replay(
                    argparse.Namespace(session=args.session, home=args.home, json=False)
                )
                continue
            # The per-task timer is omitted here: the session-wide timer above
            # already covers this task, and two timers would race on the same
            # tasks.
            #
            # Ctrl-C during a task interrupts *the task*, not the session. The
            # drain inside the pipeline still finishes every tool body it started
            # and records a result, so the work is not lost -- but the user needs
            # to be told that, or an interrupted command looks like a failed one.
            try:
                await _run_task(runtime, session, line, verbose=True)
            except KeyboardInterrupt:
                print("\n[interrupted -- started tool calls were finished and recorded]")
            print()
    finally:
        if timer is not None:
            await timer.stop()


def _cmd_replay(args: argparse.Namespace) -> int:
    """Re-derive a stored session and print exactly what the model saw.

    This is the M1 acceptance check made operational: if the derivation were not
    faithful, this command would show something the model never received.
    """
    path = session_path(args.session, args.home)
    if not path.exists():
        print(f"no session at {path}", file=sys.stderr)
        return 1

    result = read_log(path)
    session = Session.from_events(args.session, result.events)
    messages = session.derive()

    if args.json:
        print(json.dumps([m.to_record() for m in messages], ensure_ascii=False, indent=2))
        return 0

    print(f"session {args.session}: {len(result.events)} events, {len(messages)} messages")
    if result.truncate_to is not None:
        print(f"  (torn tail at byte {result.truncate_to})")
    print()
    for index, message in enumerate(messages):
        print(f"--- [{index}] {message.role} ---")
        if message.tool_calls:
            for call in message.tool_calls:
                print(f"  tool_call {call.get('name')} {json.dumps(dict(call.get('arguments', {})))}")
        if message.text:
            print(message.text)
        print()
    return 0


def _cmd_sessions(args: argparse.Namespace) -> int:
    directory = session_dir(args.home)
    if not directory.exists():
        print(f"no sessions in {directory}")
        return 0
    for path in sorted(directory.glob("*.jsonl")):
        result = read_log(path)
        print(f"{path.stem}\t{len(result.events)} events")
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    """List the skills the runtime would offer the model.

    Showing the catalogue rather than the files is deliberate: what matters is
    what the model will be told, not what happens to be on disk.
    """
    skills = SkillRegistry()
    skills.register_provider(
        FilesystemSkillProvider(
            FilesystemSkillProvider.default_roots(
                cwd=Path.cwd(),
                home=Path.home(),
                extra=[
                    Path(p)
                    for p in (os.environ.get("ARM_SKILL_DIRS") or "").split(":")
                    if p
                ],
            )
        )
    )
    snapshot = skills.snapshot()

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": s.name,
                        "description": s.description,
                        "model_invocable": s.model_invocable,
                        "user_invocable": s.user_invocable,
                        "source": s.source,
                    }
                    for s in snapshot.skills
                ],
                indent=2,
            )
        )
        return 0

    if not snapshot.skills:
        print("no skills found")
        print()
        print("skill roots are searched in this order:")
        for _rank, source, path in FilesystemSkillProvider.default_roots(
            cwd=Path.cwd(), home=Path.home()
        ):
            marker = "" if path.is_dir() else "  (missing)"
            print(f"  {source:16} {path}{marker}")
        return 0

    print(f"{len(snapshot.skills)} skill(s):")
    for skill in snapshot.skills:
        flags = []
        if not skill.model_invocable:
            flags.append("user-only")
        if not skill.user_invocable:
            flags.append("model-only")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {skill.name:24} {skill.description}{suffix}")
    if not snapshot.complete:
        print()
        print("  ! the catalogue is incomplete: a provider failed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arm", description="arm — an agent runtime built on one rule: what the model can see is recorded."
    )
    parser.add_argument(
        "--home",
        default=None,
        help=f"session store (default: {DEFAULT_HOME}, or $ARM_HOME)",
    )
    parser.add_argument("--model", default=os.environ.get("ARM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument(
        "--approval",
        choices=["ask", "never"],
        default=None,
        help=(
            "approval policy (default: ask, or $ARM_APPROVAL). 'never' consults "
            "nobody and denies every request that needs approval."
        ),
    )
    parser.add_argument(
        "--sandbox",
        default=None,
        help="sandbox preset: default / cautious / unattended / locked",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one task to completion")
    run.add_argument("task")
    run.add_argument("--session", default="default")
    run.set_defaults(func=_cmd_run)

    chat = sub.add_parser("chat", help="interactive session")
    chat.add_argument("--session", default="default")
    chat.set_defaults(func=_cmd_chat)

    replay = sub.add_parser("replay", help="re-derive a stored session")
    replay.add_argument("--session", default="default")
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(func=_cmd_replay)

    sessions = sub.add_parser("sessions", help="list stored sessions")
    sessions.set_defaults(func=_cmd_sessions)

    skills_parser = sub.add_parser("skills", help="list available skills")
    skills_parser.add_argument("--json", action="store_true")
    skills_parser.set_defaults(func=_cmd_skills)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

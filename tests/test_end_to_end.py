"""End-to-end acceptance tests: the whole runtime, no network.

These tests exist to answer one question -- does the system actually work when
assembled? Unit tests can all pass while the pieces disagree about their
interfaces, so this module drives a real ``Runtime`` against a real local HTTP
server that speaks the OpenAI wire format.

The server is deliberately a *real socket*, not a patched function. A mock that
replaces ``httpx`` would not exercise the streaming parser, the message
rendering, or the tool-call reassembly -- which are exactly the parts most likely
to be wrong.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


from arm.agent.loop import Runtime
from arm.llm.openai_adapter import OpenAICompatibleAdapter
from arm.prompt.assembler import PromptAssembler
from arm.session.persistence import JsonlLog, read_log
from arm.session.session import Session
from arm.tools.builtin import builtin_tools
from arm.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# A minimal OpenAI-compatible server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Serves scripted streaming completions.

    The script is a list of response *plans*; each request consumes one. A plan
    is either ``{"text": ...}`` or ``{"tool": (name, arguments)}``.
    """

    script: list[dict[str, Any]] = []
    received: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Silence the server's per-request logging during tests."""
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append(body)

        index = min(len(type(self).received) - 1, len(type(self).script) - 1)
        plan = type(self).script[index] if type(self).script else {"text": "ok"}

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        for chunk in _chunks_for(plan):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _chunks_for(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Render a plan as a sequence of streamed chunks."""
    chunks: list[dict[str, Any]] = []

    text = plan.get("text")
    if text:
        # Split into several chunks so the streaming path is genuinely exercised.
        size = max(1, len(text) // 3)
        for start in range(0, len(text), size):
            chunks.append(
                {
                    "choices": [
                        {"delta": {"content": text[start : start + size]}, "index": 0}
                    ]
                }
            )

    tool = plan.get("tool")
    if tool:
        name, arguments = tool
        payload = json.dumps(arguments)
        # The name arrives first, then the arguments in fragments -- the shape a
        # real provider produces, and the reason reassembly is needed.
        chunks.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": name, "arguments": ""},
                                }
                            ]
                        },
                        "index": 0,
                    }
                ]
            }
        )
        midpoint = max(1, len(payload) // 2)
        for fragment in (payload[:midpoint], payload[midpoint:]):
            chunks.append(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": fragment}}
                                ]
                            },
                            "index": 0,
                        }
                    ]
                }
            )

    chunks.append(
        {
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    return chunks


class _FailingHandler(_Handler):
    """Always answers 500, to exercise the provider-error path."""

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":"server exploded"}')


class FakeServer:
    """A real HTTP server on a real port, driven by a script.

    Each instance installs its own script onto the shared handler class. Tests
    must not mutate the handler in place, because the class is process-global and
    a mutation would leak into every later test.
    """

    def __init__(
        self, script: list[dict[str, Any]], handler: type[_Handler] = _Handler
    ) -> None:
        self._handler = handler
        self._handler.script = script
        self._handler.received = []
        self._server = HTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def received(self) -> list[dict[str, Any]]:
        return self._handler.received


class FailingServer(FakeServer):
    """A server that always returns HTTP 500."""

    def __init__(self) -> None:
        super().__init__([], handler=_FailingHandler)


def build_runtime(
    base_url: str, *, workspace: str | None = None
) -> tuple[Runtime, Session]:
    session = Session.create("e2e")
    adapter = OpenAICompatibleAdapter(
        api_key="test-key", base_url=base_url, model="test-model"
    )
    registry = ToolRegistry()
    registry.register_many(builtin_tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    return Runtime(session, adapter, registry, assembler=assembler, model="test-model"), session


# ---------------------------------------------------------------------------
# The whole loop, over HTTP
# ---------------------------------------------------------------------------


def test_full_turn_over_http():
    """A complete turn: prompt, stream, assistant message, turn/end."""
    with FakeServer([{"text": "hello from the server"}]) as server:
        runtime, session = build_runtime(server.base_url)
        runtime.submit("say hello")
        outcomes = asyncio.run(runtime.run_until_idle())

    assert outcomes[0].reason == "completed"
    texts = [m.text for m in session.derive()]
    # The transcript is: system prompt, the user's question, the tool
    # announcement, then the answer. Each is a separate surface node.
    assert texts[0] == "you are arm"
    assert "say hello" in texts
    assert texts[-1] == "hello from the server"
    assert texts.index("say hello") < texts.index("hello from the server")

    # The request really carried the system prompt and the user message.
    sent = server.received[0]
    assert sent["model"] == "test-model"
    assert sent["stream"] is True
    # system prompt, the user's question, and the developer notice announcing
    # the tool set -- in that order.
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user", "developer"]


def test_streamed_text_is_reassembled_across_chunks():
    """The streaming parser must join fragments, not emit them separately."""
    long_text = "abcdefghijklmnopqrstuvwxyz" * 4
    with FakeServer([{"text": long_text}]) as server:
        runtime, session = build_runtime(server.base_url)
        runtime.submit("go")
        asyncio.run(runtime.run_until_idle())

    assert [m.text for m in session.derive()][-1] == long_text


def test_tool_call_arguments_are_reassembled_from_fragments():
    """Tool-call arguments arrive split; they must be joined before parsing."""
    with FakeServer(
        [
            {"tool": ("bash", {"command": "echo hello"})},
            {"text": "the command printed hello"},
        ]
    ) as server:
        runtime, session = build_runtime(server.base_url)
        runtime.submit("run echo")
        outcomes = asyncio.run(runtime.run_until_idle())

    assert outcomes[0].reason == "completed"
    assert outcomes[0].steps == 2

    result = next(e for e in session if e.type == "tool/result")
    assert result.data["is_error"] is False
    assert "hello" in result.data["message"]["content"][0]["text"]


def test_tool_schemas_are_sent_to_the_model():
    """The model must be told what it can call."""
    with FakeServer([{"text": "ok"}]) as server:
        runtime, _ = build_runtime(server.base_url)
        runtime.submit("hi")
        asyncio.run(runtime.run_until_idle())

    tools = server.received[0].get("tools")
    assert tools, "tools must be advertised"
    names = {t["function"]["name"] for t in tools}
    assert {"read", "write", "edit", "bash", "glob", "grep"} <= names
    # The schema is the normalized one: no $defs, no presentation keywords.
    for tool in tools:
        assert "$defs" not in json.dumps(tool["function"]["parameters"])


def test_provider_http_error_becomes_a_turn_error():
    """A 500 is an outcome the log records, not an exception the caller sees."""
    with FailingServer() as server:
        runtime, session = build_runtime(server.base_url)
        runtime.submit("hi")
        outcomes = asyncio.run(runtime.run_until_idle())

    assert outcomes[0].reason == "error"
    types = [e.type for e in session]
    assert "error/turn-failure" in types
    assert types[-1] == "turn/end"


def test_multi_step_turn_records_every_step():
    """A tool-using turn is several steps of one turn."""
    with FakeServer(
        [
            {"tool": ("bash", {"command": "echo one"})},
            {"tool": ("bash", {"command": "echo two"})},
            {"text": "both commands ran"},
        ]
    ) as server:
        runtime, session = build_runtime(server.base_url)
        runtime.submit("run two commands")
        outcomes = asyncio.run(runtime.run_until_idle())

    # Three steps: two tool-using steps plus the final answer.
    assert outcomes[0].steps == 3
    types = [e.type for e in session]
    assert types.count("step/start") == 3
    assert types.count("step/end") == 3
    assert types.count("turn/start") == 1
    assert types[-1] == "turn/end"
    # Both tool results are present and paired.
    assert types.count("tool/call") == 2
    assert types.count("tool/result") == 2


# ---------------------------------------------------------------------------
# The M1 acceptance criterion, end to end
# ---------------------------------------------------------------------------


def test_replay_reproduces_the_request_byte_for_byte(tmp_path):
    """M1 acceptance: the log reconstructs exactly what the model received.

    This is the central claim of the whole design, so it is checked against a
    real request that a real HTTP server received, not against an internal
    expectation.
    """
    with FakeServer(
        [
            {"tool": ("bash", {"command": "echo hi"})},
            {"text": "done"},
        ]
    ) as server:
        runtime, session = build_runtime(server.base_url)
        runtime.submit("do it")
        asyncio.run(runtime.run_until_idle())

        # Persist, then rehydrate from disk alone.
        path = tmp_path / "session.jsonl"
        JsonlLog(path).append(session.events())
        restored = Session.from_events("e2e", read_log(path).events)

    # Every request the server saw must be reproducible from the log: each
    # request was a prefix of the final transcript at the moment it was sent.
    replayed = [m.to_record() for m in restored.derive()]
    for sent in server.received:
        wire = _render_wire_messages(sent["messages"])
        # The replayed transcript starts with exactly what was sent.
        assert wire == replayed[: len(wire)], (
            "the replayed prefix does not match what the server received"
        )


def _render_wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize wire messages into the shape ``Message.to_record`` produces.

    Only the fields the runtime controls are compared; provider-specific extras
    (``content: null`` for a tool-calling assistant message, for instance) are
    normalized away so the comparison is about content, not formatting.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        record: dict[str, Any] = {"role": message["role"]}
        content = message.get("content")
        if isinstance(content, str) and content:
            record["content"] = [{"type": "text", "text": content}]
        else:
            record["content"] = []
        if message.get("tool_call_id"):
            record["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            record["name"] = message["name"]
        if message.get("tool_calls"):
            record["tool_calls"] = [
                {
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]),
                }
                for call in message["tool_calls"]
            ]
        out.append(record)
    return out


def test_session_survives_a_restart(tmp_path):
    """A stored session replays to the same view after a fresh process boundary."""
    with FakeServer([{"text": "first answer"}]) as server:
        runtime, session = build_runtime(server.base_url)
        runtime.submit("first question")
        asyncio.run(runtime.run_until_idle())

        path = tmp_path / "s.jsonl"
        JsonlLog(path).append(session.events())
        original = [m.to_record() for m in session.derive()]

    restored = Session.from_events("e2e", read_log(path).events)
    assert [m.to_record() for m in restored.derive()] == original


def test_second_turn_continues_the_same_transcript(tmp_path):
    """Rehydrating and continuing must append, not restart."""
    with FakeServer([{"text": "answer one"}]) as server:
        runtime, session = build_runtime(server.base_url)
        runtime.submit("question one")
        asyncio.run(runtime.run_until_idle())
        path = tmp_path / "s.jsonl"
        JsonlLog(path).append(session.events())

    with FakeServer([{"text": "answer two"}]) as server2:
        restored = Session.from_events("e2e", read_log(path).events)
        adapter = OpenAICompatibleAdapter(
            api_key="k", base_url=server2.base_url, model="test-model"
        )
        registry = ToolRegistry()
        registry.register_many(builtin_tools())
        assembler = PromptAssembler()
        assembler.set("core", "you are arm", priority=10)
        runtime2 = Runtime(
            restored, adapter, registry, assembler=assembler, model="test-model"
        )
        runtime2.submit("question two")
        asyncio.run(runtime2.run_until_idle())

        # The second request carried the whole earlier conversation. The count
        # is over *questions*: the skill catalogue is also a user-role message,
        # so counting the role alone would conflate the two.
        messages = server2.received[0]["messages"]
        questions = [
            m for m in messages
            if m["role"] == "user"
            and not str(m.get("content", "")).startswith("<system-reminder>")
        ]
        assert len(questions) == 2, [m.get("content") for m in questions]

    # The prompt and tool announcement appear once each, not once per turn: the
    # resumed runtime recognised both from the log and did not re-record them.
    texts = [m.text for m in restored.derive()]
    assert texts.count("you are arm") == 1
    assert sum(1 for t in texts if t.startswith("Tools now available")) == 1
    conversation = [t for t in texts if t in {"question one", "answer one", "question two", "answer two"}]
    assert conversation == ["question one", "answer one", "question two", "answer two"]


# ---------------------------------------------------------------------------
# Persistence through the real CLI wiring
# ---------------------------------------------------------------------------


def test_cli_wiring_persists_a_restorable_session(tmp_path):
    """A session written by the CLI must be restorable from disk alone.

    This pins a bug that no unit test caught: the persistence listener was
    attached *after* ``Session.create`` had already written its seed events, so
    the stored log began at seq 2. A log that is not contiguous from 0 cannot be
    restored at all, so every CLI session was silently unresumable.
    """
    from arm.cli.main import build_runtime

    home = str(tmp_path)
    with FakeServer([{"text": "the answer"}]) as server:
        runtime, session, handle, _scheduler = build_runtime(
            session_id="cli",
            model="test-model",
            api_key="k",
            base_url=server.base_url,
            home=home,
        )
        runtime.submit("a question")
        outcomes = asyncio.run(runtime.run_until_idle())
        assert handle is not None
        handle.close()

    assert outcomes[0].reason == "completed"

    path = tmp_path / "sessions" / "cli.jsonl"
    assert path.exists()

    result = read_log(path)
    seqs = [e.seq for e in result.events]
    assert seqs == list(range(len(seqs))), (
        f"the stored log must be contiguous from 0, got {seqs[:5]}..."
    )

    # And it round-trips: the same messages come back.
    restored = Session.from_events("cli", result.events)
    original = [m.to_record() for m in session.derive()]
    assert [m.to_record() for m in restored.derive()] == original


def test_cli_resumes_an_existing_session(tmp_path):
    """A second run against the same session appends rather than restarts."""
    from arm.cli.main import build_runtime

    home = str(tmp_path)
    with FakeServer([{"text": "first answer"}]) as server:
        runtime, session, handle, _scheduler = build_runtime(
            session_id="resume",
            model="m",
            api_key="k",
            base_url=server.base_url,
            home=home,
        )
        runtime.submit("first question")
        asyncio.run(runtime.run_until_idle())
        assert handle is not None
        handle.close()

    with FakeServer([{"text": "second answer"}]) as server2:
        runtime2, session2, handle2, _s2 = build_runtime(
            session_id="resume",
            model="m",
            api_key="k",
            base_url=server2.base_url,
            home=home,
        )
        # The resumed session already knows the earlier conversation.
        assert len(session2) > 2, "the earlier events should have been loaded"
        runtime2.submit("second question")
        asyncio.run(runtime2.run_until_idle())
        assert handle2 is not None
        handle2.close()

        # The second request carried the whole earlier conversation. The count
        # is over *questions*: the skill catalogue is also a user-role message,
        # so counting the role alone would conflate the two.
        messages = server2.received[0]["messages"]
        questions = [
            m for m in messages
            if m["role"] == "user"
            and not str(m.get("content", "")).startswith("<system-reminder>")
        ]
        assert len(questions) == 2, [m.get("content") for m in questions]

    texts = [m.text for m in session2.derive()]
    assert "first question" in texts
    assert "second question" in texts

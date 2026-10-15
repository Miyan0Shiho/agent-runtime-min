"""Todo tests.

Two claims are under test, and both are about restraint:

* **The list is replaced wholesale.** One operation, one code path, and a log
  entry that needs no prior state to interpret.
* **The list is never injected into the prompt.** The model wrote it, so feeding
  it back would spend tokens restating its own draft and present that draft as
  though it were an instruction.

The second claim is checked by asserting that no prompt-visible text contains the
list -- a negative test, because the failure mode is something appearing that
should not.
"""

from __future__ import annotations

import asyncio

import pytest

from arm.session.events import TURN_START, TODO_WRITE
from arm.session.session import Session
from arm.todo.store import (
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    TodoError,
    TodoItem,
    TodoStore,
    TodoSummary,
    project,
    validate,
)
from arm.todo.tools import TodoTools


def make_store(**kwargs) -> tuple[TodoStore, Session]:
    session = Session.create("todo")
    return TodoStore(session, **kwargs), session


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_content_is_refused():
    """An empty task is not a task."""
    with pytest.raises(TodoError) as info:
        validate([TodoItem(content="   ")])
    assert info.value.code == "TODO_EMPTY_CONTENT"


def test_duplicate_content_is_refused():
    """A duplicated entry makes "which one is done?" unanswerable.

    The whole-table replacement means there is no id to disambiguate with, so the
    content *is* the identity -- and identities must be unique.
    """
    with pytest.raises(TodoError) as info:
        validate([TodoItem(content="same"), TodoItem(content="same")])
    assert info.value.code == "TODO_DUPLICATE"


def test_duplicate_detection_ignores_surrounding_whitespace():
    """``"task"`` and ``" task "`` are the same task."""
    with pytest.raises(TodoError) as info:
        validate([TodoItem(content="task"), TodoItem(content="  task  ")])
    assert info.value.code == "TODO_DUPLICATE"


def test_at_most_one_task_may_be_in_progress():
    """Two things in progress is usually a wish list, not a plan."""
    with pytest.raises(TodoError) as info:
        validate(
            [
                TodoItem(content="a", status=STATUS_IN_PROGRESS),
                TodoItem(content="b", status=STATUS_IN_PROGRESS),
            ]
        )
    assert info.value.code == "TODO_PARALLEL_IN_PROGRESS"


def test_parallel_progress_can_be_enabled():
    """Sometimes work genuinely is parallel."""
    items = validate(
        [
            TodoItem(content="a", status=STATUS_IN_PROGRESS),
            TodoItem(content="b", status=STATUS_IN_PROGRESS),
        ],
        allow_parallel_in_progress=True,
    )
    assert len(items) == 2


def test_an_unknown_status_is_refused():
    """A typo in a status must fail, not be silently normalized."""
    with pytest.raises(TodoError) as info:
        TodoItem(content="x", status="nearly-done")
    assert info.value.code == "TODO_INVALID_STATUS"


def test_validation_normalizes_whitespace():
    """The stored content is the trimmed form, so equality is meaningful."""
    items = validate([TodoItem(content="  padded  ")])
    assert items[0].content == "padded"


def test_an_empty_list_is_valid():
    """Clearing the list is a legitimate write."""
    assert validate([]) == []


# ---------------------------------------------------------------------------
# Whole-table replacement
# ---------------------------------------------------------------------------


def test_a_write_replaces_the_previous_list():
    """There is no add and no delete: one write, one whole table."""
    store, _ = make_store()
    store.write([TodoItem("first"), TodoItem("second")])
    store.write([TodoItem("only this")])

    assert [i.content for i in store.current()] == ["only this"]


def test_clearing_is_an_empty_replacement():
    """Clearing is a write of nothing, not a deletion."""
    store, session = make_store()
    store.write([TodoItem("x")])
    store.clear()

    assert store.current() == []
    writes = [e for e in session if e.type == TODO_WRITE]
    assert len(writes) == 2, "clearing should write an event, not remove one"
    assert list(writes[-1].data["todos"]) == []


def test_the_write_is_recorded_in_the_log():
    """Durability is the point."""
    store, session = make_store()
    store.write([TodoItem("task", status=STATUS_IN_PROGRESS)])

    writes = [e for e in session if e.type == TODO_WRITE]
    assert len(writes) == 1
    assert writes[0].data["todos"][0]["content"] == "task"
    assert writes[0].data["todos"][0]["status"] == STATUS_IN_PROGRESS


def test_the_list_survives_a_restart():
    """Folded from the log, so recovery is free."""
    store, session = make_store()
    store.write([TodoItem("a"), TodoItem("b", status=STATUS_COMPLETED)])

    restored = TodoStore(Session.from_events("todo", session.events()))
    assert [i.content for i in restored.current()] == ["a", "b"]


def test_history_records_every_version():
    """A reader can see how the plan evolved."""
    store, _ = make_store()
    store.write([TodoItem("a")])
    store.write([TodoItem("a"), TodoItem("b")])
    store.clear()

    history = store.history()
    assert [len(h) for h in history] == [1, 2, 0]


def test_an_unwritten_list_is_empty():
    """A session that never used todos has no list."""
    store, _ = make_store()
    assert store.current() == []


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_the_summary_counts_each_status():
    """The one line the model gets back must be accurate."""
    store, _ = make_store()
    summary = store.write(
        [
            TodoItem("a", status=STATUS_COMPLETED),
            TodoItem("b", status=STATUS_IN_PROGRESS),
            TodoItem("c"),
            TodoItem("d"),
        ]
    )
    assert (summary.completed, summary.in_progress, summary.pending) == (1, 1, 2)
    assert summary.total == 4


def test_the_summary_renders_one_line():
    """The model already has the list; echoing it would be waste."""
    rendered = TodoSummary(pending=2, in_progress=1, completed=3).render()
    assert rendered == "Updated todo list: 2 pending, 1 in progress, 3 completed."
    assert "\n" not in rendered


def test_the_store_summary_matches_the_list():
    """The summary is derived, so it cannot drift."""
    store, _ = make_store()
    store.write([TodoItem("a"), TodoItem("b", status=STATUS_COMPLETED)])
    assert store.summary().total == 2


# ---------------------------------------------------------------------------
# The projection is stale-aware
# ---------------------------------------------------------------------------


def test_the_projection_hides_a_stale_list():
    """A list from three turns ago describes work that is probably finished.

    Showing it as current would be a lie, so the projection returns ``None``
    rather than a list that is merely old.
    """
    store, session = make_store()
    store.write([TodoItem("from an earlier turn")])

    # A new turn begins, so the list is no longer current.
    session.append(TURN_START, {"turn": 1, "claimed_seqs": [], "kind": "turn"})
    assert project(session) is None


def test_the_projection_shows_a_current_list():
    """A list written during the open turn is current."""
    session = Session.create("todo")
    session.append(TURN_START, {"turn": 0, "claimed_seqs": [], "kind": "turn"})
    TodoStore(session).write([TodoItem("current work")])

    projected = project(session)
    assert projected is not None
    assert [i.content for i in projected] == ["current work"]


def test_the_projection_is_none_when_nothing_was_written():
    """No list is not an empty list."""
    session = Session.create("todo")
    assert project(session) is None


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


async def invoke(tool, arguments: dict):
    return await tool.body(arguments, None)


def make_tools(**kwargs):
    store, session = make_store(**kwargs)
    tools = {t.name: t for t in TodoTools(store).tools()}
    return tools, store, session


def test_there_is_exactly_one_todo_tool():
    """One operation, one tool."""
    tools, _store, _session = make_tools()
    assert list(tools) == ["todo_write"]


def test_the_tool_writes_a_list():
    """The ordinary path."""
    tools, store, _session = make_tools()
    result = asyncio.run(
        invoke(
            tools["todo_write"],
            {
                "todos": [
                    {"content": "first", "status": "completed"},
                    {"content": "second", "status": "in_progress"},
                ]
            },
        )
    )
    assert not result.is_error
    # Two items: one completed, one in progress. Nothing is pending.
    assert "0 pending, 1 in progress, 1 completed" in result.content
    assert [i.content for i in store.current()] == ["first", "second"]


def test_the_tool_returns_only_counts():
    """The model supplied the list, so echoing it would restate what it has."""
    tools, _store, _session = make_tools()
    result = asyncio.run(
        invoke(
            tools["todo_write"],
            {"todos": [{"content": "a unique task name", "status": "pending"}]},
        )
    )
    assert "a unique task name" not in result.content
    assert "1 pending" in result.content


def test_the_tool_reports_a_refusal_as_a_result():
    """A refusal is an ordinary outcome, so the model can fix the list."""
    tools, _store, _session = make_tools()
    result = asyncio.run(
        invoke(
            tools["todo_write"],
            {
                "todos": [
                    {"content": "a", "status": "in_progress"},
                    {"content": "b", "status": "in_progress"},
                ]
            },
        )
    )
    assert result.is_error
    assert result.error_code == "TODO_PARALLEL_IN_PROGRESS"
    assert "at most one" in result.content.lower()


def test_the_tool_reports_a_duplicate_as_a_result():
    """The message must say what is wrong."""
    tools, _store, _session = make_tools()
    result = asyncio.run(
        invoke(
            tools["todo_write"],
            {"todos": [{"content": "same"}, {"content": "same"}]},
        )
    )
    assert result.is_error
    assert "duplicate" in result.content.lower()


def test_the_tool_can_clear_the_list():
    """An empty list is a valid write."""
    tools, store, _session = make_tools()
    asyncio.run(invoke(tools["todo_write"], {"todos": [{"content": "x"}]}))
    result = asyncio.run(invoke(tools["todo_write"], {"todos": []}))

    assert not result.is_error
    assert store.current() == []


def test_the_tool_schema_normalizes():
    """The schema must survive the model-facing subset."""
    import json

    tools, _store, _session = make_tools()
    rendered = json.dumps(dict(tools["todo_write"].definition.parameters))
    assert "title" not in rendered
    assert "$defs" not in rendered
    # The status enum must reach the model, so it cannot invent a status.
    assert "pending" in rendered
    assert "in_progress" in rendered


# ---------------------------------------------------------------------------
# The list never reaches the prompt
# ---------------------------------------------------------------------------


def test_the_todo_list_is_not_injected_into_the_prompt():
    """The central negative claim of this module.

    Feeding the model's own scratchpad back to it would spend tokens restating
    something it just wrote, and would present its draft as though it were an
    instruction from the user.
    """
    from arm.agent.loop import Runtime
    from arm.llm.mock import MockAdapter, text_response
    from arm.prompt.assembler import PromptAssembler
    from arm.tools.registry import ToolRegistry

    session = Session.create("todo-prompt")
    store = TodoStore(session)
    registry = ToolRegistry()
    registry.register_many(TodoTools(store).tools())
    assembler = PromptAssembler()
    assembler.set("core", "you are arm", priority=10)
    adapter = MockAdapter([text_response("ok")])
    runtime = Runtime(
        session, adapter, registry, assembler=assembler, model="mock"
    )

    store.write([TodoItem("a task the prompt must not mention")])
    runtime.submit("do the work")
    asyncio.run(runtime.run_until_idle())

    sent = adapter.requests[0]
    for message in sent.messages:
        assert "a task the prompt must not mention" not in message.text, (
            f"the todo list leaked into a {message.role} message"
        )
    assert "a task the prompt must not mention" not in assembler.render()

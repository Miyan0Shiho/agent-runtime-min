"""M3.2: does a pydantic-generated schema survive a real provider?

The contract's concern: "上游对「原始 JSON Schema 子集」有明确边界... `pydantic` 会
生成 `$defs`/`anyOf` 等结构，模型未必都能正确理解。验证方法:用生成的 schema 跑一次
真实工具调用,统计失败率。"

These tests need a real model and are skipped without one. That is the honest
trade, and it is what the contract asked for: the question is whether a *provider*
accepts what pydantic emits, and no mock can answer it.

Run with:

    ARM_TEST_MODEL_KEY=... ARM_TEST_MODEL_BASE=https://api.deepseek.com \\
        pytest tests/test_m32_real_model.py -q

**What was measured (2026-10-16, deepseek-chat):** every schema is plain JSON
Schema -- no `$defs`, no `anyOf`, no `$ref` -- because the tool parameters are
written by hand rather than generated from a pydantic model. A real call
round-trips: the model returns a tool call with correctly-typed arguments, and
`usage` and `finish_reason` are both populated.

An earlier attempt measured "acceptance" as "did the model call the tool I had in
mind", which produced 10/12 and looked like a schema failure rate. It was not:
`edit` was passed over in favour of `read` because reading first is sensible. With
`edit` as the only option it is chosen every time. **The distinction matters** --
conflating "the model preferred another tool" with "the schema was rejected" would
have produced a number that meant nothing.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from arm.llm.openai_adapter import OpenAICompatibleAdapter
from arm.llm.seam import Request, StreamCollector, ToolSchema
from arm.session.messages import Block, Message
from arm.tools.builtin import builtin_tools

KEY = os.environ.get("ARM_TEST_MODEL_KEY", "")
BASE = os.environ.get("ARM_TEST_MODEL_BASE", "https://api.deepseek.com")
MODEL = os.environ.get("ARM_TEST_MODEL", "deepseek-chat")

pytestmark = pytest.mark.skipif(
    not KEY, reason="no ARM_TEST_MODEL_KEY; M3.2 needs a real provider"
)


def _adapter() -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(api_key=KEY, base_url=BASE, model=MODEL)


def _schemas(names: tuple[str, ...] | None = None) -> tuple[ToolSchema, ...]:
    tools = builtin_tools()
    if names is not None:
        tools = [t for t in tools if t.definition.name in names]
    return tuple(
        ToolSchema(
            name=t.definition.name,
            description=t.definition.description,
            parameters=t.definition.parameters,
        )
        for t in tools
    )


async def _call(prompt: str, names: tuple[str, ...] | None = None):
    collector = StreamCollector()
    request = Request(
        messages=(Message(role="user", content=(Block(kind="text", text=prompt),)),),
        tools=_schemas(names),
        model=MODEL,
    )
    async for delta in _adapter().stream(request):
        collector.add(delta)
    return collector.result()


# ===========================================================================
# The generated schemas are plain JSON Schema
# ===========================================================================


def test_no_generated_schema_uses_constructs_models_often_mishandle():
    """The contract's specific worry: `$defs`, `anyOf`, `$ref`.

    None appear, because the tool parameters are written out rather than derived
    from a pydantic model. That is why this item was less risky than the contract
    assumed -- and the test exists so that adding a generated schema is a visible
    change rather than a quiet one.
    """
    risky = {"$defs", "anyOf", "oneOf", "allOf", "$ref", "definitions"}
    offenders: dict[str, list[str]] = {}

    def walk(node, found: set[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in risky:
                    found.add(key)
                walk(value, found)
        elif isinstance(node, list):
            for value in node:
                walk(value, found)

    for tool in builtin_tools():
        found: set[str] = set()
        walk(tool.definition.parameters, found)
        if found:
            offenders[tool.definition.name] = sorted(found)

    assert not offenders, (
        f"these tool schemas use constructs a provider may not understand: {offenders}"
    )


def test_every_schema_is_an_object_with_typed_properties():
    """A schema a provider will accept: an object, with described properties."""
    for tool in builtin_tools():
        schema = tool.definition.parameters
        assert schema.get("type") == "object", f"{tool.definition.name}: not an object"
        for name, prop in (schema.get("properties") or {}).items():
            assert "type" in prop, f"{tool.definition.name}.{name} has no type"
        for required in schema.get("required") or ():
            assert required in (schema.get("properties") or {}), (
                f"{tool.definition.name} requires {required!r}, which is not declared"
            )


# ===========================================================================
# A real provider accepts them
# ===========================================================================


def test_a_real_provider_returns_a_tool_call_with_typed_arguments():
    """The round trip the contract asks for."""
    result = asyncio.run(
        _call("Read the file pyproject.toml using the read tool.", ("read",))
    )

    calls = result.tool_calls or ()
    assert calls, f"the provider returned no tool call: {result.text!r}"
    call = calls[0]
    assert call.name == "read"
    assert isinstance(call.arguments, dict)
    assert isinstance(call.arguments.get("path"), str), call.arguments


def test_a_real_provider_reports_usage_and_a_finish_reason():
    """Both were requested and discarded before this project fixed it.

    Worth asserting against a real provider rather than a fake one: the fake
    always sends them, so a fake-based test passes whether or not they are parsed.
    """
    result = asyncio.run(_call("What is 2 + 2? Reply with just the number.", ()))

    assert result.usage is not None, "the provider reported no usage"
    assert result.usage.prompt_tokens > 0
    assert result.finish_reason in {"stop", "length"}, result.finish_reason


def test_the_model_can_select_a_tool_that_is_its_only_option():
    """Separating schema acceptance from tool preference.

    Measured earlier: with `edit` and `read` both available, the model reads
    first. That is sensible, and scoring it as a schema failure produced a
    meaningless 83%. With `edit` alone it is chosen -- which is the question the
    contract actually asks.
    """
    result = asyncio.run(
        _call("Replace the string 'foo' with 'bar' in file.txt.", ("edit",))
    )

    calls = result.tool_calls or ()
    assert calls, f"the provider returned no tool call: {result.text!r}"
    assert calls[0].name == "edit", calls[0].name

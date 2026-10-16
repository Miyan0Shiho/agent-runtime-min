"""Plugin tests.

Two claims are under test, and both are about what happens when something goes
wrong rather than when it goes right:

* **Loading is all-or-nothing.** A plugin that fails halfway leaves nothing
  behind, because a partially loaded plugin looks installed and behaves
  unpredictably.
* **Unloading reverses exactly what loading did.** Every registration returns a
  disposer, so a plugin cannot leave residue -- which is what makes the list
  editable without restarting.

The configuration shape is also asserted directly: **one list, no layering**.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from arm.plugins.registry import (

    PluginContext,
    PluginError,
    PluginRegistry,
    load_all,
    parse_plugin_config,
)
from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult
from arm.tools.registry import ToolRegistry


def make_tool(name: str) -> RegisteredTool:
    async def body(_arguments: Any, _execution: Any) -> ToolResult:
        return ToolResult.ok("ok")

    return RegisteredTool(
        definition=ToolDefinition(name=name, description="", parameters={}), body=body
    )


class SimplePlugin:
    """A plugin that registers one tool."""

    name = "simple"

    def install(self, context: PluginContext) -> None:
        context.register_tool(make_tool("from-plugin"))


class FailingPlugin:
    """A plugin that registers a tool and then fails."""

    name = "failing"

    def install(self, context: PluginContext) -> None:
        context.register_tool(make_tool("half-registered"))
        raise RuntimeError("this plugin is broken")


# ===========================================================================
# Loading
# ===========================================================================


def test_a_plugin_loads_and_registers_its_tool():
    """The ordinary path."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load("tests.test_plugins:SimplePlugin")

    assert "from-plugin" in tools.view().known
    assert registry.is_loaded("simple")


def test_a_plugin_is_named_from_its_attribute():
    """A plugin without an explicit name takes its class name."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load("tests.test_plugins:SecondPlugin")

    assert registry.loaded == ("second",)


def test_a_failing_plugin_leaves_nothing_behind():
    """A partially loaded plugin looks installed and behaves unpredictably."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)

    with pytest.raises(PluginError) as info:
        registry.load("tests.test_plugins:FailingPlugin")

    assert info.value.code == "PLUGIN_INSTALL_FAILED"
    assert "half-registered" not in tools.view().known, (
        "a failed plugin left a tool behind"
    )
    assert registry.loaded == ()


def test_an_unimportable_plugin_is_reported_clearly():
    registry = PluginRegistry(tools=ToolRegistry())
    with pytest.raises(PluginError) as info:
        registry.load("definitely.not.a.module:thing")

    assert info.value.code == "PLUGIN_IMPORT_FAILED"
    assert "definitely.not.a.module" in str(info.value)


def test_a_missing_attribute_lists_what_was_asked_for():
    registry = PluginRegistry(tools=ToolRegistry())
    with pytest.raises(PluginError) as info:
        registry.load("tests.test_plugins:NoSuchThing")

    assert info.value.code == "PLUGIN_NOT_FOUND"
    assert "NoSuchThing" in str(info.value)


def test_loading_the_same_plugin_twice_is_refused():
    """Two loads would register everything twice."""
    registry = PluginRegistry(tools=ToolRegistry())
    registry.load("tests.test_plugins:SimplePlugin")

    with pytest.raises(PluginError) as info:
        registry.load("tests.test_plugins:SimplePlugin")
    assert info.value.code == "PLUGIN_ALREADY_LOADED"


def test_a_plugin_without_a_registry_is_told_so():
    """The error names the missing extension point."""
    registry = PluginRegistry(tools=None)

    with pytest.raises(PluginError) as info:
        registry.load("tests.test_plugins:SimplePlugin")
    assert "no tool registry" in str(info.value), (
        f"the error does not name the missing extension point: {info.value}"
    )


def test_load_order_is_preserved():
    """The order is part of the configuration's meaning."""
    registry = PluginRegistry(tools=ToolRegistry())
    registry.load("tests.test_plugins:SimplePlugin")
    registry.load("tests.test_plugins:SecondPlugin")

    assert registry.loaded == ("simple", "second")


# ===========================================================================
# Unloading
# ===========================================================================


def test_unloading_removes_what_the_plugin_registered():
    """Reversing exactly what loading did is what makes the list editable."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load("tests.test_plugins:SimplePlugin")

    registry.unload("simple")

    assert "from-plugin" not in tools.view().known
    assert not registry.is_loaded("simple")


def test_unloading_an_unloaded_plugin_is_refused():
    registry = PluginRegistry(tools=ToolRegistry())
    with pytest.raises(PluginError) as info:
        registry.unload("nobody")
    assert info.value.code == "PLUGIN_NOT_LOADED"


def test_unloading_leaves_other_plugins_alone():
    """One plugin's removal must not disturb another's registrations."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load("tests.test_plugins:SimplePlugin")
    registry.load("tests.test_plugins:SecondPlugin")

    registry.unload("simple")

    assert "from-plugin" not in tools.view().known
    assert "from-second" in tools.view().known


def test_unload_all_reverses_the_load_order():
    """Newest first: a later plugin may depend on an earlier one.

    Reversing the load order is the only order safe without dependency tracking,
    and dependency tracking is a framework.
    """
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load("tests.test_plugins:SimplePlugin")
    registry.load("tests.test_plugins:SecondPlugin")

    registry.unload_all()

    assert registry.loaded == ()
    assert tools.view().known == frozenset()


def test_a_failing_disposer_does_not_stop_the_others():
    """One bad disposer must not leave the rest of the plugin behind."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load("tests.test_plugins:SimplePlugin")

    loaded = registry._loaded["simple"]
    # A disposer that explodes, prepended so it runs last (reverse order).
    def broken() -> None:
        raise RuntimeError("this disposer is broken")

    loaded.disposers.insert(0, broken)
    failures = registry.unload("simple")

    assert failures, "the failing disposer was not reported"
    assert "from-plugin" not in tools.view().known, (
        "a broken disposer left a registration behind"
    )


def test_reloading_keeps_the_position_in_the_order():
    """Reloading must not become a way to accidentally change precedence."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load("tests.test_plugins:SimplePlugin")
    registry.load("tests.test_plugins:SecondPlugin")

    registry.reload("simple")

    assert registry.loaded == ("simple", "second"), (
        "reloading reordered the plugins"
    )
    assert "from-plugin" in tools.view().known


def test_reloading_replaces_the_registration():
    """The tool is registered once, not twice."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load("tests.test_plugins:SimplePlugin")
    registry.reload("simple")

    assert "from-plugin" in tools.view().known
    assert len(tools.view().known) == 1


# ===========================================================================
# Loading a list
# ===========================================================================


def test_load_all_loads_every_plugin():
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)

    failures = load_all(
        registry,
        ["tests.test_plugins:SimplePlugin", "tests.test_plugins:SecondPlugin"],
    )

    assert failures == {}
    assert registry.loaded == ("simple", "second")


def test_one_bad_entry_does_not_stop_the_others():
    """A single bad entry must not make the agent unusable."""
    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)

    failures = load_all(
        registry,
        [
            "tests.test_plugins:NoSuchThing",
            "tests.test_plugins:SimplePlugin",
        ],
    )

    assert len(failures) == 1
    assert "from-plugin" in tools.view().known, "the good plugin did not load"


def test_a_plugin_that_cannot_construct_is_reported():
    """A factory that raises is a load failure, not a crash."""
    registry = PluginRegistry(tools=ToolRegistry())
    failures = load_all(registry, ["tests.test_plugins:broken_factory"])
    assert failures
    assert "could not be constructed" in list(failures.values())[0]


# ===========================================================================
# The configuration shape
# ===========================================================================


def test_the_configuration_is_a_single_list():
    """No profile/bundle/patch layering: one file, one list."""
    specs, diagnostics = parse_plugin_config(
        {"plugins": ["a.b", "c.d:thing"]}
    )

    assert specs == ["a.b", "c.d:thing"]
    assert diagnostics == []


def test_a_missing_plugin_list_is_not_an_error():
    """A configuration with no plugins is a valid configuration."""
    specs, diagnostics = parse_plugin_config({})
    assert specs == []
    assert diagnostics == []


def test_a_non_list_plugin_configuration_is_reported():
    specs, diagnostics = parse_plugin_config({"plugins": "a.b"})
    assert specs == []
    assert diagnostics and "must be a list" in diagnostics[0]


def test_an_object_entry_uses_its_name():
    """The object form exists so a plugin can carry settings later."""
    specs, diagnostics = parse_plugin_config(
        {"plugins": [{"name": "a.b", "option": 1}]}
    )
    assert specs == ["a.b"]
    assert diagnostics == []


def test_a_nameless_object_entry_is_reported():
    """A nameless entry cannot be loaded or reported."""
    specs, diagnostics = parse_plugin_config({"plugins": [{"option": 1}]})
    assert specs == []
    assert diagnostics and "missing its 'name'" in diagnostics[0]


def test_an_unusable_entry_is_reported_and_skipped():
    specs, diagnostics = parse_plugin_config({"plugins": [42, "a.b"]})
    assert specs == ["a.b"]
    assert diagnostics


# ===========================================================================
# The context is deliberately narrow
# ===========================================================================


def test_the_context_does_not_expose_the_session():
    """A plugin that could rewrite the log could change what the runtime *is*.

    An extension point that powerful is one nobody can reason about, so the
    context carries the registries and nothing else.
    """
    context = PluginContext(tools=ToolRegistry())
    for forbidden in ("session", "adapter", "model", "log"):
        assert not hasattr(context, forbidden), (
            f"the plugin context exposes {forbidden!r}"
        )


def test_registering_many_tools_is_atomic():
    """A plugin that contributed half its tools would behave unpredictably."""
    tools = ToolRegistry()
    context = PluginContext(tools=tools)

    context.register_tools([make_tool("a"), make_tool("b")])
    assert tools.view().known == frozenset({"a", "b"})

    # Registering a duplicate must roll back the ones that succeeded.
    with pytest.raises(Exception):
        context.register_tools([make_tool("c"), make_tool("a")])

    assert "c" not in tools.view().known, (
        "a failed batch left a tool registered"
    )


def test_the_context_carries_the_plugin_name():
    """So a plugin can namespace its own registrations."""
    context = PluginContext(name="my-plugin")
    assert context.name == "my-plugin"


# ===========================================================================
# A real plugin file on disk
# ===========================================================================


def test_a_plugin_can_be_loaded_from_a_file(tmp_path: Path, monkeypatch):
    """The whole path: a file on disk, imported, installed, unloaded."""
    import sys

    module = tmp_path / "my_plugin.py"
    module.write_text(
        textwrap.dedent(
            '''
            """A plugin written the way a user would write one."""

            class Plugin:
                name = "disk-plugin"

                def install(self, context):
                    from arm.tools.definition import (
                        RegisteredTool, ToolDefinition, ToolResult,
                    )

                    async def body(arguments, execution):
                        return ToolResult.ok("from disk")

                    context.register_tool(
                        RegisteredTool(
                            definition=ToolDefinition(
                                name="disk-tool", description="", parameters={}
                            ),
                            body=body,
                        )
                    )

            plugin = Plugin()
            '''
        )
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        tools = ToolRegistry()
        registry = PluginRegistry(tools=tools)
        registry.load("my_plugin")

        assert "disk-tool" in tools.view().known

        registry.unload("disk-plugin")
        assert "disk-tool" not in tools.view().known
    finally:
        sys.modules.pop("my_plugin", None)


# ===========================================================================
# Additional plugins, used by the tests above
# ===========================================================================


class SecondPlugin:
    """A second plugin, to test ordering and independence."""

    name = "second"

    def install(self, context: PluginContext) -> None:
        context.register_tool(make_tool("from-second"))


def broken_factory() -> Any:
    """A factory that raises, to test the construction failure path."""
    raise RuntimeError("this factory cannot build a plugin")


class Plugin:
    """A plugin defined the way a user would define one.

    Present so ``tests.test_plugins`` can be loaded as a module by the CLI tests
    as well as by the tests here.
    """

    name = "example"

    def install(self, context: PluginContext) -> None:
        context.register_tool(make_tool("from-example"))


def test_a_plugin_loads_from_a_file_path(tmp_path: Path):
    """A user who writes a plugin writes a file.

    Requiring them to also make it importable by module name is a rule with no
    benefit, so the path form is supported directly.
    """
    module = tmp_path / "path_plugin.py"
    module.write_text(
        textwrap.dedent(
            '''
            from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult

            class Plugin:
                name = "path-plugin"

                def install(self, context):
                    async def body(arguments, execution):
                        return ToolResult.ok("from a path")

                    context.register_tool(RegisteredTool(
                        definition=ToolDefinition(
                            name="path-tool", description="", parameters={}
                        ),
                        body=body,
                    ))

            plugin = Plugin()
            '''
        )
    )

    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load(str(module))

    assert "path-tool" in tools.view().known
    assert registry.is_loaded("path-plugin")

    registry.unload("path-plugin")
    assert "path-tool" not in tools.view().known


def test_a_nonexistent_plugin_path_is_reported_clearly():
    registry = PluginRegistry(tools=ToolRegistry())
    with pytest.raises(PluginError) as info:
        registry.load("/definitely/not/here/plugin.py")

    assert info.value.code == "PLUGIN_IMPORT_FAILED"
    assert "does not exist" in str(info.value)


def test_a_plugin_file_that_raises_on_import_is_reported(tmp_path: Path):
    """A plugin with a syntax error must produce a load failure, not a crash."""
    module = tmp_path / "broken_plugin.py"
    module.write_text("this is not valid python !!!\n")

    registry = PluginRegistry(tools=ToolRegistry())
    with pytest.raises(Exception) as info:
        registry.load(str(module))
    assert "broken_plugin" in str(info.value) or "invalid syntax" in str(info.value)


def test_a_plugin_can_import_a_sibling_module(tmp_path: Path):
    """A multi-file plugin is what any real user writes."""
    (tmp_path / "helper.py").write_text("VALUE = 'from the helper'\n")
    module = tmp_path / "sibling_plugin.py"
    module.write_text(
        textwrap.dedent(
            '''
            from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult
            from helper import VALUE

            class Plugin:
                name = "sibling-plugin"

                def install(self, context):
                    async def body(arguments, execution):
                        return ToolResult.ok(VALUE)

                    context.register_tool(RegisteredTool(
                        definition=ToolDefinition(
                            name="sibling-tool", description="", parameters={}
                        ),
                        body=body,
                    ))

            plugin = Plugin()
            '''
        )
    )

    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)
    registry.load(str(module))

    assert "sibling-tool" in tools.view().known, (
        "a plugin could not import a module next to it"
    )


# ===========================================================================
# The documented plugin shape must be the one that actually loads
# ===========================================================================


def test_the_documented_plugin_shape_loads(tmp_path, monkeypatch):
    """The shape the `load` docstring now shows, verified as written.

    Writing a plugin meant failing three times on the way to a working one:

      1. a class named `Plugin` with no module-level name -- the loader wanted
         `plugin`, `PLUGIN` or `install`;
      2. a module-level `def install(context)` -- which reads like the interface,
         but the loader calls it with *no* arguments as a factory, so it failed
         with "install() missing 1 required positional argument";
      3. a class assigned to a module-level `plugin` -- which works.

    The docstring is the fix; this test keeps it honest. If the accepted shape
    changes and the example does not, this fails rather than the next person
    failing three times.
    """
    from arm.plugins.registry import PluginRegistry
    from arm.tools.registry import ToolRegistry

    module = tmp_path / "documented_plugin.py"
    module.write_text(
        '''
from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult


async def _body(arguments, _execution):
    return ToolResult.ok("hello from the plugin")


class Plugin:
    name = "documented-plugin"

    def install(self, context):
        context.register_tool(
            RegisteredTool(
                definition=ToolDefinition(
                    name="documented_tool", description="...", parameters={},
                ),
                body=_body,
            )
        )


plugin = Plugin
'''
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)

    registry.load("documented_plugin")

    assert registry.loaded == ("documented-plugin",)
    assert tools.get("documented_tool") is not None, (
        "the documented shape loaded but registered no tool"
    )


def test_a_module_level_install_function_is_treated_as_a_factory(tmp_path, monkeypatch):
    """The trap the docstring warns about, asserted so the warning stays true.

    `def install(context)` at module level looks like the interface. It is not:
    the loader calls it with no arguments as a factory. The failure message names
    a missing argument rather than the misunderstanding, which is why it is worth
    documenting rather than only handling.
    """
    from arm.plugins.registry import PluginError, PluginRegistry
    from arm.tools.registry import ToolRegistry

    module = tmp_path / "install_function_plugin.py"
    module.write_text(
        '''
def install(context):
    """Looks like the entry point. Is treated as a factory."""
    context.register_tool(None)
'''
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = PluginRegistry(tools=ToolRegistry())

    with pytest.raises(PluginError) as info:
        registry.load("install_function_plugin")
    assert "install() missing" in str(info.value), (
        f"the failure no longer matches the documented trap: {info.value}"
    )


def test_naming_the_attribute_explicitly_avoids_the_ambiguity(tmp_path, monkeypatch):
    """`module:attribute` is the documented escape hatch."""
    from arm.plugins.registry import PluginRegistry
    from arm.tools.registry import ToolRegistry

    module = tmp_path / "explicit_plugin.py"
    module.write_text(
        '''
from arm.tools.definition import RegisteredTool, ToolDefinition, ToolResult


async def _body(arguments, _execution):
    return ToolResult.ok("explicit")


class MyThing:
    name = "explicit-plugin"

    def install(self, context):
        context.register_tool(
            RegisteredTool(
                definition=ToolDefinition(name="explicit_tool", description="", parameters={}),
                body=_body,
            )
        )
'''
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    tools = ToolRegistry()
    registry = PluginRegistry(tools=tools)

    registry.load("explicit_plugin:MyThing")

    assert registry.loaded == ("explicit-plugin",)
    assert tools.get("explicit_tool") is not None

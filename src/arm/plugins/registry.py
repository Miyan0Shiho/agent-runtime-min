"""Plugins: one config file, a list of extensions, one load order.

The requirements fix the shape, and the shape is the point:

    **a single configuration file plus a plugin list. No profile/bundle/patch
    layering.**

That is a deliberate refusal, not a simplification. Three-layer composition
(profile → bundle → patch) exists to let one installation serve several
deployments, and it buys that by making "what is actually configured right now"
a question that requires resolving three sources. With one file, that question
has a one-line answer.

So this module implements a **registry with a declared order**, and it enforces
the two properties that make an extension list safe:

**Loading is all-or-nothing per plugin.** A plugin that fails halfway leaves
nothing behind. A partially loaded plugin is worse than an absent one: the
runtime would hold some of its registrations and none of the code that maintains
them.

**Unloading reverses exactly what loading did.** Every registration returns a
disposer, and unload calls them in reverse order. That is what makes the list
editable without a restart, and it is why a plugin cannot leave residue.

A plugin is a Python object with an ``install(context)`` method. It is given a
context exposing the runtime's registries and nothing else -- not the session,
not the adapter. A plugin that could reach the session could rewrite history, and
an extension point that powerful is one nobody can reason about.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..kernel.errors import ArmError, describe

log = logging.getLogger("arm.plugins")


class PluginError(ArmError):
    """A plugin could not be loaded or unloaded."""

    def __init__(self, message: str, *, code: str = "PLUGIN_INVALID") -> None:
        super().__init__(message)
        self.code = code


class Plugin(Protocol):
    """What a plugin must provide.

    One method. A plugin that needed a lifecycle of its own would be a framework,
    and the point of this design is that an extension is a small, removable
    addition.
    """

    name: str

    def install(self, context: "PluginContext") -> None: ...


@dataclass
class PluginContext:
    """What a plugin is allowed to touch.

    Deliberately narrow. The registries are the runtime's extension points; the
    session and the model adapter are not exposed, because a plugin that could
    rewrite the log or swap the model would be able to change what the runtime
    *is* rather than what it can do.
    """

    tools: Any = None
    hooks: Any = None
    prompt: Any = None
    #: Arbitrary key-value state a plugin may use to coordinate with itself.
    state: dict[str, Any] = field(default_factory=dict)
    #: The plugin's own name, so a plugin can namespace its registrations.
    name: str = ""

    def register_tool(self, tool: Any) -> Callable[[], None]:
        """Register a tool, returning a disposer."""
        if self.tools is None:
            raise PluginError("this runtime exposes no tool registry to plugins")
        return self.tools.register(tool)

    def register_tools(self, tools: Sequence[Any]) -> Callable[[], None]:
        """Register several tools atomically.

        If any registration fails, the ones already made are rolled back. A
        plugin that contributed half its tools would look installed and behave
        unpredictably.
        """
        if self.tools is None:
            raise PluginError("this runtime exposes no tool registry to plugins")
        disposers: list[Callable[[], None]] = []
        try:
            for tool in tools:
                disposers.append(self.tools.register(tool))
        except Exception:
            for dispose in reversed(disposers):
                dispose()
            raise

        def dispose_all() -> None:
            for dispose in reversed(disposers):
                dispose()

        return dispose_all


@dataclass
class LoadedPlugin:
    """A plugin that is currently loaded."""

    name: str
    plugin: Any
    disposers: list[Callable[[], None]] = field(default_factory=list)
    #: Where it came from, for a diagnostic.
    source: str = ""

    def unload(self) -> list[str]:
        """Reverse everything the plugin registered.

        Returns the names of disposers that failed. Failures are collected rather
        than raised immediately, so one bad disposer does not leave the rest of
        the plugin's registrations behind.
        """
        failures: list[str] = []
        for dispose in reversed(self.disposers):
            try:
                dispose()
            except Exception as exc:  # noqa: BLE001 - one bad disposer is not fatal
                failures.append(describe(exc))
        self.disposers = []
        return failures


class PluginRegistry:
    """The runtime's extension points, plus the plugins attached to them."""

    def __init__(self, *, tools: Any = None, hooks: Any = None, prompt: Any = None) -> None:
        self.tools = tools
        self.hooks = hooks
        self.prompt = prompt
        self._loaded: dict[str, LoadedPlugin] = {}
        self._order: list[str] = []

    @property
    def loaded(self) -> tuple[str, ...]:
        """Loaded plugin names, in load order."""
        return tuple(self._order)

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def load(self, spec: str) -> LoadedPlugin:
        """Import and install a plugin, by ``module:attribute`` or ``module``.

        A failure leaves nothing behind: the disposers recorded so far are called
        in reverse before the error propagates. A partially loaded plugin would
        look installed and behave unpredictably.
        """
        name, factory = self._resolve(spec)

        if name in self._loaded:
            raise PluginError(
                f"the plugin {name!r} is already loaded", code="PLUGIN_ALREADY_LOADED"
            )

        try:
            plugin = factory()
        except Exception as exc:  # noqa: BLE001 - a factory failure is a load failure
            raise PluginError(
                f"the plugin {name!r} could not be constructed: {describe(exc)}",
                code="PLUGIN_CONSTRUCTION_FAILED",
            ) from exc

        context = PluginContext(
            tools=self.tools, hooks=self.hooks, prompt=self.prompt, name=name
        )
        # A plugin may register through the context, which returns disposers, or
        # it may return them from install(). Both are supported because both are
        # natural to write, and forcing one would be a style rule rather than a
        # design decision.
        collector = _DisposerCollector(context)
        try:
            result = plugin.install(collector.context)
        except Exception as exc:  # noqa: BLE001
            collector.rollback()
            raise PluginError(
                f"the plugin {name!r} failed to install: {describe(exc)}",
                code="PLUGIN_INSTALL_FAILED",
            ) from exc

        if isinstance(result, (list, tuple)):
            for dispose in result:
                if callable(dispose):
                    # Wrapped so a disposer that returns something other than
                    # None is still accepted; a teardown function's return value
                    # is not meaningful, and refusing it would be pedantic.
                    collector.record(_as_disposer(dispose))

        loaded = LoadedPlugin(
            name=name, plugin=plugin, disposers=collector.disposers, source=spec
        )
        self._loaded[name] = loaded
        self._order.append(name)
        log.info("loaded plugin %s", name)
        return loaded

    def _resolve(self, spec: str) -> tuple[str, Callable[[], Any]]:
        """Turn a spec into a name and a factory.

        ``module:attribute`` names an attribute; a bare ``module`` looks for
        ``plugin``, then ``PLUGIN``, then ``install``. A clear error listing what
        was tried is worth more than a clever convention.
        """
        if ":" in spec:
            module_name, _, attribute = spec.partition(":")
        else:
            module_name, attribute = spec, ""

        module = None
        if module_name.endswith(".py") or "/" in module_name or "\\" in module_name:
            # A path. A user who writes a plugin writes a file, and requiring
            # them to also make it importable by module name is a rule with no
            # benefit -- so the path form is supported directly.
            module = _import_from_path(module_name)
            if module is None:
                raise PluginError(
                    f"could not load a plugin from {module_name!r}: the file does "
                    "not exist or is not readable",
                    code="PLUGIN_IMPORT_FAILED",
                )
        else:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                raise PluginError(
                    f"could not import {module_name!r}: {describe(exc)}",
                    code="PLUGIN_IMPORT_FAILED",
                ) from exc

        if attribute:
            candidate = getattr(module, attribute, None)
            if candidate is None:
                raise PluginError(
                    f"{module_name!r} has no attribute {attribute!r}",
                    code="PLUGIN_NOT_FOUND",
                )
        else:
            candidate = None
            for name in ("plugin", "PLUGIN", "install"):
                candidate = getattr(module, name, None)
                if candidate is not None:
                    break
            if candidate is None:
                raise PluginError(
                    f"{module_name!r} defines none of 'plugin', 'PLUGIN' or "
                    "'install'; name one explicitly as module:attribute",
                    code="PLUGIN_NOT_FOUND",
                )

        name = getattr(candidate, "name", None) or module_name.rsplit(".", 1)[-1]

        if callable(candidate) and not hasattr(candidate, "install"):
            # A factory function: call it to get the plugin.
            return str(name), candidate

        # An instance or class with install().
        if isinstance(candidate, type):
            return str(name), candidate
        return str(name), lambda: candidate

    def unload(self, name: str) -> list[str]:
        """Unload a plugin, reversing exactly what it registered."""
        loaded = self._loaded.pop(name, None)
        if loaded is None:
            raise PluginError(f"the plugin {name!r} is not loaded", code="PLUGIN_NOT_LOADED")
        self._order.remove(name)
        failures = loaded.unload()
        log.info("unloaded plugin %s", name)
        return failures

    def unload_all(self) -> dict[str, list[str]]:
        """Unload every plugin, newest first.

        Newest first because a plugin loaded later may depend on one loaded
        earlier; reversing the load order is the only order that is safe without
        dependency tracking, and dependency tracking is a framework.
        """
        failures: dict[str, list[str]] = {}
        for name in reversed(list(self._order)):
            try:
                problems = self.unload(name)
            except PluginError:
                continue
            if problems:
                failures[name] = problems
        return failures

    def reload(self, name: str) -> LoadedPlugin:
        """Unload and load again, keeping the plugin's position in the order."""
        loaded = self._loaded.get(name)
        if loaded is None:
            raise PluginError(f"the plugin {name!r} is not loaded", code="PLUGIN_NOT_LOADED")
        spec = loaded.source
        position = self._order.index(name)
        self.unload(name)
        fresh = self.load(spec)
        # Restore the position, so reloading does not reorder the list. A changed
        # order would make reload a way to accidentally change precedence.
        self._order.remove(name)
        self._order.insert(position, name)
        return fresh


def _as_disposer(candidate: Any) -> Callable[[], None]:
    """Adapt a callable into a no-argument disposer."""

    def dispose() -> None:
        candidate()

    return dispose


class _DisposerCollector:
    """Wraps a context so every registration's disposer is captured.

    A plugin writes ``ctx.register_tool(...)`` and does not have to think about
    teardown. That is the difference between an extension point people use
    correctly and one they use correctly by remembering to.
    """

    def __init__(self, context: PluginContext) -> None:
        self.disposers: list[Callable[[], None]] = []
        self.context = PluginContext(
            tools=_TrackingRegistry(context.tools, self, name=context.name),
            hooks=context.hooks,
            prompt=context.prompt,
            state=context.state,
            name=context.name,
        )

    def record(self, dispose: Callable[[], None]) -> None:
        self.disposers.append(dispose)

    def rollback(self) -> None:
        for dispose in reversed(self.disposers):
            try:
                dispose()
            except Exception as exc:  # noqa: BLE001
                log.warning("rolling back a plugin registration failed: %s", exc)
        self.disposers = []


class _TrackingRegistry:
    """A proxy that records every registration's disposer.

    Deliberately a proxy rather than a subclass: it must forward everything the
    real registry offers, and a subclass would silently miss a method added
    later.
    """

    def __init__(self, inner: Any, collector: _DisposerCollector, *, name: str = "") -> None:
        self._inner = inner
        self._collector = collector
        self._name = name

    def _require(self) -> Any:
        """The real registry, or a clear error naming what is missing.

        A plugin that hits ``NoneType has no attribute 'register'`` learns
        nothing; the message has to say that this runtime exposes no tool
        registry, because that is the fact the plugin author needs.
        """
        if self._inner is None:
            raise PluginError(
                f"the plugin {self._name!r} tried to register a tool, but this "
                "runtime exposes no tool registry to plugins"
            )
        return self._inner

    def __getattr__(self, item: str) -> Any:
        return getattr(self._require(), item)

    def register(self, tool: Any) -> Callable[[], None]:
        dispose = self._require().register(tool)
        self._collector.record(dispose)
        return dispose

    def register_many(self, tools: Sequence[Any]) -> Callable[[], None]:
        dispose = self._require().register_many(tools)
        self._collector.record(dispose)
        return dispose


def _import_from_path(path_text: str) -> Any:
    """Import a plugin from a file path.

    The module is registered in ``sys.modules`` under a name derived from the
    path, because a plugin that imports a sibling module relies on normal package
    semantics -- and a module that is not in ``sys.modules`` breaks that.

    The file's own directory is put on the path so a plugin can import something
    next to it, which is what any user writing a multi-file plugin will do.
    """
    import sys
    from pathlib import Path

    path = Path(path_text).expanduser()
    if not path.is_file():
        return None
    if path.suffix != ".py":
        return None

    directory = str(path.parent.resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)

    name = f"arm_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - a plugin that fails to execute is a load failure
        sys.modules.pop(name, None)
        raise
    return module


def load_all(registry: PluginRegistry, specs: Sequence[str]) -> dict[str, str]:
    """Load every plugin in a list, in order.

    Returns the failures rather than raising: one plugin failing must not prevent
    the others from loading, because the alternative is that a single bad entry in
    a configuration makes the agent unusable. The caller decides what to do with
    the list.
    """
    failures: dict[str, str] = {}
    for spec in specs:
        try:
            registry.load(spec)
        except PluginError as exc:
            failures[spec] = str(exc)
            log.warning("plugin %s did not load: %s", spec, exc)
    return failures


def parse_plugin_config(raw: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Read the plugin list from the configuration.

    A single list, with no layering. Diagnostics are returned rather than raised,
    for the same reason: a malformed entry must not prevent the others.
    """
    diagnostics: list[str] = []
    plugins = raw.get("plugins")
    if plugins is None:
        return [], []
    if not isinstance(plugins, Sequence) or isinstance(plugins, (str, bytes)):
        return [], ["'plugins' must be a list of module names"]

    specs: list[str] = []
    for entry in plugins:
        if isinstance(entry, str) and entry.strip():
            specs.append(entry.strip())
        elif isinstance(entry, Mapping):
            # An object form is accepted so a plugin can carry settings later,
            # but `name` is required: a nameless entry cannot be loaded or
            # reported.
            name = str(entry.get("name", "")).strip()
            if name:
                specs.append(name)
            else:
                diagnostics.append("a plugin entry is missing its 'name'")
        else:
            diagnostics.append(f"ignoring an unusable plugin entry: {entry!r}")
    return specs, diagnostics

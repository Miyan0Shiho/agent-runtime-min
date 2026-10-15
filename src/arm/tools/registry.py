"""The tool registry: one traversal, three answers, reversible registration.

The registry answers three questions the model-facing layer needs:

* which tools are **visible** (what goes in the request),
* which names are **known** (what may be called),
* which names are **filterable** (what a restriction may refer to).

These three must come from a *single* traversal. Implementing them separately
means three places must agree about shadowing and inheritance, and three
implementations of one rule always drift.

Registration is reversible: every ``register`` returns a disposer, so unloading
a plugin leaves no residue. Restrictions are checked eagerly and loudly --
an allow-list naming a tool that does not exist is a configuration bug that would
otherwise fail silently at call time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..kernel.errors import ArmError
from .definition import RESERVED_TOOL_NAMES, RegisteredTool, ToolDefinition

log = logging.getLogger("arm.tools.registry")


class ToolRegistryError(ArmError):
    """Base class for registry misuse."""


class DuplicateToolError(ToolRegistryError):
    """A name was registered twice in the same layer."""


class UnknownToolRestrictionError(ToolRegistryError):
    """A restriction named a tool that does not exist."""


class EmptyRestrictionError(ToolRegistryError):
    """A restriction was given no filter at all.

    An empty filter is almost always a bug that materialised an empty config, and
    the consequence would be "no tools are callable", which looks like a model
    failure rather than a configuration error.
    """


@dataclass(frozen=True, slots=True)
class ToolView:
    """The result of one traversal: three consistent answers."""

    visible: tuple[ToolDefinition, ...]
    known: frozenset[str]
    filterable: frozenset[str]

    def get(self, name: str) -> ToolDefinition | None:
        for definition in self.visible:
            if definition.name == name:
                return definition
        return None


@dataclass
class _Layer:
    """One registration layer.

    Layers exist so a sub-agent can restrict what it inherits without editing the
    parent's registry. Restrictions apply to the *inherited* surface only; a
    layer's own registrations are permanently exempt, which is what lets a
    sub-agent keep the tools it needs to report back.
    """

    name: str
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    #: Names this layer may not use, even if inherited.
    denied: frozenset[str] = frozenset()
    #: When set, only these inherited names are usable.
    allowed: frozenset[str] | None = None


class ToolRegistry:
    """A layered, reversible tool registry.

    The root layer is always consulted; child layers add or restrict. A name in a
    child layer shadows the same name in a parent, which is how a sub-agent can
    substitute a narrower version of a capability.
    """

    def __init__(self, *, name: str = "root") -> None:
        self._layers: list[_Layer] = [_Layer(name=name)]

    # -- registration ----------------------------------------------------

    def register(self, tool: RegisteredTool) -> callable:  # type: ignore[valid-type]
        """Register a tool in the current layer. Returns a disposer."""
        return self.register_many([tool])

    def register_many(self, tools: Sequence[RegisteredTool]) -> callable:  # type: ignore[valid-type]
        """Register several tools atomically. Returns one disposer for all."""
        layer = self._layers[-1]
        added: list[str] = []

        for tool in tools:
            name = tool.name
            if name in RESERVED_TOOL_NAMES:
                raise ToolRegistryError(
                    f"tool name {name!r} is reserved and cannot be registered"
                )
            if name in layer.tools:
                raise DuplicateToolError(
                    f"tool {name!r} is already registered in layer {layer.name!r}"
                )
            layer.tools[name] = tool
            added.append(name)

        def _dispose() -> None:
            for name in added:
                if layer.tools.get(name) is not None:
                    del layer.tools[name]

        return _dispose

    def register_definition(self, definition: ToolDefinition, body) -> callable:  # type: ignore[valid-type]
        return self.register(RegisteredTool(definition=definition, body=body))

    def adopt_from(self, source: "ToolRegistry") -> None:
        """Copy every tool from ``source`` into this registry's root layer.

        A **copy**, not a shared reference. A child registry must not be able to
        affect the parent's, and the parent must not be able to affect a child
        that already started -- which is what makes the capability set a child
        receives meaningful rather than "whatever the parent has by the time the
        tool is called".

        The copy is by tool *object*: the registered tools are shared, which is
        correct because a tool definition is immutable. What is copied is the
        membership, and membership is the thing that changes.
        """
        root = self._layers[0]
        for current in source._layers:
            for name, tool in current.tools.items():
                # Later layers win, matching the source's own resolution order.
                root.tools[name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool from every layer. Returns whether one was removed.

        Removes from all layers rather than the top one: a shadowed tool that
        stays in an inner layer reappears as soon as the shadowing layer is
        popped, which would make "remove" mean "hide for now".
        """
        removed = False
        for layer in self._layers:
            if name in layer.tools:
                del layer.tools[name]
                removed = True
        return removed

    def replace(self, tool: "RegisteredTool") -> Callable[[], None]:
        """Register a tool, replacing any existing one with the same name.

        Unlike ``register`` this does not raise on a duplicate: the caller is
        deliberately superseding a definition it inherited, which is exactly what
        a child agent does when it needs its own delegation tool at its own depth.
        """
        name = tool.definition.name
        self.unregister(name)
        return self.register(tool)

    # -- traversal -------------------------------------------------------

    def view(self, *, layer: int | None = None) -> ToolView:
        """Single traversal producing all three answers (M3.1).

        Walk from the root down so that later layers shadow earlier ones, and
        apply each layer's restriction to what it inherited -- never to what it
        registered itself.
        """
        index = len(self._layers) - 1 if layer is None else layer
        # Later layers win, so iterate from the root upward and overwrite.
        merged: dict[str, RegisteredTool] = {}
        own_names: set[str] = set()

        for position, current in enumerate(self._layers[: index + 1]):
            for name, tool in current.tools.items():
                merged[name] = tool
                if position == index:
                    own_names.add(name)

        visible: list[ToolDefinition] = []
        known: set[str] = set()
        filterable: set[str] = set()
        active = self._layers[index]

        for name, tool in merged.items():
            # A layer's own registrations are permanently exempt from its own
            # restrictions (M3.2): otherwise a sub-agent could restrict away the
            # very tools it registered to report results.
            if name in own_names:
                known.add(name)
                filterable.add(name)
                if not tool.definition.hidden:
                    visible.append(tool.definition)
                continue

            if name in active.denied:
                continue
            if active.allowed is not None and name not in active.allowed:
                continue

            known.add(name)
            filterable.add(name)
            if not tool.definition.hidden:
                visible.append(tool.definition)

        return ToolView(
            visible=tuple(visible),
            known=frozenset(known),
            filterable=frozenset(filterable),
        )

    def get(self, name: str, *, layer: int | None = None) -> RegisteredTool | None:
        """Fetch a registered tool by name, respecting restrictions."""
        view = self.view(layer=layer)
        if name not in view.known:
            return None
        index = len(self._layers) - 1 if layer is None else layer
        for current in reversed(self._layers[: index + 1]):
            if name in current.tools:
                return current.tools[name]
        return None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.view().known))

    # -- restriction -----------------------------------------------------

    def restrict(
        self,
        *,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        layer_name: str = "restricted",
    ) -> int:
        """Push a restricted child layer and return its index.

        ``allow`` and ``deny`` are validated against the tools that exist *now*
        (M3.14), because an allow-list entry that names nothing is a silent
        configuration failure. A tool registered later may legitimately be added
        to a restriction afterwards.
        """
        if allow is None and deny is None:
            raise EmptyRestrictionError(
                "a restriction must name at least one tool to allow or deny"
            )

        allow_set = frozenset(allow) if allow is not None else None
        deny_set = frozenset(deny) if deny is not None else frozenset()

        if allow_set is not None and not allow_set:
            raise EmptyRestrictionError(
                "an allow-list must not be empty; an empty list would make every "
                "tool uncallable"
            )

        available = self.view().filterable
        named = (allow_set or frozenset()) | deny_set
        unknown = sorted(named - available)
        if unknown:
            raise UnknownToolRestrictionError(
                f"restriction names unknown tools {unknown}; known tools are "
                f"{sorted(available)}"
            )

        self._layers.append(
            _Layer(name=layer_name, allowed=allow_set, denied=deny_set)
        )
        return len(self._layers) - 1

    def pop_layer(self) -> None:
        """Remove the newest layer. The root layer cannot be removed."""
        if len(self._layers) == 1:
            raise ToolRegistryError("cannot remove the root layer")
        self._layers.pop()

    @property
    def depth(self) -> int:
        return len(self._layers)

    def __len__(self) -> int:
        return len(self.view().visible)

"""Tool definitions: what a tool is, independent of how it runs.

A capability is split into three roles living at different rates of change:

    definition  -- what the capability *is* (slowest: changing it changes the
                   model-facing contract)
    provider    -- how it actually runs (medium: swapping a sandbox backend)
    consumer    -- what the model programs against (fastest)

The split exists so that replacing an execution backend cannot disturb the
schema the model sees. This module holds the first role.

**Discipline: do not split pre-emptively.** With exactly one plausible provider
and one consumer, a single module is correct. Splitting costs wiring, packaging
and indirection, and buys nothing until a second implementation actually exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from ..kernel.errors import ReservedToolNameError
from ..kernel.freeze import FrozenDict, assert_json_lossless, deep_freeze
from ..session.messages import Block, Message

#: Text handed to the model when a tool reports a failure. Kept in one place so
#: every failure path produces a recognisable, greppable shape.
ERROR_PREFIX = "Error: "

#: Names the runtime reserves. Registering one is refused rather than shadowed:
#: two tools with one name would make the advertised set differ from the
#: callable set.
RESERVED_TOOL_NAMES = frozenset(
    {
        "system",
        "assistant",
        "user",
        "tool",
        "error",
        "runtime",
    }
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of one tool call.

    Exactly one arm is populated. A failure carries no ``value`` -- keeping that
    field absent rather than ``None`` means "failed" and "succeeded with nothing"
    cannot be confused.

    ``content`` is what the model sees. ``value`` is execution-local data that
    never reaches the model and never reaches the log.
    """

    content: str = ""
    is_error: bool = False
    value: Any = None
    #: Structured error identity. Persisted for diagnosis, never shown to the
    #: model -- internal details leaking into the transcript is its own bug.
    error_code: str | None = None
    error_info: Mapping[str, Any] = field(default_factory=lambda: FrozenDict())
    #: Extra context the tool wants to contribute to the conversation.
    additional_context: tuple[str, ...] = ()
    #: When true, this result ends the turn instead of continuing the loop. This
    #: is the *only* successful path that interrupts a step.
    concludes_turn: bool = False
    #: Blocks to place on the wire. Defaults to a single text block of content.
    blocks: tuple[Block, ...] = ()

    def __post_init__(self) -> None:
        if self.is_error and self.value is not None:
            raise ValueError("a failed result must not carry a value")
        if self.error_info and not isinstance(self.error_info, FrozenDict):
            assert_json_lossless(dict(self.error_info), what="error_info")
            object.__setattr__(self, "error_info", deep_freeze(dict(self.error_info)))
        if not self.blocks and self.content:
            object.__setattr__(self, "blocks", (Block.text_block(self.content),))

    @staticmethod
    def ok(
        content: str,
        *,
        value: Any = None,
        additional_context: Sequence[str] = (),
        concludes_turn: bool = False,
    ) -> "ToolResult":
        return ToolResult(
            content=content,
            value=value,
            additional_context=tuple(additional_context),
            concludes_turn=concludes_turn,
        )

    @staticmethod
    def fail(
        message: str,
        *,
        code: str | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        """Build a failure result, prefixed so the model can recognise it."""
        text = message if message.startswith(ERROR_PREFIX) else ERROR_PREFIX + message
        return ToolResult(
            content=text,
            is_error=True,
            error_code=code,
            error_info=info or {},
        )

    def to_message(self, *, call_id: str, name: str | None = None) -> Message:
        """Render as the ``tool``-role message that goes on the wire."""
        return Message(
            role="tool",
            content=self.blocks or (Block.text_block(self.content),),
            tool_call_id=call_id,
            name=name,
        )


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """What a tool declares about itself.

    ``parameters`` is a JSON Schema object produced from a pydantic model, so the
    validation the model is promised and the validation that actually runs come
    from the same declaration.

    ``concurrency_safe`` is fail-closed: only an explicit ``True`` licenses
    parallel dispatch. A new tool that forgot to think about concurrency is
    therefore treated as exclusive, which is the safe default.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    #: Called with the parsed arguments; returns True only if parallel is safe.
    concurrency_safe: Callable[[Mapping[str, Any]], bool] | None = None
    #: Whether this tool may be exposed to the model at all.
    hidden: bool = False
    #: Arbitrary metadata for policy layers. Never sent to the model.
    meta: Mapping[str, Any] = field(default_factory=lambda: FrozenDict())

    #: Whether a call to this tool needs explicit approval.
    #:
    #: Declared by the tool rather than configured centrally, because the tool is
    #: the thing that knows what it does. A central list would drift: every new
    #: tool would need someone to remember to add it, and the failure is silent --
    #: a dangerous tool simply runs.
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if self.name in RESERVED_TOOL_NAMES:
            raise ReservedToolNameError(f"tool name {self.name!r} is reserved")

        # Normalization happens here, at the one place every tool definition must
        # pass through, rather than at each construction site.
        #
        # This is deliberately not the caller's job. When normalization lived in a
        # helper that callers had to remember to use, it was bypassed by every
        # real tool: the built-ins shipped raw pydantic schemas, so keywords the
        # model-facing subset rejects reached the wire and the normalizer's
        # correctness was a property of a code path nothing executed.
        if not isinstance(self.parameters, FrozenDict):
            assert_json_lossless(dict(self.parameters), what=f"{self.name} parameters")
            from .schema import normalize_schema

            normalized = normalize_schema(dict(self.parameters))
            object.__setattr__(self, "parameters", deep_freeze(dict(normalized)))

    def allows_parallel(self, arguments: Mapping[str, Any]) -> bool:
        """Whether this specific call may run concurrently. Fail-closed."""
        if self.concurrency_safe is None:
            return False
        try:
            return self.concurrency_safe(arguments) is True
        except Exception:  # noqa: BLE001 - a broken predicate must not enable parallel
            return False


#: A tool body. Receives validated arguments and a context, returns a result.
ToolBody = Callable[[Mapping[str, Any], Any], Awaitable[ToolResult]]


class ParameterModel(Protocol):
    """What a pydantic model must offer to serve as a parameter schema."""

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls, obj: Any) -> Any: ...


def tool(
    *,
    name: str,
    description: str,
    parameters: type[ParameterModel] | Mapping[str, Any],
    concurrency_safe: Callable[[Mapping[str, Any]], bool] | None = None,
    hidden: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> Callable[[ToolBody], "RegisteredTool"]:
    """Decorator turning an async function into a :class:`RegisteredTool`."""

    def _wrap(body: ToolBody) -> "RegisteredTool":
        schema: Mapping[str, Any]
        if isinstance(parameters, Mapping):
            schema = parameters
        else:
            schema = parameters.model_json_schema()
        return RegisteredTool(
            definition=ToolDefinition(
                name=name,
                description=description,
                parameters=schema,
                concurrency_safe=concurrency_safe,
                hidden=hidden,
                meta=meta or {},
            ),
            body=body,
            validator=None if isinstance(parameters, Mapping) else parameters,
        )

    return _wrap


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """A definition bound to its implementation."""

    definition: ToolDefinition
    body: ToolBody
    #: The pydantic model used to validate arguments, when one was supplied.
    validator: type[ParameterModel] | None = None

    @property
    def name(self) -> str:
        return self.definition.name

    def validate_arguments(self, raw: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
        """Validate and normalize arguments.

        Returns ``(arguments, None)`` on success or ``({}, message)`` on failure.
        Failure is reported rather than raised: an invalid argument list is an
        ordinary outcome that becomes an error *result*, not an exception that
        unwinds the step.
        """
        if self.validator is None:
            return dict(raw), None
        try:
            model = self.validator.model_validate(dict(raw))
        except Exception as exc:  # noqa: BLE001 - pydantic raises many types
            return {}, _format_violations(exc)
        dumped = model.model_dump(mode="json") if hasattr(model, "model_dump") else dict(raw)
        return dict(dumped), None


def _format_violations(exc: BaseException) -> str:
    """Render pydantic's errors as a compact, model-readable list."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts: list[str] = []
            for item in errors():  # type: ignore[union-attr]
                location = ".".join(str(p) for p in item.get("loc", ())) or "(root)"
                parts.append(f"{location}: {item.get('msg', 'invalid')}")
            if parts:
                return "; ".join(parts)
        except Exception:  # noqa: BLE001 - fall back to str()
            pass
    from ..kernel.errors import describe

    return describe(exc)


# NOTE: there is deliberately no ``schema_of`` helper here.
#
# One existed, and it was dead code: normalization lived in a helper that callers
# had to remember to invoke, so every real tool bypassed it and shipped a raw
# pydantic schema. Normalization now happens inside ``ToolDefinition``, which is
# the one place a tool definition must pass through -- a guarantee that cannot be
# forgotten by a caller.

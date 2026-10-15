"""Model-visible messages: the derived projection's output type.

A :class:`Message` is what actually goes on the wire. It is always frozen,
because :meth:`Session.derive` hands out a fresh list that shares the *same*
message objects -- so if a caller could mutate one, it would rewrite history for
every other reader.

Content is a tuple of blocks rather than a bare string. That keeps the shape
uniform across text and images, and makes "this block is not text" explicit
instead of implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..kernel.freeze import FrozenDict, assert_json_lossless, deep_freeze, thaw

BlockKind = Literal["text", "image", "tool-result"]


@dataclass(frozen=True, slots=True)
class Block:
    """One content block.

    ``text`` blocks carry ``text``; ``image`` blocks carry an opaque ``data``
    reference. Non-text blocks are never sliced by the pruner, which is why the
    pruner counts them as zero-length instead of trying to measure them.
    """

    kind: BlockKind
    text: str = ""
    data: Mapping[str, Any] = field(default_factory=lambda: FrozenDict())

    def __post_init__(self) -> None:
        if self.kind == "text":
            if not isinstance(self.text, str):
                raise TypeError("text block must carry a str")
        else:
            raw = self.data
            if not isinstance(raw, FrozenDict):
                assert_json_lossless(dict(raw), what=f"{self.kind} block data")
                object.__setattr__(self, "data", deep_freeze(dict(raw)))

    @staticmethod
    def text_block(text: str) -> "Block":
        return Block(kind="text", text=text)

    @staticmethod
    def image(data: Mapping[str, Any]) -> "Block":
        return Block(kind="image", data=data)

    @property
    def length(self) -> int:
        """Characters counted for pruning. Non-text blocks count as zero."""
        return len(self.text) if self.kind == "text" else 0

    def to_record(self) -> dict[str, Any]:
        if self.kind == "text":
            return {"type": "text", "text": self.text}
        return {"type": self.kind, **thaw(self.data)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Block":
        kind = str(record.get("type", "text"))
        if kind == "text":
            return cls(kind="text", text=str(record.get("text", "")))
        payload = {k: v for k, v in record.items() if k != "type"}
        return cls(kind="image", data=payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Message:
    """A single model-visible message.

    Exactly one of ``tool_call_id`` / ``tool_calls`` is typically set; plain
    messages carry neither.
    """

    role: str
    content: tuple[Block, ...] = ()
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    #: Set on an assistant message whose stream was cut short after content had
    #: already been produced. The partial content is kept rather than dropped.
    interrupted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("message role must be a non-empty string")
        if not isinstance(self.content, tuple):
            object.__setattr__(self, "content", tuple(self.content))
        if self.tool_calls and not isinstance(self.tool_calls[0], FrozenDict):
            frozen = tuple(deep_freeze(dict(call)) for call in self.tool_calls)
            object.__setattr__(self, "tool_calls", frozen)

    # -- content helpers -------------------------------------------------

    @property
    def text(self) -> str:
        """All text blocks joined. Convenience for checks and summaries."""
        return "".join(b.text for b in self.content if b.kind == "text")

    @property
    def char_length(self) -> int:
        return sum(b.length for b in self.content)

    @property
    def is_empty(self) -> bool:
        return not self.content

    @staticmethod
    def system(text: str) -> "Message":
        return Message(role="system", content=(Block.text_block(text),))

    @staticmethod
    def user(text: str) -> "Message":
        return Message(role="user", content=(Block.text_block(text),))

    @staticmethod
    def developer(text: str) -> "Message":
        return Message(role="developer", content=(Block.text_block(text),))

    @staticmethod
    def assistant(
        text: str = "",
        *,
        tool_calls: Sequence[Mapping[str, Any]] = (),
        interrupted: bool = False,
    ) -> "Message":
        blocks = (Block.text_block(text),) if text else ()
        return Message(
            role="assistant",
            content=blocks,
            tool_calls=tuple(tool_calls),
            interrupted=interrupted,
        )

    @staticmethod
    def tool_result(tool_call_id: str, text: str, *, name: str | None = None) -> "Message":
        return Message(
            role="tool",
            content=(Block.text_block(text),),
            tool_call_id=tool_call_id,
            name=name,
        )

    # -- serialization ---------------------------------------------------

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "role": self.role,
            "content": [b.to_record() for b in self.content],
        }
        if self.name is not None:
            record["name"] = self.name
        if self.tool_call_id is not None:
            record["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            record["tool_calls"] = [thaw(c) for c in self.tool_calls]
        if self.interrupted:
            record["interrupted"] = True
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Message":
        content = record.get("content") or []
        blocks: Iterable[Block]
        if isinstance(content, str):
            blocks = (Block.text_block(content),) if content else ()
        else:
            blocks = tuple(Block.from_record(b) for b in content)
        return cls(
            role=str(record.get("role", "user")),
            content=tuple(blocks),
            name=record.get("name"),
            tool_call_id=record.get("tool_call_id"),
            tool_calls=tuple(record.get("tool_calls") or ()),
            interrupted=bool(record.get("interrupted", False)),
        )

"""Core message types used by the agent execution loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class MessageRole(StrEnum):
    """Roles supported by the agent's internal conversation history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy a mapping so callers cannot mutate a frozen value from outside."""

    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation requested by the model.

    Attributes:
        id: Provider-generated identifier used to match the eventual tool result.
        name: Name registered in the local tool registry.
        arguments: Parsed JSON arguments supplied by the model.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("tool call id must not be empty")
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")

        object.__setattr__(self, "arguments", _immutable_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class Message:
    """One normalized message in the agent conversation history.

    `tool_calls` is valid only for assistant messages. A tool message must carry
    `tool_call_id` so that it can be paired with the originating `ToolCall`.
    """

    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("message content must be a string or None")

        normalized_calls = tuple(self.tool_calls)
        if any(not isinstance(call, ToolCall) for call in normalized_calls):
            raise TypeError("tool_calls must contain only ToolCall values")
        object.__setattr__(self, "tool_calls", normalized_calls)

        if normalized_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError("only assistant messages may contain tool calls")

        if self.role is MessageRole.TOOL:
            if self.tool_call_id is None or not self.tool_call_id.strip():
                raise ValueError("tool messages require a non-empty tool_call_id")
            if self.content is None:
                raise ValueError("tool messages require content")
        elif self.tool_call_id is not None:
            raise ValueError("only tool messages may contain tool_call_id")

        if self.role in {MessageRole.SYSTEM, MessageRole.USER} and self.content is None:
            raise ValueError("system and user messages require content")

        if (
            self.role is MessageRole.ASSISTANT
            and self.content is None
            and not normalized_calls
        ):
            raise ValueError("assistant messages require content or tool calls")


"""Provider-independent model client boundary."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from coding_agent.messages import Message


ToolSchema = Mapping[str, Any]


class ModelClientError(RuntimeError):
    """Base class for errors exposed by a model client adapter."""


class ModelRequestError(ModelClientError):
    """Raised when the model request could not be completed."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ModelResponseError(ModelClientError):
    """Raised when a provider returns an unusable response."""


class LLMClient(Protocol):
    """Minimal interface consumed by the future Agent execution loop."""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] = (),
    ) -> Message:
        """Return one normalized assistant message for the supplied history."""


"""Core result type shared by all local tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Contract implemented by every local tool exposed to the model."""

    name: str
    description: str
    parameters: Mapping[str, Any]

    def execute(self, arguments: Mapping[str, Any]) -> "ToolResult":
        """Execute validated arguments and return a normalized result."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalized outcome of a local tool execution.

    A successful result cannot contain an error. A failed result must explain
    the failure, while `output` may still carry useful partial stdout or context.
    """

    success: bool
    output: str = ""
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if not isinstance(self.output, str):
            raise TypeError("output must be a string")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        if self.success and self.error is not None:
            raise ValueError("successful tool results cannot contain an error")
        if not self.success and (self.error is None or not self.error.strip()):
            raise ValueError("failed tool results require a non-empty error")

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def ok(
        cls,
        output: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        """Build a successful result."""

        return cls(success=True, output=output, metadata=metadata or {})

    @classmethod
    def fail(
        cls,
        error: str,
        *,
        output: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        """Build a failed result with optional partial output."""

        return cls(
            success=False,
            output=output,
            error=error,
            metadata=metadata or {},
        )

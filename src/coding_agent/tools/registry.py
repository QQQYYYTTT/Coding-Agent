"""Tool registration, schema export, argument validation, and dispatch."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping, Sequence

from coding_agent.messages import ToolCall
from coding_agent.tools.base import Tool, ToolResult


_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


class ToolRegistrationError(ValueError):
    """Raised when a tool cannot be safely registered."""


class ToolArgumentError(ValueError):
    """Raised internally when arguments do not match a tool schema."""


def _snapshot_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Validate JSON serializability and detach the public schema from a tool."""

    try:
        snapshot = json.loads(json.dumps(dict(parameters), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ToolRegistrationError("tool parameters must be JSON serializable") from exc
    if not isinstance(snapshot, dict) or snapshot.get("type") != "object":
        raise ToolRegistrationError("tool parameters must be an object JSON Schema")
    properties = snapshot.get("properties", {})
    if not isinstance(properties, dict):
        raise ToolRegistrationError("tool schema properties must be an object")
    required = snapshot.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(name, str) for name in required
    ):
        raise ToolRegistrationError("tool schema required must be a list of names")
    unknown_required = set(required) - set(properties)
    if unknown_required:
        names = ", ".join(sorted(unknown_required))
        raise ToolRegistrationError(
            f"required tool parameters are not declared in properties: {names}"
        )
    return snapshot


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ToolArgumentError(f"unsupported schema type: {expected}")


def _validate_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    """Validate the JSON Schema subset needed by local tool arguments."""

    expected_type = schema.get("type")
    if not isinstance(expected_type, str):
        raise ToolArgumentError(f"{path}: schema must declare one type")
    if not _matches_type(value, expected_type):
        raise ToolArgumentError(f"{path}: expected {expected_type}")

    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list):
            raise ToolArgumentError(f"{path}: schema enum must be a list")
        if value not in enum:
            raise ToolArgumentError(f"{path}: value is not in the allowed set")

    if expected_type == "object":
        assert isinstance(value, Mapping)
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise ToolArgumentError(f"{path}: invalid object schema")
        missing = [name for name in required if name not in value]
        if missing:
            raise ToolArgumentError(
                f"{path}: missing required parameter(s): {', '.join(missing)}"
            )
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ToolArgumentError(
                    f"{path}: unexpected parameter(s): {', '.join(extras)}"
                )
        for name, child_value in value.items():
            child_schema = properties.get(name)
            if child_schema is None:
                continue
            if not isinstance(child_schema, Mapping):
                raise ToolArgumentError(f"{path}.{name}: invalid property schema")
            _validate_value(child_value, child_schema, f"{path}.{name}")

    elif expected_type == "array":
        assert isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        item_schema = schema.get("items")
        if item_schema is not None:
            if not isinstance(item_schema, Mapping):
                raise ToolArgumentError(f"{path}: invalid item schema")
            for index, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{index}]")

    elif expected_type == "string":
        assert isinstance(value, str)
        minimum_length = schema.get("minLength")
        maximum_length = schema.get("maxLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            raise ToolArgumentError(f"{path}: string is too short")
        if isinstance(maximum_length, int) and len(value) > maximum_length:
            raise ToolArgumentError(f"{path}: string is too long")

    elif expected_type in {"integer", "number"}:
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise ToolArgumentError(f"{path}: value is below minimum {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise ToolArgumentError(f"{path}: value is above maximum {maximum}")


class ToolRegistry:
    """Own tool definitions and provide a safe model-to-local dispatch boundary."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._parameters: dict[str, dict[str, Any]] = {}
        for tool in tools:
            self.register(tool)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic registration order."""

        return tuple(self._tools)

    def register(self, tool: Tool) -> None:
        """Register one tool after validating its public contract."""

        if not isinstance(tool, Tool):
            raise ToolRegistrationError("tool does not implement the Tool protocol")
        if not _TOOL_NAME_PATTERN.fullmatch(tool.name):
            raise ToolRegistrationError(
                "tool name must match [A-Za-z_][A-Za-z0-9_-]{0,63}"
            )
        if tool.name in self._tools:
            raise ToolRegistrationError(f"duplicate tool name: {tool.name}")
        if not isinstance(tool.description, str) or not tool.description.strip():
            raise ToolRegistrationError("tool description must not be empty")
        if not isinstance(tool.parameters, Mapping):
            raise ToolRegistrationError("tool parameters must be a mapping")

        self._tools[tool.name] = tool
        self._parameters[tool.name] = _snapshot_parameters(tool.parameters)

    def get(self, name: str) -> Tool | None:
        """Find a registered tool without executing it."""

        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """Export detached OpenAI-compatible function tool definitions."""

        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": deepcopy(self._parameters[name]),
                },
            }
            for name, tool in self._tools.items()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        """Validate and dispatch a model-requested call without crashing the loop."""

        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.fail(
                f"unknown tool: {call.name}",
                metadata={
                    "kind": "unknown_tool",
                    "tool_name": call.name,
                    "available_tools": self.names,
                },
            )

        try:
            _validate_value(call.arguments, self._parameters[call.name], "arguments")
        except ToolArgumentError as exc:
            return ToolResult.fail(
                str(exc),
                metadata={
                    "kind": "invalid_arguments",
                    "tool_name": call.name,
                },
            )

        try:
            result = tool.execute(call.arguments)
        except Exception as exc:
            return ToolResult.fail(
                f"tool raised {type(exc).__name__}: {exc}",
                metadata={
                    "kind": "execution_error",
                    "tool_name": call.name,
                    "exception_type": type(exc).__name__,
                },
            )
        if not isinstance(result, ToolResult):
            return ToolResult.fail(
                "tool returned an invalid result type",
                metadata={
                    "kind": "invalid_result",
                    "tool_name": call.name,
                },
            )
        return result

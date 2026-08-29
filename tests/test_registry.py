import unittest
from typing import Any, Mapping

from coding_agent.messages import ToolCall
from coding_agent.tools.base import ToolResult
from coding_agent.tools.registry import ToolRegistrationError, ToolRegistry


class EchoTool:
    name = "echo"
    description = "Return the supplied text."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "times": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls += 1
        times = arguments.get("times", 1)
        return ToolResult.ok(str(arguments["text"]) * int(times))


class RaisingTool:
    name = "raising_tool"
    description = "Raise an exception for testing."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raise RuntimeError("controlled failure")


class InvalidResultTool:
    name = "invalid_result"
    description = "Return the wrong result type for testing."
    parameters = {"type": "object", "properties": {}}

    def execute(self, arguments: Mapping[str, Any]):
        return "not a ToolResult"


class ToolRegistryTests(unittest.TestCase):
    def test_registers_and_finds_tool(self) -> None:
        tool = EchoTool()
        registry = ToolRegistry([tool])

        self.assertEqual(len(registry), 1)
        self.assertIn("echo", registry)
        self.assertEqual(registry.names, ("echo",))
        self.assertIs(registry.get("echo"), tool)
        self.assertIsNone(registry.get("missing"))

    def test_exports_openai_compatible_schema(self) -> None:
        registry = ToolRegistry([EchoTool()])

        schema = registry.schemas()[0]

        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "echo")
        self.assertEqual(
            schema["function"]["parameters"]["required"],
            ["text"],
        )

    def test_exported_schema_does_not_mutate_registry(self) -> None:
        registry = ToolRegistry([EchoTool()])
        exported = registry.schemas()
        exported[0]["function"]["parameters"]["required"].clear()

        fresh = registry.schemas()

        self.assertEqual(fresh[0]["function"]["parameters"]["required"], ["text"])

    def test_executes_valid_tool_call(self) -> None:
        tool = EchoTool()
        registry = ToolRegistry([tool])

        result = registry.execute(
            ToolCall(
                id="call-1",
                name="echo",
                arguments={"text": "go", "times": 2},
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output, "gogo")
        self.assertEqual(tool.calls, 1)

    def test_rejects_missing_required_argument_without_execution(self) -> None:
        tool = EchoTool()
        registry = ToolRegistry([tool])

        result = registry.execute(ToolCall(id="call-1", name="echo", arguments={}))

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "invalid_arguments")
        self.assertIn("missing required", result.error or "")
        self.assertEqual(tool.calls, 0)

    def test_rejects_unexpected_argument(self) -> None:
        registry = ToolRegistry([EchoTool()])

        result = registry.execute(
            ToolCall(
                id="call-1",
                name="echo",
                arguments={"text": "hello", "extra": True},
            )
        )

        self.assertFalse(result.success)
        self.assertIn("unexpected parameter", result.error or "")

    def test_rejects_argument_with_wrong_type(self) -> None:
        registry = ToolRegistry([EchoTool()])

        result = registry.execute(
            ToolCall(id="call-1", name="echo", arguments={"text": 123})
        )

        self.assertFalse(result.success)
        self.assertIn("expected string", result.error or "")

    def test_unknown_tool_is_a_failed_result(self) -> None:
        registry = ToolRegistry([EchoTool()])

        result = registry.execute(ToolCall(id="call-1", name="missing"))

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "unknown_tool")
        self.assertEqual(result.metadata["available_tools"], ("echo",))

    def test_tool_exception_is_a_failed_result(self) -> None:
        registry = ToolRegistry([RaisingTool()])

        result = registry.execute(ToolCall(id="call-1", name="raising_tool"))

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "execution_error")
        self.assertIn("RuntimeError", result.error or "")

    def test_invalid_tool_return_is_a_failed_result(self) -> None:
        registry = ToolRegistry([InvalidResultTool()])

        result = registry.execute(ToolCall(id="call-1", name="invalid_result"))

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "invalid_result")

    def test_rejects_duplicate_name(self) -> None:
        registry = ToolRegistry([EchoTool()])

        with self.assertRaisesRegex(ToolRegistrationError, "duplicate"):
            registry.register(EchoTool())

    def test_rejects_invalid_tool_name(self) -> None:
        tool = EchoTool()
        tool.name = "invalid tool name"

        with self.assertRaisesRegex(ToolRegistrationError, "tool name"):
            ToolRegistry([tool])

    def test_rejects_required_property_missing_from_schema(self) -> None:
        tool = EchoTool()
        tool.parameters = {
            "type": "object",
            "properties": {},
            "required": ["text"],
        }

        with self.assertRaisesRegex(ToolRegistrationError, "not declared"):
            ToolRegistry([tool])


if __name__ == "__main__":
    unittest.main()


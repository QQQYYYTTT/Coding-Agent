import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from coding_agent.agent import AgentRunner, AgentTraceKind
from coding_agent.messages import Message, MessageRole, ToolCall
from coding_agent.tools.filesystem import (
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
)
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tools.shell import RunCommandTool


class ScriptedWorkflowClient:
    """Deterministic model substitute for one complete coding workflow."""

    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[tuple[Message, ...], list[dict]]] = []

    def complete(self, messages, tools=()):
        self.calls.append((tuple(messages), list(tools)))
        if not self.responses:
            raise AssertionError("scripted workflow client ran out of responses")
        return self.responses.pop(0)


def _tool_payload(messages: tuple[Message, ...], call_id: str) -> dict:
    message = next(
        item
        for item in messages
        if item.role is MessageRole.TOOL and item.tool_call_id == call_id
    )
    return json.loads(message.content or "")


class ObserveModifyVerifyWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.fixture = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "buggy_shipping"
        )
        self.workspace = Path(self.temporary_directory.name) / "shipping_project"
        shutil.copytree(self.fixture, self.workspace)

    def test_agent_observes_modifies_and_verifies_buggy_project(self) -> None:
        source_path = self.workspace / "shipping.py"
        tests_path = self.workspace / "tests" / "test_shipping.py"
        original_source = source_path.read_text(encoding="utf-8")
        original_tests = tests_path.read_text(encoding="utf-8")
        old_text = (
            "    if weight < STANDARD_LIMIT_KG:\n"
            "        return STANDARD_FEE\n"
            "    if weight < HEAVY_LIMIT_KG:\n"
            "        return HEAVY_FEE"
        )
        new_text = (
            "    if weight <= STANDARD_LIMIT_KG:\n"
            "        return STANDARD_FEE\n"
            "    if weight <= HEAVY_LIMIT_KG:\n"
            "        return HEAVY_FEE"
        )
        test_command = [
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
        ]
        client = ScriptedWorkflowClient(
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="list-project",
                            name="list_files",
                            arguments={
                                "path": ".",
                                "recursive": True,
                                "max_depth": 3,
                            },
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="read-requirements",
                            name="read_file",
                            arguments={"path": "README.md"},
                        ),
                        ToolCall(
                            id="read-source",
                            name="read_file",
                            arguments={"path": "shipping.py"},
                        ),
                        ToolCall(
                            id="read-tests",
                            name="read_file",
                            arguments={"path": "tests/test_shipping.py"},
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="run-failing-tests",
                            name="run_command",
                            arguments={"argv": test_command},
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="fix-boundaries",
                            name="replace_text",
                            arguments={
                                "path": "shipping.py",
                                "old_text": old_text,
                                "new_text": new_text,
                            },
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="run-passing-tests",
                            name="run_command",
                            arguments={"argv": test_command},
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Fixed the boundary comparisons; all 8 tests pass.",
                ),
            ]
        )
        registry = ToolRegistry(
            [
                ListFilesTool(self.workspace),
                ReadFileTool(self.workspace),
                ReplaceTextTool(self.workspace),
                RunCommandTool(self.workspace),
            ]
        )
        events = []
        runner = AgentRunner(client, registry, trace_sink=events.append)

        result = runner.run(
            "Inspect the project, fix the bug without changing tests, and verify it."
        )

        list_result = _tool_payload(client.calls[1][0], "list-project")
        self.assertTrue(list_result["success"])
        self.assertIn("shipping.py", list_result["output"])
        self.assertIn("tests/test_shipping.py", list_result["output"])

        requirements_result = _tool_payload(client.calls[2][0], "read-requirements")
        self.assertIn("0 < weight <= 5", requirements_result["output"])
        source_result = _tool_payload(client.calls[2][0], "read-source")
        test_result = _tool_payload(client.calls[2][0], "read-tests")
        self.assertIn("weight < STANDARD_LIMIT_KG", source_result["output"])
        self.assertIn("test_exactly_five_kg", test_result["output"])

        failing_result = _tool_payload(client.calls[3][0], "run-failing-tests")
        self.assertFalse(failing_result["success"])
        self.assertEqual(failing_result["metadata"]["kind"], "nonzero_exit")
        self.assertEqual(failing_result["metadata"]["exit_code"], 1)
        self.assertIn("FAILED (failures=2)", failing_result["output"])

        edit_result = _tool_payload(client.calls[4][0], "fix-boundaries")
        self.assertTrue(edit_result["success"])
        self.assertEqual(edit_result["metadata"]["replacements"], 1)

        passing_result = _tool_payload(client.calls[5][0], "run-passing-tests")
        self.assertTrue(passing_result["success"])
        self.assertEqual(passing_result["metadata"]["exit_code"], 0)
        self.assertIn("Ran 8 tests", passing_result["output"])
        self.assertIn("OK", passing_result["output"])

        fixed_source = source_path.read_text(encoding="utf-8")
        self.assertNotEqual(fixed_source, original_source)
        self.assertIn("weight <= STANDARD_LIMIT_KG", fixed_source)
        self.assertIn("weight <= HEAVY_LIMIT_KG", fixed_source)
        self.assertEqual(tests_path.read_text(encoding="utf-8"), original_tests)
        self.assertEqual(
            (self.fixture / "shipping.py").read_text(encoding="utf-8"),
            original_source,
        )

        tool_sequence = [
            event.tool_name
            for event in events
            if event.kind is AgentTraceKind.TOOL_START
        ]
        self.assertEqual(
            tool_sequence,
            [
                "list_files",
                "read_file",
                "read_file",
                "read_file",
                "run_command",
                "replace_text",
                "run_command",
            ],
        )
        self.assertEqual(result.model_turns, 6)
        self.assertEqual(result.tool_calls, 7)
        self.assertEqual(
            result.output,
            "Fixed the boundary comparisons; all 8 tests pass.",
        )


if __name__ == "__main__":
    unittest.main()

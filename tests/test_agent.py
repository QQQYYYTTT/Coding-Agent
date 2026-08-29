import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_agent.agent import (
    AgentLimitError,
    AgentLoopError,
    AgentRunner,
    AgentTraceKind,
)
from coding_agent.messages import Message, MessageRole, ToolCall
from coding_agent.tools.filesystem import ListFilesTool, ReadFileTool
from coding_agent.tools.registry import ToolRegistry


class ScriptedClient:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[tuple[Message, ...], list[dict]]] = []

    def complete(self, messages, tools=()):
        self.calls.append((tuple(messages), list(tools)))
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        return self.responses.pop(0)


class AgentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name)
        self.registry = ToolRegistry([ReadFileTool(self.workspace)])

    def test_returns_immediate_text_response(self) -> None:
        client = ScriptedClient(
            [Message(role=MessageRole.ASSISTANT, content="Hello.")]
        )
        runner = AgentRunner(client, self.registry)

        result = runner.run("Say hello", system_prompt="Be concise.")

        self.assertEqual(result.output, "Hello.")
        self.assertEqual(result.model_turns, 1)
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(
            [message.role for message in result.history],
            [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT],
        )
        self.assertEqual(client.calls[0][1][0]["function"]["name"], "read_file")

    def test_executes_read_file_and_returns_result_to_model(self) -> None:
        (self.workspace / "README.md").write_text(
            "A tiny demo project.",
            encoding="utf-8",
        )
        client = ScriptedClient(
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="call-1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="The README describes a tiny demo project.",
                ),
            ]
        )
        runner = AgentRunner(client, self.registry)

        result = runner.run("Read README.md and summarize it.")

        self.assertEqual(result.model_turns, 2)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.output, "The README describes a tiny demo project.")
        second_history = client.calls[1][0]
        self.assertEqual(
            [message.role for message in second_history],
            [
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.TOOL,
            ],
        )
        tool_message = second_history[-1]
        self.assertEqual(tool_message.tool_call_id, "call-1")
        tool_payload = json.loads(tool_message.content or "")
        self.assertTrue(tool_payload["success"])
        self.assertEqual(tool_payload["output"], "A tiny demo project.")
        self.assertEqual(tool_payload["metadata"]["path"], "README.md")

    def test_returns_tool_failure_to_model_for_recovery(self) -> None:
        client = ScriptedClient(
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="call-missing",
                            name="read_file",
                            arguments={"path": "missing.txt"},
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="The requested file does not exist.",
                ),
            ]
        )

        result = AgentRunner(client, self.registry).run("Read missing.txt")

        payload = json.loads(client.calls[1][0][-1].content or "")
        self.assertFalse(payload["success"])
        self.assertEqual(payload["metadata"]["kind"], "not_found")
        self.assertEqual(result.output, "The requested file does not exist.")

    def test_emits_bounded_structured_trace_events(self) -> None:
        file_contents = "content that must not appear in trace summary"
        (self.workspace / "trace.txt").write_text(file_contents, encoding="utf-8")
        client = ScriptedClient(
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="call-trace",
                            name="read_file",
                            arguments={"path": "trace.txt"},
                        ),
                    ),
                ),
                Message(role=MessageRole.ASSISTANT, content="Done."),
            ]
        )
        events = []
        runner = AgentRunner(client, self.registry, trace_sink=events.append)

        runner.run("Read trace.txt")

        self.assertEqual(
            [event.kind for event in events],
            [
                AgentTraceKind.MODEL_START,
                AgentTraceKind.MODEL_TOOL_CALLS,
                AgentTraceKind.TOOL_START,
                AgentTraceKind.TOOL_FINISH,
                AgentTraceKind.MODEL_START,
                AgentTraceKind.MODEL_FINAL,
            ],
        )
        self.assertEqual(events[2].tool_name, "read_file")
        self.assertEqual(events[2].arguments["path"], "trace.txt")
        self.assertTrue(events[3].success)
        self.assertIn("characters=", events[3].summary or "")
        self.assertNotIn(file_contents, events[3].summary or "")

    def test_supports_multiple_tool_calls_in_one_model_turn(self) -> None:
        (self.workspace / "a.txt").write_text("A", encoding="utf-8")
        (self.workspace / "b.txt").write_text("B", encoding="utf-8")
        client = ScriptedClient(
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(id="call-a", name="read_file", arguments={"path": "a.txt"}),
                        ToolCall(id="call-b", name="read_file", arguments={"path": "b.txt"}),
                    ),
                ),
                Message(role=MessageRole.ASSISTANT, content="A and B"),
            ]
        )

        result = AgentRunner(client, self.registry).run("Read both files")

        second_history = client.calls[1][0]
        self.assertEqual(second_history[-2].tool_call_id, "call-a")
        self.assertEqual(second_history[-1].tool_call_id, "call-b")
        self.assertEqual(result.tool_calls, 2)

    def test_discovers_then_reads_unknown_file(self) -> None:
        (self.workspace / "README.md").write_text("discovered demo", encoding="utf-8")
        registry = ToolRegistry(
            [ListFilesTool(self.workspace), ReadFileTool(self.workspace)]
        )
        client = ScriptedClient(
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(id="call-list", name="list_files", arguments={}),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(
                            id="call-read",
                            name="read_file",
                            arguments={"path": "README.md"},
                        ),
                    ),
                ),
                Message(role=MessageRole.ASSISTANT, content="Found the demo."),
            ]
        )

        result = AgentRunner(client, registry).run(
            "Find the project description and summarize it."
        )

        list_payload = json.loads(client.calls[1][0][-1].content or "")
        read_payload = json.loads(client.calls[2][0][-1].content or "")
        self.assertIn("README.md", list_payload["output"])
        self.assertEqual(read_payload["output"], "discovered demo")
        self.assertEqual(result.model_turns, 3)
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.output, "Found the demo.")

    def test_raises_after_maximum_model_turns(self) -> None:
        (self.workspace / "loop.txt").write_text("loop", encoding="utf-8")
        client = ScriptedClient(
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(id="call-1", name="read_file", arguments={"path": "loop.txt"}),
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCall(id="call-2", name="read_file", arguments={"path": "loop.txt"}),
                    ),
                ),
            ]
        )

        with self.assertRaises(AgentLimitError) as captured:
            AgentRunner(client, self.registry, max_turns=2).run("Keep reading")

        self.assertEqual(captured.exception.max_turns, 2)
        self.assertEqual(len(captured.exception.history), 5)

    def test_rejects_reused_tool_call_id(self) -> None:
        (self.workspace / "file.txt").write_text("content", encoding="utf-8")
        repeated = ToolCall(
            id="same-id",
            name="read_file",
            arguments={"path": "file.txt"},
        )
        client = ScriptedClient(
            [
                Message(role=MessageRole.ASSISTANT, tool_calls=(repeated,)),
                Message(role=MessageRole.ASSISTANT, tool_calls=(repeated,)),
            ]
        )

        with self.assertRaisesRegex(AgentLoopError, "reused tool call id"):
            AgentRunner(client, self.registry).run("Read repeatedly")

    def test_rejects_non_assistant_model_message(self) -> None:
        client = ScriptedClient([Message(role=MessageRole.USER, content="wrong role")])

        with self.assertRaisesRegex(AgentLoopError, "non-assistant"):
            AgentRunner(client, self.registry).run("Hello")

    def test_rejects_empty_user_prompt(self) -> None:
        runner = AgentRunner(ScriptedClient([]), self.registry)

        with self.assertRaisesRegex(ValueError, "non-empty"):
            runner.run("   ")

    def test_rejects_invalid_turn_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            AgentRunner(ScriptedClient([]), self.registry, max_turns=0)


if __name__ == "__main__":
    unittest.main()

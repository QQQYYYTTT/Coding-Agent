import unittest

from coding_agent.messages import Message, MessageRole, ToolCall


class ToolCallTests(unittest.TestCase):
    def test_copies_arguments(self) -> None:
        arguments = {"path": "README.md"}
        call = ToolCall(id="call-1", name="read_file", arguments=arguments)
        arguments["path"] = "changed.txt"
        self.assertEqual(call.arguments["path"], "README.md")

    def test_rejects_empty_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "tool name"):
            ToolCall(id="call-1", name="  ")


class MessageTests(unittest.TestCase):
    def test_assistant_can_request_tools_without_text(self) -> None:
        call = ToolCall(
            id="call-1",
            name="read_file",
            arguments={"path": "README.md"},
        )
        message = Message(role=MessageRole.ASSISTANT, tool_calls=(call,))
        self.assertIsNone(message.content)
        self.assertEqual(message.tool_calls, (call,))

    def test_tool_message_requires_matching_call_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "tool_call_id"):
            Message(role=MessageRole.TOOL, content="file contents")

    def test_non_assistant_rejects_tool_calls(self) -> None:
        call = ToolCall(id="call-1", name="read_file")
        with self.assertRaisesRegex(ValueError, "assistant"):
            Message(
                role=MessageRole.USER,
                content="read it",
                tool_calls=(call,),
            )

    def test_assistant_requires_text_or_tool_calls(self) -> None:
        with self.assertRaisesRegex(ValueError, "content or tool calls"):
            Message(role=MessageRole.ASSISTANT)


if __name__ == "__main__":
    unittest.main()


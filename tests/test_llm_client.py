import unittest

from coding_agent.config import AppConfig
from coding_agent.llm.base import ModelResponseError
from coding_agent.llm.compatible import OpenAICompatibleClient
from coding_agent.messages import Message, MessageRole, ToolCall


class RecordingTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple] = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append((url, payload, headers, timeout))
        return self.response


class OpenAICompatibleClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            api_key="test-secret",
            base_url="https://gateway.example/v1",
            model="example-model",
            request_timeout=12.0,
        )

    def test_completes_a_no_tool_conversation(self) -> None:
        transport = RecordingTransport(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello from the model.",
                        }
                    }
                ]
            }
        )
        client = OpenAICompatibleClient(self.config, transport=transport)

        reply = client.complete(
            [Message(role=MessageRole.USER, content="Hello")]
        )

        self.assertEqual(reply.role, MessageRole.ASSISTANT)
        self.assertEqual(reply.content, "Hello from the model.")
        url, payload, headers, timeout = transport.calls[0]
        self.assertEqual(url, "https://gateway.example/v1/chat/completions")
        self.assertEqual(payload["model"], "example-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello"}])
        self.assertNotIn("tools", payload)
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        self.assertEqual(timeout, 12.0)

    def test_parses_future_function_tool_calls(self) -> None:
        transport = RecordingTransport(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        client = OpenAICompatibleClient(self.config, transport=transport)

        reply = client.complete(
            [Message(role=MessageRole.USER, content="Read the README")]
        )

        self.assertEqual(reply.tool_calls[0].name, "read_file")
        self.assertEqual(reply.tool_calls[0].arguments["path"], "README.md")

    def test_sends_registry_style_function_schema(self) -> None:
        transport = RecordingTransport(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "No tool needed.",
                        }
                    }
                ]
            }
        )
        client = OpenAICompatibleClient(self.config, transport=transport)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo text.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ]

        client.complete(
            [Message(role=MessageRole.USER, content="Hello")],
            tools=tools,
        )

        payload = transport.calls[0][1]
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_serializes_tool_result_message_with_matching_call_id(self) -> None:
        transport = RecordingTransport(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The file is a demo.",
                        }
                    }
                ]
            }
        )
        client = OpenAICompatibleClient(self.config, transport=transport)
        tool_call = ToolCall(
            id="call-1",
            name="read_file",
            arguments={"path": "README.md"},
        )

        client.complete(
            [
                Message(role=MessageRole.USER, content="Read README.md"),
                Message(role=MessageRole.ASSISTANT, tool_calls=(tool_call,)),
                Message(
                    role=MessageRole.TOOL,
                    content='{"success":true,"output":"demo"}',
                    tool_call_id="call-1",
                ),
            ]
        )

        messages = transport.calls[0][1]["messages"]
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")

    def test_rejects_response_without_choices(self) -> None:
        client = OpenAICompatibleClient(
            self.config,
            transport=RecordingTransport({"choices": []}),
        )

        with self.assertRaisesRegex(ModelResponseError, "no choices"):
            client.complete([Message(role=MessageRole.USER, content="Hello")])


if __name__ == "__main__":
    unittest.main()

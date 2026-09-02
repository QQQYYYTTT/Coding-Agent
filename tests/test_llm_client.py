import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from coding_agent.config import AppConfig
from coding_agent.llm.base import ModelRequestError, ModelResponseError
from coding_agent.llm.compatible import OpenAICompatibleClient, _http_post_json
from coding_agent.messages import Message, MessageRole, ToolCall


class RecordingTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple] = []

    def __call__(self, url, payload, headers, timeout, max_response_bytes):
        self.calls.append((url, payload, headers, timeout, max_response_bytes))
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
        url, payload, headers, timeout, max_response_bytes = transport.calls[0]
        self.assertEqual(url, "https://gateway.example/v1/chat/completions")
        self.assertEqual(payload["model"], "example-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello"}])
        self.assertNotIn("tools", payload)
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        self.assertEqual(timeout, 12.0)
        self.assertEqual(max_response_bytes, self.config.max_model_response_bytes)

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

    def test_retries_retryable_request_errors_with_backoff(self) -> None:
        attempts: list[tuple] = []
        delays: list[float] = []

        def transport(*args):
            attempts.append(args)
            if len(attempts) <= 2:
                raise ModelRequestError("temporary failure", retryable=True)
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "Recovered."}}
                ]
            }

        client = OpenAICompatibleClient(
            self.config,
            transport=transport,
            sleeper=delays.append,
        )
        reply = client.complete([Message(role=MessageRole.USER, content="Hello")])

        self.assertEqual(reply.content, "Recovered.")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(delays, [0.5, 1.0])
        self.assertEqual(attempts[0][4], self.config.max_model_response_bytes)

    def test_does_not_retry_non_retryable_request_errors(self) -> None:
        attempts = 0
        delays: list[float] = []

        def transport(*args):
            nonlocal attempts
            attempts += 1
            raise ModelRequestError("bad request", retryable=False)

        client = OpenAICompatibleClient(
            self.config,
            transport=transport,
            sleeper=delays.append,
        )

        with self.assertRaisesRegex(ModelRequestError, "bad request"):
            client.complete([Message(role=MessageRole.USER, content="Hello")])

        self.assertEqual(attempts, 1)
        self.assertEqual(delays, [])

    def test_stops_after_configured_retry_limit(self) -> None:
        config = AppConfig(
            api_key="test-secret",
            base_url="https://gateway.example/v1",
            model="example-model",
            max_retries=1,
        )
        attempts = 0

        def transport(*args):
            nonlocal attempts
            attempts += 1
            raise ModelRequestError("still unavailable", retryable=True)

        client = OpenAICompatibleClient(
            config,
            transport=transport,
            sleeper=lambda _: None,
        )

        with self.assertRaisesRegex(ModelRequestError, "still unavailable"):
            client.complete([Message(role=MessageRole.USER, content="Hello")])

        self.assertEqual(attempts, 2)

    def test_http_transport_rejects_oversized_response(self) -> None:
        class OversizedResponse:
            requested_size = None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                self.requested_size = size
                return b"x" * size

        response = OversizedResponse()
        with patch(
            "coding_agent.llm.compatible.urlopen",
            return_value=response,
        ):
            with self.assertRaisesRegex(ModelResponseError, "exceeded 1024 bytes"):
                _http_post_json(
                    "https://gateway.example/v1/chat/completions",
                    {},
                    {},
                    1.0,
                    1024,
                )

        self.assertEqual(response.requested_size, 1025)

    def test_http_status_retry_classification(self) -> None:
        statuses = (
            (400, False), (408, True), (409, True), (429, True), (503, True)
        )
        for status, expected in statuses:
            with self.subTest(status=status):
                error = HTTPError(
                    "https://gateway.example/v1/chat/completions",
                    status,
                    "provider error",
                    {},
                    BytesIO(b'{"error":{"message":"failed"}}'),
                )
                with patch(
                    "coding_agent.llm.compatible.urlopen",
                    side_effect=error,
                ):
                    with self.assertRaises(ModelRequestError) as captured:
                        _http_post_json(
                            "https://gateway.example/v1/chat/completions",
                            {},
                            {},
                            1.0,
                            1024,
                        )

                self.assertEqual(captured.exception.retryable, expected)

if __name__ == "__main__":
    unittest.main()

"""OpenAI-compatible Chat Completions client implemented over ordinary HTTP."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping, Sequence, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from coding_agent.config import AppConfig
from coding_agent.llm.base import (
    LLMClient,
    ModelRequestError,
    ModelResponseError,
    ToolSchema,
)
from coding_agent.messages import Message, MessageRole, ToolCall


JsonObject = Mapping[str, Any]
Transport = Callable[
    [str, JsonObject, Mapping[str, str], float, int], JsonObject
]


def _provider_error_message(body: bytes, fallback: str) -> str:
    """Extract a short provider error without exposing request headers."""

    try:
        payload = json.loads(body.decode("utf-8"))
        error = payload.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            return cast(str, error["message"])[:2_000]
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        pass
    return fallback


def _http_post_json(
    url: str,
    payload: JsonObject,
    headers: Mapping[str, str],
    timeout: float,
    max_response_bytes: int,
) -> JsonObject:
    """POST JSON while bounding memory used by provider responses."""

    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(max_response_bytes + 1)
            if len(body) > max_response_bytes:
                raise ModelResponseError(
                    f"model API response exceeded {max_response_bytes} bytes"
                )
    except HTTPError as exc:
        body = exc.read(max_response_bytes + 1)
        detail = _provider_error_message(body, exc.reason or "HTTP request failed")
        retryable = exc.code in {408, 409, 429} or 500 <= exc.code <= 599
        raise ModelRequestError(
            f"model API returned HTTP {exc.code}: {detail}",
            retryable=retryable,
        ) from exc
    except URLError as exc:
        raise ModelRequestError(
            f"could not reach model API: {exc.reason}",
            retryable=True,
        ) from exc
    except TimeoutError as exc:
        raise ModelRequestError(
            "model API request timed out",
            retryable=True,
        ) from exc

    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelResponseError("model API returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ModelResponseError("model API returned a non-object JSON response")
    return decoded


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(dict(call.arguments), ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _parse_tool_call(raw_call: Any) -> ToolCall:
    if not isinstance(raw_call, Mapping):
        raise ModelResponseError("tool call must be an object")
    call_id = raw_call.get("id")
    function = raw_call.get("function")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ModelResponseError("tool call is missing an id")
    if not isinstance(function, Mapping):
        raise ModelResponseError("tool call is missing its function object")

    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(name, str) or not name.strip():
        raise ModelResponseError("tool call is missing its function name")
    if not isinstance(raw_arguments, str):
        raise ModelResponseError("tool call arguments must be a JSON string")

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ModelResponseError("tool call arguments contain invalid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise ModelResponseError("tool call arguments must decode to an object")
    return ToolCall(id=call_id, name=name, arguments=arguments)


class OpenAICompatibleClient(LLMClient):
    """Normalize an OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        config: AppConfig,
        *,
        transport: Transport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or _http_post_json
        self._sleeper = sleeper

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] = (),
    ) -> Message:
        if not messages:
            raise ValueError("at least one message is required")

        request_payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [_message_payload(message) for message in messages],
        }
        if tools:
            request_payload["tools"] = [dict(tool) for tool in tools]
            request_payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "nju-coding-agent/0.1.0",
        }
        retries = 0
        while True:
            try:
                response = self._transport(
                    f"{self._config.base_url}/chat/completions",
                    request_payload,
                    headers,
                    self._config.request_timeout,
                    self._config.max_model_response_bytes,
                )
                return self._parse_response(response)
            except ModelRequestError as exc:
                if not exc.retryable or retries >= self._config.max_retries:
                    raise
                self._sleeper(min(0.5 * (2 ** retries), 4.0))
                retries += 1

    @staticmethod
    def _parse_response(response: JsonObject) -> Message:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelResponseError("model response contains no choices")

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise ModelResponseError("model response choice must be an object")
        raw_message = first_choice.get("message")
        if not isinstance(raw_message, Mapping):
            raise ModelResponseError("model response choice contains no message")

        content = raw_message.get("content")
        if content is not None and not isinstance(content, str):
            raise ModelResponseError("assistant content must be a string or null")

        raw_calls = raw_message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise ModelResponseError("assistant tool_calls must be a list")
        tool_calls = tuple(_parse_tool_call(call) for call in raw_calls)

        try:
            return Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            )
        except (TypeError, ValueError) as exc:
            raise ModelResponseError(f"invalid assistant message: {exc}") from exc

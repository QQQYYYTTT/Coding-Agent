"""Minimal model-tool execution loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping

from coding_agent.llm.base import LLMClient
from coding_agent.messages import Message, MessageRole, ToolCall
from coding_agent.tools.base import ToolResult
from coding_agent.tools.registry import ToolRegistry


class AgentLoopError(RuntimeError):
    """Raised when the local orchestration loop cannot continue safely."""


class AgentLimitError(AgentLoopError):
    """Raised when the model does not finish within the configured turn limit."""

    def __init__(self, max_turns: int, history: tuple[Message, ...]) -> None:
        super().__init__(f"agent reached the maximum of {max_turns} model turns")
        self.max_turns = max_turns
        self.history = history


class AgentContextError(AgentLoopError):
    """Raised when protected instructions alone exceed the context budget."""


class AgentNoProgressError(AgentLoopError):
    """Raised after repeated identical tool batches produce identical outcomes."""

    def __init__(
        self,
        repetitions: int,
        tool_names: tuple[str, ...],
        history: tuple[Message, ...],
    ) -> None:
        rendered_names = ", ".join(tool_names)
        super().__init__(
            "agent repeated the same tool batch without progress "
            f"{repetitions} times: {rendered_names}"
        )
        self.repetitions = repetitions
        self.tool_names = tool_names
        self.history = history


class AgentTraceKind(StrEnum):
    """Structured event types emitted by the agent loop."""

    MODEL_START = "model_start"
    MODEL_TOOL_CALLS = "model_tool_calls"
    TOOL_START = "tool_start"
    TOOL_FINISH = "tool_finish"
    CONTEXT_TRIMMED = "context_trimmed"
    NO_PROGRESS = "no_progress"
    MODEL_FINAL = "model_final"


@dataclass(frozen=True, slots=True)
class AgentTraceEvent:
    """A bounded trace event that never contains model credentials or tool output."""

    kind: AgentTraceKind
    model_turn: int
    tool_name: str | None = None
    arguments: Mapping[str, Any] | None = None
    success: bool | None = None
    summary: str | None = None
    tool_count: int | None = None
    omitted_messages: int | None = None
    context_characters: int | None = None
    repetitions: int | None = None

    def __post_init__(self) -> None:
        if self.arguments is not None:
            object.__setattr__(
                self,
                "arguments",
                MappingProxyType(dict(self.arguments)),
            )


TraceSink = Callable[[AgentTraceEvent], None]


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Successful outcome plus the complete inspectable conversation history."""

    final_message: Message
    history: tuple[Message, ...]
    model_turns: int
    tool_calls: int

    @property
    def output(self) -> str:
        """Return the final assistant text."""

        return self.final_message.content or ""


def _tool_result_content(result: ToolResult) -> str:
    """Serialize a local result into one Chat Completions tool message."""

    payload = {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "metadata": dict(result.metadata),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _tool_result_summary(result: ToolResult) -> str:
    """Summarize status and bounded metadata without exposing tool output."""

    if not result.success:
        kind = result.metadata.get("kind", "tool_error")
        return f"failed; kind={kind}"

    details: list[str] = []
    for key in (
        "entries",
        "files",
        "directories",
        "characters",
        "bytes",
        "created",
        "replacements",
        "matches",
        "exit_code",
        "timed_out",
        "duration_ms",
        "stdout_chars",
        "stderr_chars",
        "total_lines",
        "truncated",
    ):
        if key in result.metadata:
            value = result.metadata[key]
            if isinstance(value, bool):
                value = str(value).lower()
            details.append(f"{key}={value}")
    return "success" + (f"; {', '.join(details)}" if details else "")


def _tool_attempt_fingerprint(call: ToolCall, result: ToolResult) -> str:
    """Hash one action and its stable outcome without retaining sensitive arguments."""

    metadata = {
        str(key): value
        for key, value in result.metadata.items()
        if key != "duration_ms"
    }
    payload: dict[str, Any] = {
        "tool": call.name,
        "arguments": dict(call.arguments),
        "success": result.success,
        "error": result.error,
        "metadata": metadata,
    }
    output = result.output
    if call.name == "run_command":
        output = re.sub(r"\b\d+(?:\.\d+)?s\b", "<duration>", output)
    payload["output"] = output
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_MESSAGE_OVERHEAD = 16
_COMPACTED_OUTPUT = "[omitted because the conversation context was compacted]"


def _message_characters(message: Message) -> int:
    """Return a deterministic character estimate for one serialized message."""

    total = _MESSAGE_OVERHEAD + len(message.role.value)
    total += len(message.content or "")
    total += len(message.tool_call_id or "")
    for call in message.tool_calls:
        total += len(call.id) + len(call.name)
        total += len(
            json.dumps(
                dict(call.arguments),
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
    return total


def _context_characters(messages: tuple[Message, ...]) -> int:
    return sum(_message_characters(message) for message in messages)


def _conversation_blocks(
    messages: list[Message],
) -> tuple[list[Message], list[list[Message]]]:
    """Split protected prompts from atomic assistant/tool exchanges."""

    prefix: list[Message] = []
    index = 0
    while index < len(messages) and messages[index].role in {
        MessageRole.SYSTEM,
        MessageRole.USER,
    }:
        prefix.append(messages[index])
        index += 1

    blocks: list[list[Message]] = []
    current: list[Message] = []
    for message in messages[index:]:
        if message.role is MessageRole.ASSISTANT:
            if current:
                blocks.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        blocks.append(current)
    return prefix, blocks


def _compact_tool_block(
    block: list[Message],
    budget: int,
) -> list[Message] | None:
    """Keep tool-call pairing valid while replacing oversized details."""

    if not block or block[0].role is not MessageRole.ASSISTANT:
        return None
    assistant = block[0]
    if not assistant.tool_calls:
        return None

    compact_calls = tuple(
        ToolCall(id=call.id, name=call.name, arguments={})
        for call in assistant.tool_calls
    )
    compact: list[Message] = [
        Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=compact_calls,
        )
    ]
    for message in block[1:]:
        if message.role is not MessageRole.TOOL:
            return None
        success: bool | None = None
        error: str | None = None
        metadata: Mapping[str, Any] = {}
        try:
            payload = json.loads(message.content or "")
            if isinstance(payload, dict):
                success = payload.get("success")
                raw_error = payload.get("error")
                error = raw_error[:200] if isinstance(raw_error, str) else None
                raw_metadata = payload.get("metadata")
                if isinstance(raw_metadata, dict):
                    metadata = {
                        str(key): value
                        for key, value in raw_metadata.items()
                        if value is None or isinstance(value, (bool, int, float, str))
                    }
        except (json.JSONDecodeError, TypeError):
            pass
        compact_content = json.dumps(
            {
                "success": success,
                "output": _COMPACTED_OUTPUT,
                "error": error,
                "metadata": metadata,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        compact.append(
            Message(
                role=MessageRole.TOOL,
                content=compact_content,
                tool_call_id=message.tool_call_id,
            )
        )

    if _context_characters(tuple(compact)) <= budget:
        return compact

    minimal = [compact[0]] + [
        Message(
            role=MessageRole.TOOL,
            content='{"context_compacted":true}',
            tool_call_id=message.tool_call_id,
        )
        for message in block[1:]
        if message.role is MessageRole.TOOL
    ]
    return minimal if _context_characters(tuple(minimal)) <= budget else None


def _bounded_context(
    history: list[Message],
    max_context_characters: int,
) -> tuple[tuple[Message, ...], int]:
    """Keep protected prompts and the newest complete exchanges within budget."""

    prefix, blocks = _conversation_blocks(history)
    prefix_cost = _context_characters(tuple(prefix))
    if prefix_cost > max_context_characters:
        raise AgentContextError(
            "system and user prompts exceed the configured context budget "
            f"({prefix_cost} > {max_context_characters} characters)"
        )

    remaining = max_context_characters - prefix_cost
    selected_reversed: list[list[Message]] = []
    affected_messages = 0
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        block_cost = _context_characters(tuple(block))
        if block_cost <= remaining:
            selected_reversed.append(block)
            remaining -= block_cost
            continue

        compact = _compact_tool_block(block, remaining)
        if compact is not None:
            selected_reversed.append(compact)
        affected_messages += sum(len(item) for item in blocks[: index + 1])
        break

    selected: list[Message] = []
    for block in reversed(selected_reversed):
        selected.extend(block)
    return tuple(prefix + selected), affected_messages


class AgentRunner:
    """Alternate between model responses and local tool execution."""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        *,
        max_turns: int = 20,
        max_context_characters: int = 100_000,
        max_no_progress_turns: int = 3,
        trace_sink: TraceSink | None = None,
    ) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            raise TypeError("max_turns must be an integer")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        if not isinstance(max_context_characters, int) or isinstance(
            max_context_characters, bool
        ):
            raise TypeError("max_context_characters must be an integer")
        if max_context_characters < 1_000:
            raise ValueError("max_context_characters must be at least 1000")
        if not isinstance(max_no_progress_turns, int) or isinstance(
            max_no_progress_turns, bool
        ):
            raise TypeError("max_no_progress_turns must be an integer")
        if max_no_progress_turns < 2:
            raise ValueError("max_no_progress_turns must be at least 2")
        self._client = client
        self._registry = registry
        self._max_turns = max_turns
        self._max_context_characters = max_context_characters
        self._max_no_progress_turns = max_no_progress_turns
        self._trace_sink = trace_sink

    def _emit(self, event: AgentTraceEvent) -> None:
        if self._trace_sink is not None:
            self._trace_sink(event)

    def run(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        """Run until the model returns text without requesting another tool."""

        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")

        history: list[Message] = []
        if system_prompt:
            history.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
        history.append(Message(role=MessageRole.USER, content=user_prompt.strip()))

        schemas = self._registry.schemas()
        seen_call_ids: set[str] = set()
        tool_call_count = 0
        previous_tool_batch: tuple[str, ...] | None = None
        repeated_tool_batches = 0

        for model_turn in range(1, self._max_turns + 1):
            context, omitted_messages = _bounded_context(
                history,
                self._max_context_characters,
            )
            if omitted_messages:
                self._emit(
                    AgentTraceEvent(
                        kind=AgentTraceKind.CONTEXT_TRIMMED,
                        model_turn=model_turn,
                        omitted_messages=omitted_messages,
                        context_characters=_context_characters(context),
                    )
                )
            self._emit(
                AgentTraceEvent(
                    kind=AgentTraceKind.MODEL_START,
                    model_turn=model_turn,
                )
            )
            response = self._client.complete(context, schemas)
            if response.role is not MessageRole.ASSISTANT:
                raise AgentLoopError("model client returned a non-assistant message")
            history.append(response)

            if not response.tool_calls:
                self._emit(
                    AgentTraceEvent(
                        kind=AgentTraceKind.MODEL_FINAL,
                        model_turn=model_turn,
                    )
                )
                return AgentRunResult(
                    final_message=response,
                    history=tuple(history),
                    model_turns=model_turn,
                    tool_calls=tool_call_count,
                )

            self._emit(
                AgentTraceEvent(
                    kind=AgentTraceKind.MODEL_TOOL_CALLS,
                    model_turn=model_turn,
                    tool_count=len(response.tool_calls),
                )
            )

            current_tool_batch: list[str] = []
            for call in response.tool_calls:
                if call.id in seen_call_ids:
                    raise AgentLoopError(f"model reused tool call id: {call.id}")
                seen_call_ids.add(call.id)

                self._emit(
                    AgentTraceEvent(
                        kind=AgentTraceKind.TOOL_START,
                        model_turn=model_turn,
                        tool_name=call.name,
                        arguments=call.arguments,
                    )
                )
                result = self._registry.execute(call)
                current_tool_batch.append(_tool_attempt_fingerprint(call, result))
                tool_call_count += 1
                self._emit(
                    AgentTraceEvent(
                        kind=AgentTraceKind.TOOL_FINISH,
                        model_turn=model_turn,
                        tool_name=call.name,
                        success=result.success,
                        summary=_tool_result_summary(result),
                    )
                )
                history.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=_tool_result_content(result),
                        tool_call_id=call.id,
                    )
                )

            completed_batch = tuple(current_tool_batch)
            if completed_batch == previous_tool_batch:
                repeated_tool_batches += 1
            else:
                previous_tool_batch = completed_batch
                repeated_tool_batches = 1
            if repeated_tool_batches >= self._max_no_progress_turns:
                tool_names = tuple(call.name for call in response.tool_calls)
                self._emit(
                    AgentTraceEvent(
                        kind=AgentTraceKind.NO_PROGRESS,
                        model_turn=model_turn,
                        tool_count=len(tool_names),
                        repetitions=repeated_tool_batches,
                    )
                )
                raise AgentNoProgressError(
                    repeated_tool_batches,
                    tool_names,
                    tuple(history),
                )

        raise AgentLimitError(self._max_turns, tuple(history))

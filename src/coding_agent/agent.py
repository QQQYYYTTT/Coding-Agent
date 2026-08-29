"""Minimal model-tool execution loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping

from coding_agent.llm.base import LLMClient
from coding_agent.messages import Message, MessageRole
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


class AgentTraceKind(StrEnum):
    """Structured event types emitted by the agent loop."""

    MODEL_START = "model_start"
    MODEL_TOOL_CALLS = "model_tool_calls"
    TOOL_START = "tool_start"
    TOOL_FINISH = "tool_finish"
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
        "total_lines",
        "truncated",
    ):
        if key in result.metadata:
            value = result.metadata[key]
            if isinstance(value, bool):
                value = str(value).lower()
            details.append(f"{key}={value}")
    return "success" + (f"; {', '.join(details)}" if details else "")


class AgentRunner:
    """Alternate between model responses and local tool execution."""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        *,
        max_turns: int = 20,
        trace_sink: TraceSink | None = None,
    ) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            raise TypeError("max_turns must be an integer")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        self._client = client
        self._registry = registry
        self._max_turns = max_turns
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

        for model_turn in range(1, self._max_turns + 1):
            self._emit(
                AgentTraceEvent(
                    kind=AgentTraceKind.MODEL_START,
                    model_turn=model_turn,
                )
            )
            response = self._client.complete(tuple(history), schemas)
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

        raise AgentLimitError(self._max_turns, tuple(history))

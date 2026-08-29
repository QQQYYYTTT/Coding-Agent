"""Public types for the coding agent package."""

from coding_agent.agent import (
    AgentLimitError,
    AgentLoopError,
    AgentRunner,
    AgentRunResult,
    AgentTraceEvent,
    AgentTraceKind,
)
from coding_agent.config import AppConfig, ConfigurationError
from coding_agent.messages import Message, MessageRole, ToolCall
from coding_agent.tools.base import Tool, ToolResult
from coding_agent.tools.filesystem import (
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    WriteFileTool,
)
from coding_agent.tools.registry import ToolRegistrationError, ToolRegistry
from coding_agent.tools.shell import RunCommandTool

__all__ = [
    "AgentLimitError",
    "AgentLoopError",
    "AgentRunner",
    "AgentRunResult",
    "AgentTraceEvent",
    "AgentTraceKind",
    "AppConfig",
    "ConfigurationError",
    "Message",
    "MessageRole",
    "ListFilesTool",
    "ReadFileTool",
    "ReplaceTextTool",
    "RunCommandTool",
    "ToolCall",
    "Tool",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolResult",
    "WriteFileTool",
]

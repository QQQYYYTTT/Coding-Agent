"""Command-line entry point for the minimal coding agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from coding_agent.agent import (
    AgentLoopError,
    AgentRunner,
    AgentTraceEvent,
    AgentTraceKind,
)
from coding_agent.config import AppConfig, ConfigurationError
from coding_agent.llm.base import ModelClientError
from coding_agent.llm.compatible import OpenAICompatibleClient
from coding_agent.tools.filesystem import (
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    WriteFileTool,
)
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tools.shell import RunCommandTool


DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent working inside a local workspace. Use the provided "
    "tools whenever the user's request depends on project structure or file contents. "
    "Use list_files to discover unknown paths and read_file to inspect relevant text. "
    "Use write_file only to create a new file. Use replace_text for precise edits "
    "after reading the target file; old_text must identify exactly one occurrence. "
    "Use run_command to run tests after code changes; pass an argv array and inspect "
    "its exit code, stdout, and stderr before claiming success. "
    "Never claim to have observed the workspace without calling a tool. Paths passed "
    "to tools must be relative to the workspace. After receiving tool results, answer "
    "the user directly and concisely."
)

_SENSITIVE_ARGUMENT_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)

_CONTENT_ARGUMENT_NAMES = {"content", "old_text", "new_text"}


def _is_sensitive_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_ARGUMENT_PARTS)


def _sanitize_argv(value: Sequence[Any]) -> list[Any]:
    """Keep commands inspectable while redacting credential-shaped arguments."""

    sanitized: list[Any] = []
    redact_next = False
    for argument in list(value)[:64]:
        if not isinstance(argument, str):
            sanitized.append(f"<{type(argument).__name__}>")
            redact_next = False
            continue
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue

        flag, separator, _ = argument.partition("=")
        if _is_sensitive_name(flag):
            if separator:
                sanitized.append(f"{flag}=<redacted>")
            else:
                sanitized.append(argument[:80])
                redact_next = True
            continue
        lowered = argument.casefold()
        if lowered.startswith("sk-") or lowered.startswith("bearer "):
            sanitized.append("<redacted>")
            continue
        sanitized.append(argument if len(argument) <= 160 else argument[:157] + "...")
    if len(value) > 64:
        sanitized.append("<additional arguments omitted>")
    return sanitized


def _sanitize_trace_value(
    value: Any,
    *,
    name: str | None = None,
    depth: int = 0,
) -> Any:
    """Redact secret-shaped fields and bound data before terminal display."""

    if name is not None and _is_sensitive_name(name):
        return "<redacted>"
    if name in _CONTENT_ARGUMENT_NAMES and isinstance(value, str):
        return f"<{len(value)} characters omitted>"
    if name == "argv" and isinstance(value, (list, tuple)):
        return _sanitize_argv(value)
    if depth >= 4:
        return "<nested value omitted>"
    if isinstance(value, Mapping):
        items = list(value.items())[:20]
        sanitized = {
            str(key): _sanitize_trace_value(
                child,
                name=str(key),
                depth=depth + 1,
            )
            for key, child in items
        }
        if len(value) > len(items):
            sanitized["..."] = "<additional fields omitted>"
        return sanitized
    if isinstance(value, (list, tuple)):
        items = list(value)[:20]
        sanitized_items = [
            _sanitize_trace_value(item, depth=depth + 1) for item in items
        ]
        if len(value) > len(items):
            sanitized_items.append("<additional items omitted>")
        return sanitized_items
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def format_trace_event(event: AgentTraceEvent) -> str:
    """Render a structured trace without credentials or full tool results."""

    if event.kind is AgentTraceKind.MODEL_START:
        return f"[Turn {event.model_turn}] Calling model"
    if event.kind is AgentTraceKind.MODEL_TOOL_CALLS:
        return (
            f"[Turn {event.model_turn}] Model requested "
            f"{event.tool_count or 0} tool(s)"
        )
    if event.kind is AgentTraceKind.TOOL_START:
        safe_arguments = _sanitize_trace_value(event.arguments or {})
        rendered = json.dumps(safe_arguments, ensure_ascii=False, separators=(",", ":"))
        return f"[Tool] {event.tool_name} {rendered}"
    if event.kind is AgentTraceKind.TOOL_FINISH:
        return f"[Tool] {event.tool_name} {event.summary}"
    if event.kind is AgentTraceKind.MODEL_FINAL:
        return f"[Turn {event.model_turn}] Model returned final answer"
    return f"[Trace] {event.kind}"


def print_trace_event(event: AgentTraceEvent) -> None:
    """Write diagnostics to stderr so stdout remains the final answer channel."""

    print(format_trace_event(event), file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Run a minimal model-tool coding agent in a local workspace.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="User task. If omitted, the program asks for it interactively.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Directory available to local tools (default: current directory).",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System instruction for this run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show model turns and sanitized tool execution summaries.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prompt = args.prompt
    if prompt is None:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nConversation cancelled.", file=sys.stderr)
            return 130
    if not prompt:
        print("Error: prompt must not be empty.", file=sys.stderr)
        return 2

    try:
        config = AppConfig.from_env()
        client = OpenAICompatibleClient(config)
        registry = ToolRegistry(
            [
                ListFilesTool(
                    args.workspace,
                    max_output_chars=config.max_tool_output,
                ),
                ReadFileTool(
                    args.workspace,
                    max_output_chars=config.max_tool_output,
                ),
                WriteFileTool(args.workspace),
                ReplaceTextTool(args.workspace),
                RunCommandTool(
                    args.workspace,
                    max_output_chars=config.max_tool_output,
                ),
            ]
        )
        runner = AgentRunner(
            client,
            registry,
            max_turns=config.max_turns,
            trace_sink=print_trace_event if args.verbose else None,
        )
        result = runner.run(prompt, system_prompt=args.system)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    except ModelClientError as exc:
        print(f"Model API error: {exc}", file=sys.stderr)
        return 1
    except AgentLoopError as exc:
        print(f"Agent error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nAgent interrupted.", file=sys.stderr)
        return 130

    print(f"Assistant> {result.output}")
    return 0

"""Bounded, timeout-aware command execution inside one workspace."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO, Mapping, Sequence

from coding_agent.tools.base import ToolResult
from coding_agent.tools.filesystem import _normalize_workspace


_DEFAULT_ALLOWED_COMMANDS = (
    "git",
    "mypy",
    "py",
    "pytest",
    "python",
    "python3",
    "ruff",
)
_PYTHON_COMMANDS = {"py", "python", "python3"}
_PYTHON_SAFE_MODULES = {"compileall", "pytest", "unittest"}
_GIT_READ_ONLY_SUBCOMMANDS = {
    "diff",
    "grep",
    "log",
    "ls-files",
    "rev-parse",
    "show",
    "status",
}
_SECRET_ENV_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_TRUNCATION_MARKER = "\n...[truncated]"


def _normalized_command_name(value: str) -> str:
    name = value.casefold()
    return name[:-4] if name.endswith(".exe") else name


def _is_secret_environment_name(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return any(part in normalized for part in _SECRET_ENV_PARTS)


def _bounded_environment() -> dict[str, str]:
    """Inherit normal process settings without exposing model credentials."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if not _is_secret_environment_name(name)
    }
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return environment


def _clip(text: str, budget: int, *, already_truncated: bool) -> tuple[str, bool]:
    needs_truncation = already_truncated or len(text) > budget
    if not needs_truncation:
        return text, False
    marker = _TRUNCATION_MARKER[: max(budget, 0)]
    content_budget = max(budget - len(marker), 0)
    return text[:content_budget] + marker, True


def _format_output(
    stdout: str,
    stderr: str,
    *,
    exit_code: int | None,
    max_output_chars: int,
    stdout_capture_truncated: bool,
    stderr_capture_truncated: bool,
) -> tuple[str, bool, bool]:
    """Share one fixed output budget between stdout and stderr."""

    prefix = f"exit_code: {exit_code if exit_code is not None else 'none'}\n"
    stdout_header = "stdout:\n"
    stderr_header = "\nstderr:\n"
    available = max(
        max_output_chars - len(prefix) - len(stdout_header) - len(stderr_header),
        0,
    )

    stdout_need = len(stdout) + (
        len(_TRUNCATION_MARKER) if stdout_capture_truncated else 0
    )
    stderr_need = len(stderr) + (
        len(_TRUNCATION_MARKER) if stderr_capture_truncated else 0
    )
    stdout_budget = min(stdout_need, available // 2)
    stderr_budget = min(stderr_need, available - stdout_budget)
    remaining = available - stdout_budget - stderr_budget
    if remaining:
        extra_stdout = min(max(stdout_need - stdout_budget, 0), remaining)
        stdout_budget += extra_stdout
        remaining -= extra_stdout
    if remaining:
        stderr_budget += min(max(stderr_need - stderr_budget, 0), remaining)

    shown_stdout, stdout_truncated = _clip(
        stdout,
        stdout_budget,
        already_truncated=stdout_capture_truncated,
    )
    shown_stderr, stderr_truncated = _clip(
        stderr,
        stderr_budget,
        already_truncated=stderr_capture_truncated,
    )
    output = prefix + stdout_header + shown_stdout + stderr_header + shown_stderr
    return output[:max_output_chars], stdout_truncated, stderr_truncated


def _drain_stream(
    stream: BinaryIO,
    *,
    max_capture_bytes: int,
    destination: dict[str, tuple[bytes, bool]],
    key: str,
) -> None:
    chunks: list[bytes] = []
    stored = 0
    truncated = False
    try:
        while True:
            chunk = stream.read(8_192)
            if not chunk:
                break
            remaining = max_capture_bytes - stored
            if remaining > 0:
                kept = chunk[:remaining]
                chunks.append(kept)
                stored += len(kept)
            if len(chunk) > max(remaining, 0):
                truncated = True
    finally:
        stream.close()
    destination[key] = (b"".join(chunks), truncated)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Best-effort termination of the command and descendants after timeout."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()


class RunCommandTool:
    """Run selected development commands without invoking a command shell."""

    name = "run_command"
    description = (
        "Run a bounded development command in the workspace to execute tests or "
        "inspect results. Pass an argv array, not a shell command string. The working "
        "directory is always the workspace; shell operators are unavailable. Allowed "
        "commands are python test scripts/modules, pytest, ruff, mypy, and read-only git."
    )
    parameters = {
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": (
                    "Command argument vector, for example "
                    "['python', '-m', 'unittest', 'discover', '-s', 'tests', '-v']."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 60,
                "description": "Timeout from 1 to 60 seconds. Defaults to 30.",
            },
        },
        "required": ["argv"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path | str,
        *,
        default_timeout_seconds: int = 30,
        max_timeout_seconds: int = 60,
        max_output_chars: int = 20_000,
        allowed_commands: Sequence[str] = _DEFAULT_ALLOWED_COMMANDS,
    ) -> None:
        for value, name in (
            (default_timeout_seconds, "default_timeout_seconds"),
            (max_timeout_seconds, "max_timeout_seconds"),
            (max_output_chars, "max_output_chars"),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if default_timeout_seconds > max_timeout_seconds:
            raise ValueError(
                "default_timeout_seconds must not exceed max_timeout_seconds"
            )
        if max_timeout_seconds > 60:
            raise ValueError("max_timeout_seconds must not exceed 60")
        if max_output_chars < 100:
            raise ValueError("max_output_chars must be at least 100")
        if isinstance(allowed_commands, (str, bytes)):
            raise TypeError("allowed_commands must be a sequence of command names")

        normalized_allowed: set[str] = set()
        for command in allowed_commands:
            if not isinstance(command, str) or not command.strip():
                raise ValueError("allowed_commands must contain non-empty strings")
            if Path(command).name != command or "/" in command or "\\" in command:
                raise ValueError("allowed_commands must contain command names, not paths")
            normalized_allowed.add(_normalized_command_name(command))
        if not normalized_allowed:
            raise ValueError("allowed_commands must not be empty")

        self._workspace = _normalize_workspace(workspace)
        self._default_timeout_seconds = default_timeout_seconds
        self._max_timeout_seconds = max_timeout_seconds
        self._max_output_chars = max_output_chars
        self._allowed_commands = frozenset(normalized_allowed)

    @property
    def workspace(self) -> Path:
        return self._workspace

    def _validate_argv(
        self,
        raw_argv: Any,
    ) -> tuple[list[str] | None, ToolResult | None]:
        if (
            not isinstance(raw_argv, Sequence)
            or isinstance(raw_argv, (str, bytes))
            or not raw_argv
        ):
            return None, ToolResult.fail(
                "argv must be a non-empty array of strings",
                metadata={"kind": "invalid_arguments"},
            )
        if len(raw_argv) > 64:
            return None, ToolResult.fail(
                "argv must contain at most 64 items",
                metadata={"kind": "invalid_arguments"},
            )

        argv: list[str] = []
        total_characters = 0
        for value in raw_argv:
            if not isinstance(value, str) or not value or "\x00" in value:
                return None, ToolResult.fail(
                    "every argv item must be a non-empty string without null bytes",
                    metadata={"kind": "invalid_arguments"},
                )
            total_characters += len(value)
            if len(value) > 2_000 or total_characters > 8_000:
                return None, ToolResult.fail(
                    "command arguments exceed the safety limit",
                    metadata={"kind": "invalid_arguments"},
                )
            argv.append(value)

        requested = argv[0]
        if (
            Path(requested).name != requested
            or "/" in requested
            or "\\" in requested
        ):
            return None, ToolResult.fail(
                "argv[0] must be an allowed command name, not a path",
                metadata={"kind": "command_not_allowed"},
            )
        command_name = _normalized_command_name(requested)
        if command_name not in self._allowed_commands:
            return None, ToolResult.fail(
                f"command is not allowed: {requested}",
                metadata={"kind": "command_not_allowed", "command": requested},
            )

        policy_error = self._validate_command_policy(command_name, argv[1:])
        if policy_error is not None:
            return None, policy_error

        if command_name in _PYTHON_COMMANDS:
            argv[0] = sys.executable
        else:
            executable = shutil.which(requested)
            if executable is None:
                return None, ToolResult.fail(
                    f"command was not found: {requested}",
                    metadata={"kind": "command_not_found", "command": requested},
                )
            argv[0] = executable
        return argv, None

    def _validate_command_policy(
        self,
        command_name: str,
        arguments: Sequence[str],
    ) -> ToolResult | None:
        if command_name in _PYTHON_COMMANDS:
            return self._validate_python_arguments(arguments)
        if command_name == "git":
            if not arguments or arguments[0].casefold() not in _GIT_READ_ONLY_SUBCOMMANDS:
                return ToolResult.fail(
                    "only read-only git subcommands are allowed",
                    metadata={"kind": "command_not_allowed", "command": "git"},
                )
        return None

    def _validate_python_arguments(
        self,
        arguments: Sequence[str],
    ) -> ToolResult | None:
        if not arguments:
            return ToolResult.fail(
                "interactive Python is not allowed",
                metadata={"kind": "command_not_allowed", "command": "python"},
            )
        if "-c" in arguments or arguments[-1] == "-":
            return ToolResult.fail(
                "inline or stdin Python execution is not allowed",
                metadata={"kind": "command_not_allowed", "command": "python"},
            )

        if arguments[0] == "-m":
            if len(arguments) < 2 or arguments[1].casefold() not in _PYTHON_SAFE_MODULES:
                return ToolResult.fail(
                    "this Python module is not allowed",
                    metadata={"kind": "command_not_allowed", "command": "python"},
                )
            return None

        script_argument = next(
            (argument for argument in arguments if not argument.startswith("-")),
            None,
        )
        if script_argument is None:
            return ToolResult.fail(
                "Python must run a workspace script or an allowed -m module",
                metadata={"kind": "command_not_allowed", "command": "python"},
            )
        script_path = Path(script_argument)
        if script_path.is_absolute() or script_path.suffix.casefold() != ".py":
            return ToolResult.fail(
                "Python scripts must be relative .py files inside the workspace",
                metadata={"kind": "command_not_allowed", "command": "python"},
            )
        try:
            resolved_script = (self._workspace / script_path).resolve(strict=True)
        except (OSError, RuntimeError):
            return ToolResult.fail(
                "Python script does not exist or cannot be resolved",
                metadata={"kind": "not_found", "path": script_path.as_posix()},
            )
        if not resolved_script.is_relative_to(self._workspace) or not resolved_script.is_file():
            return ToolResult.fail(
                "Python script must stay inside the workspace",
                metadata={"kind": "path_outside_workspace"},
            )
        return None

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        unexpected = sorted(set(arguments) - {"argv", "timeout_seconds"})
        if unexpected:
            return ToolResult.fail(
                f"unexpected parameter(s): {', '.join(unexpected)}",
                metadata={"kind": "invalid_arguments"},
            )

        argv, argv_error = self._validate_argv(arguments.get("argv"))
        if argv_error is not None:
            return argv_error
        assert argv is not None

        timeout_seconds = arguments.get(
            "timeout_seconds",
            self._default_timeout_seconds,
        )
        if (
            not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
            or not 1 <= timeout_seconds <= self._max_timeout_seconds
        ):
            return ToolResult.fail(
                f"timeout_seconds must be an integer from 1 to {self._max_timeout_seconds}",
                metadata={"kind": "invalid_arguments"},
            )

        creation_flags = 0
        popen_arguments: dict[str, Any] = {}
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_arguments["start_new_session"] = True

        started = time.monotonic()
        try:
            process = subprocess.Popen(
                argv,
                cwd=self._workspace,
                env=_bounded_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=creation_flags,
                **popen_arguments,
            )
        except FileNotFoundError:
            return ToolResult.fail(
                "command was not found",
                metadata={"kind": "command_not_found"},
            )
        except OSError as exc:
            return ToolResult.fail(
                f"could not start command: {type(exc).__name__}",
                metadata={"kind": "start_error"},
            )

        assert process.stdout is not None and process.stderr is not None
        captured: dict[str, tuple[bytes, bool]] = {}
        max_capture_bytes = self._max_output_chars * 4
        threads = [
            threading.Thread(
                target=_drain_stream,
                kwargs={
                    "stream": process.stdout,
                    "max_capture_bytes": max_capture_bytes,
                    "destination": captured,
                    "key": "stdout",
                },
                daemon=True,
            ),
            threading.Thread(
                target=_drain_stream,
                kwargs={
                    "stream": process.stderr,
                    "max_capture_bytes": max_capture_bytes,
                    "destination": captured,
                    "key": "stderr",
                },
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        except KeyboardInterrupt:
            _terminate_process_tree(process)
            process.wait()
            raise
        finally:
            for thread in threads:
                thread.join(timeout=5)

        elapsed_ms = round((time.monotonic() - started) * 1_000)
        stdout_bytes, stdout_capture_truncated = captured.get("stdout", (b"", True))
        stderr_bytes, stderr_capture_truncated = captured.get("stderr", (b"", True))
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = None if timed_out else process.returncode
        output, stdout_truncated, stderr_truncated = _format_output(
            stdout,
            stderr,
            exit_code=exit_code,
            max_output_chars=self._max_output_chars,
            stdout_capture_truncated=stdout_capture_truncated,
            stderr_capture_truncated=stderr_capture_truncated,
        )
        metadata = {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": elapsed_ms,
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "truncated": stdout_truncated or stderr_truncated,
        }
        if timed_out:
            return ToolResult.fail(
                f"command exceeded the {timeout_seconds}-second timeout",
                output=output,
                metadata={"kind": "timeout", **metadata},
            )
        if process.returncode != 0:
            return ToolResult.fail(
                f"command exited with code {process.returncode}",
                output=output,
                metadata={"kind": "nonzero_exit", **metadata},
            )
        return ToolResult.ok(output, metadata=metadata)

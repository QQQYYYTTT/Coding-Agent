"""Workspace-scoped filesystem tools."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from coding_agent.tools.base import ToolResult


_TRUNCATION_MARKER = "\n...[output truncated]"
_LISTING_TRUNCATION_MARKER = "\n...[listing truncated]"
_DEFAULT_MAX_EDIT_BYTES = 1_000_000


def _normalize_workspace(workspace: Path | str) -> Path:
    root = Path(workspace).expanduser()
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("workspace does not exist or cannot be resolved") from exc
    if not resolved_root.is_dir():
        raise ValueError("workspace must be a directory")
    return resolved_root


def _resolve_workspace_path(
    workspace: Path,
    requested: str,
) -> tuple[Path | None, ToolResult | None]:
    relative_path = Path(requested)
    if relative_path.is_absolute() or relative_path.drive or relative_path.anchor:
        return None, ToolResult.fail(
            "path must be relative to the workspace",
            metadata={"kind": "invalid_path"},
        )

    try:
        target = (workspace / relative_path).resolve(strict=False)
    except (OSError, RuntimeError):
        return None, ToolResult.fail(
            "path could not be resolved",
            metadata={"kind": "invalid_path"},
        )
    if not target.is_relative_to(workspace):
        return None, ToolResult.fail(
            "path escapes the workspace",
            metadata={"kind": "path_outside_workspace"},
        )
    return target, None


def _is_protected_path(relative_path: Path) -> bool:
    """Keep local credentials and repository internals away from the model."""

    lowered_parts = [part.lower() for part in relative_path.parts]
    if ".git" in lowered_parts:
        return True
    filename = relative_path.name.lower()
    if filename == ".env.example":
        return False
    return filename == ".env" or filename.startswith(".env.")


def _validate_edit_path(
    workspace: Path,
    raw_path: Any,
) -> tuple[Path | None, str | None, ToolResult | None]:
    """Resolve one editable path and block credentials and repository internals."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, None, ToolResult.fail(
            "path must be a non-empty string",
            metadata={"kind": "invalid_arguments"},
        )
    target, path_error = _resolve_workspace_path(workspace, raw_path.strip())
    if path_error is not None:
        return None, None, path_error
    assert target is not None

    relative_display = target.relative_to(workspace).as_posix()
    if _is_protected_path(Path(relative_display)):
        return None, None, ToolResult.fail(
            "access to this sensitive path is blocked",
            metadata={"kind": "sensitive_path", "path": relative_display},
        )
    return target, relative_display, None


def _encoded_text(
    value: Any,
    name: str,
    *,
    allow_empty: bool,
    max_bytes: int,
) -> tuple[bytes | None, ToolResult | None]:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "" if allow_empty else " non-empty"
        return None, ToolResult.fail(
            f"{name} must be a{qualifier} string",
            metadata={"kind": "invalid_arguments"},
        )
    if "\x00" in value:
        return None, ToolResult.fail(
            f"{name} must not contain null bytes",
            metadata={"kind": "invalid_arguments"},
        )
    encoded = value.encode("utf-8")
    if len(encoded) > max_bytes:
        return None, ToolResult.fail(
            f"{name} exceeds the {max_bytes}-byte safety limit",
            metadata={"kind": "content_too_large", "bytes": len(encoded)},
        )
    return encoded, None


def _count_occurrences(text: str, fragment: str) -> int:
    """Count every possible match position, including overlapping matches."""

    matches = 0
    start = 0
    while True:
        position = text.find(fragment, start)
        if position < 0:
            return matches
        matches += 1
        start = position + 1


class ReadFileTool:
    """Read UTF-8 text files without allowing access outside one workspace."""

    name = "read_file"
    description = (
        "Read a UTF-8 text file inside the current workspace. Paths must be "
        "relative to the workspace. Optional line numbers are one-based and inclusive."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Workspace-relative path of the text file to read.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional first line to return, using one-based indexing.",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional final line to return, inclusive.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Path | str, *, max_output_chars: int = 20_000) -> None:
        if not isinstance(max_output_chars, int) or isinstance(max_output_chars, bool):
            raise TypeError("max_output_chars must be an integer")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than zero")

        self._workspace = _normalize_workspace(workspace)
        self._max_output_chars = max_output_chars

    @property
    def workspace(self) -> Path:
        """Return the normalized workspace root."""

        return self._workspace

    def _resolve_target(self, requested: str) -> tuple[Path | None, ToolResult | None]:
        return _resolve_workspace_path(self._workspace, requested)

    @staticmethod
    def _line_number(arguments: Mapping[str, Any], name: str) -> int | None:
        value = arguments.get(name)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be an integer greater than or equal to 1")
        return value

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Read a selected line range and return a bounded text result."""

        allowed_arguments = {"path", "start_line", "end_line"}
        unexpected = sorted(set(arguments) - allowed_arguments)
        if unexpected:
            return ToolResult.fail(
                f"unexpected parameter(s): {', '.join(unexpected)}",
                metadata={"kind": "invalid_arguments"},
            )

        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return ToolResult.fail(
                "path must be a non-empty string",
                metadata={"kind": "invalid_arguments"},
            )
        requested_path = raw_path.strip()

        try:
            start_line = self._line_number(arguments, "start_line") or 1
            end_line = self._line_number(arguments, "end_line")
        except ValueError as exc:
            return ToolResult.fail(
                str(exc),
                metadata={"kind": "invalid_arguments"},
            )
        if end_line is not None and end_line < start_line:
            return ToolResult.fail(
                "end_line must be greater than or equal to start_line",
                metadata={"kind": "invalid_arguments"},
            )

        target, path_error = self._resolve_target(requested_path)
        if path_error is not None:
            return path_error
        assert target is not None

        relative_display = target.relative_to(self._workspace).as_posix()
        if _is_protected_path(Path(relative_display)):
            return ToolResult.fail(
                "access to this sensitive path is blocked",
                metadata={"kind": "sensitive_path", "path": relative_display},
            )
        if not target.exists():
            return ToolResult.fail(
                f"file does not exist: {relative_display}",
                metadata={"kind": "not_found", "path": relative_display},
            )
        if not target.is_file():
            return ToolResult.fail(
                f"path is not a regular file: {relative_display}",
                metadata={"kind": "not_a_file", "path": relative_display},
            )

        try:
            with target.open("rb") as binary_file:
                if b"\x00" in binary_file.read(8_192):
                    return ToolResult.fail(
                        f"binary files are not supported: {relative_display}",
                        metadata={"kind": "binary_file", "path": relative_display},
                    )

            chunks: list[str] = []
            output_chars = 0
            truncated = False
            total_lines = 0
            marker = _TRUNCATION_MARKER[: self._max_output_chars]

            with target.open("r", encoding="utf-8", errors="strict") as text_file:
                for total_lines, line in enumerate(text_file, start=1):
                    if "\x00" in line:
                        return ToolResult.fail(
                            f"binary files are not supported: {relative_display}",
                            metadata={
                                "kind": "binary_file",
                                "path": relative_display,
                            },
                        )
                    selected = total_lines >= start_line and (
                        end_line is None or total_lines <= end_line
                    )
                    if not selected or truncated:
                        continue
                    if output_chars + len(line) <= self._max_output_chars:
                        chunks.append(line)
                        output_chars += len(line)
                        continue

                    combined = "".join(chunks) + line
                    content_limit = self._max_output_chars - len(marker)
                    chunks = [combined[: max(content_limit, 0)] + marker]
                    output_chars = len(chunks[0])
                    truncated = True
        except UnicodeDecodeError:
            return ToolResult.fail(
                f"file is not valid UTF-8 text: {relative_display}",
                metadata={"kind": "decode_error", "path": relative_display},
            )
        except OSError as exc:
            return ToolResult.fail(
                f"could not read file: {type(exc).__name__}",
                metadata={"kind": "read_error", "path": relative_display},
            )

        if start_line > total_lines and not (total_lines == 0 and start_line == 1):
            return ToolResult.fail(
                f"start_line {start_line} exceeds file length {total_lines}",
                metadata={
                    "kind": "line_out_of_range",
                    "path": relative_display,
                    "total_lines": total_lines,
                },
            )

        actual_end = total_lines if end_line is None else min(end_line, total_lines)
        return ToolResult.ok(
            "".join(chunks),
            metadata={
                "path": relative_display,
                "start_line": start_line,
                "end_line": actual_end,
                "total_lines": total_lines,
                "characters": output_chars,
                "truncated": truncated,
            },
        )


class WriteFileTool:
    """Create a new UTF-8 text file without ever overwriting an existing path."""

    name = "write_file"
    description = (
        "Create one new UTF-8 text file inside the workspace. This tool never "
        "overwrites an existing file, does not create missing parent directories, "
        "and blocks sensitive paths."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Workspace-relative path for the new file.",
            },
            "content": {
                "type": "string",
                "description": "Complete UTF-8 text content for the new file.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path | str,
        *,
        max_content_bytes: int = _DEFAULT_MAX_EDIT_BYTES,
    ) -> None:
        if not isinstance(max_content_bytes, int) or isinstance(max_content_bytes, bool):
            raise TypeError("max_content_bytes must be an integer")
        if max_content_bytes <= 0:
            raise ValueError("max_content_bytes must be greater than zero")
        self._workspace = _normalize_workspace(workspace)
        self._max_content_bytes = max_content_bytes

    @property
    def workspace(self) -> Path:
        return self._workspace

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        unexpected = sorted(set(arguments) - {"path", "content"})
        if unexpected:
            return ToolResult.fail(
                f"unexpected parameter(s): {', '.join(unexpected)}",
                metadata={"kind": "invalid_arguments"},
            )

        target, relative_display, path_error = _validate_edit_path(
            self._workspace,
            arguments.get("path"),
        )
        if path_error is not None:
            return path_error
        assert target is not None and relative_display is not None

        encoded, content_error = _encoded_text(
            arguments.get("content"),
            "content",
            allow_empty=True,
            max_bytes=self._max_content_bytes,
        )
        if content_error is not None:
            return content_error
        assert encoded is not None

        if target.exists() or target.is_symlink():
            return ToolResult.fail(
                f"path already exists; overwrite is not allowed: {relative_display}",
                metadata={"kind": "already_exists", "path": relative_display},
            )
        if not target.parent.exists():
            return ToolResult.fail(
                "parent directory does not exist",
                metadata={"kind": "parent_not_found", "path": relative_display},
            )
        if not target.parent.is_dir():
            return ToolResult.fail(
                "parent path is not a directory",
                metadata={"kind": "parent_not_directory", "path": relative_display},
            )

        created = False
        try:
            with target.open("xb") as new_file:
                created = True
                new_file.write(encoded)
                new_file.flush()
                os.fsync(new_file.fileno())
        except FileExistsError:
            return ToolResult.fail(
                f"path already exists; overwrite is not allowed: {relative_display}",
                metadata={"kind": "already_exists", "path": relative_display},
            )
        except OSError as exc:
            if created:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
            return ToolResult.fail(
                f"could not create file: {type(exc).__name__}",
                metadata={"kind": "write_error", "path": relative_display},
            )

        return ToolResult.ok(
            f"Created {relative_display}",
            metadata={
                "path": relative_display,
                "created": True,
                "characters": len(arguments["content"]),
                "bytes": len(encoded),
            },
        )


class ReplaceTextTool:
    """Replace text only when the requested old text has exactly one match."""

    name = "replace_text"
    description = (
        "Replace an exact text fragment in an existing UTF-8 file inside the "
        "workspace. The old_text must occur exactly once; zero or multiple matches "
        "cause no change."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Workspace-relative path of the file to edit.",
            },
            "old_text": {
                "type": "string",
                "minLength": 1,
                "description": "Exact existing text, which must occur exactly once.",
            },
            "new_text": {
                "type": "string",
                "description": "Exact replacement text; may be empty to delete old_text.",
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path | str,
        *,
        max_file_bytes: int = _DEFAULT_MAX_EDIT_BYTES,
    ) -> None:
        if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool):
            raise TypeError("max_file_bytes must be an integer")
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be greater than zero")
        self._workspace = _normalize_workspace(workspace)
        self._max_file_bytes = max_file_bytes

    @property
    def workspace(self) -> Path:
        return self._workspace

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        unexpected = sorted(set(arguments) - {"path", "old_text", "new_text"})
        if unexpected:
            return ToolResult.fail(
                f"unexpected parameter(s): {', '.join(unexpected)}",
                metadata={"kind": "invalid_arguments"},
            )

        target, relative_display, path_error = _validate_edit_path(
            self._workspace,
            arguments.get("path"),
        )
        if path_error is not None:
            return path_error
        assert target is not None and relative_display is not None

        old_encoded, old_error = _encoded_text(
            arguments.get("old_text"),
            "old_text",
            allow_empty=False,
            max_bytes=self._max_file_bytes,
        )
        if old_error is not None:
            return old_error
        _, new_error = _encoded_text(
            arguments.get("new_text"),
            "new_text",
            allow_empty=True,
            max_bytes=self._max_file_bytes,
        )
        if new_error is not None:
            return new_error
        assert old_encoded is not None

        if not target.exists():
            return ToolResult.fail(
                f"file does not exist: {relative_display}",
                metadata={"kind": "not_found", "path": relative_display},
            )
        if not target.is_file():
            return ToolResult.fail(
                f"path is not a regular file: {relative_display}",
                metadata={"kind": "not_a_file", "path": relative_display},
            )
        try:
            if target.stat().st_size > self._max_file_bytes:
                return ToolResult.fail(
                    f"file exceeds the {self._max_file_bytes}-byte safety limit",
                    metadata={"kind": "file_too_large", "path": relative_display},
                )
            original_bytes = target.read_bytes()
            if len(original_bytes) > self._max_file_bytes:
                return ToolResult.fail(
                    f"file exceeds the {self._max_file_bytes}-byte safety limit",
                    metadata={"kind": "file_too_large", "path": relative_display},
                )
            if b"\x00" in original_bytes:
                return ToolResult.fail(
                    f"binary files are not supported: {relative_display}",
                    metadata={"kind": "binary_file", "path": relative_display},
                )
            original = original_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return ToolResult.fail(
                f"file is not valid UTF-8 text: {relative_display}",
                metadata={"kind": "decode_error", "path": relative_display},
            )
        except OSError as exc:
            return ToolResult.fail(
                f"could not read file: {type(exc).__name__}",
                metadata={"kind": "read_error", "path": relative_display},
            )

        old_text = arguments["old_text"]
        matches = _count_occurrences(original, old_text)
        if matches == 0:
            return ToolResult.fail(
                "old_text was not found; file was not changed",
                metadata={"kind": "text_not_found", "path": relative_display, "matches": 0},
            )
        if matches != 1:
            return ToolResult.fail(
                f"old_text occurs {matches} times; file was not changed",
                metadata={"kind": "ambiguous_match", "path": relative_display, "matches": matches},
            )

        updated = original.replace(old_text, arguments["new_text"], 1)
        updated_bytes = updated.encode("utf-8")
        if len(updated_bytes) > self._max_file_bytes:
            return ToolResult.fail(
                f"updated file exceeds the {self._max_file_bytes}-byte safety limit",
                metadata={"kind": "content_too_large", "path": relative_display, "bytes": len(updated_bytes)},
            )

        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(updated_bytes)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.chmod(temporary_path, target.stat().st_mode)
            os.replace(temporary_path, target)
            temporary_path = None
        except OSError as exc:
            return ToolResult.fail(
                f"could not replace text: {type(exc).__name__}",
                metadata={"kind": "write_error", "path": relative_display},
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

        return ToolResult.ok(
            f"Replaced text in {relative_display}",
            metadata={
                "path": relative_display,
                "replacements": 1,
                "characters": len(updated),
                "bytes": len(updated_bytes),
            },
        )


class ListFilesTool:
    """List visible workspace entries without exposing protected paths."""

    name = "list_files"
    description = (
        "List files and directories inside the current workspace. Use this tool "
        "to discover project structure before reading files. Returned paths are "
        "relative to the workspace; directories end with '/'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Directory to list, relative to the workspace. Defaults to '.'.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to include nested entries. Defaults to false.",
            },
            "max_depth": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Maximum recursive depth, from 1 to 5. Defaults to 3.",
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Include hidden entries except protected secrets. Defaults to false.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path | str,
        *,
        max_entries: int = 500,
        max_output_chars: int = 20_000,
    ) -> None:
        for value, name in (
            (max_entries, "max_entries"),
            (max_output_chars, "max_output_chars"),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        self._workspace = _normalize_workspace(workspace)
        self._max_entries = max_entries
        self._max_output_chars = max_output_chars

    @property
    def workspace(self) -> Path:
        return self._workspace

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        allowed_arguments = {"path", "recursive", "max_depth", "include_hidden"}
        unexpected = sorted(set(arguments) - allowed_arguments)
        if unexpected:
            return ToolResult.fail(
                f"unexpected parameter(s): {', '.join(unexpected)}",
                metadata={"kind": "invalid_arguments"},
            )

        raw_path = arguments.get("path", ".")
        recursive = arguments.get("recursive", False)
        max_depth = arguments.get("max_depth", 3)
        include_hidden = arguments.get("include_hidden", False)
        if not isinstance(raw_path, str) or not raw_path.strip():
            return ToolResult.fail(
                "path must be a non-empty string",
                metadata={"kind": "invalid_arguments"},
            )
        if not isinstance(recursive, bool):
            return ToolResult.fail(
                "recursive must be a boolean",
                metadata={"kind": "invalid_arguments"},
            )
        if not isinstance(include_hidden, bool):
            return ToolResult.fail(
                "include_hidden must be a boolean",
                metadata={"kind": "invalid_arguments"},
            )
        if (
            not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or not 1 <= max_depth <= 5
        ):
            return ToolResult.fail(
                "max_depth must be an integer from 1 to 5",
                metadata={"kind": "invalid_arguments"},
            )

        target, path_error = _resolve_workspace_path(
            self._workspace,
            raw_path.strip(),
        )
        if path_error is not None:
            return path_error
        assert target is not None

        relative_target = target.relative_to(self._workspace)
        display_target = relative_target.as_posix()
        if _is_protected_path(relative_target):
            return ToolResult.fail(
                "access to this sensitive path is blocked",
                metadata={"kind": "sensitive_path", "path": display_target},
            )
        if not target.exists():
            return ToolResult.fail(
                f"directory does not exist: {display_target}",
                metadata={"kind": "not_found", "path": display_target},
            )
        if not target.is_dir():
            return ToolResult.fail(
                f"path is not a directory: {display_target}",
                metadata={"kind": "not_a_directory", "path": display_target},
            )

        effective_depth = max_depth if recursive else 1
        queue: list[tuple[Path, int]] = [(target, 0)]
        visited = {target}
        lines: list[str] = []
        directories = 0
        files = 0
        skipped = 0
        truncated = False

        while queue and not truncated:
            current, current_depth = queue.pop(0)
            try:
                children = sorted(current.iterdir(), key=lambda item: item.name.casefold())
            except OSError:
                if current == target:
                    return ToolResult.fail(
                        "could not list directory",
                        metadata={"kind": "list_error", "path": display_target},
                    )
                skipped += 1
                continue

            for child in children:
                lexical_relative = child.relative_to(self._workspace)
                if _is_protected_path(lexical_relative):
                    skipped += 1
                    continue
                if not include_hidden and child.name.startswith("."):
                    skipped += 1
                    continue
                try:
                    resolved_child = child.resolve(strict=True)
                except (OSError, RuntimeError):
                    skipped += 1
                    continue
                if not resolved_child.is_relative_to(self._workspace):
                    skipped += 1
                    continue

                is_directory = resolved_child.is_dir()
                display = lexical_relative.as_posix() + ("/" if is_directory else "")
                separator_length = 1 if lines else 0
                projected_length = (
                    sum(len(line) for line in lines)
                    + max(len(lines) - 1, 0)
                    + separator_length
                    + len(display)
                )
                if len(lines) >= self._max_entries or projected_length > self._max_output_chars:
                    truncated = True
                    break

                lines.append(display)
                if is_directory:
                    directories += 1
                    child_depth = current_depth + 1
                    if child_depth < effective_depth and resolved_child not in visited:
                        visited.add(resolved_child)
                        queue.append((resolved_child, child_depth))
                else:
                    files += 1

        output = "\n".join(lines)
        if truncated:
            marker = _LISTING_TRUNCATION_MARKER[: self._max_output_chars]
            content_limit = self._max_output_chars - len(marker)
            output = output[: max(content_limit, 0)] + marker
        elif not lines:
            output = "(empty directory)"

        return ToolResult.ok(
            output,
            metadata={
                "path": display_target,
                "entries": len(lines),
                "files": files,
                "directories": directories,
                "skipped": skipped,
                "recursive": recursive,
                "max_depth": effective_depth,
                "characters": len(output),
                "truncated": truncated,
            },
        )

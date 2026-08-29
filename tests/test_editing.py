import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from coding_agent.messages import ToolCall
from coding_agent.tools.filesystem import ReplaceTextTool, WriteFileTool
from coding_agent.tools.registry import ToolRegistry


class WriteFileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace.mkdir()
        self.tool = WriteFileTool(self.workspace)

    def test_creates_new_utf8_file(self) -> None:
        result = self.tool.execute({"path": "hello.txt", "content": "你好\n"})

        self.assertTrue(result.success)
        self.assertEqual((self.workspace / "hello.txt").read_text(encoding="utf-8"), "你好\n")
        self.assertTrue(result.metadata["created"])
        self.assertEqual(result.metadata["characters"], 3)
        self.assertEqual(result.metadata["bytes"], 7)

    def test_allows_empty_file(self) -> None:
        result = self.tool.execute({"path": "empty.txt", "content": ""})

        self.assertTrue(result.success)
        self.assertEqual((self.workspace / "empty.txt").read_bytes(), b"")

    def test_refuses_to_overwrite_existing_file(self) -> None:
        target = self.workspace / "existing.txt"
        target.write_text("original", encoding="utf-8")

        result = self.tool.execute({"path": "existing.txt", "content": "changed"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "already_exists")
        self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_requires_existing_parent_directory(self) -> None:
        result = self.tool.execute({"path": "missing/file.txt", "content": "data"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "parent_not_found")
        self.assertFalse((self.workspace / "missing").exists())

    def test_blocks_traversal_and_sensitive_paths(self) -> None:
        outside = self.tool.execute({"path": "../outside.txt", "content": "secret"})
        env_file = self.tool.execute({"path": ".env", "content": "KEY=secret"})

        self.assertEqual(outside.metadata["kind"], "path_outside_workspace")
        self.assertEqual(env_file.metadata["kind"], "sensitive_path")
        self.assertFalse((self.workspace.parent / "outside.txt").exists())
        self.assertFalse((self.workspace / ".env").exists())

    def test_rejects_null_bytes_and_oversized_content(self) -> None:
        small_tool = WriteFileTool(self.workspace, max_content_bytes=3)

        null_result = self.tool.execute({"path": "null.txt", "content": "a\x00b"})
        large_result = small_tool.execute({"path": "large.txt", "content": "你好"})

        self.assertEqual(null_result.metadata["kind"], "invalid_arguments")
        self.assertEqual(large_result.metadata["kind"], "content_too_large")
        self.assertFalse((self.workspace / "null.txt").exists())
        self.assertFalse((self.workspace / "large.txt").exists())

    def test_blocks_symlink_escape(self) -> None:
        outside = self.workspace.parent / "outside"
        outside.mkdir()
        link = self.workspace / "link"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlinks are unavailable: {exc}")

        result = self.tool.execute({"path": "link/new.txt", "content": "secret"})

        self.assertEqual(result.metadata["kind"], "path_outside_workspace")
        self.assertFalse((outside / "new.txt").exists())

    def test_registry_exports_write_file_schema(self) -> None:
        registry = ToolRegistry([self.tool])

        schema = registry.schemas()[0]["function"]
        result = registry.execute(
            ToolCall(
                id="write-1",
                name="write_file",
                arguments={"path": "new.txt", "content": "new"},
            )
        )

        self.assertEqual(schema["name"], "write_file")
        self.assertEqual(schema["parameters"]["required"], ["path", "content"])
        self.assertTrue(result.success)


class ReplaceTextToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace.mkdir()
        self.tool = ReplaceTextTool(self.workspace)

    def test_replaces_one_exact_match(self) -> None:
        target = self.workspace / "app.py"
        target.write_text("name = 'old'\n", encoding="utf-8", newline="")

        result = self.tool.execute(
            {"path": "app.py", "old_text": "'old'", "new_text": "'新值'"}
        )

        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "name = '新值'\n")
        self.assertEqual(result.metadata["replacements"], 1)

    def test_allows_deleting_the_unique_text(self) -> None:
        target = self.workspace / "delete.txt"
        target.write_text("before REMOVE after", encoding="utf-8")

        result = self.tool.execute(
            {"path": "delete.txt", "old_text": "REMOVE ", "new_text": ""}
        )

        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "before after")

    def test_zero_matches_causes_no_change(self) -> None:
        target = self.workspace / "zero.txt"
        target.write_text("original", encoding="utf-8")

        result = self.tool.execute(
            {"path": "zero.txt", "old_text": "missing", "new_text": "new"}
        )

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "text_not_found")
        self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_multiple_matches_causes_no_change(self) -> None:
        target = self.workspace / "many.txt"
        target.write_text("old and old", encoding="utf-8")

        result = self.tool.execute(
            {"path": "many.txt", "old_text": "old", "new_text": "new"}
        )

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "ambiguous_match")
        self.assertEqual(result.metadata["matches"], 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "old and old")

    def test_overlapping_matches_are_ambiguous(self) -> None:
        target = self.workspace / "overlap.txt"
        target.write_text("aaa", encoding="utf-8")

        result = self.tool.execute(
            {"path": "overlap.txt", "old_text": "aa", "new_text": "b"}
        )

        self.assertEqual(result.metadata["kind"], "ambiguous_match")
        self.assertEqual(result.metadata["matches"], 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "aaa")

    def test_rejects_empty_old_text(self) -> None:
        target = self.workspace / "empty-old.txt"
        target.write_text("original", encoding="utf-8")

        result = self.tool.execute(
            {"path": "empty-old.txt", "old_text": "", "new_text": "new"}
        )

        self.assertEqual(result.metadata["kind"], "invalid_arguments")
        self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_rejects_binary_non_utf8_and_sensitive_files(self) -> None:
        (self.workspace / "binary.bin").write_bytes(b"a\x00b")
        (self.workspace / "legacy.txt").write_bytes(b"\xffbad")
        (self.workspace / ".env").write_text("KEY=secret", encoding="utf-8")

        binary = self.tool.execute(
            {"path": "binary.bin", "old_text": "a", "new_text": "b"}
        )
        legacy = self.tool.execute(
            {"path": "legacy.txt", "old_text": "bad", "new_text": "good"}
        )
        env_file = self.tool.execute(
            {"path": ".env", "old_text": "secret", "new_text": "leaked"}
        )

        self.assertEqual(binary.metadata["kind"], "binary_file")
        self.assertEqual(legacy.metadata["kind"], "decode_error")
        self.assertEqual(env_file.metadata["kind"], "sensitive_path")

    def test_failed_atomic_replace_preserves_original_and_cleans_temp(self) -> None:
        target = self.workspace / "atomic.txt"
        target.write_text("old", encoding="utf-8")

        with patch("coding_agent.tools.filesystem.os.replace", side_effect=OSError("fail")):
            result = self.tool.execute(
                {"path": "atomic.txt", "old_text": "old", "new_text": "new"}
            )

        self.assertEqual(result.metadata["kind"], "write_error")
        self.assertEqual(target.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(self.workspace.glob(".atomic.txt.*.tmp")), [])

    def test_registry_exports_replace_text_schema(self) -> None:
        target = self.workspace / "file.txt"
        target.write_text("old", encoding="utf-8")
        registry = ToolRegistry([self.tool])

        schema = registry.schemas()[0]["function"]
        result = registry.execute(
            ToolCall(
                id="replace-1",
                name="replace_text",
                arguments={"path": "file.txt", "old_text": "old", "new_text": "new"},
            )
        )

        self.assertEqual(schema["name"], "replace_text")
        self.assertEqual(
            schema["parameters"]["required"],
            ["path", "old_text", "new_text"],
        )
        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "new")


if __name__ == "__main__":
    unittest.main()

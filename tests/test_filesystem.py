import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_agent.messages import ToolCall
from coding_agent.tools.filesystem import ListFilesTool, ReadFileTool
from coding_agent.tools.registry import ToolRegistry


class ReadFileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace.mkdir()
        self.tool = ReadFileTool(self.workspace)

    def test_reads_utf8_text_file(self) -> None:
        (self.workspace / "hello.txt").write_text(
            "第一行\nsecond line\n",
            encoding="utf-8",
        )

        result = self.tool.execute({"path": "hello.txt"})

        self.assertTrue(result.success)
        self.assertEqual(result.output, "第一行\nsecond line\n")
        self.assertEqual(result.metadata["path"], "hello.txt")
        self.assertEqual(result.metadata["total_lines"], 2)
        self.assertFalse(result.metadata["truncated"])

    def test_reads_inclusive_line_range(self) -> None:
        (self.workspace / "lines.txt").write_text(
            "one\ntwo\nthree\nfour\n",
            encoding="utf-8",
        )

        result = self.tool.execute(
            {"path": "lines.txt", "start_line": 2, "end_line": 3}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output, "two\nthree\n")
        self.assertEqual(result.metadata["start_line"], 2)
        self.assertEqual(result.metadata["end_line"], 3)
        self.assertEqual(result.metadata["total_lines"], 4)

    def test_end_line_is_clipped_to_file_length(self) -> None:
        (self.workspace / "short.txt").write_text("one\ntwo\n", encoding="utf-8")

        result = self.tool.execute(
            {"path": "short.txt", "start_line": 2, "end_line": 99}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output, "two\n")
        self.assertEqual(result.metadata["end_line"], 2)

    def test_reads_empty_file(self) -> None:
        (self.workspace / "empty.txt").write_text("", encoding="utf-8")

        result = self.tool.execute({"path": "empty.txt"})

        self.assertTrue(result.success)
        self.assertEqual(result.output, "")
        self.assertEqual(result.metadata["total_lines"], 0)

    def test_reports_missing_file(self) -> None:
        result = self.tool.execute({"path": "missing.txt"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "not_found")

    def test_rejects_directory(self) -> None:
        (self.workspace / "folder").mkdir()

        result = self.tool.execute({"path": "folder"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "not_a_file")

    def test_rejects_absolute_path(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        result = self.tool.execute({"path": str(outside.resolve())})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "invalid_path")
        self.assertNotIn("secret", result.output)

    def test_rejects_parent_traversal(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        result = self.tool.execute({"path": "../outside.txt"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "path_outside_workspace")
        self.assertNotIn("secret", result.output)

    def test_rejects_symlink_that_resolves_outside_workspace(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.workspace / "link.txt"
        try:
            os.symlink(outside, link)
        except OSError as exc:
            self.skipTest(f"symlinks are unavailable: {exc}")

        result = self.tool.execute({"path": "link.txt"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "path_outside_workspace")

    def test_rejects_binary_file(self) -> None:
        (self.workspace / "binary.bin").write_bytes(b"abc\x00def")

        result = self.tool.execute({"path": "binary.bin"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "binary_file")

    def test_rejects_null_byte_after_initial_sample(self) -> None:
        (self.workspace / "late-binary.bin").write_bytes(b"a" * 9_000 + b"\x00tail")

        result = self.tool.execute({"path": "late-binary.bin"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "binary_file")

    def test_rejects_non_utf8_file(self) -> None:
        (self.workspace / "legacy.txt").write_bytes(b"\xff\xfeinvalid")

        result = self.tool.execute({"path": "legacy.txt"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "decode_error")

    def test_blocks_local_env_file(self) -> None:
        (self.workspace / ".env").write_text(
            "MODEL_API_KEY=must-not-leak",
            encoding="utf-8",
        )

        result = self.tool.execute({"path": ".env"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "sensitive_path")
        self.assertNotIn("must-not-leak", result.output)

    def test_allows_env_example_file(self) -> None:
        (self.workspace / ".env.example").write_text(
            "MODEL_API_KEY=placeholder",
            encoding="utf-8",
        )

        result = self.tool.execute({"path": ".env.example"})

        self.assertTrue(result.success)
        self.assertIn("placeholder", result.output)

    def test_truncates_long_output(self) -> None:
        tool = ReadFileTool(self.workspace, max_output_chars=30)
        (self.workspace / "long.txt").write_text("x" * 100, encoding="utf-8")

        result = tool.execute({"path": "long.txt"})

        self.assertTrue(result.success)
        self.assertTrue(result.metadata["truncated"])
        self.assertLessEqual(len(result.output), 30)
        self.assertIn("truncated", result.output)

    def test_rejects_start_line_beyond_file(self) -> None:
        (self.workspace / "short.txt").write_text("one\ntwo\n", encoding="utf-8")

        result = self.tool.execute({"path": "short.txt", "start_line": 3})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "line_out_of_range")

    def test_rejects_reversed_line_range(self) -> None:
        (self.workspace / "lines.txt").write_text("one\ntwo\n", encoding="utf-8")

        result = self.tool.execute(
            {"path": "lines.txt", "start_line": 2, "end_line": 1}
        )

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "invalid_arguments")

    def test_defensively_rejects_unexpected_arguments(self) -> None:
        result = self.tool.execute({"path": "file.txt", "secret": True})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "invalid_arguments")

    def test_registry_exports_and_executes_read_file(self) -> None:
        (self.workspace / "hello.txt").write_text("hello", encoding="utf-8")
        registry = ToolRegistry([self.tool])

        schema = registry.schemas()[0]
        result = registry.execute(
            ToolCall(id="call-1", name="read_file", arguments={"path": "hello.txt"})
        )

        self.assertEqual(schema["function"]["name"], "read_file")
        self.assertEqual(schema["function"]["parameters"]["required"], ["path"])
        self.assertTrue(result.success)
        self.assertEqual(result.output, "hello")

    def test_constructor_rejects_missing_workspace(self) -> None:
        missing = self.workspace / "missing"

        with self.assertRaisesRegex(ValueError, "workspace"):
            ReadFileTool(missing)

    def test_constructor_rejects_file_as_workspace(self) -> None:
        file_path = self.workspace / "file.txt"
        file_path.write_text("content", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "directory"):
            ReadFileTool(file_path)


class ListFilesToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace.mkdir()
        self.tool = ListFilesTool(self.workspace)

    def test_lists_top_level_entries_in_deterministic_order(self) -> None:
        (self.workspace / "b.txt").write_text("b", encoding="utf-8")
        (self.workspace / "a.txt").write_text("a", encoding="utf-8")
        source = self.workspace / "src"
        source.mkdir()
        (source / "module.py").write_text("pass", encoding="utf-8")

        result = self.tool.execute({})

        self.assertTrue(result.success)
        self.assertEqual(result.output.splitlines(), ["a.txt", "b.txt", "src/"])
        self.assertEqual(result.metadata["entries"], 3)
        self.assertEqual(result.metadata["files"], 2)
        self.assertEqual(result.metadata["directories"], 1)
        self.assertFalse(result.metadata["recursive"])

    def test_lists_subdirectory_with_workspace_relative_paths(self) -> None:
        source = self.workspace / "src"
        source.mkdir()
        (source / "module.py").write_text("pass", encoding="utf-8")

        result = self.tool.execute({"path": "src"})

        self.assertTrue(result.success)
        self.assertEqual(result.output, "src/module.py")
        self.assertEqual(result.metadata["path"], "src")

    def test_recursive_listing_honors_max_depth(self) -> None:
        source = self.workspace / "src"
        nested = source / "nested"
        nested.mkdir(parents=True)
        (self.workspace / "top.txt").write_text("top", encoding="utf-8")
        (source / "module.py").write_text("pass", encoding="utf-8")
        (nested / "deep.py").write_text("pass", encoding="utf-8")

        result = self.tool.execute({"recursive": True, "max_depth": 2})

        self.assertTrue(result.success)
        self.assertEqual(
            result.output.splitlines(),
            ["src/", "top.txt", "src/module.py", "src/nested/"],
        )
        self.assertNotIn("src/nested/deep.py", result.output)

    def test_hidden_entries_are_optional_but_secrets_stay_blocked(self) -> None:
        (self.workspace / ".hidden").write_text("hidden", encoding="utf-8")
        (self.workspace / ".gitignore").write_text("*.pyc", encoding="utf-8")
        (self.workspace / ".env").write_text("API_KEY=secret", encoding="utf-8")
        (self.workspace / ".env.example").write_text("API_KEY=", encoding="utf-8")
        (self.workspace / "visible.txt").write_text("visible", encoding="utf-8")

        default_result = self.tool.execute({})
        hidden_result = self.tool.execute({"include_hidden": True})

        self.assertEqual(default_result.output, "visible.txt")
        self.assertIn(".hidden", hidden_result.output)
        self.assertIn(".gitignore", hidden_result.output)
        self.assertIn(".env.example", hidden_result.output)
        self.assertNotIn(".env\n", hidden_result.output + "\n")
        self.assertNotIn("secret", hidden_result.output)

    def test_returns_explicit_empty_directory_result(self) -> None:
        result = self.tool.execute({})

        self.assertTrue(result.success)
        self.assertEqual(result.output, "(empty directory)")
        self.assertEqual(result.metadata["entries"], 0)

    def test_rejects_file_target(self) -> None:
        (self.workspace / "file.txt").write_text("content", encoding="utf-8")

        result = self.tool.execute({"path": "file.txt"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "not_a_directory")

    def test_rejects_missing_directory(self) -> None:
        result = self.tool.execute({"path": "missing"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "not_found")

    def test_rejects_parent_traversal(self) -> None:
        result = self.tool.execute({"path": ".."})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "path_outside_workspace")

    def test_rejects_sensitive_directory(self) -> None:
        (self.workspace / ".git").mkdir()

        result = self.tool.execute({"path": ".git"})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "sensitive_path")

    def test_rejects_invalid_depth(self) -> None:
        result = self.tool.execute({"recursive": True, "max_depth": 6})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "invalid_arguments")

    def test_truncates_by_entry_limit(self) -> None:
        for name in ("a.txt", "b.txt", "c.txt"):
            (self.workspace / name).write_text(name, encoding="utf-8")
        tool = ListFilesTool(self.workspace, max_entries=2)

        result = tool.execute({})

        self.assertTrue(result.success)
        self.assertTrue(result.metadata["truncated"])
        self.assertEqual(result.metadata["entries"], 2)
        self.assertIn("listing truncated", result.output)

    def test_registry_exports_and_executes_list_files(self) -> None:
        (self.workspace / "README.md").write_text("demo", encoding="utf-8")
        registry = ToolRegistry([self.tool])

        schema = registry.schemas()[0]
        result = registry.execute(
            ToolCall(id="call-list", name="list_files", arguments={})
        )

        self.assertEqual(schema["function"]["name"], "list_files")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "README.md")


if __name__ == "__main__":
    unittest.main()

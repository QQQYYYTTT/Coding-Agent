import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from coding_agent.messages import ToolCall
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tools.shell import RunCommandTool


class RunCommandToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace.mkdir()
        self.tool = RunCommandTool(self.workspace)
        self._write_script(
            "test_sample.py",
            "import unittest\n\n"
            "class SampleTest(unittest.TestCase):\n"
            "    def test_passes(self):\n"
            "        self.assertTrue(True)\n",
        )

    def _write_script(self, name: str, source: str) -> None:
        (self.workspace / name).write_text(source, encoding="utf-8")

    def test_runs_in_fixed_workspace_and_captures_both_streams(self) -> None:
        self._write_script(
            "inspect.py",
            "import os, sys\n"
            "print(os.getcwd())\n"
            "print('错误流', file=sys.stderr)\n",
        )

        result = self.tool.execute({"argv": ["python", "inspect.py"]})

        self.assertTrue(result.success)
        self.assertEqual(result.metadata["exit_code"], 0)
        self.assertFalse(result.metadata["timed_out"])
        self.assertIn(str(self.workspace), result.output)
        self.assertIn("错误流", result.output)
        self.assertIn("stdout:", result.output)
        self.assertIn("stderr:", result.output)

    def test_returns_nonzero_exit_with_output(self) -> None:
        self._write_script(
            "fail.py",
            "import sys\nprint('before failure')\nprint('bad', file=sys.stderr)\nraise SystemExit(7)\n",
        )

        result = self.tool.execute({"argv": ["python", "fail.py"]})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "nonzero_exit")
        self.assertEqual(result.metadata["exit_code"], 7)
        self.assertIn("before failure", result.output)
        self.assertIn("bad", result.output)

    def test_terminates_command_after_timeout(self) -> None:
        self._write_script(
            "slow.py",
            "import time\nprint('started', flush=True)\ntime.sleep(10)\n",
        )

        result = self.tool.execute(
            {"argv": ["python", "slow.py"], "timeout_seconds": 1}
        )

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "timeout")
        self.assertTrue(result.metadata["timed_out"])
        self.assertIsNone(result.metadata["exit_code"])
        self.assertIn("started", result.output)

    def test_truncates_large_stdout_and_still_keeps_stderr(self) -> None:
        self._write_script(
            "large.py",
            "import sys\nprint('x' * 10000)\nprint('important error', file=sys.stderr)\n",
        )
        tool = RunCommandTool(self.workspace, max_output_chars=200)

        result = tool.execute({"argv": ["python", "large.py"]})

        self.assertTrue(result.success)
        self.assertLessEqual(len(result.output), 200)
        self.assertTrue(result.metadata["truncated"])
        self.assertTrue(result.metadata["stdout_truncated"])
        self.assertIn("important error", result.output)
        self.assertIn("...[truncated]", result.output)

    def test_scrubs_secret_environment_variables(self) -> None:
        self._write_script(
            "environment.py",
            "import os\nprint(os.environ.get('MODEL_API_KEY', 'missing'))\n",
        )
        previous = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "must-not-leak"
        self.addCleanup(self._restore_environment, "MODEL_API_KEY", previous)

        result = self.tool.execute({"argv": ["python", "environment.py"]})

        self.assertTrue(result.success)
        self.assertIn("missing", result.output)
        self.assertNotIn("must-not-leak", result.output)

    @staticmethod
    def _restore_environment(name: str, value: str | None) -> None:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def test_blocks_commands_outside_allowlist_and_shell_launchers(self) -> None:
        result = self.tool.execute({"argv": ["powershell", "-Command", "dir"]})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "command_not_allowed")

    def test_blocks_inline_python_and_unsafe_python_modules(self) -> None:
        inline = self.tool.execute({"argv": ["python", "-c", "print('x')"]})
        pip = self.tool.execute({"argv": ["python", "-m", "pip", "install", "x"]})

        self.assertEqual(inline.metadata["kind"], "command_not_allowed")
        self.assertEqual(pip.metadata["kind"], "command_not_allowed")

    def test_blocks_python_script_outside_workspace(self) -> None:
        outside = self.workspace.parent / "outside.py"
        outside.write_text("print('outside')", encoding="utf-8")

        result = self.tool.execute({"argv": ["python", "../outside.py"]})

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["kind"], "path_outside_workspace")

    def test_allows_unittest_module(self) -> None:
        result = self.tool.execute(
            {"argv": ["python", "-m", "unittest", "discover"]}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.metadata["exit_code"], 0)

    def test_restricts_git_to_read_only_subcommands(self) -> None:
        mutation = self.tool.execute({"argv": ["git", "reset", "--hard"]})

        self.assertFalse(mutation.success)
        self.assertEqual(mutation.metadata["kind"], "command_not_allowed")

    def test_validates_timeout_and_arguments(self) -> None:
        invalid_timeout = self.tool.execute(
            {"argv": ["python", "-m", "unittest"], "timeout_seconds": 61}
        )
        empty = self.tool.execute({"argv": []})

        self.assertEqual(invalid_timeout.metadata["kind"], "invalid_arguments")
        self.assertEqual(empty.metadata["kind"], "invalid_arguments")

    def test_registry_exports_and_executes_run_command(self) -> None:
        registry = ToolRegistry([self.tool])

        schema = registry.schemas()[0]["function"]
        result = registry.execute(
            ToolCall(
                id="run-1",
                name="run_command",
                arguments={"argv": ["python", "-m", "unittest", "discover"]},
            )
        )

        self.assertEqual(schema["name"], "run_command")
        self.assertEqual(schema["parameters"]["required"], ["argv"])
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()

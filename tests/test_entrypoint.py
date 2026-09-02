import os
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InstallationEntrypointTests(unittest.TestCase):
    def test_pyproject_registers_console_command(self) -> None:
        project = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            project["project"]["scripts"]["coding-agent"],
            "coding_agent.cli:main",
        )

    def test_package_supports_module_execution(self) -> None:
        environ = os.environ.copy()
        environ["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "coding_agent", "--help"],
            cwd=PROJECT_ROOT,
            env=environ,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: coding-agent", completed.stdout)
        self.assertIn("--workspace", completed.stdout)


if __name__ == "__main__":
    unittest.main()

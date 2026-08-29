import unittest

from coding_agent.tools.base import ToolResult


class ToolResultTests(unittest.TestCase):
    def test_success_result(self) -> None:
        result = ToolResult.ok("tests passed", metadata={"exit_code": 0})
        self.assertTrue(result.success)
        self.assertEqual(result.output, "tests passed")
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["exit_code"], 0)

    def test_failed_result_can_keep_partial_output(self) -> None:
        result = ToolResult.fail(
            "command exited with status 1",
            output="partial stdout",
            metadata={"exit_code": 1},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.output, "partial stdout")
        self.assertEqual(result.error, "command exited with status 1")

    def test_failed_result_requires_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "require a non-empty error"):
            ToolResult(success=False)

    def test_success_result_rejects_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot contain an error"):
            ToolResult(success=True, error="unexpected")


if __name__ == "__main__":
    unittest.main()


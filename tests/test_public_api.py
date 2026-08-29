import unittest

import coding_agent


class PublicApiTests(unittest.TestCase):
    def test_exports_all_standard_tools(self) -> None:
        expected = {
            "ListFilesTool",
            "ReadFileTool",
            "WriteFileTool",
            "ReplaceTextTool",
            "RunCommandTool",
        }

        self.assertTrue(expected.issubset(set(coding_agent.__all__)))
        for name in expected:
            self.assertIsNotNone(getattr(coding_agent, name, None))


if __name__ == "__main__":
    unittest.main()

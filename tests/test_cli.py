import unittest

from coding_agent.agent import AgentTraceEvent, AgentTraceKind
from coding_agent.cli import build_parser, format_trace_event


class VerboseTraceTests(unittest.TestCase):
    def test_parser_enables_verbose_flag(self) -> None:
        args = build_parser().parse_args(["--verbose", "hello"])

        self.assertTrue(args.verbose)
        self.assertEqual(args.prompt, "hello")

    def test_tool_arguments_are_recursively_redacted(self) -> None:
        event = AgentTraceEvent(
            kind=AgentTraceKind.TOOL_START,
            model_turn=1,
            tool_name="example",
            arguments={
                "path": "README.md",
                "api_key": "top-secret-key",
                "nested": {
                    "access_token": "top-secret-token",
                    "password": "top-secret-password",
                },
            },
        )

        rendered = format_trace_event(event)

        self.assertIn('"path":"README.md"', rendered)
        self.assertIn("<redacted>", rendered)
        self.assertNotIn("top-secret-key", rendered)
        self.assertNotIn("top-secret-token", rendered)
        self.assertNotIn("top-secret-password", rendered)

    def test_tool_finish_displays_summary_only(self) -> None:
        event = AgentTraceEvent(
            kind=AgentTraceKind.TOOL_FINISH,
            model_turn=1,
            tool_name="read_file",
            success=True,
            summary="success; characters=42, total_lines=3, truncated=false",
        )

        rendered = format_trace_event(event)

        self.assertEqual(
            rendered,
            "[Tool] read_file success; characters=42, total_lines=3, truncated=false",
        )

    def test_edit_contents_are_omitted_from_trace(self) -> None:
        event = AgentTraceEvent(
            kind=AgentTraceKind.TOOL_START,
            model_turn=1,
            tool_name="write_file",
            arguments={
                "path": "config.py",
                "content": "API_KEY = 'must-not-appear'",
            },
        )

        rendered = format_trace_event(event)

        self.assertIn('"path":"config.py"', rendered)
        self.assertIn("27 characters omitted", rendered)
        self.assertNotIn("must-not-appear", rendered)

    def test_command_credentials_are_redacted_from_trace(self) -> None:
        event = AgentTraceEvent(
            kind=AgentTraceKind.TOOL_START,
            model_turn=1,
            tool_name="run_command",
            arguments={
                "argv": [
                    "python",
                    "script.py",
                    "--api-key",
                    "must-not-appear",
                    "--token=also-must-not-appear",
                ]
            },
        )

        rendered = format_trace_event(event)

        self.assertIn('"python"', rendered)
        self.assertIn("<redacted>", rendered)
        self.assertNotIn("must-not-appear", rendered)
        self.assertNotIn("also-must-not-appear", rendered)


if __name__ == "__main__":
    unittest.main()

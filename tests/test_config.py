import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_agent.config import AppConfig, ConfigurationError, load_env_file


class AppConfigTests(unittest.TestCase):
    def test_loads_and_normalizes_environment(self) -> None:
        config = AppConfig.from_env(
            {
                "MODEL_API_KEY": "secret-value",
                "MODEL_BASE_URL": "https://gateway.example/v1/",
                "MODEL_NAME": "example-model",
                "MODEL_TIMEOUT": "15.5",
                "MODEL_MAX_RETRIES": "4",
                "MODEL_MAX_RESPONSE_BYTES": "4096",
                "AGENT_MAX_TURNS": "12",
                "MAX_CONTEXT_CHARS": "65536",
                "MAX_NO_PROGRESS_TURNS": "4",
                "COMMAND_TIMEOUT": "30",
                "MAX_TOOL_OUTPUT": "4096",
            }
        )

        self.assertEqual(config.base_url, "https://gateway.example/v1")
        self.assertEqual(config.model, "example-model")
        self.assertEqual(config.request_timeout, 15.5)
        self.assertEqual(config.max_turns, 12)
        self.assertEqual(config.max_retries, 4)
        self.assertEqual(config.max_model_response_bytes, 4096)
        self.assertEqual(config.max_context_chars, 65536)
        self.assertEqual(config.max_no_progress_turns, 4)
        self.assertEqual(config.command_timeout, 30)
        self.assertIsInstance(config.command_timeout, int)
        self.assertEqual(config.max_tool_output, 4096)

    def test_api_key_is_not_shown_in_repr(self) -> None:
        config = AppConfig(
            api_key="do-not-log-this",
            base_url="https://api.example/v1",
            model="example-model",
        )

        self.assertNotIn("do-not-log-this", repr(config))

    def test_supports_openai_api_key_fallback(self) -> None:
        config = AppConfig.from_env(
            {
                "OPENAI_API_KEY": "fallback-key",
                "MODEL_NAME": "example-model",
            }
        )

        self.assertEqual(config.api_key, "fallback-key")

    def test_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "MODEL_API_KEY"):
            AppConfig.from_env({"MODEL_NAME": "example-model"})

    def test_rejects_invalid_numeric_value(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "AGENT_MAX_TURNS"):
            AppConfig.from_env(
                {
                    "MODEL_API_KEY": "secret",
                    "MODEL_NAME": "example-model",
                    "AGENT_MAX_TURNS": "many",
                }
            )

    def test_rejects_command_timeout_outside_supported_range(self) -> None:
        for invalid in ("0", "61", "1.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ConfigurationError, "COMMAND_TIMEOUT|command_timeout"):
                    AppConfig.from_env(
                        {
                            "MODEL_API_KEY": "secret",
                            "MODEL_NAME": "example-model",
                            "COMMAND_TIMEOUT": invalid,
                        }
                    )

    def test_rejects_context_budget_below_minimum(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "max_context_chars"):
            AppConfig.from_env(
                {
                    "MODEL_API_KEY": "secret",
                    "MODEL_NAME": "example-model",
                    "MAX_CONTEXT_CHARS": "999",
                }
            )

    def test_rejects_no_progress_threshold_below_minimum(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "max_no_progress_turns"):
            AppConfig.from_env(
                {
                    "MODEL_API_KEY": "secret",
                    "MODEL_NAME": "example-model",
                    "MAX_NO_PROGRESS_TURNS": "1",
                }
            )

    def test_enforces_minimum_tool_output(self) -> None:
        minimum = AppConfig.from_env(
            {
                "MODEL_API_KEY": "secret",
                "MODEL_NAME": "example-model",
                "MAX_TOOL_OUTPUT": "100",
            }
        )
        self.assertEqual(minimum.max_tool_output, 100)

        for invalid in ("0", "99", "1.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ConfigurationError,
                    "MAX_TOOL_OUTPUT|max_tool_output",
                ):
                    AppConfig.from_env(
                        {
                            "MODEL_API_KEY": "secret",
                            "MODEL_NAME": "example-model",
                            "MAX_TOOL_OUTPUT": invalid,
                        }
                    )

    def test_rejects_non_integer_tool_output_on_direct_config(self) -> None:
        for invalid in (True, 100.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ConfigurationError, "max_tool_output"):
                    AppConfig(
                        api_key="secret",
                        base_url="https://api.example/v1",
                        model="example-model",
                        max_tool_output=invalid,
                    )

    def test_requires_https_for_remote_model_endpoints(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "HTTPS"):
            AppConfig(
                api_key="secret",
                base_url="http://gateway.example/v1",
                model="example-model",
            )

    def test_allows_http_for_loopback_development_endpoints(self) -> None:
        for base_url in (
            "http://localhost:8000/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
        ):
            with self.subTest(base_url=base_url):
                config = AppConfig(
                    api_key="secret",
                    base_url=base_url,
                    model="example-model",
                )
                self.assertEqual(config.base_url, base_url)

    def test_enforces_model_retry_range(self) -> None:
        disabled = AppConfig.from_env(
            {
                "MODEL_API_KEY": "secret",
                "MODEL_NAME": "example-model",
                "MODEL_MAX_RETRIES": "0",
            }
        )
        self.assertEqual(disabled.max_retries, 0)

        for invalid in ("-1", "6", "1.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ConfigurationError,
                    "MODEL_MAX_RETRIES|max_retries",
                ):
                    AppConfig.from_env(
                        {
                            "MODEL_API_KEY": "secret",
                            "MODEL_NAME": "example-model",
                            "MODEL_MAX_RETRIES": invalid,
                        }
                    )

    def test_enforces_model_response_size_range(self) -> None:
        minimum = AppConfig.from_env(
            {
                "MODEL_API_KEY": "secret",
                "MODEL_NAME": "example-model",
                "MODEL_MAX_RESPONSE_BYTES": "1024",
            }
        )
        self.assertEqual(minimum.max_model_response_bytes, 1024)

        for invalid in ("1023", "10000001", "1.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ConfigurationError,
                    "MODEL_MAX_RESPONSE_BYTES|max_model_response_bytes",
                ):
                    AppConfig.from_env(
                        {
                            "MODEL_API_KEY": "secret",
                            "MODEL_NAME": "example-model",
                            "MODEL_MAX_RESPONSE_BYTES": invalid,
                        }
                    )

    def test_loads_dotenv_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# local secrets\n"
                "MODEL_API_KEY='file-secret'\n"
                "MODEL_BASE_URL=https://gateway.example/v1/\n"
                'MODEL_NAME="file-model"\n',
                encoding="utf-8",
            )

            config = AppConfig.from_env({}, dotenv_path=path)

        self.assertEqual(config.api_key, "file-secret")
        self.assertEqual(config.base_url, "https://gateway.example/v1")
        self.assertEqual(config.model, "file-model")

    def test_process_values_override_dotenv_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "MODEL_API_KEY=file-secret\nMODEL_NAME=file-model\n",
                encoding="utf-8",
            )

            config = AppConfig.from_env(
                {
                    "MODEL_API_KEY": "process-secret",
                    "MODEL_NAME": "process-model",
                },
                dotenv_path=path,
            )

        self.assertEqual(config.api_key, "process-secret")
        self.assertEqual(config.model, "process-model")

    def test_dotenv_error_does_not_reveal_value(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("INVALID NAME=super-secret\n", encoding="utf-8")

            with self.assertRaises(ConfigurationError) as captured:
                load_env_file(path)

        self.assertNotIn("super-secret", str(captured.exception))


if __name__ == "__main__":
    unittest.main()

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
                "AGENT_MAX_TURNS": "12",
                "COMMAND_TIMEOUT": "30",
                "MAX_TOOL_OUTPUT": "4096",
            }
        )

        self.assertEqual(config.base_url, "https://gateway.example/v1")
        self.assertEqual(config.model, "example-model")
        self.assertEqual(config.request_timeout, 15.5)
        self.assertEqual(config.max_turns, 12)
        self.assertEqual(config.command_timeout, 30.0)
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

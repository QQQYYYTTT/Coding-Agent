"""Validated environment-backed configuration for the coding agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """Raised when required configuration is missing or invalid."""


_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(path: Path) -> dict[str, str]:
    """Read a small dotenv file without adding a runtime dependency.

    Blank lines and lines beginning with ``#`` are ignored. Values may be
    unquoted or wrapped in matching single/double quotes. Secret values are
    never included in parser errors.
    """

    if not path.exists():
        return {}
    if not path.is_file():
        raise ConfigurationError(f"dotenv path is not a file: {path}")

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"could not read dotenv file: {path}") from exc

    values: dict[str, str] = {}
    for line_number, original_line in enumerate(lines, start=1):
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(
                f"invalid dotenv entry at {path}:{line_number}: missing '='"
            )

        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME_PATTERN.fullmatch(name):
            raise ConfigurationError(
                f"invalid dotenv variable name at {path}:{line_number}"
            )

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"missing required environment variable: {name}")
    return value


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw_value = environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _non_negative_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw_value = environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigurationError(f"{name} must be zero or greater")
    return value


def _positive_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw_value = environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime settings loaded from environment variables.

    The API key is excluded from ``repr`` to prevent accidental disclosure in
    logs or tracebacks. ``MODEL_API_KEY`` takes precedence over the conventional
    ``OPENAI_API_KEY`` fallback.
    """

    api_key: str = field(repr=False)
    base_url: str
    model: str
    request_timeout: float = 60.0
    max_retries: int = 2
    max_model_response_bytes: int = 2_000_000
    max_turns: int = 20
    max_context_chars: int = 100_000
    max_no_progress_turns: int = 3
    command_timeout: int = 60
    max_tool_output: int = 20_000

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ConfigurationError("api_key must not be empty")
        if not self.base_url.strip():
            raise ConfigurationError("base_url must not be empty")
        if not self.model.strip():
            raise ConfigurationError("model must not be empty")
        parsed_url = urlsplit(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ConfigurationError("base_url must be an absolute HTTP(S) URL")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ConfigurationError("base_url must not contain credentials")
        if parsed_url.query or parsed_url.fragment:
            raise ConfigurationError("base_url must not contain a query or fragment")
        try:
            parsed_url.port
        except ValueError as exc:
            raise ConfigurationError("base_url contains an invalid port") from exc
        if (
            parsed_url.scheme != "https"
            and parsed_url.hostname.lower() not in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ConfigurationError(
                "base_url must use HTTPS unless it targets a loopback host"
            )
        if self.request_timeout <= 0:
            raise ConfigurationError("request_timeout must be greater than zero")
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or not 0 <= self.max_retries <= 5
        ):
            raise ConfigurationError("max_retries must be an integer from 0 to 5")
        if (
            not isinstance(self.max_model_response_bytes, int)
            or isinstance(self.max_model_response_bytes, bool)
            or not 1_024 <= self.max_model_response_bytes <= 10_000_000
        ):
            raise ConfigurationError(
                "max_model_response_bytes must be an integer from 1024 to 10000000"
            )
        if self.max_turns <= 0:
            raise ConfigurationError("max_turns must be greater than zero")
        if (
            not isinstance(self.max_context_chars, int)
            or isinstance(self.max_context_chars, bool)
            or self.max_context_chars < 1_000
        ):
            raise ConfigurationError("max_context_chars must be an integer of at least 1000")
        if (
            not isinstance(self.max_no_progress_turns, int)
            or isinstance(self.max_no_progress_turns, bool)
            or self.max_no_progress_turns < 2
        ):
            raise ConfigurationError(
                "max_no_progress_turns must be an integer of at least 2"
            )
        if (
            not isinstance(self.command_timeout, int)
            or isinstance(self.command_timeout, bool)
            or not 1 <= self.command_timeout <= 60
        ):
            raise ConfigurationError("command_timeout must be an integer from 1 to 60")
        if (
            not isinstance(self.max_tool_output, int)
            or isinstance(self.max_tool_output, bool)
            or self.max_tool_output < 100
        ):
            raise ConfigurationError("max_tool_output must be an integer of at least 100")

        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        dotenv_path: Path | None = None,
    ) -> "AppConfig":
        """Create configuration from process variables and a local ``.env``.

        When ``environ`` is omitted, values are first loaded from ``.env`` in
        the current working directory and then overlaid with process environment
        variables. This means an explicitly exported variable always wins.
        Passing ``environ`` keeps tests and embedded use deterministic; a dotenv
        file is only added in that mode when ``dotenv_path`` is also provided.
        """

        file_values: dict[str, str] = {}
        if environ is None or dotenv_path is not None:
            file_values = load_env_file(dotenv_path or Path.cwd() / ".env")
        source = {**file_values, **(os.environ if environ is None else environ)}
        api_key = source.get("MODEL_API_KEY", "").strip()
        if not api_key:
            api_key = source.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "missing required environment variable: MODEL_API_KEY "
                "(or OPENAI_API_KEY)"
            )

        return cls(
            api_key=api_key,
            base_url=source.get(
                "MODEL_BASE_URL",
                "https://api.openai.com/v1",
            ).strip(),
            model=_required(source, "MODEL_NAME"),
            request_timeout=_positive_float(source, "MODEL_TIMEOUT", 60.0),
            max_retries=_non_negative_int(source, "MODEL_MAX_RETRIES", 2),
            max_model_response_bytes=_positive_int(
                source,
                "MODEL_MAX_RESPONSE_BYTES",
                2_000_000,
            ),
            max_turns=_positive_int(source, "AGENT_MAX_TURNS", 20),
            max_context_chars=_positive_int(
                source,
                "MAX_CONTEXT_CHARS",
                100_000,
            ),
            max_no_progress_turns=_positive_int(
                source,
                "MAX_NO_PROGRESS_TURNS",
                3,
            ),
            command_timeout=_positive_int(source, "COMMAND_TIMEOUT", 60),
            max_tool_output=_positive_int(source, "MAX_TOOL_OUTPUT", 20_000),
        )

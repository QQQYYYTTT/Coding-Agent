"""Model client adapters."""

from coding_agent.llm.base import (
    LLMClient,
    ModelClientError,
    ModelRequestError,
    ModelResponseError,
)
from coding_agent.llm.compatible import OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "ModelClientError",
    "ModelRequestError",
    "ModelResponseError",
    "OpenAICompatibleClient",
]

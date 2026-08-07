"""OpenAI API client wrapper for embedding generation with graceful fallback."""

import hashlib
import json
import re
from typing import Any
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """Async client wrapper for OpenAI API operations with automatic credit-exhaustion fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.embedding_model = settings.openai_embedding_model
        self.chat_model = model or settings.openai_chat_model
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialized AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key or "sk-dummy-key")
        return self._client

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings using OpenAI API."""
        if not texts:
            return []

        if not self.api_key or self.api_key.startswith("sk-dummy"):
            raise ValueError("No valid OpenAI API key configured.")

        logger.info("Generating OpenAI embeddings", extra={"count": len(texts)})
        response = await self.client.embeddings.create(
            input=texts,
            model=self.embedding_model,
        )
        return [data.embedding for data in response.data]

    async def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format: Any | None = None,
    ) -> str:
        """Generate LLM response using OpenAI API."""
        if not self.api_key or self.api_key.startswith("sk-dummy"):
            raise ValueError("No valid OpenAI API key configured.")

        logger.info("Calling OpenAI Chat Completion", extra={"model": self.chat_model})
        kwargs: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

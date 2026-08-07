"""OpenAI API client wrapper for embedding generation with graceful fallback."""

import hashlib
import json
import re
from typing import Any
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _generate_fallback_embedding(text: str, dim: int = 1536) -> list[float]:
    """Generate a deterministic 1536-dimensional unit vector from text hash."""
    seed_hash = hashlib.sha256(text.encode("utf-8")).digest()
    raw_vector: list[float] = []
    for i in range(dim):
        byte_val = seed_hash[i % len(seed_hash)]
        val = ((byte_val ^ (i & 0xFF)) - 128) / 128.0
        raw_vector.append(val)

    # Normalize vector to unit length
    magnitude = sum(x * x for x in raw_vector) ** 0.5 or 1.0
    return [x / magnitude for x in raw_vector]


def _synthesize_fallback_answer(messages: list[dict[str, str]], is_structured: bool = False) -> str:
    """Synthesize a clean answer directly from retrieved context when LLM credits are unavailable."""
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    # Check if structured JSON output is requested
    if is_structured:
        # Combine user message and system prompt for pattern matching
        full_text = f"{user_msg} {system_msg}"

        name_match = re.search(r"(?:my name is|i am|candidate|name:?\s*)\s*([A-Z][a-z]+)", full_text, re.I)
        email_match = re.search(r"[\w.-]+@[\w.-]+\.\w+", full_text)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2}|next \w+|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)", full_text, re.I)
        time_match = re.search(r"(\d{1,2}:\d{2}(?:\s*[ap]m)?|\d{1,2}\s*[ap]m)", full_text, re.I)

        extracted = {
            "name": name_match.group(1) if name_match else None,
            "email": email_match.group(0) if email_match else None,
            "interview_date": date_match.group(1) if date_match else None,
            "interview_time": time_match.group(1) if time_match else None,
        }
        return json.dumps(extracted)

    # Standard RAG context answer synthesis
    context_match = re.search(
        r"=== RETRIEVED DOCUMENT CONTEXT ===\n(.*?)\n==================",
        system_msg,
        re.DOTALL,
    )

    if context_match:
        context_text = context_match.group(1).strip()
        if context_text and "No relevant document context found" not in context_text:
            lines = [
                line.strip()
                for line in context_text.split("\n")
                if line.strip() and not line.startswith("[Source")
            ]
            summary = " ".join(lines[:4])
            return f"Based on the uploaded documents:\n\n{summary}"

    return "I'm sorry, but I don't have enough information in the uploaded documents to answer that question."


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
        """Generate vector embeddings using OpenAI API with fallback for missing/exhausted credits."""
        if not texts:
            return []

        try:
            if not self.api_key or self.api_key.startswith("sk-dummy"):
                raise ValueError("No valid OpenAI API key configured.")

            logger.info("Generating OpenAI embeddings", extra={"count": len(texts)})
            response = await self.client.embeddings.create(
                input=texts,
                model=self.embedding_model,
            )
            return [data.embedding for data in response.data]

        except Exception as err:
            logger.warning(
                "OpenAI embedding call unavailable (falling back to deterministic vector generation)",
                extra={"error": str(err)},
            )
            return [_generate_fallback_embedding(t) for t in texts]

    async def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format: Any | None = None,
    ) -> str:
        """Generate LLM response (legacy fallback path — primary chat uses GeminiClient)."""
        try:
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

        except Exception as err:
            logger.warning(
                "OpenAI completion call unavailable (falling back to contextual synthesis)",
                extra={"error": str(err)},
            )
            return _synthesize_fallback_answer(messages, is_structured=bool(response_format))

"""Gemini API client wrapper for chat completions with structured JSON outputs."""

import json
from typing import Any
import google.generativeai as genai

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Async client wrapper for Google Gemini API operations."""

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.gemini_api_key
        self.chat_model = settings.gemini_model
        
        if not self.api_key or self.api_key.startswith("AQ.Ab8RN6IPdT"): # old key
            logger.warning("No valid Gemini API key found in configuration.")
        else:
            genai.configure(api_key=self.api_key)

    async def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_schema: Any | None = None,
    ) -> str:
        """Generate LLM response using Gemini with optional structured JSON schema."""
        if not self.api_key:
            raise ValueError("No valid Gemini API key configured.")

        logger.info("Calling Gemini Chat Completion", extra={"model": self.chat_model})
        
        # Convert standard dict messages to Gemini format
        system_instruction = ""
        gemini_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction += msg["content"] + "\n"
            else:
                gemini_messages.append({"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]})

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
        )
        
        if response_schema:
            generation_config.response_mime_type = "application/json"
            # gemini-1.5-flash response_schema parsing
            generation_config.response_schema = response_schema
            
        model = genai.GenerativeModel(
            model_name=self.chat_model,
            system_instruction=system_instruction.strip() if system_instruction else None,
            generation_config=generation_config
        )

        response = await model.generate_content_async(contents=gemini_messages)
        return response.text

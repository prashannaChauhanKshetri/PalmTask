"""Redis-backed chat memory service managing windowed conversation history and booking state."""

import json
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChatTurn(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PartialBookingState(BaseModel):
    """Partially extracted booking data tracked across chat turns."""

    name: str | None = None
    email: str | None = None
    interview_date: str | None = None  # ISO format string YYYY-MM-DD
    interview_time: str | None = None  # ISO format string HH:MM:SS or HH:MM

    def is_complete(self) -> bool:
        """Check if all 4 required booking fields are present."""
        return bool(
            self.name and self.email and self.interview_date and self.interview_time
        )

    def missing_fields(self) -> list[str]:
        """Return list of field names that are still missing."""
        missing: list[str] = []
        if not self.name:
            missing.append("name")
        if not self.email:
            missing.append("email")
        if not self.interview_date:
            missing.append("interview date")
        if not self.interview_time:
            missing.append("interview time")
        return missing


# In-memory storage fallback for local manual testing when Redis container is starting up
_fallback_chat_memory: dict[str, list[ChatTurn]] = {}
_fallback_booking_memory: dict[str, PartialBookingState] = {}


class MemoryService:
    """Service managing windowed chat history and partial booking state in Redis (with in-memory fallback)."""

    def __init__(self, redis_client: aioredis.Redis | None = None) -> None:
        settings = get_settings()
        self.window_size = settings.chat_window_size
        self.ttl_seconds = settings.chat_ttl_seconds
        self._redis = redis_client

    @property
    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            from app.clients.redis_client import RedisClient

            self._redis = RedisClient().redis
        return self._redis

    def _chat_key(self, session_id: UUID | str) -> str:
        return f"chat:{session_id}"

    def _booking_key(self, session_id: UUID | str) -> str:
        return f"booking_state:{session_id}"

    async def get_history(self, session_id: UUID | str) -> list[ChatTurn]:
        """Retrieve windowed conversation turns for a session."""
        key = self._chat_key(session_id)
        try:
            raw_items = await self.redis.lrange(key, 0, -1)
            await self.redis.expire(key, self.ttl_seconds)

            turns: list[ChatTurn] = []
            for item in raw_items:
                try:
                    data = json.loads(item)
                    turns.append(ChatTurn.model_validate(data))
                except Exception:
                    pass
            return turns
        except Exception:
            return _fallback_chat_memory.get(str(session_id), [])

    async def add_turn(
        self, session_id: UUID | str, role: Literal["user", "assistant"], content: str
    ) -> ChatTurn:
        """Append a new message turn to session history and cap window to last N turns."""
        key = self._chat_key(session_id)
        turn = ChatTurn(role=role, content=content)
        turn_json = turn.model_dump_json()

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.rpush(key, turn_json)
                pipe.ltrim(key, -self.window_size, -1)
                pipe.expire(key, self.ttl_seconds)
                await pipe.execute()
        except Exception:
            if str(session_id) not in _fallback_chat_memory:
                _fallback_chat_memory[str(session_id)] = []
            _fallback_chat_memory[str(session_id)].append(turn)
            _fallback_chat_memory[str(session_id)] = _fallback_chat_memory[str(session_id)][-self.window_size:]

        return turn

    async def get_booking_state(self, session_id: UUID | str) -> PartialBookingState:
        """Retrieve ongoing partial booking state for a session."""
        key = self._booking_key(session_id)
        try:
            raw_data = await self.redis.get(key)
            if not raw_data:
                return PartialBookingState()
            data = json.loads(raw_data)
            return PartialBookingState.model_validate(data)
        except Exception:
            return _fallback_booking_memory.get(str(session_id), PartialBookingState())

    async def save_booking_state(
        self, session_id: UUID | str, state: PartialBookingState
    ) -> None:
        """Save or update partial booking state."""
        key = self._booking_key(session_id)
        state_json = state.model_dump_json()
        try:
            await self.redis.set(key, state_json, ex=self.ttl_seconds)
        except Exception:
            _fallback_booking_memory[str(session_id)] = state

    async def clear_booking_state(self, session_id: UUID | str) -> None:
        """Delete partial booking state after successful booking confirmation."""
        key = self._booking_key(session_id)
        try:
            await self.redis.delete(key)
        except Exception:
            _fallback_booking_memory.pop(str(session_id), None)

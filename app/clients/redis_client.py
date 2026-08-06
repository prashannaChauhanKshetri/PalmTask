"""Redis client wrapper using redis.asyncio for async operations."""

from typing import Any
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client wrapper supporting connection pooling and lifecycle management."""

    def __init__(self, redis_url: str | None = None) -> None:
        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self._redis: aioredis.Redis | None = None

    @property
    def redis(self) -> aioredis.Redis:
        """Lazy-initialized async Redis connection."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def close(self) -> None:
        """Close connection pool."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

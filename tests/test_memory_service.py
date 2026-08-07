"""Unit tests for Redis-backed MemoryService using AsyncMock."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.memory_service import MemoryService, PartialBookingState


@pytest.mark.asyncio
async def test_memory_add_and_get_history() -> None:
    mock_redis = AsyncMock()
    mock_redis.lrange = AsyncMock(
        return_value=[
            json.dumps({"role": "user", "content": "Hello", "timestamp": "2026-08-07T00:00:00Z"}),
            json.dumps({"role": "assistant", "content": "Hi there!", "timestamp": "2026-08-07T00:00:01Z"}),
        ]
    )

    mock_pipeline = AsyncMock()
    mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    mock_pipeline.rpush = MagicMock()
    mock_pipeline.ltrim = MagicMock()
    mock_pipeline.expire = MagicMock()
    mock_pipeline.execute = AsyncMock()

    service = MemoryService(redis_client=mock_redis)
    session_id = uuid.uuid4()

    # Test get history
    history = await service.get_history(session_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Hello"
    assert history[1].role == "assistant"
    assert history[1].content == "Hi there!"

    # Test add turn
    turn = await service.add_turn(session_id, "user", "Book an interview")
    assert turn.role == "user"
    assert turn.content == "Book an interview"


@pytest.mark.asyncio
async def test_partial_booking_state_complete_check() -> None:
    state = PartialBookingState(
        name="Prashanna",
        email="prashanna@example.com",
        interview_date="2026-08-10",
        interview_time="14:00:00",
    )
    assert state.is_complete() is True
    assert len(state.missing_fields()) == 0

    partial_state = PartialBookingState(name="Prashanna")
    assert partial_state.is_complete() is False
    assert "email" in partial_state.missing_fields()
    assert "interview date" in partial_state.missing_fields()

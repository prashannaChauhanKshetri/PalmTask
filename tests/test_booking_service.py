"""Unit tests for BookingService structured extraction and date/email validation."""

import json
import uuid
from datetime import date, time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.booking import Booking
from app.services.booking_service import BookingService
from app.services.memory_service import ChatTurn, PartialBookingState


@pytest.mark.asyncio
async def test_booking_intent_extraction_partial() -> None:
    mock_repo = AsyncMock()
    mock_memory = AsyncMock()
    mock_memory.get_booking_state = AsyncMock(return_value=PartialBookingState())
    mock_memory.save_booking_state = AsyncMock()

    mock_openai = AsyncMock()
    # LLM extracts name and email first
    mock_openai.generate_chat_completion = AsyncMock(
        return_value=json.dumps({"name": "Prashanna", "email": "prashanna@example.com"})
    )

    service = BookingService(
        booking_repo=mock_repo,
        memory_service=mock_memory,
        openai_client=mock_openai,
    )

    session_id = uuid.uuid4()
    history = [ChatTurn(role="user", content="I want to book an interview")]

    result = await service.detect_intent_and_extract(
        session_id=session_id,
        user_message="I want to book an interview",
        history=history,
    )

    assert result.is_booking_intent is True
    assert result.is_complete is False
    assert "interview date" in result.response_text or "interview time" in result.response_text


@pytest.mark.asyncio
async def test_booking_intent_extraction_complete() -> None:
    mock_repo = AsyncMock()
    created_booking = Booking(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        name="Prashanna",
        email="prashanna@example.com",
        interview_date=date(2026, 8, 10),
        interview_time=time(14, 0),
        status="confirmed",
    )
    mock_repo.create_booking = AsyncMock(return_value=created_booking)

    mock_memory = AsyncMock()
    # State has all 4 fields captured
    mock_memory.get_booking_state = AsyncMock(
        return_value=PartialBookingState(
            name="Prashanna",
            email="prashanna@example.com",
            interview_date="2026-08-10",
            interview_time="14:00",
        )
    )
    mock_memory.save_booking_state = AsyncMock()
    mock_memory.clear_booking_state = AsyncMock()

    mock_openai = AsyncMock()
    mock_openai.generate_chat_completion = AsyncMock(
        return_value=json.dumps(
            {
                "name": "Prashanna",
                "email": "prashanna@example.com",
                "interview_date": "2026-08-10",
                "interview_time": "14:00",
            }
        )
    )

    service = BookingService(
        booking_repo=mock_repo,
        memory_service=mock_memory,
        openai_client=mock_openai,
    )

    session_id = uuid.uuid4()
    result = await service.detect_intent_and_extract(
        session_id=session_id,
        user_message="Confirm my interview for 2026-08-10 at 14:00",
        history=[],
    )

    assert result.is_booking_intent is True
    assert result.is_complete is True
    assert "successfully booked" in result.response_text
    assert result.booking is not None
    assert result.booking.name == "Prashanna"

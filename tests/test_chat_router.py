"""Integration tests for Conversational RAG and Booking Chat API endpoints."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.main import app
from app.models.chat_session import ChatSession
from app.services.booking_service import BookingProcessResult
from app.services.memory_service import ChatTurn
from app.services.rag_generator import RAGResponse, SourceMetadata


@pytest.fixture
def mock_session() -> ChatSession:
    return ChatSession(
        id=uuid.uuid4(),
        created_at=datetime.now(timezone.utc),
        last_active_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_chat_message_new_session(mock_session: ChatSession) -> None:
    mock_db_session = AsyncMock()

    async def override_get_db() -> AsyncMock:
        yield mock_db_session

    app.dependency_overrides[get_db] = override_get_db

    rag_resp = RAGResponse(
        answer="PalmMind AI is a technical solutions platform.",
        sources=[
            SourceMetadata(
                document_id="doc1",
                chunk_index=0,
                score=0.91,
                text_preview="PalmMind AI info...",
            )
        ],
    )

    with patch(
        "app.repositories.chat_session_repository.ChatSessionRepository.create_session",
        new_callable=AsyncMock,
        return_value=mock_session,
    ), patch(
        "app.repositories.chat_session_repository.ChatSessionRepository.update_last_active",
        new_callable=AsyncMock,
    ), patch(
        "app.services.memory_service.MemoryService.get_history",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.services.memory_service.MemoryService.add_turn",
        new_callable=AsyncMock,
    ), patch(
        "app.services.booking_service.BookingService.detect_intent_and_extract",
        new_callable=AsyncMock,
        return_value=BookingProcessResult(
            is_booking_intent=False, is_complete=False, response_text=""
        ),
    ), patch(
        "app.services.rag_generator.RAGGenerator.generate_response",
        new_callable=AsyncMock,
        return_value=rag_resp,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            payload = {"message": "What is PalmMind AI?"}
            response = await ac.post("/api/v1/chat", json=payload)

            assert response.status_code == 200
            resp_data = response.json()
            assert resp_data["session_id"] == str(mock_session.id)
            assert resp_data["answer"] == "PalmMind AI is a technical solutions platform."
            assert len(resp_data["sources"]) == 1
            assert resp_data["sources"][0]["document_id"] == "doc1"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_history(mock_session: ChatSession) -> None:
    mock_db_session = AsyncMock()

    async def override_get_db() -> AsyncMock:
        yield mock_db_session

    app.dependency_overrides[get_db] = override_get_db

    history = [
        ChatTurn(role="user", content="Hello", timestamp="2026-08-07T00:00:00Z"),
        ChatTurn(role="assistant", content="Hi!", timestamp="2026-08-07T00:00:01Z"),
    ]

    with patch(
        "app.repositories.chat_session_repository.ChatSessionRepository.get_session_by_id",
        new_callable=AsyncMock,
        return_value=mock_session,
    ), patch(
        "app.services.memory_service.MemoryService.get_history",
        new_callable=AsyncMock,
        return_value=history,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get(f"/api/v1/chat/{mock_session.id}/history")

            assert response.status_code == 200
            resp_data = response.json()
            assert resp_data["session_id"] == str(mock_session.id)
            assert len(resp_data["history"]) == 2
            assert resp_data["history"][0]["content"] == "Hello"

    app.dependency_overrides.clear()

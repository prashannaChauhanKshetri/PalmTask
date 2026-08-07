"""FastAPI router for Conversational RAG and Interview Booking Chat API endpoints."""

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import SessionNotFoundError
from app.repositories.booking_repository import BookingRepository
from app.repositories.chat_session_repository import ChatSessionRepository
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ChatTurnSchema,
    SourceMetadataSchema,
)
from app.services.booking_service import BookingService
from app.services.memory_service import MemoryService
from app.services.rag_generator import RAGGenerator

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a conversational RAG message or interview booking intent",
)
async def chat_message(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Conversational endpoint handling vector RAG queries and multi-turn interview bookings."""
    session_repo = ChatSessionRepository(db)
    booking_repo = BookingRepository(db)
    memory_service = MemoryService()

    # 1. Resolve or create chat session
    if payload.session_id is None:
        session = await session_repo.create_session()
        session_id = session.id
    else:
        session_id = payload.session_id
        session = await session_repo.get_session_by_id(session_id)
        if not session:
            # Create session if not found to ensure graceful session recovery
            session = await session_repo.create_session(session_id=session_id)

    # 2. Retrieve windowed history from Redis
    history = await memory_service.get_history(session_id)

    # 3. Check for Interview Booking Intent
    booking_service = BookingService(
        booking_repo=booking_repo,
        memory_service=memory_service,
    )
    booking_result = await booking_service.detect_intent_and_extract(
        session_id=session_id,
        user_message=payload.message,
        history=history,
    )

    if booking_result.is_booking_intent:
        answer = booking_result.response_text
        sources: list[SourceMetadataSchema] = []
        retrieval_stats: dict[str, int] = {}
    else:
        # 4. Standard Custom RAG Pipeline Generation (with semantic search)
        rag = RAGGenerator()
        rag_output = await rag.generate_response(
            user_message=payload.message,
            history=history,
        )
        answer = rag_output.answer
        sources = [
            SourceMetadataSchema(
                document_id=src.document_id,
                chunk_index=src.chunk_index,
                score=src.score,
                text_preview=src.text_preview,
            )
            for src in rag_output.sources
        ]
        retrieval_stats = {
            "filtered_count": rag_output.filtered_count,
            "expanded_count": rag_output.expanded_count,
        }

    # 5. Append turn pair to Redis chat memory
    await memory_service.add_turn(session_id, "user", payload.message)
    await memory_service.add_turn(session_id, "assistant", answer)

    # 6. Touch DB session last_active_at timestamp
    await session_repo.update_last_active(session_id)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
        retrieval_stats=retrieval_stats,
    )


@router.get(
    "/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get windowed Redis chat history for a session",
)
async def get_chat_history(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    """Retrieve the Redis-backed windowed conversation history for demo/debugging."""
    session_repo = ChatSessionRepository(db)
    session = await session_repo.get_session_by_id(session_id)
    if not session:
        raise SessionNotFoundError(session_id)

    memory_service = MemoryService()
    history_turns = await memory_service.get_history(session_id)

    return ChatHistoryResponse(
        session_id=session_id,
        history=[
            ChatTurnSchema(
                role=turn.role,
                content=turn.content,
                timestamp=turn.timestamp,
            )
            for turn in history_turns
        ],
    )

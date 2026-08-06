"""Pydantic schemas for Conversational RAG API endpoints."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Request payload for conversational chat search and interview booking."""

    session_id: UUID | None = Field(
        default=None,
        description="Optional session UUID. If omitted, a new chat session is automatically created.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="User search query or interview booking message.",
    )


class SourceMetadataSchema(BaseModel):
    """Retrieved vector chunk metadata attribution."""

    document_id: str
    chunk_index: int
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    """Response payload for chat message queries."""

    session_id: UUID
    answer: str
    sources: list[SourceMetadataSchema] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ChatTurnSchema(BaseModel):
    """Single turn representation in session history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str


class ChatHistoryResponse(BaseModel):
    """Response payload returning Redis windowed session history."""

    session_id: UUID
    history: list[ChatTurnSchema]

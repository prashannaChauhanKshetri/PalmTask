"""Pydantic schemas for Document Ingestion API endpoints."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadResponse(BaseModel):
    """Response returned immediately upon document upload enqueue."""

    document_id: UUID
    filename: str
    status: str = Field(description="'pending' | 'processing' | 'completed' | 'failed'")
    total_chunks: int = 0
    message: str = "Document upload accepted and processing in background."

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    """Detailed document status and metadata response."""

    id: UUID
    filename: str
    file_type: str
    chunking_strategy: Literal["fixed", "recursive"]
    status: str
    total_chunks: int
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Paginated list of ingested documents."""

    items: list[DocumentResponse]
    total: int
    skip: int
    limit: int

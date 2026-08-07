"""Pydantic schemas for interview booking extraction and API responses."""

from datetime import date, datetime, time
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LLMExtractedBooking(BaseModel):
    """Structured JSON schema for OpenAI-powered interview slot extraction."""

    name: str | None = Field(default=None, description="Candidate full name if mentioned")
    email: str | None = Field(default=None, description="Candidate email address if mentioned")
    interview_date: str | None = Field(
        default=None,
        description="Interview date string (e.g. '2026-08-10', 'next Tuesday', 'tomorrow')",
    )
    interview_time: str | None = Field(
        default=None,
        description="Interview time string (e.g. '14:00', '2:30 PM', '10am')",
    )


class BookingResponse(BaseModel):
    """Response payload for confirmed interview bookings."""

    id: UUID
    session_id: UUID | None
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time
    status: Literal["pending", "confirmed", "cancelled"]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

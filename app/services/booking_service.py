"""Interview booking extraction service using OpenAI structured outputs, dateparser, and email-validator."""

import json
from dataclasses import dataclass, field
from datetime import date, time
from typing import Any
from uuid import UUID

import dateparser
from email_validator import EmailNotValidError, validate_email

from app.clients.openai_client import OpenAIClient
from app.core.exceptions import BookingValidationError
from app.core.logging import get_logger
from app.models.booking import Booking
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import LLMExtractedBooking
from app.services.memory_service import ChatTurn, MemoryService, PartialBookingState

logger = get_logger(__name__)


@dataclass(frozen=True)
class BookingProcessResult:
    """Outcome of booking extraction processing for a turn."""

    is_booking_intent: bool
    is_complete: bool
    response_text: str
    booking: Booking | None = None
    missing_fields: list[str] = field(default_factory=list)


class BookingService:
    """Service managing multi-turn interview booking extraction, date/email validation, and DB creation."""

    def __init__(
        self,
        booking_repo: BookingRepository,
        memory_service: MemoryService,
        openai_client: OpenAIClient | None = None,
    ) -> None:
        self.repo = booking_repo
        self.memory = memory_service
        self.openai_client = openai_client or OpenAIClient()

    async def detect_intent_and_extract(
        self,
        session_id: UUID | str,
        user_message: str,
        history: list[ChatTurn],
    ) -> BookingProcessResult:
        """Process user turn for interview booking intent and extract missing slot fields across turns."""
        existing_state = await self.memory.get_booking_state(session_id)

        # Check intent keywords or existing active booking state
        booking_keywords = ["book", "interview", "schedule", "appointment", "slot", "date", "time"]
        has_keyword = any(kw in user_message.lower() for kw in booking_keywords)
        has_active_state = any(
            [existing_state.name, existing_state.email, existing_state.interview_date, existing_state.interview_time]
        )

        if not (has_keyword or has_active_state):
            return BookingProcessResult(
                is_booking_intent=False,
                is_complete=False,
                response_text="",
            )

        logger.info(
            "Booking intent detected",
            extra={"session_id": str(session_id), "user_message": user_message},
        )

        # Build prompt for LLM structured extraction
        conversation_context = "\n".join([f"{t.role}: {t.content}" for t in history])
        if not history or history[-1].content != user_message:
            conversation_context += f"\nuser: {user_message}"

        extraction_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a structured data extractor. Your job is to extract interview booking details "
                    "from the conversation context. Extract any candidate name, email address, requested date, "
                    "and requested time. "
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. If a detail is missing or not provided by the user, you MUST return the exact string 'null' for that field. Do NOT guess or hallucinate.\n"
                    "2. If a date is provided (e.g. 'Next Monday', '9th to 15th Aug', 'tomorrow'), you MUST convert it to a standard 'YYYY-MM-DD' format based on the current year (assume 2026 if unclear). If it's a range, pick the first available date in the range.\n"
                    "3. If a time is provided (e.g. '2-3pm', '10am'), you MUST convert it to a standard 24-hour 'HH:MM' format (e.g. '14:00', '10:00')."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation history:\n{conversation_context}",
            },
        ]

        booking_schema = {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "email": {"type": "STRING"},
                "interview_date": {"type": "STRING"},
                "interview_time": {"type": "STRING"},
            },
        }

        # Call OpenAI with json_schema structured output
        raw_llm_output = await self.openai_client.generate_chat_completion(
            messages=extraction_prompt,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "booking_extraction",
                    "schema": booking_schema,
                    "strict": True,
                },
            },
        )

        # Parse LLM output into Pydantic model
        try:
            clean_output = raw_llm_output.strip()
            if clean_output.startswith("```"):
                clean_output = clean_output.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
            parsed_data = json.loads(clean_output)
            extracted = LLMExtractedBooking.model_validate(parsed_data)
        except Exception as err:
            logger.warning("Failed to parse LLM extracted booking JSON", extra={"error": str(err)})
            extracted = LLMExtractedBooking()

        # Merge extracted fields into Redis partial booking state
        merged_state = PartialBookingState(
            name=extracted.name or existing_state.name,
            email=extracted.email or existing_state.email,
            interview_date=extracted.interview_date or existing_state.interview_date,
            interview_time=extracted.interview_time or existing_state.interview_time,
        )

        # Save updated partial state to Redis
        await self.memory.save_booking_state(session_id, merged_state)

        # Check missing fields
        missing = merged_state.missing_fields()

        if missing:
            missing_str = ", ".join(missing)
            captured = []
            if merged_state._is_valid_value(merged_state.name):
                captured.append(f"name: {merged_state.name}")
            if merged_state._is_valid_value(merged_state.email):
                captured.append(f"email: {merged_state.email}")
            if merged_state._is_valid_value(merged_state.interview_date):
                captured.append(f"date: {merged_state.interview_date}")
            if merged_state._is_valid_value(merged_state.interview_time):
                captured.append(f"time: {merged_state.interview_time}")

            captured_str = ", ".join(captured) if captured else "nothing yet"
            prompt_msg = (
                f"Thank you! I'd be happy to help schedule your interview. "
                f"So far I have: {captured_str}.\n"
                f"Could you please also provide your {missing_str}?"
            )
            return BookingProcessResult(
                is_booking_intent=True,
                is_complete=False,
                response_text=prompt_msg,
                missing_fields=missing,
            )

        # All 4 fields present — Validate email and date/time format
        valid_email = self._validate_email(merged_state.email)
        valid_date = self._parse_date(merged_state.interview_date)
        valid_time = self._parse_time(merged_state.interview_time)

        if not valid_email or not valid_date or not valid_time:
            error_hints = []
            if not valid_email:
                error_hints.append("a valid email address (e.g. name@company.com)")
            if not valid_date:
                error_hints.append("a recognizable date (e.g. '2026-08-15', 'next Monday', 'Aug 20')")
            if not valid_time:
                error_hints.append("a specific time (e.g. '14:00', '2:30 PM', '10am')")

            retry_msg = (
                f"I couldn't quite parse some of your details. "
                f"Could you please re-enter {', '.join(error_hints)}?"
            )
            return BookingProcessResult(
                is_booking_intent=True,
                is_complete=False,
                response_text=retry_msg,
                missing_fields=missing,
            )

        # Create confirmed DB Booking
        sess_uuid = UUID(str(session_id)) if isinstance(session_id, str) else session_id
        booking = await self.repo.create_booking(
            session_id=sess_uuid,
            name=merged_state.name,  # type: ignore[arg-type]
            email=valid_email,
            interview_date=valid_date,
            interview_time=valid_time,
            status="confirmed",
        )

        # Clear Redis partial state
        await self.memory.clear_booking_state(session_id)

        confirmation_text = (
            f"✅ Your interview has been successfully booked! Here are your details:\n\n"
            f"• Name: {booking.name}\n"
            f"• Email: {booking.email}\n"
            f"• Date: {booking.interview_date.strftime('%Y-%m-%d')}\n"
            f"• Time: {booking.interview_time.strftime('%H:%M')}\n\n"
            f"You will receive a confirmation at your email address. "
            f"We look forward to speaking with you!"
        )

        return BookingProcessResult(
            is_booking_intent=True,
            is_complete=True,
            response_text=confirmation_text,
            booking=booking,
        )

    def _validate_email(self, email_str: str | None) -> str | None:
        if not email_str:
            return None
        try:
            valid = validate_email(email_str, check_deliverability=False)
            return valid.normalized
        except EmailNotValidError:
            return None

    def _parse_date(self, date_str: str | None) -> date | None:
        if not date_str:
            return None
        parsed = dateparser.parse(date_str)
        return parsed.date() if parsed else None

    def _parse_time(self, time_str: str | None) -> time | None:
        if not time_str:
            return None
        parsed = dateparser.parse(time_str)
        return parsed.time() if parsed else None

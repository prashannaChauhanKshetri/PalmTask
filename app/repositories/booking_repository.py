"""Repository pattern for Booking database persistence operations."""

from datetime import date, time
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.booking import Booking


class BookingRepository:
    """Async database repository for interview bookings."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_booking(
        self,
        session_id: UUID | None,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
        status: str = "pending",
    ) -> Booking:
        """Insert a new confirmed booking record."""
        booking = Booking(
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
            status=status,
        )
        self.session.add(booking)
        await self.session.commit()
        await self.session.refresh(booking)
        return booking

    async def get_booking_by_id(self, booking_id: UUID) -> Booking | None:
        """Fetch booking by UUID."""
        stmt = select(Booking).where(Booking.id == booking_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

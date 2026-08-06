"""ORM model for the 'bookings' table."""

import uuid
from datetime import date, datetime, time

from sqlalchemy import Date, ForeignKey, String, Text, Time, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if __import__("typing").TYPE_CHECKING:
    from app.models.chat_session import ChatSession


class Booking(Base):
    """An interview booking extracted from a chat session via structured LLM output."""

    __tablename__ = "bookings"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False)
    interview_date: Mapped[date] = mapped_column(Date, nullable=False)
    interview_time: Mapped[time] = mapped_column(Time, nullable=False)
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
        comment="'pending' | 'confirmed' | 'cancelled'",
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    # ── Relationships ───────────────────────────────────────
    session: Mapped["ChatSession"] = relationship(
        "ChatSession",
        back_populates="bookings",
    )

    def __repr__(self) -> str:
        return f"<Booking id={self.id} name={self.name!r} status={self.status!r}>"

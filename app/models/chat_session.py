"""ORM model for the 'chat_sessions' table."""

import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if __import__("typing").TYPE_CHECKING:
    from app.models.booking import Booking


class ChatSession(Base):
    """A conversation session — timestamps track activity for TTL/cleanup."""

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_active_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # ── Relationships ───────────────────────────────────────
    bookings: Mapped[list["Booking"]] = relationship(
        "Booking",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} last_active={self.last_active_at}>"

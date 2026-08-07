"""ORM model for the 'documents' table."""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if __import__("typing").TYPE_CHECKING:
    from app.models.chunk import Chunk


class Document(Base):
    """Tracks each uploaded document and its ingestion status."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(
        String, nullable=False, comment="'pdf' | 'txt'"
    )
    chunking_strategy: Mapped[str] = mapped_column(
        String, nullable=False, comment="'fixed' | 'recursive'"
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
        comment="'pending' | 'processing' | 'completed' | 'failed'",
    )
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # ── Relationships ───────────────────────────────────────
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.status!r}>"

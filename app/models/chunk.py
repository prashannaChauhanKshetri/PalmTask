"""ORM model for the 'chunks' table."""

import uuid

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if __import__("typing").TYPE_CHECKING:
    from app.models.document import Document


class Chunk(Base):
    """One chunk of an ingested document, linked to its Pinecone vector by vector_id."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_id: Mapped[str] = mapped_column(
        String, nullable=False, comment="Pinecone vector ID — join key for retrieval"
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Relationships ───────────────────────────────────────
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.document_id} idx={self.chunk_index}>"

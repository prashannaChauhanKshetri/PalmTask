"""Repository pattern for Document and Chunk database persistence operations."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document


class DocumentRepository:
    """Async database repository for documents and document chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_document(
        self,
        filename: str,
        file_type: str,
        chunking_strategy: str,
        doc_id: UUID | None = None,
    ) -> Document:
        """Insert a new pending document record."""
        doc = Document(
            filename=filename,
            file_type=file_type,
            chunking_strategy=chunking_strategy,
            status="pending",
            total_chunks=0,
        )
        if doc_id:
            doc.id = doc_id
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def get_document_by_id(self, doc_id: UUID) -> Document | None:
        """Fetch document record by UUID."""
        stmt = select(Document).where(Document.id == doc_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_documents(
        self, skip: int = 0, limit: int = 20
    ) -> tuple[list[Document], int]:
        """Fetch paginated document list and total count."""
        count_stmt = select(func.count()).select_from(Document)
        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar_one()

        list_stmt = (
            select(Document)
            .order_by(Document.uploaded_at.desc())
            .offset(skip)
            .limit(limit)
        )
        items_result = await self.session.execute(list_stmt)
        items = list(items_result.scalars().all())

        return items, total

    async def update_document_status(
        self, doc_id: UUID, status: str, total_chunks: int = 0
    ) -> None:
        """Update document status and total chunks count."""
        doc = await self.get_document_by_id(doc_id)
        if doc:
            doc.status = status
            doc.total_chunks = total_chunks
            await self.session.commit()

    async def create_chunks(self, chunks: list[Chunk]) -> None:
        """Batch insert chunk records."""
        self.session.add_all(chunks)
        await self.session.commit()

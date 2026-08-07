"""Document ingestion service orchestrating extraction, chunking, embedding, and vector upsert."""

import uuid
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.openai_client import OpenAIClient
from app.clients.pinecone_client import PineconeClient
from app.core.exceptions import UnsupportedFileTypeError
from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.services.chunking import ChunkerFactory
from app.services.extractors import ExtractorFactory

logger = get_logger(__name__)


class IngestionService:
    """Service handling document upload validation and background ingestion pipeline execution."""

    def __init__(
        self,
        repo: DocumentRepository,
        openai_client: OpenAIClient | None = None,
        pinecone_client: PineconeClient | None = None,
    ) -> None:
        self.repo = repo
        self.openai_client = openai_client or OpenAIClient()
        self.pinecone_client = pinecone_client or PineconeClient()

    async def init_upload(
        self,
        filename: str,
        content_type: str | None,
        chunking_strategy: Literal["fixed", "recursive", "hierarchical"],
    ) -> Document:
        """Validate document type and create initial DB record.

        Raises:
            UnsupportedFileTypeError: If file is not .pdf or .txt.
        """
        lower_name = filename.lower()
        if lower_name.endswith(".pdf"):
            file_type = "pdf"
        elif lower_name.endswith(".txt"):
            file_type = "txt"
        else:
            raise UnsupportedFileTypeError(content_type or filename)

        doc = await self.repo.create_document(
            filename=filename,
            file_type=file_type,
            chunking_strategy=chunking_strategy,
        )
        return doc

    async def process_document_pipeline(
        self,
        doc_id: uuid.UUID,
        file_bytes: bytes,
        filename: str,
        file_type: str,
        chunking_strategy: Literal["fixed", "recursive", "hierarchical"],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """Background worker execution pipeline for document extraction, chunking, embedding, and storage."""
        logger.info(
            "Starting background document processing",
            extra={"document_id": str(doc_id), "doc_filename": filename},
        )
        await self.repo.update_document_status(doc_id, "processing")

        try:
            # 1. Text Extraction
            extractor = ExtractorFactory.get_extractor(filename, file_type)
            extracted_text = extractor.extract(file_bytes, filename)

            # 2. Text Chunking
            chunker = ChunkerFactory.get_chunker(chunking_strategy)
            chunk_results = chunker.chunk(
                extracted_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            if not chunk_results:
                logger.warning("No chunks produced from extracted text", extra={"document_id": str(doc_id)})
                await self.repo.update_document_status(doc_id, "completed", total_chunks=0)
                return

            # 3. Embedding Generation
            texts_to_embed = [c.text for c in chunk_results]
            embeddings = await self.openai_client.get_embeddings(texts_to_embed)

            # 4. Pinecone Vector Upsert
            vector_tuples: list[tuple[str, list[float], dict[str, str | int]]] = []
            db_chunks: list[Chunk] = []

            for chunk_res, emb in zip(chunk_results, embeddings, strict=True):
                vector_id = f"{doc_id}_{chunk_res.index}"
                meta = {
                    "document_id": str(doc_id),
                    "chunk_index": chunk_res.index,
                    "text_preview": chunk_res.text[:200],
                }
                vector_tuples.append((vector_id, emb, meta))

                db_chunks.append(
                    Chunk(
                        document_id=doc_id,
                        chunk_index=chunk_res.index,
                        vector_id=vector_id,
                        token_count=chunk_res.token_count,
                    )
                )

            self.pinecone_client.upsert_vectors(vector_tuples)

            # 5. Postgres Chunks Persistence
            await self.repo.create_chunks(db_chunks)

            # 6. Update Status to Completed
            await self.repo.update_document_status(
                doc_id, "completed", total_chunks=len(db_chunks)
            )

            logger.info(
                "Document processing pipeline completed successfully",
                extra={"document_id": str(doc_id), "total_chunks": len(db_chunks)},
            )

        except Exception as err:
            logger.error(
                "Document processing pipeline failed",
                extra={"document_id": str(doc_id), "error": str(err)},
                exc_info=True,
            )
            await self.repo.update_document_status(doc_id, "failed")

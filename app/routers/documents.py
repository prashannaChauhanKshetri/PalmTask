"""FastAPI router for Document Ingestion API endpoints."""

from typing import Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import DocumentNotFoundError
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for text extraction, chunking, embedding, and vector index ingestion",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunking_strategy: Literal["fixed", "recursive", "hierarchical"] = Form("fixed"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """Accept PDF/TXT multipart upload and process chunking/embedding as a background task."""
    filename = file.filename or "uploaded_file"
    file_bytes = await file.read()

    repo = DocumentRepository(db)
    service = IngestionService(repo)

    # 1. Init upload and validate extension
    doc = await service.init_upload(
        filename=filename,
        content_type=file.content_type,
        chunking_strategy=chunking_strategy,
    )

    # 2. Enqueue background pipeline task
    background_tasks.add_task(
        service.process_document_pipeline,
        doc_id=doc.id,
        file_bytes=file_bytes,
        filename=filename,
        file_type=doc.file_type,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        status="processing",
        total_chunks=0,
        message="Document uploaded successfully. Ingestion pipeline is running in background.",
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document processing status and metadata by ID",
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Fetch status and metadata for a specific document."""
    repo = DocumentRepository(db)
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)
    return DocumentResponse.model_validate(doc)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List paginated ingested documents",
)
async def list_documents(
    skip: int = Query(0, ge=0, description="Offset pagination index"),
    limit: int = Query(20, ge=1, le=100, description="Page size limit"),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """Retrieve a paginated list of ingested document records."""
    repo = DocumentRepository(db)
    items, total = await repo.list_documents(skip=skip, limit=limit)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(doc) for doc in items],
        total=total,
        skip=skip,
        limit=limit,
    )

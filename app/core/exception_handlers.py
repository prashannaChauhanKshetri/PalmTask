"""FastAPI exception handlers — translate domain exceptions to HTTP responses."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppError,
    BookingValidationError,
    DocumentNotFoundError,
    EmptyDocumentError,
    SessionNotFoundError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to the FastAPI application."""

    @app.exception_handler(DocumentNotFoundError)
    async def _document_not_found(
        request: Request, exc: DocumentNotFoundError
    ) -> JSONResponse:
        logger.warning("Document not found", extra={"document_id": str(exc.document_id)})
        return JSONResponse(status_code=404, content={"detail": exc.detail})

    @app.exception_handler(SessionNotFoundError)
    async def _session_not_found(
        request: Request, exc: SessionNotFoundError
    ) -> JSONResponse:
        logger.warning("Session not found", extra={"session_id": str(exc.session_id)})
        return JSONResponse(status_code=404, content={"detail": exc.detail})

    @app.exception_handler(UnsupportedFileTypeError)
    async def _unsupported_file_type(
        request: Request, exc: UnsupportedFileTypeError
    ) -> JSONResponse:
        logger.warning("Unsupported file type", extra={"content_type": exc.content_type})
        return JSONResponse(status_code=415, content={"detail": exc.detail})

    @app.exception_handler(EmptyDocumentError)
    async def _empty_document(
        request: Request, exc: EmptyDocumentError
    ) -> JSONResponse:
        logger.warning("Empty/scanned document", extra={"filename": exc.filename})
        return JSONResponse(status_code=422, content={"detail": exc.detail})

    @app.exception_handler(BookingValidationError)
    async def _booking_validation(
        request: Request, exc: BookingValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.detail})

    @app.exception_handler(AppError)
    async def _generic_app_error(
        request: Request, exc: AppError
    ) -> JSONResponse:
        logger.error("Unhandled application error", extra={"detail": exc.detail})
        return JSONResponse(status_code=500, content={"detail": exc.detail})

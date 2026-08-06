"""Typed domain exceptions mapped to HTTP responses via FastAPI exception handlers."""

from uuid import UUID


class AppError(Exception):
    """Base class for all application-level errors."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


# ── Document errors ─────────────────────────────────────


class DocumentNotFoundError(AppError):
    """Raised when a document ID does not exist in the database."""

    def __init__(self, document_id: UUID) -> None:
        super().__init__(f"Document {document_id} not found.")
        self.document_id = document_id


class UnsupportedFileTypeError(AppError):
    """Raised when an uploaded file has an unsupported MIME/extension."""

    def __init__(self, content_type: str) -> None:
        super().__init__(f"Unsupported file type: {content_type}. Only .pdf and .txt are accepted.")
        self.content_type = content_type


class EmptyDocumentError(AppError):
    """Raised when a PDF or TXT file yields no extractable text."""

    def __init__(self, filename: str) -> None:
        super().__init__(
            f"No extractable text found in '{filename}'. "
            "Scanned or image-only PDFs are not supported."
        )
        self.filename = filename


# ── Chat / session errors ──────────────────────────────


class SessionNotFoundError(AppError):
    """Raised when a chat session ID does not exist."""

    def __init__(self, session_id: UUID) -> None:
        super().__init__(f"Chat session {session_id} not found.")
        self.session_id = session_id


# ── Booking errors ──────────────────────────────────────


class BookingValidationError(AppError):
    """Raised when booking field extraction/validation fails."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)

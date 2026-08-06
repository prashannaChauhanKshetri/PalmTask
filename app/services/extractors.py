"""Text extraction module matching the TextExtractor protocol.

Supports PDF (via PyMuPDF/fitz) and TXT (via UTF-8 / charset-normalizer fallback).
Detects empty or scanned image-only PDFs and raises EmptyDocumentError.
"""

from typing import Protocol, runtime_checkable

import fitz  # PyMuPDF
from charset_normalizer import from_bytes

from app.core.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class TextExtractor(Protocol):
    """Protocol defining the interface for document text extractors."""

    def extract(self, file_bytes: bytes, filename: str) -> str:
        """Extract plain text from document bytes.

        Raises:
            EmptyDocumentError: If document contains no extractable text.
            UnsupportedFileTypeError: If format cannot be parsed.
        """
        ...


class PDFTextExtractor:
    """PDF text extractor using PyMuPDF (fitz).

    Rejects scanned or image-only PDFs that yield no text (no OCR).
    """

    def extract(self, file_bytes: bytes, filename: str) -> str:
        logger.info("Extracting text from PDF", extra={"doc_filename": filename, "size_bytes": len(file_bytes)})
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as err:
            logger.warning("Failed to parse PDF file", extra={"doc_filename": filename, "error": str(err)})
            raise UnsupportedFileTypeError("application/pdf (corrupted or invalid PDF)") from err

        pages_text: list[str] = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text:
                pages_text.append(text)

        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            logger.warning("PDF yielded no extractable text (likely scanned)", extra={"doc_filename": filename})
            raise EmptyDocumentError(filename)

        logger.info(
            "PDF extraction successful",
            extra={"doc_filename": filename, "pages": len(doc), "extracted_chars": len(full_text)},
        )
        return full_text


class TXTTextExtractor:
    """Plain text extractor with UTF-8 primary parsing and charset-normalizer fallback."""

    def extract(self, file_bytes: bytes, filename: str) -> str:
        logger.info("Extracting text from TXT file", extra={"doc_filename": filename, "size_bytes": len(file_bytes)})
        
        # Try primary UTF-8 decoding first
        try:
            raw_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.info("UTF-8 decoding failed; attempting charset detection", extra={"doc_filename": filename})
            results = from_bytes(file_bytes)
            best_match = results.best()
            if best_match is None or best_match.output() is None:
                raise EmptyDocumentError(filename)
            raw_text = str(best_match.output())

        full_text = raw_text.strip()
        if not full_text:
            logger.warning("TXT file is empty", extra={"doc_filename": filename})
            raise EmptyDocumentError(filename)

        logger.info(
            "TXT extraction successful",
            extra={"doc_filename": filename, "extracted_chars": len(full_text)},
        )
        return full_text


class ExtractorFactory:
    """Factory for obtaining the appropriate TextExtractor based on file extension / MIME."""

    @staticmethod
    def get_extractor(filename: str, content_type: str | None = None) -> TextExtractor:
        """Resolve extractor by extension or MIME type.

        Raises:
            UnsupportedFileTypeError: For extensions other than .pdf or .txt.
        """
        lower_name = filename.lower()
        if lower_name.endswith(".pdf") or content_type == "application/pdf":
            return PDFTextExtractor()
        elif lower_name.endswith(".txt") or content_type in ("text/plain", "application/octet-stream"):
            return TXTTextExtractor()
        else:
            raise UnsupportedFileTypeError(content_type or filename)

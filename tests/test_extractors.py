"""Unit tests for text extraction module (PDF and TXT extractors)."""

import fitz
import pytest

from app.core.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.services.extractors import ExtractorFactory, PDFTextExtractor, TXTTextExtractor


def test_txt_extractor_utf8() -> None:
    extractor = TXTTextExtractor()
    sample_bytes = "Hello World! This is a test document.".encode("utf-8")
    result = extractor.extract(sample_bytes, "sample.txt")
    assert result == "Hello World! This is a test document."


def test_txt_extractor_empty_raises_error() -> None:
    extractor = TXTTextExtractor()
    empty_bytes = b"   \n  "
    with pytest.raises(EmptyDocumentError):
        extractor.extract(empty_bytes, "empty.txt")


def test_pdf_extractor_valid() -> None:
    extractor = PDFTextExtractor()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Sample PDF Text Content")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extractor.extract(pdf_bytes, "test.pdf")
    assert "Sample PDF Text Content" in result


def test_pdf_extractor_scanned_raises_error() -> None:
    extractor = PDFTextExtractor()
    doc = fitz.open()
    doc.new_page()  # empty page with no text
    pdf_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(EmptyDocumentError):
        extractor.extract(pdf_bytes, "scanned.pdf")


def test_extractor_factory_resolution() -> None:
    pdf_ext = ExtractorFactory.get_extractor("doc.pdf", "application/pdf")
    assert isinstance(pdf_ext, PDFTextExtractor)

    txt_ext = ExtractorFactory.get_extractor("notes.txt", "text/plain")
    assert isinstance(txt_ext, TXTTextExtractor)

    with pytest.raises(UnsupportedFileTypeError):
        ExtractorFactory.get_extractor("image.png", "image/png")

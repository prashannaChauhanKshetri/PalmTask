"""Unit tests for chunking strategies (Fixed, Recursive, and Hierarchical)."""

import pytest

from app.services.chunking import (
    ChunkerFactory,
    ChunkingError,
    FixedWindowChunker,
    HierarchicalChunker,
    RecursiveChunker,
)


@pytest.fixture
def sample_document() -> str:
    return """# Section 1: Overview
The Palm RAG Backend provides real-time document search and conversational interface.
It features vector search backed by Pinecone and chat memory powered by Redis.

# Section 2: Architecture
The architecture follows clean modular principles:
1. Routers for HTTP handling.
2. Services for core business logic.
3. Repositories and Clients for data access.

# Section 3: Ingestion
Documents are ingested via multipart upload (.pdf or .txt).
Text is extracted using format-specific extractors and processed into chunk embeddings.
"""


@pytest.fixture
def hierarchical_document() -> str:
    """Document with nested heading structure for hierarchical chunking tests."""
    return """# Introduction
PalmMind AI is a next-generation conversational retrieval platform.
It uses advanced vector search and semantic understanding.

## Core Capabilities
### Document Processing
PalmMind supports PDF and TXT file uploads with automatic text extraction.
Multiple chunking strategies are available for optimal retrieval.

### Vector Search
Powered by Pinecone serverless, PalmMind delivers sub-millisecond similarity search.
Score threshold filtering removes noise from results.

## Architecture
### API Layer
FastAPI provides async REST endpoints with automatic OpenAPI documentation.
All routes follow RESTful conventions with proper status codes.

### Service Layer
Business logic is isolated in service classes following clean architecture principles.
Each service has a single responsibility and clear interface boundaries.

### Data Layer
PostgreSQL stores document metadata and booking records.
Redis provides windowed chat history for multi-turn conversations.
Pinecone indexes vector embeddings for similarity search.
"""


def test_fixed_chunker_basic(sample_document: str) -> None:
    chunker = FixedWindowChunker()
    chunks = chunker.chunk(sample_document, chunk_size=40, chunk_overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.token_count <= 40
        assert chunk.text != ""

    # Check contiguous indexing
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_recursive_chunker_structure_preservation(sample_document: str) -> None:
    chunker = RecursiveChunker()
    chunks = chunker.chunk(sample_document, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.token_count <= 60  # Allows small combined token buffer

    # Verify paragraph header preserved intact in early chunks
    assert "Section 1" in chunks[0].text


def test_strategies_produce_different_boundaries(sample_document: str) -> None:
    fixed_chunker = FixedWindowChunker()
    recursive_chunker = RecursiveChunker()

    fixed_chunks = fixed_chunker.chunk(sample_document, chunk_size=30, chunk_overlap=5)
    recursive_chunks = recursive_chunker.chunk(sample_document, chunk_size=30, chunk_overlap=5)

    fixed_texts = [c.text for c in fixed_chunks]
    recursive_texts = [c.text for c in recursive_chunks]

    # Verify boundaries differ
    assert fixed_texts != recursive_texts


def test_invalid_parameters() -> None:
    chunker = FixedWindowChunker()
    with pytest.raises(ChunkingError):
        chunker.chunk("Test", chunk_size=0, chunk_overlap=5)

    with pytest.raises(ChunkingError):
        chunker.chunk("Test", chunk_size=50, chunk_overlap=50)


def test_factory_dispatch() -> None:
    fixed = ChunkerFactory.get_chunker("fixed")
    assert isinstance(fixed, FixedWindowChunker)

    recursive = ChunkerFactory.get_chunker("recursive")
    assert isinstance(recursive, RecursiveChunker)

    hierarchical = ChunkerFactory.get_chunker("hierarchical")
    assert isinstance(hierarchical, HierarchicalChunker)

    with pytest.raises(ChunkingError):
        ChunkerFactory.get_chunker("invalid")  # type: ignore[arg-type]


# ── Hierarchical Chunker Tests ──────────────────────────────


def test_hierarchical_detects_markdown_headers(hierarchical_document: str) -> None:
    """Hierarchical chunker should detect markdown headers and prepend section breadcrumbs."""
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(hierarchical_document, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 0

    # At least one chunk should contain a section breadcrumb prefix
    breadcrumb_chunks = [c for c in chunks if "[Section:" in c.text]
    assert len(breadcrumb_chunks) > 0, "Expected chunks with section breadcrumb prefixes"

    # Verify hierarchical nesting is preserved
    nested_chunks = [c for c in chunks if ">" in c.text and "[Section:" in c.text]
    assert len(nested_chunks) > 0, "Expected nested breadcrumb paths like 'A > B'"


def test_hierarchical_chunks_contain_body_content(hierarchical_document: str) -> None:
    """Every hierarchical chunk should contain meaningful body text, not just headers."""
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(hierarchical_document, chunk_size=150, chunk_overlap=10)

    for chunk in chunks:
        # Strip away the breadcrumb prefix and check body text exists
        body = chunk.text
        if "[Section:" in body:
            body = body[body.index("\n") + 1 :] if "\n" in body else body
        assert len(body.strip()) > 0, f"Chunk {chunk.index} has no body content"


def test_hierarchical_respects_chunk_size(hierarchical_document: str) -> None:
    """Each hierarchical chunk should respect the token size limit."""
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(hierarchical_document, chunk_size=60, chunk_overlap=10)

    for chunk in chunks:
        assert chunk.token_count <= 70, (
            f"Chunk {chunk.index} exceeds size limit: {chunk.token_count} tokens"
        )


def test_hierarchical_contiguous_indexing(hierarchical_document: str) -> None:
    """Chunk indices should be contiguous 0..N-1."""
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(hierarchical_document, chunk_size=100, chunk_overlap=10)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_hierarchical_fallback_on_unstructured() -> None:
    """Unstructured text (no headers) should fall back to recursive chunking."""
    plain_text = (
        "This is a plain document with no headers. "
        "It has multiple sentences but no markdown structure. " * 10
    )
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(plain_text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    # None should have breadcrumb prefixes since there are no headers
    assert all("[Section:" not in c.text for c in chunks)


def test_hierarchical_invalid_params() -> None:
    """Should raise ChunkingError on invalid chunk_size or chunk_overlap."""
    chunker = HierarchicalChunker()
    with pytest.raises(ChunkingError):
        chunker.chunk("Test", chunk_size=0, chunk_overlap=5)

    with pytest.raises(ChunkingError):
        chunker.chunk("Test", chunk_size=50, chunk_overlap=50)


def test_hierarchical_empty_text() -> None:
    """Empty text should produce zero chunks."""
    chunker = HierarchicalChunker()
    assert chunker.chunk("", chunk_size=100, chunk_overlap=10) == []
    assert chunker.chunk("   ", chunk_size=100, chunk_overlap=10) == []


def test_hierarchical_differs_from_recursive(hierarchical_document: str) -> None:
    """Hierarchical chunks should differ from recursive due to section context prefixes."""
    hierarchical = HierarchicalChunker()
    recursive = RecursiveChunker()

    h_chunks = hierarchical.chunk(hierarchical_document, chunk_size=80, chunk_overlap=10)
    r_chunks = recursive.chunk(hierarchical_document, chunk_size=80, chunk_overlap=10)

    h_texts = [c.text for c in h_chunks]
    r_texts = [c.text for c in r_chunks]

    # They should produce different chunk texts because hierarchical adds breadcrumbs
    assert h_texts != r_texts

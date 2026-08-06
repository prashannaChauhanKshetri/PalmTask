"""Unit tests for chunking strategies (Fixed and Recursive)."""

import pytest

from app.services.chunking import (
    ChunkerFactory,
    ChunkingError,
    FixedWindowChunker,
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

    with pytest.raises(ChunkingError):
        ChunkerFactory.get_chunker("invalid")  # type: ignore[arg-type]

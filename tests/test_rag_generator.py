"""Unit tests for semantic RAG generator with score filtering, contextual window, and re-ranking."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.clients.pinecone_client import VectorSearchResult
from app.services.memory_service import ChatTurn
from app.services.rag_generator import RAGGenerator, RAGResponse


def _make_result(doc_id: str, chunk_idx: int, score: float, text: str) -> VectorSearchResult:
    """Helper to create a VectorSearchResult for tests."""
    return VectorSearchResult(
        vector_id=f"{doc_id}_chunk{chunk_idx}",
        score=score,
        document_id=doc_id,
        chunk_index=chunk_idx,
        text_preview=text,
        metadata={"document_id": doc_id, "chunk_index": chunk_idx},
    )


@pytest.mark.asyncio
async def test_rag_generator_full_pipeline() -> None:
    """End-to-end pipeline test with score filtering, expansion, and re-ranking."""
    mock_openai = AsyncMock()
    mock_openai.get_embeddings = AsyncMock(return_value=[[0.1] * 1536])

    mock_gemini = AsyncMock()
    mock_gemini.generate_chat_completion = AsyncMock(
        return_value="PalmMind AI provides automated data search solutions."
    )

    mock_pinecone = MagicMock()
    mock_pinecone.query_similarity = MagicMock(
        return_value=[
            _make_result("doc1", 0, 0.92, "PalmMind AI provides automated data search solutions."),
            _make_result("doc1", 1, 0.85, "It supports PDF and TXT document uploads."),
            _make_result("doc2", 0, 0.40, "Unrelated low-score content."),
        ]
    )

    rag = RAGGenerator(openai_client=mock_openai, gemini_client=mock_gemini, pinecone_client=mock_pinecone)

    history = [ChatTurn(role="user", content="What is PalmMind AI?")]
    user_message = "Can you elaborate on its features?"

    response = await rag.generate_response(
        user_message=user_message,
        history=history,
        score_threshold=0.65,
        enable_contextual_window=False,  # Disable for cleaner test
    )

    assert response.answer == "PalmMind AI provides automated data search solutions."
    assert len(response.sources) == 2  # doc2 score=0.40 should be filtered out
    assert response.filtered_count == 1
    assert all(s.score >= 0.65 for s in response.sources)

    # Verify system prompt contained strict guardrail instructions
    call_args = mock_gemini.generate_chat_completion.call_args[1]
    messages = call_args["messages"]
    system_msg = messages[0]["content"]

    assert "PROMPT INJECTION DEFENSE" in system_msg
    assert "STRICT GROUNDING" in system_msg
    assert "ABSENCE OF INFORMATION" in system_msg
    assert "CONTEXTUAL COHERENCE" in system_msg


@pytest.mark.asyncio
async def test_score_threshold_filtering() -> None:
    """Score filtering should discard results below the threshold."""
    rag = RAGGenerator.__new__(RAGGenerator)

    results = [
        _make_result("doc1", 0, 0.95, "High score"),
        _make_result("doc1", 1, 0.80, "Medium score"),
        _make_result("doc2", 0, 0.55, "Low score"),
        _make_result("doc3", 0, 0.30, "Very low score"),
    ]

    filtered, discarded = rag._filter_by_score(results, threshold=0.70)

    assert len(filtered) == 2
    assert discarded == 2
    assert all(r.score >= 0.70 for r in filtered)


@pytest.mark.asyncio
async def test_score_threshold_keeps_all_above() -> None:
    """When all results are above threshold, nothing is discarded."""
    rag = RAGGenerator.__new__(RAGGenerator)

    results = [
        _make_result("doc1", 0, 0.95, "Top result"),
        _make_result("doc1", 1, 0.88, "Second result"),
    ]

    filtered, discarded = rag._filter_by_score(results, threshold=0.50)

    assert len(filtered) == 2
    assert discarded == 0


def test_rerank_groups_by_document() -> None:
    """Re-ranking should group chunks by document and order by best score."""
    results = [
        _make_result("doc2", 0, 0.70, "Doc2 first chunk"),
        _make_result("doc1", 2, 0.95, "Doc1 best chunk"),
        _make_result("doc1", 0, 0.80, "Doc1 earlier chunk"),
        _make_result("doc2", 1, 0.65, "Doc2 second chunk"),
    ]

    ranked = RAGGenerator._rerank_results(results)

    # Doc1 should come first (best score 0.95 > doc2 best 0.70)
    assert ranked[0].document_id == "doc1"
    assert ranked[1].document_id == "doc1"
    # Within doc1, chunks should be ordered by index
    assert ranked[0].chunk_index == 0
    assert ranked[1].chunk_index == 2
    # Then doc2 chunks
    assert ranked[2].document_id == "doc2"
    assert ranked[3].document_id == "doc2"


def test_rerank_empty_results() -> None:
    """Re-ranking empty list should return empty."""
    assert RAGGenerator._rerank_results([]) == []


@pytest.mark.asyncio
async def test_empty_embeddings_returns_error() -> None:
    """When embedding fails, should return error response."""
    mock_openai = AsyncMock()
    mock_openai.get_embeddings = AsyncMock(return_value=[])

    mock_pinecone = MagicMock()
    rag = RAGGenerator(openai_client=mock_openai, pinecone_client=mock_pinecone)

    response = await rag.generate_response(
        user_message="test", history=[]
    )

    assert "unable to process" in response.answer.lower()
    assert response.sources == []


@pytest.mark.asyncio
async def test_contextual_window_expansion() -> None:
    """Contextual window should fetch neighboring chunks via metadata filter."""
    mock_openai = AsyncMock()
    mock_openai.get_embeddings = AsyncMock(return_value=[[0.1] * 1536])

    mock_gemini = AsyncMock()
    mock_gemini.generate_chat_completion = AsyncMock(return_value="Answer with context.")

    # Initial query returns chunk 2 of doc1
    initial_result = _make_result("doc1", 2, 0.90, "Core chunk content.")

    # Neighbor queries return chunk 1 and chunk 3
    neighbor_1 = _make_result("doc1", 1, 0.0, "Previous chunk context.")
    neighbor_3 = _make_result("doc1", 3, 0.0, "Next chunk context.")

    def mock_query_similarity(query_embedding, top_k=5, filter_dict=None):
        if filter_dict is None:
            return [initial_result]
        chunk_idx = filter_dict.get("chunk_index", {}).get("$eq")
        if chunk_idx == 1:
            return [neighbor_1]
        elif chunk_idx == 3:
            return [neighbor_3]
        return []

    mock_pinecone = MagicMock()
    mock_pinecone.query_similarity = MagicMock(side_effect=mock_query_similarity)

    rag = RAGGenerator(openai_client=mock_openai, gemini_client=mock_gemini, pinecone_client=mock_pinecone)

    response = await rag.generate_response(
        user_message="Tell me more",
        history=[],
        score_threshold=0.0,  # Accept all scores
        enable_contextual_window=True,
    )

    # Should have 3 sources: chunk 1, 2, 3 (sorted by chunk_index)
    assert len(response.sources) == 3
    chunk_indices = [s.chunk_index for s in response.sources]
    assert chunk_indices == [1, 2, 3]
    assert response.expanded_count == 2


@pytest.mark.asyncio
async def test_response_includes_retrieval_stats() -> None:
    """RAGResponse should carry semantic search pipeline statistics."""
    mock_openai = AsyncMock()
    mock_openai.get_embeddings = AsyncMock(return_value=[[0.1] * 1536])

    mock_gemini = AsyncMock()
    mock_gemini.generate_chat_completion = AsyncMock(return_value="Test answer.")

    mock_pinecone = MagicMock()
    mock_pinecone.query_similarity = MagicMock(
        return_value=[
            _make_result("doc1", 0, 0.90, "Relevant."),
            _make_result("doc2", 0, 0.30, "Noise."),
        ]
    )

    rag = RAGGenerator(openai_client=mock_openai, gemini_client=mock_gemini, pinecone_client=mock_pinecone)

    response = await rag.generate_response(
        user_message="test",
        history=[],
        score_threshold=0.65,
        enable_contextual_window=False,
    )

    assert isinstance(response, RAGResponse)
    assert response.filtered_count == 1
    assert response.expanded_count == 0

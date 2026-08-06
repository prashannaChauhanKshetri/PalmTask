"""Unit tests for hand-crafted custom RAGGenerator with prompt assembly and guardrails."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.clients.pinecone_client import VectorSearchResult
from app.services.memory_service import ChatTurn
from app.services.rag_generator import RAGGenerator


@pytest.mark.asyncio
async def test_rag_generator_pipeline() -> None:
    mock_openai = AsyncMock()
    mock_openai.get_embeddings = AsyncMock(return_value=[[0.1] * 1536])
    mock_openai.generate_chat_completion = AsyncMock(
        return_value="PalmMind AI provides automated data search solutions."
    )

    mock_pinecone = MagicMock()
    mock_pinecone.query_similarity = MagicMock(
        return_value=[
            VectorSearchResult(
                vector_id="doc1_chunk0",
                score=0.92,
                document_id="doc1",
                chunk_index=0,
                text_preview="PalmMind AI provides automated data search solutions.",
                metadata={"document_id": "doc1", "chunk_index": 0},
            )
        ]
    )

    rag = RAGGenerator(openai_client=mock_openai, pinecone_client=mock_pinecone)

    history = [ChatTurn(role="user", content="What is PalmMind AI?")]
    user_message = "Can you elaborate on its features?"

    response = await rag.generate_response(user_message=user_message, history=history)

    assert response.answer == "PalmMind AI provides automated data search solutions."
    assert len(response.sources) == 1
    assert response.sources[0].document_id == "doc1"
    assert response.sources[0].score == 0.92

    # Verify system prompt contained strict guardrail instructions
    call_args = mock_openai.generate_chat_completion.call_args[1]
    messages = call_args["messages"]
    system_msg = messages[0]["content"]

    assert "PROMPT INJECTION DEFENSE" in system_msg
    assert "STRICT GROUNDING" in system_msg
    assert "ABMSENCE OF INFORMATION" in system_msg or "ABSENCE OF INFORMATION" in system_msg

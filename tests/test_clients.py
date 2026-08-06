"""Unit tests for OpenAI and Pinecone client wrappers with mocked external APIs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.clients.openai_client import OpenAIClient
from app.clients.pinecone_client import PineconeClient


@pytest.mark.asyncio
async def test_openai_get_embeddings() -> None:
    client = OpenAIClient(api_key="test-key")

    mock_response = MagicMock()
    mock_data_1 = MagicMock()
    mock_data_1.embedding = [0.1] * 1536
    mock_data_2 = MagicMock()
    mock_data_2.embedding = [0.2] * 1536
    mock_response.data = [mock_data_1, mock_data_2]

    with patch.object(client, "_client") as mock_async_client:
        mock_async_client.embeddings.create = AsyncMock(return_value=mock_response)

        embeddings = await client.get_embeddings(["chunk 1", "chunk 2"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert embeddings[0][0] == 0.1
        assert embeddings[1][0] == 0.2


def test_pinecone_upsert_and_query() -> None:
    client = PineconeClient(api_key="test-key", index_name="palm-rag")

    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "doc123_chunk0",
                "score": 0.95,
                "metadata": {
                    "document_id": "doc123",
                    "chunk_index": 0,
                    "text_preview": "Sample text preview",
                },
            }
        ]
    }

    with patch.object(client, "_index", mock_index):
        # Test Upsert
        vectors = [
            ("doc123_chunk0", [0.1] * 1536, {"document_id": "doc123", "chunk_index": 0})
        ]
        client.upsert_vectors(vectors)
        mock_index.upsert.assert_called_once()

        # Test Query
        results = client.query_similarity(query_embedding=[0.1] * 1536, top_k=1)
        assert len(results) == 1
        assert results[0].vector_id == "doc123_chunk0"
        assert results[0].score == 0.95
        assert results[0].document_id == "doc123"
        assert results[0].chunk_index == 0

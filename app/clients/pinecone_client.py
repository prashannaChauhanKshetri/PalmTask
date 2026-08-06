"""Pinecone Vector Database client wrapper for chunk index upserts and similarity queries."""

from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class VectorSearchResult:
    """Dataclass representing a Pinecone similarity search match."""

    vector_id: str
    score: float
    document_id: str
    chunk_index: int
    text_preview: str
    metadata: dict[str, Any]


class PineconeClient:
    """Client wrapper for Pinecone Serverless vector database."""

    def __init__(self, api_key: str | None = None, index_name: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.pinecone_api_key
        self.index_name = index_name or settings.pinecone_index_name
        self._pc: Pinecone | None = None
        self._index: Any | None = None

    @property
    def index(self) -> Any:
        """Lazy-initialized Pinecone index connection."""
        if self._index is None:
            self._pc = Pinecone(api_key=self.api_key)
            self._index = self._pc.Index(self.index_name)
        return self._index

    def upsert_vectors(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
        batch_size: int = 100,
    ) -> None:
        """Batch upsert vectors and metadata into Pinecone index.

        Args:
            vectors: List of tuples (vector_id, embedding_vector, metadata_dict).
            batch_size: Max vectors per upsert batch call.
        """
        if not vectors:
            return

        logger.info(
            "Upserting vectors to Pinecone",
            extra={"count": len(vectors), "index": self.index_name},
        )

        formatted_vectors = [
            {"id": vid, "values": emb, "metadata": meta}
            for vid, emb, meta in vectors
        ]

        for i in range(0, len(formatted_vectors), batch_size):
            batch = formatted_vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)

        logger.info(
            "Pinecone upsert completed",
            extra={"count": len(vectors), "index": self.index_name},
        )

    def query_similarity(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Query Pinecone index for top_k nearest neighbor vectors by cosine similarity.

        Args:
            query_embedding: 1536-dimensional float vector.
            top_k: Number of top matching chunks to retrieve.
            filter_dict: Optional Pinecone metadata filter.

        Returns:
            List of VectorSearchResult items sorted by score descending.
        """
        logger.info(
            "Querying Pinecone similarity",
            extra={"top_k": top_k, "index": self.index_name},
        )

        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )

        results: list[VectorSearchResult] = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append(
                VectorSearchResult(
                    vector_id=match["id"],
                    score=float(match["score"]),
                    document_id=str(meta.get("document_id", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text_preview=str(meta.get("text_preview", "")),
                    metadata=meta,
                )
            )

        logger.info(
            "Pinecone query completed",
            extra={"matches_found": len(results)},
        )
        return results

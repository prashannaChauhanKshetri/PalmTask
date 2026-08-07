"""Pinecone Vector Database client wrapper for chunk index upserts and similarity queries (with local fallback)."""

import math
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
    """Client wrapper for Pinecone Serverless vector database with in-memory fallback."""

    NAMESPACE = "palm_namespace"

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
            self._pc = Pinecone(api_key=self.api_key or "pcsk_dummy")
            self._index = self._pc.Index(self.index_name)
        return self._index

    def upsert_vectors(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
        batch_size: int = 100,
    ) -> None:
        """Batch upsert vectors and metadata into Pinecone index."""
        if not vectors:
            return

        formatted_vectors = [
            {"id": vid, "values": emb, "metadata": meta}
            for vid, emb, meta in vectors
        ]
        for i in range(0, len(formatted_vectors), batch_size):
            batch = formatted_vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.NAMESPACE)
        logger.info("Pinecone upsert completed", extra={"count": len(vectors), "namespace": self.NAMESPACE})

    def query_similarity(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Query Pinecone index for top_k nearest neighbor vectors by cosine similarity."""
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
            namespace=self.NAMESPACE,
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

        return results

    def fetch_by_id(self, vector_id: str) -> VectorSearchResult | None:
        """Fetch a specific vector exactly by its ID."""
        response = self.index.fetch(ids=[vector_id], namespace=self.NAMESPACE)
        vectors = response.get("vectors", {})
        if vector_id not in vectors:
            return None
            
        match = vectors[vector_id]
        meta = match.get("metadata", {})
        return VectorSearchResult(
            vector_id=match["id"],
            score=1.0,
            document_id=str(meta.get("document_id", "")),
            chunk_index=int(meta.get("chunk_index", 0)),
            text_preview=str(meta.get("text_preview", "")),
            metadata=meta,
        )

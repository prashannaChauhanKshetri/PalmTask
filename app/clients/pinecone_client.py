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


_local_vector_store: list[tuple[str, list[float], dict[str, Any]]] = []


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1)) or 1.0
    norm2 = math.sqrt(sum(b * b for b in v2)) or 1.0
    return dot / (norm1 * norm2)


class PineconeClient:
    """Client wrapper for Pinecone Serverless vector database with in-memory fallback."""

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
        """Batch upsert vectors and metadata into Pinecone index with local fallback."""
        if not vectors:
            return

        # Keep a copy in local vector store for offline/fallback retrieval
        _local_vector_store.extend(vectors)

        try:
            formatted_vectors = [
                {"id": vid, "values": emb, "metadata": meta}
                for vid, emb, meta in vectors
            ]
            for i in range(0, len(formatted_vectors), batch_size):
                batch = formatted_vectors[i : i + batch_size]
                self.index.upsert(vectors=batch)
            logger.info("Pinecone upsert completed", extra={"count": len(vectors)})
        except Exception as err:
            logger.warning(
                "Pinecone upsert fallback to local vector store",
                extra={"error": str(err), "count": len(vectors)},
            )

    def query_similarity(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Query Pinecone index for top_k nearest neighbor vectors by cosine similarity."""
        try:
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

            if results:
                return results
        except Exception as err:
            logger.warning("Pinecone query fallback to local vector store", extra={"error": str(err)})

        # Local vector store fallback calculation
        matches: list[tuple[float, str, list[float], dict[str, Any]]] = []
        for vid, emb, meta in _local_vector_store:
            # Apply metadata filtering if provided
            if filter_dict:
                match_filter = True
                for fk, fv in filter_dict.items():
                    if isinstance(fv, dict) and "$eq" in fv:
                        if meta.get(fk) != fv["$eq"]:
                            match_filter = False
                    elif meta.get(fk) != fv:
                        match_filter = False
                if not match_filter:
                    continue

            score = _cosine_similarity(query_embedding, emb)
            matches.append((score, vid, emb, meta))

        matches.sort(key=lambda x: x[0], reverse=True)
        top_matches = matches[:top_k]

        return [
            VectorSearchResult(
                vector_id=vid,
                score=score,
                document_id=str(meta.get("document_id", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
                text_preview=str(meta.get("text_preview", "")),
                metadata=meta,
            )
            for score, vid, emb, meta in top_matches
        ]

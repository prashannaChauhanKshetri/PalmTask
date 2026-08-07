"""Application settings loaded from environment via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — every secret/tuneable lives here, never hardcoded."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── OpenAI ──────────────────────────────────────────
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536

    # ── Pinecone ────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index_name: str = "palm-rag"
    pinecone_host: str = "https://palm-rag-8y3zdyr.svc.aped-4627-b05a.pinecone.io"

    # ── PostgreSQL ──────────────────────────────────────
    database_url: str = "postgresql+asyncpg://palm:palm@postgres:5432/palm"

    # ── Redis ───────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"

    # ── RAG tunables ────────────────────────────────────
    rag_top_k: int = 5
    rag_score_threshold: float = 0.65
    rag_contextual_window: bool = True
    chat_window_size: int = 10
    chat_ttl_seconds: int = 3600

    # ── Chunking defaults ───────────────────────────────
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50

    # ── Application ─────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Factory that can be overridden in tests via FastAPI dependency injection."""
    return Settings()  # type: ignore[call-arg]


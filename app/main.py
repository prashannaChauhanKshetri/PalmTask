"""FastAPI application factory and lifespan management."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and shutdown hooks."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Starting Palm RAG API",
        extra={"environment": settings.environment},
    )
    yield
    logger.info("Shutting down Palm RAG API")


def create_app() -> FastAPI:
    """Application factory — assembles middleware, routes, and error handlers."""
    settings = get_settings()

    app = FastAPI(
        title="Palm Mind AI — RAG Backend",
        description=(
            "Production RAG backend with document ingestion, "
            "conversational search, and interview booking extraction."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    register_exception_handlers(app)

    # Register API Routers
    from app.routers.chat import router as chat_router
    from app.routers.documents import router as documents_router

    app.include_router(documents_router)
    app.include_router(chat_router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.environment}

    return app


app = create_app()

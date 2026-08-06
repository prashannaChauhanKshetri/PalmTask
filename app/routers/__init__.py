"""Routers package init."""

from app.routers.chat import router as chat_router
from app.routers.documents import router as documents_router

__all__: list[str] = ["documents_router", "chat_router"]

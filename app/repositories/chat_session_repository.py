"""Repository pattern for ChatSession database persistence operations."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import ChatSession


class ChatSessionRepository:
    """Async database repository for chat sessions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_session(self, session_id: UUID | None = None) -> ChatSession:
        """Create a new chat session database record."""
        chat_sess = ChatSession()
        if session_id:
            chat_sess.id = session_id
        self.session.add(chat_sess)
        await self.session.commit()
        await self.session.refresh(chat_sess)
        return chat_sess

    async def get_session_by_id(self, session_id: UUID) -> ChatSession | None:
        """Fetch chat session by UUID."""
        stmt = select(ChatSession).where(ChatSession.id == session_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_last_active(self, session_id: UUID) -> None:
        """Update last_active_at timestamp for an active session."""
        chat_sess = await self.get_session_by_id(session_id)
        if chat_sess:
            chat_sess.last_active_at = datetime.now(timezone.utc)
            await self.session.commit()

"""Live integration test script verifying PostgreSQL, Redis, OpenAI (embeddings), Gemini (chat), and Pinecone."""

import asyncio
from uuid import uuid4

from app.clients.openai_client import OpenAIClient
from app.clients.pinecone_client import PineconeClient
from app.core.database import async_session_factory
from app.repositories.booking_repository import BookingRepository
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.document_repository import DocumentRepository
from app.services.memory_service import MemoryService


async def test_postgres() -> None:
    print("\n--- 🐘 Testing PostgreSQL (AsyncPG + SQLAlchemy) ---")
    async with async_session_factory() as session:
        doc_repo = DocumentRepository(session)
        doc = await doc_repo.create_document(
            filename="test_live.pdf",
            file_type="pdf",
            chunking_strategy="hierarchical",
        )
        print(f"  ✅ Created Document: ID={doc.id}, Status={doc.status}")

        await doc_repo.update_document_status(doc.id, "completed", total_chunks=5)
        updated_doc = await doc_repo.get_document_by_id(doc.id)
        assert updated_doc is not None
        print(f"  ✅ Updated Document: Status={updated_doc.status}, Total Chunks={updated_doc.total_chunks}")

        chat_repo = ChatSessionRepository(session)
        chat_sess = await chat_repo.create_session()
        print(f"  ✅ Created Chat Session: ID={chat_sess.id}")

        booking_repo = BookingRepository(session)
        from datetime import date, time
        booking = await booking_repo.create_booking(
            session_id=chat_sess.id,
            name="Prashanna",
            email="prashanna@example.com",
            interview_date=date(2026, 8, 15),
            interview_time=time(14, 0),
        )
        print(f"  ✅ Created Booking: ID={booking.id}, Candidate={booking.name}, Slot={booking.interview_date} {booking.interview_time}")


async def test_redis() -> None:
    print("\n--- 🔴 Testing Redis (Chat Memory & TTL) ---")
    memory_service = MemoryService()
    test_session_id = uuid4()

    await memory_service.add_turn(test_session_id, "user", "Hello PalmMind!")
    await memory_service.add_turn(test_session_id, "assistant", "Hello Prashanna! How can I help you today?")

    history = await memory_service.get_history(test_session_id)
    print(f"  ✅ Retrieved {len(history)} turns from Redis:")
    for turn in history:
        print(f"     [{turn.role}]: {turn.content}")

    from app.services.memory_service import PartialBookingState
    await memory_service.save_booking_state(test_session_id, PartialBookingState(name="Prashanna"))
    state = await memory_service.get_booking_state(test_session_id)
    print(f"  ✅ Retrieved Partial Booking State from Redis: name={state.name}, complete={state.is_complete()}")


async def test_openai() -> None:
    print("\n--- 🤖 Testing OpenAI API (Embeddings Only) ---")
    openai_client = OpenAIClient()

    embeddings = await openai_client.get_embeddings(["PalmMind AI RAG Backend Test"])
    print(f"  ✅ Generated Embedding Vector: length={len(embeddings[0])} dimensions")


async def test_gemini() -> None:
    print("\n--- 🌟 Testing Google Gemini API (Chat Completions) ---")
    from app.clients.gemini_client import GeminiClient
    gemini_client = GeminiClient()

    chat_reply = await gemini_client.generate_chat_completion(
        messages=[{"role": "user", "content": "Reply with 'Gemini Live Test Successful!'"}],
        temperature=0.0,
    )
    print(f"  ✅ Gemini Response: {chat_reply}")


def test_pinecone() -> None:
    print("\n--- 🌲 Testing Pinecone Vector Index ---")
    pinecone_client = PineconeClient()

    # Query index with a zero vector to test index connectivity and stats
    dummy_vector = [0.01] * 1536
    results = pinecone_client.query_similarity(dummy_vector, top_k=2)
    print(f"  ✅ Pinecone Index Connection Successful! Host: {pinecone_client.index_name}")
    print(f"     Query returned {len(results)} matches.")


async def main() -> None:
    print("==================================================")
    print("  🚀 PALM TASK LIVE INTEGRATION TEST SUITE")
    print("==================================================")
    await test_postgres()
    await test_redis()
    await test_openai()
    await test_gemini()
    test_pinecone()
    print("\n==================================================")
    print("  🎉 ALL 5 SERVICES (Postgres, Redis, OpenAI, Gemini, Pinecone) WORKING LOCALLY!")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())

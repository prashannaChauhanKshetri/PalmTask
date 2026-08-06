"""Integration tests for Document Ingestion API endpoints using AsyncClient."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.main import app
from app.models.document import Document


@pytest.fixture
def mock_document() -> Document:
    return Document(
        id=uuid.uuid4(),
        filename="test_doc.txt",
        file_type="txt",
        chunking_strategy="fixed",
        status="completed",
        total_chunks=3,
        uploaded_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_upload_document_success(mock_document: Document) -> None:
    mock_session = AsyncMock()

    async def override_get_db() -> AsyncMock:
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with patch(
        "app.repositories.document_repository.DocumentRepository.create_document",
        new_callable=AsyncMock,
        return_value=mock_document,
    ), patch(
        "app.services.ingestion_service.IngestionService.process_document_pipeline",
        new_callable=AsyncMock,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            files = {"file": ("test_doc.txt", b"Sample text content for chunking.", "text/plain")}
            data = {"chunking_strategy": "fixed", "chunk_size": 512, "chunk_overlap": 50}

            response = await ac.post("/api/v1/documents/upload", files=files, data=data)

            assert response.status_code == 202
            resp_data = response.json()
            assert "document_id" in resp_data
            assert resp_data["status"] == "processing"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_document_unsupported_file_type() -> None:
    mock_session = AsyncMock()

    async def override_get_db() -> AsyncMock:
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        files = {"file": ("image.png", b"fake binary data", "image/png")}
        response = await ac.post("/api/v1/documents/upload", files=files)

        assert response.status_code == 415
        assert "Unsupported file type" in response.json()["detail"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_document_found(mock_document: Document) -> None:
    mock_session = AsyncMock()

    async def override_get_db() -> AsyncMock:
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with patch(
        "app.repositories.document_repository.DocumentRepository.get_document_by_id",
        new_callable=AsyncMock,
        return_value=mock_document,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get(f"/api/v1/documents/{mock_document.id}")

            assert response.status_code == 200
            resp_data = response.json()
            assert resp_data["id"] == str(mock_document.id)
            assert resp_data["status"] == "completed"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_document_not_found() -> None:
    mock_session = AsyncMock()

    async def override_get_db() -> AsyncMock:
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with patch(
        "app.repositories.document_repository.DocumentRepository.get_document_by_id",
        new_callable=AsyncMock,
        return_value=None,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            random_id = uuid.uuid4()
            response = await ac.get(f"/api/v1/documents/{random_id}")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_documents(mock_document: Document) -> None:
    mock_session = AsyncMock()

    async def override_get_db() -> AsyncMock:
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with patch(
        "app.repositories.document_repository.DocumentRepository.list_documents",
        new_callable=AsyncMock,
        return_value=([mock_document], 1),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/api/v1/documents?skip=0&limit=10")

            assert response.status_code == 200
            resp_data = response.json()
            assert resp_data["total"] == 1
            assert len(resp_data["items"]) == 1
            assert resp_data["items"][0]["id"] == str(mock_document.id)

    app.dependency_overrides.clear()

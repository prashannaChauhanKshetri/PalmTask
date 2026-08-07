# Palm Mind AI — Production RAG & Interview Booking Backend

Production-grade FastAPI backend for document ingestion, custom vector-based Conversational RAG, and multi-turn interview booking extraction.

> Built for the **PalmMind AI Take-Home Assessment**. Strictly adheres to all non-negotiable constraints: **No LangChain/chain abstractions**, **No FAISS/Chroma** (Pinecone serverless only), **Backend only**, **Strict typing discipline**, and **Clean layered modular architecture**.

---

## 🏗️ Architecture Overview

The system strictly follows a clean 3-tier architecture: **Routers → Services → Repositories / Clients**. Business logic never leaks into API route handlers.

```
                     ┌──────────────────────────────┐
                     │   FastAPI Routers (/api/v1)  │
                     │  (/documents, /chat, /ops)   │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │       Service Layer          │
                     │  - IngestionService          │
                     │  - RAGGenerator (Custom RAG) │
                     │  - BookingService            │
                     │  - MemoryService             │
                     │  - Chunker & Extractor       │
                     └──────┬───────────────┬───────┘
                            │               │
            ┌───────────────┘               └────────────────┐
            ▼                                                ▼
┌──────────────────────┐                         ┌───────────────────────┐
│ Repositories / DB    │                         │  External Clients     │
│ - DocumentRepository │                         │  - OpenAIClient       │
│ - BookingRepository  │                         │  - PineconeClient     │
│ - ChatSessionRepo    │                         │  - RedisClient        │
└───────────┬──────────┘                         └───────────┬───────────┘
            │                                                │
            ▼                                                ▼
  PostgreSQL (AsyncPG)                              Pinecone / Redis / OpenAI
```

### Module Organization

```
palm_task/
├── alembic/                 # Database migrations (Async PostgreSQL)
│   └── versions/            # Versioned migration scripts
├── app/
│   ├── clients/             # External SaaS SDK wrappers (OpenAI, Pinecone, Redis)
│   ├── core/                # Central config, database session, logging, exception handlers
│   ├── models/              # SQLAlchemy 2.0 ORM models with Mapped[...] annotations
│   ├── repositories/        # Async DB query abstraction layer (PostgreSQL)
│   ├── routers/             # FastAPI HTTP route handlers and dependency injection
│   ├── schemas/             # Pydantic v2 request and response schemas
│   ├── services/            # Core business logic:
│   │   ├── extractors.py    # TextExtractor protocol (PyMuPDF & TXT charset fallback)
│   │   ├── chunking.py      # Chunker protocol (Fixed, Recursive & Hierarchical strategies)
│   │   ├── ingestion_service.py # Non-blocking background ingestion pipeline
│   │   ├── memory_service.py # Redis windowed chat history (TTL) & partial booking state
│   │   ├── rag_generator.py # Semantic RAG: score filter, context window, re-ranking
│   │   └── booking_service.py # Structured slot extraction & validation
│   └── main.py              # FastAPI app factory, lifespan hooks, health check
├── tests/                   # Complete Pytest test suite (100% mocked externals)
├── docker-compose.yml       # Docker orchestrator for api, postgres, redis
├── Dockerfile               # Production multi-stage Python 3.11 image
├── pyproject.toml           # Strict Mypy, Ruff, and Pytest configuration
└── requirements.txt         # Locked dependency versions
```

---

## 🛠️ Tech Stack & Key Decisions

| Layer | Technology | Key Decisions & Rationale |
|---|---|---|
| Framework | **FastAPI (Python 3.11+)** | Async native throughout for non-blocking I/O and automatic OpenAPI specs. |
| Vector DB | **Pinecone (Serverless)** | Vector database with cosine similarity metric (1536-dim vectors). |
| Chat Memory | **Redis 7 (Alpine)** | Fast, in-memory windowed chat history (`chat:{session_id}`) with automatic TTL expiry. |
| Relational DB | **PostgreSQL 16 + AsyncPG** | SQLAlchemy 2.0 declarative models with Alembic migrations for document/booking tracking. |
| Embeddings | **OpenAI `text-embedding-3-small`** | Cost-effective, high-accuracy 1536-dimensional embeddings. |
| LLM | **Google Gemini (`gemini-flash-latest`)** | Used for RAG generation and multi-turn structured booking extraction via structured JSON output. |
| Validation | **Pydantic v2 + dateparser + email-validator** | Robust runtime validation for extracted slots and requests. |

---

## 🧠 Chunking Strategies & Design Rationale

The backend provides **three** distinct, pluggable chunking strategies matching the `Chunker` protocol:

1. **`fixed` (Token-Count Sliding Window)**:
   - Uses `tiktoken` (`cl100k_base`) to count exact tokens instead of naive character counts.
   - Default: `chunk_size=512` tokens, `chunk_overlap=50` tokens.
   - *Rationale*: Provides a deterministic baseline suitable for unstructured, non-hierarchical documents.

2. **`recursive` (Structure-Aware Recursive Splitting)**:
   - Hierarchically splits document text on paragraph boundaries (`\n\n`), then sentence boundaries (`[.!?]`), then words.
   - Recombines adjacent small fragments up to `chunk_size` while retaining `chunk_overlap`.
   - *Rationale*: Preserves paragraph context and semantic boundaries, preventing sentence truncation across chunk boundaries.

3. **`hierarchical` (Section-Aware Chunking with Breadcrumb Context)** ✨:
   - Detects document structure via markdown headers (`#`, `##`, `###`), numbered sections (`1.`, `1.2.`), and ALL-CAPS headings.
   - Prepends hierarchical breadcrumb prefixes (e.g., `[Section: Architecture > API Layer]`) to each chunk.
   - Falls back to recursive chunking for unstructured documents without detectable headers.
   - *Rationale*: Each chunk carries its section lineage, dramatically improving retrieval quality for structured documents (technical docs, policies, manuals) since the LLM always knows *where* in the document a chunk originates.

---

## 🔍 Semantic Search Enhancements

The RAG pipeline goes beyond basic top-k vector similarity with a **multi-stage semantic retrieval pipeline**:

| Stage | Feature | Description |
|---|---|---|
| 1 | **Query Embedding** | Embeds user query via OpenAI `text-embedding-3-small` (1536-dim). |
| 2 | **Vector Similarity** | Top-k nearest neighbor search in Pinecone (cosine metric). |
| 3 | **Score Threshold Filtering** | Discards results below `RAG_SCORE_THRESHOLD` (default: 0.65) to eliminate noise. |
| 4 | **Contextual Window Expansion** | Fetches neighboring chunks (N-1, N+1) for matched results to provide surrounding document context. |
| 5 | **Document-Grouped Re-ranking** | Groups chunks by document, orders groups by best match score, preserves chunk order within groups for coherent context. |
| 6 | **Prompt Assembly** | Assembles system prompt with safety guardrails and formatted context blocks. |
| 7 | **LLM Generation** | Google Gemini (`gemini-flash-latest`) with temperature 0.2 for high precision. |

The `retrieval_stats` field in the chat response exposes pipeline metrics:
```json
{
  "retrieval_stats": {
    "filtered_count": 2,
    "expanded_count": 4
  }
}
```

Configurable via environment variables:
```env
RAG_SCORE_THRESHOLD=0.65
RAG_CONTEXTUAL_WINDOW=true
```

---

## 🛡️ AI Prompt Engineering & Safety Guardrails

The RAG Generator ([rag_generator.py](file:///Users/prashanna/Desktop/palm_task/app/services/rag_generator.py)) is **100% hand-crafted** without any LangChain abstractions. It explicitly implements:

- **Strict Grounding**: System prompt strictly limits responses to facts present in retrieved document context.
- **Hallucination Prevention**: Explicit instruction to reply *"I'm sorry, but I don't have enough information in the uploaded documents to answer that question."* if context is insufficient.
- **Prompt Injection Defense**: Mandates ignoring user or document attempts to override rules or reveal system prompts.
- **Contextual Coherence**: When multiple chunks from the same document are provided (via contextual window expansion), the LLM synthesizes them into a coherent answer.
- **Source Attribution**: Returns source metadata (`document_id`, `chunk_index`, similarity score, and text preview) alongside every answer.

---

## 🚀 Getting Started

### 1. Environment Configuration

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-proj-...          # Required for embeddings
GEMINI_API_KEY=AQ.Ab8RN6...          # Required for chat/booking LLM
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=palm-rag
DATABASE_URL=postgresql+asyncpg://palm:palm@localhost:5432/palm
REDIS_URL=redis://localhost:6379/0
```

### 2. Run with Docker Compose

Start the full stack (`api`, `postgres`, `redis`):

```bash
docker-compose up --build -d
```

Apply database migrations:

```bash
docker-compose exec api alembic upgrade head
```

The API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🧪 Running Unit & Integration Tests

The test suite runs with **100% mocked external SaaS APIs** (OpenAI, Pinecone, Redis). You can run tests without an active `.env` or network connection.

```bash
python3.11 -m pytest tests/ -v
```

---

## 📡 Example API Requests

### 1. Upload a Document (`POST /api/v1/documents/upload`)

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@sample.pdf" \
  -F "chunking_strategy=recursive" \
  -F "chunk_size=512" \
  -F "chunk_overlap=50"
```

**Response (HTTP 202 Accepted)**:
```json
{
  "document_id": "8cd63d8f-9a08-4ef7-bf09-aa2cdcb2cf4c",
  "filename": "sample.pdf",
  "status": "processing",
  "total_chunks": 0,
  "message": "Document uploaded successfully. Ingestion pipeline is running in background."
}
```

### 2. Check Document Ingestion Status (`GET /api/v1/documents/{id}`)

```bash
curl "http://localhost:8000/api/v1/documents/8cd63d8f-9a08-4ef7-bf09-aa2cdcb2cf4c"
```

**Response (HTTP 200 OK)**:
```json
{
  "id": "8cd63d8f-9a08-4ef7-bf09-aa2cdcb2cf4c",
  "filename": "sample.pdf",
  "file_type": "pdf",
  "chunking_strategy": "recursive",
  "status": "completed",
  "total_chunks": 12,
  "uploaded_at": "2026-08-07T00:00:00Z"
}
```

### 3. Conversational RAG Query (`POST /api/v1/chat`)

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What technical stack does PalmMind AI use?"
  }'
```

**Response**:
```json
{
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "answer": "PalmMind AI utilizes FastAPI for the web framework, PostgreSQL for metadata, Pinecone for vector search, and Redis for chat memory.",
  "sources": [
    {
      "document_id": "8cd63d8f-9a08-4ef7-bf09-aa2cdcb2cf4c",
      "chunk_index": 1,
      "score": 0.925,
      "text_preview": "Tech Stack: FastAPI, PostgreSQL, Pinecone serverless vector database..."
    }
  ]
}
```

### 4. Multi-turn Interview Booking Sub-flow

**Turn 1**:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "message": "I would like to book an interview slot. My name is Prashanna."
  }'
```
**Response**: *"I'd be happy to help schedule your interview! To complete your booking, please provide your email, interview date, interview time."*

**Turn 2**:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "message": "My email is prashanna@example.com and I want next Tuesday at 2pm."
  }'
```
**Response**:
```json
{
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "answer": "🎉 Your interview has been successfully booked!\n• Name: Prashanna\n• Email: prashanna@example.com\n• Date: 2026-08-11\n• Time: 14:00\nWe look forward to speaking with you!",
  "sources": []
}
```

### 5. Fetch Session Chat History (`GET /api/v1/chat/{session_id}/history`)

```bash
curl "http://localhost:8000/api/v1/chat/f47ac10b-58cc-4372-a567-0e02b2c3d479/history"
```

---

## ⚖️ License

MIT License — free to use and distribute.

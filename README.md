# Agentic RAG Chatbot — Multi-Format Document QA using Model Context Protocol (MCP)
**Presentation** - https://docs.google.com/presentation/d/1J74nwHso8BU1bjjyOsHPDL-hx5hX_UKr/edit?usp=sharing&ouid=107532975049972498907&rtpof=true&sd=true
**Video Demo** - https://drive.google.com/file/d/1B-mahle-V1tVb0MiwF8FHfXPzwRoFmbK/view?usp=sharing
> **Coding Round Submission** — Agent-based RAG system with LangGraph orchestration, hybrid retrieval, Gemini Vision OCR, and RAGAS-inspired evaluation.

---

## What This Project Does

Upload any document (PDF, PPTX, CSV, DOCX, XLSX, TXT, Markdown) and ask questions about it in natural language. The system uses **5 cooperating AI agents** that communicate exclusively through **structured MCP messages** on a shared message bus. Every agent interaction is visible in real time in the UI's MCP Trace panel.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND  (Browser)                       │
│   Documents Panel │ Chat Panel │ MCP Trace + RAG Health     │
│              Vanilla JS + CSS  (no build step)               │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP REST (JSON / multipart)
┌──────────────────────────▼──────────────────────────────────┐
│                  FASTAPI  BACKEND                            │
│   POST /upload   POST /chat   POST /evaluate   GET /trace   │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │    CoordinatorAgent     │  ← LangGraph StateGraph
              │      (Orchestrator)     │    routes all requests
              └──┬─────────┬────────┬───┘
                 │         │        │
    ┌────────────▼─┐  ┌────▼────┐  ┌▼─────────────┐
    │  Ingestion   │  │Retrieval│  │  Evaluation  │
    │    Agent     │  │  Agent  │  │    Agent     │
    └────────────┬─┘  └────┬────┘  └──────────────┘
                 │         │
                 │   ┌─────▼──────────┐
                 │   │ LLMResponse    │
                 │   │    Agent       │
                 │   └────────────────┘
                 │         │
    ┌────────────▼─────────▼────────────────────────┐
    │              MCP MESSAGE BUS                   │
    │    (in-memory, trace_id → [MCPMessage, ...])   │
    └────────────────────────────────────────────────┘
                 │         │
    ┌────────────▼─┐  ┌────▼──────────────────────┐
    │    Neon      │  │    Google Gemini API        │
    │  PostgreSQL  │  │  gemini-2.5-flash  (LLM)   │
    │  + pgvector  │  │  gemini-embedding-001       │
    │              │  │  gemini-2.0-flash  (OCR)   │
    └──────────────┘  └────────────────────────────┘
```

---

## The 5 Agents

### 1. CoordinatorAgent (`coordinator_agent.py`)
The **only agent that uses LangGraph**. Contains a compiled `StateGraph` that routes every request to the correct agent pipeline based on mode (`ingest` / `query` / `evaluate`). Does not call Gemini or the database directly — it only orchestrates.

### 2. IngestionAgent (`ingestion_agent.py`)
Receives a file path, calls `document_parser` to extract text chunks, and stores them in Neon via `NeonVectorStore`. Publishes `INGEST_REQUEST` before and `INGEST_RESULT` after to the MCP bus.

### 3. RetrievalAgent (`retrieval_agent.py`)
Runs a **4-stage hybrid retrieval pipeline**:
1. **Vector search** — pgvector cosine similarity using Gemini 768-dim embeddings
2. **BM25 keyword search** — PostgreSQL `ts_rank` on stored `tsvector` column (term frequency + IDF + length normalisation). Falls back to `ILIKE` if stemming yields no results.
3. **RRF Fusion** — Reciprocal Rank Fusion merges both ranked lists (`0.65 × vector + 0.35 × BM25 + 0.15 overlap bonus`)
4. **Gemini Reranking** — sends top candidates to `gemini-2.5-flash` to re-order by true relevance

Publishes `RETRIEVAL_REQUEST` and `RETRIEVAL_RESULT` to MCP bus.

### 4. LLMResponseAgent (`llm_response_agent.py`)
Detects follow-up questions (pronouns, continuation phrases) and rewrites them using conversation history. Builds the full RAG prompt with retrieved context and conversation turns, calls `gemini-2.5-flash`, and stores the Q&A turn in `ConversationStore`. Publishes `LLM_REQUEST` and `LLM_RESPONSE`.

### 5. EvaluationAgent (`evaluation_agent.py`)
RAGAS-inspired quality scoring. Sends the question + context + answer to `gemini-2.5-flash` acting as a judge and asks it to score 3 metrics (0.0–1.0):
- **Faithfulness** — are claims grounded in context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are retrieved chunks relevant?
- **Overall** — average of all three

Publishes `EVAL_REQUEST` and `EVAL_RESULT` to MCP bus. Triggered when user clicks "Evaluate".

---

## MCP Message Flow

All agents communicate **only** through typed MCP messages on the shared bus. No agent calls another agent directly.

### For a user question (6 messages):

```
CoordinatorAgent  ──RETRIEVAL_REQUEST──►  MCP Bus
RetrievalAgent    ──RETRIEVAL_RESULT───►  MCP Bus
CoordinatorAgent  ──LLM_REQUEST────────►  MCP Bus
LLMResponseAgent  ──LLM_RESPONSE───────►  MCP Bus
CoordinatorAgent  ──EVAL_REQUEST────────►  MCP Bus   (on Evaluate click)
EvaluationAgent   ──EVAL_RESULT─────────►  MCP Bus
```

### For a document upload (2 messages):

```
CoordinatorAgent  ──INGEST_REQUEST──►  MCP Bus
IngestionAgent    ──INGEST_RESULT───►  MCP Bus
```

### Example MCP Message:
```json
{
  "sender": "RetrievalAgent",
  "receiver": "LLMResponseAgent",
  "type": "RETRIEVAL_RESULT",
  "trace_id": "abc-123",
  "timestamp": "2026-04-30T10:23:45.123456Z",
  "payload": {
    "query": "What KPIs were tracked in Q1?",
    "top_chunks": ["slide 3: revenue up 12%...", "doc: Q1 summary..."],
    "scores": [0.87, 0.74],
    "chunk_count": 2
  }
}
```

---

## Document Parsing & Chunking

| Format | Parser | Strategy | Notes |
|--------|--------|----------|-------|
| PDF (digital) | pypdf | One chunk per page | Requires ≥80 alpha chars |
| PDF (scanned) | PyMuPDF + Gemini Vision | Sliding window on OCR text | 200 DPI render → gemini-2.0-flash OCR |
| PPTX | python-pptx | One chunk per slide | Includes speaker notes |
| DOCX | python-docx | Sliding window 600 chars / 80 overlap | Tables included |
| XLSX | pandas | 15 rows per chunk, all sheets | Each sheet parsed separately |
| CSV | pandas | 15 rows per chunk | Column=value format |
| TXT / MD | built-in | Sliding window 600 chars / 80 overlap | |

Every chunk is tagged: `[source: filename | page 3]` so citations are accurate.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM (answers + reranking + evaluation) | Google Gemini `gemini-2.5-flash` |
| OCR (scanned PDFs) | Google Gemini `gemini-2.0-flash` Vision |
| Embeddings | Google Gemini `gemini-embedding-001` (768-dim, L2-normalized) |
| Vector database | Neon PostgreSQL + `pgvector` extension |
| Agent orchestration | LangGraph `StateGraph` |
| Backend API | FastAPI (Python) |
| Frontend | Vanilla HTML + CSS + JavaScript (no build step) |
| MCP bus | Custom in-memory bus (`threading.Lock`) |
| PDF parsing | pypdf + PyMuPDF (fitz) |
| Office formats | python-pptx, python-docx, pandas |
| Conversation memory | Custom `ConversationStore` (last 4 turns per session) |

---

## Setup and Run

### Prerequisites
- Python 3.10+
- A [Neon](https://neon.tech) account (free tier works)
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier works)

### Step 1 — Clone and create virtual environment

```bash
git clone <your-repo-url>
cd agentic-rag-mcp/backend

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment

Copy `.env.example` to `.env` and fill in:

```env
GEMINI_API_KEY=your_gemini_api_key_here
NEON_DATABASE_URL=postgresql://user:password@ep-xxx.region.neon.tech/dbname?sslmode=require
```

Get your Neon connection string from the Neon dashboard → Connection Details.

### Step 4 — Start the backend

```bash
# From inside backend/ folder
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Wait for: `Startup complete: document_chunks table and indexes ready.`

### Step 5 — Open the frontend

Open `frontend/index.html` in your browser. No npm or Node.js required.

---

## Usage

1. **Upload a document** — drag and drop or click "Choose file" in the left panel
2. **Ask a question** — type in the chat box and press Send
3. **View sources** — each answer shows which file and page/chunk the answer came from
4. **Follow-up questions** — ask "what about that?" or "explain more" — the system understands context
5. **Evaluate quality** — click "Evaluate" under any answer to see RAGAS-inspired scores
6. **Inspect MCP messages** — click "MCP Trace" tab on the right to see all agent messages
7. **Check RAG Health** — click "RAG Health" tab to see the latest evaluation scores

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload and ingest a document |
| `POST` | `/chat` | Ask a question, get answer + MCP trace |
| `POST` | `/evaluate` | Score a response with RAGAS-inspired metrics |
| `GET` | `/trace/{trace_id}` | Fetch full MCP message log for a request |
| `GET` | `/documents` | List all uploaded files and chunk counts |
| `GET` | `/health` | Backend health, DB status, chunk totals |
| `DELETE` | `/reset` | Clear all documents, vectors, and session data |

---

## Project Structure

```
agentic-rag-mcp/
├── .env                        ← API keys (not committed to git)
├── .env.example                ← Template for .env
├── README.md
├── architecture.pptx           ← Presentation (see below)
├── backend/
│   ├── main.py                 ← FastAPI app, all endpoints
│   ├── config.py               ← Settings from .env
│   ├── database.py             ← Neon async session factory
│   ├── requirements.txt
│   ├── agents/
│   │   ├── coordinator_agent.py  ← LangGraph orchestrator
│   │   ├── ingestion_agent.py    ← Document parse + embed + store
│   │   ├── retrieval_agent.py    ← Hybrid search (4-stage)
│   │   ├── llm_response_agent.py ← Gemini answer generation
│   │   └── evaluation_agent.py   ← RAGAS-inspired scoring
│   ├── mcp/
│   │   ├── message.py            ← MCPMessage schema + MessageType enum
│   │   └── bus.py                ← In-memory MCP bus (publish / get_trace)
│   └── services/
│       ├── document_parser.py    ← Per-format parsing + OCR
│       ├── neon_store.py         ← pgvector store + hybrid retrieval
│       ├── prompt_assembler.py   ← RAG prompt builder
│       └── context_store.py      ← Multi-turn conversation memory
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## Key Design Decisions

**Why Gemini instead of OpenAI/HuggingFace?**
Single API key for embeddings, LLM, OCR, reranking, and evaluation — no separate services, no HuggingFace model downloads.

**Why Neon PostgreSQL instead of FAISS/Chroma?**
Neon is a serverless cloud PostgreSQL — no local files, persistent across restarts, and `pgvector` provides production-grade vector search. FAISS requires in-memory loading and is lost on restart.

**Why LangGraph for coordination?**
Makes the agent routing explicit and inspectable. Adding a new agent is one `add_node` + `add_edge` call. The state machine is compiled once at startup.

**Why in-memory MCP bus?**
Meets the assignment requirement ("You can implement MCP using in-memory messaging, REST, or pub/sub"). All messages for a request share a `trace_id` and are returned with the API response so the frontend can display them instantly.

---

## Challenges Faced

1. **Scanned PDF OCR** — pypdf extracts garbled text from scanned PDFs (text layer from scanner, not real text). Fixed by checking if extracted text has ≥80 alphabetic characters; if not, PyMuPDF renders the page as a PNG and Gemini Vision reads it.

2. **Gemini response truncation in evaluation** — `gemini-2.5-flash` uses internal thinking tokens. With low `max_output_tokens`, thinking consumed the budget and JSON was cut mid-way. Fixed by switching to a comma-separated number format (`0.9,0.8,0.75`) which needs only ~15 tokens.

3. **Hybrid retrieval fusion** — Vector search and keyword search return different ranked lists. Used Reciprocal Rank Fusion (RRF) with weights (65% vector, 35% keyword, 15% overlap bonus) to produce a single merged ranking.

4. **Follow-up question resolution** — "What about that?" has no vector signal. Solved by detecting follow-up patterns (pronouns, continuation phrases) and rewriting the question using conversation history before retrieval.

---

## Future Improvements

- Persistent conversation memory (store in DB, survive restarts)
- Semantic chunking (split at headings/paragraph boundaries instead of fixed size)
- Auto-evaluation after every response (not just on button click)
- Support for image-heavy PDFs with multiple scanned pages
- RAGAS score history chart over multiple queries
- Authentication + multi-user session isolation

---

## Alternative Implementation Plan

This section describes an alternative architecture for the same system using a different toolchain. It serves as a design comparison — the actual implementation uses the stack described above (Gemini + Neon + Vanilla JS).

### Alternative Architecture

```
React Frontend (Vite + JSX)
       ↕  REST + Server-Sent Events (SSE streaming)
FastAPI Backend
       ↕
CoordinatorAgent  ←  LangGraph StateGraph
       ↕  MCP messages
┌─────────────────────────────────────────────────────┐
│  IngestionAgent   →  RetrievalAgent  →  LLMAgent   │
│        ↕                  ↕                          │
│    Docling           ChromaDB                        │
│  (PDF + OCR)      BM25 (rank-bm25)                  │
│                   Cross-encoder Reranker             │
└─────────────────────────────────────────────────────┘
       ↕
EvaluationAgent  →  RAGAS library (faithfulness / relevancy / precision)
```

### Alternative Tech Stack

| Component | This Project | Alternative |
|-----------|-------------|-------------|
| LLM | Google Gemini `gemini-2.5-flash` | OpenAI `gpt-4o-mini` or Ollama `llama3.2` (free, local) |
| Embeddings | Gemini `gemini-embedding-001` (768-dim) | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim, runs locally) |
| OCR | Gemini Vision `gemini-2.0-flash` | Docling (handles PDFs, scanned pages, tables natively) |
| Vector Store | Neon PostgreSQL + pgvector (cloud) | ChromaDB (local, file-based, no cloud needed) |
| Keyword Search | PostgreSQL `ts_rank` / tsvector (BM25-like) | `rank-bm25` library (`BM25Okapi`, pure Python) |
| Reranker | Gemini re-orders top-K via prompt | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, HuggingFace) |
| RAGAS Evaluation | Custom Gemini-judged scoring | `ragas` library + HuggingFace `datasets` |
| Frontend | Vanilla HTML/CSS/JS (no build step) | React (Vite + JSX components) |
| Deployment | FastAPI serves frontend via `StaticFiles` | Separate backend (port 8000) + frontend dev server (port 3000) |

### Alternative Folder Structure

```
agentic-rag-chatbot/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   ├── mcp/
│   │   └── message.py               ← MCPMessage Pydantic schema
│   ├── agents/
│   │   ├── coordinator.py           ← LangGraph StateGraph
│   │   ├── ingestion_agent.py       ← Docling parser + chunker
│   │   ├── retrieval_agent.py       ← Hybrid search + reranker
│   │   ├── llm_response_agent.py    ← OpenAI / Ollama with streaming
│   │   └── evaluation_agent.py      ← RAGAS library evaluation
│   ├── parsers/
│   │   └── document_parser.py       ← Docling + format routing
│   └── retrieval/
│       ├── embedder.py              ← sentence-transformers model
│       ├── vector_store.py          ← ChromaDB wrapper
│       ├── bm25_index.py            ← rank-bm25 index (pickle-persisted)
│       ├── hybrid_retriever.py      ← RRF fusion (semantic + BM25)
│       └── reranker.py              ← cross-encoder scoring
└── frontend/
    ├── package.json
    └── src/
        ├── App.jsx
        ├── api.js
        └── components/
            ├── ChatWindow.jsx
            ├── DocumentUpload.jsx
            ├── MCPTrace.jsx
            └── RAGHealthDashboard.jsx
```

### Alternative: Document Parser (Docling)

Docling is a document understanding library that handles PDF, DOCX, PPTX natively including tables and multi-column layouts:

```python
from docling.document_converter import DocumentConverter

def parse_pdf_with_docling(file_path: str):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    # Exports with layout structure preserved
    full_text = result.document.export_to_markdown()
    return full_text
```

Compared to the current approach (pypdf + PyMuPDF + Gemini Vision), Docling runs entirely locally with no API calls for OCR but requires more disk space (~2GB for models).

### Alternative: ChromaDB Vector Store

```python
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./storage/chroma_db")
collection = client.get_or_create_collection("rag_docs", metadata={"hnsw:space": "cosine"})

# Add chunks
embeddings = model.encode([c["text"] for c in chunks]).tolist()
collection.add(ids=[f"chunk_{i}" for i in range(len(chunks))],
               embeddings=embeddings,
               documents=[c["text"] for c in chunks])

# Query
results = collection.query(query_embeddings=[model.encode([query])[0].tolist()], n_results=10)
```

Compared to Neon pgvector: ChromaDB stores locally and requires no cloud account, but data is lost if the storage folder is deleted and it does not support SQL joins or full-text search in the same database.

### Alternative: BM25 with rank-bm25

```python
from rank_bm25 import BM25Okapi

def build_bm25_index(chunks):
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)

def bm25_search(bm25, chunks, query, top_k=10):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"chunk": chunks[i], "score": s} for i, s in ranked if s > 0]
```

Compared to the current `ts_rank` approach: `rank-bm25` runs in Python memory (no database round-trip) but the index is lost on restart and must be rebuilt. The current implementation stores BM25 state implicitly in the `tsvector` column which survives restarts.

### Alternative: Cross-Encoder Reranker

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, chunks, top_k=5):
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
```

A cross-encoder reads the query and chunk together (not separately like bi-encoders) so it captures more nuanced relevance. More accurate than embedding similarity alone, but ~10× slower, making it unsuitable for first-stage retrieval. Used only to re-score the top-20 candidates — same RRF fusion strategy as this project.

### Alternative: RAGAS Library Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

data = Dataset.from_dict({
    "question": [question],
    "answer": [answer],
    "contexts": [context_chunks],
})
result = evaluate(data, metrics=[faithfulness, answer_relevancy, context_precision])
# result["faithfulness"], result["answer_relevancy"], result["context_precision"]
```

Compared to the current approach (Gemini-as-judge with comma-separated scores): the RAGAS library calls an LLM internally (OpenAI by default, configurable) and runs multiple sub-prompts per metric. It is more standardised but requires an extra API key and adds latency. The current Gemini-based approach achieves the same three metrics in one API call.

### Trade-offs Summary

| Aspect | This Project | Alternative |
|--------|-------------|-------------|
| **API keys needed** | 2 (Gemini + Neon) | 1–2 (OpenAI + optional) |
| **Cost at zero usage** | Neon free tier + Gemini free tier | Ollama = $0 (fully local) |
| **Setup complexity** | Simple (`pip install`, one `.env`) | More complex (Node.js for React, HuggingFace model downloads ~500MB) |
| **Persistence** | Cloud Neon DB survives restarts | ChromaDB file-based (must back up `storage/` folder) |
| **Streaming answers** | Not implemented (full response) | SSE streaming token-by-token with React |
| **OCR quality** | Gemini Vision (very high, cloud) | Docling (good, local, no API cost) |
| **Reranking accuracy** | Gemini prompt (fast, 1 API call) | Cross-encoder (more accurate, local, slower) |
| **Frontend build step** | None (open HTML directly) | `npm install && npm run dev` required |

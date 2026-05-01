import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from backend import database
from backend.agents.coordinator_agent import CoordinatorAgent
from backend.config import settings
from backend.mcp.bus import mcp_bus
from backend.services.context_store import conversation_store
from backend.services.neon_store import neon_store

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Agentic RAG Chatbot with MCP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


coordinator = CoordinatorAgent()


ALLOWED_EXTENSIONS = {"pdf", "pptx", "csv", "docx", "txt", "md", "xlsx"}


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    context_chunks: List[str]
    session_id: str = "default"


@app.on_event("startup")
async def on_startup():
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    await database.setup_vector_table()
    logger.info("Startup complete: document_chunks table and indexes ready.")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: "
                   f"{', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    target_path = upload_dir / safe_name

    contents = await file.read()
    with open(target_path, "wb") as f:
        f.write(contents)

    result = await asyncio.to_thread(
        coordinator.handle_ingest, str(target_path), file.filename
    )
    return result


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result = await asyncio.to_thread(
        coordinator.handle_query, req.question.strip(), req.session_id
    )
    return result


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    """Run RAGAS-inspired evaluation on a question/answer/context triple."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="Answer must not be empty.")

    result = await asyncio.to_thread(
        coordinator.handle_evaluate,
        req.question.strip(),
        req.answer.strip(),
        req.context_chunks,
        req.session_id,
    )
    return result


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    return mcp_bus.get_trace(trace_id)


@app.get("/documents")
async def get_documents():
    return await asyncio.to_thread(neon_store.get_doc_list)


@app.get("/health")
async def health():
    total = await asyncio.to_thread(neon_store.total_chunks)
    docs = await asyncio.to_thread(neon_store.get_doc_list)
    db_ok = await database.check_db_health()
    return {
        "status": "ok",
        "total_chunks": total,
        "total_docs": len(docs),
        "db_healthy": db_ok,
    }


@app.delete("/reset")
async def reset():
    await asyncio.to_thread(neon_store.clear)
    mcp_bus.clear()
    conversation_store.clear_all()
    return {"status": "reset", "message": "Vector store, MCP bus, and all sessions cleared."}


_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount(
        "/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static"
    )

    @app.get("/")
    async def serve_index():
        return FileResponse(_FRONTEND_DIR / "index.html")

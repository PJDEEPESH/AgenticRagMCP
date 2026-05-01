"""LangGraph-orchestrated CoordinatorAgent.

State machine routes three flows:
  ingest  → IngestionAgent → END
  query   → RetrievalAgent → LLMResponseAgent → END
  evaluate → EvaluationAgent → END

Every node communicates via MCPMessage objects published to the shared bus.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.ingestion_agent import IngestionAgent
from backend.agents.llm_response_agent import LLMResponseAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.mcp.bus import mcp_bus
from backend.services.context_store import ConversationTurn, conversation_store

logger = logging.getLogger(__name__)


# ── Shared graph state ──────────────────────────────────────────────────────

class RAGState(TypedDict):
    mode: str                           # "ingest" | "query" | "evaluate"
    session_id: str
    trace_id: str
    # ingest fields
    file_path: Optional[str]
    filename: Optional[str]
    # query fields
    query: Optional[str]
    # evaluate fields (pre-populated when mode == "evaluate")
    eval_question: Optional[str]
    eval_answer: Optional[str]
    eval_chunks: Optional[List[str]]
    # outputs populated by nodes
    ingestion_result: Optional[Dict[str, Any]]
    retrieval_result: Optional[Dict[str, Any]]
    llm_result: Optional[Dict[str, Any]]
    eval_result: Optional[Dict[str, Any]]
    error: Optional[str]


# ── Agent singletons (module-level, instantiated once) ─────────────────────

_ingestion_agent = IngestionAgent()
_retrieval_agent = RetrievalAgent()
_llm_agent = LLMResponseAgent()


# ── Node functions ──────────────────────────────────────────────────────────

def _ingest_node(state: RAGState) -> Dict[str, Any]:
    msg = _ingestion_agent.ingest(
        state["file_path"], state["filename"], state["trace_id"]
    )
    return {
        "ingestion_result": msg.payload,
        "error": msg.payload.get("error"),
    }


def _retrieve_node(state: RAGState) -> Dict[str, Any]:
    history = conversation_store.get_context_window(state["session_id"], n=4)
    resolved = _llm_agent.resolve_followup(state["query"], history)

    msg = _retrieval_agent.retrieve(resolved, state["trace_id"])
    return {
        "retrieval_result": {
            **msg.payload,
            "resolved_query": resolved,
        },
    }


def _answer_node(state: RAGState) -> Dict[str, Any]:
    retrieval = state["retrieval_result"] or {}
    chunks = retrieval.get("top_chunks", [])
    scores = retrieval.get("scores", [])
    resolved = retrieval.get("resolved_query", state["query"])

    history = conversation_store.get_context_window(state["session_id"], n=4)

    msg = _llm_agent.generate(
        question=state["query"],
        resolved_question=resolved,
        chunks=chunks,
        scores=scores,
        trace_id=state["trace_id"],
        history=history,
    )
    answer = msg.payload.get("answer", "")

    conversation_store.add_turn(
        state["session_id"],
        ConversationTurn(
            turn_id=str(uuid.uuid4()),
            question=state["query"],
            answer=answer,
            source_chunks=chunks,
            trace_id=state["trace_id"],
            timestamp=datetime.utcnow().isoformat() + "Z",
        ),
    )
    return {"llm_result": msg.payload}


def _evaluate_node(state: RAGState) -> Dict[str, Any]:
    from backend.agents.evaluation_agent import EvaluationAgent  # lazy import
    scores = EvaluationAgent().evaluate(
        question=state["eval_question"] or "",
        answer=state["eval_answer"] or "",
        context_chunks=state["eval_chunks"] or [],
        trace_id=state["trace_id"],
    )
    return {"eval_result": scores}


# ── Graph construction ──────────────────────────────────────────────────────

def _route(state: RAGState) -> str:
    return state["mode"]  # "ingest" | "query" | "evaluate"


def _build_graph() -> Any:
    g = StateGraph(RAGState)

    g.add_node("ingest", _ingest_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("answer", _answer_node)
    g.add_node("evaluate", _evaluate_node)

    g.add_conditional_edges(
        START,
        _route,
        {"ingest": "ingest", "query": "retrieve", "evaluate": "evaluate"},
    )
    g.add_edge("ingest", END)
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    g.add_edge("evaluate", END)

    return g.compile()


_graph = _build_graph()


# ── Public facade ───────────────────────────────────────────────────────────

class CoordinatorAgent:
    AGENT_NAME = "CoordinatorAgent"

    def handle_ingest(self, file_path: str, filename: str) -> Dict[str, Any]:
        trace_id = str(uuid.uuid4())
        state = _graph.invoke(
            {
                "mode": "ingest",
                "session_id": "system",
                "trace_id": trace_id,
                "file_path": file_path,
                "filename": filename,
                "query": None,
                "eval_question": None,
                "eval_answer": None,
                "eval_chunks": None,
                "ingestion_result": None,
                "retrieval_result": None,
                "llm_result": None,
                "eval_result": None,
                "error": None,
            }
        )
        result = state.get("ingestion_result") or {}
        return {
            "trace_id": trace_id,
            "filename": result.get("filename"),
            "chunks_stored": result.get("chunks_stored"),
            "status": result.get("status"),
            "mcp_trace": mcp_bus.get_trace(trace_id),
        }

    def handle_query(
        self, question: str, session_id: str = "default"
    ) -> Dict[str, Any]:
        trace_id = str(uuid.uuid4())
        state = _graph.invoke(
            {
                "mode": "query",
                "session_id": session_id,
                "trace_id": trace_id,
                "file_path": None,
                "filename": None,
                "query": question,
                "eval_question": None,
                "eval_answer": None,
                "eval_chunks": None,
                "ingestion_result": None,
                "retrieval_result": None,
                "llm_result": None,
                "eval_result": None,
                "error": None,
            }
        )
        llm = state.get("llm_result") or {}
        retrieval = state.get("retrieval_result") or {}
        chunks = retrieval.get("top_chunks", [])
        scores = retrieval.get("scores", [])

        return {
            "trace_id": trace_id,
            "session_id": session_id,
            "question": question,
            "resolved_question": retrieval.get("resolved_query", question),
            "answer": llm.get("answer", ""),
            "source_chunks": [
                {"chunk": c, "score": s} for c, s in zip(chunks, scores)
            ],
            "mcp_trace": mcp_bus.get_trace(trace_id),
        }

    def handle_evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
        session_id: str = "default",
    ) -> Dict[str, Any]:
        trace_id = str(uuid.uuid4())
        state = _graph.invoke(
            {
                "mode": "evaluate",
                "session_id": session_id,
                "trace_id": trace_id,
                "file_path": None,
                "filename": None,
                "query": None,
                "eval_question": question,
                "eval_answer": answer,
                "eval_chunks": context_chunks,
                "ingestion_result": None,
                "retrieval_result": None,
                "llm_result": None,
                "eval_result": None,
                "error": None,
            }
        )
        return {
            "trace_id": trace_id,
            "scores": state.get("eval_result") or {},
            "mcp_trace": mcp_bus.get_trace(trace_id),
        }

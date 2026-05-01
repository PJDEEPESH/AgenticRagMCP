"""EvaluationAgent — RAGAS-inspired metrics evaluated by Gemini.

Computes three metrics in a single LLM call:
  - Faithfulness      : Are all answer claims grounded in the retrieved context?
  - Answer Relevancy  : Does the answer actually address the question?
  - Context Precision : Are the retrieved chunks relevant to the question?

All three are scored 0.0–1.0. An 'overall' average is also returned.
Results and requests are published to the MCP bus so they appear in the trace.

NOTE: We do NOT use the 'ragas' pip library.
      Gemini acts as the judge LLM — same metrics, no external dependency.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging
import re
from typing import Any, Dict, List

import google.generativeai as genai

from backend.config import settings
from backend.mcp.bus import mcp_bus
from backend.mcp.message import MCPMessage, MessageType

logger = logging.getLogger(__name__)

_JUDGE_MODEL = "gemini-2.5-flash"
_MAX_CHUNK_CHARS = 400
_MAX_CONTEXT_CHUNKS = 5


class EvaluationAgent:
    AGENT_NAME = "EvaluationAgent"

    def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
        trace_id: str,
    ) -> Dict[str, Any]:
        request = MCPMessage.create(
            sender="CoordinatorAgent",
            receiver=self.AGENT_NAME,
            type=MessageType.EVAL_REQUEST,
            trace_id=trace_id,
            payload={
                "question": question[:200],
                "answer_len": len(answer),
                "chunks_count": len(context_chunks),
            },
        )
        mcp_bus.publish(request)

        scores = self._score(question, answer, context_chunks)

        result = MCPMessage.create(
            sender=self.AGENT_NAME,
            receiver="CoordinatorAgent",
            type=MessageType.EVAL_RESULT,
            trace_id=trace_id,
            payload={"scores": scores},
        )
        mcp_bus.publish(result)

        return scores

    # ── Internals ─────────────────────────────────────────────────────────

    def _score(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
    ) -> Dict[str, float]:
        context_text = "\n---\n".join(
            c[:_MAX_CHUNK_CHARS] for c in context_chunks[:_MAX_CONTEXT_CHUNKS]
        )
        if not context_text.strip():
            context_text = "(no context retrieved)"

        prompt = (
            "You are a RAG quality evaluator. "
            "Given a question, retrieved context, and a generated answer, "
            "score three metrics each as a decimal from 0.0 to 1.0.\n\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Metrics:\n"
            "1. faithfulness – every claim in the answer is supported by the context\n"
            "2. answer_relevancy – the answer directly addresses the question\n"
            "3. context_precision – the retrieved context chunks are relevant to the question\n\n"
            "Output format: three numbers separated by commas, nothing else.\n"
            "Example output: 0.95,0.88,0.72"
        )

        try:
            raw = self._call_gemini(prompt)
            print(f"[EvaluationAgent] raw Gemini response: {raw!r}", flush=True)
            logger.info(f"EvaluationAgent raw response: {raw!r}")
            scores = self._parse_scores(raw)
            print(f"[EvaluationAgent] parsed scores: {scores}", flush=True)
            return scores
        except Exception as e:
            logger.error(f"EvaluationAgent scoring failed — {type(e).__name__}: {e}")
            print(f"[EvaluationAgent] ERROR: {type(e).__name__}: {e}", flush=True)
            return {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "overall": 0.5,
                "error": str(e),
            }

    def _call_gemini(self, prompt: str) -> str:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(_JUDGE_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 512,
            },
        )

        # Primary path — works when finish_reason is STOP
        try:
            text = response.text
            if text:
                return text
        except Exception as e:
            print(f"[EvaluationAgent] response.text raised: {e}", flush=True)
            logger.warning(f"response.text raised: {e}")

        # Fallback — extract from candidates directly
        try:
            if response.candidates:
                parts = response.candidates[0].content.parts
                joined = "".join(p.text for p in parts if hasattr(p, "text"))
                if joined:
                    return joined
        except Exception as e:
            print(f"[EvaluationAgent] candidates fallback raised: {e}", flush=True)
            logger.warning(f"candidates fallback raised: {e}")

        # Last resort — try prompt_feedback for blocked responses
        try:
            print(f"[EvaluationAgent] finish_reason: {response.candidates[0].finish_reason}", flush=True)
        except Exception:
            pass

        return ""

    def _parse_scores(self, raw: str) -> Dict[str, float]:
        # Accept any three numbers: integers or decimals, 0–1 range
        nums = re.findall(r"0?\.\d+|1\.0+|[01](?!\d)", raw.strip())
        if len(nums) < 3:
            # Broader fallback: grab any numeric token
            nums = re.findall(r"\d+\.?\d*", raw.strip())
        if len(nums) < 3:
            raise ValueError(f"Expected 3 scores, got {len(nums)} in: {raw!r}")

        def _clamp(v: str) -> float:
            try:
                return round(max(0.0, min(1.0, float(v))), 3)
            except (TypeError, ValueError):
                return 0.5

        f  = _clamp(nums[0])
        ar = _clamp(nums[1])
        cp = _clamp(nums[2])
        return {
            "faithfulness": f,
            "answer_relevancy": ar,
            "context_precision": cp,
            "overall": round((f + ar + cp) / 3, 3),
        }

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging
from typing import List

import google.generativeai as genai

from backend.config import settings
from backend.mcp.bus import mcp_bus
from backend.mcp.message import MCPMessage, MessageType
from backend.services.context_store import ConversationTurn
from backend.services.prompt_assembler import (
    build_followup_resolver_prompt,
    build_rag_answer_prompt,
    is_followup,
)

logger = logging.getLogger(__name__)


class LLMResponseAgent:
    AGENT_NAME = "LLMResponseAgent"
    MODEL = "gemini-2.5-flash"

    def resolve_followup(
        self,
        question: str,
        history: List[ConversationTurn],
    ) -> str:
        if not history or not is_followup(question, history):
            return question
        resolver_prompt = build_followup_resolver_prompt(question, history)
        resolved = self._call_gemini(resolver_prompt, max_tokens=200).strip()
        return resolved or question

    def generate(
        self,
        question: str,
        resolved_question: str,
        chunks: List[str],
        scores: List[float],
        trace_id: str,
        history: List[ConversationTurn],
    ) -> MCPMessage:
        request = MCPMessage.create(
            sender="CoordinatorAgent",
            receiver=self.AGENT_NAME,
            type=MessageType.LLM_REQUEST,
            trace_id=trace_id,
            payload={
                "question": question,
                "resolved_question": resolved_question,
                "chunks_count": len(chunks),
                "top_scores": scores[:3],
                "history_turns": len(history),
            },
        )
        mcp_bus.publish(request)

        full_prompt = build_rag_answer_prompt(resolved_question, chunks, history)
        answer = self._call_gemini(full_prompt, max_tokens=1500)

        response = MCPMessage.create(
            sender=self.AGENT_NAME,
            receiver="CoordinatorAgent",
            type=MessageType.LLM_RESPONSE,
            trace_id=trace_id,
            payload={
                "question": question,
                "resolved_question": resolved_question,
                "answer": answer,
                "model": self.MODEL,
                "chunks_used": len(chunks),
            },
        )
        mcp_bus.publish(response)
        return response

    def _call_gemini(self, prompt: str, max_tokens: int = 1500) -> str:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(self.MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.2,
            },
        )

        try:
            text = response.text
            if text:
                return text
        except Exception as e:
            logger.warning(f"Gemini response.text raised: {e}")

        try:
            if response.candidates:
                parts = response.candidates[0].content.parts
                return "".join(p.text for p in parts if hasattr(p, "text"))
        except Exception as e:
            logger.warning(f"Could not extract Gemini text from candidates: {e}")

        return ""

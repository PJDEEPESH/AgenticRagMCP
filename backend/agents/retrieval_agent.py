import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging
from typing import Optional

from backend.config import settings
from backend.mcp.bus import mcp_bus
from backend.mcp.message import MCPMessage, MessageType
from backend.services.neon_store import neon_store

logger = logging.getLogger(__name__)


class RetrievalAgent:
    AGENT_NAME = "RetrievalAgent"

    def retrieve(
        self, query: str, trace_id: str, top_k: Optional[int] = None
    ) -> MCPMessage:
        if top_k is None:
            top_k = settings.TOP_K_CHUNKS

        request = MCPMessage.create(
            sender="CoordinatorAgent",
            receiver=self.AGENT_NAME,
            type=MessageType.RETRIEVAL_REQUEST,
            trace_id=trace_id,
            payload={"query": query, "top_k": top_k},
        )
        mcp_bus.publish(request)

        if neon_store.total_chunks() <= 0:
            results = []
        else:
            results = neon_store.search(query, top_k=top_k)

        top_chunks = [r["chunk"] for r in results]
        scores = [round(float(r["score"]), 4) for r in results]

        result_msg = MCPMessage.create(
            sender=self.AGENT_NAME,
            receiver="LLMResponseAgent",
            type=MessageType.RETRIEVAL_RESULT,
            trace_id=trace_id,
            payload={
                "query": query,
                "top_chunks": top_chunks,
                "scores": scores,
                "chunk_count": len(results),
            },
        )
        mcp_bus.publish(result_msg)
        return result_msg

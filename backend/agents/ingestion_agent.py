import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging

from backend.mcp.bus import mcp_bus
from backend.mcp.message import MCPMessage, MessageType
from backend.services.document_parser import parse_document
from backend.services.neon_store import neon_store

logger = logging.getLogger(__name__)


class IngestionAgent:
    AGENT_NAME = "IngestionAgent"

    def ingest(self, file_path: str, filename: str, trace_id: str) -> MCPMessage:
        request = MCPMessage.create(
            sender="CoordinatorAgent",
            receiver=self.AGENT_NAME,
            type=MessageType.INGEST_REQUEST,
            trace_id=trace_id,
            payload={"file_path": file_path, "filename": filename},
        )
        mcp_bus.publish(request)

        try:
            chunks = parse_document(file_path, filename)
            inserted = neon_store.add_documents(chunks, doc_id=filename)

            ext = os.path.splitext(filename)[1].lstrip(".").upper()
            result = MCPMessage.create(
                sender=self.AGENT_NAME,
                receiver="CoordinatorAgent",
                type=MessageType.INGEST_RESULT,
                trace_id=trace_id,
                payload={
                    "filename": filename,
                    "chunks_stored": inserted,
                    "status": "success",
                    "format": ext,
                },
            )
            mcp_bus.publish(result)
            return result

        except Exception as e:
            err = MCPMessage.create(
                sender=self.AGENT_NAME,
                receiver="CoordinatorAgent",
                type=MessageType.ERROR,
                trace_id=trace_id,
                payload={"error": str(e), "filename": filename},
            )
            mcp_bus.publish(err)
            logger.exception("Ingestion failed")
            raise

"""Model Context Protocol message schema."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel


class MessageType(str, Enum):
    INGEST_REQUEST = "INGEST_REQUEST"
    INGEST_RESULT = "INGEST_RESULT"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESULT = "RETRIEVAL_RESULT"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    EVAL_REQUEST = "EVAL_REQUEST"
    EVAL_RESULT = "EVAL_RESULT"
    ERROR = "ERROR"


class MCPMessage(BaseModel):
    sender: str
    receiver: str
    type: MessageType
    trace_id: str
    timestamp: str
    payload: Dict[str, Any]

    @classmethod
    def create(
        cls,
        sender: str,
        receiver: str,
        type: MessageType,
        trace_id: str,
        payload: Dict[str, Any],
    ) -> "MCPMessage":
        return cls(
            sender=sender,
            receiver=receiver,
            type=type,
            trace_id=trace_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            payload=payload,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type.value,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

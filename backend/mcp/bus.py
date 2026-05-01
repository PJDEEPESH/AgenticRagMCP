"""In-memory MCP bus shared across all agents."""
import threading
from collections import defaultdict
from typing import Any, Dict, List

from backend.mcp.message import MCPMessage


class MCPBus:
    def __init__(self):
        self._traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(self, message: MCPMessage) -> None:
        with self._lock:
            self._traces[message.trace_id].append(message.to_dict())
        print(
            f"[MCP BUS] {message.sender} -> {message.receiver} | "
            f"{message.type.value} | trace={message.trace_id[:8]}"
        )

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._traces.get(trace_id, []))

    def get_all_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return dict(self._traces)

    def clear(self) -> None:
        with self._lock:
            self._traces = defaultdict(list)


mcp_bus = MCPBus()

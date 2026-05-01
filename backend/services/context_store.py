"""In-memory conversation history per session."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConversationTurn:
    turn_id: str
    question: str
    answer: str
    source_chunks: List[str] = field(default_factory=list)
    trace_id: str = ""
    timestamp: str = ""
    intent: str = "RAG"


class ConversationStore:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._sessions: Dict[str, List[ConversationTurn]] = defaultdict(list)

    def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        bucket = self._sessions[session_id]
        bucket.append(turn)
        if len(bucket) > self.max_turns:
            del bucket[0:len(bucket) - self.max_turns]

    def get_history(self, session_id: str) -> List[ConversationTurn]:
        return list(self._sessions.get(session_id, []))

    def get_context_window(self, session_id: str, n: int = 4) -> List[ConversationTurn]:
        history = self._sessions.get(session_id, [])
        if not history:
            return []
        return list(history[-n:])

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def clear_all(self) -> None:
        self._sessions.clear()


conversation_store = ConversationStore()

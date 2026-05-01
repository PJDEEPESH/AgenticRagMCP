"""Prompt construction for follow-up resolution and RAG answering."""
from typing import List

from backend.services.context_store import ConversationTurn


_REFERENTIAL = {
    " it ", " its ", " that ", " this ", " those ", " them ",
    " their ", " same ", " above ", "the previous",
}
_CONTINUATION = {
    "what about", "how about", "same for", "and the", "also show",
    "now show", "and what", "and how", "show me only",
}
_REASONING = {
    "why ", "why?", "explain", "summarize", "elaborate",
    "tell me more", "what does this", "interpret", "in short",
}
_QUESTION_STARTS = (
    "what", "who", "where", "when", "how", "which", "whose",
    "is", "are", "was", "were", "does", "do", "did",
    "can", "could", "should", "would", "will",
)


def is_followup(question: str, history: List[ConversationTurn]) -> bool:
    if not history:
        return False
    normalized = " " + question.lower().strip() + " "
    for phrase in _REFERENTIAL | _CONTINUATION | _REASONING:
        if phrase in normalized:
            return True
    tokens = question.lower().strip().split()
    if 0 < len(tokens) <= 4 and not tokens[0].startswith(_QUESTION_STARTS):
        return True
    return False


def build_followup_resolver_prompt(
    question: str, history: List[ConversationTurn]
) -> str:
    parts = ["CONVERSATION HISTORY", ""]
    last = history[-2:] if len(history) >= 2 else history
    for i, turn in enumerate(last, start=1):
        parts.append(f"Turn {i}:")
        parts.append(f"User: {turn.question}")
        ans = turn.answer or ""
        parts.append(f"Assistant: {ans[:200]}")
        parts.append("")

    parts.append("INSTRUCTIONS")
    parts.append(
        "Your task is to rewrite the current follow-up message into a complete "
        "standalone question. Pronouns like 'it', 'that', 'this', 'those', 'them' "
        "must be replaced with the concrete subject from the conversation history. "
        "Any filters or topics from the previous turn should be carried forward "
        "unless the user explicitly changes them. Output ONLY the rewritten "
        "question — no quotes, no explanation, no prefix."
    )
    parts.append("")
    parts.append("CURRENT USER MESSAGE")
    parts.append(question)
    parts.append("")
    parts.append("REWRITTEN QUESTION:")
    return "\n".join(parts)


def build_rag_answer_prompt(
    question: str,
    chunks: List[str],
    history: List[ConversationTurn],
) -> str:
    sections: List[str] = []

    sections.append("SYSTEM")
    sections.append(
        "You are a precise document QA assistant. Answer ONLY using the provided "
        "context chunks below. Always cite the source tag such as "
        "[source: file.pdf | page 3] when you reference any information. If the "
        "answer is not found in the context chunks respond with exactly this "
        "sentence: The provided documents do not contain information about this "
        "topic.\n"
        "\n"
        "FORMATTING RULES (very important):\n"
        "- Use Markdown. Render lists as real Markdown bullet points (one item per "
        "line, starting with '- ').\n"
        "- When the answer covers multiple distinct items, facts, or sources, "
        "ALWAYS use a bulleted list — never cram them into one paragraph.\n"
        "- Group facts by source: when citing a source, put the citation tag at "
        "the END of the bullet it belongs to, not inline mid-sentence.\n"
        "- Use a short bold lead-in (e.g. **Key metrics:**) before lists when "
        "helpful.\n"
        "- Use blank lines between paragraphs and between a heading and a list.\n"
        "- Keep it concise. Do not invent content not in the chunks."
    )
    sections.append("")

    if history:
        sections.append("CONVERSATION HISTORY (for follow-up awareness only)")
        last = history[-3:]
        for i, turn in enumerate(last, start=1):
            ans = (turn.answer or "")[:200]
            sections.append(f"{i}. User: {turn.question}")
            sections.append(f"   Assistant: {ans}")
        sections.append("")

    sections.append("DOCUMENT CONTEXT")
    if not chunks:
        sections.append(
            "No documents have been uploaded yet. Please upload a file first."
        )
    else:
        for n, chunk in enumerate(chunks, start=1):
            sections.append(f"Chunk {n}")
            sections.append(chunk)
            sections.append("")
    sections.append("")

    sections.append("USER QUESTION")
    sections.append(question)
    sections.append("")

    sections.append(
        "Answer based only on the context above. Cite the source tags. Use "
        "Markdown bullets when listing multiple items. Be concise."
    )
    sections.append("ANSWER:")
    return "\n".join(sections)

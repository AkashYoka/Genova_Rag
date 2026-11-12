"""
Database-backed conversation memory for chat sessions.

Source of truth is ChatMessage rows scoped by (user_id, session_id).
No process-level conversation buffers — history is rebuilt per request.

Also owns query-intent classification so the agent can choose between
chat-memory, document RAG, capability answers, or hybrid context without
relying solely on brittle hardcoded phrases.
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatSession

logger = logging.getLogger(__name__)

# How many recent messages to keep verbatim in the prompt (user+assistant rows).
DEFAULT_RECENT_LIMIT = 24
# Absolute max rows to scan from DB when building digests for long threads.
DEFAULT_SCAN_LIMIT = 200
# Per-message truncation inside the recent window.
DEFAULT_MAX_CHARS_PER_MSG = 1200
# Soft budget for the whole history block injected into prompts.
DEFAULT_PROMPT_CHAR_BUDGET = 14000

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "that", "this", "these", "those", "it", "its", "from",
    "by", "as", "into", "about", "what", "which", "who", "whom", "how", "why",
    "when", "where", "your", "you", "me", "my", "our", "we", "i", "please",
    "tell", "explain", "give", "show", "just", "any", "some", "all",
}


class QueryIntent(str, Enum):
    """Routing decision for a single user turn."""

    CAPABILITY = "capability"
    CHAT_MEMORY = "chat_memory"
    DOCUMENT = "document"
    HYBRID = "hybrid"
    GENERAL = "general"


# Phrase markers remain as a fast path; regex + history overlap do the real work.
_MEMORY_MARKERS = (
    "from memory of this chat",
    "from memory of this session",
    "from this chat session",
    "from our conversation",
    "from the conversation",
    "from chat history",
    "with no documents attached",
    "with no document attached",
    "no documents attached",
    "no document attached",
    "without any document",
    "without a document attached",
    "what did i ask",
    "what did i just ask",
    "what was my previous",
    "what was your previous",
    "previous answer",
    "previous question",
    "earlier in this chat",
    "earlier in our conversation",
    "what technology did i",
    "what database did i",
    "what did i mention",
    "remind me what i said",
    "summarize our conversation",
    "summarise our conversation",
    "summarize this chat",
    "summarise this chat",
    "recap our conversation",
    "based on what we discussed",
    "as i said earlier",
    "as i mentioned",
)

_MEMORY_REGEX = (
    re.compile(
        r"\b(what|which)\s+(is|was|are|were)\s+my\b",
        re.I,
    ),
    re.compile(
        r"\bwhat\s+(was|is)\s+(my|the|your)\s+"
        r"(first|previous|last|earlier|prior)\s+"
        r"(question|answer|message|response|reply)\b",
        re.I,
    ),
    re.compile(
        r"\b(summariz|summaris|recap)\w*.*\b"
        r"(conversation|chat|discussed|discussion|so far|we talked|we discussed)\b",
        re.I,
    ),
    re.compile(
        r"\bwhat\s+did\s+(i|you)\s+(ask|say|tell|answer|mention|reply)\b",
        re.I,
    ),
    re.compile(
        r"\b(remind me|from memory|in this (chat|conversation|session)|"
        r"earlier in (this|our) (chat|conversation))\b",
        re.I,
    ),
    re.compile(
        r"\b(which|what)\s+(database|technology|project|name|stack|framework)"
        r".*\b(i|we)\b.*(mention|say|said|told|use|used|chose)\b",
        re.I,
    ),
    re.compile(
        r"\b(which|what)\s+(database|technology|project|name).*\b"
        r"(did i|have i|i mention|i say|i told|i said)\b",
        re.I,
    ),
)

_FOLLOWUP_MARKERS = (
    "in more detail",
    "more detail",
    "elaborate",
    "go deeper",
    "tell me more",
    "expand on",
    "that stage",
    "the retrieval",
    "the previous",
    "as above",
    "same topic",
    "regarding that",
    "about that",
    "continue",
    "going back to",
    "as we discussed",
    "from what we",
)

_DOC_MARKERS = (
    "in the document",
    "in this document",
    "in the pdf",
    "in this pdf",
    "according to the document",
    "according to the pdf",
    "uploaded document",
    "the invoice",
    "this invoice",
    "the prescribing",
    "prescribing information",
    "the textbook",
    "the label",
    "page ",
    "cited source",
    "from the source",
    "below content",
    "provided content",
    "adverse event",
    "special situation",
    "off-label",
    "off label",
)

_DOC_REGEX = (
    re.compile(
        r"\b(invoice|pdf|document|prescribing|textbook|chunk|citation|igst|"
        r"cgst|sgst|taxable value|grand total|page\s*\d+|adverse event|"
        r"special situation|off[- ]?label|dosage|dosing|dose|anaphylaxis|"
        r"contraindication|indication)\b",
        re.I,
    ),
)


def normalize_role(role: Optional[str]) -> str:
    """Map stored roles to prompt roles: only user | assistant."""
    r = (role or "").strip().lower()
    if r in {"user", "human"}:
        return "user"
    if r in {"assistant", "ai", "bot", "model"}:
        return "assistant"
    # Never invent a third speaker in the prompt — treat unknown as assistant text.
    return "assistant"


def _tokens(text: str) -> Set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", (text or "").lower())
        if t not in _STOPWORDS
    }


def is_greeting(question: str) -> bool:
    """True iff normalized question is in the fixed greetings set."""
    greetings = {
        "hi",
        "hello",
        "hey",
        "how are you",
        "what's up",
        "whats up",
        "good morning",
        "good evening",
        "good afternoon",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "help",
        "what can you do",
        "who are you",
    }
    return (question or "").lower().strip().rstrip("!?.") in greetings


def is_capability_question(question: str) -> bool:
    """True for product/capability questions that must not drift into document RAG."""
    q = (question or "").lower().strip()
    if not q:
        return False
    if is_greeting(q):
        return True
    markers = (
        "what can you do",
        "what genova can",
        "what you can help",
        "how can you help",
        "list what you can do",
        "when no pdf",
        "when no document",
        "no pdf is attached",
        "no document is attached",
        "without relying on a specific attachment",
        "briefly say what genova",
        "who are you",
    )
    return any(m in q for m in markers)


def is_chat_memory_question(question: str) -> bool:
    """True when the user is asking about the conversation itself."""
    q = (question or "").lower().strip()
    if not q:
        return False
    if any(m in q for m in _MEMORY_MARKERS):
        return True
    return any(rx.search(q) for rx in _MEMORY_REGEX)


def is_conversation_followup(question: str) -> bool:
    """Heuristic: question likely depends on prior turns (not only memory phrases)."""
    q = (question or "").lower().strip()
    if not q:
        return False
    if is_chat_memory_question(q):
        return True
    if any(m in q for m in _FOLLOWUP_MARKERS):
        return True
    # Pronoun / anaphoric openers common in follow-ups
    if re.match(
        r"^(it|that|this|those|these|they|he|she|the same|and |also |"
        r"going back)\b",
        q,
    ):
        return True
    return False


def is_document_oriented_question(question: str) -> bool:
    """True when the question clearly targets document/PDF content."""
    q = (question or "").lower().strip()
    if not q:
        return False
    if any(m in q for m in _DOC_MARKERS):
        return True
    return any(rx.search(q) for rx in _DOC_REGEX)


# Unit-bearing numerics that make a clinical/document case self-describing.
_GROUNDED_NUMERIC_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*"
    r"(?:mg|kg|g|ml|l|%|iu/?mL|ppm|years?\s+old|yo\b|weeks?|months?|days?)\b"
)
_TABLE_LOOKUP_CUE_RE = re.compile(
    r"(?i)\b(table|chart|grid|matrix|intersection|lookup|row|column|cell|"
    r"dosing\s+table|dose\s+table|dosage\s+table)\b"
)


def is_self_contained_grounded_question(question: str) -> bool:
    """True when the turn can be answered without prior chat patient/case params.

    Domain-agnostic: document/table language, dual-axis numeric cases, or other
    multi-numeric grounded asks that are not anaphoric follow-ups.
    """
    q = (question or "").strip()
    if not q:
        return False
    if is_conversation_followup(q):
        return False
    if is_chat_memory_question(q) or is_capability_question(q):
        return False
    if is_document_oriented_question(q):
        return True
    if _TABLE_LOOKUP_CUE_RE.search(q):
        return True
    nums = _GROUNDED_NUMERIC_RE.findall(q)
    return len(nums) >= 2


def should_inject_prompt_history(
    question: str,
    *,
    has_attach: bool = False,
) -> bool:
    """Whether RAG prompts should prepend prior chat turns.

    Self-contained document/case questions must not inherit prior numeric
    patient params. Anaphoric follow-ups and chat-memory turns still get history.
    """
    q = (question or "").strip()
    if not q:
        return False
    if is_chat_memory_question(q) or is_capability_question(q):
        return True
    if is_conversation_followup(q):
        return True
    if is_self_contained_grounded_question(q):
        # Standalone grounded ask — skip history even without an attachment when
        # the question itself clearly targets document content.
        return False
    # Ambiguous short asks: keep history so anaphora / prior context can help.
    # has_attach is part of the public API for call-site clarity.
    return True


def _user_statements(chat_history: Optional[Sequence[Dict[str, Any]]]) -> List[str]:
    if not chat_history:
        return []
    out: List[str] = []
    for m in chat_history:
        if normalize_role(m.get("role")) != "user":
            continue
        content = (m.get("content") or "").strip()
        if content:
            out.append(content)
    return out


def history_likely_answers(
    question: str,
    chat_history: Optional[Sequence[Dict[str, Any]]],
) -> bool:
    """
    True when prior USER turns appear sufficient to answer the question.

    Uses first-person possessives ("my project") and lexical overlap with
    short user statements — not a full LLM classifier, but far stronger than
    phrase lists alone.
    """
    statements = _user_statements(chat_history)
    if not statements:
        return False

    q = (question or "").strip()
    q_lower = q.lower()
    q_tokens = _tokens(q)
    if not q_tokens:
        return False

    # "my X" / "our X" recall against user statements that mention X.
    possessives = re.findall(
        r"\b(?:my|our)\s+([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})",
        q_lower,
    )
    for phrase in possessives:
        head = phrase.split()[0]
        for stmt in statements:
            s_lower = stmt.lower()
            if phrase in s_lower or head in _tokens(stmt):
                # Prefer statements that look like facts / introductions.
                if _looks_like_fact(stmt) or re.search(
                    rf"\b{re.escape(head)}\b.{{0,48}}\b"
                    rf"(is|named|called|uses|use|using)\b",
                    s_lower,
                ):
                    return True

    # Significant overlap with a short declarative user fact (never prior questions).
    for stmt in statements:
        if not _looks_like_fact(stmt):
            continue
        s_tokens = _tokens(stmt)
        if not s_tokens:
            continue
        overlap = q_tokens & s_tokens
        # Proper-looking tokens (PersistenceTest, PostgreSQL) count heavily.
        strong = {
            t
            for t in overlap
            if len(t) >= 8
            or t in {"postgresql", "mysql", "mongodb", "redis"}
        }
        if strong or len(overlap) >= 2:
            return True
    return False


def classify_query_intent(
    question: str,
    chat_history: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    has_attach: bool = False,
) -> QueryIntent:
    """
    Decide how the agent should ground the answer.

    Priority:
    1. Capability / greeting — never hijacked by PDF or corpus.
    2. Explicit chat-memory / conversation-meta questions — even with PDF attached.
    3. Personal fact statements the user is teaching the chat — even with PDF attached.
    4. Document questions (attached PDF or explicit doc language) — must retrieve.
    5. Hybrid when history overlap / follow-up needs prior chat + document context.
    6. Otherwise general (may still retrieve when no attach).

    Important: lexical overlap with prior turns (``history_likely_answers``) must
    NOT force ``CHAT_MEMORY`` when a PDF is attached. That path skips retrieval
    Chat history is still injected into the RAG prompt for DOCUMENT/HYBRID turns.
    """
    q = (question or "").strip()
    if not q:
        return QueryIntent.GENERAL

    if is_capability_question(q):
        return QueryIntent.CAPABILITY

    memory_q = is_chat_memory_question(q)
    history_hit = history_likely_answers(q, chat_history)
    doc_q = is_document_oriented_question(q)
    followup = is_conversation_followup(q)
    has_history = bool(chat_history)
    personal_fact = is_personal_fact_statement(q)

    # Explicit conversation-meta / personal teaching beats attached PDFs.
    if memory_q or personal_fact:
        return QueryIntent.CHAT_MEMORY

    # History overlap with an attached PDF → keep RAG (citations).
    # Use HYBRID when the turn also looks like a follow-up / doc-grounded ask.
    if history_hit:
        if has_attach:
            if doc_q or followup:
                return QueryIntent.HYBRID
            return QueryIntent.DOCUMENT
        return QueryIntent.CHAT_MEMORY

    if has_attach:
        if followup and has_history and not doc_q:
            # Ambiguous follow-up with a PDF open — use both.
            return QueryIntent.HYBRID
        return QueryIntent.DOCUMENT

    if doc_q:
        return QueryIntent.DOCUMENT

    if followup and has_history:
        return QueryIntent.CHAT_MEMORY

    return QueryIntent.GENERAL


def resolve_session(
    db: Session,
    *,
    user_id: str,
    session_id: Optional[str] = None,
    create_if_missing: bool = True,
) -> ChatSession:
    """Resolve a session owned by the user; optionally create a default."""
    if session_id:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
        if not session:
            raise ValueError("Chat session not found")
        return session

    session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.created_at.desc())
        .first()
    )
    if session:
        return session
    if not create_if_missing:
        raise ValueError("Chat session not found")
    session = ChatSession(user_id=user_id, title="Default Chat")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def load_session_messages(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    limit: int = DEFAULT_SCAN_LIMIT,
) -> List[ChatMessage]:
    """Load messages for a session in chronological order (oldest → newest)."""
    rows = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user_id,
        )
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .limit(max(1, limit))
        .all()
    )
    return list(rows)


def messages_to_dicts(
    messages: Sequence[ChatMessage],
    *,
    include_sources: bool = False,
) -> List[Dict[str, Any]]:
    """Convert ORM messages to prompt-safe dicts."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        item: Dict[str, Any] = {
            "role": normalize_role(m.role),
            "content": (m.content or "").strip(),
            "id": str(m.id),
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        if include_sources and m.sources_json:
            try:
                parsed = json.loads(m.sources_json)
                item["sources"] = parsed
            except Exception:
                item["sources"] = None
        if item["content"]:
            out.append(item)
    return out


def _looks_like_fact(text: str) -> bool:
    """Prefer short declarative user lines when building long-thread digests."""
    t = (text or "").strip()
    if not t or len(t) > 280:
        return False
    if t.endswith("?"):
        return False
    lower = t.lower()
    if lower.startswith(
        ("what ", "how ", "why ", "when ", "where ", "who ", "explain ", "tell ")
    ):
        return False
    return True


_PERSONAL_FACT_RE = re.compile(
    r"\b("
    r"my\s+\w[\w\s-]{0,40}\b(is named|is called|named|called|is|are|was|were|uses|use|using)\b"
    r"|our\s+\w[\w\s-]{0,40}\b(is named|is called|named|called|is|are|uses|use|using)\b"
    r"|we\s+use\b|i\s+use\b|i'?m\s+using\b|we'?re\s+using\b"
    r"|i\s+am\s+testing\b|i\s+prefer\b|we\s+prefer\b"
    r")",
    re.I,
)

_FIRST_USER_TURN_RE = re.compile(
    r"\b(what|which)\s+(was|is)\s+(my|the|our)\s+"
    r"(first|earliest|initial)\s+(question|message|request|prompt|thing)\b",
    re.I,
)
_LAST_USER_TURN_RE = re.compile(
    r"\b(what|which)\s+(was|is)\s+(my|the|our)\s+"
    r"(last|previous|prior|earlier|most recent)\s+"
    r"(question|message|request|prompt)\b",
    re.I,
)


def is_personal_fact_statement(question: str) -> bool:
    """
    True for first-person declarative facts the user is teaching the chat
    (e.g. "My project is named PersistenceTest.") — not document questions.
    """
    q = (question or "").strip()
    if not q or not _looks_like_fact(q):
        return False
    if is_document_oriented_question(q):
        return False
    return bool(_PERSONAL_FACT_RE.search(q))


def resolve_deterministic_memory_answer(
    question: str,
    chat_history: Optional[Sequence[Dict[str, Any]]],
) -> Optional[str]:
    """
    Answer high-confidence conversation-meta questions without an LLM call.

    Treats the first/last *user turn* as the answer even when it is a statement
    rather than an interrogative (users often say "first question" loosely).
    """
    statements = _user_statements(chat_history)
    q = (question or "").strip()
    if not q:
        return None

    if _FIRST_USER_TURN_RE.search(q):
        if not statements:
            return (
                "There are no earlier user messages in this conversation yet."
            )
        first = statements[0]
        return f'Your first message in this conversation was: "{first}"'

    if _LAST_USER_TURN_RE.search(q):
        if not statements:
            return (
                "There are no earlier user messages in this conversation yet."
            )
        last = statements[-1]
        return f'Your previous message in this conversation was: "{last}"'

    if is_personal_fact_statement(q) and not statements:
        # Brand-new personal fact with empty history — acknowledge; do not RAG.
        return (
            "Got it — I'll remember that for this conversation. "
            "Ask me about it anytime."
        )

    return None


def build_prompt_history(
    messages: Sequence[Dict[str, Any]],
    *,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
    max_chars_per_msg: int = DEFAULT_MAX_CHARS_PER_MSG,
    char_budget: int = DEFAULT_PROMPT_CHAR_BUDGET,
) -> str:
    """
    Build a prompt history block from DB-backed messages.

    Strategy:
    - Keep the most recent ``recent_limit`` messages nearly verbatim.
    - From older user turns, keep short factual statements in a digest so
      long conversations do not lose early facts under a hard window cut.
    """
    if not messages:
        return ""

    cleaned: List[Dict[str, str]] = []
    for m in messages:
        role = normalize_role(m.get("role"))
        content = (m.get("content") or "").strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})

    if not cleaned:
        return ""

    recent = cleaned[-recent_limit:]
    older = cleaned[:-recent_limit] if len(cleaned) > recent_limit else []

    lines: List[str] = [
        "Previous conversation (context only — answer the CURRENT question; "
        "do not mix in unrelated earlier topics unless asked):"
    ]

    if older:
        facts: List[str] = []
        for m in older:
            if m["role"] != "user":
                continue
            if _looks_like_fact(m["content"]):
                snippet = m["content"]
                if len(snippet) > 220:
                    snippet = snippet[:217].rstrip() + "..."
                facts.append(f"- {snippet}")
            if len(facts) >= 40:
                break
        if facts:
            lines.append("Earlier facts the user stated in this conversation:")
            lines.extend(facts)

    used = sum(len(x) for x in lines)
    for m in recent:
        label = "User" if m["role"] == "user" else "Assistant"
        content = m["content"]
        if len(content) > max_chars_per_msg:
            content = content[: max_chars_per_msg - 3].rstrip() + "..."
        line = f"{label}: {content}"
        if used + len(line) + 1 > char_budget:
            lines.append(
                "(Older recent turns omitted to fit the context window; "
                "earlier facts above still apply.)"
            )
            break
        lines.append(line)
        used += len(line) + 1

    return "\n".join(lines)


def load_prompt_history(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
) -> tuple[List[Dict[str, Any]], str]:
    """Load session messages and return (dicts, formatted prompt history)."""
    rows = load_session_messages(
        db, user_id=user_id, session_id=session_id, limit=scan_limit
    )
    dicts = messages_to_dicts(rows, include_sources=True)
    formatted = build_prompt_history(dicts, recent_limit=recent_limit)
    return dicts, formatted


def last_assistant_sources(messages: Sequence[Dict[str, Any]]) -> list:
    """Recover citation payload from the most recent assistant turn that has sources."""
    for m in reversed(list(messages)):
        if normalize_role(m.get("role")) != "assistant":
            continue
        sources = m.get("sources")
        if not sources:
            continue
        if isinstance(sources, dict):
            chunks = (
                sources.get("chunks")
                or sources.get("sources")
                or []
            )
            return chunks if isinstance(chunks, list) else []
        if isinstance(sources, list):
            return sources
    return []


def persist_chat_message(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    document_id: Optional[str] = None,
    sources: Any = None,
    allow_empty: bool = False,
) -> Optional[ChatMessage]:
    """
    Persist one chat turn. Commits immediately so history survives later failures.

    Returns the ORM row, or None when an empty assistant turn is intentionally skipped.
    Raises on DB errors after rollback — callers must not report success if this fails.
    """
    role_n = normalize_role(role)
    text = (content or "").strip()
    if not text and not allow_empty:
        logger.info(
            "Skipping empty %s message persist (session=%s user=%s)",
            role_n,
            session_id,
            user_id,
        )
        return None

    sources_payload = None
    if sources is not None:
        sources_payload = (
            sources if isinstance(sources, str) else json.dumps(sources)
        )

    msg = ChatMessage(
        user_id=user_id,
        document_id=document_id,
        session_id=session_id,
        role=role_n,
        content=text if text else (content or ""),
        sources_json=sources_payload,
    )
    try:
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg
    except Exception:
        db.rollback()
        logger.exception(
            "Failed to persist %s message (session=%s user=%s)",
            role_n,
            session_id,
            user_id,
        )
        raise

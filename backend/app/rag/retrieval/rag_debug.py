"""Per-query RAG debug traces for retrieval / prompt investigations.

True PII spans (emails, phones, SSN-like) are redacted when
``RAG_DEBUG_REDACT=True`` (default). Quantities, units, and table
band labels are intentionally NOT redacted — that is the document
content admins need to inspect. Enable full prompt assembly via
``RAG_DEBUG_ENABLED=True``.
"""
from __future__ import annotations

import re
import threading
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Only identity / credential-like spans — never clinical label vocabulary.
_PII_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PII_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"
)
_PII_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PII_ACCOUNT_RE = re.compile(
    r"(?i)\b(?:ssn|social\s*security|account\s*(?:no|number|#)|invoice\s*(?:id|number)|"
    r"password|api[_-]?key|secret[_-]?key)\b\s*[:=]?\s*\S{3,}"
)

rag_debug_var: ContextVar[Optional["RagDebugTrace"]] = ContextVar(
    "rag_debug_trace", default=None
)
_lock = threading.Lock()


@dataclass
class RagDebugTrace:
    original_query: str = ""
    rewritten_queries: List[str] = field(default_factory=list)
    retrieval_method: str = ""
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    rerank_enabled: bool = False
    rerank_results: List[Dict[str, Any]] = field(default_factory=list)
    selected_context: str = ""
    selected_prompt_domain: str = ""
    selected_prompt_type: str = ""
    system_prompt_preview: str = ""
    assembled_user_prompt: str = ""
    final_answer_preview: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if redact:
            data["original_query"] = redact_sensitive_spans(data.get("original_query") or "")
            data["rewritten_queries"] = [
                redact_sensitive_spans(q) for q in (data.get("rewritten_queries") or [])
            ]
            data["selected_context"] = redact_sensitive_spans(
                (data.get("selected_context") or "")[:4000]
            )
            data["assembled_user_prompt"] = redact_sensitive_spans(
                (data.get("assembled_user_prompt") or "")[:6000]
            )
            for chunk in data.get("retrieved_chunks") or []:
                if isinstance(chunk, dict) and "text" in chunk:
                    chunk["text"] = redact_sensitive_spans(str(chunk.get("text") or "")[:500])
            for chunk in data.get("rerank_results") or []:
                if isinstance(chunk, dict) and "text" in chunk:
                    chunk["text"] = redact_sensitive_spans(str(chunk.get("text") or "")[:500])
        return data


def redact_sensitive_spans(text: str) -> str:
    """Redact identity/credential PII only — keep clinical document text intact.

    Older builds wiped quantity/unit/band vocabulary which destroyed
    table content in admin traces. That behavior is intentionally gone.
    """
    if not text:
        return text
    redacted = _PII_EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = _PII_PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = _PII_SSN_RE.sub("[REDACTED_SSN]", redacted)
    redacted = _PII_ACCOUNT_RE.sub("[REDACTED]", redacted)
    return redacted


# Back-compat alias for earlier call sites / notebooks.
redact_patient_spans = redact_sensitive_spans


def begin_rag_debug(query: str) -> RagDebugTrace:
    trace = RagDebugTrace(original_query=query or "")
    rag_debug_var.set(trace)
    return trace


def current_rag_debug() -> Optional[RagDebugTrace]:
    return rag_debug_var.get()


def record_rewrites(queries: List[str]) -> None:
    trace = rag_debug_var.get()
    if trace is not None:
        trace.rewritten_queries = list(queries or [])


def record_retrieval(
    *,
    method: str,
    chunks: List[Dict[str, Any]],
    rerank_enabled: bool = False,
) -> None:
    trace = rag_debug_var.get()
    if trace is None:
        return
    trace.retrieval_method = method
    trace.rerank_enabled = bool(rerank_enabled)
    slim = []
    for chunk in chunks or []:
        slim.append(
            {
                "id": chunk.get("id"),
                "filename": chunk.get("filename"),
                "page": chunk.get("page"),
                "score": chunk.get("score"),
                "confidence": chunk.get("confidence"),
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
                "chunk_type": chunk.get("chunk_type") or chunk.get("content_type"),
                "text": (chunk.get("text") or "")[:500],
            }
        )
    if rerank_enabled:
        trace.rerank_results = slim
    else:
        trace.retrieved_chunks = slim


def record_prompt_assembly(
    *,
    domain: str,
    prompt_type: str,
    system_prompt: str,
    user_prompt: str,
    context: str,
) -> None:
    trace = rag_debug_var.get()
    if trace is None:
        return
    trace.selected_prompt_domain = domain or ""
    trace.selected_prompt_type = prompt_type or ""
    trace.system_prompt_preview = (system_prompt or "")[:1500]
    trace.assembled_user_prompt = user_prompt or ""
    trace.selected_context = context or ""


def record_answer(answer: str) -> None:
    trace = rag_debug_var.get()
    if trace is not None:
        trace.final_answer_preview = (answer or "")[:2000]


def clear_rag_debug() -> None:
    rag_debug_var.set(None)

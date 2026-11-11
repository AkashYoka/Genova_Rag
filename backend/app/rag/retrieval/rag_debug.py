"""Per-query RAG debug traces for retrieval / prompt investigations.

Sensitive numeric / identity-adjacent spans are redacted when
``RAG_DEBUG_REDACT=True`` (default). Enable full prompt assembly via
``RAG_DEBUG_ENABLED=True``.
"""
from __future__ import annotations

import re
import threading
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_SENSITIVE_FIELD_RE = re.compile(
    r"(?i)\b("
    r"age|years?\s*old|body\s*weight|weight|dose|amount|total|"
    r"every\s+\d+\s+(?:weeks?|days?|hours?)|prescribed|patient|"
    r"invoice\s*(?:id|number)|account\s*(?:no|number)"
    r")\b[^.\n]{0,40}"
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
    """Best-effort redaction of common sensitive narrative spans for logs."""
    if not text:
        return text
    redacted = _SENSITIVE_FIELD_RE.sub("[REDACTED]", text)
    # Mask standalone numbers with common units in prose.
    redacted = re.sub(
        r"\b\d{2,4}(?:\.\d+)?\s*(?:kg|mg|g|ml|%|\$|IU(?:/mL)?|years?)\b",
        "[REDACTED]",
        redacted,
    )
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

"""Shared AI processing trace context for admin observability.

Tracing is automatic for chat questions when ``AI_TRACE_ENABLED`` is true.
Persistence failures are logged and never raised into the user-facing path.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
)

ai_trace_var: ContextVar[Optional["AITraceContext"]] = ContextVar(
    "ai_trace_context", default=None
)


def _settings():
    from app.config import get_settings

    return get_settings()


def _enabled() -> bool:
    try:
        return bool(getattr(_settings(), "AI_TRACE_ENABLED", True))
    except Exception:
        return True


def _capture_payloads() -> bool:
    try:
        return bool(getattr(_settings(), "AI_TRACE_CAPTURE_PAYLOADS", False))
    except Exception:
        return False


def _preview_chars() -> int:
    try:
        return int(getattr(_settings(), "AI_TRACE_CHUNK_PREVIEW_CHARS", 500) or 500)
    except Exception:
        return 500


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return str(value)[:8000]


def _safe_text(value: Any, *, limit: Optional[int] = None) -> str:
    """Truncate text for storage. Does not scrub clinical document content.

    Optional light PII masking (emails/phones/SSN) when ``AI_TRACE_REDACT_PII``
    is enabled. Never strips quantity/unit vocabulary from table text.
    """
    text = "" if value is None else str(value)
    try:
        if bool(getattr(_settings(), "AI_TRACE_REDACT_PII", False)):
            from app.rag.retrieval.rag_debug import redact_sensitive_spans

            text = redact_sensitive_spans(text)
    except Exception:
        pass
    if limit is not None and len(text) > limit:
        return text[:limit] + "…"
    return text


def normalize_trace_id(candidate: Optional[str] = None) -> str:
    """Return a valid UUID string; generate one when candidate is missing/invalid."""
    raw = (candidate or "").strip()
    if raw and _UUID_RE.match(raw):
        try:
            return str(uuid.UUID(raw.replace("-", "") if len(raw) == 32 else raw))
        except ValueError:
            pass
    return str(uuid.uuid4())


@dataclass
class _StepRecord:
    sequence: int
    name: str
    status: str = "running"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    temp_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class _RetrievalRecord:
    sequence: int
    search_query: Optional[str] = None
    search_method: Optional[str] = None
    document_ids: List[str] = field(default_factory=list)
    document_names: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None
    rejected: Optional[List[Dict[str, Any]]] = None
    selected_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result_count: int = 0
    duration_ms: Optional[float] = None
    step_temp_id: Optional[str] = None


@dataclass
class _LLMCallRecord:
    sequence: int
    model_name: Optional[str] = None
    provider: Optional[str] = None
    system_prompt_version: Optional[str] = None
    temperature: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    retry_count: int = 0
    validation_result: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    context_text: Optional[str] = None
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_temp_id: Optional[str] = None


@dataclass
class AITraceContext:
    trace_id: str
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    document_id: Optional[str] = None
    request_type: str = "rag"
    status: str = "success"
    early_exit: bool = False
    early_exit_reason: Optional[str] = None
    started_at: datetime = field(default_factory=_now)
    finished_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None
    retrieved_chunk_count: Optional[int] = None
    final_answer: Optional[str] = None
    final_answer_status: Optional[str] = None
    error_summary: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    steps: List[_StepRecord] = field(default_factory=list)
    retrievals: List[_RetrievalRecord] = field(default_factory=list)
    llm_calls: List[_LLMCallRecord] = field(default_factory=list)
    _open_steps: List[_StepRecord] = field(default_factory=list)
    _seq_step: int = 0
    _seq_retrieval: int = 0
    _seq_llm: int = 0
    _t0: float = field(default_factory=time.perf_counter)
    _persisted: bool = False


def current_ai_trace() -> Optional[AITraceContext]:
    return ai_trace_var.get()


def current_trace_id() -> Optional[str]:
    ctx = ai_trace_var.get()
    return ctx.trace_id if ctx else None


def begin_ai_trace(
    *,
    question: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
    request_type: str = "rag",
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[AITraceContext]:
    if not _enabled():
        return None
    tid = normalize_trace_id(trace_id)
    try:
        from app.observability import request_id_var

        request_id_var.set(tid)
    except Exception:
        pass
    ctx = AITraceContext(
        trace_id=tid,
        question=question or "",
        user_id=str(user_id) if user_id else None,
        session_id=str(session_id) if session_id else None,
        document_id=str(document_id) if document_id else None,
        request_type=request_type or "rag",
        metadata=dict(metadata or {}),
    )
    ai_trace_var.set(ctx)
    return ctx


def bind_message_id(message_id: Optional[str]) -> None:
    ctx = ai_trace_var.get()
    if ctx is None or not message_id:
        return
    ctx.message_id = str(message_id)


def mark_early_exit(reason: str) -> None:
    ctx = ai_trace_var.get()
    if ctx is None:
        return
    ctx.early_exit = True
    ctx.early_exit_reason = (reason or "")[:128]


def add_warning(message: str) -> None:
    ctx = ai_trace_var.get()
    if ctx is None or not message:
        return
    ctx.warnings.append(str(message)[:500])


def set_request_type(request_type: str) -> None:
    ctx = ai_trace_var.get()
    if ctx is None or not request_type:
        return
    ctx.request_type = request_type[:64]


def set_final_answer(
    answer: str,
    *,
    status: str = "success",
    sources: Optional[List[Dict[str, Any]]] = None,
    ctx: Optional[AITraceContext] = None,
) -> None:
    if ctx is None:
        ctx = ai_trace_var.get()
    if ctx is None:
        return
    ctx.final_answer = _safe_text(answer, limit=20000 if _capture_payloads() else 2000)
    ctx.final_answer_status = status
    if sources is not None:
        ctx.retrieved_chunk_count = len(sources)
        ctx.metadata["source_count"] = len(sources)


def begin_step(
    name: str,
    *,
    input_summary: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[_StepRecord]:
    """Open a pipeline step without a context manager (safe around ``yield``)."""
    ctx = ai_trace_var.get()
    if ctx is None or not _enabled():
        return None
    ctx._seq_step += 1
    step = _StepRecord(
        sequence=ctx._seq_step,
        name=name[:128],
        started_at=_now(),
        input_summary=_safe_text(input_summary, limit=2000) if input_summary is not None else None,
        metadata=dict(metadata or {}),
    )
    # stash timer on the record via metadata for end_step
    step.metadata["_t0"] = time.perf_counter()
    ctx.steps.append(step)
    ctx._open_steps.append(step)
    return step


def end_step(
    step: Optional[_StepRecord],
    *,
    status: Optional[str] = None,
    error: Optional[Any] = None,
    output_summary: Any = None,
) -> None:
    """Close a step opened by ``begin_step``."""
    if step is None:
        return
    ctx = ai_trace_var.get()
    t0 = None
    if isinstance(step.metadata, dict):
        t0 = step.metadata.pop("_t0", None)
    step.finished_at = _now()
    if t0 is not None:
        step.duration_ms = round((time.perf_counter() - float(t0)) * 1000, 3)
    if output_summary is not None:
        step.output_summary = _safe_text(output_summary, limit=4000)
    if error is not None:
        step.status = "failed"
        step.error_message = _safe_text(error, limit=4000)
        if ctx is not None:
            ctx.status = "failed"
            ctx.error_summary = step.error_message
    elif status:
        step.status = status
    elif step.status == "running":
        step.status = "success"
    if ctx is not None and ctx._open_steps:
        try:
            ctx._open_steps.remove(step)
        except ValueError:
            pass


@contextmanager
def trace_step(
    name: str,
    *,
    input_summary: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[_StepRecord]]:
    step = begin_step(name, input_summary=input_summary, metadata=metadata)
    if step is None:
        yield None
        return
    try:
        yield step
        if step.status == "running":
            end_step(step, status="success")
        else:
            end_step(step, status=step.status)
    except Exception as exc:
        end_step(step, error=exc)
        raise


def _active_step_temp_id(ctx: AITraceContext) -> Optional[str]:
    if ctx._open_steps:
        return ctx._open_steps[-1].temp_id
    return None


def _slim_chunk(chunk: Dict[str, Any], *, preview: Optional[int] = None) -> Dict[str, Any]:
    if preview is None:
        preview = _preview_chars() if _capture_payloads() else 0
    text = chunk.get("text") or chunk.get("content") or ""
    item: Dict[str, Any] = {
        "id": chunk.get("id"),
        "filename": chunk.get("filename"),
        "document_id": chunk.get("document_id"),
        "page": chunk.get("page"),
        "score": chunk.get("score"),
        "rrf_score": chunk.get("rrf_score"),
        "bm25_score": chunk.get("bm25_score"),
        "rerank_score": chunk.get("rerank_score"),
        "confidence": chunk.get("confidence"),
        "section": chunk.get("section"),
        "subsection": chunk.get("subsection"),
        "chunk_type": chunk.get("chunk_type") or chunk.get("content_type"),
        "table_id": chunk.get("table_id"),
        "char_count": len(text) if isinstance(text, str) else 0,
    }
    # Always keep a short preview so admins can see chunk shape even when
    # full payload capture is off; expand when capture is enabled.
    keep = preview if preview and preview > 0 else 240
    if text:
        item["text"] = _safe_text(text, limit=keep)
    return {k: v for k, v in item.items() if v is not None}


def bind_ai_trace(ctx: Optional[AITraceContext]) -> None:
    """Re-bind the ContextVar (needed after ``await`` on the WebSocket path)."""
    if ctx is not None:
        ai_trace_var.set(ctx)


def record_prompt_assembly(
    *,
    question: str,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    context_text: Optional[str] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    template_note: Optional[str] = None,
) -> None:
    """Record how retrieved chunks were injected into the LLM prompt."""
    ctx = ai_trace_var.get()
    if ctx is None or not _enabled():
        return
    cards = [_slim_chunk(c) for c in (chunks or []) if isinstance(c, dict)]
    payload = {
        "question": _safe_text(question, limit=2000),
        "chunk_count": len(cards),
        "chunks_sent_to_llm": cards,
        "context_char_count": len(context_text or ""),
        "user_prompt_char_count": len(user_prompt or ""),
        "system_prompt_char_count": len(system_prompt or ""),
        "assembly": template_note
        or (
            "Chunks are formatted as numbered source blocks, concatenated into "
            "`context`, then injected into the RAG user prompt via "
            "`format_task(context=..., question=...)`. Optional chat history is "
            "prepended to the user prompt."
        ),
    }
    if _capture_payloads():
        payload["context_text"] = _safe_text(context_text, limit=20000) if context_text else None
        payload["system_prompt"] = _safe_text(system_prompt, limit=8000) if system_prompt else None
        payload["user_prompt"] = _safe_text(user_prompt, limit=20000) if user_prompt else None
    else:
        payload["context_preview"] = _safe_text(context_text, limit=800) if context_text else None
        payload["user_prompt_preview"] = _safe_text(user_prompt, limit=800) if user_prompt else None
    ctx.metadata["prompt_assembly"] = payload
    # Attach to the current open step when present (usually prompt_assembly).
    if ctx._open_steps:
        step = ctx._open_steps[-1]
        try:
            step.output_summary = _safe_text(
                json.dumps(payload, default=str)[:12000],
                limit=12000,
            )
        except Exception:
            step.output_summary = _safe_text(payload, limit=12000)
        step.metadata["chunk_count"] = len(cards)


def record_retrieval(
    *,
    search_query: Optional[str] = None,
    search_method: Optional[str] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    filters: Optional[Dict[str, Any]] = None,
    rejected: Optional[List[Dict[str, Any]]] = None,
    selected_chunk_ids: Optional[List[str]] = None,
    duration_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    ctx = ai_trace_var.get()
    if ctx is None or not _enabled():
        return
    chunks = chunks or []
    ctx._seq_retrieval += 1
    doc_ids: List[str] = []
    doc_names: List[str] = []
    chunk_ids: List[str] = []
    slim: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        slim.append(_slim_chunk(chunk))
        cid = chunk.get("id")
        if cid:
            chunk_ids.append(str(cid))
        did = chunk.get("document_id")
        if did and str(did) not in doc_ids:
            doc_ids.append(str(did))
        name = chunk.get("filename")
        if name and str(name) not in doc_names:
            doc_names.append(str(name))
    selected = selected_chunk_ids or list(chunk_ids)
    ctx.retrievals.append(
        _RetrievalRecord(
            sequence=ctx._seq_retrieval,
            search_query=_safe_text(search_query, limit=2000) if search_query else None,
            search_method=(search_method or "")[:64] or None,
            document_ids=doc_ids,
            document_names=doc_names,
            chunk_ids=chunk_ids,
            chunks=slim,
            filters=filters,
            rejected=rejected,
            selected_chunk_ids=[str(x) for x in selected],
            metadata=dict(metadata or {}),
            result_count=len(slim),
            duration_ms=duration_ms,
            step_temp_id=_active_step_temp_id(ctx),
        )
    )
    ctx.retrieved_chunk_count = len(slim)


def record_llm_call(
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    system_prompt_version: Optional[str] = None,
    temperature: Optional[float] = None,
    usage: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
    retry_count: int = 0,
    validation_result: Optional[str] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    context_text: Optional[str] = None,
    raw_response: Optional[str] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    ctx = ai_trace_var.get()
    if ctx is None or not _enabled():
        return
    meta = dict(metadata or {})
    # Enrich tokens + estimated USD cost from the shared pricing registry.
    try:
        from app.pricing import enrich_usage_with_pricing

        usage = enrich_usage_with_pricing(
            usage or {},
            provider=provider,
            model_id=model_name,
        )
    except Exception:
        usage = usage or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cached_tokens = usage.get("cached_tokens") or usage.get("cache_read_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        extra = cached_tokens if isinstance(cached_tokens, int) else 0
        total_tokens = input_tokens + output_tokens + extra

    # Fallback estimate when the provider did not return usage (failed stream, etc.)
    if not isinstance(input_tokens, int) and (system_prompt or user_prompt):
        approx_in = max(1, (len(system_prompt or "") + len(user_prompt or "")) // 4)
        input_tokens = approx_in
        meta["input_tokens_estimated"] = True
    if not isinstance(output_tokens, int) and raw_response:
        approx_out = max(1, len(raw_response) // 4)
        output_tokens = approx_out
        meta["output_tokens_estimated"] = True
    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens
        try:
            from app.pricing import enrich_usage_with_pricing

            usage = enrich_usage_with_pricing(
                {
                    **usage,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                provider=provider,
                model_id=model_name,
            )
        except Exception:
            pass

    if usage.get("estimated_cost_usd") is not None:
        meta["estimated_cost_usd"] = usage.get("estimated_cost_usd")
    if usage.get("cost_type"):
        meta["cost_type"] = usage.get("cost_type")
    if usage.get("pricing_note"):
        meta["pricing_note"] = usage.get("pricing_note")

    prompt_chunks = [_slim_chunk(c) for c in (chunks or []) if isinstance(c, dict)]
    if not prompt_chunks and isinstance(ctx.metadata.get("prompt_assembly"), dict):
        prompt_chunks = list(
            ctx.metadata["prompt_assembly"].get("chunks_sent_to_llm") or []
        )
    if prompt_chunks:
        meta["chunks_sent_to_llm"] = prompt_chunks
        meta["chunk_count"] = len(prompt_chunks)

    capture = _capture_payloads()
    ctx._seq_llm += 1
    ctx.llm_calls.append(
        _LLMCallRecord(
            sequence=ctx._seq_llm,
            model_name=(model_name or "")[:128] or None,
            provider=(provider or "")[:64] or None,
            system_prompt_version=(system_prompt_version or "")[:64] or None,
            temperature=temperature,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            cached_tokens=cached_tokens if isinstance(cached_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            latency_ms=latency_ms,
            retry_count=int(retry_count or 0),
            validation_result=(validation_result or "")[:64] or None,
            status=status or "success",
            error_message=_safe_text(error_message, limit=4000) if error_message else None,
            system_prompt=_safe_text(system_prompt, limit=8000) if capture and system_prompt else None,
            user_prompt=_safe_text(user_prompt, limit=20000) if capture and user_prompt else (
                _safe_text(user_prompt, limit=800) if user_prompt else None
            ),
            context_text=_safe_text(context_text, limit=20000) if capture and context_text else (
                _safe_text(context_text, limit=800) if context_text else None
            ),
            raw_response=_safe_text(raw_response, limit=20000) if capture and raw_response else (
                _safe_text(raw_response, limit=800) if raw_response else None
            ),
            metadata=meta,
            step_temp_id=_active_step_temp_id(ctx),
        )
    )
    if model_name:
        ctx.model_name = str(model_name)[:128]
    if provider:
        ctx.provider = str(provider)[:64]
    # Aggregate tokens across calls
    def _sum(attr: str) -> Optional[int]:
        vals = [getattr(c, attr) for c in ctx.llm_calls if isinstance(getattr(c, attr), int)]
        return sum(vals) if vals else None

    ctx.input_tokens = _sum("input_tokens")
    ctx.output_tokens = _sum("output_tokens")
    ctx.cached_tokens = _sum("cached_tokens")
    ctx.total_tokens = _sum("total_tokens")
    # Aggregate estimated cost into top-level metadata
    costs = [
        c.metadata.get("estimated_cost_usd")
        for c in ctx.llm_calls
        if isinstance(c.metadata.get("estimated_cost_usd"), (int, float))
    ]
    if costs:
        ctx.metadata["estimated_cost_usd_total"] = round(sum(float(x) for x in costs), 8)
    if status == "failed":
        ctx.status = "failed"
        if error_message:
            ctx.error_summary = _safe_text(error_message, limit=4000)


def _persist_trace(ctx: AITraceContext) -> None:
    from app.database import SessionLocal
    from app.models import (
        AITrace,
        AITraceFeedback,
        AITraceLLMCall,
        AITraceRetrieval,
        AITraceStep,
    )

    db = SessionLocal()
    try:
        existing = (
            db.query(AITrace).filter(AITrace.trace_id == ctx.trace_id).first()
        )
        row = existing or AITrace(trace_id=ctx.trace_id)
        row.user_id = ctx.user_id
        row.session_id = ctx.session_id
        row.message_id = ctx.message_id
        row.document_id = ctx.document_id
        row.question = ctx.question or ""
        row.request_type = ctx.request_type or "rag"
        row.status = ctx.status or "success"
        row.early_exit = bool(ctx.early_exit)
        row.early_exit_reason = ctx.early_exit_reason
        row.started_at = ctx.started_at
        row.finished_at = ctx.finished_at or _now()
        row.duration_ms = ctx.duration_ms
        row.input_tokens = ctx.input_tokens
        row.output_tokens = ctx.output_tokens
        row.cached_tokens = ctx.cached_tokens
        row.total_tokens = ctx.total_tokens
        row.model_name = ctx.model_name
        row.provider = ctx.provider
        row.retrieved_chunk_count = ctx.retrieved_chunk_count
        row.final_answer = ctx.final_answer
        row.final_answer_status = ctx.final_answer_status
        row.error_summary = ctx.error_summary
        row.warnings_json = _json_dumps(ctx.warnings) if ctx.warnings else None
        row.metadata_json = _json_dumps(ctx.metadata) if ctx.metadata else None
        if not existing:
            row.created_at = _now()
            db.add(row)
            db.flush()
        else:
            # Replace children on re-persist
            for rel in (row.steps, row.retrievals, row.llm_calls):
                for child in list(rel):
                    db.delete(child)
            db.flush()

        step_id_by_temp: Dict[str, str] = {}
        for step in ctx.steps:
            srow = AITraceStep(
                trace_pk=row.id,
                sequence=step.sequence,
                name=step.name,
                status=step.status,
                started_at=step.started_at,
                finished_at=step.finished_at,
                duration_ms=step.duration_ms,
                input_summary=step.input_summary,
                output_summary=step.output_summary,
                error_message=step.error_message,
                metadata_json=_json_dumps(step.metadata) if step.metadata else None,
            )
            db.add(srow)
            db.flush()
            step_id_by_temp[step.temp_id] = str(srow.id)

        for ret in ctx.retrievals:
            db.add(
                AITraceRetrieval(
                    trace_pk=row.id,
                    step_id=step_id_by_temp.get(ret.step_temp_id or ""),
                    sequence=ret.sequence,
                    search_query=ret.search_query,
                    search_method=ret.search_method,
                    document_ids_json=_json_dumps(ret.document_ids),
                    document_names_json=_json_dumps(ret.document_names),
                    chunk_ids_json=_json_dumps(ret.chunk_ids),
                    chunks_json=_json_dumps(ret.chunks),
                    filters_json=_json_dumps(ret.filters),
                    rejected_json=_json_dumps(ret.rejected),
                    selected_chunk_ids_json=_json_dumps(ret.selected_chunk_ids),
                    metadata_json=_json_dumps(ret.metadata) if ret.metadata else None,
                    result_count=ret.result_count,
                    duration_ms=ret.duration_ms,
                )
            )

        for call in ctx.llm_calls:
            db.add(
                AITraceLLMCall(
                    trace_pk=row.id,
                    step_id=step_id_by_temp.get(call.step_temp_id or ""),
                    sequence=call.sequence,
                    model_name=call.model_name,
                    provider=call.provider,
                    system_prompt_version=call.system_prompt_version,
                    temperature=call.temperature,
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    cached_tokens=call.cached_tokens,
                    total_tokens=call.total_tokens,
                    latency_ms=call.latency_ms,
                    retry_count=call.retry_count,
                    validation_result=call.validation_result,
                    status=call.status,
                    error_message=call.error_message,
                    system_prompt=call.system_prompt,
                    user_prompt=call.user_prompt,
                    context_text=call.context_text,
                    raw_response=call.raw_response,
                    metadata_json=_json_dumps(call.metadata) if call.metadata else None,
                )
            )

        db.commit()
        ctx._persisted = True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def finish_ai_trace(
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    message_id: Optional[str] = None,
    ctx: Optional[AITraceContext] = None,
) -> Optional[str]:
    """Finalize and soft-fail persist the current trace. Returns trace_id.

    Pass ``ctx`` explicitly for SSE/stream paths where middleware may have
    cleared the ContextVar before the generator finishes.
    """
    if ctx is None:
        ctx = ai_trace_var.get()
    if ctx is None:
        return None
    try:
        # Re-bind so nested record_* helpers still see this trace if needed.
        ai_trace_var.set(ctx)
        if message_id:
            ctx.message_id = str(message_id)
        if status:
            ctx.status = status
        elif ctx.status == "running":
            ctx.status = "success"
        if error:
            ctx.error_summary = _safe_text(error, limit=4000)
            if ctx.status == "success":
                ctx.status = "failed"
        # Close any still-open steps
        for step in list(ctx._open_steps):
            if step.status == "running":
                step.status = "failed" if error else "partial"
                step.finished_at = _now()
                if error and not step.error_message:
                    step.error_message = _safe_text(error, limit=4000)
        ctx._open_steps.clear()
        ctx.finished_at = _now()
        ctx.duration_ms = round((time.perf_counter() - ctx._t0) * 1000, 3)

        # Mirror table_rag / rag_debug extras into metadata when present
        try:
            from app.rag.table_rag.telemetry import current_table_rag_trace

            tr = current_table_rag_trace()
            if tr is not None:
                ctx.metadata["table_rag"] = tr.to_dict()
        except Exception:
            pass
        try:
            from app.rag.retrieval.rag_debug import current_rag_debug

            dbg = current_rag_debug()
            if dbg is not None:
                ctx.metadata["rag_debug"] = dbg.to_dict(redact=True)
        except Exception:
            pass

        if _enabled():
            try:
                _persist_trace(ctx)
            except Exception as exc:
                logger.warning(
                    "AI trace persistence failed for %s: %s",
                    ctx.trace_id,
                    exc,
                    exc_info=True,
                )
        return ctx.trace_id
    except Exception as exc:
        logger.warning("finish_ai_trace failed: %s", exc, exc_info=True)
        return getattr(ctx, "trace_id", None)
    finally:
        try:
            ai_trace_var.set(None)
        except Exception:
            pass


def link_feedback_to_trace(
    *,
    message_id: str,
    feedback_id: str,
    feedback: str,
    reason: Optional[str] = None,
    comment: Optional[str] = None,
) -> None:
    """Attach feedback to an existing persisted trace for the message (soft-fail)."""
    if not _enabled() or not message_id:
        return
    try:
        from app.database import SessionLocal
        from app.models import AITrace, AITraceFeedback

        db = SessionLocal()
        try:
            trace = (
                db.query(AITrace)
                .filter(AITrace.message_id == str(message_id))
                .order_by(AITrace.created_at.desc())
                .first()
            )
            if trace is None:
                return
            existing = (
                db.query(AITraceFeedback)
                .filter(AITraceFeedback.feedback_id == str(feedback_id))
                .first()
            )
            if existing:
                existing.feedback = feedback
                existing.reason = reason
                existing.comment = comment
            else:
                db.add(
                    AITraceFeedback(
                        trace_pk=trace.id,
                        feedback_id=str(feedback_id),
                        message_id=str(message_id),
                        feedback=feedback,
                        reason=reason,
                        comment=comment,
                        created_at=_now(),
                    )
                )
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning("link_feedback_to_trace failed: %s", exc)


def cleanup_expired_traces(*, retention_days: Optional[int] = None) -> int:
    """Delete traces older than retention. Returns deleted count. Soft-fail."""
    try:
        days = retention_days
        if days is None:
            days = int(getattr(_settings(), "AI_TRACE_RETENTION_DAYS", 30) or 30)
        if days <= 0:
            return 0
        cutoff = _now() - timedelta(days=days)
        from app.database import SessionLocal
        from app.models import AITrace

        db = SessionLocal()
        try:
            q = db.query(AITrace).filter(AITrace.created_at < cutoff)
            count = q.count()
            q.delete(synchronize_session=False)
            db.commit()
            return int(count)
        finally:
            db.close()
    except Exception as exc:
        logger.warning("cleanup_expired_traces failed: %s", exc)
        return 0


def clear_ai_trace() -> None:
    ai_trace_var.set(None)

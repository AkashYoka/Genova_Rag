"""Structured TableRAG telemetry (contextvar + JSON log event)."""
from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

table_rag_trace_var: ContextVar[Optional["TableRagTrace"]] = ContextVar(
    "table_rag_trace", default=None
)


@dataclass
class TableRagTrace:
    document_id: str = ""
    index_version: str = ""
    rewrite_triggered: bool = False
    top1_rrf_score: float = 0.0
    candidate_table_ids: List[str] = field(default_factory=list)
    selector_status: str = ""
    selector_latency_ms: float = 0.0
    legacy_fallback_reason: Optional[str] = None
    reconcile_action: str = "none"
    question_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def begin_table_rag_trace(*, question: str = "", document_id: str = "") -> TableRagTrace:
    trace = TableRagTrace(
        document_id=str(document_id or ""),
        question_preview=(question or "")[:160],
    )
    table_rag_trace_var.set(trace)
    return trace


def current_table_rag_trace() -> Optional[TableRagTrace]:
    return table_rag_trace_var.get()


def record_rewrite_gate(*, top1_rrf_score: float, rewrite_triggered: bool) -> None:
    trace = table_rag_trace_var.get()
    if trace is None:
        return
    trace.top1_rrf_score = float(top1_rrf_score or 0.0)
    trace.rewrite_triggered = bool(rewrite_triggered)


def record_selector_metrics(
    *,
    status: str,
    latency_ms: float = 0.0,
    candidate_table_ids: Optional[List[str]] = None,
    legacy_fallback_reason: Optional[str] = None,
    index_version: str = "",
    document_id: str = "",
) -> None:
    trace = table_rag_trace_var.get()
    if trace is None:
        # Always keep a trace for table-eligible paths even if agent forgot begin().
        trace = begin_table_rag_trace(document_id=document_id)
    if document_id and not trace.document_id:
        trace.document_id = document_id
    if index_version:
        trace.index_version = index_version
    trace.selector_status = str(status or "")
    trace.selector_latency_ms = float(latency_ms or 0.0)
    if candidate_table_ids is not None:
        trace.candidate_table_ids = [str(t) for t in candidate_table_ids if t]
    if legacy_fallback_reason:
        trace.legacy_fallback_reason = legacy_fallback_reason
        logger.warning(
            "TableRAG legacy_fallback reason=%s status=%s",
            legacy_fallback_reason,
            status,
        )


def record_reconcile_action(action: str) -> None:
    trace = table_rag_trace_var.get()
    if trace is None:
        return
    trace.reconcile_action = str(action or "none")


def emit_table_rag_trace() -> Optional[Dict[str, Any]]:
    """Emit a single structured JSON log event and clear the contextvar."""
    trace = table_rag_trace_var.get()
    if trace is None:
        return None
    payload = {"event": "table_rag_trace", **trace.to_dict()}
    try:
        logger.info("%s", json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:
        logger.warning("table_rag_trace emit failed: %s", exc)
    table_rag_trace_var.set(None)
    return payload


def clear_table_rag_trace() -> None:
    table_rag_trace_var.set(None)


class TimerMs:
    """Simple wall-clock timer in milliseconds."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 2)

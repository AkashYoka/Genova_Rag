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
    answer_state: str = ""
    question_preview: str = ""
    # Decision audit for one structured lookup (inspectable, not user-facing).
    retrieved_table_ids: List[str] = field(default_factory=list)
    eligibility_decisions: List[Dict[str, Any]] = field(default_factory=list)
    rejection_reasons: Dict[str, str] = field(default_factory=dict)
    applicability_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    numeric_match: Dict[str, Any] = field(default_factory=dict)
    cell_verification: Dict[str, Any] = field(default_factory=dict)
    remaining_conflicts: List[str] = field(default_factory=list)
    # Iterative evidence retrieval audit.
    iteration_count: int = 0
    retrieval_rounds: List[Dict[str, Any]] = field(default_factory=list)

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


def record_candidate_audit(
    *,
    retrieved_table_ids: Optional[List[str]] = None,
    eligibility: Optional[List[Dict[str, Any]]] = None,
    applicability_comparisons: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Record retrieved candidates, eligibility decisions and conflicts."""
    trace = table_rag_trace_var.get()
    if trace is None:
        return
    if retrieved_table_ids is not None:
        trace.retrieved_table_ids = [
            str(t) for t in retrieved_table_ids if str(t or "").strip()
        ]
    if eligibility is not None:
        trace.eligibility_decisions = [dict(c) for c in eligibility if c]
        reasons: Dict[str, str] = {}
        conflicts: List[str] = []
        for item in trace.eligibility_decisions:
            tid = str(item.get("table_id") or "")
            state = str(item.get("state") or "")
            if tid and state and state != "eligible":
                reasons[tid] = ",".join(item.get("reasons") or []) or state
            for other in item.get("conflicts_with") or []:
                pair = ",".join(sorted([tid, str(other)]))
                if pair and pair not in conflicts:
                    conflicts.append(pair)
        trace.rejection_reasons = reasons
        trace.remaining_conflicts = conflicts
    if applicability_comparisons is not None:
        trace.applicability_comparisons = [
            dict(c) for c in applicability_comparisons if c
        ]


def record_cell_verification(
    *,
    table_id: str = "",
    row_index: Optional[int] = None,
    col_index: Optional[int] = None,
    row_headers: Optional[List[str]] = None,
    col_headers: Optional[List[str]] = None,
    status: str = "",
    reason: str = "",
) -> None:
    trace = table_rag_trace_var.get()
    if trace is None:
        return
    trace.numeric_match = {
        "table_id": table_id,
        "row_index": row_index,
        "col_index": col_index,
        "row_headers": list(row_headers or []),
        "col_headers": list(col_headers or []),
    }
    trace.cell_verification = {"status": status, "reason": reason}


def record_answer_state(state: str) -> None:
    trace = table_rag_trace_var.get()
    if trace is None:
        return
    trace.answer_state = str(state or "")


def record_reconcile_action(action: str) -> None:
    trace = table_rag_trace_var.get()
    if trace is None:
        return
    trace.reconcile_action = str(action or "none")


def record_iteration(
    *,
    round_i: int,
    queries: Optional[List[str]] = None,
    new_chunk_count: int = 0,
    outcome: str = "",
) -> None:
    """Record one iterative evidence-retrieval round."""
    trace = table_rag_trace_var.get()
    if trace is None:
        return
    trace.iteration_count = max(int(trace.iteration_count or 0), int(round_i or 0))
    trace.retrieval_rounds.append(
        {
            "round": int(round_i or 0),
            "queries": [str(q) for q in (queries or []) if q],
            "new_chunk_count": int(new_chunk_count or 0),
            "outcome": str(outcome or ""),
        }
    )


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

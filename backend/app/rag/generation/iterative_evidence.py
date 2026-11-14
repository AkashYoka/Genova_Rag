"""Iterative evidence retrieval for table-aware RAG.

Extends the single-shot ``interpret_table_question`` path: when the analyst
reports missing/ambiguous evidence and proposes follow-up queries, run another
``retrieve()`` round, merge new chunks (deduped), and re-evaluate.

Domain-agnostic. No medical/finance/engineering selection rules.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.rag.generation.llm_table_reasoning import (
    TableInterpretationResult,
    interpret_table_question,
)
from app.rag.generation.table_evidence import InsufficientEvidence, VerifiedCell

logger = logging.getLogger(__name__)


def _chunk_key(chunk: Dict[str, Any]) -> str:
    for key in ("id", "chunk_id"):
        val = chunk.get(key)
        if val:
            return str(val)
    doc = str(chunk.get("document_id") or "")
    idx = chunk.get("chunk_index")
    if doc and idx is not None:
        return f"{doc}:{idx}"
    tid = str(chunk.get("table_id") or "")
    page = chunk.get("page") or chunk.get("page_start") or ""
    text = str(chunk.get("text") or "")[:80]
    return f"{doc}:{tid}:{page}:{hash(text)}"


def union_chunks(
    existing: Sequence[Dict[str, Any]],
    incoming: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge chunks preserving order; prefer fuller table grids on collision."""
    by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for chunk in list(existing or []) + list(incoming or []):
        if not isinstance(chunk, dict):
            continue
        key = _chunk_key(chunk)
        if key not in by_key:
            by_key[key] = dict(chunk)
            order.append(key)
            continue
        prev = by_key[key]
        # Prefer richer table payload when both reference the same table.
        prev_rows = len((prev.get("table") or {}).get("rows") or [])
        new_rows = len((chunk.get("table") or {}).get("rows") or [])
        if new_rows > prev_rows or (
            float(chunk.get("score") or 0) > float(prev.get("score") or 0)
            and new_rows >= prev_rows
        ):
            merged = dict(prev)
            merged.update(chunk)
            by_key[key] = merged
    return [by_key[k] for k in order]


def _followup_queries(
    question: str,
    result: TableInterpretationResult,
    *,
    max_queries: int = 3,
) -> List[str]:
    """Build targeted retrieval queries from the analyst payload."""
    raw = result.raw_llm if isinstance(result.raw_llm, dict) else {}
    queries: List[str] = []
    for item in raw.get("additional_search_queries") or []:
        text = str(item or "").strip()
        if text and text.lower() not in {q.lower() for q in queries}:
            queries.append(text)
    missing = []
    if result.insufficient and result.insufficient.missing_dimensions:
        missing.extend(
            str(x).strip()
            for x in result.insufficient.missing_dimensions
            if str(x or "").strip()
        )
    for item in raw.get("missing_information") or []:
        text = str(item or "").strip()
        if text and text not in missing:
            missing.append(text)
    if missing and not queries:
        # Domain-agnostic synthesis: question + missing fact labels.
        focus = " ".join(missing[:4])
        queries.append(f"{(question or '').strip()} {focus}".strip())
        for dim in missing[:max_queries]:
            q = f"{dim} {(question or '').strip()}".strip()
            if q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
    # Always keep the original question as a last resort variant if empty.
    if not queries and (question or "").strip():
        queries.append(question.strip())
    return queries[: max(1, int(max_queries or 3))]


def _should_iterate(result: TableInterpretationResult) -> bool:
    if result.verified is not None:
        return False
    if not result.is_structured_lookup:
        return False
    insuff = result.insufficient
    if insuff is None:
        return False
    raw = result.raw_llm if isinstance(result.raw_llm, dict) else {}
    summary = str(
        raw.get("reasoning_summary") or insuff.reason or ""
    ).strip().lower()
    # Never thrash on selector/transport failures — those need a tighter
    # prompt retry inside reason_over_tables, not another hybrid retrieve.
    if "selector_parse_failed" in summary or "selector_error" in summary:
        return False
    if "selector_unavailable" in summary:
        return False
    has_followups = bool(
        [
            str(x).strip()
            for x in (raw.get("additional_search_queries") or [])
            if str(x or "").strip()
        ]
    )
    has_missing = bool(
        (insuff.missing_dimensions or [])
        or [
            str(x).strip()
            for x in (raw.get("missing_information") or [])
            if str(x or "").strip()
        ]
    )
    # Only iterate when the analyst explicitly signals missing evidence
    # and/or proposes follow-up searches. Blind not_found loops pollute
    # context with sibling tables and hurt accuracy.
    return bool(has_followups or has_missing)


def run_iterative_evidence_loop(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    max_iterations: Optional[int] = None,
    top_k_per_round: Optional[int] = None,
) -> Tuple[TableInterpretationResult, List[Dict[str, Any]], Dict[str, Any]]:
    """Run interpret → (optional) targeted retrieve → re-interpret loops.

    Returns ``(interpretation, merged_chunks, stats)``.
    """
    settings = get_settings()
    enabled = bool(getattr(settings, "TABLE_RAG_ITERATIVE_ENABLED", True))
    max_iters = int(
        max_iterations
        if max_iterations is not None
        else getattr(settings, "TABLE_RAG_MAX_ITERATIONS", 2)
        or 2
    )
    top_k = int(
        top_k_per_round
        if top_k_per_round is not None
        else getattr(settings, "TABLE_RAG_ITERATIVE_TOP_K", 6)
        or 6
    )
    max_iters = max(0, min(max_iters, 4))
    top_k = max(2, min(top_k, 12))

    merged = union_chunks([], list(chunks or []))
    t0 = time.perf_counter()
    result = interpret_table_question(question, merged)
    stats: Dict[str, Any] = {
        "iterative_enabled": enabled,
        "iterations": 0,
        "rounds": [
            {
                "round": 0,
                "chunk_count": len(merged),
                "verified": bool(result.verified),
                "status": (
                    result.verified.status
                    if isinstance(result.verified, VerifiedCell)
                    else (
                        result.insufficient.status
                        if isinstance(result.insufficient, InsufficientEvidence)
                        else ""
                    )
                ),
                "missing": list(
                    (result.insufficient.missing_dimensions or [])
                    if result.insufficient
                    else []
                ),
            }
        ],
        "queries_used": [],
        "new_chunks_total": 0,
        "elapsed_s": 0.0,
    }

    if not enabled or max_iters <= 0:
        stats["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return result, merged, stats

    seen_queries: set[str] = set()
    for round_i in range(1, max_iters + 1):
        if not _should_iterate(result):
            break
        queries = _followup_queries(question, result)
        queries = [q for q in queries if q.lower() not in seen_queries]
        if not queries:
            break
        for q in queries:
            seen_queries.add(q.lower())

        from app.rag.retrieval.retriever import retrieve

        new_chunks: List[Dict[str, Any]] = []
        for q in queries:
            try:
                hits = retrieve(
                    query=q,
                    user_id=user_id,
                    document_id=document_id,
                    document_ids=document_ids,
                    top_k=top_k,
                )
            except Exception as exc:
                logger.warning("Iterative retrieve failed for %r: %s", q[:80], exc)
                hits = []
            new_chunks.extend(hits or [])
            stats["queries_used"].append(q)

        before = len(merged)
        merged = union_chunks(merged, new_chunks)
        added = len(merged) - before
        stats["new_chunks_total"] += max(0, added)
        stats["iterations"] = round_i

        if added <= 0:
            stats["rounds"].append(
                {
                    "round": round_i,
                    "chunk_count": len(merged),
                    "added": 0,
                    "queries": list(queries),
                    "stopped": "no_new_chunks",
                }
            )
            break

        result = interpret_table_question(question, merged)
        stats["rounds"].append(
            {
                "round": round_i,
                "chunk_count": len(merged),
                "added": added,
                "queries": list(queries),
                "verified": bool(result.verified),
                "status": (
                    result.verified.status
                    if isinstance(result.verified, VerifiedCell)
                    else (
                        result.insufficient.status
                        if isinstance(result.insufficient, InsufficientEvidence)
                        else ""
                    )
                ),
                "missing": list(
                    (result.insufficient.missing_dimensions or [])
                    if result.insufficient
                    else []
                ),
            }
        )
        try:
            from app.rag.table_rag.telemetry import record_iteration

            record_iteration(
                round_i=round_i,
                queries=queries,
                new_chunk_count=added,
                outcome=(
                    "verified"
                    if result.verified
                    else (
                        result.insufficient.status
                        if result.insufficient
                        else "unknown"
                    )
                ),
            )
        except Exception:
            pass

    stats["elapsed_s"] = round(time.perf_counter() - t0, 3)
    if isinstance(result.raw_llm, dict):
        result.raw_llm = dict(result.raw_llm)
        result.raw_llm["_iterative_stats"] = stats
    return result, merged, stats

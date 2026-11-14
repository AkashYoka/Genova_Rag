"""Iterative / agentic evidence retrieval for table-aware and text+table RAG.

Extends the single-shot ``interpret_table_question`` path:

1. Structured table lookups may run targeted retrieve → re-interpret rounds when
   the analyst reports missing/ambiguous evidence.
2. When the path is NOT a structured table lookup, OR the table path is only
   soft-insufficient, a lightweight TEXT+TABLE evidence assessment may propose
   additional queries and expand the chunk set under the same iteration budget.

Domain-agnostic. No medical/finance/engineering selection rules. Assessment
stores structured sufficiency fields only — no private chain-of-thought.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.rag.generation.llm_table_reasoning import (
    TableInterpretationResult,
    interpret_table_question,
)
from app.rag.generation.table_evidence import InsufficientEvidence, VerifiedCell

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{[\s\S]*\}")

_EVIDENCE_ASSESS_SYSTEM = """You assess whether retrieved document evidence is sufficient to answer a user question.
Use ONLY the supplied evidence snippets (text and/or tables). Never invent facts.
Return JSON only (no markdown fences) with this schema:
{
  "sufficient": true,
  "evidence_ids": ["chunk ids that support an answer"],
  "doc_ids": ["document ids"],
  "pages": [1, 2],
  "text_refs": ["short text provenance notes"],
  "table_refs": ["table ids or captions"],
  "missing_info": ["concrete missing facts, if any"],
  "conflicts": ["short conflict notes, if any"],
  "additional_queries": ["short retrieval queries for missing document facts"],
  "can_answer_safely": true,
  "rationale": "one short sentence, no private chain-of-thought"
}
Rules:
- sufficient=true only when the snippets contain the facts needed for a grounded answer.
- can_answer_safely=false when evidence is thin, conflicting, or off-scope.
- Propose additional_queries only for missing DOCUMENT facts (not world knowledge).
- Prefer empty additional_queries when sufficient and can_answer_safely are true.
"""


@dataclass
class EvidenceAssessment:
    """Structured TEXT+TABLE sufficiency assessment (no private CoT)."""

    sufficient: bool = False
    evidence_ids: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)
    pages: List[Any] = field(default_factory=list)
    text_refs: List[str] = field(default_factory=list)
    table_refs: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    additional_queries: List[str] = field(default_factory=list)
    can_answer_safely: bool = False
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def apply_context_budget(
    chunks: Sequence[Dict[str, Any]],
    *,
    max_chunks: int = 12,
    max_chars: int = 24000,
) -> List[Dict[str, Any]]:
    """Trim merged evidence to a bounded context budget (count + characters)."""
    capped = list(chunks or [])[: max(1, int(max_chunks or 12))]
    out: List[Dict[str, Any]] = []
    used = 0
    for chunk in capped:
        text = str(chunk.get("text") or "")
        table = chunk.get("table") or {}
        extra = 0
        if isinstance(table, dict):
            extra = len(json.dumps(table.get("headers") or [])) + len(
                json.dumps((table.get("rows") or [])[:8])
            )
        cost = len(text) + extra
        if out and used + cost > max_chars:
            break
        out.append(chunk)
        used += cost
    return out or capped[:1]


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


def _is_soft_insufficient(result: TableInterpretationResult) -> bool:
    insuff = result.insufficient
    if insuff is None:
        return False
    if bool(insuff.soft):
        return True
    raw = result.raw_llm if isinstance(result.raw_llm, dict) else {}
    return bool(raw.get("soft_insufficient"))


def _should_run_agentic_assessment(result: TableInterpretationResult) -> bool:
    """Run TEXT+TABLE assessment when not a hard table verify path."""
    if result.verified is not None:
        return False
    if not result.is_structured_lookup:
        return True
    return _is_soft_insufficient(result)


def _parse_assessment_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1)
    else:
        match = _JSON_RE.search(text)
        if match:
            text = match.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _chunk_preview(chunks: Sequence[Dict[str, Any]], *, limit: int = 10) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for chunk in list(chunks or [])[:limit]:
        table = chunk.get("table") if isinstance(chunk.get("table"), dict) else {}
        preview.append(
            {
                "id": str(chunk.get("id") or chunk.get("chunk_id") or ""),
                "document_id": str(chunk.get("document_id") or ""),
                "page": chunk.get("page") or chunk.get("page_start"),
                "filename": chunk.get("filename") or chunk.get("original_name") or "",
                "table_id": str(
                    chunk.get("table_id")
                    or (table.get("table_id") if isinstance(table, dict) else "")
                    or ""
                ),
                "caption": str(
                    (table.get("caption") if isinstance(table, dict) else "")
                    or chunk.get("caption")
                    or ""
                )[:160],
                "text": str(chunk.get("text") or "")[:500],
                "has_table": bool(table),
            }
        )
    return preview


def _heuristic_assessment(
    question: str,
    chunks: Sequence[Dict[str, Any]],
) -> EvidenceAssessment:
    """Offline-safe fallback when the assessment LLM is unavailable."""
    ids = [
        str(c.get("id") or c.get("chunk_id") or "")
        for c in (chunks or [])
        if c.get("id") or c.get("chunk_id")
    ]
    doc_ids = sorted(
        {
            str(c.get("document_id"))
            for c in (chunks or [])
            if c.get("document_id")
        }
    )
    pages = sorted(
        {
            c.get("page") or c.get("page_start")
            for c in (chunks or [])
            if (c.get("page") or c.get("page_start")) is not None
        }
    )
    table_refs = [
        str(c.get("table_id") or "")
        for c in (chunks or [])
        if c.get("table_id") or (isinstance(c.get("table"), dict) and c.get("table"))
    ]
    text_blob = " ".join(str(c.get("text") or "") for c in (chunks or [])).lower()
    q_tokens = [t for t in re.findall(r"[a-z0-9]{3,}", (question or "").lower()) if t]
    overlap = sum(1 for t in q_tokens[:12] if t in text_blob) if q_tokens else 0
    sufficient = bool(chunks) and overlap >= max(2, min(4, len(q_tokens) // 3 or 1))
    missing: List[str] = []
    queries: List[str] = []
    if not sufficient and (question or "").strip():
        missing.append("additional supporting passages for the question")
        queries.append(question.strip())
    return EvidenceAssessment(
        sufficient=sufficient,
        evidence_ids=[i for i in ids if i][:12],
        doc_ids=doc_ids,
        pages=list(pages)[:12],
        text_refs=[str(c.get("filename") or "") for c in (chunks or [])[:4]],
        table_refs=[t for t in table_refs if t][:8],
        missing_info=missing,
        conflicts=[],
        additional_queries=queries[:3],
        can_answer_safely=sufficient,
        rationale="heuristic overlap assessment",
    )


def assess_evidence_sufficiency(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    client: Any = None,
    llm_provider: Optional[str] = None,
) -> EvidenceAssessment:
    """LLM (preferred) or heuristic assessment of TEXT+TABLE evidence sufficiency."""
    if not chunks:
        return EvidenceAssessment(
            sufficient=False,
            missing_info=["no retrieved evidence"],
            additional_queries=[question.strip()] if (question or "").strip() else [],
            can_answer_safely=False,
            rationale="empty evidence set",
        )
    user_payload = {
        "question": question,
        "evidence": _chunk_preview(chunks),
    }
    try:
        if client is None:
            from app.rag.llm.agent import (
                get_llm_client,
                llm_model_for_provider,
                normalize_llm_provider,
            )

            provider = normalize_llm_provider(llm_provider)
            client = get_llm_client(llm_provider=provider)
            model = llm_model_for_provider(provider)
        else:
            model = None
        kwargs: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _EVIDENCE_ASSESS_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        if model:
            kwargs["model"] = model
        response = client.chat_completion(**kwargs)
        raw = ""
        if response and getattr(response, "choices", None):
            raw = (response.choices[0].message.content or "").strip()
        data = _parse_assessment_json(raw)
        if not data:
            return _heuristic_assessment(question, chunks)

        def _str_list(key: str) -> List[str]:
            vals = data.get(key) or []
            if not isinstance(vals, list):
                return []
            out: List[str] = []
            for item in vals:
                text = str(item or "").strip()
                if text and text not in out:
                    out.append(text)
            return out

        pages_raw = data.get("pages") or []
        pages: List[Any] = []
        if isinstance(pages_raw, list):
            for p in pages_raw:
                if p is None or p == "":
                    continue
                pages.append(p)

        sufficient = bool(data.get("sufficient"))
        can_safe = bool(data.get("can_answer_safely"))
        return EvidenceAssessment(
            sufficient=sufficient,
            evidence_ids=_str_list("evidence_ids"),
            doc_ids=_str_list("doc_ids"),
            pages=pages,
            text_refs=_str_list("text_refs"),
            table_refs=_str_list("table_refs"),
            missing_info=_str_list("missing_info"),
            conflicts=_str_list("conflicts"),
            additional_queries=_str_list("additional_queries")[:3],
            can_answer_safely=can_safe,
            rationale=str(data.get("rationale") or "")[:300],
        )
    except Exception as exc:
        logger.warning("Evidence assessment LLM failed: %s", exc)
        return _heuristic_assessment(question, chunks)


def _assessment_followup_queries(
    question: str,
    assessment: EvidenceAssessment,
    *,
    max_queries: int = 3,
) -> List[str]:
    queries: List[str] = []
    for item in assessment.additional_queries or []:
        text = str(item or "").strip()
        if text and text.lower() not in {q.lower() for q in queries}:
            queries.append(text)
    if not queries and assessment.missing_info:
        focus = " ".join(str(x).strip() for x in assessment.missing_info[:3] if str(x).strip())
        if focus:
            queries.append(f"{(question or '').strip()} {focus}".strip())
    return queries[: max(1, int(max_queries or 3))]


def _should_iterate_assessment(assessment: EvidenceAssessment) -> bool:
    if assessment.sufficient and assessment.can_answer_safely:
        return False
    return bool(assessment.additional_queries or assessment.missing_info)


def _retrieve_for_queries(
    queries: Sequence[str],
    *,
    user_id: str,
    document_id: Optional[str],
    document_ids: Optional[List[str]],
    top_k: int,
    stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
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
    return new_chunks


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

    After the table path, may run a lightweight TEXT+TABLE evidence assessment
    loop when the question is not a structured lookup or the table path is only
    soft-insufficient.

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
    budget_chunks = max(
        top_k,
        int(getattr(settings, "TOP_K_RERANK", 8) or 8) + top_k,
    )
    budget_chars = int(getattr(settings, "TABLE_RAG_CONTEXT_CHAR_BUDGET", 24000) or 24000)

    merged = apply_context_budget(
        union_chunks([], list(chunks or [])),
        max_chunks=budget_chunks,
        max_chars=budget_chars,
    )
    t0 = time.perf_counter()
    result = interpret_table_question(question, merged)
    stats: Dict[str, Any] = {
        "iterative_enabled": enabled,
        "iterations": 0,
        "agentic_iterations": 0,
        "rounds": [
            {
                "round": 0,
                "phase": "table",
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
        "evidence_assessment": None,
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

        new_chunks = _retrieve_for_queries(
            queries,
            user_id=user_id,
            document_id=document_id,
            document_ids=document_ids,
            top_k=top_k,
            stats=stats,
        )

        before = len(merged)
        merged = apply_context_budget(
            union_chunks(merged, new_chunks),
            max_chunks=budget_chunks,
            max_chars=budget_chars,
        )
        added = len(merged) - before
        stats["new_chunks_total"] += max(0, added)
        stats["iterations"] = round_i

        if added <= 0:
            stats["rounds"].append(
                {
                    "round": round_i,
                    "phase": "table",
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
                "phase": "table",
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

    # --- Agentic TEXT+TABLE evidence assessment loop ---
    remaining = max_iters - int(stats.get("iterations") or 0)
    last_assessment: Optional[EvidenceAssessment] = None
    if remaining > 0 and _should_run_agentic_assessment(result):
        assessment = assess_evidence_sufficiency(question, merged)
        last_assessment = assessment
        stats["evidence_assessment"] = assessment.to_dict()
        agentic_base = int(stats.get("iterations") or 0)
        for step in range(1, remaining + 1):
            if not _should_iterate_assessment(assessment):
                stats["rounds"].append(
                    {
                        "round": agentic_base + step,
                        "phase": "agentic",
                        "chunk_count": len(merged),
                        "stopped": "sufficient" if assessment.sufficient else "no_gaps",
                        "sufficient": assessment.sufficient,
                        "can_answer_safely": assessment.can_answer_safely,
                    }
                )
                break
            queries = _assessment_followup_queries(question, assessment)
            queries = [q for q in queries if q.lower() not in seen_queries]
            if not queries:
                stats["rounds"].append(
                    {
                        "round": agentic_base + step,
                        "phase": "agentic",
                        "chunk_count": len(merged),
                        "stopped": "no_new_queries",
                    }
                )
                break
            for q in queries:
                seen_queries.add(q.lower())

            new_chunks = _retrieve_for_queries(
                queries,
                user_id=user_id,
                document_id=document_id,
                document_ids=document_ids,
                top_k=top_k,
                stats=stats,
            )
            before = len(merged)
            merged = apply_context_budget(
                union_chunks(merged, new_chunks),
                max_chunks=budget_chunks,
                max_chars=budget_chars,
            )
            added = len(merged) - before
            stats["new_chunks_total"] += max(0, added)
            stats["agentic_iterations"] = step
            stats["iterations"] = agentic_base + step

            if added <= 0:
                stats["rounds"].append(
                    {
                        "round": agentic_base + step,
                        "phase": "agentic",
                        "chunk_count": len(merged),
                        "added": 0,
                        "queries": list(queries),
                        "stopped": "no_new_chunks",
                    }
                )
                break

            assessment = assess_evidence_sufficiency(question, merged)
            last_assessment = assessment
            stats["evidence_assessment"] = assessment.to_dict()
            # Soft table path: re-interpret after new evidence arrives.
            if result.is_structured_lookup or any(
                c.get("table") or c.get("table_id") or c.get("table_json")
                for c in merged
            ):
                result = interpret_table_question(question, merged)
            stats["rounds"].append(
                {
                    "round": agentic_base + step,
                    "phase": "agentic",
                    "chunk_count": len(merged),
                    "added": added,
                    "queries": list(queries),
                    "sufficient": assessment.sufficient,
                    "can_answer_safely": assessment.can_answer_safely,
                    "missing": list(assessment.missing_info or []),
                    "verified": bool(result.verified),
                }
            )
            if result.verified is not None:
                break
            try:
                from app.rag.table_rag.telemetry import record_iteration

                record_iteration(
                    round_i=agentic_base + step,
                    queries=queries,
                    new_chunk_count=added,
                    outcome=(
                        "verified"
                        if result.verified
                        else (
                            "sufficient"
                            if assessment.sufficient and assessment.can_answer_safely
                            else "agentic_expand"
                        )
                    ),
                )
            except Exception:
                pass

    stats["elapsed_s"] = round(time.perf_counter() - t0, 3)
    if isinstance(result.raw_llm, dict):
        result.raw_llm = dict(result.raw_llm)
        result.raw_llm["_iterative_stats"] = stats
        if last_assessment is not None:
            result.raw_llm["_evidence_assessment"] = last_assessment.to_dict()
    elif last_assessment is not None:
        result.raw_llm = {
            "_iterative_stats": stats,
            "_evidence_assessment": last_assessment.to_dict(),
        }
    return result, merged, stats

"""Orchestrate structured table lookup over retrieved RAG chunks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_INVALID_TABLE_STRUCTURE,
    STATUS_MATCHED,
    STATUS_NOT_FOUND,
    InsufficientEvidence,
    VerifiedCell,
    lookup_from_chunk,
)
from app.rag.generation.table_calculator import VerifiedCalculation
from app.rag.generation.query_constraints import extract_query_constraints
from app.rag.generation.table_selection import (
    build_table_applicability,
    profile_query_attr_hits,
    select_eligible_table_chunks,
)
from app.rag.ingestion.pdf_layout.table_structure import (
    STRUCTURE_INVALID,
    validate_table_structure,
)

_STRUCTURED_LOOKUP_RE = re.compile(
    r"(?i)\b("
    r"table|chart|grid|matrix|intersection|lookup|"
    r"row|column|cell"
    r")\b"
)
# Two distinct unit-bearing numerics strongly suggest a matrix lookup.
_UNIT_NUM_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*"
    r"(kg|mg|g|µg|ug|μg|mcg|ml|l|%|iu/?mL|ng/?mL|"
    r"mpa|psi|mm|cm|ppm|usd|eur|t)\b"
)


@dataclass
class TableInterpretationResult:
    is_structured_lookup: bool
    result: Optional[Union[VerifiedCell, InsufficientEvidence]] = None
    prompt_prefix: str = ""
    sources_update: Dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> Optional[VerifiedCell]:
        return self.result if isinstance(self.result, VerifiedCell) else None

    @property
    def insufficient(self) -> Optional[InsufficientEvidence]:
        return self.result if isinstance(self.result, InsufficientEvidence) else None


def is_structured_lookup_question(question: str) -> bool:
    """True only for questions that likely need a matrix cell intersection.

    Domain-agnostic: requires explicit table language OR ≥2 distinct
    unit-bearing numeric cues. Ordinary prose questions should NOT force
    structured lookup / insufficient-table banners.
    """
    q = (question or "").strip()
    if not q:
        return False
    if _STRUCTURED_LOOKUP_RE.search(q):
        return True
    units = {m.group(1).lower().replace(" ", "") for m in _UNIT_NUM_RE.finditer(q)}
    return len(units) >= 2


def _table_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for chunk in chunks or []:
        chunk_type = str(
            chunk.get("chunk_type") or chunk.get("content_type") or ""
        ).lower()
        text = str(chunk.get("text") or "")
        if (
            chunk.get("table_json")
            or chunk.get("table")
            or chunk_type
            in (
                "table",
                "table_card",
                "table_row_window",
                "table_summary",
                "table_interpretation",
                "table_row",
            )
            or "table facts:" in text.lower()
            or "table card" in text.lower()
            or "table interpretation" in text.lower()
        ):
            out.append(chunk)
    return out


def _parse_table(chunk: Dict[str, Any]) -> Dict[str, Any]:
    from app.rag.table_rag.ids import parse_table_payload

    raw = chunk.get("table")
    if isinstance(raw, dict) and (raw.get("rows") is not None or raw.get("headers")):
        return dict(raw)
    payload = parse_table_payload(chunk.get("table_json") or raw)
    return payload.to_dict() if payload else {}


def _dedupe_by_table_id(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer hydrated full tables; keep one chunk per table_id."""
    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    orphans: List[Dict[str, Any]] = []
    for chunk in chunks:
        table = _parse_table(chunk)
        tid = str(
            chunk.get("table_id")
            or table.get("table_id")
            or ""
        ).strip()
        if not tid:
            orphans.append(chunk)
            continue
        if tid not in best:
            order.append(tid)
            best[tid] = chunk
            continue
        # Prefer chunks that already carry a hydrated full table with rows.
        prev = _parse_table(best[tid])
        cur_rows = len(table.get("rows") or [])
        prev_rows = len(prev.get("rows") or [])
        if cur_rows > prev_rows or (
            cur_rows == prev_rows and chunk.get("table") and not best[tid].get("table")
        ):
            best[tid] = chunk
    return [best[tid] for tid in order] + orphans


def _interpret_with_llm_selector(
    question: str,
    table_chunks: List[Dict[str, Any]],
) -> Optional[TableInterpretationResult]:
    """Phase-3 path: LLM coordinates + Python SQL cell verification."""
    from app.rag.table_rag.telemetry import TimerMs, record_selector_metrics

    timer = TimerMs()
    try:
        from app.config import get_settings
        from app.rag.generation.table_selector import interpret_with_selector

        settings = get_settings()
        if not bool(getattr(settings, "TABLE_RAG_SELECTOR_ENABLED", True)):
            record_selector_metrics(
                status="legacy_fallback",
                latency_ms=timer.ms(),
                legacy_fallback_reason="selector_disabled",
                index_version=str(
                    getattr(settings, "TABLE_RAG_INDEX_VERSION", "") or ""
                ),
            )
            return None
        if not table_chunks:
            return None
        outcome, decision, candidates = interpret_with_selector(
            question, table_chunks, settings=settings
        )
    except Exception as exc:
        # Fall through to legacy regex path.
        import logging

        logging.getLogger(__name__).warning(
            "LLM table selector unavailable, falling back: %s", exc
        )
        record_selector_metrics(
            status="legacy_fallback",
            latency_ms=timer.ms(),
            legacy_fallback_reason=f"selector_exception:{type(exc).__name__}",
        )
        return None

    latency_ms = timer.ms()
    candidate_ids = [
        str(c.get("table_id") or "") for c in candidates if c.get("table_id")
    ]
    # Soft-fail to legacy axis matcher when the selector cannot run or finds
    # no hydrated grids (unit tests / incomplete retrieve).
    rationale = (decision.rationale or "").lower()
    if (
        rationale.startswith("selector_unavailable")
        or rationale.startswith("selector_error")
        or rationale.startswith("selector_parse_failed")
        or rationale.startswith("no candidate tables")
        or not candidates
    ):
        record_selector_metrics(
            status="legacy_fallback",
            latency_ms=latency_ms,
            candidate_table_ids=candidate_ids,
            legacy_fallback_reason=rationale or "no_candidates",
        )
        return None

    index_version = ""
    document_id = ""
    for chunk in table_chunks:
        index_version = index_version or str(chunk.get("index_version") or "")
        document_id = document_id or str(chunk.get("document_id") or "")

    if isinstance(outcome, VerifiedCell):
        verified_meta = {
            "value": outcome.value,
            "normalized_value": outcome.normalized_value or outcome.value,
            "row_headers": outcome.row_headers,
            "col_headers": outcome.col_headers,
            "row_index": outcome.row_index,
            "col_index": outcome.col_index,
            "page": outcome.page,
            "pages": (outcome.evidence or {}).get("pages") or [],
            "confidence": outcome.confidence,
            "status": STATUS_MATCHED,
            "table_id": outcome.table_id,
            "evidence": dict(outcome.evidence or {}),
            "selector_status": decision.status,
            "selector_rationale": decision.rationale,
        }
        record_selector_metrics(
            status="matched",
            latency_ms=latency_ms,
            candidate_table_ids=candidate_ids,
            index_version=index_version,
            document_id=document_id,
        )
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=outcome,
            prompt_prefix=outcome.to_prompt_block() + "\n\n",
            sources_update={"verified_cell": verified_meta},
        )

    if isinstance(outcome, VerifiedCalculation):
        meta = outcome.to_meta()
        meta["selector_status"] = "requires_calculation"
        meta["selector_rationale"] = decision.rationale
        record_selector_metrics(
            status="requires_calculation",
            latency_ms=latency_ms,
            candidate_table_ids=candidate_ids,
            index_version=index_version,
            document_id=document_id,
        )
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=None,
            prompt_prefix=outcome.to_prompt_block() + "\n\n",
            sources_update={"verified_cell": meta, "verified_calculation": meta},
        )

    insuff = outcome if isinstance(outcome, InsufficientEvidence) else InsufficientEvidence(
        "selector produced no verified cell",
        status=STATUS_NOT_FOUND,
        soft=True,
    )
    soft = insuff.status == STATUS_NOT_FOUND and insuff.soft
    status_label = str(decision.status or insuff.status or "not_in_table")
    if getattr(insuff, "requires_calculation", False):
        status_label = "requires_calculation"
    if insuff.status == STATUS_AMBIGUOUS:
        status_label = "needs_disambiguation"
    if str(insuff.status) == "out_of_range":
        status_label = "out_of_range"
    record_selector_metrics(
        status=status_label,
        latency_ms=latency_ms,
        candidate_table_ids=candidate_ids,
        index_version=index_version,
        document_id=document_id,
    )
    prefix = ""
    fallback_prompt = getattr(insuff, "calculation_fallback_prompt", None)
    if fallback_prompt:
        prefix = str(fallback_prompt)
    else:
        prefix = insuff.to_prompt_block() + "\n\n"
    return TableInterpretationResult(
        is_structured_lookup=True,
        result=insuff,
        prompt_prefix=prefix,
        sources_update={
            "insufficient_table_evidence": True,
            "reason": insuff.reason,
            "status": insuff.status,
            "soft_insufficient": soft,
            "missing_dimensions": list(insuff.missing_dimensions or []),
            "candidate_table_ids": candidate_ids,
            "selector_status": status_label,
            "selector_rationale": decision.rationale,
        },
    )


def interpret_table_question(
    question: str,
    chunks: List[Dict[str, Any]],
) -> TableInterpretationResult:
    """Run verified cell lookup when the question looks like a table query.

    Preferred path (``TABLE_RAG_SELECTOR_ENABLED``): LLM coordinate selector
    over indexed hydrated grids, then Python verifies the exact SQL cell.

    Legacy path: applicability filters + regex axis/boundary matching.
    """
    if not is_structured_lookup_question(question):
        return TableInterpretationResult(is_structured_lookup=False)

    table_chunks = _dedupe_by_table_id(_table_chunks(chunks))
    selector_result = _interpret_with_llm_selector(question, table_chunks)
    if selector_result is not None:
        return selector_result

    # Explicit legacy fallback telemetry (selector returned None).
    try:
        from app.rag.table_rag.telemetry import current_table_rag_trace, record_selector_metrics

        trace = current_table_rag_trace()
        if trace is None or trace.selector_status != "legacy_fallback":
            record_selector_metrics(
                status="legacy_fallback",
                legacy_fallback_reason="selector_returned_none",
                candidate_table_ids=[
                    str(c.get("table_id") or "")
                    for c in table_chunks
                    if c.get("table_id")
                ],
            )
    except Exception:
        pass

    if not table_chunks:
        insuff = InsufficientEvidence(
            "no table chunks retrieved for structured lookup",
            status=STATUS_NOT_FOUND,
            soft=True,
        )
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=insuff,
            prompt_prefix="",
            sources_update={
                "insufficient_table_evidence": True,
                "reason": insuff.reason,
                "status": insuff.status,
                "soft_insufficient": True,
            },
        )

    decision = select_eligible_table_chunks(
        question, table_chunks, get_table=_parse_table
    )
    working = decision.eligible_chunks or []
    if not working:
        insuff = InsufficientEvidence(
            decision.reason or "no eligible tables after applicability filtering",
            status=STATUS_AMBIGUOUS if decision.requires_disambiguation else STATUS_NOT_FOUND,
            soft=False,
            missing_dimensions=list(decision.missing_dimensions),
        )
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=insuff,
            prompt_prefix=insuff.to_prompt_block() + "\n\n",
            sources_update={
                "insufficient_table_evidence": True,
                "reason": insuff.reason,
                "status": insuff.status,
                "soft_insufficient": False,
                "missing_dimensions": insuff.missing_dimensions,
                "excluded_tables": decision.excluded,
            },
        )

    verified: List[VerifiedCell] = []
    last_insuff: Optional[InsufficientEvidence] = None
    structure_failures: List[Dict[str, Any]] = list(decision.structure_blocked or [])
    for chunk in working:
        table = _parse_table(chunk)
        validation = validate_table_structure(table) if table else None
        if validation is not None and (
            validation.status == STRUCTURE_INVALID or not validation.reliable_for_lookup
        ):
            structure_failures.append(
                {
                    "table_id": chunk.get("table_id") or table.get("table_id"),
                    "reason": "unreliable_table_structure",
                    "status": validation.status,
                    "reason_codes": list(validation.reason_codes),
                }
            )
            last_insuff = InsufficientEvidence(
                "table structure is unreliable for definitive lookup",
                status=STATUS_INVALID_TABLE_STRUCTURE,
                soft=False,
            )
            continue
        outcome = lookup_from_chunk(chunk, question)
        if outcome is None:
            continue
        if isinstance(outcome, VerifiedCell):
            if not outcome.table_id:
                outcome.table_id = str(
                    chunk.get("table_id") or table.get("table_id") or ""
                )
                evidence = dict(outcome.evidence or {})
                evidence["table_id"] = outcome.table_id
                outcome.evidence = evidence
            # Never pin a matched cell from a needs-verification grid without
            # marking caution — selection already filters unreliable ones.
            verified.append(outcome)
        elif isinstance(outcome, InsufficientEvidence):
            last_insuff = outcome

    if not verified:
        insuff = last_insuff or InsufficientEvidence(
            "no verified cell match in retrieved tables",
            status=STATUS_NOT_FOUND,
            soft=True,
        )
        if structure_failures and insuff.status == STATUS_NOT_FOUND:
            insuff = InsufficientEvidence(
                "eligible tables failed structural validation or lookup",
                status=STATUS_INVALID_TABLE_STRUCTURE
                if all(
                    f.get("status") == STRUCTURE_INVALID for f in structure_failures
                )
                else insuff.status,
                soft=False,
            )
        # Soft not_found → prose RAG may still answer from surrounding text.
        # Ambiguous / out_of_range / invalid structure stay hard abstentions.
        soft = insuff.status == STATUS_NOT_FOUND
        insuff.soft = soft
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=insuff,
            # Soft notice guides answer-gen without forcing abstain boilerplate.
            prompt_prefix=insuff.to_prompt_block() + "\n\n",
            sources_update={
                "insufficient_table_evidence": True,
                "reason": insuff.reason,
                "status": insuff.status,
                "soft_insufficient": soft,
                "structure_blocked": structure_failures,
                "excluded_tables": decision.excluded,
                "query_constraints": decision.query_constraints,
            },
        )

    # Deduplicate identical table_id hits.
    by_table: Dict[str, VerifiedCell] = {}
    for cell in verified:
        key = cell.table_id or f"page:{cell.page}:{cell.row_index}:{cell.col_index}"
        prev = by_table.get(key)
        if prev is None or cell.confidence >= prev.confidence:
            by_table[key] = cell
    unique_cells = list(by_table.values())
    values = {c.value for c in unique_cells}

    # Sibling tables with distinct applicability + missing query dims → ambiguous
    # even before value disagreement, when selection flagged it.
    if decision.requires_disambiguation and len(unique_cells) > 1:
        insuff = InsufficientEvidence(
            "multiple applicable tables could answer this lookup; "
            "query lacks differentiating constraints",
            status=STATUS_AMBIGUOUS,
            soft=False,
            candidate_values=sorted(values),
            candidate_table_ids=sorted(
                {c.table_id for c in unique_cells if c.table_id}
            ),
            missing_dimensions=list(decision.missing_dimensions)
            or ["applicability_class"],
        )
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=insuff,
            prompt_prefix=insuff.to_prompt_block() + "\n\n",
            sources_update={
                "insufficient_table_evidence": True,
                "reason": insuff.reason,
                "status": STATUS_AMBIGUOUS,
                "soft_insufficient": False,
                "candidate_values": insuff.candidate_values,
                "candidate_table_ids": insuff.candidate_table_ids,
                "missing_dimensions": insuff.missing_dimensions,
            },
        )

    # Selection said ambiguous but only one cell verified — abstain unless the
    # winner's attribute values are explicitly hit by the query (domain-agnostic
    # evidence that the pin is not retrieval-rank luck).
    if (
        decision.requires_disambiguation
        and len(decision.eligible_chunks) > 1
        and decision.missing_dimensions
        and len(unique_cells) == 1
    ):
        winner = unique_cells[0]
        winner_hits = False
        q_tokens = extract_query_constraints(question).tokens
        winner_chunk = None
        for chunk in decision.eligible_chunks:
            tid = str(chunk.get("table_id") or "")
            if winner.table_id and tid == winner.table_id:
                winner_chunk = chunk
                break
        if winner_chunk is None and decision.eligible_chunks:
            # Fall back when table_id was not populated on the cell.
            winner_chunk = decision.eligible_chunks[0]
        if winner_chunk is not None:
            table = _parse_table(winner_chunk)
            if table:
                profile = build_table_applicability(
                    table, chunk=winner_chunk
                )
                winner_hits = bool(
                    profile_query_attr_hits(profile, q_tokens)
                    or (set(profile.purpose_tokens or set()) & q_tokens)
                )
        if not winner_hits:
            insuff = InsufficientEvidence(
                "query lacks differentiating constraints among eligible tables; "
                "refusing to pin a single table by retrieval rank",
                status=STATUS_AMBIGUOUS,
                soft=False,
                candidate_values=sorted(values),
                candidate_table_ids=sorted(
                    {
                        str(c.get("table_id") or "")
                        for c in decision.eligible_chunks
                        if c.get("table_id")
                    }
                ),
                missing_dimensions=list(decision.missing_dimensions),
            )
            return TableInterpretationResult(
                is_structured_lookup=True,
                result=insuff,
                prompt_prefix=insuff.to_prompt_block() + "\n\n",
                sources_update={
                    "insufficient_table_evidence": True,
                    "reason": insuff.reason,
                    "status": STATUS_AMBIGUOUS,
                    "soft_insufficient": False,
                    "candidate_values": insuff.candidate_values,
                    "candidate_table_ids": insuff.candidate_table_ids,
                    "missing_dimensions": insuff.missing_dimensions,
                },
            )

    if len(values) > 1:
        # Hard ambiguous: never break ties with lexical overlap.
        insuff = InsufficientEvidence(
            "retrieved tables disagree on the intersecting cell value",
            status=STATUS_AMBIGUOUS,
            soft=False,
            candidate_values=sorted(values),
            candidate_table_ids=sorted(
                {c.table_id for c in unique_cells if c.table_id}
            ),
            missing_dimensions=list(decision.missing_dimensions)
            or ["applicability_class"],
        )
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=insuff,
            prompt_prefix=insuff.to_prompt_block() + "\n\n",
            sources_update={
                "insufficient_table_evidence": True,
                "reason": insuff.reason,
                "status": STATUS_AMBIGUOUS,
                "soft_insufficient": False,
                "candidate_values": insuff.candidate_values,
                "candidate_table_ids": insuff.candidate_table_ids,
                "missing_dimensions": insuff.missing_dimensions,
            },
        )

    # Identical values from multiple tables → accept with multi-source evidence.
    best = max(unique_cells, key=lambda c: c.confidence)
    verified_meta = {
        "value": best.value,
        "normalized_value": best.normalized_value or best.value,
        "row_headers": best.row_headers,
        "col_headers": best.col_headers,
        "row_index": best.row_index,
        "col_index": best.col_index,
        "page": best.page,
        "confidence": best.confidence,
        "status": STATUS_MATCHED,
        "table_id": best.table_id,
        "evidence": dict(best.evidence or {}),
        "agreeing_table_ids": sorted(
            {c.table_id for c in unique_cells if c.table_id}
        ),
        "missing_dimensions": list(decision.missing_dimensions),
        "query_constraints": decision.query_constraints,
    }
    if (best.evidence or {}).get("prescribed_value") is not None:
        verified_meta["compare_mode"] = True
        verified_meta["prescribed_value"] = best.evidence.get("prescribed_value")
        verified_meta["prescribed_unit"] = best.evidence.get("prescribed_unit")

    return TableInterpretationResult(
        is_structured_lookup=True,
        result=best,
        prompt_prefix=best.to_prompt_block() + "\n\n",
        sources_update={"verified_cell": verified_meta},
    )

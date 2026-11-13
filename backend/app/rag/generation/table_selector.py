"""LLM coordinate selector + Python deterministic verification gate.

The LLM proposes ``(table_id, row_index, col_index)``; Python always reads the
raw cell from the SQL-hydrated grid and constructs ``VerifiedCell``. Never trust
the LLM for cell *values*.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from app.config import get_settings
from app.rag.generation.structured_lookup import (
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_NOT_FOUND,
    STATUS_OUT_OF_RANGE,
    InsufficientEvidence,
    VerifiedCell,
)
from app.rag.generation.table_calculator import (
    VerifiedCalculation,
    calculation_fallback_prompt,
    try_calculate_from_decision,
)
from app.rag.table_rag.ids import parse_table_payload
from app.rag.table_rag.representations import render_indexed_table_for_selector
from app.rag.table_rag.schema import TableLookupDecision, coerce_table_dict

logger = logging.getLogger(__name__)

SELECTOR_SYSTEM_PROMPT = """You are a deterministic table coordinate selector. Given a user question and candidate tables with explicit [row_idx, col_idx] coordinates:
1. Check if multiple candidate tables apply to different cohorts/conditions/periods/categories and the user's question omits a required differentiating dimension. If so, return status="needs_disambiguation" and list the exact `missing_dimensions`.
2. Enforce strict mathematical boundary semantics (< vs <=, > vs >=). Never guess across ambiguous boundary edges.
3. If a unique cell answers the question, return status="matched" with exact `table_id`, `row_index`, and `col_index`.
4. If the question requires multi-row aggregation, comparison, or arithmetic, return status="requires_calculation" with `table_id`, `operation` (sum|difference|ratio|product|average), and `operands` listing each exact `[row_index, col_index]` cell required. Also mention those coordinates as [r,c] in rationale.
5. If no table contains the answer, return status="not_in_table".
6. Stated/observed amounts in the question are NOT the lookup target when the question also gives axis sizing factors (row/column keys such as ranges, grades, periods, quantities). Select the LABEL / matrix cell for those axis factors on the applicability-matched table. Never pick a cell only because its value equals a stated amount.
7. Classification or comparison questions still need the labeled matrix cell when sizing factors are present. Return status="matched" for that LABEL cell. Do NOT return not_in_table merely because classification wording is absent from the matrix — your job is to pin the axis intersection cell. Note any related schedule/frequency/unit column in rationale when present.
8. Scope-lock on caption/cohort/period before coordinates: match the table whose caption/section fits the question's applicability (population, period, category, scenario). Never pick a sibling matrix for another cohort or period just because the numeric axes look similar.

Return JSON only matching:
{
  "status": "matched|needs_disambiguation|not_in_table|requires_calculation",
  "table_id": "..." or null,
  "row_index": 0 or null,
  "col_index": 0 or null,
  "missing_dimensions": ["..."],
  "operands": [{"row_index": 0, "col_index": 1}],
  "operation": "sum|difference|ratio|product|average|",
  "rationale": "..."
}
No markdown fences."""

_EMPTY_OR_UNSUPPORTED_CELL_RE = re.compile(
    r"(?i)^\s*(?:[-–—]+|n/?a|none|null|empty|"
    r"insufficient(?:\s+data)?(?:\s+to\s+recommend(?:\s+a\s+\w+)?)?|"
    r"not\s+recommended|no\s+value|not\s+applicable)\s*$"
)
_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _table_dict_from_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    raw = chunk.get("table")
    if isinstance(raw, dict) and (raw.get("rows") is not None or raw.get("headers")):
        return dict(raw)
    payload = parse_table_payload(chunk.get("table_json") or raw)
    return payload.to_dict() if payload else {}


def _leaf_headers(table: Dict[str, Any]) -> List[str]:
    paths = table.get("normalized_header_paths") or []
    headers = [str(h or "") for h in (table.get("headers") or [])]
    if paths and len(paths) >= len(headers) and headers:
        leaves = []
        for path in paths[: len(headers)]:
            if isinstance(path, (list, tuple)) and path:
                leaves.append(str(path[-1] or ""))
            else:
                leaves.append(str(path or ""))
        if any(leaves):
            return leaves
    return headers


def collect_candidate_tables(
    chunks: Sequence[Dict[str, Any]],
    *,
    top_k: int = 4,
    question: str = "",
) -> List[Dict[str, Any]]:
    """Dedupe hydrated chunks by table_id; prefer fuller grids; cap at top_k.

    When ``question`` is provided, re-rank / filter by indication·cohort scope
    so sibling matrices (different captions/populations) are not offered equally.
    """
    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for chunk in chunks or []:
        table = _table_dict_from_chunk(chunk)
        tid = str(chunk.get("table_id") or table.get("table_id") or "").strip()
        if not tid or not table:
            continue
        cur_score = float(chunk.get("score") or chunk.get("confidence") or 0)
        cur_caption = str(
            table.get("caption") or chunk.get("caption") or ""
        ).strip()
        if cur_caption and not str(table.get("caption") or "").strip():
            table = dict(table)
            table["caption"] = cur_caption
        if tid not in best:
            order.append(tid)
            best[tid] = {
                "chunk": chunk,
                "table": table,
                "table_id": tid,
                "score": cur_score,
            }
            continue
        prev = best[tid]
        prev_table = prev.get("table") or {}
        cur_rows = len(table.get("rows") or [])
        prev_rows = len(prev_table.get("rows") or [])
        prev_caption = str(
            prev_table.get("caption") or (prev.get("chunk") or {}).get("caption") or ""
        ).strip()
        # Never let a higher-scoring empty-caption retrieval wipe a recovered title.
        if cur_caption and not prev_caption:
            merged = dict(prev_table)
            merged.update({k: v for k, v in table.items() if k != "caption" or v})
            merged["caption"] = cur_caption
            best[tid] = {
                "chunk": chunk,
                "table": merged,
                "table_id": tid,
                "score": max(cur_score, float(prev.get("score") or 0)),
            }
            continue
        if cur_rows > prev_rows or (
            cur_rows == prev_rows and cur_score > prev["score"]
        ):
            merged = dict(table)
            if prev_caption and not cur_caption:
                merged["caption"] = prev_caption
            best[tid] = {
                "chunk": chunk,
                "table": merged,
                "table_id": tid,
                "score": cur_score,
            }
    ordered = [best[tid] for tid in order]
    if question.strip():
        ordered = scope_rank_candidates(question, ordered)
    else:
        ordered = sorted(ordered, key=lambda item: item["score"], reverse=True)
    return ordered[: max(1, int(top_k or 4))]


# Age/cohort words are scored via ``_age_prior`` only. Including them in
# distinctive caption hits lets "Adult X" questions prefer an unrelated
# sibling whose caption merely says "Adult …" over the indication-matched table.
_AGE_COHORT_TOKENS = frozenset(
    {
        "adult",
        "adults",
        "adolescent",
        "adolescents",
        "pediatric",
        "paediatric",
        "child",
        "children",
        "infant",
        "infants",
        "neonate",
        "neonates",
        "neonatal",
        "geriatric",
        "elderly",
        "years",
        "year",
        "age",
        "ages",
        "older",
        "younger",
    }
)


def scope_rank_candidates(
    question: str,
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank/filter sibling tables by caption·purpose·age scope vs the question.

    Domain-agnostic: prefers axis-unit overlap, then soft-boosts distinctive
    caption/purpose tokens that appear in the question. Hard-drops only
    age-incompatible cohorts when the question states a calendar age.
    """
    from app.rag.generation.table_selection import (
        _tokens,
        build_table_applicability,
        question_age_constraints,
        question_table_number,
        table_number_match,
        _bound_compatible,
        _header_units,
    )
    from app.rag.generation.structured_lookup import (
        ROLE_AXIS,
        extract_numeric_constraints,
    )

    items = list(candidates or [])
    if len(items) <= 1:
        return items

    q = question or ""
    q_tokens = _tokens(q) - _AGE_COHORT_TOKENS
    age_bounds = question_age_constraints(q)
    table_num = question_table_number(q)
    axis_units = {
        (c.unit or "").lower().replace(" ", "")
        for c in extract_numeric_constraints(q)
        if c.role == ROLE_AXIS and c.unit
    }

    # Hard preference: when the question cites Table N and at least one candidate
    # caption matches, drop non-matching siblings (domain-agnostic).
    if table_num:
        matched_num = [
            item
            for item in items
            if table_number_match(
                item.get("table") or {},
                item.get("chunk") or {},
                table_num,
            )
        ]
        if matched_num:
            items = matched_num

    enriched: List[Dict[str, Any]] = []
    for item in items:
        table = item.get("table") or {}
        chunk = item.get("chunk") or {}
        profile = build_table_applicability(table, chunk=chunk)
        ranges = profile.bound_ranges or profile.age_ranges
        if age_bounds and ranges:
            if not any(_bound_compatible(b, ranges) for b in age_bounds):
                continue  # hard drop incompatible age cohort
        header_units = profile.header_units or _header_units(table)
        axis_overlap = len(axis_units & header_units) if axis_units else 0
        caption = str(table.get("caption") or chunk.get("caption") or "")
        scope_tokens = set(profile.purpose_tokens or set()) | _tokens(caption)
        scope_tokens |= set(profile.cue_tokens or set())
        for val in (profile.attributes or {}).values():
            scope_tokens |= _tokens(str(val))
        dosing_like = 1.0 if re.search(
            r"(?i)\b(dose|doses|dosing|dosage|recommended|schedule|regimen)\b",
            caption + " " + " ".join(profile.cues[:4]),
        ) else 0.0
        age_prior = 0.0
        cap_low = caption.lower()
        q_low = q.lower()
        pediatric_q = bool(
            re.search(r"(?i)\b(pediatric|child|children|infant)\b", q_low)
            or any(b < 12 for b in age_bounds)
        )
        adult_q = bool(
            re.search(r"(?i)\b(adult|adolescents?|12 years|≥\s*12|>=\s*12)\b", q_low)
            or any(b >= 12 for b in age_bounds)
        )
        pediatric_t = bool(
            re.search(r"(?i)\b(pediatric|child|6\s*to\s*<\s*12|<12)\b", cap_low)
        )
        adult_t = bool(
            re.search(
                r"(?i)\b(12 years|≥\s*12|>=\s*12|and older|adult)\b",
                cap_low,
            )
        )
        if pediatric_q and pediatric_t:
            age_prior += 1.0
        if adult_q and adult_t and not pediatric_t:
            age_prior += 1.0
        if adult_q and pediatric_t and not pediatric_q:
            age_prior -= 1.5
        if pediatric_q and adult_t and not pediatric_t:
            age_prior -= 1.0
        # Prefer earlier pages among otherwise equal siblings (primary tables).
        try:
            page_key = int(
                table.get("page")
                or table.get("page_start")
                or chunk.get("page")
                or 10_000
            )
        except (TypeError, ValueError):
            page_key = 10_000
        indication_prior = 0.0
        if scope_tokens & q_tokens:
            indication_prior += 0.5
        enriched.append(
            {
                **item,
                "_scope_tokens": scope_tokens,
                "_axis_overlap": axis_overlap,
                "_dosing_like": dosing_like,
                "_age_prior": age_prior,
                "_indication_prior": indication_prior,
                "_page_key": page_key,
                "_caption": caption,
            }
        )

    if not enriched:
        return list(items)  # age filter emptied — keep original rather than fail closed

    for item in enriched:
        others: set = set()
        for other in enriched:
            if other is item:
                continue
            others |= other["_scope_tokens"]
        # Content tokens only — age/cohort handled by ``_age_prior``.
        distinctive = (item["_scope_tokens"] - others) - _AGE_COHORT_TOKENS
        content_scope = item["_scope_tokens"] - _AGE_COHORT_TOKENS
        item["_q_hit"] = distinctive & q_tokens
        item["_q_overlap"] = content_scope & q_tokens

    if axis_units:
        with_axes = [item for item in enriched if item["_axis_overlap"] > 0]
        if with_axes:
            enriched = with_axes

    for item in enriched:
        hit = len(item.get("_q_hit") or ())
        if hit:
            item["_indication_prior"] = float(item.get("_indication_prior") or 0) + min(
                2.0, float(hit)
            )
    ranked = sorted(
        enriched,
        key=lambda item: (
            float(item.get("_indication_prior") or 0),
            float(item.get("_age_prior") or 0),
            float(item.get("_axis_overlap") or 0),
            float(item.get("_dosing_like") or 0),
            len(item.get("_q_hit") or ()),
            len(item.get("_q_overlap") or ()),
            float(item.get("score") or 0),
            -float(item.get("_page_key") or 10_000),
        ),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    for item in ranked:
        out.append(
            {
                "chunk": item.get("chunk"),
                "table": item.get("table"),
                "table_id": item.get("table_id"),
                "score": float(item.get("score") or 0)
                + 6.0 * float(item.get("_indication_prior") or 0)
                + 5.0 * float(item.get("_axis_overlap") or 0)
                + 4.0 * float(item.get("_dosing_like") or 0)
                + 3.0 * float(item.get("_age_prior") or 0)
                + 3.0 * len(item.get("_q_hit") or ())
                + 0.5 * len(item.get("_q_overlap") or ())
                - 0.1 * float(item.get("_page_key") or 0),
            }
        )
    return out


def _extract_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def select_table_coordinates_llm(
    question: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    client: Any = None,
    settings: Any = None,
) -> TableLookupDecision:
    """Low-temperature structured LLM call → ``TableLookupDecision``."""
    settings = settings or get_settings()
    if not candidates:
        return TableLookupDecision(
            status="not_in_table",
            rationale="no candidate tables",
        )

    blocks = []
    for item in candidates:
        table = item.get("table") or {}
        chunk = item.get("chunk") or {}
        record = {
            "table_id": item.get("table_id") or table.get("table_id"),
            "table": table,
            "pages": table.get("pages")
            or chunk.get("pages")
            or [table.get("page") or chunk.get("page")],
            "section": chunk.get("section") or "",
            "caption": table.get("caption") or "",
        }
        blocks.append(render_indexed_table_for_selector(record))

    q = (question or "").strip()
    user_prompt = (
        f"User question:\n{q}\n\n"
        f"Candidate tables ({len(blocks)}):\n\n"
        + "\n\n-----\n\n".join(blocks)
    )

    if client is None:
        try:
            from app.rag.llm.agent import get_llm_client

            client = get_llm_client()
        except Exception as exc:
            logger.warning("Table selector LLM unavailable: %s", exc)
            return TableLookupDecision(
                status="not_in_table",
                rationale=f"selector_unavailable: {exc}",
            )

    try:
        from app.rag.llm.agent import llm_model_for_provider

        model = llm_model_for_provider()
    except Exception:
        model = getattr(settings, "LLM_MODEL", None)

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            max_tokens=512,
            temperature=0.0,
            stream=False,
        )
        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = ""
        parsed = _extract_json(content)
        if not parsed:
            return TableLookupDecision(
                status="not_in_table",
                rationale="selector_parse_failed",
            )
        return TableLookupDecision.model_validate(parsed)
    except Exception as exc:
        logger.warning("Table selector LLM call failed: %s", exc)
        return TableLookupDecision(
            status="not_in_table",
            rationale=f"selector_error: {exc}",
        )


def verify_lookup_decision(
    decision: TableLookupDecision,
    candidates: Sequence[Dict[str, Any]],
    *,
    question: str = "",
) -> Union[VerifiedCell, VerifiedCalculation, InsufficientEvidence]:
    """Python gate: hydrate cell from SQL grid; never trust LLM values."""
    status = decision.status
    if status == "needs_disambiguation":
        return InsufficientEvidence(
            decision.rationale
            or "multiple applicable tables; missing differentiating dimensions",
            status=STATUS_AMBIGUOUS,
            soft=False,
            missing_dimensions=list(decision.missing_dimensions or []),
            candidate_table_ids=[
                str(c.get("table_id") or "") for c in candidates if c.get("table_id")
            ],
        )
    if status == "requires_calculation":
        table_id = str(decision.table_id or "").strip()
        by_id = {str(c.get("table_id") or ""): c for c in candidates}
        record = by_id.get(table_id) if table_id else None
        if record is None and len(candidates) == 1:
            record = candidates[0]
            table_id = str(record.get("table_id") or "")
        if record is None:
            return InsufficientEvidence(
                decision.rationale
                or "requires_calculation without a valid table_id",
                status=STATUS_NOT_FOUND,
                soft=True,
                missing_dimensions=list(decision.missing_dimensions or []),
            )
        table = coerce_table_dict(record.get("table"))
        chunk = record.get("chunk") or {}
        try:
            calc = try_calculate_from_decision(
                decision,
                table,
                question=question,
                filename=str(chunk.get("filename") or ""),
            )
        except Exception as exc:
            logger.warning("Calculation helper failed: %s", exc)
            calc = None
        if calc is not None:
            return calc  # type: ignore[return-value]
        # Graceful degrade: inject full grid + step-by-step instruction.
        prompt = calculation_fallback_prompt(
            table,
            question=question,
            table_id=table_id,
            rationale=decision.rationale or "",
        )
        insuff = InsufficientEvidence(
            decision.rationale
            or "calculation operands could not be verified automatically",
            status=STATUS_NOT_FOUND,
            soft=True,
            missing_dimensions=list(decision.missing_dimensions or []),
            candidate_table_ids=[table_id] if table_id else [],
        )
        # Attach fallback prompt via a dynamic attribute for interpretation.
        setattr(insuff, "calculation_fallback_prompt", prompt)
        setattr(insuff, "requires_calculation", True)
        return insuff
    if status != "matched":
        return InsufficientEvidence(
            decision.rationale or "answer not found in candidate tables",
            status=STATUS_NOT_FOUND,
            soft=True,
        )

    table_id = str(decision.table_id or "").strip()
    by_id = {str(c.get("table_id") or ""): c for c in candidates}
    record = by_id.get(table_id)
    if record is None:
        return InsufficientEvidence(
            f"selector returned unknown table_id={table_id!r}",
            status=STATUS_NOT_FOUND,
            soft=False,
        )

    table = coerce_table_dict(record.get("table"))
    rows = [list(r) for r in (table.get("rows") or [])]
    headers = _leaf_headers(table)
    row_index = decision.row_index
    col_index = decision.col_index
    if row_index is None or col_index is None:
        return InsufficientEvidence(
            "selector matched without row/col coordinates",
            status=STATUS_NOT_FOUND,
            soft=False,
        )
    try:
        r_i = int(row_index)
        c_i = int(col_index)
    except (TypeError, ValueError):
        return InsufficientEvidence(
            "selector coordinates are not integers",
            status=STATUS_NOT_FOUND,
            soft=False,
        )
    if r_i < 0 or r_i >= len(rows):
        return InsufficientEvidence(
            f"selector row_index={r_i} out of bounds (n_rows={len(rows)})",
            status=STATUS_NOT_FOUND,
            soft=False,
        )
    row = list(rows[r_i] or [])
    if c_i < 0 or c_i >= max(len(row), len(headers)):
        return InsufficientEvidence(
            f"selector col_index={c_i} out of bounds",
            status=STATUS_NOT_FOUND,
            soft=False,
        )
    # Pad short rows.
    if c_i >= len(row):
        row = row + [""] * (c_i - len(row) + 1)
    raw_val = str(row[c_i] if row[c_i] is not None else "").strip()
    if not raw_val or _EMPTY_OR_UNSUPPORTED_CELL_RE.match(raw_val):
        return InsufficientEvidence(
            f"matched cell is empty/unsupported: {raw_val!r}",
            status=STATUS_OUT_OF_RANGE,
            soft=False,
            candidate_table_ids=[table_id],
        )

    # Row axis label = first column when present; col header from leaf headers.
    row_headers = [str(row[0])] if row and str(row[0] or "").strip() else [f"row {r_i}"]
    col_headers = [headers[c_i]] if c_i < len(headers) and headers[c_i] else [f"col {c_i}"]
    pages = table.get("pages") or []
    page = None
    if pages:
        try:
            page = int(pages[0])
        except (TypeError, ValueError):
            page = table.get("page")
    else:
        page = table.get("page")
    chunk = record.get("chunk") or {}
    filename = str(chunk.get("filename") or "")

    # When the LLM names the right band in rationale but returns the wrong
    # column/row index, correct via deterministic axis matching.
    if question and not _selector_axes_cover_question(
        row_headers, col_headers, question
    ):
        try:
            from app.rag.generation.structured_lookup import lookup_cell_in_table

            corrected = lookup_cell_in_table(
                table,
                question,
                filename=filename,
                page=int(page) if page is not None else None,
            )
        except Exception as exc:
            logger.warning("Selector axis correction failed: %s", exc)
            corrected = None
        if isinstance(corrected, VerifiedCell):
            evidence = dict(corrected.evidence or {})
            evidence.update(
                {
                    "table_id": table_id,
                    "table_caption": table.get("caption") or "",
                    "table_section": chunk.get("section") or "",
                    "pages": pages,
                    "selector_rationale": decision.rationale or "",
                    "question": question,
                    "verification": "python_sql_hydrate_axis_corrected",
                    "selector_coords_overridden": {
                        "row_index": r_i,
                        "col_index": c_i,
                        "value": raw_val,
                    },
                }
            )
            corrected.evidence = evidence
            logger.info(
                "Corrected selector coords (%s,%s)=%r → (%s,%s)=%r",
                r_i,
                c_i,
                raw_val,
                corrected.row_index,
                corrected.col_index,
                corrected.value,
            )
            return corrected

    evidence = {
        "table_id": table_id,
        "table_caption": table.get("caption") or "",
        "table_section": chunk.get("section") or "",
        "pages": pages,
        "selector_rationale": decision.rationale or "",
        "question": question,
        "verification": "python_sql_hydrate",
    }
    return VerifiedCell(
        value=raw_val,
        row_headers=row_headers,
        col_headers=col_headers,
        row_index=r_i,
        col_index=c_i,
        page=int(page) if page is not None else None,
        filename=filename,
        confidence=100.0,
        table_headers=headers,
        evidence=evidence,
        status=STATUS_MATCHED,
        table_id=table_id,
        normalized_value=raw_val,
    )


def _selector_axes_cover_question(
    row_headers: Sequence[str],
    col_headers: Sequence[str],
    question: str,
) -> bool:
    """True when selected row/col labels cover ≥2 distinct axis constraints."""
    try:
        from app.rag.generation.structured_lookup import (
            _axis_constraints,
            _score_label_match,
            extract_numeric_constraints,
            parse_range_label,
            range_contains,
        )
    except Exception:
        return True

    axis_cs = _axis_constraints(extract_numeric_constraints(question))
    if len(axis_cs) < 2:
        return True
    labels = [str(x or "") for x in list(row_headers) + list(col_headers)]
    matched = 0
    used: set[int] = set()
    for i, constraint in enumerate(axis_cs):
        for lab in labels:
            parsed = parse_range_label(lab)
            hit = False
            if parsed is not None:
                try:
                    hit = range_contains(parsed, float(constraint.value), constraint.unit)
                except Exception:
                    hit = False
            if not hit and _score_label_match(lab, constraint) > 0:
                hit = True
            if hit:
                matched += 1
                used.add(i)
                break
    return matched >= 2 and len(used) >= 2


def expand_document_sibling_chunks(
    chunks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pull same-document dosing-like sibling tables from SQL into the pool.

    Retrieval often returns only one of several indication/cohort matrices.
    Expanding siblings lets scope-rank pick the indication-matched table even
    when it was not in the top retrieved hits.
    """
    from app.rag.generation.table_selection import _header_units

    out: List[Dict[str, Any]] = [dict(c) for c in (chunks or [])]
    seen: set = set()
    doc_ids: set = set()
    user_id = ""
    for chunk in out:
        tid = str(chunk.get("table_id") or "").strip()
        doc = str(chunk.get("document_id") or "").strip()
        if tid:
            seen.add(tid)
        if doc:
            doc_ids.add(doc)
        if not user_id:
            user_id = str(chunk.get("user_id") or "").strip()

    if not doc_ids:
        return out

    try:
        from app.rag.table_rag.store import list_document_tables
    except Exception:
        return out

    for doc in doc_ids:
        try:
            rows = list_document_tables(
                document_id=doc,
                user_id=user_id or None,
            )
        except Exception as exc:
            logger.warning("Sibling table expand failed for %s: %s", doc, exc)
            continue
        for row in rows or []:
            tid = str(row.get("table_id") or "").strip()
            if not tid:
                continue
            table = dict(row.get("table") or {})
            if not table.get("rows"):
                continue
            table.setdefault("table_id", tid)
            # Prefer non-empty SQL caption/understanding over stale empty
            # values left in retrieved Chroma payloads.
            sql_caption = str(row.get("caption") or "").strip()
            if sql_caption:
                table["caption"] = sql_caption
            else:
                table.setdefault("caption", "")
            table.setdefault("page", row.get("page_start"))
            if row.get("llm_understanding"):
                table["llm_understanding"] = row.get("llm_understanding")
            units = _header_units(table)
            caption = str(table.get("caption") or row.get("caption") or "")
            # Patch already-retrieved siblings that arrived with empty captions.
            if tid in seen:
                if sql_caption:
                    for existing in out:
                        if str(existing.get("table_id") or "") != tid:
                            continue
                        existing["caption"] = sql_caption
                        et = dict(existing.get("table") or {})
                        et["caption"] = sql_caption
                        if row.get("llm_understanding"):
                            et["llm_understanding"] = row.get("llm_understanding")
                        existing["table"] = et
                continue
            # Keep matrices with axis units or any non-empty caption (domain-agnostic).
            if not caption.strip() and len(units) < 1:
                continue
            seen.add(tid)
            out.append(
                {
                    "table_id": tid,
                    "document_id": doc,
                    "user_id": user_id or row.get("user_id") or "",
                    "table": table,
                    "caption": caption,
                    "section": row.get("section") or "",
                    "page": row.get("page_start"),
                    "filename": row.get("filename") or "",
                    "score": 0.15,
                    "chunk_type": "table",
                    "index_kind": "table_sibling_expand",
                }
            )
    return out


def interpret_with_selector(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    client: Any = None,
    settings: Any = None,
) -> Tuple[
    Union[VerifiedCell, VerifiedCalculation, InsufficientEvidence],
    TableLookupDecision,
    List[Dict[str, Any]],
]:
    """End-to-end: scope-rank candidates → LLM decision → Python verify/calc."""
    settings = settings or get_settings()
    top_k = int(getattr(settings, "TABLE_RAG_SELECTOR_TOP_K", 4) or 4)
    expanded = expand_document_sibling_chunks(chunks)
    # Rank the full expanded sibling pool before truncating by retrieval score.
    pool_k = max(len(expanded), top_k * 8, 32)
    raw = collect_candidate_tables(expanded, top_k=pool_k, question="")
    candidates = scope_rank_candidates(question, raw)[: max(1, top_k)]
    if not candidates:
        candidates = collect_candidate_tables(
            expanded, top_k=top_k, question=question
        )
    # When the top caption-scoped table already yields a unique axis cell,
    # prefer Python over the LLM sibling choice.
    if candidates:
        try:
            from app.rag.generation.structured_lookup import lookup_cell_in_table

            scoped = candidates[0]
            deterministic = lookup_cell_in_table(
                coerce_table_dict(scoped.get("table")),
                question,
                filename=str((scoped.get("chunk") or {}).get("filename") or ""),
                page=(scoped.get("table") or {}).get("page"),
            )
            if isinstance(deterministic, VerifiedCell):
                decision = TableLookupDecision(
                    status="matched",
                    table_id=str(scoped.get("table_id") or ""),
                    row_index=deterministic.row_index,
                    col_index=deterministic.col_index,
                    rationale="deterministic_scope_rank_axis_lookup",
                )
                evidence = dict(deterministic.evidence or {})
                evidence["verification"] = "python_scope_rank_deterministic"
                deterministic.evidence = evidence
                return deterministic, decision, candidates
        except Exception as exc:
            logger.warning("Deterministic scope-rank lookup failed: %s", exc)
    decision = select_table_coordinates_llm(
        question, candidates, client=client, settings=settings
    )
    # If the LLM still picks an out-of-scope sibling, force the top scoped table.
    if (
        decision.status == "matched"
        and decision.table_id
        and candidates
        and str(decision.table_id) != str(candidates[0].get("table_id") or "")
        and not _table_in_scope(question, decision.table_id, candidates)
    ):
        logger.info(
            "Selector table %s out of scope; preferring %s",
            decision.table_id,
            candidates[0].get("table_id"),
        )
        # Keep coordinates only if same table; otherwise clear for axis correction
        # / deterministic lookup on the preferred table via verify path.
        decision = TableLookupDecision(
            status="matched",
            table_id=str(candidates[0].get("table_id") or ""),
            row_index=None,
            col_index=None,
            rationale=(
                (decision.rationale or "")
                + " [scope-lock overridden to indication-matched table]"
            ).strip(),
        )
        # Deterministic axis lookup on the scoped table.
        try:
            from app.rag.generation.structured_lookup import lookup_cell_in_table

            scoped = candidates[0]
            corrected = lookup_cell_in_table(
                coerce_table_dict(scoped.get("table")),
                question,
                filename=str((scoped.get("chunk") or {}).get("filename") or ""),
                page=(scoped.get("table") or {}).get("page"),
            )
            if isinstance(corrected, VerifiedCell):
                evidence = dict(corrected.evidence or {})
                evidence["verification"] = "python_scope_lock_override"
                evidence["selector_rationale"] = decision.rationale
                corrected.evidence = evidence
                return corrected, decision, candidates
        except Exception as exc:
            logger.warning("Scope-lock override lookup failed: %s", exc)

    outcome = verify_lookup_decision(decision, candidates, question=question)
    return outcome, decision, candidates


def _table_in_scope(
    question: str,
    table_id: str,
    candidates: Sequence[Dict[str, Any]],
) -> bool:
    """True only for the top scoped candidate — no sibling near-ties."""
    tid = str(table_id or "").strip()
    if not tid or not candidates:
        return False
    return tid == str(candidates[0].get("table_id") or "")

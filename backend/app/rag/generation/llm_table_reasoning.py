"""LLM-driven, domain-agnostic table selection and evidence extraction.

Python handles candidate packaging, schema validation, and anti-hallucination
checks against supplied grids. The LLM decides which table/cells answer the
question. There is no regex domain lexicon and no legacy lookup fallback.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from app.config import get_settings
from app.rag.generation.table_evidence import (
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_NOT_FOUND,
    STATUS_SELECTOR_FAILED,
    STATUS_VERIFICATION_FAILED,
    InsufficientEvidence,
    VerifiedCell,
)
from app.rag.table_rag.ids import parse_table_payload
from app.rag.table_rag.representations import render_indexed_table_for_selector

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{[\s\S]*\}")
_TABLE_CUE_RE = re.compile(
    r"(?i)\b(table|chart|grid|matrix|lookup|row|column|cell|"
    r"schedule|rate|price|measurement|specification)\b"
)
_UNIT_NUM_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*"
    r"(?:[a-zA-Z%/μµ²³°]{1,12})\b"
)

TABLE_REASONING_SYSTEM_PROMPT = """You are a domain-agnostic table evidence analyst for retrieval-augmented generation.

You receive a user question and candidate document chunks that may contain tables.
Your job is to understand the supplied content, select the relevant table(s), extract exact evidence, and decide whether the evidence is sufficient.

Rules:
1. Use ONLY the supplied candidate chunks and tables. Never invent tables, rows, columns, values, units, or captions.
2. Do not choose a table merely because its caption shares keywords. Compare purpose, subject, applicability, units, footnotes, captions, and section_path using the supplied content.
3. Sibling matrices: when several tables share similar headers/axes but different applicability (population, period, category, grade, age band, indication, product, scope), you MUST use caption/purpose/applicability/section differences to pick the matching table. If the question names a use-case/population that appears in one caption but not others, that caption wins even if another sibling has a numerically convenient cell. If the question names a table number (e.g. "Table 1"), prefer the candidate whose caption starts with that number. If applicability still cannot be resolved, return status="ambiguous" with missing_information.
4. Prefer exact cell values from the provided [row_idx, col_idx] grids. Always include row_index and col_index for extracted cells. Preserve units and numeric text exactly. When matching a question's numeric axis factors to range headers (e.g. ">60 - 70", "≥30-100", ">200 - 300"), honor the inequality symbols and endpoints in the header text; do not choose a neighboring band when the boundary symbols already resolve the value.
5. When the question asks for a quantity AND a schedule/frequency, put the quantity cell first in selected_evidence and include the schedule cell second. Put the quantity in "answer".
6. When the question states an administered/prescribed amount AND also gives axis factors for a label/spec lookup (even if the user asks to classify deviation/error/off-label/special situation), extract the LABEL/spec cell for those axis factors (not the stated amount itself). Put that label cell in selected_evidence.value and status="answered". Never use the administered/prescribed amount as the row/column lookup key — it is only the comparator.
7. Perform arithmetic only when the question requires it and every operand is present as supplied evidence. Put the computed result in answer and list each operand in selected_evidence.
8. If the matching cell itself states that data is insufficient / not recommended / blank / do-not-use, return status="answered" with that exact cell text in selected_evidence and answer — that IS the evidence.
9. If evidence conflicts across tables, return status="ambiguous".
10. If no supplied table can answer, return status="insufficient_evidence" or "not_found".
11. Keep reasoning_summary short. Do not expose private chain-of-thought.
12. Citations must reference supplied chunk_id / table_id / page when available.
13. Prefer table content + footnotes + section_path over captions alone.
14. If evidence is incomplete, also set answer_ready=false and optionally propose short additional_search_queries for missing document facts (not chain-of-thought). When evidence is sufficient, set answer_ready=true.

Return JSON only (no markdown fences) matching:
{
  "status": "answered | ambiguous | insufficient_evidence | not_found",
  "answer_ready": true,
  "reasoning_summary": "Short explanation of table selection",
  "selected_chunks": [
    {
      "chunk_id": "source chunk identifier",
      "table_id": "source table identifier",
      "caption": "table caption",
      "relevance": "Why this table is relevant"
    }
  ],
  "selected_evidence": [
    {
      "chunk_id": "source chunk identifier",
      "table_id": "source table identifier",
      "row_index": 0,
      "col_index": 0,
      "row_reference": "Source row or row label",
      "column_reference": "Source column or header",
      "value": "Exact extracted value",
      "unit": "Unit if present",
      "source_context": "Supporting table content"
    }
  ],
  "missing_information": [],
  "additional_search_queries": [],
  "answer": "Evidence-grounded short answer value or conclusion",
  "citations": []
}
"""

_TABLE_NUM_RE = re.compile(r"(?i)\btable\s+(\d+[a-z]?)\b")
_VALUE_COL_RE = re.compile(
    r"(?i)\b(dose|amount|value|price|cost|rate|qty|quantity|total|limit|strength|result)\b"
)
_AXIS_COL_RE = re.compile(
    r"(?i)\b(freq|frequency|schedule|interval|period|category|class|group|band)\b"
)
_Q_UNIT_NUM_RE = re.compile(
    r"(?i)(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z%/μµ]{1,12})\b"
)
# Units that are grammatical/time/count words, not measurement axes.
_AXIS_UNIT_STOP = frozenset(
    {
        "for", "and", "the", "with", "from", "that", "this", "which", "what",
        "of", "in", "to", "or", "at", "by", "per", "on", "as", "an", "a",
        "year", "years", "yr", "yrs", "week", "weeks", "day", "days",
        "month", "months", "hour", "hours", "minute", "minutes",
        "patient", "patients", "table", "page", "pages", "age", "ages",
        "older", "old", "every", "times", "time", "dose", "doses",
    }
)
_RANGE_IN_HEADER_RE = re.compile(
    r"(?i)(?P<lo_op>≥|=>|>=|>)?\s*(?P<lo>\d+(?:\.\d+)?)"
    r"\s*[-–—to]+\s*"
    r"(?P<hi_op>≤|<=|<)?\s*(?P<hi>\d+(?:\.\d+)?)"
)
_STOP_TOKENS = frozenset(
    {
        "the", "and", "for", "with", "from", "what", "which", "that", "this",
        "patient", "patients", "table", "dose", "doses", "dosing", "every",
        "weeks", "week", "label", "body", "weight", "years", "year", "age",
        "older", "received", "prescribed", "intentionally", "classify",
        "about", "does", "show", "recommend", "or", "any", "was", "were",
        "mg", "kg", "iu",
    }
)


@dataclass
class TableInterpretationResult:
    is_structured_lookup: bool
    result: Optional[Union[VerifiedCell, InsufficientEvidence]] = None
    prompt_prefix: str = ""
    sources_update: Dict[str, Any] = field(default_factory=dict)
    reasoning_summary: str = ""
    raw_llm: Dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> Optional[VerifiedCell]:
        return self.result if isinstance(self.result, VerifiedCell) else None

    @property
    def insufficient(self) -> Optional[InsufficientEvidence]:
        return self.result if isinstance(self.result, InsufficientEvidence) else None


def is_structured_lookup_question(question: str) -> bool:
    """True when the question likely needs matrix evidence.

    Domain-agnostic: table language OR ≥2 unit-bearing numeric cues.
    """
    q = (question or "").strip()
    if not q:
        return False
    if _TABLE_CUE_RE.search(q):
        return True
    return len(_UNIT_NUM_RE.findall(q)) >= 2


def _parse_table(chunk: Dict[str, Any]) -> Dict[str, Any]:
    raw = chunk.get("table")
    if isinstance(raw, dict) and (raw.get("rows") is not None or raw.get("headers")):
        return dict(raw)
    payload = parse_table_payload(chunk.get("table_json") or raw)
    return payload.to_dict() if payload else {}


def _is_table_chunk(chunk: Dict[str, Any]) -> bool:
    chunk_type = str(
        chunk.get("chunk_type") or chunk.get("content_type") or ""
    ).lower()
    text = str(chunk.get("text") or "")
    return bool(
        chunk.get("table_json")
        or chunk.get("table")
        or chunk_type
        in {
            "table",
            "table_card",
            "table_row_window",
            "table_summary",
            "table_row",
        }
        or "table facts:" in text.lower()
        or "table card" in text.lower()
    )


def collect_table_candidates(
    chunks: Sequence[Dict[str, Any]],
    *,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """Dedupe hydrated table chunks by table_id; prefer fuller grids."""
    best: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    orphans: List[Dict[str, Any]] = []
    for chunk in chunks or []:
        if not _is_table_chunk(chunk):
            continue
        table = _parse_table(chunk)
        tid = str(chunk.get("table_id") or table.get("table_id") or "").strip()
        if not tid:
            orphans.append({"chunk": chunk, "table": table, "table_id": "", "score": float(chunk.get("score") or 0)})
            continue
        cur_rows = len(table.get("rows") or [])
        item = {
            "chunk": chunk,
            "table": table,
            "table_id": tid,
            "score": float(chunk.get("score") or 0),
        }
        if tid not in best:
            order.append(tid)
            best[tid] = item
            continue
        prev_rows = len((best[tid].get("table") or {}).get("rows") or [])
        if cur_rows > prev_rows or (
            cur_rows == prev_rows and chunk.get("table") and not best[tid]["chunk"].get("table")
        ):
            best[tid] = item
    ordered = [best[tid] for tid in order] + orphans
    ordered.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
    return ordered[: max(1, int(top_k or 6))]


def _token_set(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1}


def _soft_token_overlap(q_tokens: set, cap_tokens: set) -> set:
    """Exact overlap plus prefix matches (old~older), not arbitrary substrings."""
    out = set(q_tokens & cap_tokens)
    for qt in q_tokens:
        if len(qt) < 3:
            continue
        for ct in cap_tokens:
            if len(ct) < 3:
                continue
            # Prefix only (old→older, allerg→allergy); avoid old∈adult.
            if ct.startswith(qt) or qt.startswith(ct):
                out.add(qt)
                break
    return out


def _candidate_caption(item: Dict[str, Any]) -> str:
    table = item.get("table") or {}
    chunk = item.get("chunk") or {}
    return str(table.get("caption") or chunk.get("caption") or "")


def _candidate_section(item: Dict[str, Any]) -> str:
    table = item.get("table") or {}
    chunk = item.get("chunk") or {}
    return str(chunk.get("section") or table.get("section") or "")


def _rank_candidates_for_question(
    question: str, candidates: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Order candidates by caption/section overlap with the question (stable)."""
    q_tokens = _token_set(question)
    table_nums = {n.lower() for n in _TABLE_NUM_RE.findall(question or "")}

    def key(item: Dict[str, Any]) -> Tuple[int, int, float]:
        caption = _candidate_caption(item)
        section = _candidate_section(item)
        blob = f"{caption} {section}"
        overlap = len(q_tokens & _token_set(blob))
        num_hit = 0
        for num in table_nums:
            if re.search(rf"(?i)\btable\s+{re.escape(num)}\b", caption):
                num_hit = 5
                break
        return (num_hit, overlap, float(item.get("score") or 0))

    return sorted(candidates, key=key, reverse=True)


def _record_to_candidate(record: Dict[str, Any], *, score: float = 0.0) -> Dict[str, Any]:
    table = dict(record.get("table") or {})
    tid = str(record.get("table_id") or table.get("table_id") or "").strip()
    chunk = {
        "chunk_id": f"store:{record.get('document_id')}:{tid}",
        "id": f"store:{record.get('document_id')}:{tid}",
        "document_id": record.get("document_id"),
        "user_id": record.get("user_id"),
        "filename": record.get("filename") or "",
        "page": record.get("page_start") or table.get("page"),
        "page_start": record.get("page_start"),
        "section": record.get("section") or "",
        "caption": record.get("caption") or table.get("caption") or "",
        "chunk_type": "table",
        "table_id": tid,
        "table": table,
        "score": score,
        "text": str(record.get("summary_text") or "")[:1200],
    }
    if not table.get("caption") and chunk["caption"]:
        table["caption"] = chunk["caption"]
    if not table.get("page") and chunk.get("page") is not None:
        table["page"] = chunk["page"]
    if not table.get("section") and chunk.get("section"):
        table["section"] = chunk["section"]
    und = record.get("llm_understanding")
    if und and not table.get("llm_understanding"):
        table["llm_understanding"] = und
    return {"chunk": chunk, "table": table, "table_id": tid, "score": score}


def expand_sibling_and_referenced_tables(
    question: str,
    candidates: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]],
    *,
    max_total: int = 10,
) -> List[Dict[str, Any]]:
    """Add same-section siblings and explicitly referenced Table N matrices.

    Domain-agnostic: uses stored document tables + shared section_path / caption
    table numbers. Does not hardcode product or indication lexicons.
    """
    settings = get_settings()
    if not bool(getattr(settings, "TABLE_RAG_SIBLING_EXPAND", True)):
        return list(candidates)

    out: List[Dict[str, Any]] = [dict(c) for c in (candidates or [])]
    seen = {str(c.get("table_id") or "") for c in out if c.get("table_id")}
    sections = {
        _candidate_section(c).strip().lower()
        for c in out
        if _candidate_section(c).strip()
    }
    table_nums = {n.lower() for n in _TABLE_NUM_RE.findall(question or "")}

    doc_ids: List[str] = []
    user_id: Optional[str] = None
    for src in list(chunks or []) + [c.get("chunk") or {} for c in out]:
        did = str(src.get("document_id") or "").strip()
        if did and did not in doc_ids:
            doc_ids.append(did)
        if not user_id and src.get("user_id"):
            user_id = str(src.get("user_id"))

    if not doc_ids:
        return out[: max(1, int(max_total or 10))]

    try:
        from app.rag.table_rag.store import list_document_tables
    except Exception as exc:
        logger.debug("Sibling expand unavailable: %s", exc)
        return out[: max(1, int(max_total or 10))]

    extras: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        try:
            records = list_document_tables(document_id=doc_id, user_id=user_id)
        except Exception as exc:
            logger.debug("list_document_tables(%s) failed: %s", doc_id, exc)
            continue
        for rec in records:
            tid = str(rec.get("table_id") or "").strip()
            if not tid or tid in seen:
                continue
            caption = str(rec.get("caption") or (rec.get("table") or {}).get("caption") or "")
            section = str(rec.get("section") or "").strip().lower()
            num_hit = any(
                re.search(rf"(?i)\btable\s+{re.escape(num)}\b", caption)
                for num in table_nums
            )
            section_hit = bool(section and section in sections)
            if not (num_hit or section_hit):
                continue
            score = 0.04 if section_hit else 0.03
            if num_hit:
                score = 0.08
            extras.append(_record_to_candidate(rec, score=score))
            seen.add(tid)

    # Prefer question-referenced / overlapping extras first, then original order.
    extras = _rank_candidates_for_question(question, extras)
    merged = out + extras
    merged = _rank_candidates_for_question(question, merged)
    return merged[: max(len(out), int(max_total or 10))]


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


def _cell_at(table: Dict[str, Any], row_index: Any, col_index: Any) -> Optional[str]:
    try:
        r = int(row_index)
        c = int(col_index)
    except (TypeError, ValueError):
        return None
    rows = table.get("rows") or []
    if r < 0 or r >= len(rows):
        return None
    row = rows[r]
    if not isinstance(row, (list, tuple)) or c < 0 or c >= len(row):
        return None
    return str(row[c] if row[c] is not None else "").strip()


def _headers_for(table: Dict[str, Any], row_index: int, col_index: int) -> Tuple[List[str], List[str]]:
    headers = [str(h or "") for h in (table.get("headers") or [])]
    col_headers = [headers[col_index]] if 0 <= col_index < len(headers) else []
    rows = table.get("rows") or []
    row_headers: List[str] = []
    if 0 <= row_index < len(rows):
        row = rows[row_index]
        if isinstance(row, (list, tuple)) and row:
            row_headers = [str(row[0] or "")]
    return row_headers, col_headers


def _normalize_value(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _find_value_coords(
    table: Dict[str, Any], value: str
) -> Optional[Tuple[int, int, str]]:
    """Locate a value in the grid; prefer exact normalized equality."""
    target = _normalize_value(value)
    if not target:
        return None
    exact: Optional[Tuple[int, int, str]] = None
    partial: Optional[Tuple[int, int, str]] = None
    for r_i, row in enumerate(table.get("rows") or []):
        if not isinstance(row, (list, tuple)):
            continue
        for c_i, cell in enumerate(row):
            cell_raw = str(cell if cell is not None else "").strip()
            cell_s = _normalize_value(cell_raw)
            if not cell_s:
                continue
            if target == cell_s:
                return r_i, c_i, cell_raw
            if exact is None and (target in cell_s or cell_s in target):
                partial = (r_i, c_i, cell_raw)
    return exact or partial


def _value_in_table(table: Dict[str, Any], value: str) -> bool:
    return _find_value_coords(table, value) is not None


def _candidate_index(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        tid = str(item.get("table_id") or "").strip()
        if tid:
            out[tid] = item
        chunk = item.get("chunk") or {}
        cid = str(chunk.get("chunk_id") or chunk.get("id") or "").strip()
        if cid and cid not in out:
            out[cid] = item
    return out


def _parse_range_bounds(header: str) -> Optional[Tuple[float, float, bool, bool]]:
    """Return (lo, hi, lo_inclusive, hi_inclusive) from a range header, if any."""
    match = _RANGE_IN_HEADER_RE.search(header or "")
    if not match:
        return None
    lo = float(match.group("lo"))
    hi = float(match.group("hi"))
    lo_op = (match.group("lo_op") or "").strip()
    hi_op = (match.group("hi_op") or "").strip()
    # Bare "a - b" usually means up to b inclusive. Treat "=>" like ">" —
    # OCR often emits "=>" for a exclusive-lower bound that continues the
    # previous band (e.g. prior row ended at 300, next starts >300).
    lo_inclusive = lo_op in {"", "≥", ">="}
    hi_inclusive = hi_op not in {"<", "≤", "<="} and hi_op != "<"
    if lo_op in {">", "=>"}:
        lo_inclusive = False
    return lo, hi, lo_inclusive, hi_inclusive


def _range_covers(header: str, number: float) -> Optional[bool]:
    bounds = _parse_range_bounds(header)
    if not bounds:
        return None
    lo, hi, lo_incl, hi_incl = bounds
    if number < lo or (number == lo and not lo_incl):
        return False
    if number > hi or (number == hi and not hi_incl):
        return False
    return True


def _is_unitless_range_header(header: str) -> bool:
    """True when a header is a numeric range with no alphabetic unit tokens."""
    if not _parse_range_bounds(header or ""):
        return False
    return re.search(r"[A-Za-z]", header or "") is None


def _question_unit_values(question: str) -> List[Tuple[float, str]]:
    """Extract measurement-like ``number + unit`` factors from the question.

    Skips grammatical pairings (``1 for``, ``12 years``) so axis verification
    does not treat narrative counts as table-band keys.
    """
    out: List[Tuple[float, str]] = []
    for match in _Q_UNIT_NUM_RE.finditer(question or ""):
        try:
            num = float(match.group("num"))
        except ValueError:
            continue
        unit = match.group("unit").lower()
        unit_root = unit.split("/")[0]
        if unit in _AXIS_UNIT_STOP or unit_root in _AXIS_UNIT_STOP:
            continue
        out.append((num, unit))
    return out


def _table_axis_headers(table: Dict[str, Any]) -> List[str]:
    headers = [str(h or "") for h in (table.get("headers") or []) if h]
    body = table.get("rows") or table.get("body") or []
    for row in body[:40]:
        if isinstance(row, (list, tuple)) and row:
            headers.append(str(row[0] or ""))
    return headers


def _missing_table_axis_unit(
    question: str,
    *,
    selected_headers: Sequence[str],
    table: Dict[str, Any],
) -> Optional[str]:
    """Reject cells that omit a unit axis present both in the question and table."""
    factors = _question_unit_values(question)
    if not factors or not table:
        return None
    selected = [str(h) for h in selected_headers if h]
    table_headers = _table_axis_headers(table)
    missing: List[str] = []
    for _number, unit in factors:
        table_has = any(
            unit in h.lower() and _parse_range_bounds(h) for h in table_headers
        )
        if not table_has:
            continue
        if not any(unit in h.lower() for h in selected):
            missing.append(unit)
    if not missing:
        return None
    seen: set = set()
    uniq: List[str] = []
    for unit in missing:
        if unit in seen:
            continue
        seen.add(unit)
        uniq.append(unit)
    return f"selected cell omits table axis unit(s) {uniq}"


def _axis_hard_conflict(
    question: str,
    *,
    row_headers: Sequence[str],
    col_headers: Sequence[str],
    table: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return a conflict note when selected range headers exclude question axis values.

    Domain-agnostic: uses only inequality/endpoint math on the supplied headers.
    Unit-bearing headers are matched by unit token; remaining factors are checked
    against unitless numeric range headers (typical row bands).
    """
    factors = _question_unit_values(question)
    if not factors:
        return None
    headers = [str(h) for h in list(row_headers or []) + list(col_headers or []) if h]
    if not headers:
        return None

    conflicts: List[str] = []
    if table is not None:
        omitted = _missing_table_axis_unit(
            question, selected_headers=headers, table=table
        )
        if omitted:
            conflicts.append(omitted)

    remaining: List[Tuple[float, str]] = []
    used: set = set()

    for number, unit in factors:
        unit_headers = [
            h
            for h in headers
            if unit in h.lower() and _parse_range_bounds(h)
        ]
        if unit_headers:
            if any(_range_covers(h, number) is True for h in unit_headers):
                for h in unit_headers:
                    if _range_covers(h, number) is True:
                        used.add(h)
                        break
            elif any(_range_covers(h, number) is False for h in unit_headers):
                conflicts.append(
                    f"{number:g} {unit} excluded by selected header(s) "
                    f"{unit_headers}"
                )
            continue
        remaining.append((number, unit))

    unitless = [
        h
        for h in headers
        if h not in used and _is_unitless_range_header(h)
    ]
    # Unitless row/axis bands (e.g. ">300-400") must cover at least one
    # remaining measurement factor. Extra value-like amounts (prescribed mg,
    # prices) that do not fit the band are ignored when another factor does.
    if unitless and remaining:
        covered = [
            (number, unit)
            for number, unit in remaining
            if any(_range_covers(h, number) is True for h in unitless)
        ]
        if not covered:
            excluded = [
                f"{number:g} {unit}"
                for number, unit in remaining
                if any(_range_covers(h, number) is False for h in unitless)
            ]
            if excluded:
                conflicts.append(
                    f"unitless range header(s) {unitless} exclude remaining "
                    f"axis factor(s) {excluded}"
                )

    return "; ".join(conflicts) if conflicts else None


def _axis_consistency_bonus(
    question: str,
    *,
    row_headers: Sequence[str],
    col_headers: Sequence[str],
) -> int:
    """Reward headers that cover the question's unit-bearing numbers; penalize misses."""
    if _axis_hard_conflict(
        question, row_headers=row_headers, col_headers=col_headers
    ):
        return -120
    factors = _question_unit_values(question)
    if not factors:
        return 0
    headers = [str(h) for h in list(row_headers or []) + list(col_headers or []) if h]
    if not headers:
        return 0
    bonus = 0
    for number, unit in factors:
        unit_headers = [
            h for h in headers if unit in h.lower() and _parse_range_bounds(h)
        ]
        if not unit_headers:
            unit_headers = [h for h in headers if _is_unitless_range_header(h)]
        matched = False
        conflict = False
        for header in unit_headers:
            covers = _range_covers(header, number)
            if covers is True:
                matched = True
            elif covers is False:
                conflict = True
        if matched:
            bonus += 40
        elif conflict:
            bonus -= 80
    return bonus


def _axis_coverage_hints(
    question: str, candidates: Sequence[Dict[str, Any]]
) -> str:
    """List which supplied range headers cover each unit-bearing question factor."""
    factors = _question_unit_values(question)
    if not factors or not candidates:
        return ""
    lines: List[str] = []
    for item in candidates[:8]:
        table = item.get("table") or {}
        tid = str(item.get("table_id") or table.get("table_id") or "")
        headers = [str(h or "") for h in (table.get("headers") or []) if h]
        # Include first-column row labels (common axis bands).
        body = table.get("rows") or table.get("body") or []
        row_labels: List[str] = []
        for row in body[:24]:
            if isinstance(row, (list, tuple)) and row:
                row_labels.append(str(row[0] or ""))
        axis_headers = headers + row_labels
        per_factor: List[str] = []
        for number, unit in factors:
            covering = [
                h
                for h in axis_headers
                if _parse_range_bounds(h) and _range_covers(h, number) is True
            ]
            unit_cover = [h for h in covering if unit in h.lower()]
            unitless_cover = [h for h in covering if _is_unitless_range_header(h)]
            chosen = unit_cover or unitless_cover or covering
            seen: set = set()
            chosen_u: List[str] = []
            for h in chosen:
                if h in seen:
                    continue
                seen.add(h)
                chosen_u.append(h)
            per_factor.append(
                f"{number:g} {unit} → {chosen_u[:6] or ['(no covering range header)']}"
            )
        if per_factor:
            lines.append(f"- {tid}: " + "; ".join(per_factor))
    if not lines:
        return ""
    return (
        "Axis coverage (honor inequality symbols; pick headers that cover "
        "each question factor):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def _caption_overlap_hints(
    question: str, candidates: Sequence[Dict[str, Any]]
) -> str:
    q_tokens = _token_set(question) - _STOP_TOKENS
    lines: List[str] = []
    if q_tokens:
        for item in candidates:
            caption = _candidate_caption(item)
            overlap = sorted(q_tokens & _token_set(caption))
            if overlap:
                lines.append(
                    f"- {item.get('table_id')}: caption overlap tokens={overlap}"
                )
    sibling_lines: List[str] = []
    # Flag sibling-looking matrices (similar headers, different captions).
    header_sets: List[Tuple[str, set, str]] = []
    for item in candidates:
        table = item.get("table") or {}
        headers = [str(h or "").lower() for h in (table.get("headers") or []) if h]
        if len(headers) < 2:
            continue
        tokens: set = set()
        for h in headers:
            tokens |= _token_set(h)
        header_sets.append(
            (str(item.get("table_id") or ""), tokens, _candidate_caption(item))
        )
    for i in range(len(header_sets)):
        for j in range(i + 1, len(header_sets)):
            tid_a, tok_a, cap_a = header_sets[i]
            tid_b, tok_b, cap_b = header_sets[j]
            if not tok_a or not tok_b:
                continue
            inter = len(tok_a & tok_b)
            union = len(tok_a | tok_b) or 1
            if inter / union < 0.45:
                continue
            cap_diff = sorted(
                (_token_set(cap_a) | _token_set(cap_b))
                - (_token_set(cap_a) & _token_set(cap_b))
                - _STOP_TOKENS
            )
            if not cap_diff:
                continue
            sibling_lines.append(
                f"- {tid_a} vs {tid_b}: similar axes; differing caption tokens={cap_diff[:12]}"
            )
    parts: List[str] = []
    if lines:
        parts.append(
            "Caption/question overlap hints (prefer compatible captions):\n"
            + "\n".join(lines)
        )
    if sibling_lines:
        parts.append(
            "Sibling-matrix warning (choose by caption/applicability, not numeric convenience):\n"
            + "\n".join(sibling_lines[:8])
        )
    primary = _primary_caption_matches(question, candidates)
    if primary:
        best = primary[0]
        parts.insert(
            0,
            "Primary caption match (prefer unless evidence is missing): "
            f"{best.get('table_id')} tokens={sorted((_token_set(question) - _STOP_TOKENS) & _token_set(_candidate_caption(best)))} "
            f"caption={_candidate_caption(best)[:140]}",
        )
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


def _header_token_set(item: Dict[str, Any]) -> set:
    table = item.get("table") or {}
    tokens: set = set()
    for header in table.get("headers") or []:
        tokens |= _token_set(str(header or ""))
    return tokens


def _sibling_clusters(
    candidates: Sequence[Dict[str, Any]],
    *,
    jaccard_min: float = 0.45,
) -> List[List[Dict[str, Any]]]:
    """Group candidates with similar header axes (likely sibling matrices)."""
    items = [c for c in candidates if _header_token_set(c)]
    clusters: List[List[Dict[str, Any]]] = []
    used = set()
    for i, item in enumerate(items):
        tid = str(item.get("table_id") or i)
        if tid in used:
            continue
        cluster = [item]
        used.add(tid)
        tok_i = _header_token_set(item)
        for j in range(i + 1, len(items)):
            other = items[j]
            oid = str(other.get("table_id") or j)
            if oid in used:
                continue
            tok_j = _header_token_set(other)
            union = len(tok_i | tok_j) or 1
            if len(tok_i & tok_j) / union >= jaccard_min:
                cluster.append(other)
                used.add(oid)
        clusters.append(cluster)
    return clusters


def _primary_caption_matches(
    question: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    min_overlap: float = 1,
) -> List[Dict[str, Any]]:
    """Prefer caption-compatible siblings; drop incompatible peers in a cluster.

    Within a similar-header sibling cluster, if the question overlaps some
    captions and not others, keep only the overlapping peers. This is
    domain-agnostic disambiguation of related matrices.

    Explicit ``Table N`` references in the question always keep matching captions.
    """
    q_tokens = _token_set(question) - _STOP_TOKENS
    if not q_tokens or not candidates:
        return []

    table_nums = {n.lower() for n in _TABLE_NUM_RE.findall(question or "")}
    referenced: List[Dict[str, Any]] = []
    if table_nums:
        for item in candidates:
            caption = _candidate_caption(item)
            if any(
                re.search(rf"(?i)\btable\s+{re.escape(num)}\b", caption)
                for num in table_nums
            ):
                referenced.append(item)
        if referenced:
            return _rank_candidates_for_question(question, referenced)

    keep_ids: set = set()
    dropped_ids: set = set()
    clustered_ids: set = set()
    for cluster in _sibling_clusters(candidates):
        for item in cluster:
            clustered_ids.add(str(item.get("table_id") or ""))
        scored: List[Tuple[int, set, set, Dict[str, Any]]] = []
        for item in cluster:
            cap_tokens = _token_set(_candidate_caption(item))
            exact = q_tokens & cap_tokens
            soft = _soft_token_overlap(q_tokens, cap_tokens)
            scored.append((len(soft), exact, soft, item))
        positive = [
            (exact, soft, item)
            for score, exact, soft, item in scored
            if score >= min_overlap
        ]
        if positive and len(positive) < len(cluster):
            # Modal token from EXACT overlaps only so soft matches (old~older)
            # do not revive incompatible siblings.
            token_hits: Dict[str, int] = {}
            for exact, _soft, _item in positive:
                for tok in exact:
                    token_hits[tok] = token_hits.get(tok, 0) + 1
            modal_tok = ""
            modal_n = 0
            for tok, count in token_hits.items():
                if count > modal_n:
                    modal_tok, modal_n = tok, count
            if modal_tok and modal_n >= 2:
                positive = [
                    (exact, soft, item)
                    for exact, soft, item in positive
                    if modal_tok in exact
                ]
            kept_now = {str(item.get("table_id") or "") for _e, _s, item in positive}
            for _exact, _soft, item in positive:
                keep_ids.add(str(item.get("table_id") or ""))
            for item in cluster:
                tid = str(item.get("table_id") or "")
                if tid not in kept_now:
                    dropped_ids.add(tid)
        else:
            for item in cluster:
                keep_ids.add(str(item.get("table_id") or ""))

    # Keep non-clustered tables that still overlap the question.
    for item in candidates:
        tid = str(item.get("table_id") or "")
        if tid in clustered_ids or tid in dropped_ids or tid in keep_ids:
            continue
        if q_tokens & _token_set(_candidate_caption(item)):
            keep_ids.add(tid)

    focused = [
        item
        for item in candidates
        if str(item.get("table_id") or "") in keep_ids
        and str(item.get("table_id") or "") not in dropped_ids
        and _soft_token_overlap(q_tokens, _token_set(_candidate_caption(item)))
    ]
    if not focused:
        return []

    # Prefer peers with richer soft caption overlap (e.g. old~older vs pediatric-only).
    def _focus_key(item: Dict[str, Any]) -> Tuple[int, int, float]:
        soft = _soft_token_overlap(q_tokens, _token_set(_candidate_caption(item)))
        exact = q_tokens & _token_set(_candidate_caption(item))
        return (len(soft), len(exact), float(item.get("score") or 0))

    focused = sorted(focused, key=_focus_key, reverse=True)
    best_soft = _focus_key(focused[0])[0]
    if best_soft > 1:
        focused = [c for c in focused if _focus_key(c)[0] >= best_soft]
    return focused


def _score_verified_evidence(
    ev: Dict[str, Any],
    value: str,
    *,
    payload: Dict[str, Any],
    col_headers: Sequence[str],
    row_headers: Sequence[str],
    question: str = "",
    table_caption: str = "",
) -> int:
    """Rank verified evidence items; prefer answer-aligned value cells."""
    score = 0
    q = question or str(payload.get("question") or "")
    answer = _normalize_value(str(payload.get("answer") or ""))
    norm_val = _normalize_value(value)
    if norm_val and answer and (norm_val in answer or answer in norm_val):
        score += 100
    col_blob = " ".join(
        [
            str(ev.get("column_reference") or ""),
            *[str(h) for h in (col_headers or [])],
        ]
    )
    if _VALUE_COL_RE.search(col_blob):
        score += 25
    if _AXIS_COL_RE.search(col_blob) and not _VALUE_COL_RE.search(col_blob):
        score -= 15
    if re.search(r"\d", value or ""):
        score += 5
    # Prefer quantity-bearing cells over pure schedule text when both exist.
    if re.search(r"(?i)\bevery\b", value or "") and not re.search(r"\d", value or ""):
        score -= 10
    score += _axis_consistency_bonus(
        q,
        row_headers=row_headers,
        col_headers=list(col_headers or [])
        + [str(ev.get("column_reference") or ""), str(ev.get("row_reference") or "")],
    )
    # Prefer tables whose caption shares non-stop tokens with the question.
    overlap = (_token_set(q) - _STOP_TOKENS) & _token_set(table_caption)
    score += 25 * len(overlap)
    return score


def _same_row_companions(
    table: Dict[str, Any],
    row_index: int,
    col_index: int,
) -> List[Dict[str, str]]:
    """Collect non-empty same-row cells that look like schedule/axis companions."""
    headers = [str(h or "") for h in (table.get("headers") or [])]
    rows = table.get("rows") or []
    if not (0 <= row_index < len(rows)):
        return []
    row = rows[row_index]
    if not isinstance(row, (list, tuple)):
        return []
    out: List[Dict[str, str]] = []
    for c_i, cell in enumerate(row):
        if c_i == col_index:
            continue
        text = str(cell if cell is not None else "").strip()
        if not text:
            continue
        header = headers[c_i] if c_i < len(headers) else f"col_{c_i}"
        blob = f"{header} {text}"
        if _AXIS_COL_RE.search(blob) or re.search(
            r"(?i)\b(every|weekly|daily|monthly|quarterly|annually)\b", text
        ):
            out.append({"header": header, "value": text})
    return out[:4]


def _build_verified_cell(
    ev: Dict[str, Any],
    *,
    item: Dict[str, Any],
    value: str,
    row_index: int,
    col_index: int,
    payload: Dict[str, Any],
) -> VerifiedCell:
    table = item.get("table") or {}
    chunk = item.get("chunk") or {}
    chunk_id = str(ev.get("chunk_id") or "")
    table_id = str(ev.get("table_id") or table.get("table_id") or "")
    row_headers, col_headers = _headers_for(table, max(row_index, 0), max(col_index, 0))
    if ev.get("row_reference"):
        row_headers = [str(ev.get("row_reference"))]
    if ev.get("column_reference"):
        col_headers = [str(ev.get("column_reference"))]
    page = (
        table.get("page")
        or table.get("page_start")
        or chunk.get("page")
        or chunk.get("page_start")
    )
    try:
        page_i = int(page) if page is not None else None
    except (TypeError, ValueError):
        page_i = None
    companions = _same_row_companions(table, row_index, col_index)
    return VerifiedCell(
        value=value,
        row_headers=row_headers,
        col_headers=col_headers,
        row_index=row_index,
        col_index=col_index,
        page=page_i,
        filename=str(chunk.get("filename") or chunk.get("source") or ""),
        confidence=0.0,
        table_headers=[str(h or "") for h in (table.get("headers") or [])],
        evidence={
            "table_caption": str(table.get("caption") or chunk.get("caption") or ""),
            "table_section": str(chunk.get("section") or ""),
            "chunk_id": str(chunk.get("chunk_id") or chunk.get("id") or chunk_id),
            "original_cell_text": value,
            "verification": "llm_select_python_verify",
            "reasoning_summary": str(payload.get("reasoning_summary") or ""),
            "unit": str(ev.get("unit") or ""),
            "source_context": str(ev.get("source_context") or ""),
            "same_row_companions": companions,
        },
        status=STATUS_MATCHED,
        table_id=table_id,
        chunk_id=str(chunk.get("chunk_id") or chunk.get("id") or chunk_id),
    )


def _try_verify_evidence_item(
    ev: Dict[str, Any],
    *,
    by_id: Dict[str, Dict[str, Any]],
    payload: Dict[str, Any],
    question: str = "",
    candidates: Sequence[Dict[str, Any]] = (),
) -> Optional[Tuple[int, VerifiedCell]]:
    table_id = str(ev.get("table_id") or "").strip()
    chunk_id = str(ev.get("chunk_id") or "").strip()
    item = by_id.get(table_id) or by_id.get(chunk_id)
    if not item:
        return None
    _ = candidates
    table = item.get("table") or {}
    value = str(ev.get("value") or "").strip()
    if not value:
        return None

    row_index = ev.get("row_index")
    col_index = ev.get("col_index")
    cell = _cell_at(table, row_index, col_index)
    if cell is not None and (
        _normalize_value(value) == _normalize_value(cell)
        or _normalize_value(value) in _normalize_value(cell)
        or _normalize_value(cell) in _normalize_value(value)
    ):
        value = cell
        r_i = int(row_index)
        c_i = int(col_index)
    else:
        found = _find_value_coords(table, value)
        if not found:
            return None
        r_i, c_i, value = found

    # Axis checks use grid headers from coordinates (authoritative), not only
    # the LLM's free-text row/column_reference which can misstate the band.
    grid_row_headers, grid_col_headers = _headers_for(table, r_i, c_i)
    conflict = _axis_hard_conflict(
        question,
        row_headers=grid_row_headers,
        col_headers=grid_col_headers,
        table=table,
    )
    if conflict:
        logger.info(
            "Rejecting table evidence for axis conflict table=%s value=%s "
            "row=%s col=%s detail=%s",
            table_id or (item.get("table_id")),
            value,
            grid_row_headers,
            grid_col_headers,
            conflict,
        )
        return None

    verified = _build_verified_cell(
        ev,
        item=item,
        value=value,
        row_index=r_i,
        col_index=c_i,
        payload=payload,
    )
    caption = str(
        (item.get("table") or {}).get("caption")
        or (item.get("chunk") or {}).get("caption")
        or ""
    )
    rank = _score_verified_evidence(
        ev,
        value,
        payload=payload,
        col_headers=verified.col_headers,
        row_headers=verified.row_headers,
        question=question,
        table_caption=caption,
    )
    return rank, verified


def verify_llm_evidence(
    payload: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    *,
    question: str = "",
) -> Union[VerifiedCell, InsufficientEvidence]:
    """Validate LLM output against supplied tables; fail closed on invention."""
    status = str(payload.get("status") or "").strip().lower()
    missing = [str(x) for x in (payload.get("missing_information") or []) if x]
    selected = list(payload.get("selected_evidence") or [])
    by_id = _candidate_index(candidates)
    candidate_table_ids = [
        str(item.get("table_id") or "")
        for item in candidates
        if item.get("table_id")
    ]
    q = question or str(payload.get("question") or "")

    if status in {"ambiguous"}:
        values = [
            str(ev.get("value") or "")
            for ev in selected
            if str(ev.get("value") or "").strip()
        ]
        return InsufficientEvidence(
            reason=str(payload.get("reasoning_summary") or "Ambiguous table evidence"),
            status=STATUS_AMBIGUOUS,
            soft=False,
            candidate_values=values,
            candidate_table_ids=candidate_table_ids,
            missing_dimensions=missing,
        )

    verified_ranked: List[Tuple[int, VerifiedCell]] = []
    for ev in selected:
        got = _try_verify_evidence_item(
            ev,
            by_id=by_id,
            payload=payload,
            question=q,
            candidates=candidates,
        )
        if got is not None:
            verified_ranked.append(got)

    if verified_ranked:
        verified_ranked.sort(key=lambda pair: pair[0], reverse=True)
        # Prefer a value-column cell over a schedule/axis-only cell when both verify.
        value_cells = [
            cell
            for _rank, cell in verified_ranked
            if _VALUE_COL_RE.search(
                " ".join(str(h) for h in (cell.col_headers or []))
            )
        ]
        if value_cells:
            return value_cells[0]
        return verified_ranked[0][1]

    if status in {"insufficient_evidence", "not_found"} or not selected:
        return InsufficientEvidence(
            reason=str(
                payload.get("reasoning_summary")
                or payload.get("answer")
                or "No verified table evidence"
            ),
            status=STATUS_NOT_FOUND,
            soft=False,
            candidate_values=[],
            candidate_table_ids=candidate_table_ids,
            missing_dimensions=missing,
        )

    return InsufficientEvidence(
        reason="LLM evidence failed Python verification against supplied tables",
        status=STATUS_VERIFICATION_FAILED,
        # Soft: allow answer-generation prose for classification / applicability
        # questions when the selector could not pin a unique cell. Hard INVALID
        # abstention over-blocked previously-passing PV cases.
        soft=True,
        candidate_values=[
            str(ev.get("value") or "")
            for ev in selected
            if str(ev.get("value") or "").strip()
        ],
        candidate_table_ids=candidate_table_ids,
        missing_dimensions=missing,
    )


def reason_over_tables(
    question: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    client: Any = None,
    settings: Any = None,
    feedback: str = "",
) -> Dict[str, Any]:
    """Call the LLM for structured table selection/extraction."""
    settings = settings or get_settings()
    if not candidates:
        return {
            "status": "not_found",
            "reasoning_summary": "no candidate tables",
            "selected_chunks": [],
            "selected_evidence": [],
            "missing_information": [],
            "answer": "",
            "citations": [],
        }

    ranked = _rank_candidates_for_question(question, candidates)
    blocks: List[str] = []
    for item in ranked:
        table = item.get("table") or {}
        chunk = item.get("chunk") or {}
        record = {
            "table_id": item.get("table_id") or table.get("table_id"),
            "table": table,
            "pages": table.get("pages")
            or chunk.get("pages")
            or [table.get("page") or chunk.get("page")],
            "section": chunk.get("section") or table.get("section") or "",
            "caption": table.get("caption") or chunk.get("caption") or "",
            "chunk_id": chunk.get("chunk_id") or chunk.get("id") or "",
        }
        rendered = render_indexed_table_for_selector(record)
        chunk_id = str(record.get("chunk_id") or "")
        if chunk_id:
            rendered = f"chunk_id: {chunk_id}\n{rendered}"
        # Surrounding prose / footnotes from the retrieved chunk when present.
        preview = str(chunk.get("text") or "").strip()
        if preview:
            preview_cap = 800 if len(ranked) >= 6 else 1200
            rendered = (
                f"{rendered}\n\nchunk_text_preview:\n{preview[:preview_cap]}"
            )
        blocks.append(rendered)

    overlap = _caption_overlap_hints(question, ranked)
    axis_hints = _axis_coverage_hints(question, ranked)
    feedback_block = (
        f"Prior selection rejected:\n{feedback.strip()}\n\n" if feedback.strip() else ""
    )
    user_prompt = (
        f"User question:\n{(question or '').strip()}\n\n"
        "Compare sibling tables using caption, purpose, applicability, "
        "section_path, and footnotes before extracting a cell. "
        "Honor range-boundary symbols when matching axis factors. "
        "Never select a range header that excludes a question axis value.\n\n"
        f"{feedback_block}"
        f"{overlap}"
        f"{axis_hints}"
        f"Candidate tables ({len(blocks)}):\n\n"
        + "\n\n-----\n\n".join(blocks)
    )

    if client is None:
        try:
            from app.rag.llm.agent import get_llm_client

            client = get_llm_client()
        except Exception as exc:
            logger.warning("Table reasoning LLM unavailable: %s", exc)
            return {
                "status": "insufficient_evidence",
                "reasoning_summary": f"selector_unavailable: {exc}",
                "selected_chunks": [],
                "selected_evidence": [],
                "missing_information": [],
                "additional_search_queries": [],
                "answer_ready": False,
                "answer": "",
                "citations": [],
            }

    try:
        from app.rag.llm.agent import llm_model_for_provider

        model = llm_model_for_provider()
    except Exception:
        model = getattr(settings, "LLM_MODEL", None)

    def _call_selector(prompt: str, *, max_tokens: int) -> Dict[str, Any]:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": TABLE_REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
        )
        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = ""
        parsed_local = _extract_json(content)
        return parsed_local if isinstance(parsed_local, dict) else {}

    started = time.perf_counter()
    try:
        parsed = _call_selector(user_prompt, max_tokens=1200)
        # One compact retry when the first response is truncated/non-JSON.
        if not parsed:
            compact_blocks = []
            for item in ranked[:6]:
                table = item.get("table") or {}
                chunk = item.get("chunk") or {}
                record = {
                    "table_id": item.get("table_id") or table.get("table_id"),
                    "table": table,
                    "pages": table.get("pages")
                    or chunk.get("pages")
                    or [table.get("page") or chunk.get("page")],
                    "section": chunk.get("section") or table.get("section") or "",
                    "caption": table.get("caption") or chunk.get("caption") or "",
                    "chunk_id": chunk.get("chunk_id") or chunk.get("id") or "",
                }
                rendered = render_indexed_table_for_selector(record)
                compact_blocks.append(rendered)
            compact_prompt = (
                f"User question:\n{(question or '').strip()}\n\n"
                "Return JSON only. Prefer exact [row_idx,col_idx] cells.\n\n"
                f"Candidate tables ({len(compact_blocks)}):\n\n"
                + "\n\n-----\n\n".join(compact_blocks)
            )
            parsed = _call_selector(compact_prompt, max_tokens=1200)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not parsed:
            parsed = {
                "status": "insufficient_evidence",
                "reasoning_summary": "selector_parse_failed",
                "selected_chunks": [],
                "selected_evidence": [],
                "missing_information": [],
                "additional_search_queries": [],
                "answer_ready": False,
                "answer": "",
                "citations": [],
            }
        try:
            from app.rag.table_rag.telemetry import record_selector_metrics

            record_selector_metrics(
                status=str(parsed.get("status") or ""),
                latency_ms=latency_ms,
                candidate_table_ids=[
                    str(item.get("table_id") or "")
                    for item in ranked
                    if item.get("table_id")
                ],
            )
        except Exception:
            pass
        return parsed
    except Exception as exc:
        logger.warning("Table reasoning LLM call failed: %s", exc)
        return {
            "status": "insufficient_evidence",
            "reasoning_summary": f"selector_error: {exc}",
            "selected_chunks": [],
            "selected_evidence": [],
            "missing_information": [],
            "additional_search_queries": [],
            "answer_ready": False,
            "answer": "",
            "citations": [],
        }


def _result_from_outcome(
    outcome: Union[VerifiedCell, InsufficientEvidence],
    *,
    reasoning_summary: str = "",
    raw_llm: Optional[Dict[str, Any]] = None,
) -> TableInterpretationResult:
    if isinstance(outcome, VerifiedCell):
        return TableInterpretationResult(
            is_structured_lookup=True,
            result=outcome,
            prompt_prefix=outcome.to_prompt_block() + "\n\n",
            sources_update={"verified_cell": outcome.to_dict()},
            reasoning_summary=reasoning_summary,
            raw_llm=raw_llm or {},
        )
    update = {
        "insufficient_table_evidence": True,
        "status": outcome.status,
        "soft_insufficient": bool(outcome.soft),
        "reason": outcome.reason,
        "candidate_values": list(outcome.candidate_values or []),
        "candidate_table_ids": list(outcome.candidate_table_ids or []),
        "missing_dimensions": list(outcome.missing_dimensions or []),
    }
    return TableInterpretationResult(
        is_structured_lookup=True,
        result=outcome,
        prompt_prefix=outcome.to_prompt_block() + "\n\n",
        sources_update=update,
        reasoning_summary=reasoning_summary,
        raw_llm=raw_llm or {},
    )


def interpret_table_question(
    question: str,
    chunks: List[Dict[str, Any]],
    *,
    client: Any = None,
    settings: Any = None,
) -> TableInterpretationResult:
    """Answer-time table understanding entrypoint used by the agent."""
    settings = settings or get_settings()
    q = (question or "").strip()
    table_chunks_present = any(_is_table_chunk(c) for c in (chunks or []))
    if not is_structured_lookup_question(q) and not table_chunks_present:
        return TableInterpretationResult(is_structured_lookup=False)

    if not bool(getattr(settings, "TABLE_RAG_SELECTOR_ENABLED", True)):
        return _result_from_outcome(
            InsufficientEvidence(
                reason="LLM table reasoning disabled",
                status=STATUS_SELECTOR_FAILED,
            )
        )

    top_k = int(getattr(settings, "TABLE_RAG_SELECTOR_TOP_K", 8) or 8)
    max_total = max(top_k, min(12, top_k + 4))
    candidates = collect_table_candidates(chunks, top_k=top_k)
    if candidates:
        candidates = expand_sibling_and_referenced_tables(
            q, candidates, chunks, max_total=max_total
        )
    if not candidates:
        # Soft: allow prose answer when no table chunks arrived.
        if not table_chunks_present:
            return TableInterpretationResult(is_structured_lookup=False)
        return _result_from_outcome(
            InsufficientEvidence(
                reason="No structured tables in retrieved chunks",
                status=STATUS_NOT_FOUND,
                soft=True,
            )
        )

    payload = reason_over_tables(
        q, candidates, client=client, settings=settings
    )
    if isinstance(payload, dict):
        payload.setdefault("question", q)
    outcome = verify_llm_evidence(payload, candidates, question=q)

    # One corrective selector pass when the first pick fails axis-boundary math.
    if isinstance(outcome, InsufficientEvidence) and isinstance(payload, dict):
        conflict_notes: List[str] = []
        by_id = _candidate_index(candidates)
        for ev in list(payload.get("selected_evidence") or []):
            item = by_id.get(str(ev.get("table_id") or "")) or by_id.get(
                str(ev.get("chunk_id") or "")
            )
            if not item:
                continue
            table = item.get("table") or {}
            r_i, c_i = ev.get("row_index"), ev.get("col_index")
            row_headers, col_headers = [], []
            try:
                if r_i is not None and c_i is not None:
                    row_headers, col_headers = _headers_for(
                        table, int(r_i), int(c_i)
                    )
            except Exception:
                row_headers, col_headers = [], []
            note = _axis_hard_conflict(
                q,
                row_headers=row_headers,
                col_headers=col_headers,
                table=table,
            )
            if note:
                conflict_notes.append(
                    f"table={ev.get('table_id')} value={ev.get('value')}: {note}"
                )
        if conflict_notes:
            retry_payload = reason_over_tables(
                q,
                candidates,
                client=client,
                settings=settings,
                feedback=(
                    "Selected range headers excluded question axis values. "
                    "Re-select cells whose row/column range headers cover every "
                    "unit-bearing factor.\n" + "\n".join(conflict_notes[:6])
                ),
            )
            if isinstance(retry_payload, dict):
                retry_payload.setdefault("question", q)
                retry_outcome = verify_llm_evidence(
                    retry_payload, candidates, question=q
                )
                if isinstance(retry_outcome, VerifiedCell) or (
                    isinstance(retry_outcome, InsufficientEvidence)
                    and retry_outcome.status != STATUS_VERIFICATION_FAILED
                ):
                    payload = retry_payload
                    outcome = retry_outcome

    return _result_from_outcome(
        outcome,
        reasoning_summary=str(payload.get("reasoning_summary") or ""),
        raw_llm=payload if isinstance(payload, dict) else {},
    )


# Compatibility aliases used by older scripts/tests during migration.
interpret_with_selector = interpret_table_question
select_eligible_table_chunks = collect_table_candidates


def lookup_cell_in_table(*_args: Any, **_kwargs: Any) -> InsufficientEvidence:
    """Removed deterministic lookup — fail closed."""
    return InsufficientEvidence(
        reason="Deterministic cell lookup removed; use LLM table reasoning",
        status=STATUS_SELECTOR_FAILED,
    )


def verify_lookup_decision(*_args: Any, **_kwargs: Any) -> InsufficientEvidence:
    """Removed coordinate-only verify path — fail closed."""
    return InsufficientEvidence(
        reason="Legacy selector verification removed; use LLM table reasoning",
        status=STATUS_SELECTOR_FAILED,
    )

"""Expand a cellular table into independent cell facts for the answer prompt.

Mirrors the assessment-pipeline pattern: each answerable cell becomes a
self-contained ``header=value`` fact so the LLM matches the question against
atomic lines in the user prompt — no domain vocabularies, no product rules.

Wide matrices (many parallel value columns) emit one fact per value cell,
binding only the row-key columns plus that single value column. Narrow tables
with no parallel columns keep one fact per row.

When a query is provided, facts are pruned to query-near rows/columns so wide
matrices do not explode the LLM context.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_NUM_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z]+")
_RANGE_PAIR_RE = re.compile(
    r"(?i)(≥|>=|>|≤|<=|<)?\s*(\d+(?:\.\d+)?)\s*(?:–|-|—|to)\s*(\d+(?:\.\d+)?)"
)
_SINGLE_BOUND_RE = re.compile(
    r"(?i)(≥|>=|>|≤|<=|<)\s*(\d+(?:\.\d+)?)"
)


def _cell(value: Any) -> str:
    return str(value or "").strip()


def _header_signature(header: str) -> Tuple[str, ...]:
    """Token signature with numbers stripped so parallel band columns group."""
    text = _NUM_TOKEN_RE.sub(" ", (header or "").lower())
    toks = _WORD_RE.findall(text)
    stop = {"to", "and", "or", "the", "a", "an", "of", "per"}
    return tuple(t for t in toks if t not in stop)


def _parallel_value_columns(headers: Sequence[str]) -> Set[int]:
    """Column indices that share a signature with at least one other column."""
    sig_to_cols: Dict[Tuple[str, ...], List[int]] = {}
    for ci, header in enumerate(headers):
        sig = _header_signature(header)
        if not sig:
            continue
        sig_to_cols.setdefault(sig, []).append(ci)
    out: Set[int] = set()
    for cols in sig_to_cols.values():
        if len(cols) >= 2:
            out.update(cols)
    return out


_TABLE_REF_NUM_RE = re.compile(
    r"(?i)\b(?:tables?|tbl\.?)\s*([0-9]{1,3}|[A-Z])\b"
)


def _query_numbers(query: str) -> List[float]:
    """Numeric tokens from the question, excluding bare Table-N identifiers."""
    q = query or ""
    # Mask "Table 1" / "Tbl. 2" so the index is not treated as a band value.
    masked = _TABLE_REF_NUM_RE.sub("table", q)
    out: List[float] = []
    for m in _NUM_TOKEN_RE.finditer(masked):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def _text_covers_number(text: str, num: float) -> bool:
    """True when ``num`` falls in a range/bound expressed in ``text``."""
    raw = text or ""
    range_hits = list(_RANGE_PAIR_RE.finditer(raw))
    if range_hits:
        # Prefer explicit a–b bands; do not also treat ">200-300" as ">200".
        for m in range_hits:
            op = (m.group(1) or "").strip()
            try:
                lo = float(m.group(2))
                hi = float(m.group(3))
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            # >200-300 → (200, 300]; ≥30-100 → [30, 100]; bare 30-60 → [30, 60].
            if op in {">"}:
                if lo < num <= hi:
                    return True
            elif op in {"≥", ">="}:
                if lo <= num <= hi:
                    return True
            elif op in {"<", "≤", "<="}:
                if lo <= num < hi:
                    return True
            else:
                if lo <= num <= hi:
                    return True
        return False

    for m in _SINGLE_BOUND_RE.finditer(raw):
        op = (m.group(1) or "").strip()
        try:
            bound = float(m.group(2))
        except ValueError:
            continue
        if op in {">"} and num > bound:
            return True
        if op in {"≥", ">="} and num >= bound:
            return True
        if op in {"<"} and num < bound:
            return True
        if op in {"≤", "<="} and num <= bound:
            return True
    # Exact numeric token present (e.g. header "85" or cell "300").
    for m in _NUM_TOKEN_RE.finditer(raw):
        try:
            if abs(float(m.group(0)) - num) < 1e-9:
                return True
        except ValueError:
            continue
    return False


def _row_match_score(row: Sequence[Any], key_cols: Sequence[int], qnums: Sequence[float]) -> int:
    if not qnums:
        return 0
    blobs = [_cell(row[ci]) if ci < len(row) else "" for ci in key_cols]
    blob = " ".join(blobs)
    return sum(1 for n in qnums if _text_covers_number(blob, n))


def _col_match_score(header: str, qnums: Sequence[float]) -> int:
    if not qnums:
        return 0
    return sum(1 for n in qnums if _text_covers_number(header or "", n))


def select_query_near_indices(
    table: Dict[str, Any],
    query: Optional[str],
    *,
    max_rows: int = 6,
    max_cols: int = 4,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Pick row/value-column indices whose bands cover query numbers.

    Returns (row_indices, value_col_indices). Either may be None (= keep all).
    Domain-agnostic: uses numeric range geometry only.
    """
    qnums = _query_numbers(query or "")
    if len(qnums) < 1:
        return None, None

    headers = [_cell(h) for h in (table.get("headers") or [])]
    rows = [list(r) for r in (table.get("rows") or [])]
    if not headers or not rows:
        return None, None

    value_cols = sorted(_parallel_value_columns(headers))
    key_cols = [ci for ci in range(len(headers)) if ci not in set(value_cols)]

    row_scores: List[Tuple[int, int]] = []
    for ri, row in enumerate(rows):
        score = _row_match_score(row, key_cols or [0], qnums)
        if score > 0:
            row_scores.append((score, ri))
    row_scores.sort(key=lambda x: (-x[0], x[1]))
    picked_rows = [ri for _s, ri in row_scores[: max(1, max_rows)]]

    picked_cols: List[int] = []
    if value_cols:
        col_scores: List[Tuple[int, int]] = []
        for ci in value_cols:
            score = _col_match_score(headers[ci], qnums)
            if score > 0:
                col_scores.append((score, ci))
        col_scores.sort(key=lambda x: (-x[0], x[1]))
        picked_cols = [ci for _s, ci in col_scores[: max(1, max_cols)]]

    # If nothing matched, fall back to no pruning (caller keeps full facts).
    if not picked_rows and not picked_cols:
        return None, None
    if not picked_rows:
        picked_rows = list(range(min(len(rows), max_rows)))
    return picked_rows, (picked_cols or None)


def independent_cell_facts(
    table: Dict[str, Any],
    *,
    max_rows: Optional[int] = None,
    relevant_row_indices: Optional[Sequence[int]] = None,
    relevant_col_indices: Optional[Sequence[int]] = None,
    query: Optional[str] = None,
) -> List[str]:
    """One prompt line per answerable cell (or per row if the table is narrow)."""
    headers = [_cell(h) for h in (table.get("headers") or [])]
    rows = [list(r) for r in (table.get("rows") or [])]
    if not headers or not rows:
        return []

    # Query-near prune when caller did not already restrict rows/cols.
    if query and relevant_row_indices is None and relevant_col_indices is None:
        near_rows, near_cols = select_query_near_indices(table, query)
        if near_rows is not None:
            relevant_row_indices = near_rows
        if near_cols is not None:
            relevant_col_indices = near_cols

    if relevant_row_indices:
        keep = sorted(
            {
                int(i)
                for i in relevant_row_indices
                if 0 <= int(i) < len(rows)
            }
        )
        indexed = [(i, rows[i]) for i in keep]
    else:
        indexed = list(enumerate(rows))

    if max_rows is not None and max_rows > 0 and not relevant_row_indices:
        indexed = indexed[: int(max_rows)]

    width = len(headers)
    all_value_cols = _parallel_value_columns(headers)
    value_cols = set(all_value_cols)
    if relevant_col_indices is not None and value_cols:
        keep_cols = {int(c) for c in relevant_col_indices}
        value_cols = {c for c in value_cols if c in keep_cols}
    key_cols = [ci for ci in range(width) if ci not in all_value_cols]

    facts: List[str] = []

    def _bind(ci: int, row: Sequence[Any]) -> Optional[str]:
        header = headers[ci] or f"col_{ci}"
        value = _cell(row[ci]) if ci < len(row) else ""
        if not value:
            return None
        return f"{header}={value}"

    for ri, row in indexed:
        if not value_cols:
            parts = [p for ci in range(width) if (p := _bind(ci, row))]
            if parts:
                facts.append(f"cell_fact[r{ri}]: " + " | ".join(parts))
            continue

        key_parts = [p for ci in key_cols if (p := _bind(ci, row))]
        for ci in sorted(value_cols):
            value_part = _bind(ci, row)
            if not value_part:
                continue
            parts = key_parts + [value_part]
            facts.append(f"cell_fact[r{ri},c{ci}]: " + " | ".join(parts))
    return facts


def independent_row_facts(
    table: Dict[str, Any],
    *,
    max_rows: Optional[int] = None,
    relevant_row_indices: Optional[Sequence[int]] = None,
    relevant_col_indices: Optional[Sequence[int]] = None,
    query: Optional[str] = None,
) -> List[str]:
    return independent_cell_facts(
        table,
        max_rows=max_rows,
        relevant_row_indices=relevant_row_indices,
        relevant_col_indices=relevant_col_indices,
        query=query,
    )


def independent_rows_prompt_block(
    table: Dict[str, Any],
    *,
    max_rows: Optional[int] = None,
    relevant_row_indices: Optional[Sequence[int]] = None,
    relevant_col_indices: Optional[Sequence[int]] = None,
    query: Optional[str] = None,
    include_grid: bool = True,
    grid_block: str = "",
) -> str:
    """Assessment-style prompt block: caption + independent cell facts (+ optional grid)."""
    caption = _cell(table.get("caption") or table.get("table_title") or "")
    table_id = _cell(table.get("table_id") or "")
    page = table.get("page") or table.get("page_start")
    header_bits = ["INDEPENDENT CELL FACTS"]
    if caption:
        header_bits.append(caption)
    elif table_id:
        header_bits.append(table_id)
    if page not in (None, ""):
        header_bits.append(f"page {page}")

    lines = [
        " — ".join(header_bits),
        "LOOKUP RULE: each cell_fact is self-contained. Match the question to "
        "exactly one cell_fact using its header=value pairs; do not mix values "
        "across different cell_fact lines. Prefer a cell_fact whose bands cover "
        "the query numbers; never borrow values from a neighboring fact.",
    ]
    facts = independent_cell_facts(
        table,
        max_rows=max_rows,
        relevant_row_indices=relevant_row_indices,
        relevant_col_indices=relevant_col_indices,
        query=query,
    )
    if not facts:
        return grid_block.strip()
    lines.extend(facts)
    if include_grid and grid_block.strip():
        lines.append("")
        lines.append(grid_block.strip())
    return "\n".join(lines)

"""Resolve per-cell status without inventing values. Domain-agnostic."""
from __future__ import annotations

import re
from typing import List, Sequence

from ste.models import Cell, normalize_ws, pad_row
from ste.shared_attribute_expander import _NUMERIC_RE, _numeric_ratio, _label_like

# Explicit unavailable / missing patterns (source wording preserved via status).
_STATUS_PATTERNS = [
    ("insufficient_data", re.compile(r"(?i)\binsufficient\s+data\b")),
    ("not_applicable", re.compile(r"(?i)\b(not\s+applicable|n/?a)\b")),
    (
        "not_reported",
        re.compile(r"(?i)\b(not\s+reported|none\s+reported|not\s+available|\bnr\b)\b"),
    ),
    ("redacted", re.compile(r"(?i)\b(redacted|\*{3,}|\[redacted\])\b")),
    ("illegible", re.compile(r"(?i)\b(illegible|unreadable)\b")),
]

_UNAVAIL_FILL_RE = re.compile(r"(?i)\binsufficient\s+data\b")


def classify_status(text: str) -> tuple[str, str]:
    """Return (status, display_text). Never invents numeric values."""
    original = normalize_ws(text)
    if not original:
        return "empty", ""
    for status, pat in _STATUS_PATTERNS:
        if pat.search(original):
            return status, original
    return "present", original


def fill_unavailable_empties(rows: List[List[str]]) -> List[str]:
    """Fill empty cells in numeric columns only when unavailable markers exist."""
    warnings: List[str] = []
    if not rows:
        return warnings
    width = max(len(r) for r in rows)
    body = [pad_row(r, width) for r in rows]
    has_status = any(
        _UNAVAIL_FILL_RE.search(normalize_ws(c) or "") for row in body for c in row
    )
    if not has_status:
        rows[:] = body
        return warnings
    filled = 0
    for col in range(width):
        if _label_like(col, body):
            continue
        if _numeric_ratio(col, body) < 0.35 and not any(
            _UNAVAIL_FILL_RE.search(normalize_ws(r[col]) or "")
            for r in body
            if col < len(r)
        ):
            continue
        for row in body:
            if normalize_ws(row[col]):
                continue
            row[col] = "Insufficient Data"
            filled += 1
    if filled:
        warnings.append(f"unavailable_empties_filled:{filled}")
    rows[:] = body
    return warnings


def build_cells(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    confidence: float = 0.0,
    cell_region_map: dict | None = None,
    cell_bbox_map: dict | None = None,
    cell_span_map: dict | None = None,
) -> List[Cell]:
    """Build one :class:`Cell` per logical position.

    Args:
        headers: Reconstructed header path per column.
        rows: Body rows (already rectangular/padded).
        confidence: Cell confidence when the engine provides one.
        cell_region_map: ``(row, col) -> {status, explanation, region_id}``
            for connected unavailable-data regions.
        cell_bbox_map: ``(row, col) -> BBox`` source geometry (PDF points),
            when the extraction backend reports it.
        cell_span_map: ``(row, col) -> {"colspan": int, "rowspan": int}``
            merge information from the extraction backend.
    """
    width = max(len(headers), max((len(r) for r in rows), default=0), 1)
    hdrs = pad_row(list(headers), width)
    body = [pad_row(r, width) for r in rows]
    region_map = cell_region_map or {}
    bbox_map = cell_bbox_map or {}
    span_map = cell_span_map or {}
    cells: List[Cell] = []
    for ri, row in enumerate(body):
        for ci, text in enumerate(row):
            status, value_text = classify_status(text)
            original = normalize_ws(text)
            explanation = ""
            region_id = ""
            info = region_map.get((ri, ci))
            if info:
                status = str(info.get("status") or status)
                explanation = normalize_ws(str(info.get("explanation") or ""))
                region_id = str(info.get("region_id") or "")
                if status == "insufficient_data":
                    value_text = explanation or value_text
            unit = ""
            header = hdrs[ci] if ci < len(hdrs) else ""
            m = re.search(r"\(([^)]{1,16})\)", header or "")
            if m:
                unit = normalize_ws(m.group(1))
            span_info = span_map.get((ri, ci), {})
            cells.append(
                Cell(
                    row=ri,
                    col=ci,
                    text=value_text,
                    status=status,
                    original=original,
                    unit=unit,
                    bbox=bbox_map.get((ri, ci)),
                    confidence=confidence,
                    explanation=explanation,
                    region_id=region_id,
                    colspan=int(span_info.get("colspan", 1)),
                    rowspan=int(span_info.get("rowspan", 1)),
                )
            )
    return cells

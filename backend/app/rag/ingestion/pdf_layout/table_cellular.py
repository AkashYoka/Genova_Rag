"""Domain-agnostic fully-cellular table reconstruction.

DetectDocumentText has no native merged-cell model. Geometry grids therefore
leave blank cells where a PDF used rowspan/colspan or a single banner across a
region. This module expands those blanks into independent cells so every row is
self-contained.

Rules (structural only — no domain vocabularies):
- Never emit rowspan / colspan / merged_cells.
- Do not rewrite or flood header rows (leaf headers come from reconstruction).
- Repeat shared categorical attributes into every applicable row.
- In predominantly numeric columns, treat non-numeric text as a marker and copy
  that source text into contiguous empty neighbors (dedupe first; do not invent
  numeric values).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.ingestion.pdf_layout.types import normalize_ws

_NUMERIC_VALUE_RE = re.compile(
    r"^-?\d+(?:[.,]\d+)?(?:\s*[a-zA-Z%/μµ²³°]{0,8})?$"
)
_INSUFFICIENT_DATA_RE = re.compile(r"(?i)\binsufficient\s+data\b")


def _pad_grid(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    header_rows: Optional[Sequence[Sequence[Any]]] = None,
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    width = max(
        len(headers or []),
        max((len(r) for r in (rows or [])), default=0),
        max((len(r) for r in (header_rows or [])), default=0),
        1,
    )
    hdrs = [normalize_ws(h) for h in list(headers or [])]
    if len(hdrs) < width:
        hdrs = hdrs + [f"Column {i + 1}" for i in range(len(hdrs), width)]
    hdrs = hdrs[:width]
    body: List[List[str]] = []
    for row in rows or []:
        cells = [normalize_ws(c) for c in list(row)]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        body.append(cells[:width])
    hrows: List[List[str]] = []
    for row in header_rows or []:
        cells = [normalize_ws(c) for c in list(row)]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        hrows.append(cells[:width])
    return hdrs, body, hrows


def _is_numeric_like(value: str) -> bool:
    return bool(_NUMERIC_VALUE_RE.match(normalize_ws(value)))


def collapse_repeated_phrase(text: str) -> str:
    """Collapse duplicated banner phrases into a single marker string."""
    value = normalize_ws(text)
    if not value:
        return ""
    tokens = value.split()
    n = len(tokens)
    if n >= 2:
        for size in range(1, n // 2 + 1):
            if n % size != 0:
                continue
            unit = tokens[:size]
            if all(tokens[i : i + size] == unit for i in range(0, n, size)):
                value = " ".join(unit)
                break
    if _INSUFFICIENT_DATA_RE.search(value):
        # Canonical short form used by quality / retrieval prompts.
        return "Insufficient Data"
    return value


def _column_numeric_ratio(rows: Sequence[Sequence[str]], col: int) -> float:
    values = [normalize_ws(r[col]) for r in rows if col < len(r) and normalize_ws(r[col])]
    if not values:
        return 0.0
    numeric = sum(1 for v in values if _is_numeric_like(v))
    return float(numeric) / float(len(values))


def _column_is_label_like(rows: Sequence[Sequence[str]], col: int) -> bool:
    values = [normalize_ws(r[col]) for r in rows if col < len(r) and normalize_ws(r[col])]
    if not values:
        return False
    ratio = _column_numeric_ratio(rows, col)
    if ratio >= 0.55:
        return False
    # Prefer categorical / cadence columns with shortish labels.
    avg_len = sum(len(v) for v in values) / max(1, len(values))
    return avg_len <= 48.0 or ratio <= 0.25


def expand_shared_attributes(rows: List[List[str]]) -> List[List[str]]:
    """Fill blank cells in categorical columns from neighboring filled values."""
    if not rows:
        return rows
    width = max((len(r) for r in rows), default=0)
    rows = _pad_grid([], rows)[1]
    for col in range(width):
        if not _column_is_label_like(rows, col):
            continue
        # Forward fill within contiguous blank bands using last non-empty.
        last = ""
        for r in range(len(rows)):
            val = normalize_ws(rows[r][col])
            if val:
                last = val
            elif last:
                rows[r][col] = last
        # Backward fill remaining leading blanks from the next known value.
        nxt = ""
        for r in reversed(range(len(rows))):
            val = normalize_ws(rows[r][col])
            if val:
                nxt = val
            elif nxt:
                rows[r][col] = nxt
    return rows


def _numeric_columns(rows: Sequence[Sequence[str]]) -> List[int]:
    if not rows:
        return []
    width = max((len(r) for r in rows), default=0)
    return [c for c in range(width) if _column_numeric_ratio(rows, c) >= 0.45]


def _dedupe_join_parts(parts: Sequence[str]) -> str:
    cleaned = []
    for part in parts:
        text = collapse_repeated_phrase(normalize_ws(part))
        if text:
            cleaned.append(text)
    if not cleaned:
        return ""
    # Preserve order, drop exact duplicates.
    unique = list(dict.fromkeys(cleaned))
    if len(unique) == 1:
        return unique[0]
    # If every part collapses to the same insufficient-data marker, keep one.
    if all(_INSUFFICIENT_DATA_RE.search(p) for p in unique):
        return "Insufficient Data"
    return " ".join(unique)


def _join_non_value_fragments(
    row: Sequence[str], start: int, end: int
) -> str:
    parts: List[str] = []
    for idx, cell in enumerate(row):
        if idx < start or idx > end:
            continue
        text = normalize_ws(cell)
        if not text or _is_numeric_like(text):
            continue
        parts.append(text)
    return _dedupe_join_parts(parts)


def expand_marker_regions(rows: List[List[str]]) -> List[List[str]]:
    """Expand non-numeric markers across contiguous empty numeric-column cells."""
    if not rows:
        return rows
    width = max((len(r) for r in rows), default=0)
    rows = _pad_grid([], rows)[1]
    numeric_cols = set(_numeric_columns(rows))
    if not numeric_cols:
        return rows

    # Seed markers per cell.
    markers: List[List[str]] = [[""] * width for _ in rows]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            text = collapse_repeated_phrase(normalize_ws(cell))
            if not text or _is_numeric_like(text):
                continue
            if c in numeric_cols or _INSUFFICIENT_DATA_RE.search(text):
                markers[r][c] = text
                rows[r][c] = text

    # Horizontal expand within a row across empty numeric cells.
    for r in range(len(rows)):
        last = ""
        for c in range(width):
            if c not in numeric_cols and not markers[r][c]:
                last = ""
                continue
            val = normalize_ws(rows[r][c])
            if markers[r][c]:
                last = markers[r][c]
            elif not val and last:
                rows[r][c] = last
                markers[r][c] = last
        last = ""
        for c in reversed(range(width)):
            if c not in numeric_cols and not markers[r][c]:
                last = ""
                continue
            val = normalize_ws(rows[r][c])
            if markers[r][c]:
                last = markers[r][c]
            elif not val and last:
                rows[r][c] = last
                markers[r][c] = last

    # Vertical expand of markers in numeric columns.
    for c in range(width):
        if c not in numeric_cols:
            continue
        last = ""
        for r in range(len(rows)):
            val = normalize_ws(rows[r][c])
            if markers[r][c] or (
                val and not _is_numeric_like(val)
            ):
                last = markers[r][c] or collapse_repeated_phrase(val)
                rows[r][c] = last
                markers[r][c] = last
            elif not val and last:
                rows[r][c] = last
                markers[r][c] = last
        last = ""
        for r in reversed(range(len(rows))):
            val = normalize_ws(rows[r][c])
            if markers[r][c] or (
                val and not _is_numeric_like(val)
            ):
                last = markers[r][c] or collapse_repeated_phrase(val)
                rows[r][c] = last
                markers[r][c] = last
            elif not val and last:
                rows[r][c] = last
                markers[r][c] = last

    # Join only contiguous marker fragments inside numeric columns (banner spam).
    for r, row in enumerate(rows):
        for c in range(width):
            if c not in numeric_cols or not markers[r][c]:
                continue
            left = c
            while left - 1 >= 0 and left - 1 in numeric_cols and markers[r][left - 1]:
                left -= 1
            right = c
            while right + 1 < width and right + 1 in numeric_cols and markers[r][right + 1]:
                right += 1
            joined = _join_non_value_fragments(row, left, right)
            if joined and not _is_numeric_like(joined):
                rows[r][c] = joined
                markers[r][c] = joined
    return rows


def fill_unavailable_empty_cells(rows: List[List[str]]) -> List[List[str]]:
    """Copy unavailable markers into remaining empty cells of marker columns."""
    if not rows:
        return rows
    width = max((len(r) for r in rows), default=0)
    rows = _pad_grid([], rows)[1]
    for col in range(width):
        if _column_is_label_like(rows, col) and _column_numeric_ratio(rows, col) < 0.2:
            continue
        markers = [
            normalize_ws(r[col])
            for r in rows
            if normalize_ws(r[col]) and not _is_numeric_like(normalize_ws(r[col]))
        ]
        if not markers:
            continue
        # Prefer explicit unavailable markers.
        marker = ""
        for candidate in markers:
            low = candidate.lower()
            if (
                "n/a" == low
                or "not applicable" in low
                or "insufficient data" in low
                or low in {"na", "-", "—"}
            ):
                marker = collapse_repeated_phrase(candidate)
                break
        if not marker:
            continue
        for r in range(len(rows)):
            if not normalize_ws(rows[r][col]):
                rows[r][col] = marker
    return rows


def strip_merge_metadata(cells: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cell in cells or []:
        item = dict(cell or {})
        text = collapse_repeated_phrase(normalize_ws(str(item.get("text") or "")))
        item["text"] = text
        item["row_span"] = 1
        item["col_span"] = 1
        out.append(item)
    return out


def expand_to_fully_cellular(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    header_rows: Optional[Sequence[Sequence[Any]]] = None,
    cells: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a fully cellular grid with no merged-cell semantics."""
    hdrs, body, hrows = _pad_grid(headers, rows, header_rows=header_rows)
    warnings: List[str] = []
    if any(normalize_ws(c) for row in body for c in row):
        body = expand_shared_attributes(body)
        body = expand_marker_regions(body)
        body = fill_unavailable_empty_cells(body)
    cleaned_cells = strip_merge_metadata(cells or [])
    # Drop duplicate synthetic cells by (row,col).
    if cleaned_cells:
        uniq = {}
        for cell in cleaned_cells:
            key = (int(cell.get("row_index", -1)), int(cell.get("col_index", -1)))
            uniq[key] = cell
        cleaned_cells = list(uniq.values())
    return {
        "headers": list(hdrs),
        "rows": body,
        "header_rows": hrows,
        "merged_cells": [],
        "cells": cleaned_cells,
        "warnings": warnings,
        "fully_cellular": True,
    }


def table_data_to_html(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    caption: str = "",
) -> str:
    """Render a plain HTML table with no rowspan/colspan attributes."""
    width = max(len(headers or []), max((len(r) for r in (rows or [])), default=0), 0)
    parts: List[str] = ["<table>"]
    if caption:
        parts.append(f"<caption>{normalize_ws(caption)}</caption>")
    if headers:
        parts.append("<thead><tr>")
        hdrs = list(headers) + [""] * width
        for cell in hdrs[:width]:
            parts.append(f"<th>{normalize_ws(cell)}</th>")
        parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in rows or []:
        cells = list(row) + [""] * width
        parts.append("<tr>")
        for cell in cells[:width]:
            parts.append(f"<td>{normalize_ws(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)

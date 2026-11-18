"""Expand shared categorical attributes into every applicable row.

Domain-agnostic: no product/vocabulary rules. Handles vertically merged labels
whose text was OCR'd in the middle of a rowspan by assigning empty cells to the
nearest label *block* (consecutive identical values), using row index or
optional geometric row positions.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from ste.models import normalize_ws, pad_row

_NUMERIC_RE = re.compile(
    r"^-?\d+(?:[.,]\d+)?(?:\s*[a-zA-Z%/μµ²³°]{0,8})?$"
)
_INSUFFICIENT_RE = re.compile(r"(?i)\binsufficient\s+data\b")


def collapse_repeated_phrase(text: str) -> str:
    token = normalize_ws(text)
    if not token:
        return ""
    words = token.split()
    if len(words) >= 2:
        for size in range(1, len(words) // 2 + 1):
            if len(words) % size != 0:
                continue
            chunk = words[:size]
            chunks = [words[i : i + size] for i in range(0, len(words), size)]
            if len(chunks) >= 2 and all(c == chunk for c in chunks):
                token = " ".join(chunk)
                break
    if _INSUFFICIENT_RE.search(token):
        return "Insufficient Data"
    return token


def _numeric_ratio(col: int, rows: Sequence[Sequence[str]]) -> float:
    vals = [normalize_ws(r[col]) for r in rows if col < len(r) and normalize_ws(r[col])]
    if not vals:
        return 0.0
    return sum(1 for v in vals if _NUMERIC_RE.match(v)) / float(len(vals))


def _label_like(col: int, rows: Sequence[Sequence[str]]) -> bool:
    vals = [normalize_ws(r[col]) for r in rows if col < len(r) and normalize_ws(r[col])]
    if not vals:
        return col <= 1
    if _numeric_ratio(col, rows) >= 0.6:
        return False
    # Rate/count cells (e.g. "2 (1.1%)") are values, not shared labels.
    rateish = sum(
        1 for v in vals if "%" in v or re.search(r"\(\s*\d", v) or _NUMERIC_RE.match(v)
    )
    if rateish >= max(2, int(0.4 * len(vals))):
        return False
    unique = {v.lower() for v in vals}
    if len(vals) >= 2 and len(unique) <= max(2, len(vals) // 2):
        return True
    return col == 0


def _is_jammed_multi_value(text: str) -> bool:
    """True when a cell looks like several range labels smashed together."""
    from ste.header_hierarchy import split_jammed_ranges

    parts = split_jammed_ranges(text)
    return len(parts) >= 3


def _row_position(ri: int, positions: Optional[Sequence[float]]) -> float:
    if positions is not None and 0 <= ri < len(positions):
        try:
            return float(positions[ri])
        except (TypeError, ValueError):
            pass
    return float(ri)


def _label_blocks(
    body: Sequence[Sequence[str]],
    col: int,
    positions: Optional[Sequence[float]],
) -> List[Tuple[float, str]]:
    """Collapse consecutive identical non-empty cells into (centroid, value)."""
    blocks: List[Tuple[float, str]] = []
    i = 0
    n = len(body)
    while i < n:
        cur = normalize_ws(body[i][col])
        if not cur or _is_jammed_multi_value(cur):
            i += 1
            continue
        j = i + 1
        while j < n and normalize_ws(body[j][col]) == cur:
            j += 1
        centroid = sum(_row_position(k, positions) for k in range(i, j)) / float(j - i)
        blocks.append((centroid, cur))
        i = j
    return blocks


def expand_shared_attributes(
    rows: List[List[str]],
    *,
    row_positions: Optional[Sequence[float]] = None,
) -> List[str]:
    """Fill empty shared-label cells from the nearest label block.

    Engines often OCR a vertically merged categorical value in the *middle* of
    its visual span (wrapped fragments on a few body rows, blanks above/below).
    Forward-fill then paints that label past the true merge boundary into the
    next span. Instead, consecutive identical values form a block; each empty
    cell inherits the value of the nearest block centroid (row index, or
    geometric ``row_positions`` when provided). Exact ties prefer the earlier
    block.
    """
    warnings: List[str] = []
    if not rows:
        return warnings
    width = max(len(r) for r in rows)
    body = [pad_row(r, width) for r in rows]
    positions = row_positions
    if positions is not None and len(positions) != len(body):
        positions = None

    for col in range(width):
        if not _label_like(col, body):
            continue
        blocks = _label_blocks(body, col, positions)
        if not blocks:
            continue
        filled = 0
        for ri, row in enumerate(body):
            cur = normalize_ws(row[col])
            if cur:
                continue
            pos = _row_position(ri, positions)
            # Nearest block by centroid distance; earlier block wins ties.
            _cent, best_val = min(
                blocks,
                key=lambda item: (abs(item[0] - pos), item[0]),
            )
            row[col] = best_val
            filled += 1
        if filled:
            warnings.append(f"shared_attr_col_{col}_filled:{filled}")
    for row in body:
        for i, cell in enumerate(row):
            row[i] = collapse_repeated_phrase(cell)
    rows[:] = body
    return warnings


def body_row_positions_from_cells_meta(
    cells_meta: list | None,
    *,
    header_row_count: int,
    body_rows: int,
) -> Optional[List[float]]:
    """Average cell mid-Y per body row from extractor geometry (when present)."""
    if not cells_meta or body_rows <= 0:
        return None
    sums = [0.0] * body_rows
    counts = [0] * body_rows
    for meta in cells_meta:
        try:
            row = int(meta.get("row", -1))
            box = meta.get("bbox")
            if not box or len(box) < 4:
                continue
            body_row = row - int(header_row_count or 0)
            if body_row < 0 or body_row >= body_rows:
                continue
            y0, y1 = float(box[1]), float(box[3])
        except (TypeError, ValueError):
            continue
        sums[body_row] += (y0 + y1) * 0.5
        counts[body_row] += 1
    covered = sum(1 for c in counts if c > 0)
    if covered < max(2, (body_rows + 1) // 2):
        return None
    out: List[float] = []
    last = 0.0
    for i in range(body_rows):
        if counts[i]:
            last = sums[i] / float(counts[i])
            out.append(last)
        else:
            # Interpolate missing rows from neighbors when possible.
            out.append(last if i == 0 else last + 1.0)
    # Forward-fill any leading zeros from first known.
    first_known = next((i for i, c in enumerate(counts) if c), None)
    if first_known is not None and first_known > 0:
        base = out[first_known]
        for i in range(first_known):
            out[i] = base - (first_known - i)
    return out


def expand_marker_regions(rows: List[List[str]]) -> List[str]:
    """Copy non-numeric markers into contiguous empty neighbors in numeric cols."""
    warnings: List[str] = []
    if not rows:
        return warnings
    width = max(len(r) for r in rows)
    body = [pad_row(r, width) for r in rows]
    filled = 0
    for col in range(width):
        if _numeric_ratio(col, body) < 0.35:
            continue
        # Find marker cells
        for ri, row in enumerate(body):
            cur = normalize_ws(row[col])
            if not cur or _NUMERIC_RE.match(cur):
                continue
            marker = collapse_repeated_phrase(cur)
            row[col] = marker
            # Spread to empty neighbors in same column (contiguous)
            for rj in range(ri + 1, len(body)):
                if normalize_ws(body[rj][col]):
                    break
                body[rj][col] = marker
                filled += 1
            for rj in range(ri - 1, -1, -1):
                if normalize_ws(body[rj][col]):
                    break
                body[rj][col] = marker
                filled += 1
    if filled:
        warnings.append(f"marker_expanded:{filled}")
    rows[:] = body
    return warnings

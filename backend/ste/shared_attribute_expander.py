"""Expand shared categorical attributes into every applicable row."""
from __future__ import annotations

import re
from typing import List, Sequence

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


def expand_shared_attributes(rows: List[List[str]]) -> List[str]:
    warnings: List[str] = []
    if not rows:
        return warnings
    width = max(len(r) for r in rows)
    body = [pad_row(r, width) for r in rows]
    for col in range(width):
        if not _label_like(col, body):
            continue
        filled = 0
        last = ""
        for row in body:
            cur = normalize_ws(row[col])
            if cur:
                # Never propagate jammed multi-range garbage as a "shared" attr.
                last = "" if _is_jammed_multi_value(cur) else cur
            elif last:
                row[col] = last
                filled += 1
        last = ""
        for row in reversed(body):
            cur = normalize_ws(row[col])
            if cur:
                last = "" if _is_jammed_multi_value(cur) else cur
            elif last:
                row[col] = last
                filled += 1
        if filled:
            warnings.append(f"shared_attr_col_{col}_filled:{filled}")
    for row in body:
        for i, cell in enumerate(row):
            row[i] = collapse_repeated_phrase(cell)
    rows[:] = body
    return warnings


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

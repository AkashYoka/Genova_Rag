"""Build a rectangular text grid from positioned words in a table region."""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ste.models import Word, normalize_ws


def _cluster_rows(words: Sequence[Word]) -> List[List[Word]]:
    if not words:
        return []
    median_h = sorted(w.bbox.height for w in words)[len(words) // 2] or 8.0
    tol = max(2.5, median_h * 0.55)
    ordered = sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.height / 2, w.bbox.x0))
    rows: List[List[Word]] = []
    for w in ordered:
        if not rows:
            rows.append([w])
            continue
        prev = rows[-1]
        cy = sum(x.bbox.y0 + x.bbox.height / 2 for x in prev) / len(prev)
        wy = w.bbox.y0 + w.bbox.height / 2
        if abs(wy - cy) <= tol:
            prev.append(w)
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w.bbox.x0)
    return rows


def _column_anchors(rows: Sequence[Sequence[Word]], page_width: float) -> List[float]:
    """Estimate column x-centers from word starts."""
    xs: List[float] = []
    for row in rows:
        for w in row:
            xs.append(w.bbox.x0)
    if not xs:
        return []
    xs.sort()
    # Cluster x positions
    gap = max(8.0, page_width * 0.02)
    clusters: List[List[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] <= gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    anchors = [sum(c) / len(c) for c in clusters]
    return anchors


def _assign_col(x0: float, anchors: Sequence[float]) -> int:
    best_i, best_d = 0, abs(x0 - anchors[0])
    for i, a in enumerate(anchors[1:], start=1):
        d = abs(x0 - a)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def grid_from_words(
    words: Sequence[Word],
    *,
    page_width: float,
) -> Tuple[List[List[str]], List[float]]:
    """Return (grid_rows, column_anchors)."""
    rows = _cluster_rows(words)
    if len(rows) < 2:
        return [], []
    anchors = _column_anchors(rows, page_width)
    if len(anchors) < 2:
        return [], []
    width = len(anchors)
    grid: List[List[str]] = []
    for row in rows:
        cells = [""] * width
        # Collect words per column then join.
        buckets: List[List[str]] = [[] for _ in range(width)]
        for w in row:
            ci = _assign_col(w.bbox.x0, anchors)
            buckets[ci].append(w.text)
        for ci in range(width):
            cells[ci] = normalize_ws(" ".join(buckets[ci]))
        if any(cells):
            grid.append(cells)
    return grid, anchors


_PURE_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def split_header_body(grid: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    """Heuristic: leading non-data rows → header bands; rest → body."""
    if not grid:
        return [], []

    def is_data(row: Sequence[str]) -> bool:
        nonempty = [normalize_ws(c) for c in row if normalize_ws(c)]
        if not nonempty:
            return False
        pure = sum(1 for c in nonempty if _PURE_NUM_RE.match(c))
        if pure >= 2:
            return True
        if row and re.search(r"\d", row[0] or "") and pure >= 1:
            if re.search(r"[<>≥≤]|-|to", row[0] or "", re.I):
                return True
        return False

    promote = 0
    for i, row in enumerate(grid):
        if is_data(row):
            break
        promote = i + 1
    # At least one header row when possible.
    if promote == 0 and grid:
        promote = 1
    # Cap header depth.
    promote = min(promote, max(1, min(6, len(grid) // 2)))
    return [list(r) for r in grid[:promote]], [list(r) for r in grid[promote:]]

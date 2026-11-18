"""Detect table regions on a page (native + OCR word alignment)."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ste.models import BBox, Word


def native_table_bboxes(page: Any) -> List[BBox]:
    out: List[BBox] = []
    try:
        if not hasattr(page, "find_tables"):
            return out
        finder = page.find_tables()
        for t in list(getattr(finder, "tables", []) or []):
            try:
                b = t.bbox
                out.append(BBox(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            except Exception:
                continue
    except Exception:
        return out
    return out


def _cluster_y(words: Sequence[Word], tol: float) -> List[List[Word]]:
    ordered = sorted(words, key=lambda w: (w.bbox.y0, w.bbox.x0))
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
    return rows


def alignment_regions(
    words: Sequence[Word],
    *,
    page_width: float,
    page_height: float,
) -> List[BBox]:
    """Heuristic: dense multi-column word clusters → table-like regions."""
    if len(words) < 8:
        return []
    median_h = sorted(w.bbox.height for w in words)[len(words) // 2] or 8.0
    row_tol = max(3.0, median_h * 0.6)
    rows = _cluster_y(words, row_tol)
    # Keep rows with ≥2 words spanning distinct x positions.
    rich = []
    for row in rows:
        xs = sorted(w.bbox.x0 for w in row)
        if len(row) >= 2 and (xs[-1] - xs[0]) > page_width * 0.15:
            rich.append(row)
    if len(rich) < 3:
        return []

    # Group contiguous rich rows into regions.
    regions: List[BBox] = []
    start = 0
    while start < len(rich):
        end = start
        while end + 1 < len(rich):
            # Continuity by vertical gap
            a = rich[end][0].bbox.y1
            b = rich[end + 1][0].bbox.y0
            if b - a > median_h * 4:
                break
            end += 1
        group = [w for r in rich[start : end + 1] for w in r]
        if len(group) >= 12:
            regions.append(
                BBox(
                    min(w.bbox.x0 for w in group),
                    min(w.bbox.y0 for w in group),
                    max(w.bbox.x1 for w in group),
                    max(w.bbox.y1 for w in group),
                )
            )
        start = end + 1
    return regions


def merge_regions(regions: Sequence[BBox], *, iou_thresh: float = 0.4) -> List[BBox]:
    kept: List[BBox] = []
    for box in sorted(regions, key=lambda b: (b.y0, b.x0)):
        if box.area <= 0:
            continue
        if any(box.iou(k) >= iou_thresh for k in kept):
            continue
        kept.append(box)
    return kept


def detect_regions(
    page: Any,
    words: Sequence[Word],
    *,
    page_width: float,
    page_height: float,
) -> List[BBox]:
    native = native_table_bboxes(page)
    aligned = alignment_regions(
        words, page_width=page_width, page_height=page_height
    )
    merged = merge_regions(list(native) + list(aligned))
    if merged:
        return merged
    # Fallback: union of all words on page (single region) when gated positive.
    if words:
        return [
            BBox(
                min(w.bbox.x0 for w in words),
                min(w.bbox.y0 for w in words),
                max(w.bbox.x1 for w in words),
                max(w.bbox.y1 for w in words),
            )
        ]
    return []


def words_in_region(words: Sequence[Word], region: BBox, *, pad: float = 2.0) -> List[Word]:
    rx0, ry0, rx1, ry1 = region.x0 - pad, region.y0 - pad, region.x1 + pad, region.y1 + pad
    out: List[Word] = []
    for w in words:
        cx = (w.bbox.x0 + w.bbox.x1) / 2
        cy = (w.bbox.y0 + w.bbox.y1) / 2
        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
            out.append(w)
    return out

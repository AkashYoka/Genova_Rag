"""Cheap gate: decide whether a page likely contains tables (no OCR)."""
from __future__ import annotations

import re
from typing import Any

from ste.models import PageGateResult
from ste.pdf_loader import page_text

_TABLE_CUE_RE = re.compile(
    r"(?i)\b(?:"
    r"table\s+\d+|table\s+[a-z]|tables?\s*:|"
    r"see\s+table|refer\s+to\s+table"
    r")\b"
)


def _ruled_significant_count(page: Any) -> int:
    """Count mid-size ruled boxes from vector drawings (generic)."""
    try:
        drawings = page.get_drawings() or []
    except Exception:
        return 0
    page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
    boxes = []
    for d in drawings:
        try:
            r = d.get("rect")
            if r is None:
                continue
            x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if (0.08 * page_area) <= area <= (0.65 * page_area):
                boxes.append((x0, y0, x1, y1))
        except Exception:
            continue
    return len(boxes)


def page_has_tables(page: Any, page_number: int) -> PageGateResult:
    """Return whether this page should be rendered/OCR'd for tables."""
    find_count = 0
    try:
        if hasattr(page, "find_tables"):
            finder = page.find_tables()
            found = list(getattr(finder, "tables", []) or [])
            find_count = len(found)
            if found:
                return PageGateResult(
                    page=page_number,
                    has_tables=True,
                    reason="find_tables",
                    native_table_count=find_count,
                )
    except Exception:
        find_count = 0

    text = page_text(page)
    has_cue = bool(_TABLE_CUE_RE.search(text))
    ruled = _ruled_significant_count(page)
    if ruled > 0 and has_cue:
        return PageGateResult(
            page=page_number,
            has_tables=True,
            reason="ruled_boxes_with_cue",
            native_table_count=max(find_count, 1),
        )
    if has_cue and find_count == 0 and ruled == 0:
        # Caption-only mentions are weak; still allow when "Table N." appears
        # with dense numeric layout heuristics later — keep conservative.
        if re.search(r"(?i)\btable\s+\d+\.", text):
            return PageGateResult(
                page=page_number,
                has_tables=True,
                reason="table_caption_cue",
                native_table_count=0,
            )
    return PageGateResult(
        page=page_number,
        has_tables=False,
        reason="no_tabular_signal",
        native_table_count=0,
    )

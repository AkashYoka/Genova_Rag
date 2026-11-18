"""Conservative multi-page table stitching.

Docling (like Textract) reports one physical table per page. Continuation
pages usually repeat the header band at the top of the next page. This module
merges only when that repeated header matches the previous table *and* the
layout looks like a page break, so independent same-shape tables are not glued
together.
"""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ste.backends.base import RawTable
from ste.models import normalize_ws

_MIN_BOTTOM_RATIO = 0.65
_MAX_TOP_RATIO = 0.35


def _row_key(row: Sequence[str]) -> Tuple[str, ...]:
    return tuple(
        re.sub(r"\s+", " ", normalize_ws(c)).lower() for c in row if normalize_ws(c)
    )


def _strip_page_noise(text: str) -> str:
    """Ignore page numbers embedded in repeated header rows."""
    return re.sub(r"(?i)\bpage\s*\d+\b|\b\d+\s*/\s*\d+\b", "", text).strip()


def is_continuation(
    prev: RawTable,
    nxt: RawTable,
    *,
    page_sizes: Mapping[int, Tuple[float, float]],
) -> bool:
    """Return True when ``nxt`` is the same table continued on the next page."""
    if not prev.grid or not nxt.grid:
        return False
    if nxt.page != prev.pages[-1] + 1:
        return False
    if prev.n_cols != nxt.n_cols:
        return False
    if not nxt.header_row_count:
        return False

    prev_key = [_strip_page_noise(c) for c in _row_key(prev.grid[0])]
    next_key = [_strip_page_noise(c) for c in _row_key(nxt.grid[0])]
    if prev_key != next_key or not prev_key:
        return False

    prev_h = page_sizes.get(prev.pages[-1], (0.0, 0.0))[1]
    next_h = page_sizes.get(nxt.page, (0.0, 0.0))[1]
    if prev_h > 0 and prev.bbox.y1 < prev_h * _MIN_BOTTOM_RATIO:
        return False
    if next_h > 0 and nxt.bbox.y0 > next_h * _MAX_TOP_RATIO:
        return False
    return True


def _offset_cells_meta(
    meta: Sequence[Dict[str, object]],
    *,
    row_offset: int,
    drop_rows: int,
    page: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in meta:
        try:
            row = int(item.get("row", -1))
            col = int(item.get("col", -1))
        except Exception:
            continue
        if row < drop_rows or row < 0 or col < 0:
            continue
        clone = dict(item)
        clone["row"] = row - drop_rows + row_offset
        clone["col"] = col
        try:
            origin_row = int(item.get("origin_row", row) or row)
        except Exception:
            origin_row = row
        clone["origin_row"] = origin_row - drop_rows + row_offset
        clone["page"] = int(item.get("page") or page)
        out.append(clone)
    return out


def stitch_multipage_tables(
    tables: Sequence[RawTable],
    *,
    page_sizes: Optional[Mapping[int, Tuple[float, float]]] = None,
    enabled: bool = True,
) -> Tuple[List[RawTable], List[str]]:
    """Merge conservative page-break continuations.

    Args:
        tables: Raw tables in arbitrary order.
        page_sizes: ``{page: (width, height)}`` in PDF points.
        enabled: When False, tables are only sorted.

    Returns:
        ``(merged_tables, warnings)``.
    """
    sizes = dict(page_sizes or {})
    ordered = sorted(tables, key=lambda t: (t.page, t.bbox.y0, t.bbox.x0))
    if not enabled:
        return ordered, []

    merged: List[RawTable] = []
    warnings: List[str] = []
    for table in ordered:
        if not merged:
            merged.append(table)
            continue
        prev = merged[-1]
        if not is_continuation(prev, table, page_sizes=sizes):
            merged.append(table)
            continue

        drop = int(table.header_row_count or 0)
        if drop >= len(table.grid):
            warnings.append(f"multipage_skipped_header_only:{table.page}")
            merged.append(table)
            continue
        row_offset = len(prev.grid)
        prev.grid.extend([list(r) for r in table.grid[drop:]])
        prev.cells_meta.extend(
            _offset_cells_meta(
                table.cells_meta,
                row_offset=row_offset,
                drop_rows=drop,
                page=table.page,
            )
        )
        prev.pages = list(dict.fromkeys([*prev.pages, *table.pages]))
        prev.metadata = dict(prev.metadata or {})
        prev.metadata.setdefault("continuation_pages", []).append(table.page)
        prev.metadata.setdefault("stitched_sources", []).append(table.source)
        warnings.append(f"multipage_table_stitched:p{table.page}->p{prev.page}")
    return merged, warnings

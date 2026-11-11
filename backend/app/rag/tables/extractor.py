"""Layout-aware table extraction with geometry, merges, captions and footnotes.

PyMuPDF's table finder exposes cell rectangles and header information, but its
grid can be structurally wrong on complex merged headers (caption text leaking
into headers, several labels stacked in one cell, phantom columns). This module
normalizes the PyMuPDF grid, detects when that grid is untrustworthy, and falls
back to Camelot's lattice/stream parsers for the same page region. Cell
coordinates are preserved from whichever extractor produced the chosen grid.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pymupdf as fitz

from app.rag.ingestion.pdf_layout.types import BBox, TableData, TextLine, normalize_ws
from app.rag.tables.normalize import (
    detect_header_row_count,
    drop_empty_columns,
    normalize_table_data,
    resolve_merged_cells,
)

logger = logging.getLogger(__name__)

_FOOTNOTE_RE = re.compile(
    r"^\s*(?:[*†‡§¶]|note\b|notes\b|source\b|superscript|[\d]{1,2}\s+\S)",
    re.I,
)


def _norm(value: Any) -> str:
    return normalize_ws(str(value or ""))


def _raw(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _bbox_tuple(value: Any) -> Optional[List[float]]:
    box = BBox.from_any(value)
    return list(box.as_tuple()) if box else None


def _same_box(left: Optional[Sequence[float]], right: Optional[Sequence[float]], tol: float = 1.0) -> bool:
    if not left or not right or len(left) < 4 or len(right) < 4:
        return False
    return all(abs(float(left[i]) - float(right[i])) <= tol for i in range(4))


def _bbox_iou(left: Optional[BBox], right: Optional[BBox]) -> float:
    if left is None or right is None:
        return 0.0
    intersection = left.intersection_area(right)
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


def _spans_from_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect merged cells by repeated cell rectangles."""
    spans: Dict[Tuple[int, int], Dict[str, int]] = {}
    anchors: Dict[Tuple[float, float, float, float], Tuple[int, int]] = {}
    for cell in cells:
        bbox = cell.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        key = tuple(round(float(v), 1) for v in bbox[:4])
        anchor = anchors.get(key)
        if anchor is None:
            anchors[key] = (int(cell["row"]), int(cell["col"]))
            continue
        span = spans.setdefault(
            anchor, {"row": anchor[0], "col": anchor[1], "row_span": 1, "col_span": 1}
        )
        span["row_span"] = max(span["row_span"], int(cell["row"]) - anchor[0] + 1)
        span["col_span"] = max(span["col_span"], int(cell["col"]) - anchor[1] + 1)
        cell["merged_into"] = [anchor[0], anchor[1]]
    return list(spans.values())


def _cells_from_table(table: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Derive per-cell coordinates from a PyMuPDF table."""
    try:
        raw_rows = list(getattr(table, "rows", []) or [])
    except Exception:
        return [], []
    if not raw_rows:
        return [], []

    box_rows: List[List[Optional[List[float]]]] = []
    for row in raw_rows:
        try:
            boxes = list(getattr(row, "cells", []) or [])
        except Exception:
            boxes = []
        box_rows.append([_bbox_tuple(box) for box in boxes])

    width = max((len(row) for row in box_rows), default=0)
    cells: List[Dict[str, Any]] = []
    for row_index, row in enumerate(box_rows):
        for col_index in range(width):
            box = row[col_index] if col_index < len(row) else None
            cells.append(
                {
                    "row": row_index,
                    "col": col_index,
                    "text": "",
                    "row_span": 1,
                    "col_span": 1,
                    "header": False,
                    "bbox": box,
                }
            )
    return cells, _spans_from_cells(cells)


def _header_hint(table: Any) -> Optional[int]:
    header = getattr(table, "header", None)
    if header is None:
        return None
    try:
        header_box = BBox.from_any(getattr(header, "bbox", None))
        if header_box is None:
            return None
        count = 0
        for row in getattr(table, "rows", []) or []:
            row_box = BBox.from_any(getattr(row, "bbox", None))
            if row_box is not None and row_box.center_y <= header_box.y1 + 1.0:
                count += 1
        return count or None
    except Exception:
        return None


def _footnotes_near(
    lines: Sequence[TextLine],
    table_bbox: BBox,
    page_height: float,
    *,
    max_lines: int = 6,
) -> List[str]:
    """Capture short annotation lines immediately below a table."""
    if not lines:
        return []
    distance = max(28.0, page_height * 0.045)
    candidates: List[Tuple[float, str]] = []
    for line in lines:
        text = normalize_ws(line.text)
        if not text or len(text) > 400:
            continue
        if line.bbox.y0 < table_bbox.y1 - 2.0:
            continue
        if line.bbox.y0 > table_bbox.y1 + distance:
            continue
        if line.bbox.x1 < table_bbox.x0 or line.bbox.x0 > table_bbox.x1:
            continue
        if _FOOTNOTE_RE.match(text):
            candidates.append((line.bbox.y0, text))
    candidates.sort(key=lambda item: item[0])
    return [text for _y, text in candidates[:max_lines]]


def _table_score(table: TableData) -> float:
    """Structural quality score used to choose between extractors."""
    headers = [_norm(header) for header in table.headers]
    body = [[_norm(cell) for cell in row] for row in table.rows]
    score = 0.0
    if headers:
        score += min(6.0, float(len(headers)))
        score -= 6.0 * max(0, len(headers) - len(set(headers)))
        score -= 12.0 * sum(1 for header in headers if len(header) > 140)
    if body:
        score += 1.2 * len(body)
        labels = [row[0] for row in body if row]
        distinct = {label for label in labels if label}
        score += 1.5 * len(distinct)
        empty = sum(1 for label in labels if not label)
        if labels and empty >= max(2, len(labels) // 2):
            score -= 10.0
        if labels and len(re.findall(r"\d+\s*[-\u2212]\s*\d+", labels[0])) >= 3:
            score -= 10.0
    width = max(1, table.n_cols)
    non_empty = sum(1 for row in body for cell in row if cell)
    if body:
        score += min(10.0, (non_empty / (len(body) * width)) * 10.0)
    return score


_RANGE_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:[-\u2212\u2013\u2014]|to)\s*[>≥]?\s*\d+(?:\.\d+)?",
    re.I,
)


def _needs_camelot(table: TableData) -> bool:
    """True when the native grid shows known structural-breakage signatures."""
    headers = [_norm(header) for header in table.headers]
    if headers and len(headers) > len(set(headers)):
        return True
    if any(len(header) > 140 for header in headers):
        return True
    # Body row labels collapsed into a header cell (several ranges in one label).
    if any(len(_RANGE_TOKEN_RE.findall(header)) >= 3 for header in headers):
        return True
    placeholders = sum(
        1 for header in headers if re.fullmatch(r"column\s+\d+", header, re.I)
    )
    if placeholders >= 2:
        return True
    if not table.rows:
        return False
    labels = [_norm(row[0]) for row in table.rows if row]
    empty = sum(1 for label in labels if not label)
    if labels and empty >= max(2, len(labels) // 2):
        return True
    if labels and len(re.findall(r"\d+\s*[-\u2212]\s*\d+", labels[0])) >= 3:
        return True
    if (
        len(labels) >= 4
        and len(set(labels)) == 1
        and len(labels[0]) > 80
    ):
        return True
    # Frequency/schedule fragments used as row labels (Every / 4 / weeks / ...)
    # while measurable values sit elsewhere — a classic collapsed-grid smell.
    from app.rag.tables.normalize import _looks_like_phrase_fragment

    if labels and sum(
        1 for label in labels if _looks_like_phrase_fragment(label)
    ) >= max(3, len(labels) // 2):
        return True
    return False


def _camelot_bbox(bbox: Any, page_height: float) -> Optional[BBox]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in list(bbox)[:4])
    except (TypeError, ValueError):
        return None
    # Camelot uses a bottom-left origin; convert to top-left points.
    return BBox.from_any([x0, page_height - y1, x1, page_height - y0])


def _camelot_cells(table: Any, page_height: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw = getattr(table, "cells", None) or []
    cells: List[Dict[str, Any]] = []
    for row_index, row in enumerate(raw):
        for col_index, cell in enumerate(row or []):
            if cell is None:
                continue
            try:
                x0 = float(getattr(cell, "x1"))
                x1 = float(getattr(cell, "x2"))
                y_bottom = float(getattr(cell, "y1"))
                y_top = float(getattr(cell, "y2"))
            except Exception:
                continue
            cells.append(
                {
                    "row": row_index,
                    "col": col_index,
                    "text": "",
                    "row_span": 1,
                    "col_span": 1,
                    "header": False,
                    "bbox": [x0, page_height - y_top, x1, page_height - y_bottom],
                }
            )
    return cells, _spans_from_cells(cells)


def _camelot_tables_for_page(
    filepath: str,
    *,
    page_number: int,
    page_height: float,
    document_id: str = "",
) -> List[TableData]:
    try:
        import camelot
    except Exception as exc:
        logger.debug("Camelot unavailable for table fallback: %s", exc)
        return []

    candidates: List[TableData] = []
    for flavor in ("stream", "lattice", "network"):
        try:
            try:
                tables = camelot.read_pdf(
                    filepath,
                    pages=str(int(page_number)),
                    flavor=flavor,
                    suppress_stdout=True,
                )
            except TypeError:
                tables = camelot.read_pdf(
                    filepath, pages=str(int(page_number)), flavor=flavor
                )
        except Exception as exc:
            logger.debug(
                "Camelot flavor=%s page=%s failed: %s", flavor, page_number, exc
            )
            continue
        for found in tables or []:
            try:
                raw_rows = [
                    [_raw(cell) for cell in row]
                    for row in found.df.values.tolist()
                ]
                raw_rows = [row for row in raw_rows if any(_norm(cell) for cell in row)]
                if len(raw_rows) < 2:
                    continue
                width = max(len(row) for row in raw_rows)
                if width < 2:
                    continue
                raw_rows = [
                    (row + [""] * (width - len(row)))[:width] for row in raw_rows
                ]
                bbox = _camelot_bbox(getattr(found, "_bbox", None), page_height)
                if bbox is None:
                    continue
                cells, spans = _camelot_cells(found, page_height)
                table = TableData(
                    page=page_number,
                    bbox=bbox,
                    headers=raw_rows[0],
                    rows=raw_rows[1:],
                    extraction_method=f"camelot_{flavor}",
                    source_lines=len(raw_rows),
                    header_row_count=1,
                    grid=raw_rows,
                    cells=cells,
                    spans=spans,
                    # The raw grid below is the original representation; Camelot's
                    # to_markdown() writes to a path and is not needed here.
                    original_markdown="",
                )
                normalize_table_data(table, document_id=document_id)
                candidates.append(table)
            except Exception as exc:
                logger.debug("Camelot table conversion failed: %s", exc)
        if candidates and max(_table_score(candidate) for candidate in candidates) >= 0:
            break
    return candidates


def _choose_tables(
    native: Sequence[TableData],
    camelot_candidates: Sequence[TableData],
) -> List[TableData]:
    """Replace structurally broken native grids with better fallback grids."""
    if not camelot_candidates:
        return list(native)
    chosen: List[TableData] = []
    for table in native:
        if not _needs_camelot(table):
            chosen.append(table)
            continue
        best: Optional[TableData] = None
        best_iou = 0.0
        for candidate in camelot_candidates:
            iou = _bbox_iou(table.bbox, candidate.bbox)
            if iou > best_iou:
                best_iou = iou
                best = candidate
        if best is None:
            best = max(camelot_candidates, key=_table_score)
        if _table_score(best) > _table_score(table) + 3.0:
            best.caption = best.caption or table.caption
            best.footnotes = list(best.footnotes or table.footnotes or [])
            logger.info(
                "TABLE_FLOW %s",
                {
                    "event": "table_extractor_fallback",
                    "page": table.page,
                    "native_method": table.extraction_method,
                    "fallback_method": best.extraction_method,
                    "native_score": round(_table_score(table), 2),
                    "fallback_score": round(_table_score(best), 2),
                    "bbox_iou": round(best_iou, 3),
                },
            )
            chosen.append(best)
        else:
            chosen.append(table)
    return chosen


def extract_page_tables(
    page: fitz.Page,
    *,
    page_number: int,
    document_id: str = "",
    lines: Sequence[TextLine] = (),
    method: str = "native",
    filepath: str = "",
) -> List[TableData]:
    """Extract every complete table on a page with layout metadata."""
    if not hasattr(page, "find_tables"):
        return []
    try:
        finder = page.find_tables()
        found = list(getattr(finder, "tables", []) or [])
    except Exception as exc:
        logger.debug("Table detection failed on page %s: %s", page_number, exc)
        return []

    page_height = float(page.rect.height)
    tables: List[TableData] = []

    for found_table in found:
        try:
            raw_grid = [
                [_raw(cell) for cell in (row or [])]
                for row in (found_table.extract() or [])
            ]
            raw_grid = [row for row in raw_grid if any(_norm(cell) for cell in row)]
            if len(raw_grid) < 2:
                continue
            width = max(len(row) for row in raw_grid)
            if width < 2:
                continue
            raw_grid = [
                (row + [""] * (width - len(row)))[:width] for row in raw_grid
            ]

            cells, spans = _cells_from_table(found_table)
            raw_grid, spans, cells = drop_empty_columns(raw_grid, spans, cells)
            header_count = detect_header_row_count(
                raw_grid, hint=_header_hint(found_table)
            )
            bbox = BBox.from_any(getattr(found_table, "bbox", None))
            if bbox is None:
                continue
            table = TableData(
                page=page_number,
                bbox=bbox,
                headers=raw_grid[0],
                rows=raw_grid[1:],
                extraction_method=method,
                source_lines=len(raw_grid),
                header_row_count=header_count,
                grid=raw_grid,
                cells=cells,
                spans=spans,
                original_markdown=(
                    found_table.to_markdown()
                    if hasattr(found_table, "to_markdown")
                    else ""
                ),
            )
            normalize_table_data(table, document_id=document_id)
            for cell in table.cells:
                row_index = int(cell.get("row", 0))
                col_index = int(cell.get("col", 0))
                if row_index < len(raw_grid) and col_index < len(raw_grid[row_index]):
                    cell["text"] = _norm(raw_grid[row_index][col_index])
                cell["header"] = row_index < table.header_row_count
            table.footnotes = _footnotes_near(lines, table.bbox, page_height)
            if table.headers and table.rows:
                tables.append(table)
        except Exception as exc:
            logger.debug("Skipping native table on page %s: %s", page_number, exc)

    if filepath and any(_needs_camelot(table) for table in tables):
        candidates = _camelot_tables_for_page(
            filepath,
            page_number=page_number,
            page_height=page_height,
            document_id=document_id,
        )
        tables = _choose_tables(tables, candidates)

    return tables


def attach_footnotes_from_lines(
    tables: Sequence[TableData],
    lines: Sequence[TextLine],
    page_height: float,
) -> None:
    """Attach footnotes once page lines are available (native path ordering)."""
    for table in tables:
        if table.footnotes:
            continue
        table.footnotes = _footnotes_near(lines, table.bbox, page_height)

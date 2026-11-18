"""Table reconstruction with optional merged-cell (colspan/rowspan) support.

Domain-agnostic: spans come from the extraction backend (e.g. Docling
TableFormer) via ``cells_meta``, not from hardcoded domain knowledge.
"""
from __future__ import annotations

import re
from typing import List, Sequence

from ste.cell_status import build_cells, fill_unavailable_empties
from ste.grid_postprocess import is_unit_banner_row, postprocess_grid
from ste.header_hierarchy import expand_jammed_headers, header_paths, leaf_headers
from ste.models import BBox, ExtractedTable, Word, normalize_ws, pad_row
from ste.shared_attribute_expander import (
    body_row_positions_from_cells_meta,
    expand_marker_regions,
    expand_shared_attributes,
)
from ste.structure_extractor import grid_from_words, split_header_body
from ste.unavailable_regions import (
    expand_unavailable_regions,
    validate_unavailable_region_coverage,
)


def cell_bbox_map(
    cells_meta: list | None,
    header_row_count: int,
) -> dict:
    """Map body-relative ``(row, col)`` to source cell bounding boxes.

    ``cells_meta`` coordinates are grid-absolute (header rows included), so
    the header band is subtracted. Cells without backend geometry are simply
    absent from the map.
    """
    out: dict = {}
    for meta in cells_meta or []:
        try:
            row = int(meta.get("row", -1))
            col = int(meta.get("col", -1))
        except (TypeError, ValueError):
            continue
        body_row = row - int(header_row_count or 0)
        if body_row < 0 or col < 0:
            continue
        box = meta.get("bbox")
        if not box or len(box) < 4:
            continue
        try:
            out[(body_row, col)] = BBox(
                float(box[0]), float(box[1]), float(box[2]), float(box[3])
            )
        except (TypeError, ValueError):
            continue
    return out


def cell_span_map(
    cells_meta: list | None,
    header_row_count: int,
) -> dict:
    """Map body-relative ``(row, col)`` to ``{"colspan": int, "rowspan": int}``.

    Only the *origin* cell of a span (dr=0, dc=0) reports non-1 spans.
    Covered positions report colspan=rowspan=1 so the HTML exporter can skip
    them. ``cells_meta`` coordinates are grid-absolute (header rows included),
    so the header band is subtracted.
    """
    out: dict = {}
    for meta in cells_meta or []:
        try:
            row = int(meta.get("row", -1))
            col = int(meta.get("col", -1))
            origin_row = int(meta.get("origin_row", row))
            origin_col = int(meta.get("origin_col", col))
        except (TypeError, ValueError):
            continue
        body_row = row - int(header_row_count or 0)
        if body_row < 0 or col < 0:
            continue
        row_span = int(meta.get("row_span", 1) or 1)
        col_span = int(meta.get("col_span", 1) or 1)
        # Only the origin cell carries the real span; covered positions are 1x1.
        if (row, col) == (origin_row, origin_col) and (row_span > 1 or col_span > 1):
            out[(body_row, col)] = {"colspan": col_span, "rowspan": row_span}
        else:
            out.setdefault((body_row, col), {"colspan": 1, "rowspan": 1})
    return out


def _looks_like_body_row(row: Sequence[str]) -> bool:
    """True when a candidate header band is really the first data row.

    Textract AnalyzeDocument often marks the first body row as COLUMN_HEADER,
    which then becomes ``Header / value`` paths and drops a real data row.
    """
    nonempty = [normalize_ws(c) for c in row if normalize_ws(c)]
    if not nonempty:
        return False
    # Sentence-length cells are almost never header labels.
    if any(len(c.split()) >= 6 for c in nonempty):
        return True
    if any(len(c) > 48 for c in nonempty):
        return True
    # Mixed short + long (e.g. "UAE" + long Peppol description) → body.
    longish = sum(1 for c in nonempty if len(c) > 32 or len(c.split()) >= 4)
    shortish = sum(1 for c in nonempty if len(c) <= 24)
    if longish >= 2 and shortish >= 1:
        return True
    return False


def _header_continuation(row: Sequence[str]) -> bool:
    """Extend under-counted headers using structural signals only."""
    if is_unit_banner_row(row):
        return True
    # Never promote a row that already looks like body content.
    if _looks_like_body_row(row):
        return False
    nonempty = [normalize_ws(c) for c in row if normalize_ws(c)]
    if not nonempty:
        return False
    if any("%" in c or re.search(r"\(\s*\d", c) for c in nonempty):
        return False
    letterish = sum(1 for c in nonempty if re.search(r"[A-Za-z]{2,}", c))
    if letterish < max(1, (len(nonempty) + 1) // 2):
        return False
    if any(
        re.search(
            r"(?i)(?:≥|>=|>|≤|<=|<)\s*\d+(?:\.\d+)?\s*(?:–|-|—|to)\s*\d+",
            c,
        )
        or re.search(r"(?i)\d+(?:\.\d+)?\s*(?:–|-|—|to)\s*\d+.*\b[a-z]{1,6}\b", c)
        or re.match(r"(?i)^(?:\([^)]{0,12}\)|[a-z]{2,12})$", c)
        for c in nonempty
    ):
        return True
    pure_num = sum(1 for c in nonempty if re.fullmatch(r"-?\d+(?:\.\d+)?", c))
    short_labels = sum(1 for c in nonempty if len(c) <= 28 and re.search(r"[A-Za-z]", c))
    if pure_num == 0 and short_labels >= max(2, len(nonempty) - 1):
        if any(len(c.split()) >= 6 for c in nonempty):
            return False
        if any(re.search(r"(?i)\d+\s*(mg|ml|mL|kg|%)", c) for c in nonempty):
            return False
        return True
    return False


def reconstruct_from_grid(
    grid: List[List[str]],
    *,
    table_id: str,
    page: int,
    bbox: BBox,
    caption: str = "",
    extraction_method: str = "geometry",
    confidence: float = 0.0,
    provenance: dict | None = None,
    header_row_count: int | None = None,
    document_id: str = "",
    cells_meta: list | None = None,
    image_bytes: bytes | None = None,
) -> ExtractedTable | None:
    if len(grid) < 2:
        return None
    width = max(len(r) for r in grid)
    if width < 2:
        return None
    original_grid = [pad_row(r, width) for r in grid]
    grid = [list(r) for r in original_grid]
    if header_row_count is not None and 1 <= header_row_count < len(grid):
        header_rows = [list(r) for r in grid[:header_row_count]]
        body = [list(r) for r in grid[header_row_count:]]
    else:
        header_rows, body = split_header_body(grid)
    if not body:
        return None

    extended = 0
    while body and extended < 3 and _header_continuation(body[0]):
        header_rows.append(body.pop(0))
        extended += 1
    # Demote Textract false-positive header bands (first data row as header).
    while len(header_rows) > 1 and _looks_like_body_row(header_rows[-1]):
        body.insert(0, header_rows.pop())
    if not body:
        return None

    original_headers = (
        leaf_headers(header_rows)
        if header_rows
        else [f"Column {i + 1}" for i in range(width)]
    )
    headers = list(original_headers)
    headers, header_rows, body = expand_jammed_headers(headers, header_rows, body)
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    headers = pad_row(headers, width)
    header_rows = [pad_row(r, width) for r in header_rows]
    body = [pad_row(r, width) for r in body]
    if header_rows:
        headers = leaf_headers(header_rows)
        original_headers = list(headers)

    warnings: List[str] = []
    headers, header_rows, body, post_warnings = postprocess_grid(
        headers, header_rows, body
    )
    warnings.extend(post_warnings)

    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    headers = pad_row(headers, width)
    header_rows = [pad_row(r, width) for r in header_rows]
    body = [pad_row(r, width) for r in body]
    original_headers = pad_row(original_headers, width)

    used_header_count = len(header_rows) if header_rows else int(header_row_count or 0)
    row_positions = body_row_positions_from_cells_meta(
        cells_meta,
        header_row_count=used_header_count,
        body_rows=len(body),
    )
    warnings.extend(expand_shared_attributes(body, row_positions=row_positions))
    warnings.extend(expand_marker_regions(body))
    warnings.extend(fill_unavailable_empties(body))

    body, unavail_regions, unavail_warnings, cell_region_map = expand_unavailable_regions(
        body,
        cells_meta=cells_meta,
        image_bytes=image_bytes,
        header_row_count=used_header_count,
    )
    warnings.extend(unavail_warnings)
    coverage = validate_unavailable_region_coverage(body, unavail_regions, cell_region_map)
    if not coverage.get("ok", True):
        warnings.append(f"unavailable_region_coverage_issues:{coverage.get('issues')}")

    reconstructed_headers = (
        header_paths(header_rows) if header_rows else list(headers)
    )
    if any(" / " in h for h in headers):
        reconstructed_headers = list(headers)
    elif headers and headers != original_headers:
        reconstructed_headers = list(headers)
    else:
        reconstructed_headers = pad_row(reconstructed_headers, width)

    conf = float(confidence)
    bbox_map = cell_bbox_map(cells_meta, used_header_count)
    span_map = cell_span_map(cells_meta, used_header_count)
    cells = build_cells(
        reconstructed_headers,
        body,
        confidence=conf if conf <= 100 else conf,
        cell_region_map=cell_region_map,
        cell_bbox_map=bbox_map,
        cell_span_map=span_map,
    )
    has_spans = any(
        v.get("colspan", 1) > 1 or v.get("rowspan", 1) > 1 for v in span_map.values()
    )
    prov = {
        "fully_cellular": not has_spans,
        "original_headers": list(original_headers),
        "reconstructed_headers": list(reconstructed_headers),
        "original_grid": [list(r) for r in original_grid],
        "document_id": document_id,
        "unavailable_regions": unavail_regions,
        "unavailable_region_coverage": coverage,
        "cell_bboxes_available": bool(bbox_map),
        "cell_spans_available": bool(span_map),
        "has_spans": has_spans,
    }
    if provenance:
        prov.update(provenance)
    return ExtractedTable(
        table_id=table_id,
        page=page,
        bbox=bbox,
        caption=normalize_ws(caption),
        headers=reconstructed_headers,
        header_rows=header_rows,
        rows=body,
        cells=cells,
        pages=[page],
        extraction_method=extraction_method,
        confidence=(conf / 100.0) if conf > 1 else conf,
        warnings=warnings,
        provenance=prov,
    )


def reconstruct_from_words(
    words: Sequence[Word],
    *,
    table_id: str,
    page: int,
    bbox: BBox,
    page_width: float,
    caption: str = "",
    extraction_method: str = "geometry",
    document_id: str = "",
) -> ExtractedTable | None:
    grid, _anchors = grid_from_words(words, page_width=page_width)
    mean_conf = (
        sum(w.confidence for w in words) / len(words) if words else 0.0
    )
    return reconstruct_from_grid(
        grid,
        table_id=table_id,
        page=page,
        bbox=bbox,
        caption=caption,
        extraction_method=extraction_method,
        confidence=mean_conf,
        provenance={"word_count": len(words)},
        document_id=document_id,
    )


def native_tables_from_page(page) -> List[tuple[BBox, List[List[str]]]]:
    out: List[tuple[BBox, List[List[str]]]] = []
    try:
        if not hasattr(page, "find_tables"):
            return out
        finder = page.find_tables()
        for t in list(getattr(finder, "tables", []) or []):
            try:
                b = t.bbox
                bbox = BBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                raw = t.extract()
                grid: List[List[str]] = []
                for row in raw or []:
                    grid.append(
                        [normalize_ws("" if c is None else str(c)) for c in row]
                    )
                if len(grid) >= 2 and max(len(r) for r in grid) >= 2:
                    out.append((bbox, grid))
            except Exception:
                continue
    except Exception:
        return out
    return out


def caption_near_region(page_text_lines: Sequence[tuple[str, float]], region: BBox) -> str:
    candidates = []
    for text, y in page_text_lines:
        if y >= region.y0:
            continue
        if region.y0 - y > 80:
            continue
        if re.search(r"(?i)\btable\s+\d+", text):
            candidates.append((region.y0 - y, text))
    if not candidates:
        return ""
    candidates.sort(key=lambda t: t[0])
    return normalize_ws(candidates[0][1])

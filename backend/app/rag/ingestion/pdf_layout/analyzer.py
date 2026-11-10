"""Coordinate-based layout analysis: lines, columns, tables, headers/footers.

The analyzer deliberately works from positioned words. Tesseract's own reading
order is not trusted for multi-column pages or tables; order is reconstructed
from coordinates and page structure.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pymupdf as fitz

from app.config import get_settings
from app.rag.ingestion.pdf_layout.types import (
    BBox,
    TableData,
    TextBlock,
    TextLine,
    TextWord,
    normalize_ws,
)

logger = logging.getLogger(__name__)

_LIST_RE = re.compile(r"^\s*(?:[•●▪◦*\-–—]|\(?[a-zA-Z]\)|\(?\d{1,2}[.)])\s+\S")
_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?|table|exhibit|chart|graph|plate)\s+[A-Z0-9]", re.I)
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?[ivxlcdm]{1,6}\s*$|^\s*\d{1,4}\s*$", re.I)


def median(values: Sequence[float]) -> float:
    values = [float(v) for v in values if v is not None]
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def is_list_line(text: str) -> bool:
    return bool(_LIST_RE.match(text or ""))


def is_caption_line(text: str) -> bool:
    return bool(_CAPTION_RE.match(text or ""))


def _words_in_bbox(words: Iterable[TextWord], bbox: BBox) -> List[TextWord]:
    return [w for w in words if bbox.contains(w.bbox) or bbox.intersection_area(w.bbox) > 0.5 * w.bbox.area]


def group_words_by_rows(words: Sequence[TextWord], page_width: float) -> List[List[TextWord]]:
    """Group words into visual rows, then split rows at large horizontal gaps."""
    if not words:
        return []
    heights = [w.bbox.height for w in words if w.bbox.height > 0]
    tolerance = max(2.5, 0.55 * median(heights))
    ordered = sorted(words, key=lambda w: (w.bbox.center_y, w.bbox.x0))
    rows: List[List[TextWord]] = []
    row_centers: List[float] = []
    for word in ordered:
        center = word.bbox.center_y
        if rows and abs(center - row_centers[-1]) <= tolerance:
            rows[-1].append(word)
            row_centers[-1] = sum(w.bbox.center_y for w in rows[-1]) / len(rows[-1])
        else:
            rows.append([word])
            row_centers.append(center)
    for row in rows:
        row.sort(key=lambda w: w.bbox.x0)
    return rows


def split_row_into_cells(row: Sequence[TextWord], page_width: float) -> List[TextLine]:
    """Split a visual row at gutters wide enough to be column/cell boundaries."""
    if not row:
        return []
    gap_threshold = max(9.0, page_width * 0.018)
    cells: List[List[TextWord]] = [[row[0]]]
    for word in row[1:]:
        previous = cells[-1][-1]
        gap = word.bbox.x0 - previous.bbox.x1
        if gap > gap_threshold:
            cells.append([word])
        else:
            cells[-1].append(word)
    lines: List[TextLine] = []
    for cell in cells:
        text = normalize_ws(" ".join(w.text for w in cell))
        if not text:
            continue
        lines.append(
            TextLine(
                words=list(cell),
                page=cell[0].page,
                font_size=median([w.bbox.height for w in cell]),
                source="ocr",
            )
        )
    return lines


def words_to_lines(words: Sequence[TextWord], page_width: float) -> List[TextLine]:
    rows = group_words_by_rows(words, page_width)
    lines: List[TextLine] = []
    for row in rows:
        lines.extend(split_row_into_cells(row, page_width))
    lines.sort(key=lambda line: (line.bbox.center_y, line.bbox.x0))
    return lines


def detect_column_gutter(lines: Sequence[TextLine], page_width: float, page_height: float) -> Optional[float]:
    """Detect a dominant vertical gutter for two-column prose.

    Only prose-like lines away from the page margins vote. A candidate must have
    substantial text on both sides and few lines crossing it, which prevents
    table rows and full-width headings from creating false columns.
    """
    if len(lines) < 8 or page_width <= 0:
        return None
    votes = [
        line for line in lines
        if len(line.words) >= 3
        and line.bbox.width > page_width * 0.12
        and line.bbox.y0 > page_height * 0.08
        and line.bbox.y1 < page_height * 0.92
        and line.words
        and sum(1 for w in line.words if w.text[:1].isdigit()) / len(line.words) < 0.45
    ]
    if len(votes) < 8:
        return None

    # Track the winning X explicitly. Returning the loop's terminal x always
    # collapsed to ~0.72*page_width and interleaved two-column prose.
    best_score: Optional[float] = None
    best_x: Optional[float] = None
    step = max(2.0, page_width * 0.01)
    x = page_width * 0.28
    while x <= page_width * 0.72:
        left = [line for line in votes if line.bbox.x1 <= x]
        right = [line for line in votes if line.bbox.x0 >= x]
        crossing = [line for line in votes if line.bbox.x0 < x - 1 and line.bbox.x1 > x + 1]
        if len(left) >= 3 and len(right) >= 3 and len(crossing) <= max(2, int(len(votes) * 0.12)):
            score = float(min(len(left), len(right)) * 2 - len(crossing))
            # Prefer a gutter nearer the geometric gap between columns when tied.
            if left and right:
                gap_center = (max(line.bbox.x1 for line in left) + min(line.bbox.x0 for line in right)) / 2.0
                score -= abs(x - gap_center) / page_width
            if best_score is None or score > best_score:
                best_score = score
                best_x = x
        x += step
    return round(best_x, 2) if best_x is not None else None


def _column_index(bbox: BBox, gutter: Optional[float]) -> int:
    if gutter is None:
        return 0
    return 1 if bbox.center_x > gutter else 0


def _crosses_gutter(bbox: BBox, gutter: Optional[float], tolerance: float = 4.0) -> bool:
    if gutter is None:
        return False
    return bbox.x0 < gutter - tolerance and bbox.x1 > gutter + tolerance


def _looks_like_heading_line(line: TextLine, body_size: float) -> bool:
    text = line.text
    if not text:
        return False
    size = line.font_size or line.height
    if size >= max(12.0, body_size * 1.35):
        return True
    letters = [c for c in text if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    if letters and upper / len(letters) >= 0.82 and len(text) <= 100:
        return True
    if re.match(r"^(chapter|appendix|part|section)\b", text, re.I) and len(text) <= 120:
        return True
    if re.match(r"^\d+(?:\.\d+){1,3}\s+\S", text) and len(text) <= 120:
        return True
    return False


def merge_lines_to_blocks(
    lines: Sequence[TextLine],
    page_width: float,
    gutter: Optional[float] = None,
    body_size: float = 10.0,
) -> List[TextBlock]:
    """Merge adjacent lines in the same column into paragraph-like blocks."""
    if not lines:
        return []
    heights = [line.height for line in lines if line.height > 0]
    line_height = median(heights) or 10.0
    # With a gutter: read each column top-to-bottom independently so adjacent
    # Y-neighbors from opposite columns never form one interleaved block.
    if gutter is not None:
        ordered = sorted(
            lines,
            key=lambda line: (
                0 if _crosses_gutter(line.bbox, gutter) else 1,
                _column_index(line.bbox, gutter),
                line.bbox.center_y,
                line.bbox.x0,
            ),
        )
    else:
        ordered = sorted(lines, key=lambda line: (line.bbox.center_y, line.bbox.x0))
    blocks: List[TextBlock] = []
    current: List[TextLine] = []
    current_column = 0

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(TextBlock(lines=list(current), page=current[0].page, column=current_column))
            current = []

    for line in ordered:
        column = _column_index(line.bbox, gutter)
        heading = _looks_like_heading_line(line, body_size)
        if not current:
            current = [line]
            current_column = column
            continue
        previous = current[-1]
        same_column = column == current_column and not _crosses_gutter(line.bbox, gutter)
        vertical_gap = line.bbox.y0 - previous.bbox.y1
        overlap = min(line.bbox.x1, previous.bbox.x1) - max(line.bbox.x0, previous.bbox.x0)
        narrower = max(1.0, min(line.bbox.width, previous.bbox.width))
        aligned = overlap / narrower >= 0.25
        if (
            same_column
            and aligned
            and -2.0 <= vertical_gap <= max(line_height * 2.0, 24.0)
            and not heading
            and not _looks_like_heading_line(previous, body_size)
        ):
            current.append(line)
        else:
            flush()
            current = [line]
            current_column = column
    flush()
    return blocks


def order_objects(objects: Sequence[Any], gutter: Optional[float], page_width: float) -> List[Any]:
    """Order text blocks/tables by bands and columns (top-down, left-to-right)."""
    if not objects:
        return []
    tolerance = max(4.0, page_width * 0.008)
    if gutter is None:
        return sorted(objects, key=lambda obj: (obj.bbox.y0, obj.bbox.x0))

    full_width = [
        obj for obj in objects
        if _crosses_gutter(obj.bbox, gutter, tolerance) or obj.bbox.width >= page_width * 0.72
    ]
    full_ids = {id(obj) for obj in full_width}
    others = [obj for obj in objects if id(obj) not in full_ids]
    full_width.sort(key=lambda obj: (obj.bbox.y0, obj.bbox.x0))

    bands: List[List[Any]] = []
    boundaries = [obj.bbox.y0 for obj in full_width]
    for index in range(len(boundaries) + 1):
        low = boundaries[index - 1] if index > 0 else -math.inf
        high = boundaries[index] if index < len(boundaries) else math.inf
        band = [obj for obj in others if low <= obj.bbox.center_y < high]
        if band:
            left = [obj for obj in band if obj.bbox.center_x <= gutter]
            right = [obj for obj in band if obj.bbox.center_x > gutter]
            left.sort(key=lambda obj: (obj.bbox.y0, obj.bbox.x0))
            right.sort(key=lambda obj: (obj.bbox.y0, obj.bbox.x0))
            bands.append(left + right)
        else:
            bands.append([])

    ordered: List[Any] = []
    for index, band in enumerate(bands):
        ordered.extend(band)
        if index < len(full_width):
            ordered.append(full_width[index])
    return ordered


# ── Table detection ─────────────────────────────────────────────────────────


def _page_image(page: fitz.Page, dpi: int) -> Tuple[Any, float, float]:
    import numpy as np

    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        image = image[:, :, :3]
    gray = image.mean(axis=2).astype("uint8") if image.ndim == 3 else image
    return gray, pix.width, pix.height


def _line_masks(gray: Any) -> Tuple[Any, Any]:
    import cv2
    import numpy as np

    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = binary.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 30), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 30)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    return horizontal, vertical


def detect_ruled_table_boxes(page: fitz.Page, dpi: int = 200) -> List[BBox]:
    """Find ruled table regions using horizontal/vertical line morphology."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    try:
        gray, px_w, px_h = _page_image(page, dpi)
        horizontal, vertical = _line_masks(gray)
        combined = cv2.add(horizontal, vertical)
        combined = cv2.dilate(combined, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception as exc:
        logger.debug("Ruled-table detection failed on page %s: %s", page.number + 1, exc)
        return []

    page_w, page_h = float(page.rect.width), float(page.rect.height)
    sx = page_w / max(1, px_w)
    sy = page_h / max(1, px_h)
    page_area = page_w * page_h
    boxes: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = BBox(x * sx, y * sy, (x + w) * sx, (y + h) * sy)
        if bbox.width < page_w * 0.18 or bbox.height < page_h * 0.025:
            continue
        if bbox.area < page_area * 0.008 or bbox.area > page_area * 0.72:
            continue
        # A real table must contain at least two horizontal and two vertical lines.
        x0, x1 = max(0, x), min(px_w, x + w)
        y0, y1 = max(0, y), min(px_h, y + h)
        h_count = cv2.countNonZero(horizontal[y0:y1, x0:x1])
        v_count = cv2.countNonZero(vertical[y0:y1, x0:x1])
        if h_count < max(30, w * 2) or v_count < max(20, h * 2):
            continue
        boxes.append(bbox)
    boxes.sort(key=lambda b: (b.y0, b.x0))
    # Remove duplicates/near-duplicates.
    unique: List[BBox] = []
    for box in boxes:
        if not any(box.iou(other) >= 0.7 for other in unique):
            unique.append(box)
    return unique


def extract_grid_positions(
    page: fitz.Page,
    bbox: BBox,
    *,
    dpi: int = 200,
) -> Tuple[List[float], List[float]]:
    """Return vertical and horizontal rule positions inside a table bbox."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return [], []
    try:
        gray, px_w, px_h = _page_image(page, dpi)
        horizontal, vertical = _line_masks(gray)
        sx = float(page.rect.width) / max(1, px_w)
        sy = float(page.rect.height) / max(1, px_h)
        x0 = max(0, int(bbox.x0 / sx))
        x1 = min(px_w, int(bbox.x1 / sx))
        y0 = max(0, int(bbox.y0 / sy))
        y1 = min(px_h, int(bbox.y1 / sy))
        if x1 <= x0 or y1 <= y0:
            return [], []
        h_crop = horizontal[y0:y1, x0:x1]
        v_crop = vertical[y0:y1, x0:x1]
        v_score = (v_crop > 0).sum(axis=0)
        h_score = (h_crop > 0).sum(axis=1)
        xs = _peaks(v_score, threshold=max(5, h_crop.shape[0] * 0.25))
        ys = _peaks(h_score, threshold=max(5, v_crop.shape[1] * 0.25))
        return [round((x0 + x) * sx, 2) for x in xs], [round((y0 + y) * sy, 2) for y in ys]
    except Exception:
        return [], []


def _peaks(scores: Any, threshold: float) -> List[int]:
    peaks: List[int] = []
    for index, value in enumerate(scores):
        if value < threshold:
            continue
        if index > 0 and scores[index - 1] > value:
            continue
        if index + 1 < len(scores) and scores[index + 1] > value:
            continue
        if peaks and index - peaks[-1] <= 2:
            continue
        peaks.append(index)
    return peaks


def table_from_words(
    words: Sequence[TextWord],
    bbox: BBox,
    *,
    page: int,
    page_width: float,
    page_height: float,
    grid_x: Optional[Sequence[float]] = None,
    grid_y: Optional[Sequence[float]] = None,
    method: str = "alignment",
) -> Optional[TableData]:
    """Reconstruct a row/column table from words inside a region.

    Grid lines are used when available; otherwise cells are inferred from
    horizontal whitespace and aligned across rows. Values are never normalized
    or rounded, so financial figures remain exact.
    """
    region = [w for w in words if bbox.contains(w.bbox) and w.text.strip()]
    if len(region) < 4:
        return None

    rows = group_words_by_rows(region, page_width)
    if len(rows) < 2:
        return None

    if grid_x and len(grid_x) >= 2 and grid_y and len(grid_y) >= 2:
        xs = sorted(grid_x)
        ys = sorted(grid_y)
        grid_rows: List[List[str]] = []
        for r_index in range(len(ys) - 1):
            top, bottom = ys[r_index], ys[r_index + 1]
            row_cells = [""] * max(1, len(xs) - 1)
            for word in region:
                cy = word.bbox.center_y
                if not (top <= cy <= bottom):
                    continue
                cx = word.bbox.center_x
                c_index = None
                for c in range(len(xs) - 1):
                    if xs[c] <= cx <= xs[c + 1]:
                        c_index = c
                        break
                if c_index is not None:
                    row_cells[c_index] = normalize_ws(row_cells[c_index] + " " + word.text)
            if any(row_cells):
                grid_rows.append(row_cells)
        if len(grid_rows) >= 2:
            return _table_from_rows(grid_rows, bbox, page, method, len(region))

    cell_rows: List[List[TextLine]] = []
    for row in rows:
        cells = split_row_into_cells(row, page_width)
        if cells:
            cell_rows.append(cells)

    if len(cell_rows) < 2:
        return None

    # Infer column anchors from cell centers, requiring repeated positions.
    tolerance = max(7.0, page_width * 0.012)
    centers: List[float] = [cell.bbox.center_x for row in cell_rows for cell in row]
    anchors: List[float] = []
    for center in sorted(centers):
        if not anchors or abs(center - anchors[-1]) > tolerance:
            anchors.append(center)
        else:
            anchors[-1] = (anchors[-1] + center) / 2.0
    anchors = [
        anchor for anchor in anchors
        if sum(1 for center in centers if abs(center - anchor) <= tolerance) >= 2
    ]
    if len(anchors) < 2:
        max_cells = max(len(row) for row in cell_rows)
        if max_cells < 2:
            return None
        anchors = [
            page_width * ((index + 0.5) / max_cells)
            for index in range(max_cells)
        ]

    matrix: List[List[str]] = []
    confidence_matrix: List[List[float]] = []
    for row in cell_rows:
        values = [""] * len(anchors)
        confidences = [100.0] * len(anchors)
        for cell in row:
            center = cell.bbox.center_x
            index = min(range(len(anchors)), key=lambda i: abs(anchors[i] - center))
            if abs(anchors[index] - center) > page_width * 0.09:
                continue
            values[index] = normalize_ws(values[index] + " " + cell.text)
            confidences[index] = min(confidences[index], float(cell.confidence or 100.0))
        if any(values):
            matrix.append(values)
            confidence_matrix.append(confidences)

    if len(matrix) < 2:
        return None
    return _table_from_rows(
        matrix,
        bbox,
        page,
        method,
        len(region),
        confidences=confidence_matrix,
    )


def _table_from_rows(
    rows: List[List[str]],
    bbox: BBox,
    page: int,
    method: str,
    source_lines: int,
    confidences: Optional[Sequence[Sequence[float]]] = None,
) -> Optional[TableData]:
    cleaned: List[List[str]] = []
    cleaned_conf: List[List[float]] = []
    for row_index, row in enumerate(rows):
        values = [normalize_ws(cell) for cell in row]
        if not any(values):
            continue
        cleaned.append(values)
        if confidences and row_index < len(confidences):
            conf_row = list(confidences[row_index]) + [100.0] * max(0, len(values) - len(confidences[row_index]))
            cleaned_conf.append(conf_row[: len(values)])
        else:
            cleaned_conf.append([100.0] * len(values))
    if len(cleaned) < 2:
        return None
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    cleaned_conf = [row + [100.0] * (width - len(row)) for row in cleaned_conf]
    # Drop columns that are entirely empty.
    keep = [i for i in range(width) if any(row[i] for row in cleaned)]
    if len(keep) < 2:
        return None
    cleaned = [[row[i] for i in keep] for row in cleaned]
    cleaned_conf = [[row[i] for i in keep] for row in cleaned_conf]

    # Only collapse columns that stayed completely empty after keep[]. Do not
    # merge sparse-but-valued company columns (e.g. U.S. Rubber vs Firestone).
    cleaned, cleaned_conf = _drop_trailing_spacer_columns(cleaned, cleaned_conf)

    first = cleaned[0]
    numeric_first = sum(1 for cell in first if _numeric_like(cell))
    headers = first if numeric_first <= len(first) * 0.4 else []
    data = cleaned[1:] if headers else cleaned
    data_conf = cleaned_conf[1:] if headers else cleaned_conf
    if not data:
        return None

    low_confidence_cells: List[Dict[str, Any]] = []
    try:
        from app.config import get_settings

        min_conf = float(get_settings().PDF_OCR_MIN_CONFIDENCE or 20.0)
    except Exception:
        min_conf = 20.0
    # Financial values need higher trust than the global OCR floor.
    finance_conf = max(min_conf, 55.0)
    for r_index, row in enumerate(data):
        for c_index, cell in enumerate(row):
            if not cell or not _numeric_like(cell):
                continue
            conf = data_conf[r_index][c_index] if r_index < len(data_conf) else 100.0
            if conf < finance_conf:
                low_confidence_cells.append(
                    {
                        "row": r_index,
                        "col": c_index,
                        "text": cell,
                        "confidence": round(float(conf), 2),
                    }
                )

    return TableData(
        page=page,
        bbox=bbox,
        headers=headers,
        rows=data,
        extraction_method=method,
        source_lines=source_lines,
        low_confidence_cells=low_confidence_cells,
    )


def _drop_trailing_spacer_columns(
    rows: List[List[str]],
    confidences: List[List[float]],
) -> Tuple[List[List[str]], List[List[float]]]:
    """Drop leading/trailing all-empty columns left by OCR alignment noise."""
    if not rows:
        return rows, confidences
    width = len(rows[0])
    keep = [i for i in range(width) if any(row[i] for row in rows)]
    if len(keep) < 2:
        return rows, confidences
    return (
        [[row[i] for i in keep] for row in rows],
        [[row[i] for i in keep] for row in confidences],
    )


def _numeric_like(value: str) -> bool:
    text = normalize_ws(value)
    if not text:
        return False
    digits = sum(1 for ch in text if ch.isdigit())
    return digits >= max(1, len(text) // 3) and bool(re.search(r"\d", text))


def alignment_table_regions(
    lines: Sequence[TextLine],
    page_width: float,
    page_height: float,
    *,
    excluded: Sequence[BBox] = (),
) -> List[Tuple[BBox, List[TextLine]]]:
    """Find borderless ledger-style tables from short, aligned, numeric rows."""
    candidates = [
        line for line in lines
        if not any(box.intersection_area(line.bbox) > 0.5 * line.bbox.area for box in excluded)
        and line.text
    ]
    if len(candidates) < 6:
        return []
    rows = group_words_by_rows([w for line in candidates for w in line.words], page_width)
    row_cells: List[List[TextLine]] = []
    for row in rows:
        row_cells.append(split_row_into_cells(row, page_width))

    regions: List[Tuple[BBox, List[TextLine]]] = []
    current: List[List[TextLine]] = []
    current_y: List[float] = []

    def flush() -> None:
        nonlocal current, current_y
        if len(current) >= 4:
            multi = sum(1 for row in current if len(row) >= 2)
            all_cells = [cell for row in current for cell in row]
            short = sum(1 for cell in all_cells if len(cell.text) <= 40)
            numeric_cells = sum(1 for cell in all_cells if _numeric_like(cell.text))
            x0 = min(cell.bbox.x0 for cell in all_cells)
            x1 = max(cell.bbox.x1 for cell in all_cells)
            if (
                multi >= max(3, int(len(current) * 0.6))
                and all_cells
                and short / len(all_cells) >= 0.7
                and numeric_cells / len(all_cells) >= 0.18
                and (x1 - x0) >= page_width * 0.35
            ):
                bbox = BBox(x0, min(cell.bbox.y0 for cell in all_cells), x1, max(cell.bbox.y1 for cell in all_cells))
                regions.append((bbox, [cell for row in current for cell in row]))
        current = []
        current_y = []

    heights = [line.height for line in candidates if line.height > 0]
    tolerance = max(3.0, median(heights) * 2.2)
    for row in row_cells:
        if not row:
            continue
        y = median([cell.bbox.center_y for cell in row])
        if current_y and abs(y - current_y[-1]) > tolerance:
            flush()
        current.append(row)
        current_y.append(y)
    flush()
    return regions


def detect_repeating_margins(
    pages_lines: Dict[int, List[TextLine]],
    page_height: float,
) -> Tuple[Dict[int, List[TextLine]], Dict[str, Any]]:
    """Remove confidently repeated running headers/footers and page numbers."""
    if not pages_lines:
        return pages_lines, {"removed_lines": 0, "patterns": []}

    top_zone = page_height * 0.085
    bottom_zone = page_height * 0.915
    occurrences: Counter = Counter()
    page_hits: Dict[str, set] = {}
    for page_num, lines in pages_lines.items():
        for line in lines:
            zone = "top" if line.bbox.y1 <= top_zone else "bottom" if line.bbox.y0 >= bottom_zone else ""
            if not zone:
                continue
            key = _margin_key(line.text)
            if not key:
                continue
            occurrences[key] += 1
            page_hits.setdefault(key, set()).add(page_num)

    threshold = max(3, int(math.ceil(len(pages_lines) * 0.5)))
    repeated = {
        key for key, pages in page_hits.items()
        if len(pages) >= threshold and any(ch.isalpha() for ch in key)
    }

    removed = 0
    patterns: List[str] = []
    cleaned: Dict[int, List[TextLine]] = {}
    for page_num, lines in pages_lines.items():
        kept: List[TextLine] = []
        for line in lines:
            in_margin = line.bbox.y1 <= top_zone or line.bbox.y0 >= bottom_zone
            key = _margin_key(line.text)
            is_page_number = bool(_PAGE_NUMBER_RE.match(line.text)) and in_margin
            if in_margin and (key in repeated or is_page_number):
                removed += 1
                if key and key not in patterns:
                    patterns.append(key)
                continue
            kept.append(line)
        cleaned[page_num] = kept

    return cleaned, {
        "removed_lines": removed,
        "repeated_patterns": patterns[:20],
        "threshold_pages": threshold,
    }


def _margin_key(text: str) -> str:
    value = normalize_ws(text).lower()
    value = re.sub(r"\d+", "#", value)
    value = re.sub(r"[^a-z#]+", " ", value)
    return normalize_ws(value)

"""AWS Textract DetectDocumentText → geometry tables → TableData.

Active production table extractor (Docling remains in-tree but disabled by
default). Reconstructs row/column structure from WORD/LINE geometry because
DetectDocumentText does **not** return native TABLE/CELL blocks.

Modes
-----
* Multipage PDFs (preferred): ``StartDocumentTextDetection`` /
  ``GetDocumentTextDetection`` with S3 + NextToken pagination.
* Single-page / per-page fallback: sync ``DetectDocumentText`` on a PNG.

Never calls ``AnalyzeDocument`` (TABLES). Cost ~$1.50 / 1,000 pages.
IAM: ``textract:DetectDocumentText``, ``textract:StartDocumentTextDetection``,
``textract:GetDocumentTextDetection``.

On client/API errors the parser records ``textract_error`` and leaves tables
empty — there is no img2table or Tesseract table fallback. Body text is never
taken from Textract.
"""
from __future__ import annotations

import logging
import random
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from botocore.config import Config

from app.config import get_settings
from app.rag.ingestion.pdf_layout.analyzer import (
    alignment_table_regions,
    detect_ruled_table_boxes,
    extract_grid_positions,
    table_from_words,
    words_to_lines,
)
from app.rag.ingestion.pdf_layout.header_reconstruction import (
    reconstruct_table_headers_cached,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData, TextWord, normalize_ws

logger = logging.getLogger(__name__)

_textract_client = None
_EXTRACTION_METHOD = "textract_detect"

# Process-local API usage counters for cost / audit reporting.
_API_USAGE: Dict[str, int] = {
    "detect_document_text_pages": 0,
    "start_document_text_detection_jobs": 0,
    "get_document_text_detection_calls": 0,
    "analyze_document_tables_pages": 0,  # must remain 0
    "pages_via_async": 0,
    "pages_via_sync": 0,
}


def get_textract_api_usage() -> Dict[str, int]:
    """Return a copy of DetectDocumentText API usage counters."""
    return dict(_API_USAGE)


def reset_textract_api_usage() -> None:
    """Test helper — clear API usage counters."""
    for key in list(_API_USAGE.keys()):
        _API_USAGE[key] = 0


def _textract_region() -> str:
    settings = get_settings()
    region = (settings.PDF_TEXTRACT_REGION or "").strip()
    if region:
        return region
    return (
        (settings.BEDROCK_REGION or "").strip()
        or __import__("os").environ.get("AWS_REGION", "")
        or __import__("os").environ.get("AWS_DEFAULT_REGION", "")
        or "us-east-1"
    )


def _textract_dpi() -> int:
    settings = get_settings()
    dpi = int(settings.PDF_TEXTRACT_DPI or 0)
    if dpi > 0:
        return dpi
    return max(72, int(settings.PDF_OCR_DPI or 300))


def get_textract_client():
    """Lazy boto3 Textract client (ambient AWS credentials)."""
    global _textract_client
    if _textract_client is not None:
        return _textract_client
    settings = get_settings()
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for Textract tables. Install with: pip install boto3"
        ) from exc
    timeout = max(5, int(settings.PDF_TEXTRACT_TIMEOUT_SECONDS or 60))
    cfg = Config(
        read_timeout=timeout,
        connect_timeout=10,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    _textract_client = boto3.client(
        "textract", region_name=_textract_region(), config=cfg
    )
    return _textract_client


def reset_textract_client() -> None:
    """Test helper to clear the cached client."""
    global _textract_client
    _textract_client = None


def _geometry_to_pdf_bbox(
    geometry: Dict[str, Any],
    *,
    page_width: float,
    page_height: float,
) -> BBox:
    box = (geometry or {}).get("BoundingBox") or {}
    left = float(box.get("Left") or 0.0)
    top = float(box.get("Top") or 0.0)
    width = float(box.get("Width") or 0.0)
    height = float(box.get("Height") or 0.0)
    return BBox(
        left * page_width,
        top * page_height,
        (left + width) * page_width,
        (top + height) * page_height,
    )


def words_from_detect_blocks(
    blocks: Sequence[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    min_confidence: float = 0.0,
) -> List[TextWord]:
    """Convert DetectDocumentText WORD blocks into positioned TextWord objects."""
    words: List[TextWord] = []
    for block in blocks:
        if block.get("BlockType") != "WORD":
            continue
        text = normalize_ws(str(block.get("Text") or ""))
        if not text:
            continue
        conf = float(block.get("Confidence") or 0.0)
        if conf < min_confidence:
            continue
        bbox = _geometry_to_pdf_bbox(
            block.get("Geometry") or {},
            page_width=page_width,
            page_height=page_height,
        )
        if bbox.area <= 0:
            continue
        words.append(
            TextWord(
                text=text,
                bbox=bbox,
                page=page_number,
                confidence=conf,
            )
        )
    return words


def lines_from_detect_blocks(
    blocks: Sequence[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
) -> List[Tuple[str, BBox]]:
    """LINE blocks with PDF bboxes (region hints / diagnostics)."""
    lines: List[Tuple[str, BBox]] = []
    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue
        text = normalize_ws(str(block.get("Text") or ""))
        if not text:
            continue
        bbox = _geometry_to_pdf_bbox(
            block.get("Geometry") or {},
            page_width=page_width,
            page_height=page_height,
        )
        if bbox.area <= 0:
            continue
        lines.append((text, bbox))
    return lines


_BANNER_TOKEN_RE = re.compile(
    r"(?i)^(?:dose|value|amount|unit|rate|freq(?:uency)?|"
    r"total|count|%|\(\s*[a-zμµ/%]{1,6}\s*\))$"
)


def _is_label_banner_row(row: Sequence[str]) -> bool:
    """Detect non-data banner rows like blank|blank|Dose|(mg)|blank."""
    cells = [normalize_ws(c) for c in row]
    nonempty = [c for c in cells if c]
    if len(nonempty) < 2 or len(nonempty) > max(3, len(cells) // 3):
        return False
    if cells and cells[0]:
        return False
    for cell in nonempty:
        if re.search(r"\d", cell) and not cell.startswith("("):
            return False
        if not _BANNER_TOKEN_RE.match(cell):
            return False
    return True


def _leaf_headers(header_rows: Sequence[Sequence[str]]) -> List[str]:
    if not header_rows:
        return []
    width = max(len(r) for r in header_rows)
    if width == 0:
        return []
    padded = [list(r) + [""] * (width - len(r)) for r in header_rows]
    if len(padded) == 1:
        return [normalize_ws(c) for c in padded[0]]
    leaves: List[str] = []
    for c in range(width):
        parts: List[str] = []
        for r in padded:
            val = normalize_ws(r[c])
            if val and (not parts or parts[-1] != val):
                parts.append(val)
        if not parts:
            leaves.append(f"Column {c + 1}")
            continue
        child = parts[-1]
        if len(parts) >= 2 and re.search(r"\d", child) and not re.search(r"\d", parts[0]):
            leaves.append(child)
        else:
            leaves.append(" > ".join(parts))
    return leaves


def _merge_soft_continuation_rows(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    """Merge multi-line cell wraps where a following row continues empty columns."""
    if not rows:
        return []
    out: List[List[str]] = [list(rows[0])]
    for raw in rows[1:]:
        prev = list(out[-1])
        row = list(raw)
        width = max(len(prev), len(row))
        prev.extend([""] * (width - len(prev)))
        row.extend([""] * (width - len(row)))

        nonempty_idx = [i for i, c in enumerate(row) if normalize_ws(c)]
        # Only merge when the leading key cell is empty — a true multi-line wrap.
        if not nonempty_idx or normalize_ws(row[0]):
            out.append(row)
            continue

        if all(not normalize_ws(prev[i]) for i in nonempty_idx):
            for i in nonempty_idx:
                prev[i] = normalize_ws(row[i])
            out[-1] = prev
            continue

        # Soft wrap into prior cells that are empty or hyphen-truncated.
        if all(
            (not normalize_ws(prev[i])) or prev[i].endswith("-")
            for i in nonempty_idx
        ):
            for i in nonempty_idx:
                left = normalize_ws(prev[i])
                right = normalize_ws(row[i])
                if not right:
                    continue
                if left.endswith("-"):
                    prev[i] = left[:-1] + right
                else:
                    prev[i] = normalize_ws(f"{left} {right}" if left else right)
            out[-1] = prev
            continue

        out.append(row)
    return out


def _pad_row(row: Sequence[str], width: int) -> List[str]:
    values = [normalize_ws(c) for c in row]
    if len(values) < width:
        values.extend([""] * (width - len(values)))
    return values[:width]


_PURE_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_HEADERISH_TOKEN_RE = re.compile(
    r"(?i)(?:\b(?:kg|g|mg|ml|l|%|usd|eur|mpa|psi|hz|v|a|w)\b|"
    r"freq|unit|value|amount|rate|total|weight|weeks?|months?|"
    r"period|grade|size|range|column\s+\d+|\(\s*[a-z]{1,6}\s*\))"
)


def _is_data_row(row: Sequence[str]) -> bool:
    """True when a row looks like table body (dose integers / IgE bins)."""
    cells = [normalize_ws(c) for c in row]
    nonempty = [c for c in cells if c]
    if not nonempty:
        return False
    pure_nums = sum(1 for c in nonempty if _PURE_NUM_RE.match(c))
    if pure_nums >= 2:
        return True
    # IgE/range key + at least one numeric dose.
    if cells and re.search(r"\d", cells[0]) and pure_nums >= 1:
        if re.search(r"[<>≥≤]|-|to", cells[0], re.I):
            return True
    return False


def _is_weak_header_label(text: str) -> bool:
    value = normalize_ws(text)
    if not value:
        return True
    return bool(re.match(r"(?i)^column\s+\d+$", value))


def _drop_empty_body_columns(
    headers: List[str],
    header_rows: List[List[str]],
    body: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """Remove columns that never carry body values (alignment ghost cols)."""
    if not body:
        return headers, header_rows, body
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    headers = _pad_row(headers, width)
    header_rows = [_pad_row(r, width) for r in header_rows]
    body = [_pad_row(r, width) for r in body]
    keep = [
        i
        for i in range(width)
        if any(normalize_ws(row[i]) for row in body)
    ]
    if 0 not in keep and width > 0:
        keep = [0] + keep
    if len(keep) < 2:
        return headers, header_rows, body
    return (
        [headers[i] for i in keep],
        [[row[i] for i in keep] for row in header_rows],
        [[row[i] for i in keep] for row in body],
    )


def _promote_leading_header_bands(
    headers: List[str],
    header_rows: List[List[str]],
    body: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """Move leading non-data rows into header_rows for multi-row headers."""
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    headers = _pad_row(headers, width)
    header_rows = [_pad_row(r, width) for r in header_rows]
    body = [_pad_row(r, width) for r in body]

    # Seed header_rows from weak leaf headers produced by table_from_words.
    if not header_rows and headers and any(h for h in headers):
        header_rows = [list(headers)]
    elif headers and header_rows:
        # Prefer explicit header_rows; keep headers as leaf candidates later.
        pass

    promote_from = 0
    for index, row in enumerate(body):
        if _is_data_row(row):
            break
        nonempty = [c for c in row if c]
        headerish = sum(1 for c in nonempty if _HEADERISH_TOKEN_RE.search(c))
        # Sparse banner / label rows also stay in the header band.
        if nonempty and headerish == 0 and not any(_is_weak_header_label(c) for c in nonempty):
            # Short alphabetic labels (e.g. Dosing / Freq.) still count.
            alphaish = sum(1 for c in nonempty if re.search(r"[A-Za-z]", c) and not _PURE_NUM_RE.match(c))
            if alphaish < max(1, len(nonempty) // 2):
                break
        promote_from = index + 1

    if promote_from:
        header_rows = header_rows + [list(r) for r in body[:promote_from]]
        body = body[promote_from:]

    # Drop synthetic Column-N-only seed row when real header bands exist.
    if len(header_rows) >= 2 and header_rows[0]:
        if all(_is_weak_header_label(c) or not c for c in header_rows[0]):
            header_rows = header_rows[1:]
        elif sum(1 for c in header_rows[0] if c and not _is_weak_header_label(c)) <= 1:
            # e.g. ["Pretreatment", "", ...] above richer bands — keep it.
            pass

    leaves = _leaf_headers(header_rows) if header_rows else headers
    if not leaves and body:
        leaves = [f"Column {i + 1}" for i in range(width)]
    return leaves, header_rows, body


def _synthetic_cells(
    headers: Sequence[str],
    header_rows: Sequence[Sequence[str]],
    body: Sequence[Sequence[str]],
    *,
    table_bbox: BBox,
    mean_conf: float,
) -> List[Dict[str, Any]]:
    """Approximate per-cell metadata for downstream consumers."""
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    header_count = len(header_rows) if header_rows else (1 if headers else 0)
    total_rows = header_count + len(body)
    if total_rows <= 0 or width <= 0 or table_bbox.area <= 0:
        return []

    cell_w = table_bbox.width / width
    cell_h = table_bbox.height / total_rows
    cells: List[Dict[str, Any]] = []

    def _emit(ri: int, ci: int, text: str, is_header: bool) -> None:
        x0 = table_bbox.x0 + ci * cell_w
        y0 = table_bbox.y0 + ri * cell_h
        cells.append(
            {
                "row_index": ri,
                "col_index": ci,
                "text": normalize_ws(text),
                "confidence": mean_conf,
                "bbox": [x0, y0, x0 + cell_w, y0 + cell_h],
                "row_span": 1,
                "col_span": 1,
                "is_header": is_header,
            }
        )

    if header_rows:
        for ri, row in enumerate(header_rows):
            padded = _pad_row(row, width)
            for ci, text in enumerate(padded):
                _emit(ri, ci, text, True)
    elif headers:
        for ci, text in enumerate(_pad_row(headers, width)):
            _emit(0, ci, text, True)

    for bi, row in enumerate(body):
        ri = header_count + bi
        for ci, text in enumerate(_pad_row(row, width)):
            _emit(ri, ci, text, False)
    return cells


def _finalize_geometry_table(
    table: TableData,
    *,
    region_words: Sequence[TextWord],
    min_confidence: float,
) -> Optional[TableData]:
    """Banner cleanup, multiline merge, header reconstruction, metadata."""
    width = max(len(table.headers or []), max((len(r) for r in table.rows), default=0))
    if width < 2:
        return None

    headers = [normalize_ws(h) for h in (table.headers or [])]
    header_rows: List[List[str]] = (
        [list(r) for r in table.header_rows]
        if table.header_rows
        else ([list(headers)] if headers else [])
    )
    body = _merge_soft_continuation_rows(
        [_pad_row(r, width) for r in (table.rows or [])]
    )

    headers, header_rows, body = _promote_leading_header_bands(
        headers, header_rows, body
    )
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        width,
    )
    headers = _pad_row(headers, width)
    header_rows = [_pad_row(r, width) for r in header_rows]
    body = [_pad_row(r, width) for r in body]

    cleaned_body: List[List[str]] = []
    for row in body:
        if _is_label_banner_row(row):
            for ci, cell in enumerate(row):
                token = normalize_ws(cell)
                if not token:
                    continue
                if ci < len(headers) and not normalize_ws(headers[ci]):
                    headers[ci] = token
                elif ci < len(headers) and token.lower() not in headers[ci].lower():
                    if _BANNER_TOKEN_RE.match(token) and not re.search(r"\d", headers[ci]):
                        headers[ci] = f"{headers[ci]} {token}".strip()
            continue
        cleaned_body.append(row)
    body = cleaned_body
    headers, header_rows, body = _drop_empty_body_columns(
        headers, header_rows, body
    )
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    if header_rows:
        headers = _leaf_headers(header_rows)

    if not body and not headers:
        return None

    if header_rows and not any(headers):
        headers = _leaf_headers(header_rows)

    confidences = [float(w.confidence) for w in region_words if w.confidence is not None]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    recon = reconstruct_table_headers_cached(
        header_rows=header_rows,
        rows=body,
        cells=[],
        merged_cells=[],
        existing_headers=headers,
    )
    if recon.leaf_headers and len(recon.leaf_headers) == width:
        headers = list(recon.leaf_headers)
    if recon.reconstructed_header_rows:
        header_rows = [list(r) for r in recon.reconstructed_header_rows]
    if recon.reconstructed_rows:
        body = [list(r) for r in recon.reconstructed_rows]

    low_confidence_cells = list(table.low_confidence_cells or [])
    for w in region_words:
        if w.confidence < min_confidence and w.text:
            low_confidence_cells.append(
                {
                    "row": -1,
                    "col": -1,
                    "text": w.text,
                    "confidence": float(w.confidence),
                    "reason": "low_word_confidence",
                }
            )
    if recon.needs_verification:
        low_confidence_cells.append(
            {
                "row": -1,
                "col": -1,
                "text": "",
                "confidence": recon.normalization_confidence * 100.0,
                "reason": "header_reconstruction_uncertain",
            }
        )

    for ri, row in enumerate(body):
        empty = sum(1 for c in row[1:] if not normalize_ws(c))
        if row and empty >= max(2, (len(row) - 1) // 2):
            low_confidence_cells.append(
                {
                    "row": ri,
                    "col": -1,
                    "text": normalize_ws(row[0]) if row else "",
                    "confidence": mean_conf,
                    "reason": "sparse_row",
                }
            )

    cells = _synthetic_cells(
        headers,
        header_rows,
        body,
        table_bbox=table.bbox,
        mean_conf=mean_conf,
    )

    return TableData(
        page=table.page,
        bbox=table.bbox,
        headers=headers,
        rows=body,
        extraction_method=_EXTRACTION_METHOD,
        extraction_engine=_EXTRACTION_METHOD,
        caption=table.caption or "",
        source_lines=table.source_lines or len(region_words),
        low_confidence_cells=low_confidence_cells,
        cells=cells,
        header_rows=header_rows,
        merged_cells=[],
        textract_confidence=round(mean_conf, 2),
        ocr_confidence=round(mean_conf, 2),
        structure_confidence=float(
            recon.normalization_confidence if recon else 0.0
        ),
        provenance={
            "engine": _EXTRACTION_METHOD,
            "page": table.page,
            # DetectDocumentText has no native TABLE/CELL structure.
            "text_derived": True,
            "structurally_verified_table": False,
            "quality_notes": list((recon.warnings if recon else []) or []),
        },
        header_reconstruction=recon.to_dict(),
        normalized_header_paths=[list(p) for p in recon.normalized_header_paths],
        pages=list(table.pages or []),
    )


def _union_bbox(words: Sequence[TextWord]) -> Optional[BBox]:
    if not words:
        return None
    return BBox(
        min(w.bbox.x0 for w in words),
        min(w.bbox.y0 for w in words),
        max(w.bbox.x1 for w in words),
        max(w.bbox.y1 for w in words),
    )


def _candidate_regions(
    words: Sequence[TextWord],
    *,
    page: Any,
    page_width: float,
    page_height: float,
) -> List[Tuple[BBox, Optional[Sequence[float]], Optional[Sequence[float]]]]:
    """Ordered table regions: ruled boxes, then alignment clusters, then full page."""
    regions: List[Tuple[BBox, Optional[Sequence[float]], Optional[Sequence[float]]]] = []
    seen: List[BBox] = []

    def _add(
        bbox: BBox,
        grid_x: Optional[Sequence[float]] = None,
        grid_y: Optional[Sequence[float]] = None,
    ) -> None:
        if bbox.area <= 0:
            return
        for existing in seen:
            if existing.iou(bbox) >= 0.55 or existing.contains(bbox, tolerance=2.0):
                return
        seen.append(bbox)
        regions.append((bbox, grid_x, grid_y))

    if page is not None:
        try:
            for bbox in detect_ruled_table_boxes(page):
                grid_x, grid_y = extract_grid_positions(page, bbox)
                if len(grid_x) > 16 or len(grid_y) > 90:
                    grid_x, grid_y = [], []
                _add(bbox, grid_x or None, grid_y or None)
        except Exception as exc:
            logger.debug("Ruled table detection skipped: %s", exc)

    lines = words_to_lines(words, page_width)
    try:
        for bbox, _region_lines in alignment_table_regions(
            lines, page_width, page_height, excluded=seen
        ):
            _add(bbox)
    except Exception as exc:
        logger.debug("Alignment region detection skipped: %s", exc)

    if not regions:
        full = _union_bbox(words)
        if full is not None:
            # Slight pad so edge words are not clipped by contains().
            pad_x = max(1.0, page_width * 0.005)
            pad_y = max(1.0, page_height * 0.005)
            _add(
                BBox(
                    max(0.0, full.x0 - pad_x),
                    max(0.0, full.y0 - pad_y),
                    min(page_width, full.x1 + pad_x),
                    min(page_height, full.y1 + pad_y),
                )
            )
    return regions


def table_data_from_textract_blocks(
    blocks: Sequence[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    min_confidence: float = 50.0,
    page: Any = None,
) -> List[TableData]:
    """Convert DetectDocumentText Blocks into TableData via geometry reconstruction.

    Accepts WORD/LINE blocks only (no TABLE/CELL). ``page`` is optional and
    enables ruled-grid hints from vector drawings when present.
    """
    # Keep low-confidence words for layout; flag them later.
    words = words_from_detect_blocks(
        blocks,
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        min_confidence=0.0,
    )
    if len(words) < 4:
        return []

    tables: List[TableData] = []
    for bbox, grid_x, grid_y in _candidate_regions(
        words,
        page=page,
        page_width=page_width,
        page_height=page_height,
    ):
        region_words = [
            w for w in words if bbox.contains(w.bbox) or bbox.intersection_area(w.bbox) > 0.5 * w.bbox.area
        ]
        raw = table_from_words(
            words,
            bbox,
            page=page_number,
            page_width=page_width,
            page_height=page_height,
            grid_x=grid_x,
            grid_y=grid_y,
            method=_EXTRACTION_METHOD,
        )
        if not raw:
            continue
        finalized = _finalize_geometry_table(
            raw,
            region_words=region_words,
            min_confidence=min_confidence,
        )
        if finalized:
            tables.append(finalized)
    return tables


def _detect_document_text_bytes(png_bytes: bytes) -> Dict[str, Any]:
    client = get_textract_client()
    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            resp = client.detect_document_text(Document={"Bytes": png_bytes})
            _API_USAGE["detect_document_text_pages"] += 1
            _API_USAGE["pages_via_sync"] += 1
            return resp
        except Exception as exc:
            last_err = exc
            err = str(exc)
            is_throttle = any(
                x in err
                for x in (
                    "ThrottlingException",
                    "ProvisionedThroughputExceededException",
                    "Too Many Requests",
                    "slowdown",
                )
            )
            if not is_throttle or attempt == 3:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Textract DetectDocumentText throttled — retry in %.1fs (attempt %s/4)",
                wait,
                attempt + 1,
            )
            time.sleep(wait)
    if last_err:
        raise last_err
    return {"Blocks": []}


def _await_document_text_detection(
    client: Any,
    job_id: str,
    *,
    timeout_s: float,
    poll_s: float,
) -> List[Dict[str, Any]]:
    """Poll GetDocumentTextDetection until complete; collect all NextToken pages."""
    deadline = time.time() + max(30.0, float(timeout_s))
    poll = max(0.5, float(poll_s))
    while time.time() < deadline:
        _API_USAGE["get_document_text_detection_calls"] += 1
        resp = client.get_document_text_detection(JobId=job_id, MaxResults=1000)
        status = str(resp.get("JobStatus") or "")
        if status == "IN_PROGRESS":
            time.sleep(poll)
            continue
        if status == "FAILED":
            raise RuntimeError(
                resp.get("StatusMessage") or "StartDocumentTextDetection failed"
            )
        blocks = list(resp.get("Blocks") or [])
        token = resp.get("NextToken")
        while token:
            _API_USAGE["get_document_text_detection_calls"] += 1
            resp = client.get_document_text_detection(
                JobId=job_id, NextToken=token, MaxResults=1000
            )
            blocks.extend(resp.get("Blocks") or [])
            token = resp.get("NextToken")
        return blocks
    raise TimeoutError(
        f"Textract StartDocumentTextDetection job {job_id} timed out after {timeout_s}s"
    )


def _upload_pdf_for_async_textract(filepath: str) -> Tuple[str, str, str]:
    """Upload PDF to S3 for StartDocumentTextDetection.

    Returns ``(bucket, key, user_id_for_cleanup)``.
    """
    from app.services import s3_storage

    if not s3_storage.is_configured():
        raise RuntimeError(
            "S3 is required for multipage StartDocumentTextDetection. "
            "Set AWS_S3_BUCKET or disable PDF_TEXTRACT_ASYNC_MULTIPAGE."
        )
    path = Path(filepath)
    user_id = "textract_jobs"
    safe_name = f"{uuid.uuid4().hex}_{path.name}"
    key = s3_storage.upload_pdf(user_id, safe_name, str(path))
    bucket = (get_settings().S3_BUCKET or get_settings().AWS_S3_BUCKET or "").strip()
    return bucket, key, user_id


def extract_tables_from_pdf_async(
    filepath: str,
    *,
    page_sizes: Dict[int, Tuple[float, float]],
    doc: Any = None,
) -> Tuple[List[TableData], Dict[str, Any]]:
    """Run Start/GetDocumentTextDetection on a multipage PDF and reconstruct tables.

    Returns ``(tables, meta)``. Tables are text-derived (geometry), not native
    Textract TABLE blocks. Never calls AnalyzeDocument.
    """
    settings = get_settings()
    meta: Dict[str, Any] = {
        "api": "StartDocumentTextDetection",
        "analyze_document_tables": False,
        "pages_processed": 0,
        "job_id": "",
        "s3_key": "",
    }
    if not bool(getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False)):
        meta["skipped"] = "textract_disabled"
        return [], meta
    if not bool(getattr(settings, "PDF_TEXTRACT_TABLES_ENABLED", True)):
        meta["skipped"] = "tables_disabled"
        return [], meta

    bucket, key, cleanup_user = _upload_pdf_for_async_textract(filepath)
    meta["s3_key"] = key
    client = get_textract_client()
    try:
        start = client.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
        )
        _API_USAGE["start_document_text_detection_jobs"] += 1
        job_id = str(start.get("JobId") or "")
        meta["job_id"] = job_id
        if not job_id:
            raise RuntimeError("StartDocumentTextDetection returned empty JobId")

        blocks = _await_document_text_detection(
            client,
            job_id,
            timeout_s=float(
                getattr(settings, "PDF_TEXTRACT_ASYNC_TIMEOUT_SECONDS", 900) or 900
            ),
            poll_s=float(
                getattr(settings, "PDF_TEXTRACT_ASYNC_POLL_SECONDS", 2.0) or 2.0
            ),
        )
        by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for block in blocks:
            page_no = int(block.get("Page") or 1)
            by_page[page_no].append(block)

        min_conf = float(settings.PDF_TEXTRACT_MIN_CONFIDENCE or 50.0)
        tables: List[TableData] = []
        for page_no in sorted(by_page.keys()):
            width, height = page_sizes.get(page_no, (0.0, 0.0))
            if width <= 0 or height <= 0:
                continue
            page_obj = None
            if doc is not None:
                try:
                    page_obj = doc[page_no - 1]
                except Exception:
                    page_obj = None
            page_tables = table_data_from_textract_blocks(
                by_page[page_no],
                page_number=page_no,
                page_width=float(width),
                page_height=float(height),
                min_confidence=min_conf,
                page=page_obj,
            )
            for table in page_tables:
                prov = dict(table.provenance or {})
                prov["textract_api"] = "StartDocumentTextDetection"
                prov["text_derived"] = True
                prov["structurally_verified_table"] = False
                table.provenance = prov
            tables.extend(page_tables)
            _API_USAGE["detect_document_text_pages"] += 1
            _API_USAGE["pages_via_async"] += 1

        meta["pages_processed"] = len(by_page)
        meta["table_count"] = len(tables)
        meta["block_count"] = len(blocks)
        return tables, meta
    finally:
        try:
            from app.services import s3_storage

            s3_storage.delete_pdf(cleanup_user, Path(key).name, key=key)
        except Exception as exc:
            logger.warning("Could not delete temp Textract S3 object %s: %s", key, exc)


def extract_tables_from_pdf_detect(
    filepath: str,
    *,
    page_sizes: Dict[int, Tuple[float, float]],
    doc: Any = None,
) -> Tuple[List[TableData], Dict[str, Any]]:
    """Extract text-derived tables via DetectDocumentText for an entire PDF.

    Multipage + async enabled → Start/GetDocumentTextDetection.
    Otherwise → sync per-page PNG DetectDocumentText.
    """
    settings = get_settings()
    page_count = len(page_sizes) or 0
    async_on = bool(getattr(settings, "PDF_TEXTRACT_ASYNC_MULTIPAGE", True))
    if page_count > 1 and async_on:
        try:
            return extract_tables_from_pdf_async(
                filepath, page_sizes=page_sizes, doc=doc
            )
        except Exception as exc:
            logger.warning(
                "Async StartDocumentTextDetection failed for %s (%s); "
                "falling back to sync per-page DetectDocumentText",
                Path(filepath).name,
                exc,
            )
            # Continue to sync path below.

    meta: Dict[str, Any] = {
        "api": "DetectDocumentText",
        "analyze_document_tables": False,
        "pages_processed": 0,
        "mode": "sync_per_page",
    }
    if not bool(getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False)):
        meta["skipped"] = "textract_disabled"
        return [], meta
    if not bool(getattr(settings, "PDF_TEXTRACT_TABLES_ENABLED", True)):
        meta["skipped"] = "tables_disabled"
        return [], meta
    if doc is None:
        meta["skipped"] = "no_doc_handle"
        return [], meta

    tables: List[TableData] = []
    for page_no in sorted(page_sizes.keys()):
        try:
            page = doc[page_no - 1]
        except Exception:
            continue
        page_tables = extract_tables_from_page(page, page_no)
        tables.extend(page_tables)
        meta["pages_processed"] = int(meta["pages_processed"]) + 1
    meta["table_count"] = len(tables)
    return tables, meta


def extract_tables_from_page(page, page_number: int) -> List[TableData]:
    """Render a PDF page and extract tables via DetectDocumentText + geometry.

    Returns [] on any failure so callers can surface textract_error without
    falling back to native/OCR table extractors.
    """
    settings = get_settings()
    # Both flags must be true before any AWS call.
    if not bool(getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False)):
        return []
    if not settings.PDF_TEXTRACT_TABLES_ENABLED:
        return []

    try:
        import pymupdf as fitz

        dpi = _textract_dpi()
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        png_bytes = pix.tobytes("png")
        # Sync DetectDocumentText Bytes limit is 5 MB.
        if len(png_bytes) > 4_500_000:
            logger.warning(
                "Textract page %s PNG too large (%s bytes); skipping",
                page_number,
                len(png_bytes),
            )
            return []
        resp = _detect_document_text_bytes(png_bytes)
        blocks = list(resp.get("Blocks") or [])
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        tables = table_data_from_textract_blocks(
            blocks,
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            min_confidence=float(settings.PDF_TEXTRACT_MIN_CONFIDENCE or 50.0),
            page=page,
        )
        for table in tables:
            prov = dict(table.provenance or {})
            prov["textract_api"] = "DetectDocumentText"
            prov["text_derived"] = True
            prov["structurally_verified_table"] = False
            table.provenance = prov
        return tables
    except Exception as exc:
        logger.warning(
            "Textract DetectDocumentText failed on page %s: %s — no table fallback",
            page_number,
            exc,
        )
        return []


def usable_textract_tables(tables: Sequence[TableData]) -> List[TableData]:
    """Filter to tables with a usable grid (≥2 cols and ≥1 body row).

    Header-only reconstructions (common TOC / multi-column prose false
    positives from DetectDocumentText geometry) are dropped.
    """
    usable: List[TableData] = []
    for table in tables:
        width = max(
            len(table.headers or []),
            max((len(r) for r in table.rows), default=0),
        )
        if width >= 2 and list(table.rows or []):
            usable.append(table)
    return usable

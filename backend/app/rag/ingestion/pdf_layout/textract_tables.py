"""AWS Textract DetectDocumentText → geometry tables → TableData.

Mandatory flow
--------------
PDF pages are rendered to images first. **Never** send the original PDF file,
PDF bytes, or an S3 PDF document reference to Textract.

PDF → render each page (PNG) → ``DetectDocumentText(Document={{Bytes: png}})``
    → WORD/LINE geometry → domain-agnostic table reconstruction → TableData.

Never calls ``AnalyzeDocument`` / ``TABLES`` / ``StartDocumentTextDetection``.
Cost ~$1.50 / 1,000 DetectDocumentText pages. IAM: ``textract:DetectDocumentText``.

On client/API errors the parser records ``textract_error`` / per-page errors and
leaves that page's tables empty — there is no img2table, Tesseract, Camelot,
Tabula, or whole-PDF Textract fallback. When Textract is enabled, body text
lines are also built from DetectDocumentText WORD blocks.
"""
from __future__ import annotations

import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from app.rag.ingestion.pdf_layout.types import (
    BBox,
    PageOcrLayout,
    PageOcrWord,
    TableData,
    TextWord,
    normalize_ws,
)

logger = logging.getLogger(__name__)

_textract_client = None
_EXTRACTION_METHOD = "textract_detect"
# Sync DetectDocumentText Bytes payload limit is 5 MB; stay under with margin.
_MAX_DETECT_BYTES = 4_500_000
# Match pdf_text_v6_stable parallel Textract cap.
_MAX_TEXTRACT_WORKERS = 10
_DEFAULT_RENDER_DPI = 300

# Process-local API usage counters for cost / audit reporting.
_API_USAGE: Dict[str, int] = {
    "detect_document_text_pages": 0,
    "start_document_text_detection_jobs": 0,  # must remain 0 (PDF upload disabled)
    "get_document_text_detection_calls": 0,  # must remain 0
    "analyze_document_tables_pages": 0,  # must remain 0
    "pages_via_async": 0,  # must remain 0
    "pages_via_image": 0,
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
        return max(72, dpi)
    return max(72, int(settings.PDF_OCR_DPI or _DEFAULT_RENDER_DPI))


def _textract_max_workers() -> int:
    settings = get_settings()
    requested = int(getattr(settings, "PDF_TEXTRACT_MAX_WORKERS", 10) or 10)
    return max(1, min(requested, _MAX_TEXTRACT_WORKERS))


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


def _polygon_px_from_geometry(
    geometry: Dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> List[int]:
    """Pack Textract Polygon into v6-style ``[x0,y0,...,x3,y3]`` pixel coords."""
    polygon = (geometry or {}).get("Polygon") or []
    points: List[int] = []
    for point in polygon[:4]:
        try:
            points.extend(
                [
                    int(float(point.get("X") or 0.0) * image_width),
                    int(float(point.get("Y") or 0.0) * image_height),
                ]
            )
        except Exception:
            continue
    while len(points) < 8 and points:
        points.extend(points[-2:])
    return points[:8]


def words_from_detect_blocks(
    blocks: Sequence[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    min_confidence: float = 0.0,
    image_width: int = 0,
    image_height: int = 0,
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
        geometry = block.get("Geometry") or {}
        bbox = _geometry_to_pdf_bbox(
            geometry,
            page_width=page_width,
            page_height=page_height,
        )
        if bbox.area <= 0:
            continue
        polygon_px: List[int] = []
        if image_width > 0 and image_height > 0:
            polygon_px = _polygon_px_from_geometry(
                geometry, image_width=image_width, image_height=image_height
            )
        words.append(
            TextWord(
                text=text,
                bbox=bbox,
                page=page_number,
                confidence=conf,
                polygon_px=polygon_px,
            )
        )
    return words


def page_ocr_layout_from_blocks(
    blocks: Sequence[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
    image_format: str = "png",
    dpi: int = _DEFAULT_RENDER_DPI,
    min_confidence: float = 0.0,
) -> PageOcrLayout:
    """Build a v6-compatible page layout pack from DetectDocumentText blocks."""
    ocr_words: List[PageOcrWord] = []
    for block in blocks:
        if block.get("BlockType") != "WORD":
            continue
        text = normalize_ws(str(block.get("Text") or ""))
        if not text:
            continue
        conf = float(block.get("Confidence") or 95.0)
        if conf < min_confidence:
            continue
        geometry = block.get("Geometry") or {}
        polygon_px = _polygon_px_from_geometry(
            geometry, image_width=image_width, image_height=image_height
        )
        bbox = _geometry_to_pdf_bbox(
            geometry, page_width=page_width, page_height=page_height
        )
        ocr_words.append(
            PageOcrWord(
                text=text,
                polygon_px=polygon_px,
                confidence=conf,
                bbox_pdf=bbox if bbox.area > 0 else None,
            )
        )
    return PageOcrLayout(
        page_index=int(page_number),
        image_width=int(image_width),
        image_height=int(image_height),
        words=ocr_words,
        image_format=image_format,
        dpi=int(dpi),
    )


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
    r"(?i)^(?:"
    r"value|amount|unit|rate|freq(?:uency)?|total|count|qty|quantity|%|"
    r"\(\s*[a-zA-Zμµ%/²³.]{1,10}\s*\)|"
    r"[a-z]{1,16}\s*\(\s*[a-zA-Zμµ%/²³.]{1,10}\s*\)"
    r")$"
)
_RANGE_BAND_RE = re.compile(
    r"(?i)(?:≥|>=|>|≤|<=|<)?\s*\d+(?:\.\d+)?\s*(?:–|-|—|to)\s*\d+(?:\.\d+)?"
)
_MULTI_RANGE_SPLIT_RE = re.compile(
    r"(?i)(?=(?:≥|>=|>|≤|<=|<)\s*\d)|(?<=\d)\s+(?=(?:≥|>=|>|≤|<=|<)?\s*\d)"
)


def _is_label_banner_row(row: Sequence[str]) -> bool:
    """Detect non-data banner rows like blank|blank|Amount|($)|blank."""
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


def _is_unit_banner_row(row: Sequence[str]) -> bool:
    """True when value columns are mostly short unit/measure banners."""
    cells = [normalize_ws(c) for c in row]
    if len(cells) < 3:
        return False
    value_cells = cells[2:] if len(cells) > 2 else cells[1:]
    nonempty = [c for c in value_cells if c]
    if not nonempty:
        return False
    bannerish = sum(1 for c in nonempty if _BANNER_TOKEN_RE.match(c))
    if bannerish >= 1 and bannerish == len(nonempty):
        return True
    # Repeated identical short label across non-numeric value cells.
    labels = [
        c
        for c in nonempty
        if not (re.search(r"\d", c) and not c.startswith("("))
    ]
    if (
        len(labels) >= 2
        and len({c.lower() for c in labels}) == 1
        and len(labels[0]) <= 24
        and re.match(r"(?i)^[a-z(]", labels[0])
    ):
        return True
    if len(nonempty) < 2:
        return False
    return bannerish >= max(2, (len(nonempty) + 1) // 2)


# Back-compat alias for older call sites / tests.
_is_unit_dose_row = _is_unit_banner_row


def _split_jammed_range_header(text: str) -> List[str]:
    """Split ``≥10-12 >12-15 >15-20 …`` smashed into one header cell."""
    token = normalize_ws(text)
    if not token or not _RANGE_BAND_RE.search(token):
        return [token] if token else []
    ranges = re.findall(
        r"(?i)(?:≥|>=|>|≤|<=|<)?\s*\d+(?:\.\d+)?\s*(?:–|-|—|to)\s*\d+(?:\.\d+)?",
        token,
    )
    if len(ranges) < 2:
        return [token]
    parts = [
        normalize_ws(p) for p in _MULTI_RANGE_SPLIT_RE.split(token) if normalize_ws(p)
    ]
    cleaned: List[str] = []
    for part in parts:
        part = re.sub(r"(?i)\(\s*[a-zA-Zμµ%/²³.]{1,10}\s*\)", "", part)
        part = re.sub(r"(?i)\b(?:value|amount|unit|rate|total|count)\b", "", part)
        part = normalize_ws(part)
        if part and _RANGE_BAND_RE.search(part):
            cleaned.append(part)
    return cleaned if len(cleaned) >= 2 else [token]


_split_jammed_weight_header = _split_jammed_range_header
_WEIGHT_RANGE_RE = _RANGE_BAND_RE


def _expand_jammed_header_columns(
    headers: List[str],
    header_rows: List[List[str]],
    body: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """Spread smashed multi-range headers into neighboring empty/Column-N leaves.

    Geometry often keeps one body column per band but OCR packs several
    band labels into a single header cell (leaving ``Column N`` placeholders).
    Redistribute labels across those weak neighbors — do not insert columns.
    """
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    headers = _pad_row(headers, width)
    header_rows = [_pad_row(r, width) for r in header_rows]
    body = [_pad_row(r, width) for r in body]

    jam_col = -1
    jam_parts: List[str] = []
    for ci, h in enumerate(headers):
        parts = _split_jammed_range_header(h)
        if len(parts) >= 3:
            jam_col = ci
            jam_parts = parts
            break
    if jam_col < 0:
        return headers, header_rows, body

    def _weak(ci: int) -> bool:
        h = normalize_ws(headers[ci])
        return (
            not h
            or _is_weak_header_label(h)
            or bool(_BANNER_TOKEN_RE.match(h))
            or ci == jam_col
        )

    left = jam_col
    while left > 0 and _weak(left - 1):
        left -= 1
    right = jam_col
    while right + 1 < width and _weak(right + 1):
        right += 1
    span = list(range(left, right + 1))
    if len(span) < 2:
        return headers, header_rows, body

    assigned = list(jam_parts)
    if len(assigned) > len(span):
        assigned = assigned[: len(span)]
    while len(assigned) < len(span):
        assigned.append(assigned[-1] if assigned else "")

    for slot, ci in enumerate(span):
        band = assigned[slot]
        if not band:
            continue
        headers[ci] = band
        if header_rows:
            header_rows[-1][ci] = band
            for ri in range(len(header_rows) - 1):
                cell = normalize_ws(header_rows[ri][ci])
                if cell and len(_split_jammed_range_header(cell)) >= 3:
                    header_rows[ri][ci] = ""
    return headers, header_rows, body


def _compose_header_leaf(parts: Sequence[str]) -> str:
    """Join stacked header-path parts as-is — never invent domain labels."""
    cleaned: List[str] = []
    for part in parts:
        text = normalize_ws(part)
        if not text:
            continue
        if cleaned and cleaned[-1].lower() == text.lower():
            continue
        cleaned.append(text)
    return normalize_ws(" ".join(cleaned))


_compose_weight_dose_leaf = _compose_header_leaf


def _is_range_label_row(row: Sequence[str]) -> bool:
    """True for jammed/sparse range-band label rows still sitting in body."""
    cells = [normalize_ws(c) for c in row]
    if not cells or cells[0]:
        return False
    nonempty = [c for c in cells if c]
    if not nonempty:
        return False
    for cell in nonempty:
        if len(_split_jammed_range_header(cell)) >= 2:
            return True
    rangeish = sum(1 for c in nonempty if _RANGE_BAND_RE.search(c))
    pure_nums = sum(1 for c in nonempty if _PURE_NUM_RE.match(c))
    if rangeish and pure_nums == 0:
        return True
    if pure_nums == len(nonempty) and pure_nums <= 3:
        return True
    if rangeish == len(nonempty):
        return True
    return False


_is_weight_label_row = _is_range_label_row


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
            if val and (not parts or parts[-1].lower() != val.lower()):
                parts.append(val)
        if not parts:
            leaves.append(f"Column {c + 1}")
            continue
        # Always join stacked path parts; never invent missing domain words.
        leaves.append(_compose_header_leaf(parts) if len(parts) > 1 else parts[0])
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
    """True when a row looks like table body (numeric values / range keys)."""
    cells = [normalize_ws(c) for c in row]
    nonempty = [c for c in cells if c]
    if not nonempty:
        return False
    # Unit-banner / range-label rows are header bands, not body.
    if _is_unit_banner_row(cells) or _is_range_label_row(cells):
        return False
    pure_nums = sum(1 for c in nonempty if _PURE_NUM_RE.match(c))
    bannerish = sum(1 for c in nonempty if _BANNER_TOKEN_RE.match(c))
    if bannerish and bannerish >= pure_nums:
        return False
    if (not cells[0]) and pure_nums == len(nonempty) and pure_nums <= 3:
        return False
    if pure_nums >= 2:
        return True
    # Range/key cell + at least one numeric value.
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
        if (
            _is_label_banner_row(row)
            or _is_unit_banner_row(row)
            or _is_range_label_row(row)
        ):
            header_rows.append(_pad_row(row, width))
            for ci, cell in enumerate(row):
                token = normalize_ws(cell)
                if not token:
                    continue
                if _RANGE_BAND_RE.search(token):
                    if ci < len(headers) and not normalize_ws(headers[ci]):
                        if not _PURE_NUM_RE.match(token):
                            headers[ci] = token
                    continue
                if ci < len(headers) and not normalize_ws(headers[ci]):
                    headers[ci] = token
                elif ci < len(headers) and token.lower() not in headers[ci].lower():
                    if _BANNER_TOKEN_RE.match(token) and not re.search(
                        r"\d", headers[ci]
                    ):
                        headers[ci] = f"{headers[ci]} {token}".strip()
            continue
        cleaned_body.append(row)
    body = cleaned_body
    headers, header_rows, body = _expand_jammed_header_columns(
        headers, header_rows, body
    )
    # Second pass: unit banners can sit after the first real key row.
    kept_body: List[List[str]] = []
    for row in body:
        if _is_unit_banner_row(row) or _is_range_label_row(row):
            header_rows.append(_pad_row(row, max(len(row), len(headers), 1)))
            continue
        kept_body.append(row)
    body = kept_body
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
    headers = _pad_row(headers, width)

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

    # Fully cellular expansion: repeat shared attrs, fill unavailable cells,
    # and guarantee no merged-cell metadata survives DetectDocumentText grids.
    from app.rag.ingestion.pdf_layout.table_cellular import expand_to_fully_cellular

    cellular = expand_to_fully_cellular(
        headers,
        body,
        header_rows=header_rows,
        cells=[],
    )
    headers = list(cellular["headers"])
    body = [list(r) for r in cellular["rows"]]
    header_rows = [list(r) for r in cellular["header_rows"]]
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    quality_notes = list((recon.warnings if recon else []) or [])
    quality_notes.extend(list(cellular.get("warnings") or []))

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
            "fully_cellular": True,
            "quality_notes": quality_notes,
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


def _detect_document_text_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Call DetectDocumentText on image bytes only (never a PDF)."""
    client = get_textract_client()
    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            resp = client.detect_document_text(Document={"Bytes": image_bytes})
            _API_USAGE["detect_document_text_pages"] += 1
            _API_USAGE["pages_via_image"] += 1
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


def _render_page_image_bytes(
    page: Any, *, dpi: int
) -> Tuple[bytes, str, int, int, int]:
    """Render one PDF page to an image suitable for DetectDocumentText.

    Returns ``(bytes, format, image_width, image_height, used_dpi)``.
    Downscales until the payload is under the sync Bytes limit. Never returns
    PDF bytes. Raises if the page cannot be rendered under the limit.
    """
    import pymupdf as fitz

    current_dpi = max(72, int(dpi))
    last_size = 0
    for _ in range(8):
        scale = current_dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img_w, img_h = int(pix.width), int(pix.height)
        png_bytes = pix.tobytes("png")
        last_size = len(png_bytes)
        if last_size <= _MAX_DETECT_BYTES:
            return png_bytes, "png", img_w, img_h, current_dpi
        # Prefer JPEG when PNG is still too large at this DPI.
        try:
            jpg_bytes = pix.tobytes("jpeg", jpg_quality=85)
            if len(jpg_bytes) <= _MAX_DETECT_BYTES:
                return jpg_bytes, "jpeg", img_w, img_h, current_dpi
            last_size = len(jpg_bytes)
        except Exception:
            pass
        current_dpi = max(72, int(current_dpi * 0.75))
    raise RuntimeError(
        f"page image still too large after downscale ({last_size} bytes > "
        f"{_MAX_DETECT_BYTES}); cannot call DetectDocumentText"
    )


def _maybe_save_page_image(
    *,
    filepath: str,
    page_number: int,
    image_bytes: bytes,
    image_fmt: str,
) -> str:
    """Optionally dump page images for debug (v6 images/page-NNNN.png style)."""
    settings = get_settings()
    if not bool(getattr(settings, "PDF_TEXTRACT_SAVE_PAGE_IMAGES", False)):
        return ""
    report_dir = Path(
        getattr(settings, "PDF_LAYOUT_REPORT_DIR", "./data/layout_reports")
        or "./data/layout_reports"
    )
    stem = Path(filepath).stem or "document"
    out_dir = report_dir / "textract_pages" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if image_fmt == "jpeg" else "png"
    out_path = out_dir / f"page-{int(page_number):04d}.{ext}"
    out_path.write_bytes(image_bytes)
    return str(out_path)


def _tables_and_words_from_blocks(
    blocks: Sequence[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    page: Any = None,
    min_confidence: float,
    image_width: int = 0,
    image_height: int = 0,
    image_format: str = "png",
    dpi: int = _DEFAULT_RENDER_DPI,
) -> Tuple[List[TableData], List[TextWord], PageOcrLayout]:
    words = words_from_detect_blocks(
        blocks,
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        min_confidence=0.0,
        image_width=image_width,
        image_height=image_height,
    )
    layout = page_ocr_layout_from_blocks(
        blocks,
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        image_width=max(1, int(image_width)),
        image_height=max(1, int(image_height)),
        image_format=image_format,
        dpi=dpi,
        min_confidence=0.0,
    )
    tables = table_data_from_textract_blocks(
        blocks,
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        min_confidence=min_confidence,
        page=page,
    )
    for table in tables:
        prov = dict(table.provenance or {})
        prov["textract_api"] = "DetectDocumentText"
        prov["input"] = "page_image"
        prov["text_derived"] = True
        prov["structurally_verified_table"] = False
        prov["page"] = int(page_number)
        table.provenance = prov
        if not table.pages:
            table.pages = [int(page_number)]
    return tables, words, layout


def detect_page_tables_and_words(
    page, page_number: int
) -> Tuple[List[TableData], List[TextWord]]:
    """Render page → DetectDocumentText on image bytes → (tables, words).

    Raises on hard failures so callers can record per-page errors. Never falls
    back to another OCR engine or whole-PDF Textract.
    """
    settings = get_settings()
    if not bool(getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False)):
        return [], []
    if not settings.PDF_TEXTRACT_TABLES_ENABLED:
        return [], []

    dpi = _textract_dpi()
    image_bytes, image_fmt, img_w, img_h, used_dpi = _render_page_image_bytes(
        page, dpi=dpi
    )
    resp = _detect_document_text_bytes(image_bytes)
    blocks = list(resp.get("Blocks") or [])
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    tables, words, _layout = _tables_and_words_from_blocks(
        blocks,
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        page=page,
        min_confidence=float(settings.PDF_TEXTRACT_MIN_CONFIDENCE or 50.0),
        image_width=img_w,
        image_height=img_h,
        image_format=image_fmt,
        dpi=used_dpi,
    )
    for table in tables:
        prov = dict(table.provenance or {})
        prov["image_format"] = image_fmt
        prov["image_bytes"] = len(image_bytes)
        prov["image_width"] = img_w
        prov["image_height"] = img_h
        table.provenance = prov
    return tables, words


def extract_tables_from_pdf_detect(
    filepath: str,
    *,
    page_sizes: Dict[int, Tuple[float, float]],
    doc: Any = None,
) -> Tuple[List[TableData], Dict[str, Any]]:
    """Render every PDF page to an image and run DetectDocumentText per page.

    Never uploads or sends the PDF to Textract. Page results are combined in
    ascending page order. Controlled concurrency applies only to the Textract
    API calls after images are rendered.
    """
    settings = get_settings()
    meta: Dict[str, Any] = {
        "api": "DetectDocumentText",
        "mode": "page_images",
        "analyze_document_tables": False,
        "pdf_uploaded_to_textract": False,
        "pages_processed": 0,
        "page_errors": {},
        "workers": 1,
        "dpi": _textract_dpi(),
        "saved_page_images": [],
        "page_layouts": {},
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

    page_numbers = sorted(int(p) for p in page_sizes.keys())
    dpi = _textract_dpi()
    meta["dpi"] = dpi
    # Render sequentially (PyMuPDF page objects are not safely shared across
    # threads), then optionally fan out DetectDocumentText.
    rendered: List[Dict[str, Any]] = []
    for page_no in page_numbers:
        try:
            page = doc[page_no - 1]
        except Exception as exc:
            meta["page_errors"][str(page_no)] = f"open_page: {type(exc).__name__}: {exc}"
            logger.error(
                "Cannot open PDF page %s for Textract image render: %s",
                page_no,
                exc,
            )
            continue
        try:
            image_bytes, image_fmt, img_w, img_h, used_dpi = _render_page_image_bytes(
                page, dpi=dpi
            )
            saved = _maybe_save_page_image(
                filepath=filepath,
                page_number=page_no,
                image_bytes=image_bytes,
                image_fmt=image_fmt,
            )
            if saved:
                meta["saved_page_images"].append(saved)
            rendered.append(
                {
                    "page_no": page_no,
                    "page": page,
                    "image_bytes": image_bytes,
                    "image_fmt": image_fmt,
                    "image_width": img_w,
                    "image_height": img_h,
                    "used_dpi": used_dpi,
                    "page_width": float(page.rect.width),
                    "page_height": float(page.rect.height),
                }
            )
        except Exception as exc:
            meta["page_errors"][str(page_no)] = (
                f"render: {type(exc).__name__}: {exc}"
            )
            logger.error(
                "PDF page %s image render failed (no OCR substitute): %s",
                page_no,
                exc,
            )

    min_conf = float(settings.PDF_TEXTRACT_MIN_CONFIDENCE or 50.0)
    workers = _textract_max_workers()
    meta["workers"] = workers

    def _ocr_one(item: Dict[str, Any]) -> Dict[str, Any]:
        page_no = int(item["page_no"])
        try:
            resp = _detect_document_text_bytes(item["image_bytes"])
            blocks = list(resp.get("Blocks") or [])
            tables, words, layout = _tables_and_words_from_blocks(
                blocks,
                page_number=page_no,
                page_width=float(item["page_width"]),
                page_height=float(item["page_height"]),
                page=item.get("page"),
                min_confidence=min_conf,
                image_width=int(item["image_width"]),
                image_height=int(item["image_height"]),
                image_format=str(item["image_fmt"]),
                dpi=int(item["used_dpi"]),
            )
            for table in tables:
                prov = dict(table.provenance or {})
                prov["image_format"] = item["image_fmt"]
                prov["image_bytes"] = len(item["image_bytes"])
                prov["image_width"] = int(item["image_width"])
                prov["image_height"] = int(item["image_height"])
                table.provenance = prov
            return {
                "page_no": page_no,
                "tables": tables,
                "words": words,
                "layout": layout,
                "error": "",
            }
        except Exception as exc:
            return {
                "page_no": page_no,
                "tables": [],
                "words": [],
                "layout": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    results_by_page: Dict[int, Dict[str, Any]] = {}
    if workers <= 1 or len(rendered) <= 1:
        for item in rendered:
            results_by_page[int(item["page_no"])] = _ocr_one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_ocr_one, item): int(item["page_no"]) for item in rendered}
            for fut in as_completed(futures):
                page_no = futures[fut]
                try:
                    results_by_page[page_no] = fut.result()
                except Exception as exc:
                    results_by_page[page_no] = {
                        "page_no": page_no,
                        "tables": [],
                        "words": [],
                        "layout": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

    tables: List[TableData] = []
    words_by_page: Dict[int, List[TextWord]] = {}
    page_layouts: Dict[int, Dict[str, Any]] = {}
    for page_no in page_numbers:
        result = results_by_page.get(page_no)
        if not result:
            continue
        if result.get("error"):
            meta["page_errors"][str(page_no)] = str(result["error"])
            logger.error(
                "Textract DetectDocumentText failed on page %s image: %s",
                page_no,
                result["error"],
            )
            meta["pages_processed"] = int(meta["pages_processed"]) + 1
            continue
        page_tables = list(result.get("tables") or [])
        page_words = list(result.get("words") or [])
        tables.extend(page_tables)
        if page_words:
            words_by_page[page_no] = page_words
        layout = result.get("layout")
        if layout is not None:
            page_layouts[page_no] = layout.to_dict()
        meta["pages_processed"] = int(meta["pages_processed"]) + 1

    # Sequential LLM normalize after parallel OCR (avoid Bedrock from workers).
    try:
        from app.rag.ingestion.pdf_layout.table_llm_normalize import (
            normalize_tables_with_llm,
        )

        tables, llm_meta = normalize_tables_with_llm(tables)
        meta["llm_normalize"] = llm_meta
    except Exception as exc:
        meta["llm_normalize_error"] = f"{type(exc).__name__}: {exc}"
        logger.debug("LLM table normalize batch skipped: %s", exc)

    meta["table_count"] = len(tables)
    meta["words_by_page"] = words_by_page
    meta["page_layouts"] = page_layouts
    meta["failed_pages"] = sorted(int(p) for p in meta["page_errors"].keys())
    if meta["page_errors"] and not tables and not words_by_page:
        # Surface a hard failure when every page failed — no silent empty success.
        first = next(iter(meta["page_errors"].values()))
        raise RuntimeError(
            f"DetectDocumentText failed for all rendered pages "
            f"({len(meta['page_errors'])} errors); first: {first}"
        )
    return tables, meta


def extract_tables_from_page(page, page_number: int) -> List[TableData]:
    """Render a PDF page image and extract tables via DetectDocumentText.

    Returns [] on failure so callers can surface ``textract_error`` without
    falling back to native/OCR table extractors or whole-PDF Textract.
    """
    try:
        tables, _words = detect_page_tables_and_words(page, page_number)
        return tables
    except Exception as exc:
        logger.error(
            "Textract DetectDocumentText failed on page %s image: %s — no fallback",
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

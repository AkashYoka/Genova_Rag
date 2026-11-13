"""AWS Textract TABLES → TableData for Genova RAG.

Sole production structured-table extractor. The parser gates calls via
``_page_looks_tabular`` (when ``PDF_TEXTRACT_TABULAR_ONLY``) so ordinary text
pages never hit AWS.

Uses AnalyzeDocument(FeatureTypes=["TABLES"]) on a per-page PNG.
IAM: textract:AnalyzeDocument. Cost ~$0.015/page analyzed.

On client/API errors the parser records ``textract_error`` and leaves tables
empty — there is no img2table or Tesseract table fallback. Body text is never
taken from Textract.
"""
from __future__ import annotations

import logging
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from botocore.config import Config

from app.config import get_settings
from app.rag.ingestion.pdf_layout.header_reconstruction import (
    reconstruct_table_headers_cached,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData, normalize_ws

logger = logging.getLogger(__name__)

_textract_client = None


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


def _cell_text_from_block(
    cell: Dict[str, Any],
    blocks_by_id: Dict[str, Dict[str, Any]],
) -> str:
    texts: List[str] = []
    for rel in cell.get("Relationships") or []:
        if rel.get("Type") != "CHILD":
            continue
        for child_id in rel.get("Ids") or []:
            child = blocks_by_id.get(child_id) or {}
            btype = child.get("BlockType")
            if btype == "WORD":
                texts.append(str(child.get("Text") or ""))
            elif btype == "SELECTION_ELEMENT" and child.get("SelectionStatus") == "SELECTED":
                texts.append("X")
    return normalize_ws(" ".join(t for t in texts if t))


def _merged_span_map(
    table_block: Dict[str, Any],
    blocks_by_id: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[int, int], Tuple[int, int, str]]:
    """Map (row, col) → (row_span, col_span, text) for merged cells."""
    spans: Dict[Tuple[int, int], Tuple[int, int, str]] = {}
    for rel in table_block.get("Relationships") or []:
        if rel.get("Type") != "MERGED_CELL":
            continue
        for mid in rel.get("Ids") or []:
            merged = blocks_by_id.get(mid) or {}
            if merged.get("BlockType") != "MERGED_CELL":
                continue
            row = int(merged.get("RowIndex") or 0)
            col = int(merged.get("ColumnIndex") or 0)
            row_span = max(1, int(merged.get("RowSpan") or 1))
            col_span = max(1, int(merged.get("ColumnSpan") or 1))
            text = _cell_text_from_block(merged, blocks_by_id)
            if row < 1 or col < 1:
                continue
            for r in range(row, row + row_span):
                for c in range(col, col + col_span):
                    spans[(r, c)] = (row_span if (r, c) == (row, col) else 0, col_span if (r, c) == (row, col) else 0, text)
            spans[(row, col)] = (row_span, col_span, text)
    return spans


def _is_header_cell(cell: Dict[str, Any]) -> bool:
    entity = cell.get("EntityTypes") or []
    return "COLUMN_HEADER" in entity or "TABLE_TITLE" in entity


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
        # Prefer the most specific trailing label when earlier parts are generic
        # category words without digits (avoids "Body > >60-70 kg" noise).
        child = parts[-1]
        if len(parts) >= 2 and re.search(r"\d", child) and not re.search(r"\d", parts[0]):
            leaves.append(child)
        else:
            leaves.append(" > ".join(parts))
    return leaves


_BANNER_TOKEN_RE = re.compile(
    r"(?i)^(?:dose|value|amount|unit|rate|freq(?:uency)?|"
    r"total|count|%|\(\s*[a-zμµ/%]{1,6}\s*\))$"
)


def _is_label_banner_row(row: Sequence[str]) -> bool:
    """Detect non-data banner rows like blank|blank|Dose|(mg)|blank.

    Domain-agnostic: known label/unit tokens only, mostly empty cells, no
    numeric ranges. Single-letter body values are NOT treated as banners.
    """
    cells = [normalize_ws(c) for c in row]
    nonempty = [c for c in cells if c]
    if len(nonempty) < 2 or len(nonempty) > max(3, len(cells) // 3):
        return False
    # Require leading body key empty so real first-column labels are kept.
    if cells and cells[0]:
        return False
    for cell in nonempty:
        if re.search(r"\d", cell) and not cell.startswith("("):
            return False
        if not _BANNER_TOKEN_RE.match(cell):
            return False
    return True


def _infer_header_row_count(grid: List[List[Dict[str, Any]]]) -> int:
    if not grid:
        return 0
    headerish = 0
    for row in grid:
        if not row:
            break
        header_flags = sum(1 for cell in row if cell.get("is_header"))
        if header_flags >= max(1, len(row) // 2):
            headerish += 1
        else:
            break
    if headerish > 0:
        # Peek one more row: numeric/unit leaf bands often lack COLUMN_HEADER flags.
        if headerish < len(grid):
            nxt = grid[headerish]
            texts = [normalize_ws(c.get("text") or "") for c in nxt]
            nonempty = [t for t in texts if t]
            if (
                nonempty
                and not texts[0]
                and sum(1 for t in nonempty if re.search(r"\d", t)) >= max(2, len(nonempty) - 1)
                and sum(1 for t in nonempty if re.fullmatch(r"-?\d+(?:\.\d+)?", t))
                < max(2, len(nonempty) // 2)
            ):
                headerish += 1
        return headerish
    # Fallback: first row is headers when the table has ≥2 rows.
    return 1 if len(grid) >= 2 else 0


def table_data_from_textract_blocks(
    blocks: Sequence[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    min_confidence: float = 50.0,
) -> List[TableData]:
    """Convert Textract AnalyzeDocument Blocks into TableData objects."""
    blocks_by_id = {b.get("Id"): b for b in blocks if b.get("Id")}
    tables: List[TableData] = []

    for block in blocks:
        if block.get("BlockType") != "TABLE":
            continue
        cell_blocks: List[Dict[str, Any]] = []
        for rel in block.get("Relationships") or []:
            if rel.get("Type") != "CHILD":
                continue
            for cid in rel.get("Ids") or []:
                child = blocks_by_id.get(cid) or {}
                if child.get("BlockType") == "CELL":
                    cell_blocks.append(child)
        if not cell_blocks:
            continue

        max_row = max(int(c.get("RowIndex") or 0) for c in cell_blocks)
        max_col = max(int(c.get("ColumnIndex") or 0) for c in cell_blocks)
        if max_row < 1 or max_col < 1:
            continue

        merged = _merged_span_map(block, blocks_by_id)
        grid: List[List[Dict[str, Any]]] = [
            [
                {
                    "row_index": r,
                    "col_index": c,
                    "text": "",
                    "confidence": 0.0,
                    "bbox": None,
                    "row_span": 1,
                    "col_span": 1,
                    "is_header": False,
                }
                for c in range(1, max_col + 1)
            ]
            for r in range(1, max_row + 1)
        ]

        confidences: List[float] = []
        low_confidence_cells: List[Dict[str, Any]] = []
        cells_out: List[Dict[str, Any]] = []
        merged_out: List[Dict[str, Any]] = []

        for cell in cell_blocks:
            row = int(cell.get("RowIndex") or 0)
            col = int(cell.get("ColumnIndex") or 0)
            if row < 1 or col < 1 or row > max_row or col > max_col:
                continue
            row_span = max(1, int(cell.get("RowSpan") or 1))
            col_span = max(1, int(cell.get("ColumnSpan") or 1))
            text = _cell_text_from_block(cell, blocks_by_id)
            conf = float(cell.get("Confidence") or 0.0)
            confidences.append(conf)
            bbox = _geometry_to_pdf_bbox(
                cell.get("Geometry") or {},
                page_width=page_width,
                page_height=page_height,
            )
            is_header = _is_header_cell(cell)
            span_info = merged.get((row, col))
            if span_info:
                m_row_span, m_col_span, m_text = span_info
                if m_text:
                    text = m_text
                # Primary merged origin has positive spans; clones are (0, 0, text).
                if m_row_span > 0 and m_col_span > 0:
                    row_span = m_row_span
                    col_span = m_col_span
                    if row_span > 1 or col_span > 1:
                        merged_out.append(
                            {
                                "row_index": row - 1,
                                "col_index": col - 1,
                                "row_span": row_span,
                                "col_span": col_span,
                                "text": text,
                            }
                        )

            entry = {
                "row_index": row - 1,
                "col_index": col - 1,
                "text": text,
                "confidence": conf,
                "bbox": list(bbox.as_tuple()),
                "row_span": row_span,
                "col_span": col_span,
                "is_header": is_header,
            }
            cells_out.append(entry)
            if conf < min_confidence and text:
                low_confidence_cells.append(
                    {"row": row - 1, "col": col - 1, "text": text, "confidence": conf}
                )

            for r in range(row, min(max_row, row + row_span - 1) + 1):
                for c in range(col, min(max_col, col + col_span - 1) + 1):
                    slot = grid[r - 1][c - 1]
                    # Primary cell owns spans; filled clones keep text.
                    if (r, c) == (row, col) or not slot["text"]:
                        slot["text"] = text
                        slot["confidence"] = conf
                        slot["bbox"] = list(bbox.as_tuple())
                        slot["is_header"] = is_header or slot["is_header"]
                    if (r, c) == (row, col):
                        slot["row_span"] = row_span
                        slot["col_span"] = col_span

        header_count = _infer_header_row_count(grid)
        header_rows = [
            [normalize_ws(cell["text"]) for cell in grid[r]]
            for r in range(header_count)
        ]
        body = [
            [normalize_ws(cell["text"]) for cell in grid[r]]
            for r in range(header_count, max_row)
        ]
        headers = _leaf_headers(header_rows)
        if not headers and body:
            # Single-row table: treat as header-less body.
            headers = [f"Column {i + 1}" for i in range(max_col)]
        if not body and header_rows:
            # Only header-like rows: keep last as body if ≥2 header rows.
            if len(header_rows) >= 2:
                body = [header_rows[-1]]
                header_rows = header_rows[:-1]
                headers = _leaf_headers(header_rows)
            else:
                body = [list(headers)]
                headers = [f"Column {i + 1}" for i in range(max_col)]
                header_rows = []

        # Drop non-data banner rows (e.g. Dose | (mg) under column groups).
        cleaned_body: List[List[str]] = []
        for row in body:
            if _is_label_banner_row(row):
                # Fold banner tokens into headers when a column header is empty.
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

        if max_col < 2 or (not body and not headers):
            continue

        # Geometry-aware header path reconstruction + fragmented cell repair.
        recon = reconstruct_table_headers_cached(
            header_rows=header_rows,
            rows=body,
            cells=cells_out,
            merged_cells=merged_out,
            existing_headers=headers,
        )
        if recon.leaf_headers and len(recon.leaf_headers) == max_col:
            headers = list(recon.leaf_headers)
        if recon.reconstructed_header_rows:
            header_rows = [list(r) for r in recon.reconstructed_header_rows]
        if recon.reconstructed_rows:
            body = [list(r) for r in recon.reconstructed_rows]
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

        table_bbox = _geometry_to_pdf_bbox(
            block.get("Geometry") or {},
            page_width=page_width,
            page_height=page_height,
        )
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        # Flag sparse data rows (many empties) for downstream verification.
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
        tables.append(
            TableData(
                page=page_number,
                bbox=table_bbox,
                headers=headers,
                rows=body,
                extraction_method="textract_tables",
                source_lines=max_row,
                low_confidence_cells=low_confidence_cells,
                cells=cells_out,
                header_rows=header_rows,
                merged_cells=merged_out,
                textract_confidence=round(mean_conf, 2),
                header_reconstruction=recon.to_dict(),
                normalized_header_paths=[list(p) for p in recon.normalized_header_paths],
            )
        )
    return tables


def _analyze_document_bytes(png_bytes: bytes) -> Dict[str, Any]:
    client = get_textract_client()
    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            return client.analyze_document(
                Document={"Bytes": png_bytes},
                FeatureTypes=["TABLES"],
            )
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
                "Textract TABLES throttled — retry in %.1fs (attempt %s/4)",
                wait,
                attempt + 1,
            )
            time.sleep(wait)
    if last_err:
        raise last_err
    return {"Blocks": []}


def extract_tables_from_page(page, page_number: int) -> List[TableData]:
    """Render a PDF page and extract tables via Textract TABLES.

    Returns [] on any failure so callers can fall back to native/OCR extractors.
    """
    settings = get_settings()
    if not settings.PDF_TEXTRACT_TABLES_ENABLED:
        return []

    try:
        import pymupdf as fitz

        dpi = _textract_dpi()
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        png_bytes = pix.tobytes("png")
        # Sync AnalyzeDocument Bytes limit is 5 MB.
        if len(png_bytes) > 4_500_000:
            logger.warning(
                "Textract page %s PNG too large (%s bytes); skipping",
                page_number,
                len(png_bytes),
            )
            return []
        resp = _analyze_document_bytes(png_bytes)
        blocks = list(resp.get("Blocks") or [])
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        return table_data_from_textract_blocks(
            blocks,
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            min_confidence=float(settings.PDF_TEXTRACT_MIN_CONFIDENCE or 50.0),
        )
    except Exception as exc:
        logger.warning(
            "Textract TABLES failed on page %s: %s — falling back to native/OCR",
            page_number,
            exc,
        )
        return []


def usable_textract_tables(tables: Sequence[TableData]) -> List[TableData]:
    """Filter to tables with a usable grid (≥2 cols and ≥1 body row)."""
    usable: List[TableData] = []
    for table in tables:
        width = max(len(table.headers or []), max((len(r) for r in table.rows), default=0))
        if width >= 2 and (table.rows or table.headers):
            usable.append(table)
    return usable

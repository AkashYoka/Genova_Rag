"""Textract AnalyzeDocument TABLES on page images → cellular grids.

Unlike DetectDocumentText (words only), this returns TABLE/CELL blocks with
row/column indices and optional RowSpan/ColSpan. Still uses rendered page
images — never PDF bytes.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ste.models import BBox, normalize_ws
from ste.ocr_textract import get_client

logger = logging.getLogger(__name__)


def analyze_document_tables(
    image_bytes: bytes,
    *,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    client = get_client(region=region)
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            return client.analyze_document(
                Document={"Bytes": image_bytes},
                FeatureTypes=["TABLES"],
            )
        except Exception as exc:
            last_exc = exc
            wait = (2**attempt) + random.random()
            logger.warning(
                "AnalyzeDocument TABLES attempt %s failed: %s; sleep %.1fs",
                attempt + 1,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"AnalyzeDocument TABLES failed: {last_exc}") from last_exc


def _geometry_bbox(
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


def _index_blocks(blocks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(b.get("Id")): b for b in blocks if b.get("Id")}


def _cell_text(cell: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> str:
    parts: List[str] = []
    for rel in cell.get("Relationships") or []:
        if rel.get("Type") != "CHILD":
            continue
        for cid in rel.get("Ids") or []:
            child = by_id.get(str(cid)) or {}
            if child.get("BlockType") == "WORD":
                t = normalize_ws(str(child.get("Text") or ""))
                if t:
                    parts.append(t)
            elif child.get("BlockType") == "SELECTION_ELEMENT":
                if str(child.get("SelectionStatus") or "") == "SELECTED":
                    parts.append("[X]")
    return normalize_ws(" ".join(parts))


def tables_from_analyze_blocks(
    blocks: Sequence[Dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
) -> List[Dict[str, Any]]:
    """Return list of {bbox, grid, header_row_count, confidence, cells_meta}."""
    by_id = _index_blocks(blocks)
    tables: List[Dict[str, Any]] = []

    for block in blocks:
        if block.get("BlockType") != "TABLE":
            continue
        cell_ids: List[str] = []
        for rel in block.get("Relationships") or []:
            if rel.get("Type") == "CHILD":
                cell_ids.extend(str(i) for i in (rel.get("Ids") or []))

        # Collect cells with span expansion into a dense grid.
        sparse: Dict[Tuple[int, int], str] = {}
        cells_meta: List[Dict[str, Any]] = []
        max_r, max_c = 0, 0
        confs: List[float] = []
        for cid in cell_ids:
            cell = by_id.get(cid) or {}
            if cell.get("BlockType") != "CELL":
                continue
            row = int(cell.get("RowIndex") or 1) - 1
            col = int(cell.get("ColumnIndex") or 1) - 1
            row_span = max(1, int(cell.get("RowSpan") or 1))
            col_span = max(1, int(cell.get("ColumnSpan") or 1))
            text = _cell_text(cell, by_id)
            confs.append(float(cell.get("Confidence") or 0.0))
            geom = (cell.get("Geometry") or {}).get("BoundingBox") or {}
            bbox_norm = [
                float(geom.get("Left") or 0.0),
                float(geom.get("Top") or 0.0),
                float(geom.get("Width") or 0.0),
                float(geom.get("Height") or 0.0),
            ]
            page_bbox = _geometry_bbox(
                cell.get("Geometry") or {},
                page_width=page_width,
                page_height=page_height,
            )
            for dr in range(row_span):
                for dc in range(col_span):
                    rr, cc = row + dr, col + dc
                    # Only write text into the origin cell; mark spanned clones
                    # with the same text so every logical cell is filled
                    # (fully cellular — no implicit merges left).
                    if (rr, cc) not in sparse or not sparse[(rr, cc)]:
                        sparse[(rr, cc)] = text
                    max_r = max(max_r, rr)
                    max_c = max(max_c, cc)
                    cells_meta.append(
                        {
                            "row": rr,
                            "col": cc,
                            "origin_row": row,
                            "origin_col": col,
                            "row_span": row_span,
                            "col_span": col_span,
                            "text": text if (dr, dc) == (0, 0) else "",
                            "bbox_norm": bbox_norm,
                            "bbox": page_bbox.as_list(),
                            "confidence": float(cell.get("Confidence") or 0.0),
                        }
                    )

        if max_r < 0 or max_c < 1:
            continue
        n_rows, n_cols = max_r + 1, max_c + 1
        if n_rows < 2 or n_cols < 2:
            continue

        grid: List[List[str]] = []
        for r in range(n_rows):
            grid.append(
                [normalize_ws(sparse.get((r, c), "")) for c in range(n_cols)]
            )

        # Textract EntityTypes on first row often mark COLUMN_HEADER.
        header_rows = 1
        for c in range(n_cols):
            # Inspect first-row cells for header entity
            break
        first_row_headerish = 0
        for cid in cell_ids:
            cell = by_id.get(cid) or {}
            if int(cell.get("RowIndex") or 0) != 1:
                continue
            entities = cell.get("EntityTypes") or []
            if "COLUMN_HEADER" in entities:
                first_row_headerish += 1
        if first_row_headerish >= max(1, n_cols // 2):
            header_rows = 1
            # Check for a second header row
            second = 0
            for cid in cell_ids:
                cell = by_id.get(cid) or {}
                if int(cell.get("RowIndex") or 0) != 2:
                    continue
                entities = cell.get("EntityTypes") or []
                if "COLUMN_HEADER" in entities:
                    second += 1
            if second >= max(1, n_cols // 2):
                header_rows = 2

        bbox = _geometry_bbox(
            block.get("Geometry") or {},
            page_width=page_width,
            page_height=page_height,
        )
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        tables.append(
            {
                "page": page_number,
                "bbox": bbox,
                "grid": grid,
                "header_row_count": header_rows,
                "confidence": mean_conf,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "cells_meta": cells_meta,
            }
        )
    return tables

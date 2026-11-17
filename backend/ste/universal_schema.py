"""Emit the universal reconstruction JSON schema (domain-agnostic)."""
from __future__ import annotations

from typing import Any, Dict, List

from ste.models import Cell, ExtractedTable, normalize_ws
from ste.row_context import header_key, split_unit

_NULL_STATUSES = {
    "empty",
    "insufficient_data",
    "not_applicable",
    "not_reported",
    "illegible",
    "redacted",
    "uncertain",
}


def _cell_payload(
    *,
    header: str,
    cell: Cell | None,
    text: str,
    status: str,
    original: str,
    method: str,
) -> Dict[str, Any]:
    unit = (cell.unit if cell else "") or ""
    if not unit:
        _, unit = split_unit(header, text)
    value: Any = None if status in _NULL_STATUSES else (text or None)
    # Preserve explicit unavailable wording in original; value stays null.
    out = {
        "header": header,
        "original_value": original if original else None,
        "normalized_value": value,
        "unit": unit or None,
        "status": status,
        "source_coordinates": cell.bbox.as_list() if cell and cell.bbox else None,
        "confidence": round(float(cell.confidence), 2) if cell else 0.0,
        "reconstruction_method": method,
    }
    if cell and cell.explanation:
        out["explanation"] = cell.explanation
    if cell and cell.region_id:
        out["region_id"] = cell.region_id
        out["reconstruction_method"] = "connected_unavailable_region"
    return out


def to_universal_dict(
    table: ExtractedTable,
    *,
    document_id: str = "",
) -> Dict[str, Any]:
    """Schema aligned with the Universal PDF Table Reconstruction contract."""
    prov = dict(table.provenance or {})
    doc_id = document_id or str(prov.get("document_id") or "")
    original_headers = list(
        prov.get("original_headers") or table.headers or []
    )
    reconstructed_headers = list(
        prov.get("reconstructed_headers") or table.headers or []
    )
    original_grid = prov.get("original_grid")
    if not original_grid:
        original_grid = list(table.header_rows or []) + [list(r) for r in table.rows]

    headers = list(reconstructed_headers or table.headers or [])
    rows = [list(r) for r in (table.rows or [])]
    width = max(len(headers), max((len(r) for r in rows), default=0), 1)
    while len(headers) < width:
        headers.append(f"Column {len(headers) + 1}")
    while len(original_headers) < width:
        original_headers.append(headers[len(original_headers)])

    keys = [header_key(h, i) for i, h in enumerate(headers)]
    seen: Dict[str, int] = {}
    unique_keys: List[str] = []
    for k in keys:
        n = seen.get(k, 0)
        seen[k] = n + 1
        unique_keys.append(k if n == 0 else f"{k}_{n + 1}")

    cell_by_pos = {(c.row, c.col): c for c in (table.cells or [])}
    independent_rows: List[Dict[str, Any]] = []
    for ri, row in enumerate(rows):
        row = list(row) + [""] * (width - len(row))
        row = row[:width]
        cells_out: Dict[str, Any] = {}
        for ci, key in enumerate(unique_keys):
            cell = cell_by_pos.get((ri, ci))
            original = cell.original if cell else normalize_ws(row[ci])
            status = cell.status if cell else ("empty" if not original else "present")
            text = cell.text if cell else original
            method = "source_cell"
            if status == "insufficient_data" and original.lower() == "insufficient data":
                method = "unavailable_marker"
            cells_out[key] = _cell_payload(
                header=headers[ci],
                cell=cell,
                text=text,
                status=status,
                original=original,
                method=method,
            )
        independent_rows.append(
            {
                "row_id": f"{table.table_id}_row_{ri + 1}",
                "row_index": ri,
                "cells": cells_out,
                "source": {
                    "page": table.page,
                    "pages": list(table.pages or [table.page]),
                    "bbox": table.bbox.as_list(),
                    "caption": table.caption or "",
                },
            }
        )

    return {
        "document_id": doc_id,
        "page_number": table.page,
        "table_id": table.table_id,
        "table_title": table.caption or "",
        "original_headers": original_headers,
        "reconstructed_headers": headers,
        "original_grid": original_grid,
        "independent_rows": independent_rows,
        "unavailable_regions": list(prov.get("unavailable_regions") or []),
        "footnotes": list(table.footnotes or []),
        "extraction_confidence": round(float(table.confidence), 4),
        "reconstruction_warnings": list(table.warnings or []),
        "source_provenance": {
            "extraction_method": table.extraction_method,
            "bbox": table.bbox.as_list(),
            "pages": list(table.pages or [table.page]),
            "fully_cellular": True,
            **{
                k: v
                for k, v in prov.items()
                if k
                not in {
                    "original_headers",
                    "reconstructed_headers",
                    "original_grid",
                    "document_id",
                }
            },
        },
    }

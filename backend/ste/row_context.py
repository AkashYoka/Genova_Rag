"""Build self-contained row objects from dynamic headers (no domain keys)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from ste.models import Cell, ExtractedTable, normalize_ws


def header_key(header: str, index: int) -> str:
    token = normalize_ws(header).lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    if not token:
        token = f"column_{index + 1}"
    if token[0].isdigit():
        token = f"col_{token}"
    return token[:80]


def split_unit(header: str, value: str) -> tuple[str, str]:
    """Best-effort unit from header parentheses; leave value unchanged."""
    m = re.search(r"\(([^)]{1,16})\)", header or "")
    unit = normalize_ws(m.group(1)) if m else ""
    return value, unit


def build_self_contained_rows(table: ExtractedTable) -> List[Dict[str, Any]]:
    headers = list(table.headers or [])
    rows = [list(r) for r in (table.rows or [])]
    width = max(len(headers), max((len(r) for r in rows), default=0), 1)
    while len(headers) < width:
        headers.append(f"Column {len(headers) + 1}")
    keys = [header_key(h, i) for i, h in enumerate(headers)]
    # Ensure unique keys
    seen: Dict[str, int] = {}
    unique_keys: List[str] = []
    for k in keys:
        n = seen.get(k, 0)
        seen[k] = n + 1
        unique_keys.append(k if n == 0 else f"{k}_{n + 1}")

    cell_by_pos: Dict[tuple[int, int], Cell] = {
        (c.row, c.col): c for c in (table.cells or [])
    }
    out: List[Dict[str, Any]] = []
    for ri, row in enumerate(rows):
        row = list(row) + [""] * (width - len(row))
        row = row[:width]
        context: Dict[str, Any] = {}
        for ci, key in enumerate(unique_keys):
            cell = cell_by_pos.get((ri, ci))
            original = cell.original if cell else normalize_ws(row[ci])
            status = cell.status if cell else ("empty" if not original else "present")
            text = cell.text if cell else original
            unit = cell.unit if cell else ""
            if not unit:
                _, unit = split_unit(headers[ci], text)
            null_statuses = {
                "empty",
                "insufficient_data",
                "not_applicable",
                "not_reported",
                "illegible",
                "redacted",
                "uncertain",
            }
            entry: Dict[str, Any] = {
                "value": None if status in null_statuses else (text or None),
                "status": status,
                "original": original,
            }
            if unit:
                entry["unit"] = unit
            entry["header"] = headers[ci]
            if cell and cell.explanation:
                entry["explanation"] = cell.explanation
            if cell and cell.region_id:
                entry["region_id"] = cell.region_id
            context[key] = entry
        out.append(
            {
                "table_id": table.table_id,
                "row_id": f"{table.table_id}_row_{ri + 1}",
                "row_index": ri,
                "row_context": context,
                "source": {
                    "page": table.page,
                    "pages": list(table.pages or [table.page]),
                    "bbox": table.bbox.as_list(),
                    "caption": table.caption or "",
                },
            }
        )
    return out


def attach_self_contained_rows(table: ExtractedTable) -> ExtractedTable:
    table.self_contained_rows = build_self_contained_rows(table)
    return table

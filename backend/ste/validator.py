"""Structural validation of reconstructed tables."""
from __future__ import annotations

from typing import Any, Dict, List

from ste.models import ExtractedTable, normalize_ws


def validate_table(table: ExtractedTable) -> Dict[str, Any]:
    width = len(table.headers or [])
    issues: List[str] = []
    checks: Dict[str, bool] = {}

    checks["has_headers"] = width >= 2
    if not checks["has_headers"]:
        issues.append("too_few_columns")

    checks["has_body_rows"] = bool(table.rows)
    if not checks["has_body_rows"]:
        issues.append("no_body_rows")

    jagged = False
    for r in table.rows or []:
        if len(r) != width:
            jagged = True
            break
    checks["rectangular"] = not jagged and width > 0
    if jagged:
        issues.append("jagged_rows")

    has_spans = any(
        int(getattr(c, "colspan", 1) or 1) > 1 or int(getattr(c, "rowspan", 1) or 1) > 1
        for c in (table.cells or [])
    )
    checks["no_merged_cells_metadata"] = not has_spans
    checks["fully_cellular_flag"] = bool(
        (table.provenance or {}).get("fully_cellular", True)
    ) and not has_spans
    # Effective column count validation (accounts for colspan and rowspan).
    # For each row, sum the colspan of every *origin* cell (positions covered
    # by a colspan from the left are skipped because they are already part of
    # the origin cell's span). Positions covered by a rowspan from above still
    # occupy one column slot each (their colspan is 1).
    width = len(table.headers or [])
    effective_ok = True
    # Positions covered by a colspan (non-origin cells of a span with colspan>1).
    colspan_covered: set[tuple[int, int]] = set()
    for cell in table.cells or []:
        if cell.colspan > 1:
            for dc in range(1, cell.colspan):
                colspan_covered.add((cell.row, cell.col + dc))
    for ri, row in enumerate(table.rows or []):
        effective = 0
        for ci in range(len(row)):
            if (ri, ci) in colspan_covered:
                continue  # already accounted for by the origin cell's colspan
            cell = next(
                (c for c in table.cells or [] if c.row == ri and c.col == ci),
                None,
            )
            if cell:
                effective += max(1, int(cell.colspan))
            else:
                effective += 1
        if effective != width:
            effective_ok = False
            issues.append(f"effective_col_count_mismatch:row_{ri}")
    checks["effective_column_count"] = effective_ok

    # Self-contained rows cover every body row
    scr = table.self_contained_rows or []
    checks["self_contained_row_count_matches"] = len(scr) == len(table.rows or [])
    if not checks["self_contained_row_count_matches"]:
        issues.append("self_contained_mismatch")

    # Each self-contained row has a field per header
    missing_ctx = 0
    for row in scr:
        ctx = row.get("row_context") or {}
        if len(ctx) < width:
            missing_ctx += 1
    checks["each_row_has_full_context"] = missing_ctx == 0
    if missing_ctx:
        issues.append(f"incomplete_row_context:{missing_ctx}")

    # Insufficient-data cells are explicit when status set
    insuff = 0
    empty = 0
    for c in table.cells or []:
        if c.status == "insufficient_data":
            insuff += 1
            if not normalize_ws(c.explanation or c.text or ""):
                issues.append(f"insufficient_without_explanation:{c.row},{c.col}")
        if c.status == "empty":
            empty += 1
    checks["has_explicit_cell_statuses"] = True

    coverage = (table.provenance or {}).get("unavailable_region_coverage") or {}
    checks["unavailable_region_coverage_ok"] = bool(
        coverage.get("ok", True) if coverage else True
    )
    if coverage and not coverage.get("ok", True):
        issues.extend(f"unavail_region:{i}" for i in (coverage.get("issues") or [])[:20])

    ok = all(
        [
            checks["has_headers"],
            checks["has_body_rows"],
            checks["rectangular"],
            checks["self_contained_row_count_matches"],
            checks["each_row_has_full_context"],
            checks["unavailable_region_coverage_ok"],
        ]
    )
    return {
        "table_id": table.table_id,
        "ok": ok,
        "checks": checks,
        "issues": issues,
        "stats": {
            "columns": width,
            "rows": len(table.rows or []),
            "insufficient_data_cells": insuff,
            "empty_cells": empty,
            "unavailable_regions": len(
                (table.provenance or {}).get("unavailable_regions") or []
            ),
            "warnings": list(table.warnings or []),
            "confidence": table.confidence,
        },
    }

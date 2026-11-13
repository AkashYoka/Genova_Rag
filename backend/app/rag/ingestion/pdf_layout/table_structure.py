"""Domain-agnostic structural validation for extracted tables.

Preserves raw Textract evidence. Emits explicit validation statuses and reason
codes. Never fabricates missing cells or headers.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.rag.ingestion.pdf_layout.types import normalize_ws

# Public validation statuses (distinct from lookup statuses).
STRUCTURE_VALID = "structure_valid"
STRUCTURE_NEEDS_VERIFICATION = "structure_needs_verification"
STRUCTURE_INVALID = "structure_invalid"

# Reason codes
REASON_EMPTY_GRID = "empty_grid"
REASON_INCONSISTENT_ROW_WIDTHS = "inconsistent_row_widths"
REASON_MISSING_HEADERS = "missing_headers"
REASON_FRAGMENTED_HEADERS = "fragmented_headers"
REASON_INVALID_MERGED_SPAN = "invalid_merged_span"
REASON_SPARSE_GRID = "sparse_grid"
REASON_LOW_EXTRACTION_CONFIDENCE = "low_extraction_confidence"
REASON_UNRELIABLE_AXES = "unreliable_axes"
REASON_MISSING_CELLS = "missing_cells"
REASON_HEADER_RECONSTRUCTION_UNCERTAIN = "header_reconstruction_uncertain"

_DIGIT_RE = re.compile(r"\d")
_SYNTHETIC_HEADER_RE = re.compile(r"(?i)^column\s+\d+$")


@dataclass
class StructureValidation:
    status: str = STRUCTURE_VALID
    reason_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extraction_confidence: float = 0.0
    interpretation_confidence: Optional[float] = None
    reliable_for_lookup: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _row_widths(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[int]:
    widths = [len(headers)] if headers is not None else []
    widths.extend(len(r) for r in rows)
    return widths


def _merged_span_errors(
    merged_cells: Sequence[Dict[str, Any]],
    *,
    n_rows: int,
    n_cols: int,
) -> List[str]:
    errors: List[str] = []
    for m in merged_cells or []:
        try:
            r0 = int(m.get("row_index") or 0)
            c0 = int(m.get("col_index") or 0)
            rs = int(m.get("row_span") or 1)
            cs = int(m.get("col_span") or 1)
        except (TypeError, ValueError):
            errors.append("non_numeric_merged_span")
            continue
        if rs < 1 or cs < 1:
            errors.append("non_positive_merged_span")
        if r0 < 0 or c0 < 0:
            errors.append("negative_merged_origin")
        if n_cols > 0 and c0 + cs > n_cols + 1:
            # Allow slight overshoot from 0-based vs filled clones.
            if c0 >= n_cols or c0 + cs > n_cols + 2:
                errors.append("merged_span_exceeds_columns")
        if n_rows > 0 and r0 >= n_rows + 4:
            errors.append("merged_span_exceeds_rows")
    return errors


def validate_table_structure(
    table: Dict[str, Any],
    *,
    interpretation_confidence: Optional[float] = None,
) -> StructureValidation:
    """Validate grid integrity without mutating the table."""
    headers = [normalize_ws(h) for h in (table.get("headers") or [])]
    rows = [[normalize_ws(c) for c in (r or [])] for r in (table.get("rows") or [])]
    header_rows = table.get("header_rows") or []
    merged = table.get("merged_cells") or []
    low_conf = table.get("low_confidence_cells") or []
    recon = table.get("header_reconstruction") if isinstance(table.get("header_reconstruction"), dict) else {}
    textract_conf = float(table.get("textract_confidence") or 0.0)
    extraction_conf = textract_conf / 100.0 if textract_conf > 1.0 else textract_conf
    if extraction_conf <= 0 and low_conf:
        extraction_conf = 0.4
    elif extraction_conf <= 0:
        extraction_conf = 0.7 if headers and rows else 0.0

    reasons: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {
        "n_headers": len(headers),
        "n_body_rows": len(rows),
    }

    if not headers and not rows:
        return StructureValidation(
            status=STRUCTURE_INVALID,
            reason_codes=[REASON_EMPTY_GRID],
            extraction_confidence=extraction_conf,
            interpretation_confidence=interpretation_confidence,
            reliable_for_lookup=False,
            details=details,
        )

    widths = _row_widths(headers, rows)
    if widths:
        details["min_width"] = min(widths)
        details["max_width"] = max(widths)
        if max(widths) - min(widths) > 1:
            reasons.append(REASON_INCONSISTENT_ROW_WIDTHS)
            warnings.append(
                f"row_widths_vary:{min(widths)}-{max(widths)}"
            )

    n_cols = max(widths) if widths else 0
    n_rows = len(rows) + (1 if headers else 0)
    span_errs = _merged_span_errors(merged, n_rows=n_rows, n_cols=n_cols)
    if span_errs:
        reasons.append(REASON_INVALID_MERGED_SPAN)
        warnings.extend(span_errs[:8])

    nonempty_headers = [h for h in headers if h]
    if not nonempty_headers:
        reasons.append(REASON_MISSING_HEADERS)
    synthetic = sum(1 for h in headers if _SYNTHETIC_HEADER_RE.match(h))
    if synthetic and synthetic >= max(1, len(headers) // 2):
        reasons.append(REASON_FRAGMENTED_HEADERS)
        warnings.append(f"synthetic_headers:{synthetic}")

    if recon.get("needs_verification") or (
        float(recon.get("normalization_confidence") or 1.0) < 0.55
    ):
        reasons.append(REASON_HEADER_RECONSTRUCTION_UNCERTAIN)
        warnings.extend(list(recon.get("warnings") or [])[:6])

    # Sparse body: many empty value cells.
    if rows and n_cols >= 2:
        empty_cells = 0
        total_value_cells = 0
        for row in rows:
            for cell in row[1:n_cols]:
                total_value_cells += 1
                if not cell:
                    empty_cells += 1
        sparse_ratio = empty_cells / total_value_cells if total_value_cells else 0.0
        details["sparse_ratio"] = round(sparse_ratio, 3)
        if sparse_ratio >= 0.65 and total_value_cells >= 4:
            reasons.append(REASON_SPARSE_GRID)

    if low_conf:
        # Distinguish sparse_row annotations from truly low OCR confidence.
        low_ocr = [
            c
            for c in low_conf
            if isinstance(c, dict)
            and c.get("reason") not in {"sparse_row"}
            and float(c.get("confidence") or 100) < 50
        ]
        if low_ocr or any(
            isinstance(c, dict) and c.get("reason") == "header_reconstruction_uncertain"
            for c in low_conf
        ):
            reasons.append(REASON_LOW_EXTRACTION_CONFIDENCE)
            details["low_confidence_cell_count"] = len(low_conf)

    # Axis reliability: at least one header or first-column label with digits/units.
    axisish = 0
    for h in headers:
        if _DIGIT_RE.search(h) or len(h) >= 3:
            axisish += 1
    if rows:
        first_col = [r[0] for r in rows if r and r[0]]
        if first_col:
            axisish += 1
    if axisish == 0 and headers:
        reasons.append(REASON_UNRELIABLE_AXES)

    # Missing cells vs declared spans in cells metadata.
    cells = table.get("cells") or []
    if cells and n_cols >= 2 and n_rows >= 2:
        expected = set()
        # Body-oriented check is soft; only flag severe gaps.
        occupied = {
            (int(c.get("row_index") or -1), int(c.get("col_index") or -1))
            for c in cells
            if isinstance(c, dict)
        }
        header_count = len(header_rows) if header_rows else (1 if headers else 0)
        gaps = 0
        for r in range(header_count, header_count + len(rows)):
            for c in range(n_cols):
                expected.add((r, c))
                if (r, c) not in occupied and (not rows[r - header_count] or c >= len(rows[r - header_count]) or not rows[r - header_count][c]):
                    # Empty is allowed; only count if cell metadata claims presence elsewhere.
                    pass
        details["cell_metadata_count"] = len(occupied)

    # Deduplicate reasons preserving order.
    reasons = list(dict.fromkeys(reasons))
    hard_invalid = {
        REASON_EMPTY_GRID,
        REASON_MISSING_HEADERS,
    }
    soft_block = {
        REASON_FRAGMENTED_HEADERS,
        REASON_UNRELIABLE_AXES,
        REASON_INVALID_MERGED_SPAN,
        REASON_INCONSISTENT_ROW_WIDTHS,
    }

    if any(r in hard_invalid for r in reasons) or (
        REASON_MISSING_HEADERS in reasons and not rows
    ):
        status = STRUCTURE_INVALID
        reliable = False
    elif reasons and (any(r in soft_block for r in reasons) or extraction_conf < 0.45):
        status = STRUCTURE_NEEDS_VERIFICATION
        # Fragmented/unreliable axes: do not allow definitive lookup.
        reliable = not any(
            r in {REASON_FRAGMENTED_HEADERS, REASON_UNRELIABLE_AXES, REASON_MISSING_HEADERS}
            for r in reasons
        )
        if extraction_conf < 0.35:
            reliable = False
    elif reasons:
        status = STRUCTURE_NEEDS_VERIFICATION
        reliable = True  # sparse/low-conf alone: lookup may still work with caution
    else:
        status = STRUCTURE_VALID
        reliable = True

    if interpretation_confidence is not None and interpretation_confidence < 0.4:
        warnings.append("low_interpretation_confidence")

    return StructureValidation(
        status=status,
        reason_codes=reasons,
        warnings=list(dict.fromkeys(warnings)),
        extraction_confidence=round(max(0.0, min(1.0, extraction_conf)), 3),
        interpretation_confidence=interpretation_confidence,
        reliable_for_lookup=reliable,
        details=details,
    )


def attach_structure_validation(
    table: Dict[str, Any],
    *,
    interpretation_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a shallow copy with ``structure_validation`` attached (raw preserved)."""
    und = table.get("llm_understanding") if isinstance(table.get("llm_understanding"), dict) else {}
    interp_conf = interpretation_confidence
    if interp_conf is None and und:
        try:
            interp_conf = float(und.get("confidence"))
        except (TypeError, ValueError):
            interp_conf = None
    validation = validate_table_structure(
        table, interpretation_confidence=interp_conf
    )
    out = dict(table)
    out["structure_validation"] = validation.to_dict()
    if validation.status != STRUCTURE_VALID:
        out["needs_verification"] = True
    return out

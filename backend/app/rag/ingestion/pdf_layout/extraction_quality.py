"""Deterministic table-structure quality checks (domain-agnostic).

Separates OCR confidence from structure confidence. High OCR scores do not
imply a valid grid. Failed checks trigger DetectDocumentText fallback or
``needs_verification``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag.ingestion.pdf_layout.types import TableData, normalize_ws


@dataclass
class TableQualityResult:
    ok: bool
    structure_confidence: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "structure_confidence": self.structure_confidence,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def _grid_width(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> int:
    return max([len(headers)] + [len(r) for r in rows] or [0])


def evaluate_table_quality(
    table: TableData,
    *,
    min_cols: Optional[int] = None,
    min_rows: Optional[int] = None,
    max_empty_cell_ratio: Optional[float] = None,
    min_structure_confidence: Optional[float] = None,
) -> TableQualityResult:
    """Score structural reliability of a reconstructed table."""
    settings = get_settings()
    min_cols = int(
        min_cols
        if min_cols is not None
        else getattr(settings, "PDF_TABLE_QUALITY_MIN_COLS", 2)
    )
    min_rows = int(
        min_rows
        if min_rows is not None
        else getattr(settings, "PDF_TABLE_QUALITY_MIN_ROWS", 1)
    )
    max_empty = float(
        max_empty_cell_ratio
        if max_empty_cell_ratio is not None
        else getattr(settings, "PDF_TABLE_QUALITY_MAX_EMPTY_CELL_RATIO", 0.65)
    )
    min_conf = float(
        min_structure_confidence
        if min_structure_confidence is not None
        else getattr(settings, "PDF_TABLE_QUALITY_MIN_STRUCTURE_CONFIDENCE", 0.35)
    )

    headers = [normalize_ws(h) for h in (table.headers or [])]
    rows = [[normalize_ws(c) for c in row] for row in (table.rows or [])]
    width = _grid_width(headers, rows)
    reasons: List[str] = []
    score = 1.0

    if width < min_cols:
        reasons.append("too_few_columns")
        score -= 0.45
    if len(rows) < min_rows:
        reasons.append("too_few_rows")
        score -= 0.45

    # Jagged row widths.
    if rows:
        widths = [len(r) for r in rows]
        if max(widths) - min(widths) >= 2:
            reasons.append("jagged_row_widths")
            score -= 0.15

    # Empty-cell ratio over body.
    total_cells = sum(max(width, len(r)) for r in rows) or 0
    empty_cells = 0
    for row in rows:
        padded = list(row) + [""] * max(0, width - len(row))
        empty_cells += sum(1 for c in padded[:width] if not c)
    empty_ratio = (empty_cells / total_cells) if total_cells else 1.0
    if total_cells and empty_ratio > max_empty:
        reasons.append("high_empty_cell_ratio")
        score -= 0.25

    # Duplicate consecutive rows (OCR / geometry collapse).
    dup = 0
    for i in range(1, len(rows)):
        if rows[i] == rows[i - 1] and any(rows[i]):
            dup += 1
    if rows and dup / max(1, len(rows) - 1) > 0.4:
        reasons.append("duplicate_row_ratio")
        score -= 0.2

    # Header presence: leaf headers that are mostly synthetic Column N are weak.
    hard_fail_reasons: List[str] = []
    if headers:
        synthetic = sum(
            1 for h in headers if h.lower().startswith("column ") or not h
        )
        majority = max(2, (len(headers) + 1) // 2)
        if synthetic >= majority:
            reasons.append("weak_headers")
            score -= 0.35
            # Majority-synthetic headers are not trustworthy for TableRAG.
            hard_fail_reasons.append("weak_headers")
    elif rows:
        reasons.append("missing_headers")
        score -= 0.1

    # Bbox sanity.
    bbox = table.bbox
    if bbox is not None and getattr(bbox, "area", 0) <= 0:
        reasons.append("invalid_bbox")
        score -= 0.1

    # Prior engine confidence (structure channel if set). Keep weight modest so
    # a high prior cannot override clear structural failures.
    prior = float(table.structure_confidence or 0.0)
    if prior > 0:
        score = 0.75 * score + 0.25 * max(0.0, min(1.0, prior))

    score = max(0.0, min(1.0, round(score, 3)))
    hard_fail_reasons.extend(
        r for r in ("too_few_columns", "too_few_rows") if r in reasons
    )
    ok = score >= min_conf and not hard_fail_reasons

    return TableQualityResult(
        ok=ok,
        structure_confidence=score,
        reasons=reasons,
        metrics={
            "width": width,
            "n_rows": len(rows),
            "empty_cell_ratio": round(empty_ratio, 3),
            "duplicate_rows": dup,
            "min_structure_confidence": min_conf,
        },
    )


def filter_quality_tables(
    tables: Sequence[TableData],
) -> tuple[List[TableData], List[TableData], List[Dict[str, Any]]]:
    """Split tables into (accepted, rejected, reports)."""
    accepted: List[TableData] = []
    rejected: List[TableData] = []
    reports: List[Dict[str, Any]] = []
    for table in tables:
        result = evaluate_table_quality(table)
        table.structure_confidence = result.structure_confidence
        if table.provenance is None:
            table.provenance = {}
        notes = list((table.provenance or {}).get("quality_notes") or [])
        notes.extend(result.reasons)
        prov = dict(table.provenance or {})
        prov["quality_notes"] = list(dict.fromkeys(notes))
        prov["quality"] = result.to_dict()
        table.provenance = prov
        if not result.ok:
            table.low_confidence_cells = list(table.low_confidence_cells or []) + [
                {
                    "row": -1,
                    "col": -1,
                    "text": "",
                    "confidence": result.structure_confidence * 100.0,
                    "reason": "structure_quality_failed",
                    "details": result.reasons,
                }
            ]
            rejected.append(table)
        else:
            accepted.append(table)
        reports.append(
            {
                "page": table.page,
                "engine": table.extraction_engine or table.extraction_method,
                **result.to_dict(),
            }
        )
    return accepted, rejected, reports

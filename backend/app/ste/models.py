"""Typed structures for standalone table extraction."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def iou(self, other: "BBox") -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Word:
    text: str
    bbox: BBox
    page: int
    confidence: float = 100.0
    polygon_px: List[int] = field(default_factory=list)


@dataclass
class Cell:
    row: int
    col: int
    text: str
    status: str = "present"
    # present|empty|insufficient_data|not_applicable|not_reported|
    # illegible|redacted|uncertain
    original: str = ""
    unit: str = ""
    bbox: Optional[BBox] = None
    confidence: float = 0.0
    explanation: str = ""
    region_id: str = ""
    colspan: int = 1
    rowspan: int = 1

    def to_dict(self) -> Dict[str, Any]:
        null_statuses = {
            "empty",
            "insufficient_data",
            "not_applicable",
            "not_reported",
            "illegible",
            "redacted",
            "uncertain",
        }
        value: Any = None if self.status in null_statuses else (self.text or None)
        out = {
            "row": self.row,
            "col": self.col,
            "value": value,
            "status": self.status,
            "original": self.original or self.text,
            "unit": self.unit or None,
            "bbox": self.bbox.as_list() if self.bbox else None,
            "confidence": round(float(self.confidence), 2),
        }
        if self.explanation:
            out["explanation"] = self.explanation
        if self.region_id:
            out["region_id"] = self.region_id
        if int(self.colspan or 1) != 1:
            out["colspan"] = int(self.colspan)
        if int(self.rowspan or 1) != 1:
            out["rowspan"] = int(self.rowspan)
        return out


@dataclass
class ExtractedTable:
    table_id: str
    page: int
    bbox: BBox
    caption: str = ""
    headers: List[str] = field(default_factory=list)
    header_rows: List[List[str]] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    cells: List[Cell] = field(default_factory=list)
    footnotes: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    extraction_method: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    # Self-contained rows produced after reconstruction.
    self_contained_rows: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page": self.page,
            "pages": list(self.pages or [self.page]),
            "bbox": self.bbox.as_list(),
            "caption": self.caption,
            "headers": list(self.headers),
            "header_rows": [list(r) for r in self.header_rows],
            "rows": [list(r) for r in self.rows],
            "cells": [c.to_dict() for c in self.cells],
            "footnotes": list(self.footnotes),
            "extraction_method": self.extraction_method,
            "confidence": round(float(self.confidence), 4),
            "warnings": list(self.warnings),
            "self_contained_rows": list(self.self_contained_rows),
            "provenance": dict(self.provenance or {}),
        }


@dataclass
class PageGateResult:
    page: int
    has_tables: bool
    reason: str
    native_table_count: int = 0


@dataclass
class DocumentResult:
    source_path: str
    page_count: int
    gated_pages: List[PageGateResult] = field(default_factory=list)
    processed_pages: List[int] = field(default_factory=list)
    skipped_pages: List[int] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    ocr_backend: str = ""
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "page_count": self.page_count,
            "processed_pages": list(self.processed_pages),
            "skipped_pages": list(self.skipped_pages),
            "table_count": len(self.tables),
            "ocr_backend": self.ocr_backend,
            "errors": list(self.errors),
            "gated_pages": [asdict(g) for g in self.gated_pages],
            "meta": dict(self.meta or {}),
            "tables": [t.to_dict() for t in self.tables],
        }


def pad_row(row: Sequence[str], width: int) -> List[str]:
    cells = [normalize_ws(c) for c in row]
    if len(cells) < width:
        cells = cells + [""] * (width - len(cells))
    return cells[:width]

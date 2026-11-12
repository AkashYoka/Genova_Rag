"""Typed structures shared by the layout-aware PDF pipeline.

Coordinates are PDF user-space points (origin top-left), matching PyMuPDF.
OCR engines return raster pixel coordinates; ``ocr_engine`` converts them
before constructing these objects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_any(cls, value: Any) -> Optional["BBox"]:
        if value is None:
            return None
        if isinstance(value, BBox):
            return value
        try:
            if isinstance(value, dict):
                x0 = value.get("x0", value.get("left", 0))
                y0 = value.get("y0", value.get("top", 0))
                x1 = value.get("x1", value.get("right", 0))
                y1 = value.get("y1", value.get("bottom", 0))
            else:
                x0, y0, x1, y1 = list(value)[:4]
            x0, x1 = sorted((float(x0), float(x1)))
            y0, y1 = sorted((float(y0), float(y1)))
            return cls(x0, y0, x1, y1)
        except Exception:
            return None

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def intersection_area(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: "BBox", tolerance: float = 0.0) -> bool:
        return (
            other.x0 >= self.x0 - tolerance
            and other.y0 >= self.y0 - tolerance
            and other.x1 <= self.x1 + tolerance
            and other.y1 <= self.y1 + tolerance
        )

    def normalized(self, page_width: float, page_height: float) -> List[float]:
        if page_width <= 0 or page_height <= 0:
            return [round(self.x0, 4), round(self.y0, 4), round(self.x1, 4), round(self.y1, 4)]
        return [
            round(max(0.0, min(1.0, self.x0 / page_width)), 4),
            round(max(0.0, min(1.0, self.y0 / page_height)), 4),
            round(max(0.0, min(1.0, self.x1 / page_width)), 4),
            round(max(0.0, min(1.0, self.y1 / page_height)), 4),
        ]

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class TextWord:
    text: str
    bbox: BBox
    page: int = 1
    confidence: float = 100.0

    @property
    def height(self) -> float:
        return self.bbox.height


@dataclass
class TextLine:
    words: List[TextWord]
    page: int = 1
    font_size: float = 0.0
    bold: bool = False
    source: str = "native"

    @property
    def bbox(self) -> BBox:
        if not self.words:
            return BBox(0, 0, 0, 0)
        return BBox(
            min(w.bbox.x0 for w in self.words),
            min(w.bbox.y0 for w in self.words),
            max(w.bbox.x1 for w in self.words),
            max(w.bbox.y1 for w in self.words),
        )

    @property
    def text(self) -> str:
        return normalize_ws(" ".join(w.text for w in self.words if w.text.strip()))

    @property
    def height(self) -> float:
        return self.bbox.height

    @property
    def confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)


@dataclass
class TextBlock:
    lines: List[TextLine]
    page: int = 1
    column: int = 0
    block_type: str = "paragraph"
    bbox_override: Optional[BBox] = None

    @property
    def bbox(self) -> BBox:
        if self.bbox_override is not None:
            return self.bbox_override
        if not self.lines:
            return BBox(0, 0, 0, 0)
        boxes = [line.bbox for line in self.lines]
        return BBox(
            min(b.x0 for b in boxes),
            min(b.y0 for b in boxes),
            max(b.x1 for b in boxes),
            max(b.y1 for b in boxes),
        )

    @property
    def text(self) -> str:
        parts: List[str] = []
        for line in self.lines:
            value = line.text.strip()
            if not value:
                continue
            if parts and parts[-1].endswith("-"):
                parts[-1] = parts[-1][:-1] + value
            else:
                parts.append(value)
        return normalize_ws(" ".join(parts))

    @property
    def font_size(self) -> float:
        sizes = [line.font_size or line.height for line in self.lines if line.font_size or line.height]
        return max(sizes) if sizes else 0.0


@dataclass
class TableData:
    page: int
    bbox: BBox
    headers: List[str]
    rows: List[List[str]]
    extraction_method: str = "native"
    caption: str = ""
    source_lines: int = 0
    # Cells whose OCR confidence was too low for trustworthy financial values.
    # Each entry: {"row": int, "col": int, "text": str, "confidence": float}
    low_confidence_cells: List[Dict[str, Any]] = field(default_factory=list)
    # Populated by table_normalize.repair_table_data / normalize_table_grid.
    linearized_facts: List[str] = field(default_factory=list)
    # Optional Textract / structured-cell metadata (backward compatible).
    # cells: [{row_index, col_index, text, confidence, bbox, row_span, col_span, is_header}]
    cells: List[Dict[str, Any]] = field(default_factory=list)
    header_rows: List[List[str]] = field(default_factory=list)
    # merged_cells: [{row_index, col_index, row_span, col_span, text}]
    merged_cells: List[Dict[str, Any]] = field(default_factory=list)
    textract_confidence: float = 0.0

    @property
    def n_cols(self) -> int:
        return max([len(self.headers)] + [len(r) for r in self.rows] or [0])

    @property
    def n_rows(self) -> int:
        return len(self.rows) + (1 if self.headers else 0)

    @staticmethod
    def _cell(value: Any) -> str:
        return normalize_ws(str(value or "")).replace("|", "\\|")

    @property
    def markdown(self) -> str:
        width = self.n_cols
        if width == 0:
            return ""
        headers = (self.headers + [""] * width)[:width]
        lines = [
            "| " + " | ".join(self._cell(c) for c in headers) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        for row in self.rows:
            cells = (list(row) + [""] * width)[:width]
            lines.append("| " + " | ".join(self._cell(c) for c in cells) + " |")
        facts = list(self.linearized_facts or [])
        if not facts:
            try:
                from app.rag.ingestion.pdf_layout.table_normalize import linearize_table_facts

                facts = linearize_table_facts(
                    self.headers, self.rows, caption=self.caption or ""
                )
            except Exception:
                facts = []
        if facts:
            lines.append("")
            lines.append("Table facts:")
            lines.extend(f"- {fact}" for fact in facts)
        return "\n".join(lines)

    def to_json(self, include_markdown: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "page": self.page,
            "headers": self.headers,
            "rows": self.rows,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "extraction_method": self.extraction_method,
            "bbox": list(self.bbox.as_tuple()),
            "caption": self.caption,
            "low_confidence_cells": self.low_confidence_cells,
            "needs_verification": bool(self.low_confidence_cells),
            "linearized_facts": list(self.linearized_facts or []),
            "cells": list(self.cells or []),
            "header_rows": [list(r) for r in (self.header_rows or [])],
            "merged_cells": list(self.merged_cells or []),
            "textract_confidence": float(self.textract_confidence or 0.0),
        }
        if include_markdown:
            data["markdown"] = self.markdown
        return data


@dataclass
class DocumentElement:
    element_type: str  # heading | paragraph | list | table | figure | caption
    text: str
    page_start: int
    page_end: int
    bbox: Optional[BBox] = None
    level: int = 0
    section: str = ""
    subsection: str = ""
    extraction_method: str = "native"
    content_type: str = "paragraph"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": self.element_type,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "bbox": list(self.bbox.as_tuple()) if self.bbox else None,
            "level": self.level,
            "section": self.section,
            "subsection": self.subsection,
            "extraction_method": self.extraction_method,
            "content_type": self.content_type,
            "metadata": self.metadata,
        }


@dataclass
class PageExtraction:
    page_number: int
    classification: str
    strategy: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    lines: List[TextLine] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    elements: List[DocumentElement] = field(default_factory=list)
    status: str = "processed"
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    ocr_confidence: float = 0.0
    ocr_backend: str = ""

    def to_report(self) -> Dict[str, Any]:
        return {
            "page": self.page_number,
            "classification": self.classification,
            "strategy": self.strategy,
            "status": self.status,
            "error": self.error,
            "warnings": self.warnings,
            "elapsed_s": round(self.elapsed_s, 3),
            "ocr_backend": self.ocr_backend,
            "ocr_confidence": round(self.ocr_confidence, 2),
            "metrics": self.metrics,
            "lines": len(self.lines),
            "tables": len(self.tables),
            "elements": len(self.elements),
            "element_types": _counts(e.element_type for e in self.elements),
        }


@dataclass
class ParsedDocument:
    filepath: str
    page_count: int
    pages: List[PageExtraction]
    elements: List[DocumentElement]
    chunks: List[Dict[str, Any]]
    report: Dict[str, Any]
    elapsed_s: float = 0.0


def _counts(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts

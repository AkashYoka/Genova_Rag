"""Typed, serializable table structures shared by the whole RAG pipeline.

A ``TableRecord`` is the unit of truth for a table: it keeps the complete grid,
resolved merged cells, per-cell coordinates, caption/footnotes, a stable
``table_id`` and the original extracted representation. Retrieval, prompting
and evidence validation all operate on this object instead of re-parsing
Markdown.

Domain-agnostic by design: no product names, column names, units or values are
special-cased anywhere in this module.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.rag.ingestion.pdf_layout.types import normalize_ws

TABLE_SCHEMA_VERSION = 1


def _norm(value: Any) -> str:
    return normalize_ws(str(value or ""))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def make_table_id(
    *,
    document_id: str = "",
    page: int = 0,
    bbox: Optional[Sequence[float]] = None,
    headers: Optional[Sequence[Any]] = None,
    rows: Optional[Sequence[Sequence[Any]]] = None,
) -> str:
    """Stable table identity: same document + region + content -> same id.

    Only the first rows participate so a table split into row parts keeps the
    same identity across every part and across re-indexing.
    """
    payload = {
        "doc": str(document_id or ""),
        "page": _as_int(page),
        "bbox": [round(float(v), 1) for v in list(bbox or [])[:4]],
        "headers": [_norm(h) for h in list(headers or [])[:40]],
        "rows": [
            [_norm(c) for c in list(row or [])[:20]]
            for row in list(rows or [])[:4]
        ],
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"tbl_{digest[:16]}"


@dataclass
class TableCell:
    """One extracted cell, in full-grid coordinates (header rows included)."""

    row: int
    col: int
    text: str = ""
    row_span: int = 1
    col_span: int = 1
    header: bool = False
    merged_into: Optional[Tuple[int, int]] = None
    bbox: Optional[List[float]] = None
    confidence: float = 100.0

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "row": self.row,
            "col": self.col,
            "text": self.text,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "header": self.header,
        }
        if self.merged_into is not None:
            data["merged_into"] = list(self.merged_into)
        if self.bbox:
            data["bbox"] = list(self.bbox)
        if self.confidence != 100.0:
            data["confidence"] = self.confidence
        return data

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "TableCell":
        merged = payload.get("merged_into")
        return cls(
            row=_as_int(payload.get("row")),
            col=_as_int(payload.get("col")),
            text=_norm(payload.get("text")),
            row_span=max(1, _as_int(payload.get("row_span"), 1)),
            col_span=max(1, _as_int(payload.get("col_span"), 1)),
            header=bool(payload.get("header")),
            merged_into=(tuple(merged) if isinstance(merged, (list, tuple)) and len(merged) == 2 else None),
            bbox=list(payload.get("bbox") or []) or None,
            confidence=float(payload.get("confidence", 100.0) or 100.0),
        )


def _cols(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 0


@dataclass
class TableRecord:
    """Complete extracted table with provenance and verification data."""

    table_id: str
    document_id: str = ""
    page: int = 1
    bbox: List[float] = field(default_factory=list)
    page_width: float = 0.0
    page_height: float = 0.0
    caption: str = ""
    footnotes: List[str] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    header_row_count: int = 1
    row_labels: List[str] = field(default_factory=list)
    label_columns: List[int] = field(default_factory=lambda: [0])
    cells: List[TableCell] = field(default_factory=list)
    spans: List[Dict[str, Any]] = field(default_factory=list)
    original: Dict[str, Any] = field(default_factory=dict)
    extraction_method: str = ""
    source_lines: int = 0
    filename: str = ""
    source_index: int = 0
    schema_version: int = TABLE_SCHEMA_VERSION

    @property
    def n_cols(self) -> int:
        widths = [
            _cols(self.headers),
            max((_cols(row) for row in self.rows), default=0),
            max((cell.col + 1 for cell in self.cells), default=0),
        ]
        return max(widths or [0])

    @property
    def n_body_rows(self) -> int:
        return len(self.rows)

    def padded(self, row: Sequence[Any]) -> List[str]:
        width = self.n_cols
        values = [_norm(cell) for cell in row]
        return (values + [""] * width)[:width]

    @property
    def grid(self) -> List[List[str]]:
        """Header rows (merged labels) followed by body rows."""
        head = [self.padded(self.headers)] if self.headers else []
        return head + [self.padded(row) for row in self.rows]

    def column_label(self, col: int) -> str:
        if 0 <= col < len(self.headers) and _norm(self.headers[col]):
            return _norm(self.headers[col])
        return f"Column {col + 1}"

    def row_text(self, row: int) -> str:
        if row < 0 or row >= len(self.rows):
            return ""
        return " | ".join(value for value in self.padded(self.rows[row]) if value)

    def row_label(self, row: int) -> str:
        if 0 <= row < len(self.row_labels) and _norm(self.row_labels[row]):
            return _norm(self.row_labels[row])
        return self.row_text(row)

    def cell_text(self, row: int, col: int) -> str:
        if row < 0 or row >= len(self.rows):
            return ""
        values = self.padded(self.rows[row])
        if col < 0 or col >= len(values):
            return ""
        return values[col]

    def cell_id(self, row: int, col: int) -> str:
        return f"R{row + 1}C{col + 1}"

    def merged_note(self, row: int, col: int) -> str:
        """Describe the merge covering (row, col), if any."""
        for span in self.spans:
            try:
                span_row = _as_int(span.get("row"))
                span_col = _as_int(span.get("col"))
                row_span = max(1, _as_int(span.get("row_span"), 1))
                col_span = max(1, _as_int(span.get("col_span"), 1))
            except Exception:
                continue
            if (
                span_row <= row < span_row + row_span
                and span_col <= col < span_col + col_span
                and (row_span > 1 or col_span > 1)
            ):
                return (
                    f"merged cell anchored at R{span_row + 1}C{span_col + 1} "
                    f"(row_span={row_span}, col_span={col_span})"
                )
        return ""

    def render_for_prompt(self, *, max_chars: int = 6000) -> str:
        """Full, indexed table text for the interpretation/answer model.

        Every column and row carries a stable identifier so the model can name
        an exact intersection; the complete grid is included, not a fragment.
        """
        lines: List[str] = []
        source = f" [Source: {self.filename}, Page {self.page}]" if self.filename else ""
        lines.append(f'TABLE {self.table_id} | page {self.page}{source}')
        if self.caption:
            lines.append(f"Caption: {self.caption}")
        lines.append(f"Dimensions: {self.n_body_rows} body rows x {self.n_cols} columns")
        columns = [
            f'C{i + 1}={json.dumps(self.column_label(i), ensure_ascii=False)}'
            for i in range(self.n_cols)
        ]
        lines.append("Columns: " + " | ".join(columns))
        lines.append("Rows (R# identifiers are 1-based body rows):")
        for index, row in enumerate(self.rows):
            values = self.padded(row)
            cells = [
                f"C{col + 1}={json.dumps(values[col], ensure_ascii=False)}"
                for col in range(self.n_cols)
                if values[col]
            ]
            label = self.row_label(index)
            prefix = f'R{index + 1} (row label {json.dumps(label, ensure_ascii=False)})'
            lines.append(prefix + ": " + " | ".join(cells))
        span_notes = [
            f"R{_as_int(s.get('row')) + 1}C{_as_int(s.get('col')) + 1} spans "
            f"{max(1, _as_int(s.get('row_span'), 1))} rows x "
            f"{max(1, _as_int(s.get('col_span'), 1))} columns"
            for s in self.spans
            if max(1, _as_int(s.get("row_span"), 1)) > 1
            or max(1, _as_int(s.get("col_span"), 1)) > 1
        ]
        if span_notes:
            lines.append("Merged cells: " + "; ".join(span_notes))
        if self.footnotes:
            lines.append("Footnotes: " + " | ".join(self.footnotes))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[table truncated for context length]"
        return text

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "table_id": self.table_id,
            "document_id": self.document_id,
            "page": self.page,
            "bbox": list(self.bbox or []),
            "page_width": self.page_width,
            "page_height": self.page_height,
            "caption": self.caption,
            "footnotes": list(self.footnotes or []),
            "headers": list(self.headers or []),
            "rows": [list(row) for row in self.rows or []],
            "header_row_count": self.header_row_count,
            "row_labels": list(self.row_labels or []),
            "label_columns": list(self.label_columns or []),
            "cells": [cell.to_json() for cell in self.cells or []],
            "spans": list(self.spans or []),
            "original": self.original or {},
            "extraction_method": self.extraction_method,
            "source_lines": self.source_lines,
            "filename": self.filename,
            "n_rows": self.n_body_rows,
            "n_cols": self.n_cols,
        }

    @classmethod
    def from_json(cls, payload: Any) -> Optional["TableRecord"]:
        data = payload
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(data, dict):
            return None
        if isinstance(data.get("complete_table"), (dict, str)):
            return cls.from_json(data["complete_table"])
        if isinstance(data.get("table"), (dict, str)):
            return cls.from_json(data["table"])

        headers = [_norm(h) for h in (data.get("headers") or [])]
        rows = [
            [_norm(c) for c in (row if isinstance(row, (list, tuple)) else [row])]
            for row in (data.get("rows") or [])
        ]
        if not headers and not rows:
            return None
        width = max([len(headers)] + [len(row) for row in rows] or [0])
        if width == 0:
            return None
        headers = (headers + [""] * width)[:width]
        rows = [(row + [""] * width)[:width] for row in rows]

        document_id = str(data.get("document_id") or "")
        table_id = str(data.get("table_id") or "").strip()
        if not table_id:
            table_id = make_table_id(
                document_id=document_id,
                page=_as_int(data.get("page"), 1),
                bbox=data.get("bbox"),
                headers=headers,
                rows=rows,
            )

        row_labels = [_norm(label) for label in (data.get("row_labels") or [])]
        if len(row_labels) != len(rows):
            label_columns = list(data.get("label_columns") or [0])
            row_labels = []
            for row in rows:
                parts = [
                    row[col]
                    for col in label_columns
                    if 0 <= col < len(row) and _norm(row[col])
                ]
                row_labels.append(" ".join(parts) if parts else next((v for v in row if v), ""))

        cells = [
            TableCell.from_json(item)
            for item in (data.get("cells") or [])
            if isinstance(item, dict)
        ]

        return cls(
            table_id=table_id,
            document_id=document_id,
            page=_as_int(data.get("page"), 1),
            bbox=[float(v) for v in (data.get("bbox") or []) if isinstance(v, (int, float))],
            page_width=float(data.get("page_width") or 0.0),
            page_height=float(data.get("page_height") or 0.0),
            caption=_norm(data.get("caption")),
            footnotes=[_norm(f) for f in (data.get("footnotes") or []) if _norm(f)],
            headers=headers,
            rows=rows,
            header_row_count=max(1, _as_int(data.get("header_row_count"), 1)),
            row_labels=row_labels,
            label_columns=[_as_int(c) for c in (data.get("label_columns") or [0])],
            cells=cells,
            spans=[s for s in (data.get("spans") or []) if isinstance(s, dict)],
            original=data.get("original") if isinstance(data.get("original"), dict) else {},
            extraction_method=str(data.get("extraction_method") or ""),
            source_lines=_as_int(data.get("source_lines")),
            filename=str(data.get("filename") or ""),
            source_index=_as_int(data.get("source_index")),
        )


def records_to_json(records: Iterable[TableRecord]) -> List[Dict[str, Any]]:
    return [record.to_json() for record in records]


def _token_variants(text: str) -> List[str]:
    """Normalized numeric/text tokens used for value-presence checks."""
    cleaned = _norm(text)
    # Unify unicode minus/dashes and thousands separators.
    cleaned = cleaned.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned)
    return [token for token in re.findall(r"[0-9]+(?:\.[0-9]+)?[A-Za-z%/]*", cleaned)]

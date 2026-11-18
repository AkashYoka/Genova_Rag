"""Backend-agnostic table extraction contract.

The extraction layer is the only part of STE that knows about a document
engine. Backends return *raw* rectangular grids plus per-cell geometry; all
downstream reconstruction, validation and export stays backend independent.

Adding a new open-source engine later means implementing
:class:`TableExtractionBackend` and registering it in
``ste.backends.registry`` — no changes to reconstruction or exporters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from ste.models import BBox, normalize_ws


@dataclass
class RawTable:
    """A table exactly as the extraction engine reported it.

    Attributes:
        page: 1-based page number of the table's origin.
        bbox: Table bounding box in PDF points (top-left origin).
        grid: Rectangular-ish grid of cell text. Spanned cells are repeated
            in every covered logical position (fully cellular at the source).
        source: Engine identifier stored in ``original_extraction.json``.
        header_row_count: Number of leading header rows when the engine can
            report it, otherwise ``None`` so structural splitting is used.
        confidence: Engine confidence when available (0.0 when not provided).
        cells_meta: Per-logical-cell geometry/provenance dictionaries.
        pages: All pages covered by the table (multi-page tables).
        metadata: Engine-specific diagnostics kept for auditing.
    """

    page: int
    bbox: BBox
    grid: List[List[str]]
    source: str
    header_row_count: Optional[int] = None
    confidence: float = 0.0
    cells_meta: List[Dict[str, Any]] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.grid = [[normalize_ws(c) for c in row] for row in self.grid]
        if not self.pages:
            self.pages = [self.page]

    @property
    def n_rows(self) -> int:
        return len(self.grid)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.grid), default=0)

    def to_payload(self) -> Dict[str, Any]:
        """Serializable payload used by the extraction cache."""
        return {
            "page": self.page,
            "pages": list(self.pages),
            "bbox": self.bbox.as_list(),
            "source": self.source,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "header_row_count": self.header_row_count,
            "grid": [list(r) for r in self.grid],
            "cells_meta": list(self.cells_meta),
            "confidence": float(self.confidence),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RawTable":
        bbox = payload.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        return cls(
            page=int(payload.get("page") or 1),
            bbox=BBox(*[float(v) for v in bbox[:4]]),
            grid=[list(r) for r in (payload.get("grid") or [])],
            source=str(payload.get("source") or "unknown"),
            header_row_count=payload.get("header_row_count"),
            confidence=float(payload.get("confidence") or 0.0),
            cells_meta=list(payload.get("cells_meta") or []),
            pages=[int(p) for p in (payload.get("pages") or [payload.get("page") or 1])],
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class DocumentExtraction:
    """Backend output for one document (or a requested page subset)."""

    backend: str
    tables: List[RawTable] = field(default_factory=list)
    pages_processed: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    cache_hit: bool = False

    def to_payload(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "tables": [t.to_payload() for t in self.tables],
            "pages_processed": list(self.pages_processed),
            "warnings": list(self.warnings),
            "meta": dict(self.meta or {}),
            "elapsed_seconds": float(self.elapsed_seconds),
            "cache_hit": bool(self.cache_hit),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "DocumentExtraction":
        return cls(
            backend=str(payload.get("backend") or "unknown"),
            tables=[RawTable.from_payload(p) for p in payload.get("tables") or []],
            pages_processed=[int(p) for p in payload.get("pages_processed") or []],
            warnings=[str(w) for w in payload.get("warnings") or []],
            meta=dict(payload.get("meta") or {}),
            elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
            cache_hit=True,
        )


@runtime_checkable
class TableExtractionBackend(Protocol):
    """Minimal interface every extraction engine must satisfy."""

    name: str

    def available(self) -> bool:
        """Return True when the engine and its models can be imported."""

    def extract(
        self,
        pdf_path: str,
        *,
        pages: Optional[Sequence[int]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> DocumentExtraction:
        """Extract raw tables from ``pdf_path`` (1-based ``pages`` filter)."""

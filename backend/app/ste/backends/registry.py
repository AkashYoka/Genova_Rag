"""Backend registry: Textract AnalyzeDocument primary, PyMuPDF fallback."""
from __future__ import annotations

from typing import Dict, List, Optional

from ste.backends.base import DocumentExtraction, RawTable, TableExtractionBackend
from ste.backends.pymupdf_backend import PymupdfBackend
from ste.backends.textract_analyze_backend import (
    TextractAnalyzeBackend,
    textract_available,
)

__all__ = [
    "DocumentExtraction",
    "RawTable",
    "TableExtractionBackend",
    "TextractAnalyzeBackend",
    "PymupdfBackend",
    "available_backends",
    "create_backend",
    "resolve_backend_name",
]


def available_backends() -> List[str]:
    """Return backends whose runtime dependencies are importable."""
    names: List[str] = []
    if textract_available():
        names.append("textract")
    try:
        import pymupdf  # noqa: F401

        names.append("pymupdf")
    except Exception:
        pass
    return names


def resolve_backend_name(name: Optional[str]) -> str:
    """Resolve ``auto`` to Textract when boto3 is available, else PyMuPDF."""
    requested = (name or "textract").strip().lower()
    if requested in {"docling", "docling_tableformer"}:
        # Docling removed — map legacy names to Textract AnalyzeDocument.
        return "textract"
    if requested != "auto":
        return requested
    return "textract" if textract_available() else "pymupdf"


def create_backend(
    name: Optional[str] = "textract",
    *,
    options: Optional[Dict[str, object]] = None,
) -> TableExtractionBackend:
    """Instantiate a backend by name.

    Args:
        name: ``textract``, ``pymupdf`` or ``auto``.
        options: Backend-specific options (dpi, aws_region, stitch_multipage).
    """
    resolved = resolve_backend_name(name)
    opts: Dict[str, object] = dict(options or {})
    if resolved == "textract":
        return TextractAnalyzeBackend(
            dpi=int(opts.get("dpi") or 200),
            aws_region=str(opts["aws_region"]) if opts.get("aws_region") else None,
            tables_only=bool(opts.get("tables_only", True)),
            stitch_multipage=bool(opts.get("stitch_multipage", True)),
            max_workers=int(opts.get("max_workers") or 4),
        )
    if resolved == "pymupdf":
        return PymupdfBackend(
            dpi=int(opts.get("dpi") or 200),
            ocr_engine=str(opts.get("ocr_engine") or "tesseract"),
        )
    raise ValueError(
        f"Unknown extraction backend: {name!r}. "
        f"Available: {', '.join(available_backends()) or 'none'}"
    )

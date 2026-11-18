"""Open PDFs with PyMuPDF (no Genova imports)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple


def open_pdf(path: str | Path) -> Any:
    import pymupdf as fitz

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    return fitz.open(str(p))


def page_sizes(doc: Any) -> List[Tuple[int, float, float]]:
    """Return [(page_number_1based, width, height), ...]."""
    out: List[Tuple[int, float, float]] = []
    for i, page in enumerate(doc):
        out.append((i + 1, float(page.rect.width), float(page.rect.height)))
    return out


def page_text(page: Any) -> str:
    try:
        return page.get_text("text") or ""
    except Exception:
        return ""

"""Local PyMuPDF backend: native ruled tables, or Tesseract geometry fallback.

This is a dependency-light engine that runs fully offline and is useful when
Textract is unavailable. It contacts no cloud document service.

Strategy per page:
1. ``page.find_tables()`` (native vector ruling) -> grid.
2. Otherwise positioned words (native text layer, else local Tesseract OCR)
   -> dense word alignment grid.
"""
from __future__ import annotations

import importlib.util
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from ste.backends.base import DocumentExtraction, RawTable
from ste.models import BBox, Word, normalize_ws
from ste.multipage import stitch_multipage_tables
from ste.ocr_tesseract import words_from_image
from ste.page_renderer import render_page_image
from ste.pdf_loader import open_pdf, page_sizes, page_text
from ste.reconstructor import native_tables_from_page
from ste.structure_extractor import grid_from_words
from ste.table_detector import detect_regions, merge_regions, words_in_region

logger = logging.getLogger(__name__)


def _native_words(page: Any, page_number: int) -> List[Word]:
    words: List[Word] = []
    try:
        raw = page.get_text("words") or []
    except Exception:
        return words
    for item in raw:
        try:
            x0, y0, x1, y1, text = item[0], item[1], item[2], item[3], item[4]
        except (IndexError, TypeError):
            continue
        token = normalize_ws(str(text or ""))
        if not token:
            continue
        bbox = BBox(float(x0), float(y0), float(x1), float(y1))
        if bbox.area <= 0:
            continue
        words.append(Word(text=token, bbox=bbox, page=page_number, confidence=100.0))
    return words


class PymupdfBackend:
    """Offline PyMuPDF + local-OCR fallback backend."""

    name = "pymupdf"

    def __init__(self, *, dpi: int = 200, ocr_engine: str = "tesseract") -> None:
        self.dpi = dpi
        self.ocr_engine = ocr_engine

    def available(self) -> bool:
        return importlib.util.find_spec("pymupdf") is not None

    def _words(self, page: Any, page_number: int, width: float, height: float) -> List[Word]:
        words = _native_words(page, page_number)
        if words:
            return words
        if (self.ocr_engine or "none").lower() in {"none", "off", "disabled"}:
            return []
        try:
            image, _fmt, image_w, image_h, _dpi = render_page_image(page, dpi=self.dpi)
            return words_from_image(
                image,
                page_number=page_number,
                page_width=width,
                page_height=height,
                image_width=image_w,
                image_height=image_h,
            )
        except Exception as exc:
            logger.warning("local OCR failed on page %s: %s", page_number, exc)
            return []

    def extract(
        self,
        pdf_path: str,
        *,
        pages: Optional[Sequence[int]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> DocumentExtraction:
        started = time.perf_counter()
        doc = open_pdf(pdf_path)
        wanted = {int(p) for p in pages} if pages else None
        tables: List[RawTable] = []
        warnings: List[str] = []
        processed: List[int] = []
        page_sizes_map: Dict[int, tuple[float, float]] = {}

        try:
            for page_number, width, height in page_sizes(doc):
                if wanted is not None and page_number not in wanted:
                    continue
                processed.append(page_number)
                page_sizes_map[page_number] = (width, height)
                page = doc[page_number - 1]
                found_any = False

                for bbox, grid in native_tables_from_page(page):
                    if len(grid) < 2 or max((len(r) for r in grid), default=0) < 2:
                        continue
                    found_any = True
                    tables.append(
                        RawTable(
                            page=page_number,
                            bbox=bbox,
                            grid=grid,
                            source="pymupdf_find_tables",
                            header_row_count=None,
                            confidence=100.0,
                            metadata={"backend": "pymupdf", "native_ruling": True},
                        )
                    )
                if found_any:
                    continue

                words = self._words(page, page_number, width, height)
                if len(words) < 8:
                    continue
                regions = merge_regions(
                    detect_regions(
                        page, words, page_width=width, page_height=height
                    )
                )
                for region in regions:
                    region_words = words_in_region(words, region)
                    if len(region_words) < 8:
                        continue
                    grid, _anchors = grid_from_words(region_words, page_width=width)
                    if len(grid) < 2 or max((len(r) for r in grid), default=0) < 2:
                        continue
                    mean_conf = sum(w.confidence for w in region_words) / len(
                        region_words
                    )
                    tables.append(
                        RawTable(
                            page=page_number,
                            bbox=region,
                            grid=grid,
                            source="pymupdf_word_geometry",
                            header_row_count=None,
                            confidence=mean_conf,
                            metadata={
                                "backend": "pymupdf",
                                "native_ruling": False,
                                "word_count": len(region_words),
                                "used_text_layer": bool(
                                    (page_text(page) or "").strip()
                                ),
                            },
                        )
                    )
        finally:
            doc.close()

        tables, stitch_warnings = stitch_multipage_tables(
            tables,
            page_sizes=page_sizes_map,
            enabled=bool((options or {}).get("stitch_multipage", True)),
        )
        warnings.extend(stitch_warnings)
        return DocumentExtraction(
            backend=self.name,
            tables=tables,
            pages_processed=processed,
            warnings=warnings,
            meta={"backend": self.name, "dpi": self.dpi},
            elapsed_seconds=time.perf_counter() - started,
        )


__all__ = ["PymupdfBackend"]

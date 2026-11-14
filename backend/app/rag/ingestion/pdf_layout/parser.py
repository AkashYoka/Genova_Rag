"""Layout-aware PDF orchestration: classify → extract/OCR → analyze → chunk."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pymupdf as fitz

from app.config import get_settings
from app.rag.ingestion.pdf_layout.analyzer import (
    alignment_table_regions,
    detect_column_gutter,
    detect_repeating_margins,
    detect_ruled_table_boxes,
    extract_grid_positions,
    group_words_by_rows,
    table_from_words,
    words_to_lines,
)
from app.rag.ingestion.pdf_layout.chunking import chunk_elements, chunk_quality_report
from app.rag.ingestion.pdf_layout.classifier import classify_document, classify_page, compute_text_metrics
from app.rag.ingestion.pdf_layout.ocr_engine import (
    OcrError,
    OcrUnavailableError,
    ocr_page,
    ocr_status,
)
from app.rag.ingestion.pdf_layout.structure import (
    StructureState,
    body_font_size,
    build_page_elements,
)
from app.rag.ingestion.pdf_layout.types import (
    BBox,
    DocumentElement,
    PageExtraction,
    ParsedDocument,
    TableData,
    TextLine,
    normalize_ws,
)

logger = logging.getLogger(__name__)

class PdfLayoutError(RuntimeError):
    """Raised when the layout pipeline cannot produce usable content."""


def _textract_fallback_allowed(settings: Any = None) -> bool:
    """True only when both Textract master and tables flags are enabled.

    ``PDF_TEXTRACT_FALLBACK_ENABLED=false`` blocks every DetectDocumentText
    call (including the active Textract-primary path).
    """
    if settings is None:
        settings = get_settings()
    return bool(getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False)) and bool(
        getattr(settings, "PDF_TEXTRACT_TABLES_ENABLED", True)
    )


def _table_engine() -> str:
    """Active table engine label for diagnostics."""
    settings = get_settings()
    docling_on = bool(getattr(settings, "PDF_DOCLING_TABLES_ENABLED", False))
    textract_on = _textract_fallback_allowed(settings)
    if textract_on and not docling_on:
        return "textract_detect"
    if textract_on and docling_on:
        return "docling+textract_fallback"
    if docling_on:
        return "docling_only"
    return "none"


def _page_area(page: fitz.Page) -> float:
    return max(1.0, float(page.rect.width) * float(page.rect.height))


def _image_coverage(page: fitz.Page) -> float:
    """Fraction of the page covered by image objects (clipped union estimate)."""
    page_area = _page_area(page)
    rects: List[BBox] = []
    try:
        for info in page.get_image_info():
            bbox = BBox.from_any(info.get("bbox"))
            if bbox and bbox.area > 0:
                rects.append(bbox)
    except Exception:
        return 0.0
    if not rects:
        return 0.0
    # Sum with overlap correction via a coarse occupancy grid.
    cols, rows = 40, 60
    occupied = set()
    for rect in rects:
        c0 = max(0, int(rect.x0 / page.rect.width * cols))
        c1 = min(cols - 1, int(rect.x1 / page.rect.width * cols))
        r0 = max(0, int(rect.y0 / page.rect.height * rows))
        r1 = min(rows - 1, int(rect.y1 / page.rect.height * rows))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                occupied.add((r, c))
    return min(1.0, len(occupied) / float(cols * rows))


def _native_lines(page: fitz.Page, table_boxes: Sequence[BBox]) -> List[TextLine]:
    page_width = float(page.rect.width)
    raw = page.get_text("words") or []
    from app.rag.ingestion.pdf_layout.types import TextWord

    words: List[TextWord] = []
    for item in raw:
        try:
            x0, y0, x1, y1, text = item[0], item[1], item[2], item[3], item[4]
        except Exception:
            continue
        text = str(text or "").strip()
        if not text:
            continue
        bbox = BBox(float(x0), float(y0), float(x1), float(y1))
        if any(box.intersection_area(bbox) > 0.5 * bbox.area for box in table_boxes):
            continue
        words.append(TextWord(text=text, bbox=bbox, page=page.number + 1, confidence=100.0))
    lines = words_to_lines(words, page_width)
    for line in lines:
        line.source = "native"
    return lines


def _clean_table_rows(rows: Sequence[Sequence[Any]]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    for row in rows or []:
        values = [normalize_ws(str(cell or "")) for cell in (row or [])]
        if any(values):
            cleaned.append(values)
    return cleaned


_TABLE_CUE_RE = re.compile(
    r"(?i)\b("
    r"table\s*\d+|tbl\.?\s*\d+|"
    r"dosing\s+table|dose\s+table|schedule\s+table|"
    r"see\s+table|refer\s+to\s+table"
    r")\b"
)


def _page_looks_tabular(
    page: fitz.Page,
    *,
    fallback_tables: Sequence[TableData],
    page_text: str = "",
) -> Tuple[bool, str]:
    """Cheap gate: should Textract DetectDocumentText run on this page?

    Textract is for tabular data only — not narrative pages that merely mention
    a table. Strong signals (extracted grids / find_tables) always qualify.
    Ruled boxes qualify only when they look like mid-page grids (not full-page
    borders) and are backed by a table text cue.
    """
    if fallback_tables:
        return True, "fallback_tables"

    find_count = 0
    try:
        if hasattr(page, "find_tables"):
            finder = page.find_tables()
            found = list(getattr(finder, "tables", []) or [])
            find_count = len(found)
            if found:
                return True, "find_tables"
    except Exception:
        find_count = 0

    text = page_text or ""
    if not text:
        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""
    has_cue = bool(_TABLE_CUE_RE.search(text))

    try:
        ruled = detect_ruled_table_boxes(page)
    except Exception:
        ruled = []
    page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
    # Ignore near-full-page borders and tiny noise boxes.
    significant = [
        box
        for box in ruled
        if (0.08 * page_area) <= box.area <= (0.65 * page_area)
    ]
    if significant and (has_cue or find_count > 0):
        return True, "ruled_boxes_with_cue"

    # Narrative "see Table N" alone is not enough to spend a Textract call.
    return False, "no_tabular_signal"


def _repair_page_tables(
    tables: Sequence[TableData],
    *,
    filepath: str,
) -> List[TableData]:
    """Repair collapsed grids and attach linearized facts (drug-agnostic)."""
    from app.rag.ingestion.pdf_layout.table_normalize import repair_table_data

    repaired: List[TableData] = []
    for table in tables:
        try:
            repair_table_data(table, filepath=filepath, caption=table.caption or "")
        except Exception as exc:
            logger.debug(
                "Table repair skipped on page %s: %s",
                getattr(table, "page", "?"),
                exc,
            )
        repaired.append(table)
    return repaired


def _native_tables(page: fitz.Page) -> List[TableData]:
    if not hasattr(page, "find_tables"):
        return []
    try:
        finder = page.find_tables()
        found = list(getattr(finder, "tables", []) or [])
    except Exception as exc:
        logger.debug("Native table detection failed on page %s: %s", page.number + 1, exc)
        return []
    tables: List[TableData] = []
    for table in found:
        try:
            rows = _clean_table_rows(table.extract())
            bbox = BBox.from_any(getattr(table, "bbox", None))
            if not bbox or len(rows) < 2:
                continue
            width = max(len(row) for row in rows)
            if width < 2:
                continue
            rows = [row + [""] * (width - len(row)) for row in rows]
            headers = rows[0]
            data = rows[1:]
            if not data:
                continue
            tables.append(
                TableData(
                    page=page.number + 1,
                    bbox=bbox,
                    headers=headers,
                    rows=data,
                    extraction_method="native",
                    source_lines=len(rows),
                )
            )
        except Exception as exc:
            logger.debug("Skipping native table on page %s: %s", page.number + 1, exc)
    return tables


def _ocr_page_content(
    page: fitz.Page,
    *,
    exclude_boxes: Optional[Sequence[BBox]] = None,
    detect_tables: bool = True,
) -> Tuple[List[TextLine], List[TableData], float, str, Dict[str, Any]]:
    """OCR a page into positioned text lines.

    ``detect_tables=False`` disables the legacy ruled/alignment table
    reconstruction.  ``exclude_boxes`` removes words that belong to tables
    already extracted by another engine so table text does not leak into
    paragraph chunks.
    """
    result = ocr_page(page)
    words = result["words"]
    backend = result["backend"]
    confidence = result["mean_confidence"]
    if not words:
        raise OcrError("OCR returned no positioned words")

    excluded = list(exclude_boxes or [])
    if excluded:
        words = [
            word
            for word in words
            if not any(
                box.intersection_area(word.bbox) > 0.5 * word.bbox.area
                for box in excluded
            )
        ]

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    tables: List[TableData] = []
    consumed_boxes: List[BBox] = []
    used_words = list(words)

    if detect_tables:
        for bbox in detect_ruled_table_boxes(page):
            grid_x, grid_y = extract_grid_positions(page, bbox)
            if len(grid_x) > 16 or len(grid_y) > 90:
                grid_x, grid_y = [], []
            table = table_from_words(
                used_words,
                bbox,
                page=page.number + 1,
                page_width=page_width,
                page_height=page_height,
                grid_x=grid_x,
                grid_y=grid_y,
                method="ocr_ruled",
            )
            if table:
                tables.append(table)
                consumed_boxes.append(bbox)

    remaining = [
        word for word in used_words
        if not any(box.intersection_area(word.bbox) > 0.5 * word.bbox.area for box in consumed_boxes)
    ]
    lines = words_to_lines(remaining, page_width)

    if detect_tables:
        # Borderless ledger tables: detect aligned short numeric rows, then
        # remove their lines from normal paragraph flow.
        for bbox, _region_lines in alignment_table_regions(
            lines, page_width, page_height, excluded=consumed_boxes
        ):
            table = table_from_words(
                remaining,
                bbox,
                page=page.number + 1,
                page_width=page_width,
                page_height=page_height,
                method="ocr_alignment",
            )
            if table:
                tables.append(table)
                consumed_boxes.append(bbox)
        if consumed_boxes:
            lines = [
                line for line in lines
                if not any(box.intersection_area(line.bbox) > 0.5 * line.bbox.area for box in consumed_boxes)
            ]
    lines.sort(key=lambda line: (line.bbox.center_y, line.bbox.x0))
    details = {
        "ocr_elapsed_s": result.get("elapsed_s", 0),
        "ocr_dpi": result.get("dpi"),
        "pixel_size": result.get("pixel_size"),
        "ocr_words": len(words),
        "ocr_excluded_table_words": len(result["words"]) - len(words),
        "ocr_table_detection": bool(detect_tables),
        "ruled_tables": sum(1 for table in tables if table.extraction_method == "ocr_ruled"),
        "alignment_tables": sum(1 for table in tables if table.extraction_method == "ocr_alignment"),
    }
    return lines, tables, confidence, backend, details


def _apply_ocr_result(
    page_ext: PageExtraction,
    page: fitz.Page,
    *,
    lines: List[TextLine],
    confidence: float,
    backend: str,
    details: Dict[str, Any],
    image_coverage: float,
) -> None:
    page_ext.lines = lines
    page_ext.ocr_confidence = confidence
    page_ext.ocr_backend = backend
    page_ext.metrics.update(details)
    page_ext.metrics["ocr_chars"] = sum(len(line.text) for line in lines)
    page_ext.metrics["ocr_text_metrics"] = compute_text_metrics(
        "\n".join(line.text for line in lines),
        page_area=_page_area(page),
        image_coverage=image_coverage,
    )


def _extract_body_lines(
    page: fitz.Page,
    *,
    strategy: str,
    coverage: float,
    page_ext: PageExtraction,
    exclude_boxes: Sequence[BBox] = (),
    allow_ocr_retry: bool = True,
) -> str:
    """Extract paragraph lines without running any table engine.

    Used when table regions are already known and should be excluded from the
    body flow. Returns the (possibly changed) page strategy.
    """
    if strategy == "ocr":
        lines, _tables, confidence, backend, details = _ocr_page_content(
            page,
            exclude_boxes=exclude_boxes,
            detect_tables=False,
        )
        _apply_ocr_result(
            page_ext,
            page,
            lines=lines,
            confidence=confidence,
            backend=backend,
            details=details,
            image_coverage=coverage,
        )
        return strategy

    if strategy == "native":
        lines = _native_lines(page, exclude_boxes)
        page_ext.lines = lines
        if not lines and allow_ocr_retry and coverage >= 0.15:
            logger.warning(
                "Page %s has no native lines despite a text layer; retrying with OCR",
                page.number + 1,
            )
            page_ext.strategy = "ocr"
            strategy = "ocr"
            lines, _tables, confidence, backend, details = _ocr_page_content(
                page,
                exclude_boxes=exclude_boxes,
                detect_tables=False,
            )
            _apply_ocr_result(
                page_ext,
                page,
                lines=lines,
                confidence=confidence,
                backend=backend,
                details=details,
                image_coverage=coverage,
            )
        return strategy

    page_ext.status = "empty"
    page_ext.metrics["note"] = "blank page with no text or image"
    return strategy


def _table_extraction_report(
    pages: Sequence[PageExtraction],
    *,
    engine: str,
    engine_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate table metrics for evaluation and diagnostics."""
    processed = [page for page in pages if page.strategy != "none"]
    with_tables = [page for page in pages if page.tables]
    without_tables = [
        page.page_number for page in processed if not page.tables
    ]
    blank_pages = [page.page_number for page in pages if page.strategy == "none"]
    failures = [
        {
            "page": page.page_number,
            "error": str(
                page.metrics.get("textract_error")
                or page.metrics.get("docling_error")
                or ""
            ),
        }
        for page in pages
        if page.metrics.get("textract_error") or page.metrics.get("docling_error")
    ]

    table_entries: List[Dict[str, Any]] = []
    for page in pages:
        for index, table in enumerate(page.tables, start=1):
            table_entries.append(
                {
                    "page": page.page_number,
                    "index": index,
                    "extraction_method": table.extraction_method,
                    "extraction_engine": getattr(table, "extraction_engine", "")
                    or table.extraction_method,
                    "bbox": [round(value, 2) for value in table.bbox.as_tuple()],
                    "n_rows": table.n_rows,
                    "n_cols": table.n_cols,
                    "data_rows": len(table.rows),
                    "headers": list(table.headers),
                    "rows": [list(row) for row in table.rows],
                    "header_rows": [list(row) for row in (table.header_rows or [])],
                    "merged_cells": list(table.merged_cells or []),
                    "cell_count": len(table.cells or []),
                    "caption": table.caption,
                    "structure_confidence": float(
                        getattr(table, "structure_confidence", 0) or 0
                    ),
                    "ocr_confidence": float(getattr(table, "ocr_confidence", 0) or 0),
                }
            )

    elapsed = sum(
        float(page.metrics.get("textract_elapsed_s") or 0)
        + float(page.metrics.get("docling_elapsed_s") or 0)
        for page in pages
    )
    return {
        "engine": engine,
        "engine_info": dict(engine_info or {}),
        "pages_processed": len(processed),
        "pages_with_tables": len(with_tables),
        "pages_without_tables": without_tables,
        "blank_pages": blank_pages,
        "table_count": sum(len(page.tables) for page in pages),
        "merged_cell_count": sum(
            len(table.merged_cells or [])
            for page in pages
            for table in page.tables
        ),
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_s": round(elapsed, 3),
        "tables": table_entries,
    }


def _figure_elements(
    page: fitz.Page,
    lines: Sequence[TextLine],
    tables: Sequence[TableData],
    *,
    image_coverage: float,
    page_height: float,
) -> List[DocumentElement]:
    """Create figure elements for embedded figures on native pages.

    For fully scanned pages the page image is the transport layer, not a
    figure; OCR text is used instead of treating the whole page as a figure.
    """
    if image_coverage >= 0.8:
        return []
    figures: List[DocumentElement] = []
    captions = [line for line in lines if re.match(r"^\s*(figure|fig\.?|chart|graph|exhibit)\b", line.text, re.I)]
    try:
        infos = page.get_image_info()
    except Exception:
        return []
    for info in infos:
        bbox = BBox.from_any(info.get("bbox"))
        if not bbox or bbox.area < _page_area(page) * 0.01:
            continue
        if any(table.bbox.contains(bbox) for table in tables):
            continue
        caption = next(
            (
                line for line in captions
                if abs(line.bbox.y0 - bbox.y1) <= max(30.0, page_height * 0.04)
            ),
            None,
        )
        text = f"Figure on page {page.number + 1}"
        if caption:
            text += f": {caption.text}"
        figures.append(
            DocumentElement(
                element_type="figure",
                text=text,
                page_start=page.number + 1,
                page_end=page.number + 1,
                bbox=bbox,
                section="",
                subsection="",
                extraction_method="native",
                content_type="figure_caption",
                metadata={"has_caption": bool(caption)},
            )
        )
    return figures


def parse_pdf(
    filepath: str,
    *,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> ParsedDocument:
    """Parse a PDF into structured elements and retrieval-ready chunks."""
    settings = get_settings()
    started = time.perf_counter()
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    engine = _table_engine()
    fallback_on = _textract_fallback_allowed(settings)
    engine_info: Dict[str, Any] = {
        "engine": engine,
        "docling_enabled": bool(getattr(settings, "PDF_DOCLING_TABLES_ENABLED", False)),
        "docling_tableformer_mode": (
            getattr(settings, "PDF_DOCLING_TABLEFORMER_MODE", "accurate") or "accurate"
        ),
        "textract_fallback_enabled": fallback_on,
        "textract_primary": bool(
            fallback_on
            and not bool(getattr(settings, "PDF_DOCLING_TABLES_ENABLED", False))
        ),
        "pdf_textract_fallback_enabled": bool(
            getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False)
        ),
        "pdf_textract_tables_enabled": bool(
            getattr(settings, "PDF_TEXTRACT_TABLES_ENABLED", True)
        ),
        "textract_async_multipage": bool(
            getattr(settings, "PDF_TEXTRACT_ASYNC_MULTIPAGE", True)
        ),
        "tabular_only": bool(getattr(settings, "PDF_TEXTRACT_TABULAR_ONLY", True)),
        "analyze_document_tables": False,
        "region": (
            (getattr(settings, "PDF_TEXTRACT_REGION", "") or "").strip()
            or (getattr(settings, "BEDROCK_REGION", "") or "").strip()
            or "us-east-1"
        ),
    }
    logger.info(
        "PDF table engine: %s "
        "(docling=%s, textract_fallback=%s, tabular_only=%s, region=%s)",
        engine,
        engine_info["docling_enabled"],
        engine_info["textract_fallback_enabled"],
        engine_info["tabular_only"],
        engine_info["region"],
    )

    doc = fitz.open(str(path))
    page_sizes: Dict[int, Tuple[float, float]] = {
        i + 1: (float(p.rect.width), float(p.rect.height)) for i, p in enumerate(doc)
    }
    docling_enabled = bool(getattr(settings, "PDF_DOCLING_TABLES_ENABLED", False))
    textract_enabled = _textract_fallback_allowed(settings)
    docling_tables_all: List[TableData] = []
    docling_error = ""
    # Docling code stays intact; only runs when explicitly re-enabled.
    if docling_enabled:
        try:
            from app.rag.ingestion.pdf_layout.docling_tables import (
                extract_tables_from_pdf,
            )

            docling_tables_all = extract_tables_from_pdf(
                str(path), page_sizes=page_sizes
            )
        except Exception as exc:
            docling_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Docling table extraction failed for %s: %s — "
                "%s",
                path.name,
                exc,
                (
                    "Textract DetectDocumentText available"
                    if textract_enabled
                    else "Docling-only mode (no Textract)"
                ),
            )

    # Textract-primary path: one DetectDocumentText job for the whole PDF
    # (async multipage via S3 when enabled). Docling stays unused.
    textract_tables_by_page: Dict[int, List[TableData]] = {}
    textract_doc_meta: Dict[str, Any] = {}
    textract_doc_error = ""
    textract_primary = bool(textract_enabled and not docling_enabled)
    if textract_primary:
        try:
            from app.rag.ingestion.pdf_layout.textract_tables import (
                extract_tables_from_pdf_detect,
            )

            all_textract, textract_doc_meta = extract_tables_from_pdf_detect(
                str(path), page_sizes=page_sizes, doc=doc
            )
            for table in all_textract:
                textract_tables_by_page.setdefault(int(table.page), []).append(table)
            logger.info(
                "Textract DetectDocumentText primary for %s: pages=%s tables=%s api=%s",
                path.name,
                textract_doc_meta.get("pages_processed"),
                len(all_textract),
                textract_doc_meta.get("api"),
            )
        except Exception as exc:
            textract_doc_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Textract DetectDocumentText primary failed for %s: %s",
                path.name,
                exc,
            )
        try:
            from app.rag.ingestion.pdf_layout.textract_tables import (
                get_textract_api_usage,
            )

            engine_info["textract_doc"] = dict(textract_doc_meta or {})
            if textract_doc_error:
                engine_info["textract_doc_error"] = textract_doc_error
            engine_info["textract_api_usage"] = get_textract_api_usage()
        except Exception:
            pass

    pages: List[PageExtraction] = []
    figures_by_page: Dict[int, List[DocumentElement]] = {}
    margin_report: Dict[str, Any] = {}
    try:
        for page_index, page in enumerate(doc):
            page_started = time.perf_counter()
            page_number = page_index + 1
            native_text = page.get_text("text") or ""
            coverage = _image_coverage(page)
            classification = classify_page(
                native_text,
                page_area=_page_area(page),
                image_coverage=coverage,
            )
            label = classification["label"]
            strategy = "ocr" if classification["needs_ocr"] else "native"
            if label == "blank":
                strategy = "none"
            page_ext = PageExtraction(
                page_number=page_number,
                classification=label,
                strategy=strategy,
                metrics={
                    **classification["metrics"],
                    "classification_confidence": classification["confidence"],
                    "classification_reasons": classification["reasons"],
                    "page_width": round(float(page.rect.width), 2),
                    "page_height": round(float(page.rect.height), 2),
                },
            )

            try:
                table_source = "none"
                page_ext.metrics["table_engine"] = engine
                if docling_error:
                    page_ext.metrics["docling_error"] = docling_error
                hint_tables: List[TableData] = []
                if strategy == "none":
                    page_ext.status = "empty"
                    page_ext.metrics["note"] = "blank page with no text or image"
                else:
                    # Body text only — never use OCR/native grids as stored tables.
                    if strategy == "ocr":
                        (
                            lines,
                            _ocr_tables,
                            confidence,
                            backend,
                            details,
                        ) = _ocr_page_content(page, detect_tables=False)
                        _apply_ocr_result(
                            page_ext,
                            page,
                            lines=lines,
                            confidence=confidence,
                            backend=backend,
                            details=details,
                            image_coverage=coverage,
                        )
                    else:
                        hint_tables = _native_tables(page)
                        lines = _native_lines(
                            page, [table.bbox for table in hint_tables]
                        )
                        page_ext.lines = lines
                        if not lines and not hint_tables and coverage >= 0.15:
                            logger.warning(
                                "Page %s has no native lines despite a text layer; "
                                "retrying with OCR",
                                page_number,
                            )
                            page_ext.strategy = "ocr"
                            strategy = "ocr"
                            (
                                lines,
                                _ocr_tables,
                                confidence,
                                backend,
                                details,
                            ) = _ocr_page_content(page, detect_tables=False)
                            _apply_ocr_result(
                                page_ext,
                                page,
                                lines=lines,
                                confidence=confidence,
                                backend=backend,
                                details=details,
                                image_coverage=coverage,
                            )

                    # Tables: Textract DetectDocumentText primary (default) or
                    # Docling primary with Textract fallback when Docling is on.
                    from app.rag.ingestion.pdf_layout.extraction_quality import (
                        filter_quality_tables,
                    )
                    from app.rag.ingestion.pdf_layout.textract_tables import (
                        extract_tables_from_page,
                        usable_textract_tables,
                    )

                    tabular_only = bool(
                        getattr(settings, "PDF_TEXTRACT_TABULAR_ONLY", True)
                    )
                    looks_tabular, tabular_reason = _page_looks_tabular(
                        page,
                        fallback_tables=hint_tables,
                        page_text=native_text,
                    )
                    page_ext.metrics["tabular_candidate"] = looks_tabular
                    page_ext.metrics["tabular_reason"] = tabular_reason
                    if textract_doc_meta:
                        page_ext.metrics["textract_doc_api"] = textract_doc_meta.get(
                            "api"
                        )
                    if textract_doc_error:
                        page_ext.metrics["textract_doc_error"] = textract_doc_error

                    page_docling = [
                        t for t in docling_tables_all if int(t.page) == int(page_number)
                    ]
                    accepted, rejected, quality_reports = filter_quality_tables(
                        page_docling
                    )
                    page_ext.metrics["docling_tables_raw"] = len(page_docling)
                    page_ext.metrics["docling_tables"] = len(accepted)
                    page_ext.metrics["docling_rejected"] = len(rejected)
                    page_ext.metrics["table_quality"] = quality_reports
                    page_ext.metrics["docling_enabled"] = bool(docling_enabled)

                    final_tables: List[TableData] = list(accepted)
                    table_source = "docling" if accepted else "none"

                    def _apply_textract_page_tables(
                        raw_tables: List[TableData],
                        *,
                        source_label: str,
                    ) -> None:
                        nonlocal final_tables, table_source
                        usable = usable_textract_tables(raw_tables)
                        fb_accepted, _fb_rej, fb_reports = filter_quality_tables(usable)
                        page_ext.metrics["textract_quality"] = fb_reports
                        page_ext.metrics["textract_tables"] = len(usable)
                        if fb_accepted:
                            final_tables = list(fb_accepted)
                            table_source = source_label
                            return
                        if usable:
                            usable_uncertain = []
                            for t in usable:
                                reasons = list(
                                    ((t.provenance or {}).get("quality") or {}).get(
                                        "reasons"
                                    )
                                    or []
                                )
                                if any(
                                    r in reasons
                                    for r in ("too_few_rows", "too_few_columns")
                                ):
                                    continue
                                usable_uncertain.append(t)
                            if usable_uncertain:
                                final_tables = list(usable_uncertain)
                                table_source = source_label
                            else:
                                page_ext.metrics["textract_rejected_empty"] = True

                    textract_tables: List[TableData] = []
                    textract_error = ""
                    textract_started = time.perf_counter()
                    run_textract = False
                    need_fallback = False

                    if textract_primary:
                        # Active path: Docling disabled; DetectDocumentText is primary.
                        page_ext.metrics["textract_primary"] = True
                        skip_non_tabular = bool(
                            tabular_only and not looks_tabular
                        )
                        if skip_non_tabular:
                            page_ext.metrics["textract_skipped"] = "non_tabular_page"
                            run_textract = False
                        else:
                            run_textract = True
                            try:
                                precomputed = list(
                                    textract_tables_by_page.get(page_number) or []
                                )
                                if precomputed or not textract_doc_error:
                                    textract_tables = precomputed
                                    _apply_textract_page_tables(
                                        textract_tables,
                                        source_label="textract_detect",
                                    )
                                else:
                                    # Document-level async failed — sync page retry.
                                    textract_tables = extract_tables_from_page(
                                        page, page_number
                                    )
                                    _apply_textract_page_tables(
                                        textract_tables,
                                        source_label="textract_detect",
                                    )
                            except Exception as exc:
                                textract_error = f"{type(exc).__name__}: {exc}"
                                logger.error(
                                    "Textract DetectDocumentText failed on page %s of %s: %s",
                                    page_number,
                                    path.name,
                                    exc,
                                )
                                page_ext.warnings.append(f"textract: {textract_error}")
                        need_fallback = bool(run_textract and not final_tables)
                    else:
                        # Legacy hybrid: Docling primary → DetectDocumentText fallback.
                        need_fallback = (not accepted) and (
                            looks_tabular
                            or not tabular_only
                            or bool(rejected)
                            or bool(docling_error)
                        )
                        if (
                            tabular_only
                            and not looks_tabular
                            and not rejected
                            and not docling_error
                        ):
                            need_fallback = False
                            page_ext.metrics["textract_skipped"] = "non_tabular_page"
                        if need_fallback:
                            page_ext.metrics["textract_fallback_reason"] = (
                                "docling_error"
                                if docling_error
                                else "rejected"
                                if rejected
                                else "empty"
                                if not page_docling
                                else "quality_gate"
                            )

                        run_textract = bool(need_fallback and textract_enabled)
                        if need_fallback and not textract_enabled:
                            page_ext.metrics["textract_skipped"] = "disabled"
                            page_ext.metrics["docling_only_mode"] = True
                            fail_reason = page_ext.metrics.get(
                                "textract_fallback_reason"
                            ) or (
                                "docling_error"
                                if docling_error
                                else "rejected"
                                if rejected
                                else "empty"
                            )
                            page_ext.metrics[
                                "docling_extraction_failed_reason"
                            ] = fail_reason
                            if rejected:
                                page_ext.metrics["docling_reject_reasons"] = [
                                    list(
                                        ((t.provenance or {}).get("quality") or {}).get(
                                            "reasons"
                                        )
                                        or []
                                    )
                                    for t in rejected
                                ]
                            page_ext.warnings.append(
                                f"docling_only: extraction failed ({fail_reason}); "
                                "Textract fallback disabled"
                            )
                        if run_textract:
                            try:
                                textract_tables = extract_tables_from_page(
                                    page, page_number
                                )
                                _apply_textract_page_tables(
                                    textract_tables,
                                    source_label="textract_fallback",
                                )
                            except Exception as exc:
                                textract_error = f"{type(exc).__name__}: {exc}"
                                logger.error(
                                    "Textract DetectDocumentText fallback failed on page %s of %s: %s",
                                    page_number,
                                    path.name,
                                    exc,
                                )
                                page_ext.warnings.append(f"textract: {textract_error}")

                    page_ext.metrics["textract_elapsed_s"] = round(
                        time.perf_counter() - textract_started, 3
                    )
                    page_ext.metrics["textract_invoked"] = bool(run_textract)
                    page_ext.metrics.setdefault(
                        "textract_tables", len(textract_tables)
                    )
                    if textract_error:
                        page_ext.metrics["textract_error"] = textract_error
                    if run_textract and not textract_tables and not textract_error:
                        page_ext.metrics["textract_empty"] = True

                    # Explicit failure (never silently invent a grid).
                    if need_fallback and not final_tables:
                        page_ext.metrics["table_extraction_failed"] = True
                        if textract_primary:
                            page_ext.warnings.append(
                                "table_extraction_failed: DetectDocumentText "
                                "produced no usable text-derived table"
                            )
                        elif not textract_enabled:
                            page_ext.metrics.setdefault(
                                "docling_only_mode", True
                            )
                        else:
                            page_ext.warnings.append(
                                "table_extraction_failed: Docling quality gate and "
                                "DetectDocumentText fallback produced no usable table"
                            )
                    elif (
                        table_source in {"textract_fallback", "textract_detect"}
                        and final_tables
                        and any(
                            (t.provenance or {}).get("quality", {}).get("ok") is False
                            for t in final_tables
                        )
                    ):
                        page_ext.metrics["table_extraction_uncertain"] = True
                        for t in final_tables:
                            prov = dict(t.provenance or {})
                            prov["extraction_uncertain"] = True
                            t.provenance = prov

                    page_ext.tables = list(final_tables)

                    # Exclude table regions from body lines when present.
                    if page_ext.tables and page_ext.lines:
                        table_boxes = [
                            table.bbox
                            for table in page_ext.tables
                            if table.bbox and table.bbox.area > 0
                        ]
                        if table_boxes:
                            page_ext.lines = [
                                line
                                for line in page_ext.lines
                                if not any(
                                    box.intersection_area(line.bbox)
                                    > 0.5 * line.bbox.area
                                    for box in table_boxes
                                    if line.bbox
                                )
                            ]

                page_ext.metrics["table_source"] = table_source

                if page_ext.tables:
                    page_ext.tables = _repair_page_tables(
                        page_ext.tables, filepath=str(path)
                    )

                page_ext.metrics["gutter"] = detect_column_gutter(
                    page_ext.lines, float(page.rect.width), float(page.rect.height)
                )
                page_ext.metrics["extracted_chars"] = sum(len(line.text) for line in page_ext.lines)
                page_ext.metrics["table_count"] = len(page_ext.tables)
                figures_by_page[page_number] = _figure_elements(
                    page,
                    page_ext.lines,
                    page_ext.tables,
                    image_coverage=coverage,
                    page_height=float(page.rect.height),
                )
            except OcrUnavailableError as exc:
                page_ext.status = "failed"
                page_ext.error = str(exc)
                logger.error("OCR unavailable for page %s of %s: %s", page_number, path.name, exc)
            except OcrError as exc:
                page_ext.status = "failed"
                page_ext.error = str(exc)
                logger.error("OCR failed for page %s of %s: %s", page_number, path.name, exc)
            except Exception as exc:
                page_ext.status = "failed"
                page_ext.error = f"{type(exc).__name__}: {exc}"
                logger.exception("Layout extraction failed for page %s of %s", page_number, path.name)
            finally:
                page_ext.elapsed_s = time.perf_counter() - page_started
                pages.append(page_ext)
    finally:
        doc.close()

    # Running headers/footers are removed only when they recur confidently.
    page_height = max((p.metrics.get("page_height", 0) for p in pages), default=0)
    cleaned, margin_report = detect_repeating_margins(
        {page.page_number: page.lines for page in pages}, page_height or 792.0
    )
    for page in pages:
        page.lines = cleaned.get(page.page_number, page.lines)

    # Multipage table stitch (after Textract + header reconstruction/repair).
    try:
        from app.rag.ingestion.pdf_layout.table_normalize import stitch_multipage_tables

        stitch_multipage_tables(pages)
    except Exception as exc:
        logger.warning("Multipage table stitch skipped: %s", exc)

    body_size = body_font_size(pages)
    state = StructureState()
    elements: List[DocumentElement] = []
    for page in pages:
        page.elements = build_page_elements(
            page,
            state=state,
            body_size=body_size,
            page_width=float(page.metrics.get("page_width") or 612.0),
            page_height=float(page.metrics.get("page_height") or 792.0),
            figures=figures_by_page.get(page.page_number),
        )
        elements.extend(page.elements)

    chunks = chunk_elements(
        elements,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        document_id=document_id,
        document_name=document_name or path.name,
    )
    quality = chunk_quality_report(chunks, len(pages))
    classification = classify_document([page.to_report() for page in pages])
    elapsed = time.perf_counter() - started
    report: Dict[str, Any] = {
        "filepath": path.name,
        "document_id": document_id or "",
        "document_name": document_name or path.name,
        "page_count": len(pages),
        "document_type": classification["document_type"],
        "classification": classification,
        "page_counts": classification["page_counts"],
        "native_pages": classification["native_pages"],
        "ocr_pages": classification["ocr_pages"],
        "failed_pages": classification["failed_pages"],
        "ocr": ocr_status(),
        "table_engine": engine,
        "table_extraction": _table_extraction_report(
            pages, engine=engine, engine_info=engine_info
        ),
        "margins": margin_report,
        "layout": {
            "body_font_size": round(body_size, 2),
            "gutter_pages": [page.page_number for page in pages if page.metrics.get("gutter")],
            "headings": sum(1 for element in elements if element.element_type == "heading"),
            "tables": sum(1 for element in elements if element.element_type == "table"),
            "figures": sum(1 for element in elements if element.element_type == "figure"),
            "lists": sum(1 for element in elements if element.element_type == "list"),
            "captions": sum(1 for element in elements if element.element_type == "caption"),
        },
        "chunks": quality,
        "pages": [page.to_report() for page in pages],
        "elapsed_s": round(elapsed, 2),
    }
    parsed = ParsedDocument(
        filepath=str(path),
        page_count=len(pages),
        pages=pages,
        elements=elements,
        chunks=chunks,
        report=report,
        elapsed_s=elapsed,
    )
    if settings.PDF_LAYOUT_WRITE_DIAGNOSTICS:
        report["diagnostics"] = write_diagnostics(parsed)
    return parsed


def chunk_pdf(
    filepath: str,
    *,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper used by the chunker integration."""
    return parse_pdf(
        filepath,
        document_id=document_id,
        document_name=document_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).chunks


def write_diagnostics(parsed: ParsedDocument) -> Dict[str, str]:
    settings = get_settings()
    directory = Path(settings.PDF_LAYOUT_REPORT_DIR)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Could not create layout diagnostics directory %s: %s", directory, exc)
        return {}
    prefix = parsed.report.get("document_id") or Path(parsed.filepath).stem
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix))[:80]
    digest = hashlib.sha256(parsed.filepath.encode("utf-8")).hexdigest()[:8]
    prefix = f"{prefix}_{digest}"
    payloads: Dict[str, Any] = {
        "report": parsed.report,
        "classification": {
            "filepath": Path(parsed.filepath).name,
            "document_type": parsed.report.get("document_type"),
            "page_counts": parsed.report.get("page_counts"),
            "pages": [
                {
                    "page": page.page_number,
                    "classification": page.classification,
                    "strategy": page.strategy,
                    "status": page.status,
                    "error": page.error,
                    "metrics": page.metrics,
                }
                for page in parsed.pages
            ],
        },
        "layout": {
            "filepath": Path(parsed.filepath).name,
            "page_count": parsed.page_count,
            "elements": [element.to_json() for element in parsed.elements],
        },
        "chunks": parsed.chunks,
        "quality": parsed.report.get("chunks", {}),
    }
    written: Dict[str, str] = {}
    for name, payload in payloads.items():
        target = directory / f"{prefix}_{name}.json"
        try:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written[name] = str(target)
        except Exception as exc:
            logger.warning("Could not write layout diagnostics %s: %s", target, exc)
    return written

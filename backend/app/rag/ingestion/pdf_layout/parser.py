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
    from app.rag.ingestion.pdf_layout.table_extraction import _textract_allowed

    if settings is None:
        settings = get_settings()
    return _textract_allowed(settings)


def _ste_tables_enabled(settings: Any = None) -> bool:
    """True when STE Textract AnalyzeDocument+Bedrock table extraction is enabled."""
    from app.rag.ingestion.pdf_layout.table_extraction import _ste_enabled

    if settings is None:
        settings = get_settings()
    return _ste_enabled(settings)


def _table_engine() -> str:
    """Active table engine label for diagnostics."""
    from app.rag.ingestion.pdf_layout.table_extraction import resolve_table_engine

    return resolve_table_engine(get_settings())


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
    r"schedule\s+table|matrix|grid|"
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
                or page.metrics.get("ste_error")
                or ""
            ),
        }
        for page in pages
        if page.metrics.get("textract_error")
        or page.metrics.get("docling_error")
        or page.metrics.get("ste_error")
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

    page_elapsed = sum(
        float(page.metrics.get("textract_elapsed_s") or 0)
        + float(page.metrics.get("docling_elapsed_s") or 0)
        for page in pages
    )
    info = dict(engine_info or {})
    plan_report = dict(info.get("table_plan") or {})
    # Prefer document-plan wall time (includes STE + Detect) over page sums.
    plan_elapsed = plan_report.get("elapsed_s")
    ste_info = dict(info.get("ste") or {})
    return {
        "engine": engine,
        "engine_info": info,
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
        "elapsed_s": round(
            float(plan_elapsed if plan_elapsed is not None else page_elapsed), 3
        ),
        "timings": {
            "plan_elapsed_s": plan_report.get("elapsed_s"),
            "ste_elapsed_s": plan_report.get("ste_elapsed_s")
            or ste_info.get("elapsed_s"),
            "detect_elapsed_s": plan_report.get("detect_elapsed_s"),
            "detect_skipped": plan_report.get("detect_skipped"),
            "ste_detail": ste_info.get("timings"),
            "page_metric_sum_s": round(page_elapsed, 3),
        },
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
    progress: Optional[Any] = None,
) -> ParsedDocument:
    """Parse a PDF into structured elements and retrieval-ready chunks.

    ``progress`` is an optional ``(stage: str, pct: int) -> None`` callback used
    for long STE Textract+Bedrock runs.
    """
    settings = get_settings()
    started = time.perf_counter()
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    from app.rag.ingestion.pdf_layout.table_extraction import (
        run_document_table_extraction,
        resolve_page_tables,
    )

    doc = fitz.open(str(path))
    page_sizes: Dict[int, Tuple[float, float]] = {
        i + 1: (float(p.rect.width), float(p.rect.height)) for i, p in enumerate(doc)
    }

    # Unified table stage: STE AnalyzeDocument + DetectDocumentText → one plan.
    table_plan = run_document_table_extraction(
        str(path),
        page_sizes=page_sizes,
        doc=doc,
        document_id=document_id or "",
        document_name=document_name or path.name,
        settings=settings,
        progress=progress,
    )
    engine = table_plan.engine
    engine_info = dict(table_plan.engine_info or {})
    engine_info["table_plan"] = table_plan.to_report()
    ste_on = table_plan.ste_enabled
    textract_enabled = table_plan.textract_enabled
    textract_primary = bool(textract_enabled)
    textract_words_by_page = table_plan.textract_words_by_page
    textract_doc_meta = dict(table_plan.textract_meta or {})
    textract_doc_error = table_plan.textract_error
    docling_error = ""

    logger.info(
        "PDF table engine: %s "
        "(ste=%s, textract=%s, tabular_only=%s, region=%s)",
        engine,
        ste_on,
        textract_enabled,
        engine_info.get("tabular_only"),
        engine_info.get("region"),
    )

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
                    # Body text: prefer DetectDocumentText WORDs when Textract
                    # primary is on. Do not silently swap table extractors.
                    hint_tables = _native_tables(page)
                    textract_words = list(
                        textract_words_by_page.get(page_number) or []
                    )
                    if textract_primary and textract_words:
                        from app.rag.ingestion.pdf_layout.analyzer import (
                            words_to_lines,
                        )

                        table_boxes = [t.bbox for t in hint_tables]
                        # Exclude words that sit inside provisional table regions;
                        # final Textract tables replace these later.
                        kept = []
                        for word in textract_words:
                            if any(
                                box.intersection_area(word.bbox) > 0.5 * word.bbox.area
                                for box in table_boxes
                            ):
                                continue
                            kept.append(word)
                        lines = words_to_lines(kept, float(page.rect.width))
                        for line in lines:
                            line.source = "textract_detect"
                        page_ext.lines = lines
                        page_ext.strategy = "textract_detect"
                        page_ext.metrics["text_engine"] = "textract_detect"
                        page_ext.metrics["textract_word_count"] = len(textract_words)
                    elif strategy == "ocr":
                        if textract_primary:
                            # Textract-only mode: do not fall back to local OCR.
                            page_ext.lines = []
                            page_ext.warnings.append(
                                "textract_text_missing: OCR fallback disabled"
                            )
                            page_ext.metrics["text_engine"] = "none"
                        else:
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
                        lines = _native_lines(
                            page, [table.bbox for table in hint_tables]
                        )
                        page_ext.lines = lines
                        page_ext.metrics["text_engine"] = "native"
                        if (
                            not textract_primary
                            and not lines
                            and not hint_tables
                            and coverage >= 0.15
                        ):
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

                    # Unified per-page table resolve from the document plan.
                    final_tables, table_source, table_metrics, table_warnings = (
                        resolve_page_tables(
                            page_number,
                            table_plan,
                            page=page,
                            hint_tables=hint_tables,
                            native_text=native_text,
                            settings=settings,
                            page_looks_tabular_fn=_page_looks_tabular,
                        )
                    )
                    page_ext.metrics.update(table_metrics)
                    page_ext.warnings.extend(table_warnings)
                    page_ext.tables = list(final_tables)

                    # Rebuild body lines from DetectDocumentText words once final
                    # table boxes are known (preferred Textract-only text path).
                    textract_words = list(
                        textract_words_by_page.get(page_number) or []
                    )
                    if textract_primary and textract_words:
                        from app.rag.ingestion.pdf_layout.analyzer import (
                            words_to_lines,
                        )

                        table_boxes = [
                            table.bbox
                            for table in page_ext.tables
                            if table.bbox and table.bbox.area > 0
                        ]
                        kept = []
                        for word in textract_words:
                            if any(
                                box.intersection_area(word.bbox) > 0.5 * word.bbox.area
                                for box in table_boxes
                            ):
                                continue
                            kept.append(word)
                        lines = words_to_lines(kept, float(page.rect.width))
                        for line in lines:
                            line.source = "textract_detect"
                        page_ext.lines = lines
                        page_ext.strategy = "textract_detect"
                        page_ext.metrics["text_engine"] = "textract_detect"
                    elif page_ext.tables and page_ext.lines:
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
                    # Do not rewrite STE Bedrock-reconstructed cell values.
                    ste_kept = [
                        t
                        for t in page_ext.tables
                        if (t.extraction_engine or "").startswith("ste_")
                    ]
                    other_tables = [
                        t
                        for t in page_ext.tables
                        if not (t.extraction_engine or "").startswith("ste_")
                    ]
                    repaired_others = _repair_page_tables(
                        other_tables, filepath=str(path)
                    )
                    if ste_kept:
                        try:
                            from app.rag.ingestion.pdf_layout.table_normalize import (
                                linearize_table_facts,
                            )

                            for t in ste_kept:
                                if not t.linearized_facts:
                                    t.linearized_facts = linearize_table_facts(
                                        t.headers,
                                        t.rows,
                                        caption=t.caption or "",
                                    )
                        except Exception:
                            pass
                    page_ext.tables = list(ste_kept) + list(repaired_others)

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

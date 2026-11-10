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
) -> Tuple[List[TextLine], List[TableData], float, str, Dict[str, Any]]:
    settings = get_settings()
    result = ocr_page(page)
    words = result["words"]
    backend = result["backend"]
    confidence = result["mean_confidence"]
    if not words:
        raise OcrError("OCR returned no positioned words")

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    tables: List[TableData] = []
    consumed_boxes: List[BBox] = []
    used_words = list(words)

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

    # Borderless ledger tables: detect aligned short numeric rows, then remove
    # their lines from normal paragraph flow.
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
        "ruled_tables": sum(1 for table in tables if table.extraction_method == "ocr_ruled"),
        "alignment_tables": sum(1 for table in tables if table.extraction_method == "ocr_alignment"),
    }
    return lines, tables, confidence, backend, details


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

    doc = fitz.open(str(path))
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
                if strategy == "ocr":
                    lines, tables, confidence, backend, details = _ocr_page_content(page)
                    page_ext.lines = lines
                    page_ext.tables = tables
                    page_ext.ocr_confidence = confidence
                    page_ext.ocr_backend = backend
                    page_ext.metrics.update(details)
                    page_ext.metrics["ocr_chars"] = sum(len(line.text) for line in lines)
                    page_ext.metrics["ocr_text_metrics"] = compute_text_metrics(
                        "\n".join(line.text for line in lines),
                        page_area=_page_area(page),
                        image_coverage=coverage,
                    )
                elif strategy == "native":
                    native_tables = _native_tables(page)
                    lines = _native_lines(page, [table.bbox for table in native_tables])
                    page_ext.lines = lines
                    page_ext.tables = native_tables
                    if not lines and not native_tables and coverage >= 0.15:
                        logger.warning(
                            "Page %s has no native lines despite a text layer; retrying with OCR",
                            page_number,
                        )
                        page_ext.strategy = "ocr"
                        lines, tables, confidence, backend, details = _ocr_page_content(page)
                        page_ext.lines = lines
                        page_ext.tables = tables
                        page_ext.ocr_confidence = confidence
                        page_ext.ocr_backend = backend
                        page_ext.metrics.update(details)
                else:
                    page_ext.status = "empty"
                    page_ext.metrics["note"] = "blank page with no text or image"

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

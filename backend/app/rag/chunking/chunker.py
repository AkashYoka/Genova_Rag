"""
Smart document chunking using LangChain's RecursiveCharacterTextSplitter.
Supports PDF, DOCX, TXT, and Markdown files with page-level metadata.
"""
import json
import os
import re
import pymupdf as fitz
import docx
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def _is_word_inside_bbox(word: Dict[str, Any], bbox: tuple) -> bool:
    """Return True when the word center falls inside a pdfplumber bbox."""
    x0, top, x1, bottom = bbox
    word_x = (float(word["x0"]) + float(word["x1"])) / 2
    word_y = (float(word["top"]) + float(word["bottom"])) / 2
    return x0 <= word_x <= x1 and top <= word_y <= bottom


def _words_to_text(words: List[Dict[str, Any]], line_tolerance: float = 3.0) -> str:
    """Rebuild readable text from positioned pdfplumber words, preserving multi-column reading order."""
    if not words:
        return ""

    # 1. Group words into horizontal lines based on vertical proximity
    sorted_words = sorted(words, key=lambda item: (float(item["top"]), item["x0"]))
    lines: List[List[Dict[str, Any]]] = []

    for word in sorted_words:
        if not lines:
            lines.append([word])
            continue

        current_top = sum(float(item["top"]) for item in lines[-1]) / len(lines[-1])
        if abs(float(word["top"]) - current_top) <= line_tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])

    # 2. Split each line into segments based on horizontal gaps (column gutters)
    GAP_THRESHOLD = 20.0
    line_segments_list = []
    for line in lines:
        sorted_line_words = sorted(line, key=lambda item: item["x0"])
        segments_in_line = []
        current_seg = [sorted_line_words[0]]

        for word in sorted_line_words[1:]:
            prev_word = current_seg[-1]
            gap = float(word["x0"]) - float(prev_word["x1"])
            if gap > GAP_THRESHOLD:
                segments_in_line.append(current_seg)
                current_seg = [word]
            else:
                current_seg.append(word)
        segments_in_line.append(current_seg)
        line_segments_list.append(segments_in_line)

    # 3. Detect global vertical gutters from lines with multiple layout segments
    gutter_intervals = []
    for seg_list in line_segments_list:
        if len(seg_list) > 1:
            for i in range(len(seg_list) - 1):
                x1_prev = max(float(w["x1"]) for w in seg_list[i])
                x0_next = min(float(w["x0"]) for w in seg_list[i+1])
                if x0_next > x1_prev:
                    gutter_intervals.append((x1_prev, x0_next))

    significant_gutter_centers = []
    if gutter_intervals:
        sorted_gutters = sorted(gutter_intervals, key=lambda g: g[0])
        current_gutter_groups = []
        for g in sorted_gutters:
            inserted = False
            for group in current_gutter_groups:
                group_x0 = max(item[0] for item in group)
                group_x1 = min(item[1] for item in group)
                if max(g[0], group_x0) < min(g[1], group_x1):
                    group.append(g)
                    inserted = True
                    break
            if not inserted:
                current_gutter_groups.append([g])

        num_multi_seg_lines = sum(1 for seg_list in line_segments_list if len(seg_list) > 1)
        min_group_size = max(2, int(num_multi_seg_lines * 0.15))

        for group in current_gutter_groups:
            if len(group) >= min_group_size:
                centers = [(g[0] + g[1]) / 2.0 for g in group]
                significant_gutter_centers.append(sum(centers) / len(centers))
        significant_gutter_centers.sort()

    # 4. Flatten segments and capture absolute structural bounds
    segment_objs = []
    for seg_list in line_segments_list:
        for seg in seg_list:
            seg_x0 = min(float(w["x0"]) for w in seg)
            seg_x1 = max(float(w["x1"]) for w in seg)
            seg_top = min(float(w["top"]) for w in seg)
            seg_bottom = max(float(w["bottom"]) for w in seg)
            seg_text = " ".join(w["text"] for w in seg)
            segment_objs.append({
                "x0": seg_x0,
                "x1": seg_x1,
                "top": seg_top,
                "bottom": seg_bottom,
                "text": seg_text
            })

    # 5. Sort segments vertically into horizontal layout bands divided by full-width text structures
    segment_objs.sort(key=lambda s: s["top"])

    final_segments = []
    current_band = []
    GUTTER_TOLERANCE = 3.0

    for seg in segment_objs:
        crosses_gutter = False
        for center in significant_gutter_centers:
            if (seg["x0"] + GUTTER_TOLERANCE) < center < (seg["x1"] - GUTTER_TOLERANCE):
                crosses_gutter = True
                break

        if crosses_gutter:
            if current_band:
                def get_col_idx(s):
                    idx = 0
                    s_mid = (s["x0"] + s["x1"]) / 2.0
                    for c in significant_gutter_centers:
                        if s_mid > c:
                            idx += 1
                    return idx
                current_band.sort(key=lambda s: (get_col_idx(s), s["top"]))
                final_segments.extend(current_band)
                current_band = []
            final_segments.append(seg)
        else:
            current_band.append(seg)

    if current_band:
        def get_col_idx(s):
            idx = 0
            s_mid = (s["x0"] + s["x1"]) / 2.0
            for c in significant_gutter_centers:
                if s_mid > c:
                    idx += 1
            return idx
        current_band.sort(key=lambda s: (get_col_idx(s), s["top"]))
        final_segments.extend(current_band)

    text_lines = [seg["text"] for seg in final_segments if seg["text"].strip()]
    return "\n".join(text_lines)


def _clean_table_cell(cell: Any) -> str:
    """Normalize extracted table cell text for Markdown serialization."""
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip().replace("|", "\\|")


def _normalize_table_rows(rows: List[List[Any]]) -> List[List[str]]:
    """Pad ragged rows and clean cells while preserving empty merged-cell slots."""
    cleaned = [
        [_clean_table_cell(cell) for cell in (row or [])]
        for row in (rows or [])
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return []
    width = max(len(row) for row in cleaned)
    return [row + [""] * (width - len(row)) for row in cleaned]


def _table_to_markdown(rows: List[List[Any]]) -> str:
    """Serialize extracted table rows into Markdown for retrieval."""
    normalized = _normalize_table_rows(rows)
    if not normalized:
        return ""

    def fmt(row: List[str]) -> str:
        return "| " + " | ".join(row) + " |"

    header = normalized[0]
    separator = ["---"] * len(header)
    body = normalized[1:]
    return "\n".join([fmt(header), fmt(separator), *[fmt(row) for row in body]])


def _table_to_structured_json(
    rows: List[List[Any]],
    *,
    page: int,
    extractor: str,
    flavor: str | None = None,
    bbox: List[float] | None = None,
    footnotes: List[str] | None = None,
    accuracy: float | None = None,
) -> Dict[str, Any]:
    """Build retrieval-friendly JSON for a table (headers, rows, metadata)."""
    normalized = _normalize_table_rows(rows)
    headers = normalized[0] if normalized else []
    body = normalized[1:] if len(normalized) > 1 else []
    return {
        "page": page,
        "extractor": extractor,
        "flavor": flavor,
        "headers": headers,
        "rows": body,
        "n_rows": len(normalized),
        "n_cols": len(headers),
        "bbox": bbox,
        "footnotes": footnotes or [],
        "accuracy": accuracy,
        "markdown": _table_to_markdown(normalized),
    }


def _table_fingerprint(rows: List[List[Any]], page: int) -> str:
    """Stable content fingerprint used to dedupe tables across extractors."""
    normalized = _normalize_table_rows(rows)
    blob = f"{page}|" + "||".join("|".join(row).lower() for row in normalized)
    blob = re.sub(r"\s+", " ", blob).strip()
    return blob


def _normalize_bbox(
    bbox: Any,
    page_width: float | None = None,
    page_height: float | None = None,
    *,
    pdf_origin: bool = False,
) -> List[float] | None:
    """Return a normalized [x0, top, x1, bottom] bbox in 0–1 page coordinates.

    When ``pdf_origin`` is True the input is treated as PDF user-space
    (origin bottom-left) and converted to top-left before normalizing, matching
    pdfplumber / on-page highlight coordinates.
    """
    if bbox is None:
        return None
    try:
        if isinstance(bbox, dict):
            vals = [
                bbox.get("x0", bbox.get("left")),
                bbox.get("y0", bbox.get("top")),
                bbox.get("x1", bbox.get("right")),
                bbox.get("y1", bbox.get("bottom")),
            ]
        else:
            vals = list(bbox)[:4]
        nums = [float(v) for v in vals]
    except Exception:
        return None

    if page_width and page_height and page_width > 0 and page_height > 0 and max(nums) > 1.5:
        x0, y0, x1, y1 = nums
        if pdf_origin:
            top = page_height - max(y0, y1)
            bottom = page_height - min(y0, y1)
            return [
                round(min(x0, x1) / page_width, 4),
                round(top / page_height, 4),
                round(max(x0, x1) / page_width, 4),
                round(bottom / page_height, 4),
            ]
        return [
            round(x0 / page_width, 4),
            round(y0 / page_height, 4),
            round(x1 / page_width, 4),
            round(y1 / page_height, 4),
        ]
    return [round(v, 4) for v in nums]


def _bbox_iou(a: List[float] | None, b: List[float] | None) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _footnote_candidates_near_bbox(
    words: List[Dict[str, Any]],
    bbox: tuple,
    *,
    max_gap: float = 36.0,
) -> List[str]:
    """Collect short footnote-like lines sitting just below a table bbox."""
    if not words or not bbox:
        return []
    _x0, _top, _x1, bottom = bbox
    below = [
        w for w in words
        if float(w.get("top", 0)) >= float(bottom) - 1
        and float(w.get("top", 0)) <= float(bottom) + max_gap
    ]
    if not below:
        return []
    text = _words_to_text(below)
    lines = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if re.match(r"^(\*+|†+|‡+|§+|\d+[\).]|Note:|NOTES?:)", cleaned, flags=re.I):
            lines.append(cleaned)
        elif len(cleaned) <= 160 and cleaned[:1].islower():
            lines.append(cleaned)
    return lines[:6]


def _extractor_rank(extractor: str, flavor: str | None = None) -> int:
    """Higher rank wins during cross-extractor dedupe."""
    del flavor
    if extractor == "docling":
        return 5
    if extractor == "pdfplumber":
        return 2
    return 0


def _is_plausible_table(
    rows: List[List[str]],
    *,
    extractor: str,
    flavor: str | None = None,
    accuracy: float | None = None,
) -> bool:
    """Drop obvious false positives (e.g. multi-column prose mistaken as tables)."""
    del flavor, accuracy, extractor
    if not rows or len(rows) < 2:
        return False
    n_cols = max(len(row) for row in rows)
    if n_cols < 1:
        return False
    # Extremely tall narrow blocks are usually sidebars / TOC, not tables.
    if n_cols <= 2 and len(rows) > 25:
        return False
    if len(rows) > 60:
        return False
    return True


def _make_table_record(
    *,
    rows: List[List[Any]],
    page: int,
    extractor: str,
    flavor: str | None = None,
    bbox: Any = None,
    page_width: float | None = None,
    page_height: float | None = None,
    footnotes: List[str] | None = None,
    accuracy: float | None = None,
    pdf_origin: bool = False,
) -> Dict[str, Any] | None:
    normalized = _normalize_table_rows(rows)
    if not _is_plausible_table(
        normalized, extractor=extractor, flavor=flavor, accuracy=accuracy
    ):
        return None
    markdown = _table_to_markdown(normalized)
    if not markdown.strip():
        return None
    norm_bbox = _normalize_bbox(
        bbox, page_width, page_height, pdf_origin=pdf_origin
    )
    structured = _table_to_structured_json(
        normalized,
        page=page,
        extractor=extractor,
        flavor=flavor,
        bbox=norm_bbox,
        footnotes=footnotes,
        accuracy=accuracy,
    )
    return {
        "text": markdown,
        "page": page,
        "chunk_type": "table",
        "bbox": json.dumps(norm_bbox) if norm_bbox else "",
        "table_json": json.dumps(structured, ensure_ascii=False),
        "table_extractor": extractor,
        "table_flavor": flavor or "",
        "table_footnotes": json.dumps(footnotes or [], ensure_ascii=False),
        "_fingerprint": _table_fingerprint(normalized, page),
        "_rows": normalized,
        "_bbox": norm_bbox,
        "_rank": _extractor_rank(extractor, flavor),
        "_accuracy": accuracy,
    }


def _dedupe_table_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the richest copy of each table; drop near-duplicates across extractors."""
    survivors: List[Dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item.get("page", 0),
            -int(item.get("_rank", 0)),
            -len(item.get("_rows") or []),
        ),
    ):
        duplicate = False
        for kept in survivors:
            if kept.get("page") != record.get("page"):
                continue
            if kept.get("_fingerprint") == record.get("_fingerprint"):
                duplicate = True
                break
            if _bbox_iou(kept.get("_bbox"), record.get("_bbox")) >= 0.55:
                # Same region: keep higher-ranked extractor already in survivors.
                duplicate = True
                break
        if not duplicate:
            survivors.append(record)

    cleaned: List[Dict[str, Any]] = []
    for idx, record in enumerate(survivors):
        item = {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        item["table_index"] = idx
        cleaned.append(item)
    return cleaned


def _pdf_page_sizes(filepath: str) -> Dict[int, tuple[float, float]]:
    """Return {page_number: (width, height)} for bbox normalization."""
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            return {
                i + 1: (float(page.width), float(page.height))
                for i, page in enumerate(pdf.pages)
            }
    except Exception:
        return {}


def _extract_tables_with_docling(filepath: str) -> List[Dict[str, Any]]:
    """Legacy-chunker Docling path — TableFormer Accurate via docling_tables."""
    from app.rag.ingestion.pdf_layout.docling_tables import extract_tables_from_pdf

    page_sizes = _pdf_page_sizes(filepath)
    records: List[Dict[str, Any]] = []
    for table in extract_tables_from_pdf(filepath, page_sizes=page_sizes):
        rows = (
            [list(table.headers)] + [list(r) for r in table.rows]
            if table.headers
            else [list(r) for r in table.rows]
        )
        width_height = page_sizes.get(int(table.page))
        record = _make_table_record(
            rows=rows,
            page=int(table.page),
            extractor="docling",
            flavor="tableformer_accurate",
            bbox=list(table.bbox.as_tuple()) if table.bbox else None,
            page_width=width_height[0] if width_height else None,
            page_height=width_height[1] if width_height else None,
            accuracy=(float(table.structure_confidence or 0) * 100.0) or None,
            pdf_origin=True,
        )
        if record:
            records.append(record)
    return records


def _extract_tables_with_pdfplumber(filepath: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Coordinate-based pdfplumber extraction + paragraph text outside table bboxes.

    Returns ``(text_page_records, table_records)``.
    """
    import pdfplumber

    text_pages: List[Dict[str, Any]] = []
    table_records: List[Dict[str, Any]] = []

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables() or []
            table_bboxes = [table.bbox for table in tables]
            words = page.extract_words() or []
            paragraph_words = [
                word for word in words
                if not any(_is_word_inside_bbox(word, bbox) for bbox in table_bboxes)
            ]
            paragraph_text = _words_to_text(paragraph_words)
            if paragraph_text.strip():
                text_pages.append({
                    "text": paragraph_text,
                    "page": page_num,
                    "chunk_type": "text",
                })

            page_w, page_h = float(page.width), float(page.height)
            for table in tables:
                rows = table.extract() or []
                footnotes = _footnote_candidates_near_bbox(words, table.bbox)
                record = _make_table_record(
                    rows=rows,
                    page=page_num,
                    extractor="pdfplumber",
                    flavor="find_tables",
                    bbox=table.bbox,
                    page_width=page_w,
                    page_height=page_h,
                    footnotes=footnotes,
                )
                if record:
                    table_records.append(record)
    return text_pages, table_records


def extract_pdf(filepath: str) -> List[Dict[str, Any]]:
    """Extract PDF text while preserving tables as separate chunks.

    Extraction order:
    1. Docling TableFormer Accurate + pdfplumber text/coords (legacy path only)
    2. PyMuPDF native text
    3. OCR fallback for image-only pages
    """
    pages: List[Dict[str, Any]] = []

    try:
        pages = extract_pdf_with_tables(filepath)
    except Exception as e2:
        logger.warning(f"Table-aware PDF extraction failed, falling back: {e2}")

    if not pages:
        try:
            pages = extract_pdf_with_pymupdf(filepath)
        except Exception as e3:
            logger.warning(f"PyMuPDF extraction failed, falling back to OCR: {e3}")

    # If still no text, run a full OCR pass for scanned/image-only PDFs.
    if not pages:
        logger.info(
            "All text extractors returned empty for '%s' — running full OCR pass",
            filepath,
        )
        try:
            from app.rag.ingestion.ocr import extract_pdf_with_ocr
            pages = extract_pdf_with_ocr(filepath)
        except Exception as exc:
            logger.warning("OCR pass failed for '%s': %s", filepath, exc)

    return pages


def extract_pdf_with_pymupdf(filepath: str) -> List[Dict[str, Any]]:
    """Fallback PDF extraction with page numbers using PyMuPDF.

    For pages with no selectable text (scanned/image-only pages) OCR is
    attempted automatically using the configured OCR backend.
    """
    from app.rag.ingestion.ocr import MIN_TEXT_CHARS, OCR_BACKEND, ocr_page

    doc = fitz.open(filepath)
    pages = []

    try:
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()

            if len(text) >= MIN_TEXT_CHARS:
                pages.append({
                    "text": text,
                    "page": page_num + 1,
                    "chunk_type": "text",
                    "ocr": False,
                })
                continue

            # No selectable text — attempt OCR
            logger.info(
                "Page %d of '%s' is image-only — running OCR (backend=%s)",
                page_num + 1,
                filepath,
                OCR_BACKEND,
            )
            try:
                ocr_text = ocr_page(page)
                if ocr_text:
                    pages.append({
                        "text": ocr_text,
                        "page": page_num + 1,
                        "chunk_type": "text",
                        "ocr": True,
                    })
                else:
                    logger.warning(
                        "OCR returned no text for page %d of '%s'",
                        page_num + 1,
                        filepath,
                    )
            except ImportError:
                logger.warning(
                    "OCR backend '%s' not installed — skipping image-only page %d",
                    OCR_BACKEND,
                    page_num + 1,
                )
            except Exception as exc:
                logger.warning(
                    "OCR failed for page %d of '%s': %s",
                    page_num + 1,
                    filepath,
                    exc,
                )
    finally:
        doc.close()

    return pages


def extract_pdf_with_tables(filepath: str) -> List[Dict[str, Any]]:
    """Extract PDF text + tables for RAG chunking.

    Pipeline (legacy only; layout-aware uses Docling+Textract in parse_pdf):
    1. pdfplumber — paragraph text + coordinate-based tables
    2. Docling TableFormer Accurate — structured tables
    3. Deduplicate overlapping tables (prefer Docling > pdfplumber)

    Each table record includes Markdown text, structured JSON, page, bbox, footnotes,
    and extractor metadata suitable for chunking and retrieval.

    When the layout-aware pipeline is enabled, this legacy path returns PyMuPDF
    body text only: AWS Textract owns table extraction in ``parse_pdf``, so
    running Docling/pdfplumber here would duplicate tables.
    """
    if bool(getattr(get_settings(), "PDF_LAYOUT_AWARE_ENABLED", True)):
        logger.info(
            "Layout-aware PDF enabled: legacy chunker returns body text only "
            "for '%s' (legacy Docling/pdfplumber tables disabled; Docling+Textract owns tables in parse_pdf)",
            filepath,
        )
        return extract_pdf_with_pymupdf(filepath)

    pages: List[Dict[str, Any]] = []
    text_pages: List[Dict[str, Any]] = []
    plumber_tables: List[Dict[str, Any]] = []

    try:
        text_pages, plumber_tables = _extract_tables_with_pdfplumber(filepath)
    except Exception as exc:
        logger.warning("pdfplumber extraction failed for '%s': %s", filepath, exc)

    all_tables: List[Dict[str, Any]] = []

    try:
        docling_tables = _extract_tables_with_docling(filepath)
        all_tables.extend(docling_tables)
        logger.info(
            "Docling extracted %d table candidate(s) from '%s'",
            len(docling_tables),
            filepath,
        )
    except ImportError as exc:
        logger.info("Docling unavailable (%s); continuing without it", exc)
    except Exception as exc:
        logger.warning("Docling table extraction failed: %s", exc)

    validated: List[Dict[str, Any]] = list(all_tables)
    validated.extend(plumber_tables)
    deduped = _dedupe_table_records(validated)
    logger.info(
        "Retained %d unique table(s) after dedupe for '%s'",
        len(deduped),
        filepath,
    )

    pages.extend(text_pages)
    pages.extend(deduped)

    if not pages:
        # Absolute fallback: whole-page pdfplumber text so callers get something.
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({
                            "text": text,
                            "page": page_num,
                            "chunk_type": "text",
                        })
        except Exception as exc:
            logger.warning("pdfplumber full-text fallback failed: %s", exc)

    return pages


def extract_pdf_images(
    doc_or_path: Any,
    min_width: int = 50,
    min_height: int = 50,
    min_size: int = 10240,
) -> Any:
    """Generator to yield extracted images from a PDF page-by-page."""
    if not doc_or_path:
        return

    is_path = isinstance(doc_or_path, str)
    doc = None
    if is_path:
        try:
            doc = fitz.open(doc_or_path)
        except Exception as e:
            logger.warning(f"Could not open PDF with fitz for image extraction: {e}")
            return
    else:
        doc = doc_or_path

    try:
        for page_num, page in enumerate(doc):
            processed_xrefs = set()
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in processed_xrefs:
                    continue
                processed_xrefs.add(xref)

                try:
                    pix = fitz.Pixmap(doc, xref)
                    width, height = pix.width, pix.height
                    
                    if pix.n >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_bytes = pix.tobytes("png")
                    
                    if width < min_width or height < min_height or len(img_bytes) < min_size:
                        del img_bytes
                        del pix
                        continue

                    yield {
                        "image_bytes": img_bytes,
                        "page": page_num + 1,
                        "width": width,
                        "height": height,
                    }
                except Exception:
                    continue
    finally:
        if is_path and doc:
            doc.close()


def extract_docx(filepath: str) -> List[Dict[str, Any]]:
    """Extract text from DOCX files."""
    doc = docx.Document(filepath)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    return [{"text": full_text, "page": 1}] if full_text else []


def extract_txt(filepath: str) -> List[Dict[str, Any]]:
    """Extract text from TXT/Markdown files, preserving tables as separate chunks."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if not text.strip():
        return []

    chunks = []
    current_text_lines = []
    table_lines = []
    in_table = False

    for line in text.splitlines():
        is_table_line = line.strip().startswith("|")

        if is_table_line:
            if not in_table:
                if current_text_lines:
                    chunk_text = "\n".join(current_text_lines).strip()
                    if chunk_text:
                        chunks.append({"text": chunk_text, "page": 1, "chunk_type": "text"})
                    current_text_lines = []
                in_table = True
            table_lines.append(line)
        else:
            if in_table:
                table_text = "\n".join(table_lines).strip()
                if table_text:
                    chunks.append({"text": table_text, "page": 1, "chunk_type": "table"})
                table_lines = []
                in_table = False
            current_text_lines.append(line)

    if in_table and table_lines:
        chunks.append({"text": "\n".join(table_lines).strip(), "page": 1, "chunk_type": "table"})
    elif current_text_lines:
        chunk_text = "\n".join(current_text_lines).strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "page": 1, "chunk_type": "text"})

    return chunks


def chunk_document(
    filepath: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    document_id: str = None,
    document_name: str = None,
    progress=None,
    strategy: str = None,
) -> List[Dict[str, Any]]:
    """Load a document, extract text per page, and split into semantic chunks.

    PDFs on disk first go through the LLM-assisted hierarchical semantic
    pipeline (section analysis over layout blocks, deterministic fallback).
    The layout-aware pipeline remains the fallback, and the legacy path remains
    for older callers, fake paths in tests, and any unexpected failure.

    ``progress`` is an optional ``(stage, percent)`` callback used by Celery
    ingestion; it is only forwarded to the semantic pipeline. ``strategy``
    overrides the global ``CHUNK_STRATEGY`` for a per-document reindex.
    """
    ext = filepath.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        try:
            from app.rag.chunking.semantic.service import (
                semantic_chunk_document,
                should_use_semantic_chunking,
            )

            if should_use_semantic_chunking(filepath, strategy=strategy):
                semantic_chunks = semantic_chunk_document(
                    filepath,
                    document_id=document_id,
                    document_name=document_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    progress=progress,
                    strategy=strategy,
                )
                if semantic_chunks:
                    logger.info(
                        "Semantic chunking produced %d chunks for %s (strategy=%s, version=%s)",
                        len(semantic_chunks),
                        filepath,
                        getattr(semantic_chunks, "strategy", "semantic_llm"),
                        getattr(semantic_chunks, "index_version", ""),
                    )
                    return semantic_chunks
                logger.warning(
                    "Semantic chunking produced no chunks for %s; "
                    "falling back to layout pipeline",
                    filepath,
                )
        except Exception as exc:
            logger.warning(
                "Semantic chunking failed for %s (%s); "
                "falling back to layout pipeline",
                filepath,
                exc,
                exc_info=True,
            )

    if ext == "pdf":
        if getattr(settings, "PDF_LAYOUT_AWARE_ENABLED", True) and os.path.exists(filepath):
            try:
                from app.rag.ingestion.tessdata import apply_tessdata_env
                from app.services.layout_parser import AdvancedPDFParser

                # Ensure Celery workers find Scoop/system tessdata before OCR.
                apply_tessdata_env(getattr(settings, "PDF_OCR_TESSDATA_DIR", "") or None)

                layout_chunks = AdvancedPDFParser(filepath).chunk_document(
                    document_id=document_id,
                    document_name=document_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if layout_chunks:
                    logger.info(
                        "Layout-aware PDF pipeline produced %d chunks for %s",
                        len(layout_chunks),
                        filepath,
                    )
                    return layout_chunks
                logger.warning(
                    "Layout-aware PDF pipeline produced no chunks for %s; "
                    "falling back to legacy extraction",
                    filepath,
                )
            except Exception as exc:
                logger.warning(
                    "Layout-aware PDF pipeline failed for %s (%s); "
                    "falling back to legacy extraction",
                    filepath,
                    exc,
                    exc_info=True,
                )
        pages = extract_pdf(filepath)
    elif ext == "docx":
        pages = extract_docx(filepath)
    elif ext in ("txt", "md"):
        pages = extract_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    pdf_doc = None
    if ext == "pdf":
        try:
            pdf_doc = fitz.open(filepath)
        except Exception as e:
            logger.warning(f"Could not open PDF with fitz: {e}")

    if not pages:
        if not pdf_doc or len(pdf_doc) == 0:
            if pdf_doc:
                pdf_doc.close()
            return []
    
    if not chunk_size:
        chunk_size = settings.CHUNK_SIZE
    if not chunk_overlap:
        chunk_overlap = settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    chunk_index = 0

    try:
        max_page_in_data = max(page_data["page"] for page_data in pages) if pages else 0
        total_pages = max(max_page_in_data, len(pdf_doc) if pdf_doc else 0)

        image_iter = None
        next_image = None
        if pdf_doc:
            try:
                image_iter = iter(extract_pdf_images(pdf_doc))
                next_image = next(image_iter)
            except StopIteration:
                next_image = None
            except Exception as e:
                logger.warning(f"Could not initialize image iterator: {e}")

        pages_by_num = {}
        for p_data in pages:
            pages_by_num.setdefault(p_data["page"], []).append(p_data)

        for page_num in range(1, total_pages + 1):
            page_data_list = pages_by_num.get(page_num, [])
            for page_data in page_data_list:
                text = page_data["text"]
                chunk_type = page_data.get("chunk_type", "text")

                if chunk_type == "table":
                    table_chunk = {
                        "text": text.strip(),
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "chunk_type": "table",
                        "bbox": page_data.get("bbox", ""),
                        "table_index": page_data.get("table_index", 0),
                    }
                    for key in (
                        "table_json",
                        "table_extractor",
                        "table_flavor",
                        "table_footnotes",
                    ):
                        if page_data.get(key) not in (None, ""):
                            table_chunk[key] = page_data[key]
                    all_chunks.append(table_chunk)
                    chunk_index += 1
                    continue

                splits = splitter.split_text(text)

                for split_text in splits:
                    if split_text.strip():
                        chunk = {
                            "text": split_text.strip(),
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "chunk_type": chunk_type,
                        }

                        if pdf_doc and page_num <= len(pdf_doc):
                            try:
                                page_obj = pdf_doc[page_num - 1]
                                rects = page_obj.search_for(split_text.strip())
                                if rects:
                                    W, H = float(page_obj.rect.width), float(page_obj.rect.height)
                                    norm_rects = [
                                        [
                                            round(r.x0 / W, 4),
                                            round(r.y0 / H, 4),
                                            round(r.x1 / W, 4),
                                            round(r.y1 / H, 4)
                                        ]
                                        for r in rects
                                    ]
                                    chunk["bbox"] = json.dumps(norm_rects)
                            except Exception as e:
                                logger.warning(f"Bbox extraction error on page {page_num}: {e}")

                        all_chunks.append(chunk)
                        chunk_index += 1

            while next_image and next_image["page"] == page_num:
                img_bytes = next_image["image_bytes"]
                try:
                    from app.rag.ingestion.vision import caption_image
                    caption = caption_image(img_bytes, page=page_num)

                    if caption:
                        all_chunks.append({
                            "text": caption,
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "chunk_type": "text",
                            "is_image": True,
                            "image_caption": caption,
                        })
                        chunk_index += 1
                except Exception as e:
                    logger.warning(f"Failed to generate caption for image on page {page_num}: {e}")
                    fallback_text = f"Image on page {page_num}."
                    all_chunks.append({
                        "text": fallback_text,
                        "page": page_num,
                        "chunk_index": chunk_index,
                        "chunk_type": "text",
                        "is_image": True,
                        "image_caption": fallback_text,
                    })
                    chunk_index += 1
                finally:
                    if next_image:
                        next_image["image_bytes"] = None
                    if "img_bytes" in locals():
                        del img_bytes

                    try:
                        if image_iter is not None:
                            next_image = next(image_iter)
                        else:
                            next_image = None
                    except StopIteration:
                        next_image = None
                    except Exception as e:
                        logger.warning(f"Error getting next image from iterator: {e}")
                        next_image = None

            import gc
            gc.collect()

    finally:
        if pdf_doc:
            pdf_doc.close()

    return all_chunks


def get_page_count(filepath: str) -> int:
    """Get total page count of a document."""
    ext = filepath.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        doc = fitz.open(filepath)
        count = len(doc)
        doc.close()
        return count

    return 1

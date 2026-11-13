"""Docling + TableFormer Accurate → canonical TableData.

Primary production table extractor. Maps Docling table items (cells, spans,
headers) into Genova ``TableData`` so downstream TableRAG never depends on
the extraction engine.

Fallback for failed/unreliable pages is DetectDocumentText (see textract_tables).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.rag.ingestion.pdf_layout.header_reconstruction import (
    reconstruct_table_headers_cached,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData, normalize_ws

logger = logging.getLogger(__name__)

_EXTRACTION_ENGINE = "docling"
_converter = None


def reset_docling_converter() -> None:
    """Test helper to clear the cached converter."""
    global _converter
    _converter = None


def _tableformer_mode():
    settings = get_settings()
    mode = (getattr(settings, "PDF_DOCLING_TABLEFORMER_MODE", "accurate") or "accurate").strip().lower()
    try:
        from docling.datamodel.pipeline_options import TableFormerMode

        if mode == "fast":
            return TableFormerMode.FAST
        return TableFormerMode.ACCURATE
    except Exception:
        return mode or "accurate"


def get_docling_converter():
    """Lazy DocumentConverter configured for TableFormer Accurate."""
    global _converter
    if _converter is not None:
        return _converter
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "docling is required for primary table extraction. "
            "Install with: pip install docling"
        ) from exc

    settings = get_settings()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    try:
        pipeline_options.table_structure_options.mode = _tableformer_mode()
        pipeline_options.table_structure_options.do_cell_matching = True
    except Exception as exc:
        logger.debug("Could not set TableFormer mode: %s", exc)
    try:
        pipeline_options.do_ocr = bool(getattr(settings, "PDF_DOCLING_DO_OCR", True))
    except Exception:
        pass

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return _converter


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_ws(value)
    # Never stringify bound methods / callables (e.g. TableItem.caption_text).
    if callable(value) and not isinstance(value, (bytes, bytearray)):
        return ""
    text = getattr(value, "text", None)
    if callable(text):
        return ""
    if text is not None:
        return normalize_ws(str(text))
    return normalize_ws(str(value))


def _caption_from_docling_table(table: Any, document: Any = None) -> str:
    """Resolve a short caption string from a Docling table item.

    Docling 2.91 exposes ``caption_text(doc)`` as a method. Accessing the
    attribute without calling it previously dumped the method repr (hundreds
    of KB) into chunk text and broke Titan embeddings (50k char / 8k token
    limits).
    """
    # Preferred: callable caption_text(doc=...)
    caption_fn = getattr(table, "caption_text", None)
    if callable(caption_fn):
        try:
            if document is not None:
                try:
                    text = caption_fn(document)
                except TypeError:
                    text = caption_fn(doc=document)
            else:
                text = caption_fn()
            if text:
                return normalize_ws(str(text))
        except Exception:
            pass

    for attr in ("caption", "title"):
        val = getattr(table, attr, None)
        if val is None or callable(val):
            continue
        text = _cell_text(val)
        if text:
            return text
    return ""


def _bbox_from_docling(item: Any, page_width: float, page_height: float) -> BBox:
    """Best-effort bbox from Docling provenance / export."""
    # Prefer explicit page-local bbox if present.
    for attr in ("bbox", "bounding_box", "prov"):
        raw = getattr(item, attr, None)
        if raw is None:
            continue
        try:
            if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                return BBox(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
            if isinstance(raw, dict):
                l = float(raw.get("l", raw.get("left", raw.get("x0", 0))) or 0)
                t = float(raw.get("t", raw.get("top", raw.get("y0", 0))) or 0)
                r = float(raw.get("r", raw.get("right", raw.get("x1", page_width))) or page_width)
                b = float(raw.get("b", raw.get("bottom", raw.get("y1", page_height))) or page_height)
                return BBox(l, t, r, b)
            # Provenance list
            if isinstance(raw, (list, tuple)) and raw:
                first = raw[0]
                bbox = getattr(first, "bbox", None) or getattr(first, "bounding_box", None)
                if bbox is not None:
                    return _bbox_from_docling(type("X", (), {"bbox": bbox})(), page_width, page_height)
        except Exception:
            continue
    return BBox(0.0, 0.0, float(page_width or 0), float(page_height or 0))


def _page_number_from_table(table: Any, default: int = 1) -> int:
    for attr in ("page_no", "page", "page_number"):
        val = getattr(table, attr, None)
        if val is not None:
            try:
                return int(val) + (0 if int(val) >= 1 else 1)
            except Exception:
                pass
    prov = getattr(table, "prov", None) or []
    try:
        if prov:
            page_no = getattr(prov[0], "page_no", None)
            if page_no is not None:
                return int(page_no) + (1 if int(page_no) == 0 else 0)
    except Exception:
        pass
    return default


def _grid_from_dataframe(df: Any) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    rows_raw = [[_cell_text(c) for c in row] for row in df.values.tolist()]
    rows_raw = [r for r in rows_raw if any(r)]
    if not rows_raw:
        return [], [], []
    # Prefer DataFrame columns when they look like real headers.
    col_labels = [_cell_text(c) for c in list(df.columns)]
    if col_labels and not all(str(c).startswith("Unnamed") for c in col_labels):
        headers = col_labels
        body = rows_raw
        header_rows = [list(headers)]
    else:
        headers = list(rows_raw[0])
        body = rows_raw[1:]
        header_rows = [list(headers)]
    return headers, body, header_rows


def _grid_from_table_data(data: Any) -> Tuple[List[str], List[List[str]], List[List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse Docling TableData-like structure when dataframe export is unavailable."""
    cells_meta: List[Dict[str, Any]] = []
    merged: List[Dict[str, Any]] = []
    grid_map: Dict[Tuple[int, int], str] = {}
    max_r, max_c = -1, -1

    table_cells = getattr(data, "table_cells", None) or getattr(data, "cells", None) or []
    for cell in table_cells:
        try:
            r0 = int(getattr(cell, "start_row_offset_idx", getattr(cell, "row", 0)) or 0)
            c0 = int(getattr(cell, "start_col_offset_idx", getattr(cell, "col", 0)) or 0)
            r1 = int(getattr(cell, "end_row_offset_idx", r0 + 1) or (r0 + 1))
            c1 = int(getattr(cell, "end_col_offset_idx", c0 + 1) or (c0 + 1))
            text = _cell_text(getattr(cell, "text", cell))
            row_span = max(1, r1 - r0)
            col_span = max(1, c1 - c0)
            grid_map[(r0, c0)] = text
            max_r = max(max_r, r1 - 1)
            max_c = max(max_c, c1 - 1)
            cells_meta.append(
                {
                    "row_index": r0,
                    "col_index": c0,
                    "text": text,
                    "confidence": 100.0,
                    "bbox": None,
                    "row_span": row_span,
                    "col_span": col_span,
                    "is_header": bool(getattr(cell, "column_header", False))
                    or bool(getattr(cell, "row_header", False)),
                }
            )
            if row_span > 1 or col_span > 1:
                merged.append(
                    {
                        "row_index": r0,
                        "col_index": c0,
                        "row_span": row_span,
                        "col_span": col_span,
                        "text": text,
                    }
                )
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if (rr, cc) not in grid_map:
                        grid_map[(rr, cc)] = text
        except Exception:
            continue

    if max_r < 0 or max_c < 0:
        return [], [], [], [], []

    grid = [
        [grid_map.get((r, c), "") for c in range(max_c + 1)]
        for r in range(max_r + 1)
    ]
    header_count = 0
    for row_cells in cells_meta:
        if row_cells.get("is_header") and int(row_cells.get("row_index") or 0) == 0:
            header_count = 1
            break
    # Promote first row as header when any column_header flagged on row 0,
    # else default to first row.
    if header_count == 0 and len(grid) >= 2:
        header_count = 1
    header_rows = [list(grid[r]) for r in range(header_count)]
    headers = list(header_rows[0]) if header_rows else []
    body = [list(grid[r]) for r in range(header_count, len(grid))]
    return headers, body, header_rows, cells_meta, merged


def table_data_from_docling_table(
    table: Any,
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    document: Any = None,
) -> Optional[TableData]:
    """Convert one Docling table item into TableData."""
    headers: List[str] = []
    body: List[List[str]] = []
    header_rows: List[List[str]] = []
    cells_meta: List[Dict[str, Any]] = []
    merged: List[Dict[str, Any]] = []

    # Prefer dataframe export (stable grid). Pass ``doc`` when available —
    # Docling 2.91+ warns that export without the parent document is deprecated
    # and can degrade header reconstruction.
    df = None
    for meth in ("export_to_dataframe", "to_dataframe"):
        fn = getattr(table, meth, None)
        if callable(fn):
            try:
                if document is not None and meth == "export_to_dataframe":
                    df = fn(doc=document)
                else:
                    df = fn()
                break
            except TypeError:
                try:
                    df = fn()
                    break
                except Exception:
                    df = None
            except Exception:
                df = None
    if df is not None:
        headers, body, header_rows = _grid_from_dataframe(df)

    data = getattr(table, "data", None)
    if (not headers and not body) and data is not None:
        headers, body, header_rows, cells_meta, merged = _grid_from_table_data(data)

    if not headers and not body:
        return None

    width = max(len(headers), max((len(r) for r in body), default=0), 1)
    headers = list(headers) + [""] * max(0, width - len(headers))
    headers = headers[:width]
    body = [list(r) + [""] * max(0, width - len(r)) for r in body]
    body = [r[:width] for r in body]

    caption = _caption_from_docling_table(table, document=document)

    bbox = _bbox_from_docling(table, page_width, page_height)
    page = _page_number_from_table(table, default=page_number)

    recon = reconstruct_table_headers_cached(
        header_rows=header_rows,
        rows=body,
        cells=cells_meta,
        merged_cells=merged,
        existing_headers=headers,
    )
    if recon.leaf_headers and len(recon.leaf_headers) == width:
        headers = list(recon.leaf_headers)
    if recon.reconstructed_header_rows:
        header_rows = [list(r) for r in recon.reconstructed_header_rows]
    if recon.reconstructed_rows:
        body = [list(r) for r in recon.reconstructed_rows]

    # Honest prior for the quality gate — avoid a flat 0.85 that inflates
    # structurally weak grids. Quality filter still overwrites this score.
    structure_prior = _estimate_structure_prior(
        headers=headers,
        rows=body,
        merged=merged,
        recon_confidence=float(getattr(recon, "normalization_confidence", 0.0) or 0.0),
    )

    return TableData(
        page=page,
        bbox=bbox,
        headers=headers,
        rows=body,
        extraction_method=_EXTRACTION_ENGINE,
        extraction_engine=_EXTRACTION_ENGINE,
        caption=caption,
        source_lines=len(body) + len(header_rows),
        cells=cells_meta,
        header_rows=header_rows,
        merged_cells=merged,
        ocr_confidence=0.0,
        structure_confidence=structure_prior,
        textract_confidence=0.0,
        header_reconstruction=recon.to_dict(),
        normalized_header_paths=[list(p) for p in recon.normalized_header_paths],
        pages=[page],
        provenance={
            "engine": _EXTRACTION_ENGINE,
            "page": page,
            "tableformer_mode": str(
                getattr(get_settings(), "PDF_DOCLING_TABLEFORMER_MODE", "accurate")
            ),
            "structure_prior": structure_prior,
            "quality_notes": [],
        },
    )


def _estimate_structure_prior(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    merged: Sequence[Dict[str, Any]],
    recon_confidence: float = 0.0,
) -> float:
    """Domain-agnostic structure prior from grid signals (0–1)."""
    width = max([len(headers)] + [len(r) for r in rows] or [0])
    if width < 2 or not rows:
        return 0.25
    score = 0.55
    # Real-looking headers (not Column N / blank).
    synthetic = sum(
        1 for h in headers if not normalize_ws(h) or normalize_ws(h).lower().startswith("column ")
    )
    if synthetic == 0 and headers:
        score += 0.2
    elif synthetic >= max(2, (len(headers) + 1) // 2):
        score -= 0.25
    # Empty-cell ratio.
    total = sum(max(width, len(r)) for r in rows) or 1
    empty = 0
    for row in rows:
        padded = list(row) + [""] * max(0, width - len(row))
        empty += sum(1 for c in padded[:width] if not normalize_ws(str(c)))
    empty_ratio = empty / total
    if empty_ratio > 0.5:
        score -= 0.2
    elif empty_ratio < 0.15:
        score += 0.1
    # Merged cells are informative but not proof of quality.
    if merged:
        score += 0.05
    if recon_confidence > 0:
        score = 0.7 * score + 0.3 * max(0.0, min(1.0, recon_confidence))
    return round(max(0.15, min(0.92, score)), 3)


def _pdf_page_count(filepath: str, page_sizes: Optional[Dict[int, Tuple[float, float]]]) -> int:
    if page_sizes:
        try:
            return max(int(p) for p in page_sizes.keys())
        except Exception:
            pass
    try:
        import fitz

        doc = fitz.open(filepath)
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    except Exception:
        return 0


def _collect_docling_table_items(document: Any) -> List[Any]:
    if hasattr(document, "tables") and document.tables:
        return list(document.tables)
    if hasattr(document, "iterate_items"):
        try:
            return [
                item
                for item, _path in document.iterate_items()
                if getattr(item, "label", None) == "table"
                or item.__class__.__name__.lower().endswith("table")
            ]
        except Exception:
            return []
    return []


def _convert_page_batch(
    converter: Any,
    filepath: str,
    *,
    start_page: int,
    end_page: int,
) -> Tuple[Any, List[Any]]:
    """Convert one inclusive 1-based page range; return (document, table items)."""
    result = converter.convert(
        filepath,
        raises_on_error=False,
        page_range=(int(start_page), int(end_page)),
    )
    document = getattr(result, "document", result)
    return document, _collect_docling_table_items(document)


def extract_tables_from_pdf(
    filepath: str,
    *,
    page_sizes: Optional[Dict[int, Tuple[float, float]]] = None,
) -> List[TableData]:
    """Run Docling on a PDF and return canonical tables for all pages.

    Large PDFs are converted in ``PDF_DOCLING_PAGE_BATCH_SIZE`` page batches via
    Docling ``page_range`` to avoid torch/layout ``std::bad_alloc`` OOM on
    Windows CPU hosts. Set batch size to 0 for a single full-document convert.
    """
    settings = get_settings()
    if not getattr(settings, "PDF_DOCLING_TABLES_ENABLED", True):
        return []

    batch_size = int(getattr(settings, "PDF_DOCLING_PAGE_BATCH_SIZE", 4) or 0)
    page_sizes = page_sizes or {}
    n_pages = _pdf_page_count(filepath, page_sizes)

    try:
        converter = get_docling_converter()
    except Exception as exc:
        logger.warning("Docling converter init failed for %s: %s", filepath, exc)
        raise

    # Decide convert strategy.
    use_batches = bool(batch_size > 0 and n_pages > batch_size)
    batches: List[Optional[Tuple[int, int]]]
    if use_batches:
        batches = []
        start = 1
        while start <= n_pages:
            end = min(n_pages, start + batch_size - 1)
            batches.append((start, end))
            start = end + 1
    else:
        batches = [None]  # full document

    out: List[TableData] = []
    batch_errors: List[str] = []
    for batch in batches:
        batch_label = "full" if batch is None else f"{batch[0]}-{batch[1]}"
        default_page = 1 if batch is None else int(batch[0])
        try:
            if batch is None:
                result = converter.convert(filepath, raises_on_error=False)
                document = getattr(result, "document", result)
                tables_raw = _collect_docling_table_items(document)
            else:
                start_page, end_page = batch
                document, tables_raw = _convert_page_batch(
                    converter,
                    filepath,
                    start_page=start_page,
                    end_page=end_page,
                )
        except Exception as exc:
            msg = f"pages {batch_label}: {type(exc).__name__}: {exc}"
            batch_errors.append(msg)
            logger.warning("Docling batch failed for %s (%s)", filepath, msg)
            continue

        for ordinal, table in enumerate(tables_raw):
            page = _page_number_from_table(table, default=default_page)
            width, height = page_sizes.get(page, (612.0, 792.0))
            try:
                td = table_data_from_docling_table(
                    table,
                    page_number=page,
                    page_width=float(width),
                    page_height=float(height),
                    document=document,
                )
            except Exception as exc:
                logger.debug("Skipping Docling table %s: %s", ordinal, exc)
                continue
            if td:
                # Docling often leaves caption empty even when the PDF has
                # "Table N. …" above the grid — recover from page geometry.
                if not (td.caption or "").strip():
                    try:
                        from app.rag.ingestion.pdf_layout.caption_recovery import (
                            enrich_table_caption,
                        )

                        enrich_table_caption(
                            td,
                            filepath=filepath,
                            document=document,
                            docling_table=table,
                        )
                    except Exception as cap_exc:
                        logger.debug(
                            "Caption recovery skipped for page %s table %s: %s",
                            page,
                            ordinal,
                            cap_exc,
                        )
                out.append(td)

    if not out and batch_errors:
        # No tables and every batch failed — surface to hybrid fallback.
        raise RuntimeError(
            "Docling conversion failed for all page batches: "
            + "; ".join(batch_errors[:3])
        )
    if batch_errors:
        logger.warning(
            "Docling partial success for %s: %d tables, %d batch errors",
            filepath,
            len(out),
            len(batch_errors),
        )
    return out


def extract_tables_for_page(
    filepath: str,
    page_number: int,
    *,
    page_width: float,
    page_height: float,
    cached_doc_tables: Optional[Sequence[TableData]] = None,
) -> List[TableData]:
    """Return Docling tables for one 1-indexed page.

    When ``cached_doc_tables`` is provided (full-document convert), filter by page
    instead of re-converting.
    """
    if cached_doc_tables is not None:
        return [t for t in cached_doc_tables if int(t.page) == int(page_number)]

    all_tables = extract_tables_from_pdf(
        filepath, page_sizes={page_number: (page_width, page_height)}
    )
    return [t for t in all_tables if int(t.page) == int(page_number)]

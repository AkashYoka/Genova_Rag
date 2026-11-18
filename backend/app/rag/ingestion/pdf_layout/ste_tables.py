"""STE Textract AnalyzeDocument + Bedrock Nova table extraction for Genova layout.

Uses the in-repo package ``app/ste`` (Textract AnalyzeDocument TABLES +
optional Bedrock Nova 2 Lite reconstruction). Peer to ``textract_tables.py``:
produces ``TableData`` for ``parse_pdf`` / TableRAG. Semantic chunking for
prose is unchanged.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.rag.ingestion.pdf_layout.types import BBox, TableData

logger = logging.getLogger(__name__)

EXTRACTION_ENGINE = "ste_textract_analyze_bedrock"
DEFAULT_IOU = 0.55
ProgressCallback = Callable[[str, int], None]


def ensure_ste_importable() -> None:
    """Make ``import ste`` work via ``app.ste`` (shipped inside the app package).

    Production only deploys ``app/``; STE lives at ``app/ste`` and is aliased to
    the top-level name ``ste`` so existing ``from ste...`` imports keep working
    without a Docker image rewrite.
    """
    try:
        from app.ste_path import install_ste_alias

        install_ste_alias()
    except Exception:
        pass

    try:
        import ste  # noqa: F401

        return
    except ImportError:
        pass

    # Fallback: app.ste direct
    try:
        import importlib

        app_ste = importlib.import_module("app.ste")
        sys.modules.setdefault("ste", app_ste)
        import ste  # noqa: F401

        return
    except ImportError as exc:
        raise ModuleNotFoundError(
            "No module named 'ste' / 'app.ste'. Ensure app/ste is present in "
            "the deployed app package (STE was moved under app/ so normal app "
            "deploys include it without Docker COPY changes)."
        ) from exc


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ste_cells_to_genova(
    cells: Sequence[Any],
    *,
    header_row_count: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Map STE ``Cell`` objects to Genova cell dicts + low-confidence list."""
    out: List[Dict[str, Any]] = []
    low: List[Dict[str, Any]] = []
    for cell in cells or []:
        if isinstance(cell, dict):
            row = int(cell.get("row", cell.get("row_index", 0)) or 0)
            col = int(cell.get("col", cell.get("col_index", 0)) or 0)
            text = _cell_text(
                cell.get("text")
                if cell.get("text") is not None
                else cell.get("value")
                if cell.get("value") is not None
                else cell.get("original")
            )
            status = str(cell.get("status") or "present")
            conf = float(cell.get("confidence") or 0.0)
            bbox = cell.get("bbox")
            rowspan = int(cell.get("rowspan") or cell.get("row_span") or 1)
            colspan = int(cell.get("colspan") or cell.get("col_span") or 1)
            original = _cell_text(cell.get("original") or text)
            explanation = str(cell.get("explanation") or "")
            region_id = str(cell.get("region_id") or "")
        else:
            row = int(getattr(cell, "row", 0) or 0)
            col = int(getattr(cell, "col", 0) or 0)
            status = str(getattr(cell, "status", "present") or "present")
            raw_text = getattr(cell, "text", None)
            text = "" if raw_text is None else _cell_text(raw_text)
            conf = float(getattr(cell, "confidence", 0.0) or 0.0)
            bbox_obj = getattr(cell, "bbox", None)
            bbox = bbox_obj.as_list() if bbox_obj is not None and hasattr(bbox_obj, "as_list") else None
            rowspan = int(getattr(cell, "rowspan", 1) or 1)
            colspan = int(getattr(cell, "colspan", 1) or 1)
            original = _cell_text(getattr(cell, "original", "") or text)
            explanation = str(getattr(cell, "explanation", "") or "")
            region_id = str(getattr(cell, "region_id", "") or "")

        # Genova cell contract uses body-relative row_index for body cells and
        # absolute indices for header cells (is_header=True). STE cells are
        # body-relative (row 0 = first data row) after Bedrock rebuild.
        is_header = False
        entry: Dict[str, Any] = {
            "row_index": row,
            "col_index": col,
            "text": text,
            "confidence": conf * 100.0 if conf <= 1.0 else conf,
            "bbox": bbox,
            "row_span": 1,  # fully cellular: never retain spans for retrieval
            "col_span": 1,
            "is_header": is_header,
            "status": status,
            "original": original,
        }
        if explanation:
            entry["explanation"] = explanation
        if region_id:
            entry["region_id"] = region_id
        # Preserve span metadata for audit only (not used as merges).
        if rowspan != 1 or colspan != 1:
            entry["source_row_span"] = rowspan
            entry["source_col_span"] = colspan
        out.append(entry)

        conf_pct = entry["confidence"]
        if conf_pct and conf_pct < 50.0 and text:
            low.append(
                {
                    "row": row,
                    "col": col,
                    "text": text,
                    "confidence": conf_pct,
                }
            )
    del header_row_count  # reserved for future header-band mapping
    return out, low


def _row_values_from_independent(row: Dict[str, Any], headers: Sequence[str]) -> List[str]:
    """Build a cellular row list from one STE independent_row payload."""
    cells = row.get("cells") or {}
    if not isinstance(cells, dict):
        return []
    # Prefer header-order; fall back to insertion order of cells.
    values: List[str] = []
    if headers:
        # Match by header string on cell payload.
        by_header = {
            str((payload or {}).get("header") or ""): payload
            for payload in cells.values()
            if isinstance(payload, dict)
        }
        for header in headers:
            payload = by_header.get(str(header))
            if payload is None:
                # Try snake_case key match.
                payload = next(
                    (
                        cells[k]
                        for k in cells
                        if isinstance(cells[k], dict)
                        and str(cells[k].get("header") or "") == str(header)
                    ),
                    None,
                )
            if not isinstance(payload, dict):
                values.append("")
                continue
            status = str(payload.get("status") or "present")
            if status in {
                "empty",
                "insufficient_data",
                "not_applicable",
                "not_reported",
                "illegible",
                "redacted",
                "uncertain",
            }:
                # Keep explicit marker text when present; never invent.
                original = payload.get("original_value")
                values.append(_cell_text(original) if original else "")
            else:
                val = payload.get("normalized_value")
                if val is None:
                    val = payload.get("original_value")
                values.append(_cell_text(val))
        return values
    for payload in cells.values():
        if not isinstance(payload, dict):
            values.append("")
            continue
        val = payload.get("normalized_value")
        if val is None:
            val = payload.get("original_value")
        values.append(_cell_text(val))
    return values


def universal_dict_to_table_data(
    universal: Dict[str, Any],
    *,
    extraction_method: str = "textract_analyze_tables",
) -> TableData:
    """Map a STE ``universal.json`` dict into Genova ``TableData``."""
    headers = [
        _cell_text(h)
        for h in (
            universal.get("reconstructed_headers")
            or universal.get("original_headers")
            or []
        )
    ]
    independent_rows = list(universal.get("independent_rows") or [])
    rows: List[List[str]] = []
    if independent_rows:
        for ir in independent_rows:
            rows.append(_row_values_from_independent(ir if isinstance(ir, dict) else {}, headers))
    else:
        # Fall back to original_grid body (skip header band if present).
        grid = [list(r) for r in (universal.get("original_grid") or [])]
        if grid and headers and [ _cell_text(c) for c in grid[0] ] == headers:
            rows = [[_cell_text(c) for c in r] for r in grid[1:]]
        else:
            rows = [[_cell_text(c) for c in r] for r in grid]

    # Pad ragged rows; never invent non-empty values.
    width = max(len(headers), max((len(r) for r in rows), default=0), 0)
    if width and len(headers) < width:
        headers = headers + [f"Column {i + 1}" for i in range(len(headers), width)]
    rows = [(list(r) + [""] * width)[:width] for r in rows]

    source_prov = dict(universal.get("source_provenance") or {})
    bbox_raw = source_prov.get("bbox") or universal.get("bbox")
    bbox = BBox.from_any(bbox_raw) or BBox(0.0, 0.0, 0.0, 0.0)
    page = int(universal.get("page_number") or source_prov.get("page") or 1)
    pages = list(source_prov.get("pages") or [page])
    pages = [int(p) for p in pages] if pages else [page]

    conf = float(universal.get("extraction_confidence") or 0.0)
    warnings = list(universal.get("reconstruction_warnings") or [])
    reconstructed = [
        _cell_text(h) for h in (universal.get("reconstructed_headers") or headers)
    ]
    original_headers = [
        _cell_text(h) for h in (universal.get("original_headers") or [])
    ]

    provenance: Dict[str, Any] = {
        **source_prov,
        "ste_table_id": universal.get("table_id") or "",
        "document_id": universal.get("document_id") or source_prov.get("document_id") or "",
        "reconstructed_headers": reconstructed,
        "original_headers": original_headers,
        "original_grid": universal.get("original_grid"),
        "independent_rows": independent_rows,
        "unavailable_regions": list(universal.get("unavailable_regions") or []),
        "reconstruction_warnings": warnings,
        "original_extraction": {
            "original_headers": original_headers,
            "original_grid": universal.get("original_grid"),
            "extraction_method": extraction_method
            or source_prov.get("extraction_method")
            or "",
        },
        "fully_cellular": True,
        "extraction_engine": EXTRACTION_ENGINE,
    }

    return TableData(
        page=page,
        bbox=bbox,
        headers=reconstructed or headers,
        rows=rows,
        extraction_method=extraction_method
        or str(source_prov.get("extraction_method") or "textract_analyze_tables"),
        caption=_cell_text(universal.get("table_title") or ""),
        cells=[],
        header_rows=[original_headers] if original_headers else [],
        extraction_engine=EXTRACTION_ENGINE,
        ocr_confidence=conf * 100.0 if conf <= 1.0 else conf,
        structure_confidence=conf if conf <= 1.0 else conf / 100.0,
        provenance=provenance,
        pages=pages,
        merged_cells=[],
        independent_rows=independent_rows,
    )


def extracted_table_to_table_data(
    table: Any,
    *,
    document_id: str = "",
) -> TableData:
    """Map a STE ``ExtractedTable`` (post-Bedrock) into Genova ``TableData``."""
    from ste.universal_schema import to_universal_dict

    universal = to_universal_dict(table, document_id=document_id or "")
    td = universal_dict_to_table_data(
        universal,
        extraction_method=str(
            getattr(table, "extraction_method", "") or "textract_analyze_tables"
        ),
    )

    genova_cells, low = _ste_cells_to_genova(getattr(table, "cells", None) or [])
    td.cells = genova_cells
    td.low_confidence_cells = low
    td.header_rows = [list(r) for r in (getattr(table, "header_rows", None) or [])] or td.header_rows

    # Enrich provenance with live ExtractedTable fields (audit).
    prov = dict(td.provenance or {})
    ste_prov = dict(getattr(table, "provenance", None) or {})
    for key, value in ste_prov.items():
        prov.setdefault(key, value)
    prov["ste_table_id"] = str(getattr(table, "table_id", "") or prov.get("ste_table_id") or "")
    prov["warnings"] = list(getattr(table, "warnings", None) or [])
    if getattr(table, "self_contained_rows", None):
        prov["self_contained_rows"] = list(table.self_contained_rows)
    # Keep original_extraction snapshot when Bedrock already stashed originals.
    if ste_prov.get("original_grid") and not (prov.get("original_extraction") or {}).get(
        "original_grid"
    ):
        prov["original_extraction"] = {
            "original_headers": list(ste_prov.get("original_headers") or []),
            "original_grid": ste_prov.get("original_grid"),
            "extraction_method": getattr(table, "extraction_method", "") or "",
        }
    td.provenance = prov

    conf = float(getattr(table, "confidence", 0.0) or 0.0)
    if conf:
        td.structure_confidence = conf if conf <= 1.0 else conf / 100.0
        td.ocr_confidence = conf * 100.0 if conf <= 1.0 else conf
    return td


def table_content_fingerprint(table: TableData) -> str:
    """Stable content fingerprint for dedupe when bbox IoU is unavailable."""
    headers = "|".join(_cell_text(h).lower() for h in (table.headers or []))
    sample_rows = (table.rows or [])[:3]
    body = "||".join(
        "|".join(_cell_text(c).lower() for c in row) for row in sample_rows
    )
    pages = ",".join(str(p) for p in (table.pages or [table.page]))
    return f"{pages}::{headers}::{body}"


def merge_tables_prefer_ste(
    existing: Sequence[TableData],
    ste_tables: Sequence[TableData],
    *,
    iou_threshold: float = DEFAULT_IOU,
) -> List[TableData]:
    """Merge page tables preferring STE; drop overlapping Textract duplicates.

    Matching uses page + bbox IoU, then content fingerprint. STE always wins
    overlaps. Non-overlapping existing tables are kept (e.g. rare Textract-only
    finds STE missed).
    """
    ste_list = list(ste_tables or [])
    if not ste_list:
        return list(existing or [])
    if not existing:
        return ste_list

    kept_existing: List[TableData] = []
    ste_fps = {table_content_fingerprint(t) for t in ste_list}

    for other in existing:
        engine = (other.extraction_engine or other.extraction_method or "").lower()
        if EXTRACTION_ENGINE in engine or engine.startswith("ste_"):
            # Already STE — avoid duplicating when caller passes mixed lists.
            continue
        overlapped = False
        other_pages = set(other.pages or [other.page])
        for ste in ste_list:
            ste_pages = set(ste.pages or [ste.page])
            if other_pages.isdisjoint(ste_pages) and int(other.page) not in ste_pages:
                if int(ste.page) not in other_pages:
                    continue
            if other.bbox and ste.bbox and other.bbox.area > 0 and ste.bbox.area > 0:
                if other.bbox.iou(ste.bbox) >= float(iou_threshold):
                    overlapped = True
                    break
            if table_content_fingerprint(other) == table_content_fingerprint(ste):
                overlapped = True
                break
            if table_content_fingerprint(other) in ste_fps:
                overlapped = True
                break
        if not overlapped:
            kept_existing.append(other)

    # STE first (primary), then any non-overlapping leftovers.
    return ste_list + kept_existing


def _estimate_bedrock_cost(usage: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Attach estimated USD cost using Genova pricing tables when available."""
    out = dict(usage or {})
    try:
        from app.pricing import enrich_usage_with_pricing

        enriched = enrich_usage_with_pricing(
            {
                "input_tokens": out.get("input_tokens"),
                "output_tokens": out.get("output_tokens"),
                "total_tokens": out.get("total_tokens"),
            },
            provider="bedrock",
            model_id=model_id,
        )
        out["estimated_cost_usd"] = enriched.get("estimated_cost_usd")
        out["cost_type"] = enriched.get("cost_type")
        out["pricing_note"] = enriched.get("pricing_note")
        out["model_id"] = model_id or enriched.get("model_id")
    except Exception as exc:
        out["cost_estimate_error"] = f"{type(exc).__name__}: {exc}"
    return out


def extract_ste_tables_from_pdf(
    filepath: str,
    *,
    document_id: str = "",
    document_name: str = "",
    pages: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    settings: Any = None,
) -> Tuple[List[TableData], Dict[str, Any]]:
    """Run STE ``extract_pdf`` and return Genova tables + run metadata.

    Parameters mirror Genova settings:
    - ``PDF_STE_BEDROCK_ENABLED`` / ``PDF_STE_BEDROCK_MODEL_ID``
    - ``PDF_STE_BACKEND``, ``PDF_STE_OUTPUT_DIR``, ``PDF_STE_USE_CACHE``
    - ``BEDROCK_REGION`` / ``PDF_STE_AWS_REGION`` (Textract + Bedrock)
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    def _report(stage: str, pct: int) -> None:
        if progress:
            try:
                progress(stage, int(pct))
            except Exception:
                pass

    ensure_ste_importable()
    from ste.pipeline import extract_pdf
    from ste.settings import Settings as SteSettings

    bedrock_on = bool(getattr(settings, "PDF_STE_BEDROCK_ENABLED", True))
    model_id = (
        (getattr(settings, "PDF_STE_BEDROCK_MODEL_ID", "") or "").strip()
        or (getattr(settings, "BEDROCK_MODEL_ID", "") or "").strip()
        or "us.amazon.nova-2-lite-v1:0"
    )
    region = (
        (getattr(settings, "PDF_STE_AWS_REGION", "") or "").strip()
        or (getattr(settings, "PDF_TEXTRACT_REGION", "") or "").strip()
        or (getattr(settings, "BEDROCK_REGION", "") or "").strip()
        or "us-east-1"
    )
    backend = (getattr(settings, "PDF_STE_BACKEND", "textract") or "textract").strip()
    if backend.lower() in {"docling", "docling_tableformer"}:
        backend = "textract"
    use_cache = bool(getattr(settings, "PDF_STE_USE_CACHE", True))
    ste_dpi = int(getattr(settings, "PDF_STE_DPI", 150) or 150)
    max_workers = int(getattr(settings, "PDF_STE_MAX_WORKERS", 4) or 4)
    bedrock_max_workers = int(
        getattr(settings, "PDF_STE_BEDROCK_MAX_WORKERS", 3) or 3
    )
    bedrock_skip_simple = bool(
        getattr(settings, "PDF_STE_BEDROCK_SKIP_SIMPLE", False)
    )
    if pages is None:
        raw_pages = (getattr(settings, "PDF_STE_PAGES", "") or "").strip()
        if raw_pages:
            pages = []
            for part in raw_pages.replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    pages.append(int(part))
                except ValueError:
                    continue
            pages = sorted(set(p for p in pages if p > 0)) or None
    out_root = Path(
        output_dir
        or getattr(settings, "PDF_STE_OUTPUT_DIR", "")
        or "./data/ste_runs"
    )
    run_name = document_id or document_name or Path(filepath).stem
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)[:80]
    run_dir = out_root / safe_name / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ste_settings = SteSettings(
        backend=backend,
        dpi=ste_dpi,
        tables_only=True,
        pages=tuple(pages) if pages else (),
        use_cache=use_cache,
        stitch_multipage=True,
        bedrock=bedrock_on,
        bedrock_model_id=model_id,
        aws_region=region,
        max_workers=max_workers,
        bedrock_max_workers=bedrock_max_workers,
        bedrock_skip_simple=bedrock_skip_simple,
    )

    _report("ste_extract", 30)
    logger.info(
        "STE Textract+Bedrock starting for %s (bedrock=%s model=%s backend=%s "
        "dpi=%s workers=%s bedrock_workers=%s skip_simple=%s pages=%s)",
        Path(filepath).name,
        bedrock_on,
        model_id,
        backend,
        ste_dpi,
        max_workers,
        bedrock_max_workers,
        bedrock_skip_simple,
        pages or "all",
    )
    started = time.perf_counter()
    try:
        result = extract_pdf(
            filepath,
            output_dir=run_dir,
            settings=ste_settings,
            backend=backend,
            dpi=ste_dpi,
            tables_only=True,
            pages=pages,
            use_cache=use_cache,
            stitch_multipage=True,
            bedrock=bedrock_on,
            bedrock_model_id=model_id,
            aws_region=region,
        )
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 2)
        logger.exception("STE extract_pdf failed for %s: %s", filepath, exc)
        return [], {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": elapsed,
            "output_dir": str(run_dir),
            "extraction_engine": EXTRACTION_ENGINE,
        }

    _report("ste_map", 45)
    tables: List[TableData] = []
    for et in list(getattr(result, "tables", None) or []):
        try:
            tables.append(
                extracted_table_to_table_data(et, document_id=document_id or "")
            )
        except Exception as exc:
            logger.warning(
                "STE table map failed for %s: %s",
                getattr(et, "table_id", "?"),
                exc,
            )

    raw_usage = dict((getattr(result, "meta", None) or {}).get("bedrock_usage") or {})
    usage = _estimate_bedrock_cost(raw_usage, model_id)
    # Per-table token lines when STE provides them.
    per_table = []
    for item in list(raw_usage.get("tables") or []):
        if isinstance(item, dict):
            per_table.append(_estimate_bedrock_cost(item, model_id))
    if per_table:
        usage["tables"] = per_table

    elapsed = round(time.perf_counter() - started, 2)
    result_meta = dict(getattr(result, "meta", None) or {})
    meta: Dict[str, Any] = {
        "ok": True,
        "extraction_engine": EXTRACTION_ENGINE,
        "table_count": len(tables),
        "page_count": int(getattr(result, "page_count", 0) or 0),
        "processed_pages": list(getattr(result, "processed_pages", None) or []),
        "skipped_pages": list(getattr(result, "skipped_pages", None) or []),
        "errors": list(getattr(result, "errors", None) or []),
        "bedrock_enabled": bedrock_on,
        "bedrock_model_id": model_id,
        "bedrock_usage": usage,
        "output_dir": str(run_dir),
        "elapsed_s": elapsed,
        "backend": backend,
        "dpi": ste_dpi,
        "max_workers": max_workers,
        "bedrock_max_workers": bedrock_max_workers,
        "bedrock_skip_simple": bedrock_skip_simple,
        "timings": dict(result_meta.get("timings") or {}),
        "ste_meta": {
            k: v
            for k, v in result_meta.items()
            if k not in {"bedrock_usage"}
        },
    }
    logger.info(
        "STE finished %s: tables=%s elapsed=%ss bedrock_tokens=%s cost_usd=%s",
        Path(filepath).name,
        len(tables),
        elapsed,
        usage.get("total_tokens"),
        usage.get("estimated_cost_usd"),
    )
    _report("ste_done", 50)
    return tables, meta


def ste_tables_for_page(
    ste_tables: Sequence[TableData],
    page_number: int,
) -> List[TableData]:
    """Return STE tables whose primary page or pages list includes ``page_number``."""
    out: List[TableData] = []
    for table in ste_tables or []:
        pages = set(int(p) for p in (table.pages or [table.page]))
        if int(page_number) in pages or int(table.page) == int(page_number):
            # Avoid duplicating multipage tables onto every spanned page —
            # attach only on the primary (min) page.
            primary = min(pages) if pages else int(table.page)
            if int(page_number) == primary:
                out.append(table)
    return out

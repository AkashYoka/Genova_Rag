"""End-to-end extraction (Textract AnalyzeDocument): gate → backend → structural → LLM → exports.

The pipeline contract and output layout are unchanged from the original STE.
Only the extraction layer (and its raw-data adapter) is backend specific.
"""
from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ste.backends import DocumentExtraction, create_backend, resolve_backend_name
from ste.bedrock_reconstructor import DEFAULT_MODEL_ID, apply_bedrock_reconstruction
from ste.cache import ExtractionCache
from ste.cell_status import build_cells
from ste.exporters.csv_exporter import write_csv
from ste.exporters.html_exporter import write_html
from ste.exporters.json_exporter import write_json
from ste.metrics import (
    ResourceMonitor,
    document_facts,
    hardware_info,
    local_cost_summary,
)
from ste.models import (
    BBox,
    DocumentResult,
    ExtractedTable,
    PageGateResult,
    normalize_ws,
)
from ste.page_renderer import crop_preview_png, render_page_image
from ste.pdf_loader import open_pdf, page_sizes, page_text
from ste.reconstructor import (
    caption_near_region,
    cell_bbox_map,
    cell_span_map,
    reconstruct_from_grid,
)
from ste.row_context import attach_self_contained_rows
from ste.settings import Settings
from ste.table_page_gate import page_has_tables
from ste.unavailable_regions import (
    expand_unavailable_regions,
    validate_unavailable_region_coverage,
)
from ste.universal_schema import to_universal_dict
from ste.validator import validate_table

logger = logging.getLogger(__name__)


def _bedrock_enabled(flag: Optional[bool]) -> bool:
    if flag is not None:
        return bool(flag)
    env = (os.environ.get("STE_BEDROCK") or "1").strip().lower()
    return env not in {"0", "false", "off", "no"}


def _looks_image_only(page: Any) -> bool:
    """True when a page has essentially no text layer but carries images.

    Scanned pages cannot be rejected by the cheap text/ruling gate, so they
    are promoted to Textract AnalyzeDocument candidates.
    """
    text = re.sub(r"\s+", "", page_text(page) or "")
    if len(text) >= 80:
        return False
    try:
        images = page.get_images(full=True) or []
    except Exception:
        images = []
    return bool(images) or not text


def _caption_lines(page: Any) -> List[tuple[str, float]]:
    lines: List[tuple[str, float]] = []
    try:
        data = page.get_text("dict") or {}
        for block in data.get("blocks") or []:
            for line in block.get("lines") or []:
                spans = line.get("spans") or []
                text = " ".join(str(s.get("text") or "") for s in spans).strip()
                if not text:
                    continue
                bbox = line.get("bbox") or [0, 0, 0, 0]
                lines.append((text, float(bbox[1])))
    except Exception:
        for index, line in enumerate((page_text(page) or "").splitlines()):
            if line.strip():
                lines.append((line.strip(), float(index)))
    return lines


def extract_pdf(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    settings: Optional[Settings] = None,
    # Convenience overrides (CLI parity with the original STE).
    dpi: Optional[int] = None,
    backend: Optional[str] = None,
    gate: Optional[str] = None,
    ocr: Optional[str] = None,
    tables_only: Optional[bool] = None,
    pages: Optional[List[int]] = None,
    use_cache: Optional[bool] = None,
    cache_dir: Optional[str | Path] = None,
    stitch_multipage: Optional[bool] = None,
    bedrock: Optional[bool] = None,
    bedrock_model_id: Optional[str] = None,
    aws_region: Optional[str] = None,
    # Accepted for backward compatibility; ignored (Docling removed).
    docling_table_mode: Optional[str] = None,
    docling_cell_matching: Optional[bool] = None,
    docling_device: Optional[str] = None,
    docling_num_threads: Optional[int] = None,
    docling_artifacts_path: Optional[str] = None,
) -> DocumentResult:
    """Extract and reconstruct every table in a PDF.

    Args:
        input_path: PDF to process.
        output_dir: Run output directory (created if missing).
        settings: Full settings object; individual keyword arguments override it.
        dpi: Page render DPI used for previews, shading checks and the LLM image.
        backend: ``textract`` (default), ``pymupdf`` or ``auto``.
        gate: ``cues`` (cheap PyMuPDF gate, default) or ``layout`` (every page
            goes to the backend).
        ocr: Local OCR engine preference for PyMuPDF fallback.
        tables_only: Skip pages without a table signal.
        pages: 1-based page subset.
        use_cache: Reuse backend results for unchanged PDFs/settings.
        cache_dir: Cache directory (default ``.ste_cache`` in the project).
        stitch_multipage: Merge header-repeating page-break continuations.
        bedrock: Optional Bedrock semantic reconstruction (LLM only).
        bedrock_model_id: Bedrock model id.
        aws_region: AWS region for Bedrock / Textract.

    Returns:
        :class:`DocumentResult` with tables, errors and run metadata.
    """
    del (
        docling_table_mode,
        docling_cell_matching,
        docling_device,
        docling_num_threads,
        docling_artifacts_path,
    )
    base = settings or Settings()
    resolved_settings = base.with_overrides(
        dpi=dpi,
        backend=backend,
        gate=gate,
        ocr_engine=ocr,
        tables_only=tables_only,
        pages=tuple(pages) if pages else None,
        use_cache=use_cache,
        cache_dir=Path(cache_dir) if cache_dir else None,
        stitch_multipage=stitch_multipage,
        bedrock=bedrock,
        bedrock_model_id=bedrock_model_id,
        aws_region=aws_region,
    )

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    resolved_backend = resolve_backend_name(resolved_settings.backend)
    use_bedrock = _bedrock_enabled(resolved_settings.bedrock)
    model_id = resolved_settings.bedrock_model_id or DEFAULT_MODEL_ID
    document_id = input_path.stem
    page_filter = (
        {int(p) for p in resolved_settings.pages} if resolved_settings.pages else None
    )
    dpi = max(72, min(400, int(resolved_settings.dpi or 200)))
    gate_mode = (resolved_settings.gate or "cues").lower()

    monitor = ResourceMonitor().start()
    doc = open_pdf(input_path)
    sizes = page_sizes(doc)
    page_dims = {page: (width, height) for page, width, height in sizes}

    gated_pages: List[PageGateResult] = []
    candidates: List[int] = []
    skipped_pages: List[int] = []

    try:
        for page_number, _width, _height in sizes:
            if page_filter is not None and page_number not in page_filter:
                skipped_pages.append(page_number)
                continue
            if gate_mode == "layout" or not resolved_settings.tables_only:
                reason = (
                    "layout_mode"
                    if gate_mode == "layout"
                    else "tables_only_disabled"
                )
                gated_pages.append(PageGateResult(page_number, True, reason, 0))
                candidates.append(page_number)
                continue
            page = doc[page_number - 1]
            gate_result = page_has_tables(page, page_number)
            if not gate_result.has_tables and _looks_image_only(page):
                gate_result = PageGateResult(
                    page=page_number,
                    has_tables=True,
                    reason="image_only_page_no_text_layer",
                    native_table_count=0,
                )
            gated_pages.append(gate_result)
            if gate_result.has_tables:
                candidates.append(page_number)
            else:
                skipped_pages.append(page_number)

        backend_options = resolved_settings.backend_options()
        backend_options["dpi"] = dpi

        # ---- cached backend extraction -------------------------------------
        cache = ExtractionCache(
            resolved_settings.cache_dir,
            enabled=bool(resolved_settings.use_cache and candidates),
        )
        cache_key: Optional[str] = None
        cache_hit = False
        extraction: Optional[DocumentExtraction] = None
        if candidates:
            cache_key = cache.make_key(
                input_path,
                backend=resolved_backend,
                options=backend_options,
                pages=candidates,
            )
            cached = cache.load(cache_key)
            if cached is not None:
                extraction = DocumentExtraction.from_payload(cached)
                extraction.meta = dict(extraction.meta or {})
                extraction.meta["cache_hit"] = True
                cache_hit = True

        if extraction is None:
            try:
                engine = create_backend(
                    resolved_backend, options=backend_options
                )
                extraction = engine.extract(
                    str(input_path),
                    pages=candidates,
                    options=backend_options,
                )
            except Exception as exc:
                message = (
                    f"extraction backend {resolved_backend} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception(message)
                extraction = DocumentExtraction(backend=resolved_backend)
                extraction.warnings.append(message)
            if (
                cache.enabled
                and cache_key
                and extraction.tables
                and not (extraction.meta or {}).get("failed_ranges")
            ):
                cache.save(
                    cache_key,
                    extraction.to_payload(),
                    pdf_path=input_path,
                    backend=resolved_backend,
                    options=backend_options,
                    pages=candidates,
                    elapsed_seconds=extraction.elapsed_seconds,
                )

        result = DocumentResult(
            source_path=str(input_path.resolve()),
            page_count=len(sizes),
            ocr_backend=resolved_backend,
            gated_pages=gated_pages,
            processed_pages=[p for p in candidates],
            skipped_pages=skipped_pages,
            meta={
                "dpi": dpi,
                "tables_only": resolved_settings.tables_only,
                "document_id": document_id,
                "pages_filter": sorted(page_filter) if page_filter else None,
                "bedrock_enabled": use_bedrock,
                "bedrock_model_id": model_id if use_bedrock else None,
                "backend": resolved_backend,
                "backend_meta": dict(extraction.meta or {}),
                "cache": {**cache.stats(), "hit": cache_hit, "key": cache_key},
                "settings": resolved_settings.public_dict(),
                "extraction_mode": (
                    "textract_analyze_tables"
                    if resolved_backend == "textract"
                    else "pymupdf_native_geometry"
                ),
            },
        )
        result.errors.extend(
            w for w in extraction.warnings if "failed" in w or "error" in w
        )

        bedrock_usage: List[Dict[str, Any]] = []
        page_render_cache: Dict[int, Optional[Dict[str, Any]]] = {}
        table_index = 0
        t_bedrock = 0.0
        t_export = 0.0
        skip_simple = bool(getattr(resolved_settings, "bedrock_skip_simple", True))
        bedrock_workers = max(
            1, min(8, int(getattr(resolved_settings, "bedrock_max_workers", 3) or 3))
        )

        def _page_render(page_number: int) -> Optional[Dict[str, Any]]:
            if page_number in page_render_cache:
                return page_render_cache[page_number]
            try:
                image_bytes, image_fmt, img_w, img_h, used_dpi = render_page_image(
                    doc[page_number - 1], dpi=dpi
                )
            except Exception as exc:
                message = (
                    f"page {page_number} render failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.error(message)
                result.errors.append(message)
                page_render_cache[page_number] = None
                return None
            page_render_cache[page_number] = {
                "bytes": image_bytes,
                "format": image_fmt,
                "width": img_w,
                "height": img_h,
                "dpi": used_dpi,
            }
            return page_render_cache[page_number]

        def _apply_bedrock_one(
            table: ExtractedTable,
            image_bytes: Optional[bytes],
            image_format: str,
        ) -> Tuple[ExtractedTable, Dict[str, Any]]:
            logger.info(
                "STEP2 Bedrock reconstruction start table=%s page=%s model=%s",
                table.table_id,
                table.page,
                model_id if use_bedrock else "disabled",
            )
            t0 = time.perf_counter()
            out, bmeta = apply_bedrock_reconstruction(
                table,
                image_bytes=image_bytes or b"",
                image_format=image_format,
                document_id=document_id,
                model_id=model_id,
                region=resolved_settings.aws_region,
                enabled=use_bedrock,
                skip_simple=skip_simple,
            )
            bmeta["table_id"] = out.table_id
            bmeta["elapsed_s"] = round(time.perf_counter() - t0, 3)
            logger.info(
                "STEP2 Bedrock done table=%s applied=%s skipped=%s tokens=%s elapsed=%ss",
                out.table_id,
                bmeta.get("applied"),
                bmeta.get("skipped"),
                bmeta.get("total_tokens"),
                bmeta.get("elapsed_s"),
            )
            return out, bmeta

        def _export_one(
            table: ExtractedTable,
            original_payload: Dict[str, Any],
            image_bytes: Optional[bytes],
            bmeta: Dict[str, Any],
            page_obj: Any,
        ) -> None:
            table.provenance = dict(table.provenance or {})
            table.provenance["bedrock"] = {
                key: bmeta.get(key)
                for key in (
                    "enabled",
                    "applied",
                    "model_id",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "rejected_cells",
                    "skipped",
                    "error",
                    "elapsed_s",
                )
                if key in bmeta
            }

            header_count = int(
                len(table.header_rows or [])
                or original_payload.get("header_row_count")
                or 0
            )
            body, regions, warnings, cell_map = expand_unavailable_regions(
                [list(r) for r in table.rows],
                cells_meta=original_payload.get("cells_meta"),
                image_bytes=image_bytes,
                header_row_count=header_count,
            )
            table.rows = body
            table.warnings.extend(warnings)
            coverage = validate_unavailable_region_coverage(
                body, regions, cell_map
            )
            table.provenance["unavailable_regions"] = regions
            table.provenance["unavailable_region_coverage"] = coverage
            if not coverage.get("ok", True):
                table.warnings.append(
                    f"unavailable_region_coverage_issues:{coverage.get('issues')}"
                )

            confidence = (
                table.confidence * 100
                if table.confidence <= 1
                else table.confidence
            )
            span_map = cell_span_map(
                original_payload.get("cells_meta"), header_count
            )
            table.cells = build_cells(
                table.headers,
                table.rows,
                confidence=confidence,
                cell_region_map=cell_map,
                cell_bbox_map=cell_bbox_map(
                    original_payload.get("cells_meta"), header_count
                ),
                cell_span_map=span_map,
            )
            if span_map and any(
                v.get("colspan", 1) > 1 or v.get("rowspan", 1) > 1
                for v in span_map.values()
            ):
                table.provenance["fully_cellular"] = False
                table.provenance["has_spans"] = True
                table.provenance["cell_spans_available"] = True
            attach_self_contained_rows(table)
            logger.info(
                "STEP3 Python validate+export start table=%s", table.table_id
            )
            report = validate_table(table)
            result.tables.append(table)

            table_dir = tables_dir / table.table_id
            table_dir.mkdir(parents=True, exist_ok=True)
            write_json(table_dir / "original_extraction.json", original_payload)
            write_json(table_dir / "reconstructed.json", table.to_dict())
            write_json(
                table_dir / "universal.json",
                to_universal_dict(table, document_id=document_id),
            )
            write_html(table_dir / "reconstructed.html", table)
            write_csv(table_dir / "reconstructed.csv", table)
            write_json(table_dir / "validation_report.json", report)
            logger.info(
                "STEP3 exports written table=%s validation_ok=%s dir=%s",
                table.table_id,
                report.get("ok", report.get("passed")),
                table_dir,
            )
            try:
                preview = crop_preview_png(
                    page_obj, table.bbox.as_list(), dpi=150
                )
                (table_dir / "preview.png").write_bytes(preview)
            except Exception as exc:
                logger.debug("preview failed for %s: %s", table.table_id, exc)

        # Phase 1 (serial): reconstruct grids + render page images (PyMuPDF).
        pending: List[Dict[str, Any]] = []
        for raw in extraction.tables:
            table_index += 1
            table_id = f"table_{table_index:03d}"
            page_number = int(raw.page)
            page = doc[page_number - 1]
            caption = normalize_ws(str(raw.metadata.get("caption_hint") or ""))
            if not caption:
                caption = caption_near_region(_caption_lines(page), raw.bbox)
            need_image = use_bedrock or bool(raw.cells_meta)
            rendered = _page_render(page_number) if need_image else None
            image_bytes = rendered["bytes"] if rendered else None
            image_format = str(rendered["format"]) if rendered else "png"
            original_payload: Dict[str, Any] = {
                "table_id": table_id,
                "page": page_number,
                "pages": list(raw.pages),
                "bbox": raw.bbox.as_list(),
                "source": raw.source,
                "n_rows": raw.n_rows,
                "n_cols": raw.n_cols,
                "header_row_count": raw.header_row_count,
                "grid": [list(r) for r in raw.grid],
                "cells_meta": list(raw.cells_meta),
                "metadata": dict(raw.metadata or {}),
                "gate_reason": next(
                    (
                        g.reason
                        for g in gated_pages
                        if g.page == page_number
                    ),
                    "",
                ),
                "render": (
                    {
                        "image_format": rendered["format"],
                        "image_width": rendered["width"],
                        "image_height": rendered["height"],
                        "dpi": rendered["dpi"],
                        "image_bytes": len(rendered["bytes"]),
                    }
                    if rendered
                    else {"skipped": "no_image_required"}
                ),
                "backend": extraction.backend,
            }

            table = reconstruct_from_grid(
                raw.grid,
                table_id=table_id,
                page=page_number,
                bbox=raw.bbox,
                caption=caption,
                extraction_method=raw.source,
                confidence=raw.confidence,
                header_row_count=raw.header_row_count,
                provenance={
                    "source": raw.source,
                    "backend": extraction.backend,
                    "pages": list(raw.pages),
                    "document_id": document_id,
                    "header_row_count": raw.header_row_count,
                    "confidence_available": bool(
                        raw.metadata.get("confidence_available", True)
                    ),
                    "backend_metadata": dict(raw.metadata or {}),
                },
                document_id=document_id,
                cells_meta=raw.cells_meta,
                image_bytes=image_bytes,
            )
            if table is None:
                table_dir = tables_dir / table_id
                table_dir.mkdir(parents=True, exist_ok=True)
                write_json(table_dir / "original_extraction.json", original_payload)
                write_json(
                    table_dir / "validation_report.json",
                    {
                        "table_id": table_id,
                        "ok": False,
                        "issues": ["reconstruct_failed"],
                    },
                )
                result.errors.append(
                    f"table {table_id} on page {page_number}: reconstruct_failed"
                )
                continue
            table.pages = list(raw.pages or [page_number])
            pending.append(
                {
                    "table": table,
                    "original_payload": original_payload,
                    "image_bytes": image_bytes,
                    "image_format": image_format,
                    "page_obj": page,
                }
            )

        # Phase 2: Bedrock reconstruction (parallel when >1 table).
        bedrock_results: List[Tuple[int, ExtractedTable, Dict[str, Any]]] = []
        t_bedrock_start = time.perf_counter()
        if pending and use_bedrock and bedrock_workers > 1 and len(pending) > 1:
            workers = min(bedrock_workers, len(pending))
            logger.info(
                "STEP2 Bedrock parallel tables=%s workers=%s skip_simple=%s",
                len(pending),
                workers,
                skip_simple,
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _apply_bedrock_one,
                        item["table"],
                        item["image_bytes"],
                        item["image_format"],
                    ): idx
                    for idx, item in enumerate(pending)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    table_out, bmeta = fut.result()
                    bedrock_results.append((idx, table_out, bmeta))
            bedrock_results.sort(key=lambda x: x[0])
        else:
            for idx, item in enumerate(pending):
                table_out, bmeta = _apply_bedrock_one(
                    item["table"],
                    item["image_bytes"],
                    item["image_format"],
                )
                bedrock_results.append((idx, table_out, bmeta))
        t_bedrock = time.perf_counter() - t_bedrock_start

        # Phase 3 (serial): validate + export (stable order, shared result).
        t_export_start = time.perf_counter()
        for idx, table_out, bmeta in bedrock_results:
            item = pending[idx]
            bedrock_usage.append(bmeta)
            _export_one(
                table_out,
                item["original_payload"],
                item["image_bytes"],
                bmeta,
                item["page_obj"],
            )
        t_export = time.perf_counter() - t_export_start

        monitor.sample()
        metrics = monitor.stop()
        usage_summary = {
            "tables": bedrock_usage,
            "input_tokens": sum(
                int(u.get("input_tokens") or 0) for u in bedrock_usage
            ),
            "output_tokens": sum(
                int(u.get("output_tokens") or 0) for u in bedrock_usage
            ),
            "bedrock_wall_s": round(t_bedrock, 3),
            "export_wall_s": round(t_export, 3),
            "bedrock_workers": bedrock_workers if use_bedrock else 0,
            "bedrock_skip_simple": skip_simple,
            "bedrock_skipped_simple": sum(
                1
                for u in bedrock_usage
                if str(u.get("skipped") or "").startswith("simple:")
            ),
        }
        usage_summary["total_tokens"] = (
            usage_summary["input_tokens"] + usage_summary["output_tokens"]
        )
        result.meta["bedrock_usage"] = usage_summary
        result.meta["metrics"] = metrics
        result.meta["timings"] = {
            "backend_s": float((extraction.meta or {}).get("timings", {}).get("total_s") or extraction.elapsed_seconds or 0),
            "backend_detail": dict((extraction.meta or {}).get("timings") or {}),
            "bedrock_wall_s": round(t_bedrock, 3),
            "export_wall_s": round(t_export, 3),
        }

        facts = document_facts(str(input_path), page_count=len(sizes))
        cost = local_cost_summary(
            pages_processed=len(result.processed_pages),
            tables=len(result.tables),
        )
        cache_stats = cache.stats()

        write_json(
            output_dir / "document_metadata.json",
            {
                **facts,
                "source_path": result.source_path,
                "ocr_backend": result.ocr_backend,
                "extraction_mode": result.meta.get("extraction_mode"),
                "backend": resolved_backend,
                "backend_meta": result.meta.get("backend_meta"),
                "document_id": document_id,
                "bedrock_enabled": use_bedrock,
                "bedrock_model_id": model_id if use_bedrock else None,
                "processed_pages": result.processed_pages,
                "skipped_pages": result.skipped_pages,
                "table_count": len(result.tables),
                "settings": result.meta.get("settings"),
                "cache": result.meta.get("cache"),
            },
        )
        write_json(
            output_dir / "extraction_summary.json",
            {
                "source_path": result.source_path,
                "page_count": result.page_count,
                "processed_pages": result.processed_pages,
                "skipped_pages": result.skipped_pages,
                "processed_page_count": len(result.processed_pages),
                "skipped_page_count": len(result.skipped_pages),
                "table_count": len(result.tables),
                "ocr_backend": result.ocr_backend,
                "extraction_mode": result.meta.get("extraction_mode"),
                "backend": resolved_backend,
                "backend_meta": result.meta.get("backend_meta"),
                "document_id": document_id,
                "bedrock_enabled": use_bedrock,
                "bedrock_model_id": model_id if use_bedrock else None,
                "bedrock_usage": usage_summary,
                "cache": {**cache_stats, "hit": cache_hit, "key": cache_key},
                "metrics": metrics,
                "errors": result.errors,
                "warnings": list(extraction.warnings),
                "tables": [
                    {
                        "table_id": t.table_id,
                        "page": t.page,
                        "pages": list(t.pages),
                        "caption": t.caption,
                        "rows": len(t.rows),
                        "cols": len(t.headers),
                        "method": t.extraction_method,
                        "confidence": t.confidence,
                        "warnings": t.warnings,
                    }
                    for t in result.tables
                ],
                "gate": [
                    {
                        "page": g.page,
                        "has_tables": g.has_tables,
                        "reason": g.reason,
                        "native_table_count": g.native_table_count,
                    }
                    for g in result.gated_pages
                ],
            },
        )
        write_json(output_dir / "bedrock_usage.json", usage_summary)
        write_json(
            output_dir / "processing_metrics.json",
            {
                "document": facts,
                "metrics": metrics,
                "hardware": hardware_info(),
                "backend": result.meta.get("backend_meta"),
                "settings": result.meta.get("settings"),
                "cache": {**cache_stats, "hit": cache_hit, "key": cache_key},
                "cost": cost,
                "pages": {
                    "requested": sorted(page_filter) if page_filter else None,
                    "candidates": candidates,
                    "skipped": skipped_pages,
                },
                "tables": len(result.tables),
            },
        )
        return result
    finally:
        doc.close()


__all__ = ["extract_pdf"]

"""Textract AnalyzeDocument TABLES backend (page images → cellular grids).

Renders each PDF page to PNG/JPEG and calls ``AnalyzeDocument`` with
``FeatureTypes=["TABLES"]``. Never sends PDF bytes to Textract.

I/O-bound AnalyzeDocument calls are parallelized after serial gate+render
(PyMuPDF is not shared across threads).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ste.backends.base import DocumentExtraction, RawTable
from ste.models import BBox
from ste.multipage import stitch_multipage_tables
from ste.ocr_textract_tables import (
    analyze_document_tables,
    tables_from_analyze_blocks,
)
from ste.page_renderer import render_page_image
from ste.pdf_loader import open_pdf, page_sizes
from ste.table_page_gate import page_has_tables

logger = logging.getLogger(__name__)


def textract_available() -> bool:
    try:
        import boto3  # noqa: F401

        return True
    except Exception:
        return False


def _bbox_from_item(item: Dict[str, Any], width: float, height: float) -> BBox:
    bbox = item.get("bbox")
    if isinstance(bbox, BBox):
        return bbox
    coords = bbox if isinstance(bbox, (list, tuple)) else None
    if coords and len(coords) >= 4:
        return BBox(
            float(coords[0]),
            float(coords[1]),
            float(coords[2]),
            float(coords[3]),
        )
    return BBox(0.0, 0.0, float(width), float(height))


def _analyze_one_page(
    *,
    page_number: int,
    image_bytes: bytes,
    width: float,
    height: float,
    used_dpi: int,
    region: Optional[str],
) -> Tuple[int, List[RawTable], Optional[str], float]:
    """Return (page, tables, warning, analyze_seconds)."""
    t0 = time.perf_counter()
    try:
        resp = analyze_document_tables(image_bytes, region=region)
        blocks = list(resp.get("Blocks") or [])
        raw_tables = tables_from_analyze_blocks(
            blocks,
            page_number=page_number,
            page_width=float(width),
            page_height=float(height),
        )
        out: List[RawTable] = []
        for item in raw_tables:
            conf = float(item.get("confidence") or 0.0)
            if conf > 1.0:
                conf = conf / 100.0
            out.append(
                RawTable(
                    page=page_number,
                    bbox=_bbox_from_item(item, width, height),
                    grid=[list(r) for r in (item.get("grid") or [])],
                    source="textract_analyze_tables",
                    header_row_count=item.get("header_row_count"),
                    confidence=conf,
                    cells_meta=list(item.get("cells_meta") or []),
                    pages=[page_number],
                    metadata={
                        "backend": "textract",
                        "api": "AnalyzeDocument",
                        "feature_types": ["TABLES"],
                        "dpi": used_dpi,
                        "aws_region": region or "",
                        "page_width": float(width),
                        "page_height": float(height),
                    },
                )
            )
        logger.info(
            "STEP1 page=%s tables_found=%s blocks=%s analyze=%.2fs",
            page_number,
            len(out),
            len(blocks),
            time.perf_counter() - t0,
        )
        return page_number, out, None, time.perf_counter() - t0
    except Exception as exc:
        msg = f"textract_analyze:p{page_number}:{type(exc).__name__}: {exc}"
        logger.warning(msg)
        return page_number, [], msg, time.perf_counter() - t0


class TextractAnalyzeBackend:
    """Cloud Textract AnalyzeDocument:TABLES extraction backend."""

    name = "textract"

    def __init__(
        self,
        *,
        dpi: int = 200,
        aws_region: Optional[str] = None,
        tables_only: bool = True,
        stitch_multipage: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.dpi = int(dpi or 200)
        self.aws_region = aws_region
        self.tables_only = bool(tables_only)
        self.stitch_multipage = bool(stitch_multipage)
        self.max_workers = max(1, min(10, int(max_workers or 1)))

    def available(self) -> bool:
        return textract_available()

    def extract(
        self,
        pdf_path: str,
        *,
        pages: Optional[Sequence[int]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> DocumentExtraction:
        started = time.perf_counter()
        opts = dict(options or {})
        dpi = int(opts.get("dpi") or self.dpi)
        region = (
            str(opts.get("aws_region") or self.aws_region or "").strip() or None
        )
        tables_only = bool(opts.get("tables_only", self.tables_only))
        stitch = bool(opts.get("stitch_multipage", self.stitch_multipage))
        max_workers = max(
            1,
            min(10, int(opts.get("max_workers") or self.max_workers or 1)),
        )

        doc = open_pdf(pdf_path)
        wanted = {int(p) for p in pages} if pages else None
        tables: List[RawTable] = []
        warnings: List[str] = []
        processed: List[int] = []
        skipped: List[int] = []
        dims: Dict[int, Tuple[float, float]] = {}
        render_jobs: List[Dict[str, Any]] = []
        gate_s = 0.0
        render_s = 0.0
        analyze_s = 0.0

        try:
            # Serial gate + render (PyMuPDF safety). Parallelize AWS only.
            for page_number, width, height in page_sizes(doc):
                dims[page_number] = (float(width), float(height))
                if wanted is not None and page_number not in wanted:
                    continue
                page = doc[page_number - 1]
                if tables_only:
                    t_gate = time.perf_counter()
                    try:
                        gate = page_has_tables(page, page_number)
                        if not gate.has_tables:
                            skipped.append(page_number)
                            gate_s += time.perf_counter() - t_gate
                            continue
                    except Exception as exc:
                        warnings.append(
                            f"page_gate_error:p{page_number}:{type(exc).__name__}"
                        )
                    gate_s += time.perf_counter() - t_gate

                processed.append(page_number)
                try:
                    logger.info(
                        "STEP1 render page=%s dpi=%s (AnalyzeDocument queued)",
                        page_number,
                        dpi,
                    )
                    t_render = time.perf_counter()
                    image_bytes, _fmt, _iw, _ih, used_dpi = render_page_image(
                        page, dpi=dpi
                    )
                    render_s += time.perf_counter() - t_render
                    render_jobs.append(
                        {
                            "page_number": page_number,
                            "image_bytes": image_bytes,
                            "width": float(width),
                            "height": float(height),
                            "used_dpi": used_dpi,
                        }
                    )
                except Exception as exc:
                    msg = f"textract_render:p{page_number}:{type(exc).__name__}: {exc}"
                    warnings.append(msg)
                    logger.warning(msg)
        finally:
            doc.close()

        api_pages = 0
        if render_jobs:
            workers = min(max_workers, len(render_jobs))
            logger.info(
                "STEP1 AnalyzeDocument parallel pages=%s workers=%s",
                len(render_jobs),
                workers,
            )
            # Preserve page order when collecting results.
            by_page: Dict[int, List[RawTable]] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _analyze_one_page,
                        page_number=int(job["page_number"]),
                        image_bytes=job["image_bytes"],
                        width=float(job["width"]),
                        height=float(job["height"]),
                        used_dpi=int(job["used_dpi"]),
                        region=region,
                    )
                    for job in render_jobs
                ]
                for fut in as_completed(futures):
                    page_number, page_tables, warn, elapsed = fut.result()
                    api_pages += 1
                    analyze_s += float(elapsed)
                    if warn:
                        warnings.append(warn)
                    by_page[page_number] = page_tables
            for page_number in sorted(by_page):
                tables.extend(by_page[page_number])

        if stitch and tables:
            try:
                tables, stitch_warnings = stitch_multipage_tables(
                    tables, page_sizes=dims, enabled=True
                )
                warnings.extend(stitch_warnings)
                if stitch_warnings:
                    logger.info("STEP1 multipage stitch: %s", "; ".join(stitch_warnings))
            except Exception as exc:
                warnings.append(f"stitch_failed:{type(exc).__name__}: {exc}")

        total = time.perf_counter() - started
        logger.info(
            "STEP1 Textract done: tables=%s api_pages=%s workers=%s "
            "gate=%.2fs render=%.2fs analyze=%.2fs elapsed=%.2fs",
            len(tables),
            api_pages,
            max_workers,
            gate_s,
            render_s,
            analyze_s,
            total,
        )
        return DocumentExtraction(
            backend="textract",
            tables=tables,
            pages_processed=processed,
            warnings=warnings,
            meta={
                "api": "AnalyzeDocument",
                "feature_types": ["TABLES"],
                "analyze_document_tables_pages": api_pages,
                "skipped_pages": skipped,
                "dpi": dpi,
                "aws_region": region or "",
                "max_workers": max_workers,
                "timings": {
                    "gate_s": round(gate_s, 3),
                    "render_s": round(render_s, 3),
                    "analyze_s": round(analyze_s, 3),
                    "total_s": round(total, 3),
                },
            },
            elapsed_seconds=round(total, 3),
        )

#!/usr/bin/env python3
"""Live Docling / hybrid TableRAG extraction validation.

Usage:
  python scripts/live_docling_validate.py
  python scripts/live_docling_validate.py --pdf data/layout_probe/probe-table_pages.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _table_summary(table: Any) -> Dict[str, Any]:
    if hasattr(table, "to_json"):
        j = table.to_json(include_markdown=False)
    else:
        j = dict(table or {})
    headers = list(j.get("headers") or [])
    rows = [list(r) for r in (j.get("rows") or [])]
    return {
        "extraction_engine": j.get("extraction_engine") or j.get("extraction_method"),
        "extraction_method": j.get("extraction_method"),
        "page": j.get("page"),
        "pages": j.get("pages"),
        "headers": headers,
        "n_headers": len(headers),
        "n_rows": len(rows),
        "n_cols": j.get("n_cols") or len(headers),
        "structure_confidence": j.get("structure_confidence"),
        "ocr_confidence": j.get("ocr_confidence"),
        "needs_verification": j.get("needs_verification"),
        "merged_cells_count": len(j.get("merged_cells") or []),
        "empty_cell_ratio": _empty_ratio(headers, rows),
        "provenance": j.get("provenance") or {},
        "rows": rows[:80],
        "sample_rows": rows[:5],
        "markdown_preview": _md_preview(headers, rows),
    }


def _empty_ratio(headers: List[str], rows: List[List[str]]) -> float:
    width = max([len(headers)] + [len(r) for r in rows] or [0])
    if not rows or width == 0:
        return 1.0
    total = 0
    empty = 0
    for row in rows:
        padded = list(row) + [""] * max(0, width - len(row))
        for c in padded[:width]:
            total += 1
            if not str(c or "").strip():
                empty += 1
    return round(empty / total, 4) if total else 1.0


def _md_preview(headers: List[str], rows: List[List[str]], max_rows: int = 8) -> str:
    width = max([len(headers)] + [len(r) for r in rows] or [0])
    if width == 0:
        return ""
    h = (headers + [""] * width)[:width]
    lines = [
        "| " + " | ".join(str(c) for c in h) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in rows[:max_rows]:
        cells = (list(row) + [""] * width)[:width]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def extract_docling_only(pdf: Path) -> Dict[str, Any]:
    from app.rag.ingestion.pdf_layout.docling_tables import (
        extract_tables_from_pdf,
        reset_docling_converter,
    )
    from app.rag.ingestion.pdf_layout.extraction_quality import filter_quality_tables

    reset_docling_converter()
    started = time.perf_counter()
    error = ""
    tables: List[Any] = []
    try:
        tables = extract_tables_from_pdf(str(pdf))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    accepted, rejected, reports = filter_quality_tables(list(tables))
    return {
        "engine": "docling",
        "elapsed_s": round(time.perf_counter() - started, 3),
        "error": error,
        "raw_count": len(tables),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "quality_reports": reports,
        "accepted": [_table_summary(t) for t in accepted],
        "rejected": [_table_summary(t) for t in rejected],
        "raw": [_table_summary(t) for t in tables],
    }


def extract_textract_only(pdf: Path, max_pages: int = 3) -> Dict[str, Any]:
    """DetectDocumentText geometry path only (no AnalyzeDocument)."""
    import fitz

    from app.rag.ingestion.pdf_layout.extraction_quality import filter_quality_tables
    from app.rag.ingestion.pdf_layout.textract_tables import (
        extract_tables_from_page,
        usable_textract_tables,
    )

    started = time.perf_counter()
    error = ""
    pages_out: List[Dict[str, Any]] = []
    all_tables: List[Any] = []
    try:
        doc = fitz.open(str(pdf))
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            page_number = i + 1
            try:
                tables = usable_textract_tables(extract_tables_from_page(page, page_number))
            except Exception as exc:
                pages_out.append(
                    {
                        "page": page_number,
                        "error": f"{type(exc).__name__}: {exc}",
                        "n_tables": 0,
                    }
                )
                continue
            accepted, rejected, reports = filter_quality_tables(list(tables))
            all_tables.extend(tables)
            pages_out.append(
                {
                    "page": page_number,
                    "raw_count": len(tables),
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "quality_reports": reports,
                    "tables": [_table_summary(t) for t in tables],
                }
            )
        doc.close()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "engine": "textract_detect",
        "elapsed_s": round(time.perf_counter() - started, 3),
        "error": error,
        "pages": pages_out,
        "raw_count": len(all_tables),
        "raw": [_table_summary(t) for t in all_tables],
    }


def extract_hybrid(pdf: Path) -> Dict[str, Any]:
    from app.rag.ingestion.pdf_layout.parser import parse_pdf

    started = time.perf_counter()
    error = ""
    pages_out: List[Dict[str, Any]] = []
    try:
        parsed = parse_pdf(str(pdf))
        for page in getattr(parsed, "pages", []) or []:
            m = dict(getattr(page, "metrics", {}) or {})
            tables = list(getattr(page, "tables", []) or [])
            pages_out.append(
                {
                    "page": getattr(page, "page_number", None),
                    "table_source": m.get("table_source"),
                    "docling_tables_raw": m.get("docling_tables_raw"),
                    "docling_tables": m.get("docling_tables"),
                    "docling_rejected": m.get("docling_rejected"),
                    "docling_error": m.get("docling_error"),
                    "textract_invoked": m.get("textract_invoked"),
                    "textract_fallback_reason": m.get("textract_fallback_reason"),
                    "textract_tables": m.get("textract_tables"),
                    "textract_error": m.get("textract_error"),
                    "table_extraction_failed": m.get("table_extraction_failed"),
                    "table_extraction_uncertain": m.get("table_extraction_uncertain"),
                    "table_quality": m.get("table_quality"),
                    "textract_quality": m.get("textract_quality"),
                    "n_tables": len(tables),
                    "tables": [_table_summary(t) for t in tables],
                }
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return {
        "engine": "hybrid",
        "elapsed_s": round(time.perf_counter() - started, 3),
        "error": error,
        "n_pages": len(pages_out),
        "pages": pages_out,
    }


def validate_tablerag_path(hybrid: Dict[str, Any], pdf_name: str) -> Dict[str, Any]:
    """Chunk/index/persist path using hybrid page tables (in-memory)."""
    from app.rag.table_rag.indexing import expand_table_rag_chunks
    from app.rag.generation.table_calculator import compute_from_operands

    chunks: List[Dict[str, Any]] = []
    for page in hybrid.get("pages") or []:
        for idx, table in enumerate(page.get("tables") or []):
            headers = list(table.get("headers") or [])
            rows = [list(r) for r in (table.get("rows") or table.get("sample_rows") or [])]
            payload = {
                "table_id": f"live_{pdf_name}_p{page.get('page')}_{idx}",
                "headers": headers,
                "rows": rows,
                "page": page.get("page") or 1,
                "pages": table.get("pages") or [page.get("page") or 1],
                "extraction_engine": table.get("extraction_engine"),
                "extraction_method": table.get("extraction_method"),
                "structure_confidence": table.get("structure_confidence"),
                "ocr_confidence": table.get("ocr_confidence"),
                "needs_verification": table.get("needs_verification"),
                "provenance": table.get("provenance") or {},
                "merged_cells": [],
            }
            # Prefer full rows from markdown_preview is insufficient; stash sample.
            md = table.get("markdown_preview") or ""
            chunks.append(
                {
                    "id": f"c_{idx}",
                    "text": md or "| |",
                    "chunk_type": "table",
                    "content_type": "table",
                    "page": payload["page"],
                    "page_start": payload["page"],
                    "page_end": payload["page"],
                    "section": "LiveExtract",
                    "table_json": payload,
                    "extraction_method": payload.get("extraction_engine") or "docling",
                }
            )

    if not chunks:
        return {"ok": False, "reason": "no_tables_in_hybrid"}

    expanded, records = expand_table_rag_chunks(
        chunks,
        document_id=f"live-{pdf_name}",
        filename=pdf_name,
        enabled=True,
    )
    calc = None
    if records:
        table = records[0]["table"]
        rows = table.get("rows") or []
        # Try max over first numeric-looking column if ≥2 numeric cells.
        numeric_coords = []
        for r_i, row in enumerate(rows):
            for c_i, cell in enumerate(row):
                text = str(cell or "").replace(",", "").replace("$", "").strip()
                try:
                    float(text.rstrip("%"))
                    numeric_coords.append((r_i, c_i))
                    break
                except ValueError:
                    continue
            if len(numeric_coords) >= 3:
                break
        if len(numeric_coords) >= 2:
            try:
                calc_obj = compute_from_operands(
                    table,
                    numeric_coords[:3],
                    operation="max",
                    question="maximum value in table",
                )
                calc = {
                    "value": calc_obj.value,
                    "operation": calc_obj.operation,
                    "expression": calc_obj.expression,
                    "operands": [
                        {"r": o.row_index, "c": o.col_index, "raw": o.raw_value}
                        for o in calc_obj.operands
                    ],
                }
            except Exception as exc:
                calc = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "n_source_chunks": len(chunks),
        "n_expanded_chunks": len(expanded),
        "n_table_cards": sum(1 for c in expanded if c.get("chunk_type") == "table_card"),
        "n_records": len(records),
        "record_engines": [r.get("extraction_method") for r in records],
        "record_confidence": [r.get("confidence") for r in records],
        "record_needs_verification": [r.get("needs_verification") for r in records],
        "calculator": calc,
    }


DEFAULT_PDFS = [
    ROOT / "data" / "layout_probe" / "probe-table_pages.pdf",
    next((ROOT / "data" / "sample_subsets").glob("01_Vendor*.pdf"), None),
    ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="append", default=[])
    parser.add_argument(
        "--report-dir",
        default=str(ROOT / "data" / "layout_reports"),
    )
    parser.add_argument("--skip-textract-solo", action="store_true")
    parser.add_argument("--xolair-max-pages-note", action="store_true")
    args = parser.parse_args()

    pdfs: List[Path] = []
    if args.pdf:
        for p in args.pdf:
            path = Path(p)
            if not path.is_absolute():
                path = ROOT / path
            pdfs.append(path)
    else:
        for p in DEFAULT_PDFS:
            if p is not None:
                pdfs.append(Path(p))

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    master: Dict[str, Any] = {
        "docling_versions": {},
        "results": {},
    }
    try:
        import importlib.metadata as md

        master["docling_versions"] = {
            "docling": md.version("docling"),
            "docling-core": md.version("docling-core"),
            "docling-parse": md.version("docling-parse"),
            "docling-ibm-models": md.version("docling-ibm-models"),
        }
    except Exception as exc:
        master["docling_versions"] = {"error": str(exc)}

    from app.config import get_settings
    from app.rag.ingestion.pdf_layout.docling_tables import _tableformer_mode

    s = get_settings()
    master["config"] = {
        "PDF_DOCLING_TABLES_ENABLED": s.PDF_DOCLING_TABLES_ENABLED,
        "PDF_DOCLING_TABLEFORMER_MODE": s.PDF_DOCLING_TABLEFORMER_MODE,
        "configured_tableformer_mode": str(_tableformer_mode()),
        "PDF_TEXTRACT_TABLES_ENABLED": s.PDF_TEXTRACT_TABLES_ENABLED,
    }

    for pdf in pdfs:
        key = pdf.name
        print(f"\n=== {key} ({pdf}) size={pdf.stat().st_size if pdf.exists() else 'MISSING'} ===")
        if not pdf.exists():
            master["results"][key] = {"error": "pdf_missing", "path": str(pdf)}
            print("MISSING — not skipped silently")
            continue

        entry: Dict[str, Any] = {"path": str(pdf), "size_bytes": pdf.stat().st_size}
        print("Docling-only…")
        entry["docling"] = extract_docling_only(pdf)
        print(
            f"  raw={entry['docling']['raw_count']} "
            f"accepted={entry['docling']['accepted_count']} "
            f"rejected={entry['docling']['rejected_count']} "
            f"err={entry['docling']['error']!r} "
            f"elapsed={entry['docling']['elapsed_s']}s"
        )

        if not args.skip_textract_solo:
            # Limit Textract pages for large PDFs (cost/latency).
            max_pages = 2 if "xolair" in key.lower() else 5
            print(f"Textract-only (DetectDocumentText, max_pages={max_pages})…")
            entry["textract"] = extract_textract_only(pdf, max_pages=max_pages)
            print(
                f"  raw={entry['textract']['raw_count']} "
                f"err={entry['textract']['error']!r} "
                f"elapsed={entry['textract']['elapsed_s']}s"
            )

        print("Hybrid parse_pdf…")
        entry["hybrid"] = extract_hybrid(pdf)
        print(
            f"  pages={entry['hybrid']['n_pages']} "
            f"err={entry['hybrid']['error']!r} "
            f"elapsed={entry['hybrid']['elapsed_s']}s"
        )
        for page in entry["hybrid"].get("pages") or []:
            print(
                f"  page {page.get('page')}: source={page.get('table_source')} "
                f"n={page.get('n_tables')} "
                f"textract_invoked={page.get('textract_invoked')} "
                f"reason={page.get('textract_fallback_reason')} "
                f"failed={page.get('table_extraction_failed')} "
                f"uncertain={page.get('table_extraction_uncertain')}"
            )
            for t in page.get("tables") or []:
                print(f"    engine={t.get('extraction_engine')} "
                      f"headers={t.get('headers')[:6]} "
                      f"rows={t.get('n_rows')} cols={t.get('n_cols')} "
                      f"conf={t.get('structure_confidence')}")
                preview = t.get("markdown_preview") or ""
                if preview:
                    print(preview[:800])
                    print("---")

        entry["tablerag"] = validate_tablerag_path(entry["hybrid"], key)
        print(f"  tablerag={json.dumps(entry['tablerag'])[:500]}")

        out = report_dir / f"live_{pdf.stem}_extract.json"
        out.write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")
        print(f"  wrote {out}")
        master["results"][key] = entry

    master_path = report_dir / "live_docling_validate_master.json"
    master_path.write_text(json.dumps(master, indent=2, default=str), encoding="utf-8")
    print(f"\nMaster report: {master_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

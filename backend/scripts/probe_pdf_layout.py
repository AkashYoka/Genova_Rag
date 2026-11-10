"""Diagnostic probe for layout-aware PDF parsing (Gartley / multi-column).

Usage (from backend/):
  python scripts/probe_pdf_layout.py [--pdf PATH] [--pages 1,2,5,10] [--doc-id ID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import pymupdf as fitz  # noqa: E402

from app.rag.ingestion.pdf_layout.parser import parse_pdf, write_diagnostics  # noqa: E402


def _subset_pdf(source: Path, pages: list[int], dest: Path) -> Path:
    src = fitz.open(str(source))
    out = fitz.open()
    for page_number in pages:
        out.insert_pdf(src, from_page=page_number - 1, to_page=page_number - 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dest))
    out.close()
    src.close()
    return dest


def _reading_order_snippet(elements, limit: int = 8) -> list[dict]:
    rows = []
    for el in elements[:limit]:
        rows.append(
            {
                "type": el.element_type,
                "page": el.page_start,
                "columns": (el.metadata or {}).get("columns"),
                "section": el.section,
                "text": (el.text or "")[:220],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default=str(ROOT / "docs" / "gartley-hm-profits-3.pdf"),
    )
    parser.add_argument("--pages", default="1,2", help="1-based page numbers, comma-separated")
    parser.add_argument("--doc-id", default="probe-layout")
    parser.add_argument(
        "--out-dir",
        default=str(BACKEND / "data" / "layout_reports"),
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    pages = [int(p.strip()) for p in args.pages.split(",") if p.strip()]
    probe_pdf = BACKEND / "data" / "layout_probe" / f"{args.doc_id}_pages.pdf"
    _subset_pdf(pdf_path, pages, probe_pdf)

    parsed = parse_pdf(
        str(probe_pdf),
        document_id=args.doc_id,
        document_name=probe_pdf.name,
    )
    paths = write_diagnostics(parsed)

    # Extra reading-order / gutter diagnostic
    order_report = {
        "source_pdf": str(pdf_path),
        "probe_pdf": str(probe_pdf),
        "requested_pages": pages,
        "gutters": {
            str(page.page_number): page.metrics.get("gutter") for page in parsed.pages
        },
        "page_widths": {
            str(page.page_number): page.metrics.get("page_width") for page in parsed.pages
        },
        "ocr_confidence": {
            str(page.page_number): page.ocr_confidence for page in parsed.pages
        },
        "tables_per_page": {
            str(page.page_number): len(page.tables) for page in parsed.pages
        },
        "reading_order_preview": _reading_order_snippet(parsed.elements, limit=12),
        "interleave_checks": [],
    }

    # Heuristic: interleaved left/right fragments often contain both
    # "Management is the most important" and "grow, assuming" in one paragraph.
    for el in parsed.elements:
        if el.element_type != "paragraph":
            continue
        text = el.text or ""
        if "Management is the most important" in text and "grow, assuming" in text:
            order_report["interleave_checks"].append(
                {
                    "page": el.page_start,
                    "status": "FAIL",
                    "reason": "left/right column fragments merged into one paragraph",
                    "snippet": text[:300],
                }
            )

    if not order_report["interleave_checks"]:
        order_report["interleave_checks"].append(
            {"status": "PASS", "reason": "known Gartley page-1 interleave markers separated"}
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order_path = out_dir / f"{args.doc_id}_reading_order.json"
    order_path.write_text(json.dumps(order_report, indent=2), encoding="utf-8")

    print(json.dumps({"diagnostics": paths, "reading_order": str(order_path), "summary": {
        "pages": parsed.page_count,
        "elements": len(parsed.elements),
        "chunks": len(parsed.chunks),
        "gutters": order_report["gutters"],
        "interleave": order_report["interleave_checks"][0]["status"],
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

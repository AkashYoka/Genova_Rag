"""Dump DetectDocumentText geometry tables for xolair_prescribing.pdf.

Uses the same tabular-page gate + extract_tables_from_page path as ingest.
Writes scripts/_e2e_xolair_textract_tables.json — no index mutation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pymupdf as fitz

from app.rag.ingestion.pdf_layout.parser import _page_looks_tabular
from app.rag.ingestion.pdf_layout.textract_tables import (
    extract_tables_from_page,
    usable_textract_tables,
)

PDF = Path(__file__).resolve().parents[2] / "docs" / "xolair_prescribing.pdf"
OUT = Path(__file__).with_name("_e2e_xolair_textract_tables.json")
COST_PER_PAGE = 0.0015


def main() -> None:
    if not PDF.exists():
        raise SystemExit(f"missing PDF: {PDF}")

    doc = fitz.open(PDF)
    pages = []
    n_tables = 0
    n_textract_pages = 0
    t0 = time.perf_counter()
    for i in range(len(doc)):
        page = doc[i]
        page_no = i + 1
        looks, reason = _page_looks_tabular(page, fallback_tables=[])
        entry = {
            "page": page_no,
            "tabular_candidate": bool(looks),
            "tabular_reason": reason,
            "textract_ran": False,
            "tables": [],
        }
        if looks:
            n_textract_pages += 1
            tables = usable_textract_tables(extract_tables_from_page(page, page_no))
            entry["textract_ran"] = True
            for t in tables:
                n_tables += 1
                payload = t.to_json(include_markdown=True)
                # Keep dump readable: truncate huge markdown.
                md = payload.get("markdown") or ""
                if len(md) > 4000:
                    payload["markdown"] = md[:4000] + "\n… [truncated]"
                entry["tables"].append(payload)
        pages.append(entry)
        print(
            f"page {page_no}/{len(doc)} tabular={looks} "
            f"tables={len(entry['tables'])} reason={reason}"
        )

    doc.close()
    elapsed = time.perf_counter() - t0
    report = {
        "pdf": str(PDF),
        "page_count": len(pages),
        "textract_pages": n_textract_pages,
        "table_count": n_tables,
        "estimated_cost_usd": round(n_textract_pages * COST_PER_PAGE, 4),
        "elapsed_s": round(elapsed, 2),
        "pages": pages,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"wrote {OUT} tables={n_tables} textract_pages={n_textract_pages} "
        f"est_cost=${report['estimated_cost_usd']} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()

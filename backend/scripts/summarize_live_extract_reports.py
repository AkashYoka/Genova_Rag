#!/usr/bin/env python3
"""Build Docling / Textract / hybrid summary reports from live extract JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "layout_reports"


def _load(name: str) -> Dict[str, Any]:
    path = REPORT_DIR / name
    if not path.exists():
        return {"error": f"missing:{name}"}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_entry(entry: Dict[str, Any], pdf: str) -> Dict[str, Any]:
    docling = entry.get("docling") or {}
    textract = entry.get("textract") or {}
    hybrid = entry.get("hybrid") or {}
    pages = hybrid.get("pages") or []
    hybrid_engines = []
    for p in pages:
        for t in p.get("tables") or []:
            hybrid_engines.append(
                {
                    "page": p.get("page"),
                    "source": p.get("table_source"),
                    "engine": t.get("extraction_engine"),
                    "n_rows": t.get("n_rows"),
                    "n_cols": t.get("n_cols"),
                    "headers": (t.get("headers") or [])[:6],
                    "structure_confidence": t.get("structure_confidence"),
                    "needs_verification": t.get("needs_verification"),
                    "textract_invoked": p.get("textract_invoked"),
                    "fallback_reason": p.get("textract_fallback_reason"),
                }
            )
    return {
        "pdf": pdf,
        "docling": {
            "raw_count": docling.get("raw_count"),
            "accepted_count": docling.get("accepted_count"),
            "rejected_count": docling.get("rejected_count"),
            "elapsed_s": docling.get("elapsed_s"),
            "error": docling.get("error"),
            "quality_reports": docling.get("quality_reports"),
        },
        "textract": {
            "raw_count": textract.get("raw_count"),
            "elapsed_s": textract.get("elapsed_s"),
            "error": textract.get("error"),
            "pages": [
                {
                    "page": p.get("page"),
                    "raw_count": p.get("raw_count"),
                    "accepted_count": p.get("accepted_count"),
                    "rejected_count": p.get("rejected_count"),
                }
                for p in (textract.get("pages") or [])
            ],
        },
        "hybrid": {
            "n_pages": hybrid.get("n_pages"),
            "elapsed_s": hybrid.get("elapsed_s"),
            "error": hybrid.get("error"),
            "tables": hybrid_engines,
            "source_counts": _count_sources(pages),
        },
        "tablerag": entry.get("tablerag"),
        "note": (
            "Observed engine differences are reported; cell accuracy vs ground "
            "truth is not claimed unless a fixture expected table exists."
        ),
    }


def _count_sources(pages: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in pages:
        src = str(p.get("table_source") or "none")
        counts[src] = counts.get(src, 0) + 1
    return counts


def main() -> int:
    files = {
        "probe-table_pages.pdf": "live_probe-table_pages_extract.json",
        "01_Vendor_Invoice_Goods_CGST_SGST.pdf": "live_01_Vendor_Invoice_Goods_CGST_SGST_extract.json",
        "xolair_prescribing.pdf": "live_xolair_prescribing_extract.json",
    }
    docling_report = {"engine": "docling_primary", "pdfs": {}}
    textract_report = {"engine": "textract_detect_solo", "pdfs": {}}
    hybrid_report = {"engine": "hybrid_parse_pdf", "pdfs": {}}

    for pdf, fname in files.items():
        entry = _load(fname)
        if entry.get("error"):
            for report in (docling_report, textract_report, hybrid_report):
                report["pdfs"][pdf] = entry
            continue
        summary = summarize_entry(entry, pdf)
        docling_report["pdfs"][pdf] = summary["docling"]
        textract_report["pdfs"][pdf] = summary["textract"]
        hybrid_report["pdfs"][pdf] = {
            "hybrid": summary["hybrid"],
            "tablerag": summary["tablerag"],
        }

    out_paths = {
        "bench_docling_primary.json": docling_report,
        "bench_textract_fallback.json": textract_report,
        "bench_hybrid_final.json": hybrid_report,
        "live_docling_vs_textract_comparison.json": {
            pdf: summarize_entry(_load(fname), pdf)
            for pdf, fname in files.items()
        },
    }
    for name, payload in out_paths.items():
        path = REPORT_DIR / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

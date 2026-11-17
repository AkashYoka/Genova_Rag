"""Micro-benchmark: layout table extraction wall time (STE + optional Detect).

Usage (PowerShell):
  cd backend
  python scripts/bench_ste_layout_perf.py
  python scripts/bench_ste_layout_perf.py --pdf data/sample_subsets/xolair_prescribing.pdf --pages 5,6,19
  python scripts/bench_ste_layout_perf.py --no-cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Bench STE layout table extraction")
    parser.add_argument(
        "--pdf",
        default=str(ROOT / "data" / "uploads" / "f3d989fb-79af-4fe2-9360-12b06772348c" / "Genova_AI_v1_Review.pdf"),
    )
    parser.add_argument("--pages", default="", help="STE page subset, e.g. 5,6,19")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--keep-detect",
        action="store_true",
        help="Force DetectDocumentText even when STE primary",
    )
    parser.add_argument("--label", default="bench")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"PDF not found: {pdf}")
        return 1

    # Env must be set before settings load.
    os.environ.setdefault("PDF_STE_TABLES_ENABLED", "true")
    os.environ["PDF_STE_SKIP_DETECT_WHEN_STE"] = (
        "false" if args.keep_detect else "true"
    )
    if args.no_cache:
        os.environ["PDF_STE_USE_CACHE"] = "false"
    if args.pages:
        os.environ["PDF_STE_PAGES"] = args.pages

    import pymupdf as fitz

    from app.config import get_settings
    from app.rag.ingestion.pdf_layout.table_extraction import (
        run_document_table_extraction,
    )

    get_settings.cache_clear()
    settings = get_settings()

    doc = fitz.open(str(pdf))
    page_sizes = {
        i + 1: (float(p.rect.width), float(p.rect.height))
        for i, p in enumerate(doc)
    }
    t0 = time.perf_counter()
    try:
        plan = run_document_table_extraction(
            str(pdf),
            page_sizes=page_sizes,
            doc=doc,
            document_id=f"bench_{args.label}",
            document_name=pdf.name,
            settings=settings,
        )
    finally:
        doc.close()
    wall = round(time.perf_counter() - t0, 3)
    report = {
        "label": args.label,
        "pdf": str(pdf),
        "pages": args.pages or "all",
        "wall_s": wall,
        "plan": plan.to_report(),
        "flags": {
            "PDF_STE_SKIP_DETECT_WHEN_STE": getattr(
                settings, "PDF_STE_SKIP_DETECT_WHEN_STE", None
            ),
            "PDF_STE_MAX_WORKERS": getattr(settings, "PDF_STE_MAX_WORKERS", None),
            "PDF_STE_BEDROCK_MAX_WORKERS": getattr(
                settings, "PDF_STE_BEDROCK_MAX_WORKERS", None
            ),
            "PDF_STE_DPI": getattr(settings, "PDF_STE_DPI", None),
            "PDF_STE_BEDROCK_SKIP_SIMPLE": getattr(
                settings, "PDF_STE_BEDROCK_SKIP_SIMPLE", None
            ),
            "PDF_STE_USE_CACHE": getattr(settings, "PDF_STE_USE_CACHE", None),
        },
    }
    out_dir = ROOT / "data" / "evals" / "ste_perf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build demo HTML from the latest full-doc STE run (no AWS re-call)."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOC_ID = "43ae024f-3c0d-40f5-aa49-83e9f3682641"


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "demo_ste_xolair_pipeline",
        ROOT / "scripts" / "demo_ste_xolair_pipeline.py",
    )
    demo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = demo
    spec.loader.exec_module(demo)

    runs = sorted(
        (ROOT / "data" / "ste_runs" / DOC_ID).glob("run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        print("No STE runs found")
        return 1
    run_dir = runs[0]

    reports = sorted(
        (ROOT / "data" / "layout_reports").glob("xolair_ste_reindex_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    reindex = json.loads(reports[0].read_text(encoding="utf-8")) if reports else {}

    def _load(name: str):
        path = run_dir / name
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    meta = _load("extraction_summary.json")
    bedrock = _load("bedrock_usage.json")
    doc_meta = _load("document_metadata.json")
    tables = demo._collect_tables(run_dir)

    capture = demo.StepCapture()
    now = time.time()
    tokens = bedrock.get("total_tokens")
    if tokens is None:
        tokens = (bedrock.get("totals") or {}).get("total_tokens")
    for msg, step in [
        (
            "Full PDF scanned (72 pages); tables_only gate selected tabular candidates",
            1,
        ),
        (
            f"STEP1 AnalyzeDocument TABLES api_pages=19 tables={len(tables)} "
            "(non-tabular pages skipped)",
            1,
        ),
        ("STEP2 Bedrock reconstruction applied on extracted tables", 2),
        (f"STEP2 bedrock_tokens={tokens}", 2),
        ("STEP3 Python validate + HTML/JSON/CSV exports written per table", 3),
        (
            "Re-index ready: chroma="
            f"{(reindex.get('inspection') or {}).get('chroma_total_chunks')} "
            f"sql_tables={(reindex.get('inspection') or {}).get('sql_table_count')}",
            3,
        ),
    ]:
        capture.events.append(demo.LogEvent(now, "INFO", "demo_ste_xolair", msg, step))

    run_meta = {
        "run_dir": str(run_dir),
        "elapsed_s": meta.get("elapsed_s") or reindex.get("ingest_elapsed_s"),
        "table_count": len(tables),
        "page_count": 72,
        "processed_pages": doc_meta.get("processed_pages")
        or meta.get("processed_pages")
        or [],
        "skipped_pages": doc_meta.get("skipped_pages")
        or meta.get("skipped_pages")
        or [],
        "errors": [],
        "backend": "textract",
        "meta": {
            "backend_meta": {"analyze_document_tables_pages": 19},
            "bedrock_usage": bedrock,
        },
        "region": "us-east-1",
        "model_id": "us.amazon.nova-2-lite-v1:0",
        "pages": "ALL (tabular-gated)",
    }
    html = demo._render_demo_html(
        run_meta=run_meta,
        tables=tables,
        capture=capture,
        reindex_report={
            "document_id": reindex.get("document_id") or DOC_ID,
            "status": reindex.get("status"),
            "chunk_count": reindex.get("chunk_count"),
            "ingest_elapsed_s": reindex.get("ingest_elapsed_s"),
            "inspection": reindex.get("inspection") or {},
        },
    )
    out = (
        ROOT
        / "data"
        / "layout_reports"
        / f"xolair_ste_demo_full_{int(time.time())}.html"
    )
    out.write_text(html, encoding="utf-8")
    print(out)
    print(f"tables={len(tables)} run={run_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

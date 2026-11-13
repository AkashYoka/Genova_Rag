#!/usr/bin/env python3
"""Evaluate the live Genova TableRAG pipeline with RAGAS + table metrics.

Usage:
  python scripts/evaluate_ragas.py
  python scripts/evaluate_ragas.py --dataset medical --limit 2
  python scripts/evaluate_ragas.py --skip-llm-metrics   # table metrics only

Env:
  PDF_TEXTRACT_FALLBACK_ENABLED=false   # Docling-only extraction phase
  RAGAS_DATASET_DIR=data/evals/ragas/datasets
  RAGAS_REPORT_DIR=data/evals/ragas/reports
  RAGAS_ENABLED=true
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    # Install langchain_community Vertex AI stubs before any ragas import.
    from app.rag.evaluation.ragas_compat import ensure_ragas_langchain_compat

    ensure_ragas_langchain_compat()

    from app.config import get_settings
    from app.rag.evaluation.runner import evaluate_dataset, merge_overall_report

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset name (stem) under RAGAS_DATASET_DIR; repeatable. Default: all *.json",
    )
    parser.add_argument("--user-id", default="", help="Override dataset user_id")
    parser.add_argument("--limit", type=int, default=0, help="Max cases per dataset")
    parser.add_argument(
        "--skip-llm-metrics",
        action="store_true",
        help="Skip RAGAS LLM judge metrics; still run table-specific metrics",
    )
    parser.add_argument(
        "--report-dir",
        default="",
        help="Override RAGAS_REPORT_DIR",
    )
    parser.add_argument("--llm-provider", default=None)
    args = parser.parse_args()

    if not bool(getattr(settings, "RAGAS_ENABLED", True)):
        print("RAGAS_ENABLED=false — evaluation disabled.")
        return 2

    fallback = bool(getattr(settings, "PDF_TEXTRACT_FALLBACK_ENABLED", False))
    print(
        f"PDF_TEXTRACT_FALLBACK_ENABLED={fallback} "
        f"(Docling-only expected for this phase)"
    )
    if fallback:
        print(
            "WARNING: Textract fallback is enabled. "
            "Set PDF_TEXTRACT_FALLBACK_ENABLED=false for Docling-only testing."
        )

    dataset_dir = Path(
        getattr(settings, "RAGAS_DATASET_DIR", "data/evals/ragas/datasets")
    )
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    report_dir = Path(
        args.report_dir
        or getattr(settings, "RAGAS_REPORT_DIR", "data/evals/ragas/reports")
    )
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        paths = []
        for name in args.datasets:
            p = Path(name)
            if not p.suffix:
                p = dataset_dir / f"{name}.json"
            elif not p.is_absolute():
                p = dataset_dir / p.name if p.parent == Path(".") else ROOT / p
            paths.append(p)
    else:
        paths = sorted(dataset_dir.glob("*.json"))

    reports = []
    for path in paths:
        if not path.exists():
            print(f"SKIP missing dataset: {path}")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        cases = raw.get("cases") if isinstance(raw, dict) else raw
        if not cases:
            print(f"SKIP empty dataset: {path.name}")
            continue
        print(f"→ Evaluating {path.name} ({len(cases)} cases)")
        report = evaluate_dataset(
            path,
            user_id=args.user_id or None,
            llm_provider=args.llm_provider,
            limit=args.limit,
            skip_llm_metrics=args.skip_llm_metrics,
            report_dir=report_dir,
        )
        reports.append(report)
        print(
            json.dumps(
                {
                    "dataset": report.get("dataset"),
                    "n": report.get("n_questions"),
                    "ok": report.get("successful_queries"),
                    "fail": report.get("failed_queries"),
                    "ragas_status": (report.get("retrieval_and_generation_metrics") or {}).get(
                        "status"
                    ),
                    "ragas_scores": (report.get("retrieval_and_generation_metrics") or {}).get(
                        "scores"
                    ),
                    "report": report.get("report_path"),
                },
                indent=2,
            )
        )

    if not reports:
        print("No datasets evaluated.")
        return 1

    overall = merge_overall_report(reports, report_dir=report_dir)
    print(f"Wrote overall report: {overall.get('report_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

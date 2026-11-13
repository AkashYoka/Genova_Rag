#!/usr/bin/env python3
"""Table extraction / cell-level benchmark harness.

Compares extracted grids to optional ground-truth JSON fixtures.

Usage (unit / fixture mode — no Docling/AWS required):
  python scripts/bench_table_extraction.py --fixture-dir data/evals/table_extraction

Usage (live PDF — requires Docling and/or Textract):
  python scripts/bench_table_extraction.py \\
      --pdf data/sample_subsets/probe-table_pages.pdf \\
      --report data/layout_reports/table_extract_bench.json

Fixture JSON schema (one file per case):
  {
    "id": "finance_simple",
    "domain": "finance",
    "pdf": "optional/relative/path.pdf",
    "expected": {
      "headers": ["Year", "Revenue"],
      "rows": [["2024", "10"]]
    },
    "extracted": { ... }   // optional; used when PDF/engine unavailable
  }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def cell_metrics(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> Dict[str, Any]:
    """Cell-level correctness vs ground truth (headers + body)."""
    exp_h = [_norm(h) for h in (expected.get("headers") or [])]
    act_h = [_norm(h) for h in (actual.get("headers") or [])]
    exp_rows = [[_norm(c) for c in row] for row in (expected.get("rows") or [])]
    act_rows = [[_norm(c) for c in row] for row in (actual.get("rows") or [])]

    header_ok = exp_h == act_h
    width = max(len(exp_h), len(act_h), 1)
    total = 0
    matched = 0
    for r in range(max(len(exp_rows), len(act_rows))):
        e = list(exp_rows[r]) if r < len(exp_rows) else []
        a = list(act_rows[r]) if r < len(act_rows) else []
        e += [""] * max(0, width - len(e))
        a += [""] * max(0, width - len(a))
        for c in range(width):
            total += 1
            if e[c] == a[c]:
                matched += 1
    cell_acc = (matched / total) if total else 0.0
    row_rel_ok = len(exp_rows) == len(act_rows) and all(
        len(e) == len(a) or True for e, a in zip(exp_rows, act_rows)
    )
    return {
        "header_correct": header_ok,
        "cell_accuracy": round(cell_acc, 4),
        "cells_matched": matched,
        "cells_total": total,
        "row_count_match": len(exp_rows) == len(act_rows),
        "col_count_match": len(exp_h) == len(act_h),
        "row_col_relationship_ok": header_ok and cell_acc >= 0.99,
        "engine": actual.get("extraction_engine") or actual.get("extraction_method") or "",
        "structure_confidence": actual.get("structure_confidence"),
    }


def _load_fixtures(fixture_dir: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    if not fixture_dir.is_dir():
        return cases
    for path in sorted(fixture_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            cases.append({"id": path.stem, "error": f"fixture_load: {exc}"})
            continue
        data.setdefault("id", path.stem)
        data["_path"] = str(path)
        cases.append(data)
    return cases


def _extract_from_pdf(pdf: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run hybrid parser tables; returns (table dicts, meta)."""
    meta: Dict[str, Any] = {"pdf": str(pdf), "docling_available": False}
    try:
        import docling  # noqa: F401

        meta["docling_available"] = True
    except ImportError:
        meta["docling_available"] = False

    started = time.perf_counter()
    from app.rag.ingestion.pdf_layout.parser import parse_pdf

    doc = parse_pdf(str(pdf))
    tables: List[Dict[str, Any]] = []
    for page in getattr(doc, "pages", []) or []:
        for table in getattr(page, "tables", []) or []:
            payload = table.to_json(include_markdown=False)
            payload["page_metrics"] = dict(getattr(page, "metrics", {}) or {})
            tables.append(payload)
    meta["elapsed_s"] = round(time.perf_counter() - started, 3)
    meta["n_tables"] = len(tables)
    meta["engines"] = sorted(
        {
            str(t.get("extraction_engine") or t.get("extraction_method") or "")
            for t in tables
            if t.get("extraction_engine") or t.get("extraction_method")
        }
    )
    return tables, meta


def run_fixture_cases(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for case in cases:
        if case.get("error"):
            results.append(case)
            continue
        expected = case.get("expected") or {}
        actual = case.get("extracted") or {}
        pdf_rel = case.get("pdf")
        live_meta = None
        if pdf_rel and not actual:
            pdf_path = (ROOT / pdf_rel).resolve()
            if pdf_path.is_file():
                try:
                    tables, live_meta = _extract_from_pdf(pdf_path)
                    actual = tables[0] if tables else {}
                except Exception as exc:
                    results.append(
                        {
                            "id": case.get("id"),
                            "domain": case.get("domain"),
                            "error": f"extract_failed: {type(exc).__name__}: {exc}",
                            "verified": False,
                        }
                    )
                    continue
            else:
                results.append(
                    {
                        "id": case.get("id"),
                        "domain": case.get("domain"),
                        "error": f"pdf_missing: {pdf_rel}",
                        "verified": False,
                    }
                )
                continue
        if not expected:
            results.append(
                {
                    "id": case.get("id"),
                    "domain": case.get("domain"),
                    "error": "missing_expected",
                    "verified": False,
                }
            )
            continue
        metrics = cell_metrics(expected, actual)
        results.append(
            {
                "id": case.get("id"),
                "domain": case.get("domain"),
                "metrics": metrics,
                "live": live_meta,
                "verified": True,
                "source": "live_pdf" if live_meta else "fixture_extracted",
            }
        )
    scored = [r for r in results if r.get("verified") and r.get("metrics")]
    header_acc = (
        sum(1 for r in scored if r["metrics"]["header_correct"]) / len(scored)
        if scored
        else None
    )
    cell_acc = (
        sum(r["metrics"]["cell_accuracy"] for r in scored) / len(scored)
        if scored
        else None
    )
    return {
        "n_cases": len(results),
        "n_scored": len(scored),
        "header_accuracy": header_acc,
        "mean_cell_accuracy": cell_acc,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        default=str(ROOT / "data" / "evals" / "table_extraction"),
        help="Directory of fixture JSON cases",
    )
    parser.add_argument("--pdf", default="", help="Optional single PDF to extract")
    parser.add_argument(
        "--report",
        default="",
        help="Write JSON report path",
    )
    args = parser.parse_args()

    report: Dict[str, Any] = {"mode": []}
    if args.pdf:
        pdf = Path(args.pdf)
        if not pdf.is_file():
            pdf = ROOT / args.pdf
        tables, meta = _extract_from_pdf(pdf)
        report["mode"].append("live_pdf")
        report["live_pdf"] = {"meta": meta, "tables": tables}
        print(json.dumps({"live_pdf": meta, "n_tables": len(tables)}, indent=2))

    fixture_dir = Path(args.fixture_dir)
    if not fixture_dir.is_absolute():
        fixture_dir = ROOT / fixture_dir
    cases = _load_fixtures(fixture_dir)
    if cases:
        report["mode"].append("fixtures")
        fixture_report = run_fixture_cases(cases)
        report["fixtures"] = fixture_report
        print(
            json.dumps(
                {
                    "fixtures": {
                        "n_cases": fixture_report["n_cases"],
                        "n_scored": fixture_report["n_scored"],
                        "header_accuracy": fixture_report["header_accuracy"],
                        "mean_cell_accuracy": fixture_report["mean_cell_accuracy"],
                    }
                },
                indent=2,
            )
        )
    elif not args.pdf:
        print("No fixtures found and no --pdf provided.", file=sys.stderr)
        return 2

    if args.report:
        out = Path(args.report)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

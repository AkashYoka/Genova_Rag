#!/usr/bin/env python3
"""End-to-end TableRAG eval against the live retrieve → interpret → agent path.

Usage:
  python scripts/eval_table_rag_e2e.py --dataset data/evals/table_rag_golden.json \\
      --user-id <uuid> --report data/layout_reports/table_rag_e2e_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _resolve_document_id(
    *,
    user_id: str,
    document_id: Optional[str],
    original_name: Optional[str],
) -> Optional[str]:
    if document_id:
        return document_id
    if not original_name:
        return None
    from app.database import SessionLocal
    from app.models import Document

    db = SessionLocal()
    try:
        needle = original_name.lower()
        rows = (
            db.query(Document)
            .filter(Document.user_id == user_id, Document.is_deleted.is_(False))
            .all()
        )
        for row in rows:
            if needle in str(row.original_name or "").lower():
                return str(row.id)
    finally:
        db.close()
    return None


def _value_match(expected: Any, actual: Any) -> bool:
    if expected is None:
        return True
    exp = str(expected).strip().lower().replace(",", "")
    act = str(actual or "").strip().lower().replace(",", "")
    if not exp:
        return True
    if exp in act:
        return True
    # Numeric tolerance.
    try:
        return abs(float(exp) - float(act)) < 1e-6
    except (TypeError, ValueError):
        return False


def run_case(case: Dict[str, Any], *, user_id: str) -> Dict[str, Any]:
    from app.rag.generation.table_interpretation import interpret_table_question
    from app.rag.llm.agent import generate_answer
    from app.rag.retrieval.retriever import retrieve
    from app.rag.table_rag.representations import render_indexed_table_for_selector
    from app.rag.table_rag.telemetry import (
        begin_table_rag_trace,
        clear_table_rag_trace,
        current_table_rag_trace,
    )

    question = str(case.get("question") or "")
    doc_id = _resolve_document_id(
        user_id=user_id,
        document_id=case.get("document_id"),
        original_name=case.get("original_name") or case.get("document"),
    )
    begin_table_rag_trace(question=question, document_id=str(doc_id or ""))
    t0 = time.perf_counter()
    chunks = retrieve(
        query=question,
        user_id=user_id,
        document_id=doc_id,
        top_k=case.get("top_k") or 8,
    )
    t_retrieve = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    interpretation = interpret_table_question(question, chunks)
    t_selector = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    result = generate_answer(
        question=question,
        user_id=user_id,
        document_id=doc_id,
        top_k=case.get("top_k") or 8,
    )
    t_e2e = (time.perf_counter() - t2) * 1000.0
    # generate_answer begins its own trace; prefer post-answer sources.
    sources = result.get("sources") or []
    verified = None
    for src in sources:
        if isinstance(src.get("verified_cell"), dict):
            verified = src["verified_cell"]
            break
    answer = str(result.get("answer") or "")
    expected_value = case.get("expected_value")
    expected_status = case.get("expected_selector_status")
    selector_status = ""
    for src in sources:
        if src.get("selector_status"):
            selector_status = str(src.get("selector_status"))
            break
    if not selector_status and verified:
        selector_status = "matched" if not verified.get("calculation") else "requires_calculation"
    if not selector_status:
        trace = current_table_rag_trace()
        if trace is not None:
            selector_status = trace.selector_status
            t_selector = trace.selector_latency_ms or t_selector

    actual_value = None
    if verified:
        actual_value = verified.get("value")
    cell_ok = _value_match(expected_value, actual_value) if expected_value is not None else None
    answer_ok = _value_match(expected_value, answer) if expected_value is not None else None
    status_ok = (
        True
        if not expected_status
        else str(expected_status).lower() in str(selector_status).lower()
    )
    passed = True
    if cell_ok is False and answer_ok is False:
        passed = False
    # Prefer answer correctness when the selector did not emit a status
    # (legacy/narrative path still produced the right value).
    if expected_status and not status_ok:
        if answer_ok is True and not selector_status:
            pass
        else:
            passed = False
    if case.get("expect_abstain") and "ambiguous" not in answer.lower() and "insufficient" not in answer.lower():
        # Soft check — only fail when verified_cell was incorrectly pinned.
        if verified and not verified.get("calculation"):
            passed = False

    rendered = None
    if not passed:
        for chunk in chunks:
            table = chunk.get("table")
            if isinstance(table, dict) and table.get("rows"):
                try:
                    rendered = render_indexed_table_for_selector(
                        {
                            "table_id": chunk.get("table_id") or table.get("table_id"),
                            "table": table,
                            "pages": table.get("pages"),
                            "section": chunk.get("section"),
                            "caption": table.get("caption"),
                        }
                    )
                except Exception:
                    rendered = None
                break

    clear_table_rag_trace()
    return {
        "id": case.get("id") or question[:40],
        "question": question,
        "document_id": doc_id,
        "original_name": case.get("original_name") or case.get("document"),
        "passed": passed,
        "expected_value": expected_value,
        "actual_value": actual_value,
        "expected_selector_status": expected_status,
        "selector_status": selector_status,
        "legacy_fallback": selector_status == "legacy_fallback",
        "reconcile_action": next(
            (s.get("reconcile_action") for s in sources if s.get("reconcile_action")),
            None,
        ),
        "answer_preview": answer[:400],
        "latency_ms": {
            "retrieve": round(t_retrieve, 2),
            "selector": round(t_selector, 2),
            "e2e_generate": round(t_e2e, 2),
        },
        "rendered_selector_input": rendered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--report", default=str(ROOT / "data" / "layout_reports" / "table_rag_e2e_report.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = dataset.get("cases") if isinstance(dataset, dict) else dataset
    if args.limit:
        cases = cases[: args.limit]

    results = []
    for case in cases:
        print(f"→ {case.get('id') or case.get('question', '')[:60]}")
        try:
            results.append(run_case(case, user_id=args.user_id))
        except Exception as exc:
            results.append(
                {
                    "id": case.get("id"),
                    "question": case.get("question"),
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    e2e_lat = [r.get("latency_ms", {}).get("e2e_generate", 0) for r in results if r.get("latency_ms")]
    sel_lat = [r.get("latency_ms", {}).get("selector", 0) for r in results if r.get("latency_ms")]
    status_counts: Dict[str, int] = {}
    for r in results:
        key = str(r.get("selector_status") or "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1

    summary = {
        "n": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "failed": sum(1 for r in results if not r.get("passed")),
        "selector_status_breakdown": status_counts,
        "legacy_fallback_count": sum(1 for r in results if r.get("legacy_fallback")),
        "latency_ms": {
            "e2e_p50": round(_percentile(e2e_lat, 50), 2),
            "e2e_p95": round(_percentile(e2e_lat, 95), 2),
            "selector_p50": round(_percentile(sel_lat, 50), 2),
            "selector_p95": round(_percentile(sel_lat, 95), 2),
            "e2e_mean": round(statistics.fmean(e2e_lat), 2) if e2e_lat else 0.0,
        },
        "results": results,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Wrote {out}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

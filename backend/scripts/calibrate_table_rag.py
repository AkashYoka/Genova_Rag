#!/usr/bin/env python3
"""Calibrate TABLE_RAG_REWRITE_FALLBACK_THRESHOLD on the RRF score scale.

RRF with k=60 yields scores ≈ 1/(60+rank) ∈ ~0.015–0.033 (not cosine 0–1).

Usage:
  python scripts/calibrate_table_rag.py --queries queries.jsonl --user-id <uuid>
  python scripts/calibrate_table_rag.py --from-sql --user-id <uuid> [--document-id <id>]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_queries(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                out.append(obj)
            elif isinstance(obj, dict):
                out.append(str(obj.get("question") or obj.get("query") or ""))
        return [q for q in out if q.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item.get("question") or item.get("query") or ""))
        return [q for q in out if q.strip()]
    raise SystemExit("queries file must be JSON list or JSONL")


def _queries_from_sql(*, user_id: str, document_id: Optional[str]) -> List[str]:
    from app.database import SessionLocal
    from app.models import DocumentTable

    db = SessionLocal()
    try:
        q = db.query(DocumentTable).filter(DocumentTable.user_id == user_id)
        if document_id:
            q = q.filter(DocumentTable.document_id == document_id)
        rows = q.all()
    finally:
        db.close()

    queries: List[str] = []
    for row in rows:
        raw = getattr(row, "llm_understanding_json", None)
        if not raw:
            continue
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            continue
        for pat in (data or {}).get("suggested_qa_patterns") or []:
            if isinstance(pat, dict) and pat.get("question_pattern"):
                queries.append(str(pat["question_pattern"]))
            elif isinstance(pat, str) and pat.strip():
                queries.append(pat.strip())
    # Deduplicate while preserving order.
    return list(dict.fromkeys(q for q in queries if q.strip()))


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=str, default="", help="JSON/JSONL query file")
    parser.add_argument("--from-sql", action="store_true", help="Pull suggested_qa_patterns from SQL")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--document-ids", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--target-percentile", type=float, default=17.5, help="~P15–P20")
    args = parser.parse_args()

    if args.from_sql:
        queries = _queries_from_sql(user_id=args.user_id, document_id=args.document_id)
    elif args.queries:
        queries = _load_queries(Path(args.queries))
    else:
        raise SystemExit("Provide --queries or --from-sql")

    if not queries:
        raise SystemExit("No queries found")

    from app.rag.retrieval import retriever as retriever_mod

    # Force rewrite OFF so we measure raw top-1 RRF.
    scores: List[float] = []
    details: List[Dict[str, Any]] = []
    for q in queries:
        chunks = retriever_mod.retrieve(
            query=q,
            user_id=args.user_id,
            document_id=args.document_id,
            document_ids=args.document_ids,
            top_k=args.top_k,
        )
        top1 = 0.0
        if chunks:
            top1 = float(
                chunks[0].get("_top1_rrf_score")
                or chunks[0].get("score")
                or 0.0
            )
        scores.append(top1)
        details.append({"question": q[:120], "top1_rrf_score": top1})

    ranked = sorted(scores)
    p10 = _percentile(ranked, 10)
    p20 = _percentile(ranked, 20)
    p50 = _percentile(ranked, 50)
    p90 = _percentile(ranked, 90)
    recommended = _percentile(ranked, args.target_percentile)

    report = {
        "n_queries": len(queries),
        "mean": round(statistics.fmean(scores), 6) if scores else 0.0,
        "P10": round(p10, 6),
        "P20": round(p20, 6),
        "P50": round(p50, 6),
        "P90": round(p90, 6),
        "recommended_TABLE_RAG_REWRITE_FALLBACK_THRESHOLD": round(recommended, 6),
        "note": (
            "RRF(k=60) scores are ~0.015–0.033. Set threshold near P15–P20 so "
            "rewrites fire only on genuinely weak lexical/semantic matches."
        ),
        "samples": details[:20],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

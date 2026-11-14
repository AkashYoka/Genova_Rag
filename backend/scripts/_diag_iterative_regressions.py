"""Diagnose iterative-loop regressions on Xolair cases 1,2,3,5."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = "43ae024f-3c0d-40f5-aa49-83e9f3682641"
USER = "f3d989fb-79af-4fe2-9360-12b06772348c"

from app.config import get_settings
from app.rag.generation.iterative_evidence import run_iterative_evidence_loop
from app.rag.retrieval.retriever import retrieve

CASES = {
    1: (
        "A 35-year-old patient with allergic asthma, body weight 78 kg and "
        "pretreatment IgE 350 IU/mL was intentionally prescribed Xolair 150 mg "
        "every 4 weeks. No adverse event was reported. Identify any adverse "
        "event, special situation, or off-label usage."
    ),
    2: (
        "According to Table 1 for patients 12 years and older with asthma, what "
        "XOLAIR dose and frequency apply for pretreatment IgE 350 IU/mL and body "
        "weight 78 kg?"
    ),
    3: (
        "Asthma patient >=12 years, IgE 300 IU/mL, weight 70 kg. What is the "
        "label dose and frequency from the dosing table?"
    ),
    5: (
        "Adult asthma, 80 kg, IgE 250 IU/mL prescribed 75 mg every 4 weeks. "
        "No AE. Classify AE / special situation / off-label / dosing deviation."
    ),
}


def main() -> None:
    get_settings.cache_clear()
    s = get_settings()
    print(
        "iterative",
        s.TABLE_RAG_ITERATIVE_ENABLED,
        "max",
        s.TABLE_RAG_MAX_ITERATIONS,
    )
    for cid, q in CASES.items():
        print(f"\n=== CASE {cid} ===")
        chunks = retrieve(query=q, user_id=USER, document_id=DOC, top_k=8)
        result, merged, stats = run_iterative_evidence_loop(
            q, chunks, user_id=USER, document_id=DOC
        )
        v = result.verified
        print(
            "verified",
            None
            if v is None
            else {
                "value": v.value,
                "page": v.page,
                "table_id": v.table_id,
                "row": v.row_headers,
                "col": v.col_headers,
            },
        )
        if result.insufficient:
            print(
                "insuff",
                result.insufficient.status,
                result.insufficient.reason[:180],
                "missing",
                result.insufficient.missing_dimensions,
            )
        raw = result.raw_llm or {}
        print(
            "raw",
            raw.get("status"),
            "answer",
            raw.get("answer"),
            "ready",
            raw.get("answer_ready"),
            "queries",
            raw.get("additional_search_queries"),
        )
        print("stats", json.dumps(stats, ensure_ascii=False)[:500])
        print(
            "tables_in_merged",
            sorted(
                {
                    str(c.get("table_id"))
                    for c in merged
                    if c.get("table_id")
                }
            ),
        )


if __name__ == "__main__":
    main()

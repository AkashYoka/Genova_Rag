"""Diagnose Q5 table selection for Adult asthma 80kg IgE 250."""
from __future__ import annotations

from app.rag.generation.structured_lookup import lookup_cell_in_table
from app.rag.generation.table_selector import (
    collect_candidate_tables,
    coerce_table_dict,
    expand_document_sibling_chunks,
    interpret_with_selector,
    scope_rank_candidates,
)
from app.rag.table_rag.store import list_document_tables

DOC = "43ae024f-3c0d-40f5-aa49-83e9f3682641"
Q = (
    "Adult asthma, 80 kg, IgE 250 IU/mL prescribed 75 mg every 4 weeks. "
    "No AE. Classify AE / special situation / off-label / dosing deviation."
)


def main() -> None:
    rows = list_document_tables(document_id=DOC)
    print("=== SQL tables ===")
    for r in rows:
        print(
            r.get("page_start"),
            r.get("table_id"),
            "|",
            (r.get("caption") or "")[:110],
        )

    print("\n=== per-table lookup ===")
    for r in rows:
        table = dict(r.get("table") or {})
        table.setdefault("caption", r.get("caption") or "")
        table.setdefault("page", r.get("page_start"))
        out = lookup_cell_in_table(
            table,
            Q,
            filename=r.get("filename") or "",
            page=r.get("page_start"),
        )
        kind = type(out).__name__
        val = getattr(out, "value", None)
        print(
            r.get("page_start"),
            (r.get("caption") or "")[:55],
            kind,
            val,
            getattr(out, "evidence", {}) if hasattr(out, "evidence") else "",
        )

    # Build synthetic chunks from SQL (full sibling pool) as expand would.
    chunks = []
    for r in rows:
        table = dict(r.get("table") or {})
        table["caption"] = r.get("caption") or table.get("caption") or ""
        table["page"] = r.get("page_start")
        chunks.append(
            {
                "table_id": r.get("table_id"),
                "document_id": DOC,
                "table": table,
                "caption": r.get("caption") or "",
                "page": r.get("page_start"),
                "score": 0.2,
                "chunk_type": "table",
                "filename": r.get("filename") or "",
            }
        )

    expanded = expand_document_sibling_chunks(chunks)
    raw = collect_candidate_tables(expanded, top_k=32, question="")
    ranked = scope_rank_candidates(Q, raw)
    print("\n=== scope_rank order ===")
    for i, item in enumerate(ranked[:10]):
        t = item.get("table") or {}
        ch = item.get("chunk") or {}
        print(
            i,
            item.get("table_id"),
            "p",
            t.get("page") or ch.get("page"),
            "|",
            (item.get("caption") or t.get("caption") or ch.get("caption") or "")[:90],
            "score",
            item.get("score"),
        )

    outcome, decision, cands = interpret_with_selector(Q, chunks)
    print("\n=== interpret_with_selector ===")
    print("decision", decision)
    print("outcome", type(outcome).__name__, getattr(outcome, "value", None))
    print("evidence", getattr(outcome, "evidence", None))
    print("top cand", cands[0].get("table_id") if cands else None)


if __name__ == "__main__":
    main()

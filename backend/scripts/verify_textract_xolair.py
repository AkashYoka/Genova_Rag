"""Verify production Textract table extraction + Q1 lookup on Xolair pages.

Reindexes pages 5,6,7,11 with DetectDocumentText + geometry tables and runs:
  - Q1: IgE 350 IU/mL, weight 78 kg (expect structured ~300)
  - Q8: unsupported IgE/weight (expect abstention / no invented dose)

Compares extraction shape against the recorded three-way Case C winner.

Usage
-----
  cd backend
  python scripts/verify_textract_xolair.py
  python scripts/verify_textract_xolair.py --answer
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("PDF_LAYOUT_WRITE_DIAGNOSTICS", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("verify_textract_xolair")

Q1_NO_AGE = (
    "For a patient with pretreatment IgE of 350 IU/mL and body weight "
    "of 78 kg, what is the recommended XOLAIR dose and dosing frequency?"
)
Q1_ADULT = (
    "For a 45-year-old patient with pretreatment IgE of 350 IU/mL and "
    "body weight of 78 kg, what is the recommended XOLAIR dose and "
    "dosing frequency according to the asthma dosing table?"
)
Q1_PEDS = (
    "For an 8-year-old patient with pretreatment IgE of 350 IU/mL and "
    "body weight of 78 kg, what is the recommended XOLAIR dose and "
    "dosing frequency according to the dosing table?"
)
Q8 = (
    "What is the recommended XOLAIR dose for a patient with "
    "pretreatment IgE of 9999 IU/mL and body weight of 5 kg?"
)


def _subset_pdf(src: Path, pages: Sequence[int], dest: Path) -> Dict[int, int]:
    import fitz

    mapping: Dict[int, int] = {}
    with fitz.open(str(src)) as doc:
        out = fitz.open()
        for new_idx, page_no in enumerate(pages, start=1):
            out.insert_pdf(doc, from_page=page_no - 1, to_page=page_no - 1)
            mapping[new_idx] = page_no
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(dest))
        out.close()
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        default=str(ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf"),
    )
    parser.add_argument("--pages", default="5,6,7,11")
    parser.add_argument("--answer", action="store_true")
    parser.add_argument(
        "--json-out",
        default="data/layout_reports/textract_xolair_verify.json",
    )
    args = parser.parse_args()

    from app.config import get_settings
    from app.rag.generation.table_interpretation import interpret_table_question
    from app.rag.ingestion.pdf_layout.parser import parse_pdf
    from app.rag.llm.agent import generate_answer
    from app.rag.retrieval.retriever import retrieve
    from app.rag.table_rag import expand_table_rag_chunks, route_query
    from app.rag.vectordb.vectorstore import delete_document_chunks, store_chunks

    settings = get_settings()
    settings.PDF_TEXTRACT_TABLES_ENABLED = True
    settings.PDF_TEXTRACT_TABULAR_ONLY = False  # match bakeoff Case C
    settings.TABLE_RAG_SEPARATE_INTERPRETATION_EMBEDDING = True
    settings.TABLE_RAG_ENABLED = True

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    pages = [int(p.strip()) for p in args.pages.split(",") if p.strip()]

    user_id = f"verify-textract-{uuid.uuid4().hex[:8]}"
    document_id = str(uuid.uuid4())
    chroma = Path(tempfile.mkdtemp(prefix="verify_textract_chroma_"))
    settings.CHROMA_PERSIST_DIR = str(chroma)
    import app.rag.vectordb.vectorstore as vs

    vs._chroma_client = None

    temp = tempfile.TemporaryDirectory(prefix="verify_textract_pdf_")
    target = Path(temp.name) / "subset.pdf"
    mapping = _subset_pdf(pdf, pages, target)

    t0 = time.perf_counter()
    parsed = parse_pdf(str(target), document_id=document_id, document_name=pdf.name)
    extract_s = time.perf_counter() - t0

    chunks = list(parsed.chunks or [])
    for chunk in chunks:
        for key in ("page", "page_start", "page_end"):
            val = chunk.get(key)
            if val is None:
                continue
            mapped = mapping.get(int(val))
            if mapped:
                chunk[key] = mapped
        raw = chunk.get("table_json")
        if isinstance(raw, str) and raw:
            try:
                payload = json.loads(raw)
                mp = mapping.get(int(payload.get("page") or 0))
                if mp:
                    payload["page"] = mp
                    chunk["table_json"] = json.dumps(payload, ensure_ascii=False)
            except Exception:
                pass

    expanded, table_records = expand_table_rag_chunks(
        chunks, document_id=document_id, filename=pdf.name, enabled=True
    )
    for rec in table_records:
        mp = mapping.get(int(rec.get("page_start") or 0))
        if mp:
            rec["page_start"] = mp
            rec["page_end"] = mp
            rec["page"] = mp
            rec["table"]["page"] = mp

    store_chunks(
        chunks=expanded,
        document_id=document_id,
        filename=pdf.name,
        user_id=user_id,
    )

    tables_summary = []
    for rec in table_records:
        table = rec["table"]
        tables_summary.append(
            {
                "table_id": rec["table_id"],
                "page": rec.get("page_start"),
                "n_rows": len(table.get("rows") or []),
                "n_cols": table.get("n_cols"),
                "merged_cells": len(table.get("merged_cells") or []),
                "extraction_method": table.get("extraction_method")
                or rec.get("extraction_method"),
                "headers": (table.get("headers") or [])[:8],
                "sample_row0": (table.get("rows") or [[]])[0][:8],
            }
        )

    # In-memory hydrate
    by_id = {r["table_id"]: r for r in table_records}

    def _mem_load(*, user_id, pairs, db=None):
        out = {}
        for doc_id, tid in pairs:
            rec = by_id.get(tid)
            if rec:
                out[f"{doc_id}:{tid}"] = {
                    "table_id": tid,
                    "table": rec["table"],
                    "needs_verification": rec.get("needs_verification"),
                    "caption": rec["table"].get("caption") or "",
                }
        return out

    import app.rag.table_rag.hydrate as hydrate_mod

    hydrate_mod.load_tables_by_ids = _mem_load  # type: ignore

    results: List[dict] = []
    for qid, question in (
        ("q1_no_age", Q1_NO_AGE),
        ("q1_adult_age", Q1_ADULT),
        ("q1_peds_age", Q1_PEDS),
        ("q8", Q8),
    ):
        route = route_query(question)
        retrieved = retrieve(
            query=question, user_id=user_id, document_id=document_id, top_k=8
        )
        from app.rag.table_rag.hydrate import hydrate_retrieved_tables

        retrieved = hydrate_retrieved_tables(retrieved, user_id=user_id)
        interp = interpret_table_question(question, retrieved)
        entry = {
            "id": qid,
            "question": question,
            "route": route.modality.value,
            "table_ids": sorted(
                {str(h.get("table_id")) for h in retrieved if h.get("table_id")}
            ),
            "structured_lookup": None,
            "answer": None,
        }
        if interp.verified is not None:
            entry["structured_lookup"] = {
                "status": getattr(interp.verified, "status", "matched"),
                "value": interp.verified.value,
                "table_id": getattr(interp.verified, "table_id", None),
                "row_headers": interp.verified.row_headers,
                "col_headers": interp.verified.col_headers,
                "page": interp.verified.page,
            }
        elif interp.insufficient is not None:
            entry["structured_lookup"] = {
                "insufficient": True,
                "status": getattr(interp.insufficient, "status", None),
                "reason": getattr(interp.insufficient, "reason", None),
                "soft": getattr(interp.insufficient, "soft", None),
                "candidate_values": getattr(
                    interp.insufficient, "candidate_values", None
                ),
            }
        if args.answer:
            packed = generate_answer(
                question=question, user_id=user_id, document_id=document_id
            )
            entry["answer"] = packed.get("answer")
        results.append(entry)
        logger.info(
            "%s lookup=%s tables=%s",
            qid,
            entry["structured_lookup"],
            entry["table_ids"],
        )

    report = {
        "pdf": str(pdf),
        "pages": pages,
        "engine": "textract",
        "extract_seconds": round(extract_s, 3),
        "table_extraction": (parsed.report or {}).get("table_extraction"),
        "tables": tables_summary,
        "questions": results,
        "compare_to_case_c": {
            "expected_min_rows_page5": 6,
            "q1_expected_value": "300",
        },
    }
    out = Path(args.json_out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", out)

    print("\n=== Textract Xolair verification ===")
    print(f"Tables: {len(table_records)}")
    for t in tables_summary:
        print(
            f"  page {t['page']}: {t['n_rows']}x{t['n_cols']} "
            f"method={t['extraction_method']} headers={t['headers'][:4]}"
        )
    for q in results:
        print(f"{q['id']}: lookup={q['structured_lookup']}")
        if q.get("answer"):
            print((q["answer"] or "")[:400])

    try:
        delete_document_chunks(document_id, user_id)
    except Exception:
        pass
    temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

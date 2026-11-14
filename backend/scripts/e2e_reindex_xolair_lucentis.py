"""Reindex Xolair + Lucentis PDFs with Textract TableRAG and run E2E questions.

Uses the existing PV single-session question batteries. Writes a combined
report under ``data/layout_reports/``.

Usage
-----
  cd backend
  python scripts/e2e_reindex_xolair_lucentis.py
  python scripts/e2e_reindex_xolair_lucentis.py --no-answer   # lookup only
  python scripts/e2e_reindex_xolair_lucentis.py --docs xolair
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("PDF_LAYOUT_WRITE_DIAGNOSTICS", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("e2e_reindex")

DOCS = {
    "xolair": {
        "pdf_candidates": [
            ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf",
            Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Medcare\xolair_prescribing.pdf"),
        ],
        "questions_json": ROOT
        / "scripts"
        / "_e2e_xolair_pv_extended_single_session.json",
        # Full label is large; key dosing + safety pages for structured lookup.
        "pages": None,  # full PDF
    },
    "lucentis": {
        "pdf_candidates": [
            ROOT / "data" / "sample_subsets" / "lucentis_prescribing.pdf",
            Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Medcare\lucentis_prescribing.pdf"),
        ],
        "questions_json": ROOT / "scripts" / "_e2e_lucentis_pv_single_session.json",
        "pages": None,
    },
}


def _find_pdf(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"PDF not found in: {[str(p) for p in candidates]}")


def _load_questions(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(r["question"]) for r in data.get("results") or [] if r.get("question")]


def _subset_pdf(src: Path, pages: Sequence[int], dest: Path) -> Dict[int, int]:
    import fitz

    mapping: Dict[int, int] = {}
    with fitz.open(str(src)) as doc:
        out = fitz.open()
        for new_idx, page_no in enumerate(pages, start=1):
            if page_no < 1 or page_no > doc.page_count:
                continue
            out.insert_pdf(doc, from_page=page_no - 1, to_page=page_no - 1)
            mapping[new_idx] = page_no
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(dest))
        out.close()
    return mapping


def _answer_flags(ans: str) -> Dict[str, bool]:
    low = (ans or "").lower()
    return {
        "insuff_prefix": low.lstrip().startswith("insufficient table evidence")
        or low.lstrip().startswith("ambiguous table evidence"),
        "ambiguous": "ambiguous" in low[:200],
        "not_concordant": bool(
            re.search(
                r"not label[- ]concordant|does not match|discrepan|inappropriate|"
                r"medication error|under[- ]?dos|over[- ]?dos|incorrect",
                low,
            )
        ),
        "matches": bool(
            re.search(
                r"label[- ]concordant|matches the label|matches the approved|recommended",
                low,
            )
        )
        and not bool(re.search(r"not label[- ]concordant|does not match", low)),
        "off_label": bool(re.search(r"off[- ]?label|not indicated|not an approved", low)),
        "verified_banner": "verified table evidence is authoritative" in low
        or "verified table value" in low,
    }


def _reindex_pdf(
    *,
    pdf: Path,
    pages: Optional[Sequence[int]],
    user_id: str,
    document_id: str,
) -> Dict[str, Any]:
    from app.config import get_settings
    from app.rag.ingestion.pdf_layout.parser import parse_pdf
    from app.rag.table_rag import expand_table_rag_chunks
    from app.rag.vectordb.vectorstore import store_chunks
    import app.rag.table_rag.hydrate as hydrate_mod
    import app.rag.vectordb.vectorstore as vs

    settings = get_settings()
    settings.PDF_TEXTRACT_TABLES_ENABLED = True
    # Full PDFs: only analyze pages that look tabular (production gate).
    settings.PDF_TEXTRACT_TABULAR_ONLY = True
    settings.TABLE_RAG_ENABLED = True
    settings.TABLE_RAG_SEPARATE_INTERPRETATION_EMBEDDING = True

    temp_pdf = None
    mapping: Dict[int, int] = {}
    target = pdf
    if pages:
        temp_pdf = tempfile.TemporaryDirectory(prefix="e2e_pdf_")
        target = Path(temp_pdf.name) / "subset.pdf"
        mapping = _subset_pdf(pdf, pages, target)

    t0 = time.perf_counter()
    parsed = parse_pdf(str(target), document_id=document_id, document_name=pdf.name)
    extract_s = time.perf_counter() - t0

    chunks = list(parsed.chunks or [])
    if mapping:
        for chunk in chunks:
            for key in ("page", "page_start", "page_end"):
                val = chunk.get(key)
                if val is None:
                    continue
                mapped = mapping.get(int(val))
                if mapped:
                    chunk[key] = mapped
            raw = chunk.get("table_json")
            if isinstance(raw, str) and raw.strip():
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
    if mapping:
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

    hydrate_mod.load_tables_by_ids = _mem_load  # type: ignore

    tables_summary = [
        {
            "table_id": rec["table_id"],
            "page": rec.get("page_start"),
            "n_rows": len(rec["table"].get("rows") or []),
            "n_cols": rec["table"].get("n_cols"),
            "extraction_method": rec["table"].get("extraction_method"),
            "headers": (rec["table"].get("headers") or [])[:8],
            "caption": (rec["table"].get("caption") or "")[:120],
        }
        for rec in table_records
    ]

    return {
        "extract_seconds": round(extract_s, 3),
        "chunk_count": len(expanded),
        "table_count": len(table_records),
        "tables": tables_summary,
        "table_extraction": (parsed.report or {}).get("table_extraction"),
        "temp_pdf": temp_pdf,
        "mapping": mapping,
        "_vs": vs,
    }


def _run_questions(
    *,
    questions: Sequence[str],
    user_id: str,
    document_id: str,
    with_answer: bool,
) -> List[Dict[str, Any]]:
    from app.rag.generation.llm_table_reasoning import interpret_table_question
    from app.rag.llm.agent import generate_answer
    from app.rag.retrieval.retriever import retrieve
    from app.rag.table_rag import route_query
    from app.rag.table_rag.hydrate import hydrate_retrieved_tables

    results: List[Dict[str, Any]] = []
    for i, question in enumerate(questions, 1):
        t0 = time.perf_counter()
        route = route_query(question)
        retrieved = retrieve(
            query=question, user_id=user_id, document_id=document_id, top_k=8
        )
        retrieved = hydrate_retrieved_tables(retrieved, user_id=user_id)
        interp = interpret_table_question(question, retrieved)
        entry: Dict[str, Any] = {
            "i": i,
            "question": question,
            "route": route.modality.value,
            "elapsed_s": None,
            "table_ids": sorted(
                {str(h.get("table_id")) for h in retrieved if h.get("table_id")}
            ),
            "structured_lookup": None,
            "answer": None,
            "flags": {},
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
                "missing_dimensions": getattr(
                    interp.insufficient, "missing_dimensions", None
                ),
            }

        if with_answer:
            packed = generate_answer(
                question=question, user_id=user_id, document_id=document_id
            )
            ans = packed.get("answer") or ""
            entry["answer"] = ans
            entry["flags"] = _answer_flags(ans)
        entry["elapsed_s"] = round(time.perf_counter() - t0, 2)
        results.append(entry)

        lookup = entry["structured_lookup"] or {}
        logger.info(
            "Q%s %.1fs lookup=%s flags=%s",
            i,
            entry["elapsed_s"],
            {
                k: lookup.get(k)
                for k in ("status", "value", "page", "insufficient", "candidate_values")
                if k in lookup
            },
            entry.get("flags"),
        )
        preview = (entry.get("answer") or "")[:160].replace("\n", " ")
        if preview:
            logger.info("  ans: %s", preview)
    return results


def run_one(name: str, cfg: dict, *, with_answer: bool) -> Dict[str, Any]:
    from app.rag.vectordb.vectorstore import delete_document_chunks
    from app.config import get_settings
    import app.rag.vectordb.vectorstore as vs

    pdf = _find_pdf(cfg["pdf_candidates"])
    questions = _load_questions(cfg["questions_json"])
    logger.info("=== %s pdf=%s questions=%s ===", name, pdf, len(questions))

    settings = get_settings()
    user_id = f"e2e-{name}-{uuid.uuid4().hex[:8]}"
    document_id = str(uuid.uuid4())
    chroma = Path(tempfile.mkdtemp(prefix=f"e2e_{name}_chroma_"))
    settings.CHROMA_PERSIST_DIR = str(chroma)
    vs._chroma_client = None

    meta = _reindex_pdf(
        pdf=pdf,
        pages=cfg.get("pages"),
        user_id=user_id,
        document_id=document_id,
    )
    logger.info(
        "%s indexed: tables=%s chunks=%s extract=%.1fs",
        name,
        meta["table_count"],
        meta["chunk_count"],
        meta["extract_seconds"],
    )
    for t in meta["tables"][:12]:
        logger.info(
            "  p%s %sx%s %s | %s",
            t["page"],
            t["n_rows"],
            t["n_cols"],
            t["extraction_method"],
            (t.get("caption") or "")[:70],
        )

    results = _run_questions(
        questions=questions,
        user_id=user_id,
        document_id=document_id,
        with_answer=with_answer,
    )

    try:
        delete_document_chunks(document_id, user_id)
    except Exception:
        pass
    temp_pdf = meta.pop("temp_pdf", None)
    if temp_pdf is not None:
        temp_pdf.cleanup()
    meta.pop("_vs", None)

    return {
        "name": name,
        "pdf": str(pdf),
        "document_id": document_id,
        "user_id": user_id,
        "extract_seconds": meta["extract_seconds"],
        "chunk_count": meta["chunk_count"],
        "table_count": meta["table_count"],
        "tables": meta["tables"],
        "table_extraction": meta.get("table_extraction"),
        "questions": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs",
        default="xolair,lucentis",
        help="Comma list: xolair,lucentis",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Skip generate_answer; structured lookup only",
    )
    parser.add_argument(
        "--json-out",
        default="data/layout_reports/e2e_xolair_lucentis.json",
    )
    args = parser.parse_args()
    names = [n.strip() for n in args.docs.split(",") if n.strip()]
    with_answer = not args.no_answer

    report: Dict[str, Any] = {"docs": {}, "with_answer": with_answer}
    for name in names:
        if name not in DOCS:
            raise SystemExit(f"Unknown doc {name!r}; choose from {list(DOCS)}")
        report["docs"][name] = run_one(name, DOCS[name], with_answer=with_answer)

    out = Path(args.json_out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", out)

    print("\n=== E2E summary ===")
    for name, block in report["docs"].items():
        print(
            f"{name}: tables={block['table_count']} chunks={block['chunk_count']} "
            f"extract={block['extract_seconds']}s questions={len(block['questions'])}"
        )
        for q in block["questions"]:
            lu = q.get("structured_lookup") or {}
            status = lu.get("status") or ("insuff" if lu.get("insufficient") else "?")
            val = lu.get("value") or lu.get("candidate_values") or ""
            print(
                f"  Q{q['i']}: {status} {val} "
                f"page={lu.get('page')} {q['elapsed_s']}s"
            )
            if q.get("answer"):
                print("   ", (q["answer"] or "")[:140].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

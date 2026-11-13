"""Demonstrate TableRAG indexing + retrieval on xolair_prescribing.pdf.

Runs without a live LLM when ``--offline`` is set (hash embeddings).

Examples
--------
  cd backend
  python scripts/demo_table_rag_xolair.py --offline
  python scripts/demo_table_rag_xolair.py --pages 3,4,5 --query "..."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("demo_table_rag")


DEFAULT_QUERY = (
    "What dose applies for pretreatment assay 350 IU/mL and body weight "
    "78 kg every 2 weeks according to the dosing table?"
)


def _find_pdf(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"PDF not found: {path}")
        return path
    candidates = [
        ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf",
        ROOT / "docs" / "xolair_prescribing.pdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(
        "xolair_prescribing.pdf not found. Pass --pdf path/to/file.pdf"
    )


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
        out.save(str(dest))
        out.close()
    return mapping


def _install_hash_embeddings() -> None:
    """Deterministic bag-of-tokens embeddings for offline demos/tests."""
    import hashlib
    import math

    import app.rag.embeddings.embeddings as emb

    dim = 64

    def _vec(text: str) -> list[float]:
        vec = [0.0] * dim
        for token in (text or "").lower().split():
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    class _HashModel:
        def embed_documents(self, texts):
            return [_vec(t) for t in texts]

        def embed_query(self, text):
            return _vec(text)

    model = _HashModel()
    emb.get_embedding_model = lambda: model  # type: ignore
    emb.embed_query = lambda q: _vec(q)  # type: ignore
    emb.embed_texts = lambda texts: [_vec(t) for t in texts]  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default=None, help="Path to xolair_prescribing.pdf")
    parser.add_argument("--pages", default="3,4,5,6", help="Comma pages (empty=all)")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--offline", action="store_true", help="Hash embeddings; skip LLM")
    parser.add_argument("--answer", action="store_true", help="Also call generate_answer")
    parser.add_argument("--json-out", default=None, help="Write demo result JSON")
    args = parser.parse_args()

    pdf = _find_pdf(args.pdf)
    pages: List[int] = []
    if args.pages.strip():
        pages = [int(p.strip()) for p in args.pages.split(",") if p.strip()]

    from app.config import get_settings
    from app.rag.ingestion.pdf_layout.parser import parse_pdf
    from app.rag.table_rag import expand_table_rag_chunks, route_query
    from app.rag.table_rag.hydrate import hydrate_retrieved_tables
    from app.rag.vectordb.vectorstore import (
        store_chunks,
        query_chunks,
        delete_document_chunks,
    )
    from app.rag.embeddings.embeddings import embed_query
    from app.rag.retrieval.retriever import retrieve
    from app.rag.retrieval.bm25 import query_bm25
    from app.rag.generation.table_interpretation import interpret_table_question
    from app.rag.llm.agent import _format_context

    settings = get_settings()
    user_id = f"demo-table-rag-{uuid.uuid4().hex[:8]}"
    document_id = str(uuid.uuid4())
    filename = pdf.name

    if args.offline:
        _install_hash_embeddings()
        tmp = tempfile.mkdtemp(prefix="table_rag_chroma_")
        settings.CHROMA_PERSIST_DIR = tmp
        logger.info("Offline mode: CHROMA_PERSIST_DIR=%s", tmp)

    temp_dir = None
    target = pdf
    mapping: Dict[int, int] = {}
    try:
        if pages:
            temp_dir = tempfile.TemporaryDirectory(prefix="table_rag_pdf_")
            target = Path(temp_dir.name) / f"{pdf.stem}_pages.pdf"
            mapping = _subset_pdf(pdf, pages, target)
            logger.info("Subset pages %s → %s", pages, target)

        logger.info("Parsing %s", target)
        parsed = parse_pdf(
            str(target),
            document_id=document_id,
            document_name=filename,
        )
    finally:
        if temp_dir is not None:
            # Keep subset until parse finishes; cleanup after.
            pass

    chunks = list(parsed.chunks or [])
    # Remap page numbers in table payloads when subsetting.
    if mapping:
        for chunk in chunks:
            src_page = mapping.get(int(chunk.get("page") or 0))
            if src_page:
                chunk["page"] = src_page
                chunk["page_start"] = src_page
                chunk["page_end"] = src_page
            raw = chunk.get("table_json")
            if isinstance(raw, str) and raw:
                try:
                    payload = json.loads(raw)
                    mapped = mapping.get(int(payload.get("page") or 0))
                    if mapped:
                        payload["page"] = mapped
                        chunk["table_json"] = json.dumps(payload, ensure_ascii=False)
                except Exception:
                    pass

    expanded, table_records = expand_table_rag_chunks(
        chunks,
        document_id=document_id,
        filename=filename,
        enabled=True,
    )
    logger.info(
        "Chunks: base=%d expanded=%d tables=%d",
        len(chunks),
        len(expanded),
        len(table_records),
    )
    for rec in table_records:
        logger.info(
            "  table_id=%s page=%s rows=%s cols=%s caption=%r merged=%d",
            rec["table_id"],
            rec.get("page_start"),
            len(rec["table"].get("rows") or []),
            rec["table"].get("n_cols"),
            (rec["table"].get("caption") or "")[:60],
            len(rec["table"].get("merged_cells") or []),
        )

    stored = store_chunks(
        chunks=expanded,
        document_id=document_id,
        filename=filename,
        user_id=user_id,
    )
    logger.info("Stored %d vectors/BM25 entries", stored)

    route = route_query(args.query)
    logger.info("Route: modality=%s reason=%s", route.modality.value, route.reason)

    qvec = embed_query(args.query)
    vector_hits = query_chunks(
        query_embedding=qvec,
        user_id=user_id,
        document_id=document_id,
        top_k=8,
    )
    try:
        bm25_hits = query_bm25(
            query=args.query,
            user_id=user_id,
            document_id=document_id,
            top_k=8,
        )
    except Exception as exc:
        logger.warning("BM25 query failed: %s", exc)
        bm25_hits = []

    retrieved = retrieve(
        query=args.query,
        user_id=user_id,
        document_id=document_id,
        top_k=8,
    )
    if table_records and not any(c.get("table_prompt_block") for c in retrieved):
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
        retrieved = hydrate_retrieved_tables(retrieved, user_id=user_id)

    interpretation = interpret_table_question(args.query, retrieved)
    context = _format_context(retrieved)

    result = {
        "pdf": str(pdf),
        "document_id": document_id,
        "user_id": user_id,
        "query": args.query,
        "route": {
            "modality": route.modality.value,
            "reason": route.reason,
            "wants_structured_lookup": route.wants_structured_lookup,
        },
        "tables_indexed": [
            {
                "table_id": r["table_id"],
                "page": r.get("page_start"),
                "n_rows": len(r["table"].get("rows") or []),
                "n_cols": r["table"].get("n_cols"),
                "merged_cells": len(r["table"].get("merged_cells") or []),
            }
            for r in table_records
        ],
        "vector_top": [
            {
                "chunk_type": h.get("chunk_type"),
                "page": h.get("page"),
                "table_id": h.get("table_id"),
                "score": h.get("score"),
                "text_head": str(h.get("text") or "")[:160],
            }
            for h in vector_hits[:5]
        ],
        "bm25_top": [
            {
                "chunk_type": h.get("chunk_type"),
                "page": h.get("page"),
                "table_id": h.get("table_id"),
                "score": h.get("score"),
                "text_head": str(h.get("text") or "")[:160],
            }
            for h in bm25_hits[:5]
        ],
        "retrieved": [
            {
                "chunk_type": h.get("chunk_type"),
                "page": h.get("page"),
                "table_id": h.get("table_id"),
                "score": h.get("score"),
                "has_table_prompt": bool(h.get("table_prompt_block")),
                "text_head": str(h.get("text") or "")[:160],
            }
            for h in retrieved
        ],
        "verified_cell": None,
        "context_head": context[:1200],
        "answer": None,
    }
    if interpretation.verified is not None:
        result["verified_cell"] = {
            "value": interpretation.verified.value,
            "row_headers": interpretation.verified.row_headers,
            "col_headers": interpretation.verified.col_headers,
            "page": interpretation.verified.page,
            "confidence": interpretation.verified.confidence,
        }
        logger.info(
            "Verified cell: value=%s row=%s col=%s page=%s",
            interpretation.verified.value,
            interpretation.verified.row_headers,
            interpretation.verified.col_headers,
            interpretation.verified.page,
        )
    else:
        logger.info(
            "No verified cell (structured=%s reason=%s)",
            interpretation.is_structured_lookup,
            getattr(interpretation.insufficient, "reason", None),
        )

    if args.answer and not args.offline:
        from app.rag.llm.agent import generate_answer

        packed = generate_answer(
            question=args.query,
            user_id=user_id,
            document_id=document_id,
        )
        result["answer"] = packed.get("answer")
        logger.info("Answer head: %s", str(packed.get("answer") or "")[:400])
    elif args.answer and args.offline:
        logger.info("Skipping LLM answer in --offline mode")

    print("\n=== TableRAG demo summary ===")
    print(f"PDF: {pdf}")
    print(f"Tables indexed: {len(table_records)}")
    print(f"Route: {route.modality.value} ({route.reason})")
    print(f"Retrieved chunks: {len(retrieved)}")
    for hit in retrieved[:5]:
        print(
            f"  - type={hit.get('chunk_type')} page={hit.get('page')} "
            f"table_id={hit.get('table_id')} score={hit.get('score')}"
        )
    if result["verified_cell"]:
        print(
            f"Verified: {result['verified_cell']['value']} "
            f"(page {result['verified_cell']['page']})"
        )
    print("Context head:\n", context[:800])

    if args.json_out:
        out = Path(args.json_out)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out)

    try:
        delete_document_chunks(document_id, user_id)
    except Exception:
        pass
    if temp_dir is not None:
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

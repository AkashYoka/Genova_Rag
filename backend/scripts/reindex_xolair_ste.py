#!/usr/bin/env python3
"""Re-index xolair_prescribing.pdf with STE Textract AnalyzeDocument + Bedrock.

Enables PDF_STE_TABLES_ENABLED, re-runs ingest, then dumps how table cards /
row windows land in Chroma + document_tables SQL.

Usage (from backend/):
  python scripts/reindex_xolair_ste.py
  python scripts/reindex_xolair_ste.py --all-pages
  python scripts/reindex_xolair_ste.py --pages 5,6,19,21,32
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Known table-bearing pages from prior Textract eval (smoke subset).
DEFAULT_STE_PAGES = "5,6,7,8,11,17,19,21,32,33,35,36,38,41,42,49"

XOLAIR_PDF = ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf"
# Prefer the ready local-dev copy; fall back to the long-lived smoke id.
PREFERRED_DOC_ID = "f7ebb234-3d78-46bb-9fa5-b045b54f3d11"
FALLBACK_DOC_ID = "43ae024f-3c0d-40f5-aa49-83e9f3682641"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reindex_xolair_ste")


def _configure_env(*, all_pages: bool, pages: str) -> None:
    os.environ["PDF_STE_TABLES_ENABLED"] = "true"
    os.environ["PDF_STE_BEDROCK_ENABLED"] = "true"
    os.environ["PDF_STE_BACKEND"] = "textract"
    os.environ.setdefault("PDF_STE_USE_CACHE", "true")
    os.environ.setdefault("PDF_STE_OUTPUT_DIR", str(ROOT / "data" / "ste_runs"))
    os.environ.setdefault("TABLE_RAG_ENABLED", "true")
    # Keep Textract for prose OCR; STE wins overlapping tables.
    os.environ.setdefault("PDF_TEXTRACT_FALLBACK_ENABLED", "true")
    os.environ.setdefault("PDF_TEXTRACT_TABLES_ENABLED", "true")
    if all_pages:
        os.environ["PDF_STE_PAGES"] = ""
    else:
        os.environ["PDF_STE_PAGES"] = pages.strip()


def _clear_settings() -> Any:
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    settings.PDF_STE_TABLES_ENABLED = True
    settings.PDF_STE_BEDROCK_ENABLED = True
    settings.PDF_STE_BACKEND = "textract"
    settings.PDF_STE_PAGES = os.environ.get("PDF_STE_PAGES", "") or ""
    settings.TABLE_RAG_ENABLED = True
    settings.SEMANTIC_CHUNKING_ENABLED = True
    settings.CHUNK_STRATEGY = "semantic_llm"
    return settings


def _resolve_doc(document_id: Optional[str] = None) -> Any:
    from app.database import SessionLocal
    from app.models import Document

    db = SessionLocal()
    try:
        if document_id:
            doc = (
                db.query(Document)
                .filter(Document.id == document_id, Document.is_deleted.is_(False))
                .first()
            )
            return doc
        for doc_id in (PREFERRED_DOC_ID, FALLBACK_DOC_ID):
            doc = (
                db.query(Document)
                .filter(Document.id == doc_id, Document.is_deleted.is_(False))
                .first()
            )
            if doc:
                return doc
        doc = (
            db.query(Document)
            .filter(
                Document.original_name.ilike("%xolair_prescribing%"),
                Document.is_deleted.is_(False),
            )
            .order_by(Document.uploaded_at.desc())
            .first()
        )
        return doc
    finally:
        db.close()


def _ensure_local_copy(user_id: str, src: Path) -> Path:
    from app.config import get_settings

    settings = get_settings()
    dest_dir = Path(settings.UPLOAD_DIR) / str(user_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dest)
    return dest


def _inspect_vectors(document_id: str, user_id: str) -> Dict[str, Any]:
    from app.database import SessionLocal
    from app.models import DocumentTable
    from app.rag.vectordb.vectorstore import get_chroma_client, get_collection_name

    db = SessionLocal()
    try:
        sql_rows = (
            db.query(DocumentTable)
            .filter(DocumentTable.document_id == document_id)
            .order_by(DocumentTable.page_start, DocumentTable.ordinal)
            .all()
        )
        sql_summary = []
        for row in sql_rows:
            payload: Dict[str, Any] = {}
            if row.table_json:
                try:
                    payload = json.loads(row.table_json)
                except Exception:
                    payload = {}
            if not payload.get("headers") and row.headers_json:
                try:
                    payload["headers"] = json.loads(row.headers_json)
                except Exception:
                    pass
            if not payload.get("independent_rows"):
                # May live only inside nested provenance on table_json.
                prov = payload.get("provenance") or {}
                if isinstance(prov, dict) and prov.get("independent_rows"):
                    payload["independent_rows"] = prov["independent_rows"]
            pages_val = None
            if row.pages_json:
                try:
                    pages_val = json.loads(row.pages_json)
                except Exception:
                    pages_val = row.pages_json
            sql_summary.append(
                {
                    "table_id": row.table_id,
                    "page": row.page_start,
                    "page_end": row.page_end,
                    "pages": pages_val,
                    "ordinal": row.ordinal,
                    "n_rows": row.n_rows,
                    "n_cols": row.n_cols,
                    "extraction_method": row.extraction_method
                    or payload.get("extraction_engine")
                    or payload.get("extraction_method"),
                    "caption": (row.caption or payload.get("caption") or "")[:120],
                    "headers": (payload.get("headers") or [])[:8],
                    "has_independent_rows": bool(payload.get("independent_rows")),
                    "independent_row_count": len(payload.get("independent_rows") or []),
                    "index_version": row.index_version,
                }
            )
    finally:
        db.close()

    client = get_chroma_client()
    coll_name = get_collection_name(user_id)
    collection = client.get_or_create_collection(name=coll_name)
    # Chroma where filter on document_id.
    try:
        raw = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
    except Exception:
        # Some Chroma builds need $eq.
        raw = collection.get(
            where={"document_id": {"$eq": document_id}},
            include=["documents", "metadatas"],
        )

    ids = list(raw.get("ids") or [])
    docs = list(raw.get("documents") or [])
    metas = list(raw.get("metadatas") or [])
    table_chunks = []
    by_type: Dict[str, int] = {}
    ste_chunks = 0
    for i, meta in enumerate(metas):
        meta = meta or {}
        ctype = str(meta.get("chunk_type") or meta.get("index_kind") or "")
        by_type[ctype or "unknown"] = by_type.get(ctype or "unknown", 0) + 1
        engine = str(
            meta.get("extraction_method")
            or meta.get("extraction_engine")
            or ""
        )
        is_table = ctype in {
            "table",
            "table_card",
            "table_row_window",
            "table_summary",
            "table_row",
        } or bool(meta.get("table_id")) or bool(meta.get("table_json"))
        if not is_table:
            continue
        if "ste_" in engine:
            ste_chunks += 1
        preview = (docs[i] or "")[:240].replace("\n", " ")
        table_json = meta.get("table_json") or ""
        tj: Dict[str, Any] = {}
        if table_json:
            try:
                tj = json.loads(table_json)
            except Exception:
                tj = {}
        table_chunks.append(
            {
                "id": ids[i] if i < len(ids) else None,
                "chunk_type": ctype,
                "page": meta.get("page") or meta.get("page_start"),
                "table_id": meta.get("table_id") or tj.get("table_id"),
                "extraction_method": engine
                or tj.get("extraction_engine")
                or tj.get("extraction_method"),
                "row_index": meta.get("row_index"),
                "independent_row_id": tj.get("independent_row_id"),
                "headers": (tj.get("headers") or [])[:6],
                "n_rows_preview": len(tj.get("rows") or []),
                "text_preview": preview,
            }
        )

    return {
        "chroma_total_chunks": len(ids),
        "chroma_chunk_types": by_type,
        "chroma_table_chunks": len(table_chunks),
        "chroma_ste_table_chunks": ste_chunks,
        "chroma_table_samples": table_chunks[:40],
        "sql_table_count": len(sql_summary),
        "sql_tables": sql_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Run STE on every page (slow). Default uses known table pages.",
    )
    parser.add_argument(
        "--pages",
        default=DEFAULT_STE_PAGES,
        help="Comma-separated 1-based pages for STE (ignored with --all-pages).",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Skip ingest; only dump current Chroma/SQL table vectors.",
    )
    parser.add_argument(
        "--document-id",
        default="",
        help="Force a specific document UUID (e.g. the dev@genova.ai copy).",
    )
    parser.add_argument(
        "--skip-textract",
        action="store_true",
        help="Disable Textract DetectDocumentText (faster; native/OCR prose + STE tables).",
    )
    args = parser.parse_args()

    if not XOLAIR_PDF.is_file():
        logger.error("Missing PDF: %s", XOLAIR_PDF)
        return 1

    _configure_env(all_pages=bool(args.all_pages), pages=str(args.pages))
    if args.skip_textract:
        os.environ["PDF_TEXTRACT_FALLBACK_ENABLED"] = "false"
        os.environ["PDF_TEXTRACT_TABLES_ENABLED"] = "false"
    settings = _clear_settings()
    if args.skip_textract:
        settings.PDF_TEXTRACT_FALLBACK_ENABLED = False
        settings.PDF_TEXTRACT_TABLES_ENABLED = False
    doc = _resolve_doc((args.document_id or "").strip() or None)
    if doc is None:
        logger.error("No xolair_prescribing.pdf document row found in DB")
        return 1

    document_id = str(doc.id)
    user_id = str(doc.user_id)
    logger.info(
        "Target document_id=%s user_id=%s status=%s ste_pages=%s",
        document_id,
        user_id,
        doc.status,
        settings.PDF_STE_PAGES or "ALL",
    )

    report: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "document_id": document_id,
        "user_id": user_id,
        "pdf": str(XOLAIR_PDF),
        "ste_pages": settings.PDF_STE_PAGES or "ALL",
        "settings": {
            "PDF_STE_TABLES_ENABLED": bool(settings.PDF_STE_TABLES_ENABLED),
            "PDF_STE_BEDROCK_ENABLED": bool(settings.PDF_STE_BEDROCK_ENABLED),
            "PDF_STE_BACKEND": getattr(settings, "PDF_STE_BACKEND", ""),
            "BEDROCK_MODEL_ID": settings.BEDROCK_MODEL_ID,
            "EMBEDDING_PROVIDER": settings.EMBEDDING_PROVIDER,
            "CHROMA_PERSIST_DIR": settings.CHROMA_PERSIST_DIR,
            "TABLE_RAG_INDEX_VERSION": settings.TABLE_RAG_INDEX_VERSION,
        },
    }

    if not args.inspect_only:
        from app.database import SessionLocal
        from app.models import Document
        from app.services.document_ingestion import ingest_document

        local = _ensure_local_copy(user_id, XOLAIR_PDF)
        db = SessionLocal()
        try:
            row = db.query(Document).filter(Document.id == document_id).first()
            row.filename = local.name
            row.original_name = XOLAIR_PDF.name
            row.file_size = local.stat().st_size
            row.status = "pending"
            row.error_message = None
            row.is_deleted = False
            row.storage_backend = "local"
            row.s3_key = None
            row.chunk_strategy = "semantic_llm"
            db.commit()
        finally:
            db.close()

        t0 = time.perf_counter()
        logger.info("Starting ingest with STE Textract+Bedrock …")
        ingest_document(document_id, str(local), XOLAIR_PDF.name, user_id)
        report["ingest_elapsed_s"] = round(time.perf_counter() - t0, 2)

        db = SessionLocal()
        try:
            row = db.query(Document).filter(Document.id == document_id).first()
            report["status"] = row.status
            report["page_count"] = row.page_count
            report["chunk_count"] = row.chunk_count
            report["index_version"] = row.index_version
            report["error_message"] = row.error_message
            report["chunk_strategy"] = row.chunk_strategy
        finally:
            db.close()
        logger.info(
            "Ingest done status=%s chunks=%s elapsed=%ss",
            report.get("status"),
            report.get("chunk_count"),
            report.get("ingest_elapsed_s"),
        )

    report["inspection"] = _inspect_vectors(document_id, user_id)

    out_dir = ROOT / "data" / "layout_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xolair_ste_reindex_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    insp = report["inspection"]
    print("\n=== Xolair STE vector storage ===")
    print(f"document_id: {document_id}")
    print(f"status: {report.get('status')}")
    print(f"ste_pages: {report.get('ste_pages')}")
    print(f"chroma_total_chunks: {insp['chroma_total_chunks']}")
    print(f"chroma_chunk_types: {insp['chroma_chunk_types']}")
    print(f"chroma_table_chunks: {insp['chroma_table_chunks']}")
    print(f"chroma_ste_table_chunks: {insp['chroma_ste_table_chunks']}")
    print(f"sql_table_count: {insp['sql_table_count']}")
    for t in insp["sql_tables"][:12]:
        print(
            f"  SQL table page={t['page']} engine={t['extraction_method']} "
            f"rows={t['n_rows']} cols={t['n_cols']} indep={t['independent_row_count']} "
            f"headers={t['headers']}"
        )
    for c in insp["chroma_table_samples"][:15]:
        print(
            f"  CHROMA {c['chunk_type']} page={c['page']} engine={c['extraction_method']} "
            f"table_id={c['table_id']} row={c.get('row_index')} "
            f"indep_row={c.get('independent_row_id')}"
        )
        print(f"    text: {c['text_preview'][:160]}")
    print(f"\nFull report: {out_path}")
    return 0 if report.get("status", "ready") in {"ready", "indexed", "completed", "processed", None} else 1


if __name__ == "__main__":
    raise SystemExit(main())

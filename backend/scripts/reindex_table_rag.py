#!/usr/bin/env python3
"""Re-index documents for TableRAG v2 (card/window + multipage + profiles).

Purges Chroma/BM25 + document_tables, then re-runs ingest_document.

Usage:
  python scripts/reindex_table_rag.py --all
  python scripts/reindex_table_rag.py --document-id <uuid>
  python scripts/reindex_table_rag.py --original-name xolair_prescribing.pdf \\
      --original-name 03_Vendor_Invoice_NonPO_FB60.pdf
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reindex_table_rag")


def _resolve_targets(
    *,
    all_docs: bool,
    document_ids: Sequence[str],
    original_names: Sequence[str],
    stale_only: bool,
) -> List[Any]:
    from app.config import get_settings
    from app.database import SessionLocal
    from app.models import Document, DocumentTable

    settings = get_settings()
    current = str(getattr(settings, "TABLE_RAG_INDEX_VERSION", "v2_card_window") or "")
    db = SessionLocal()
    try:
        q = db.query(Document).filter(
            Document.is_deleted.is_(False),
            Document.status.in_(("ready", "indexed", "completed", "processed")),
        )
        docs = q.order_by(Document.uploaded_at.desc()).all()
        # Also accept pending/failed if explicitly named.
        if original_names or document_ids:
            q2 = db.query(Document).filter(Document.is_deleted.is_(False))
            docs = q2.order_by(Document.uploaded_at.desc()).all()

        selected: List[Any] = []
        name_needles = [n.lower() for n in original_names]
        id_set = {str(i) for i in document_ids if i}

        for doc in docs:
            doc_id = str(doc.id)
            oname = str(doc.original_name or "").lower()
            if id_set and doc_id not in id_set:
                continue
            if name_needles and not any(n in oname for n in name_needles):
                continue
            if all_docs or id_set or name_needles:
                if stale_only and not (id_set or name_needles):
                    # Check DocumentTable index_version parity.
                    versions = [
                        str(r.index_version or "")
                        for r in db.query(DocumentTable)
                        .filter(DocumentTable.document_id == doc.id)
                        .all()
                    ]
                    doc_ver = str(getattr(doc, "index_version", None) or "")
                    if versions and all(v == current for v in versions) and doc_ver == current:
                        continue
                selected.append(doc)
                continue
            if stale_only:
                versions = [
                    str(r.index_version or "")
                    for r in db.query(DocumentTable)
                    .filter(DocumentTable.document_id == doc.id)
                    .all()
                ]
                doc_ver = str(getattr(doc, "index_version", None) or "")
                if not versions or any(v != current for v in versions) or doc_ver != current:
                    selected.append(doc)
        # Detach for use outside session.
        out = []
        for doc in selected:
            out.append(
                type(
                    "DocRef",
                    (),
                    {
                        "id": str(doc.id),
                        "user_id": str(doc.user_id),
                        "filename": str(doc.filename or ""),
                        "original_name": str(doc.original_name or ""),
                        "filepath": str(getattr(doc, "filepath", None) or ""),
                        "status": str(doc.status or ""),
                    },
                )()
            )
        return out
    finally:
        db.close()


def _resolve_filepath(doc: Any) -> Path:
    from app.config import get_settings

    settings = get_settings()
    candidates = []
    if getattr(doc, "filepath", None):
        candidates.append(Path(doc.filepath))
    upload_dir = Path(getattr(settings, "UPLOAD_DIR", ROOT / "data" / "uploads"))
    candidates.append(upload_dir / doc.filename)
    candidates.append(ROOT / "data" / "uploads" / doc.filename)
    candidates.append(ROOT / "data" / "sample_subsets" / doc.original_name)
    for path in candidates:
        if path and path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot locate file for {doc.original_name} ({doc.id}); tried {candidates}"
    )


def _purge(doc: Any) -> None:
    from app.database import SessionLocal
    from app.rag.table_rag import delete_document_tables
    from app.rag.vectordb.vectorstore import delete_document_chunks

    logger.info("Purging Chroma/BM25 + document_tables for %s", doc.original_name)
    try:
        delete_document_chunks(document_id=doc.id, user_id=doc.user_id)
    except Exception as exc:
        logger.warning("delete_document_chunks: %s", exc)
    db = SessionLocal()
    try:
        delete_document_tables(document_id=doc.id, db=db)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("delete_document_tables: %s", exc)
    finally:
        db.close()


def _verify_parity(doc: Any) -> Dict[str, Any]:
    from app.config import get_settings
    from app.database import SessionLocal
    from app.models import DocumentTable
    from app.rag.vectordb.vectorstore import get_chroma_client, get_collection_name

    settings = get_settings()
    current = str(getattr(settings, "TABLE_RAG_INDEX_VERSION", "v2_card_window") or "")
    db = SessionLocal()
    try:
        rows = (
            db.query(DocumentTable)
            .filter(DocumentTable.document_id == doc.id)
            .all()
        )
        sql_ids = {str(r.table_id) for r in rows}
        sql_versions = {str(r.index_version or "") for r in rows}
    finally:
        db.close()

    chroma_ids: set[str] = set()
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=get_collection_name(doc.user_id))
        got = collection.get(
            where={"document_id": {"$eq": doc.id}},
            include=["metadatas"],
        )
        for meta in got.get("metadatas") or []:
            if not isinstance(meta, dict):
                continue
            kind = str(meta.get("chunk_type") or meta.get("index_kind") or "")
            if kind in ("table_card", "table_summary"):
                tid = str(meta.get("table_id") or "").strip()
                if tid:
                    chroma_ids.add(tid)
    except Exception as exc:
        logger.warning("Chroma parity check skipped: %s", exc)

    missing_in_sql = sorted(chroma_ids - sql_ids)
    missing_in_chroma = sorted(sql_ids - chroma_ids)
    version_ok = (not sql_versions) or (sql_versions == {current})
    ok = not missing_in_sql and version_ok
    return {
        "document_id": doc.id,
        "original_name": doc.original_name,
        "sql_table_count": len(sql_ids),
        "chroma_card_count": len(chroma_ids),
        "missing_in_sql": missing_in_sql,
        "missing_in_chroma": missing_in_chroma,
        "sql_index_versions": sorted(sql_versions),
        "expected_index_version": current,
        "ok": ok,
    }


def reindex_one(doc: Any, *, skip_purge: bool = False) -> Dict[str, Any]:
    from app.services.document_ingestion import ingest_document

    filepath = _resolve_filepath(doc)
    if not skip_purge:
        _purge(doc)
    logger.info("Re-ingesting %s from %s", doc.original_name, filepath)
    ingest_document(
        document_id=doc.id,
        filepath=str(filepath),
        original_name=doc.original_name,
        user_id=doc.user_id,
    )
    return _verify_parity(doc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stale-only", action="store_true", default=True)
    parser.add_argument("--include-fresh", action="store_true", help="Ignore stale filter")
    parser.add_argument("--document-id", action="append", default=[])
    parser.add_argument("--original-name", action="append", default=[])
    parser.add_argument("--skip-purge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stale_only = bool(args.stale_only) and not args.include_fresh
    if not args.all and not args.document_id and not args.original_name:
        # Default: stale docs only.
        args.all = True

    targets = _resolve_targets(
        all_docs=bool(args.all),
        document_ids=args.document_id,
        original_names=args.original_name,
        stale_only=stale_only and not (args.document_id or args.original_name),
    )
    if not targets:
        logger.info("No documents matched.")
        return 0

    logger.info("Matched %d document(s)", len(targets))
    reports = []
    for doc in targets:
        logger.info("→ %s (%s) status=%s", doc.original_name, doc.id, doc.status)
        if args.dry_run:
            continue
        try:
            report = reindex_one(doc, skip_purge=args.skip_purge)
            reports.append(report)
            logger.info("parity ok=%s sql=%s chroma_cards=%s", report["ok"], report["sql_table_count"], report["chroma_card_count"])
            if not report["ok"]:
                logger.warning("parity issues: %s", json.dumps(report, indent=2))
        except Exception as exc:
            logger.exception("Failed reindex for %s: %s", doc.original_name, exc)
            reports.append(
                {
                    "document_id": doc.id,
                    "original_name": doc.original_name,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    out_path = ROOT / "data" / "layout_reports" / "table_rag_reindex_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    failed = [r for r in reports if not r.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

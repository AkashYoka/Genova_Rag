"""Re-index data/all_Domain PDFs + Xolair via Textract DetectDocumentText.

Uses the live ingestion pipeline (semantic chunker + Bedrock Titan embeddings).
Docling remains disabled in the active path. Writes a report under
``data/layout_reports/``.

Usage
-----
  cd backend
  python scripts/reindex_all_domain_textract.py
  python scripts/reindex_all_domain_textract.py --skip-xolair
  python scripts/reindex_all_domain_textract.py --only-xolair
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PDF_DOCLING_TABLES_ENABLED", "false")
os.environ.setdefault("PDF_TEXTRACT_FALLBACK_ENABLED", "true")
os.environ.setdefault("PDF_TEXTRACT_TABLES_ENABLED", "true")
os.environ.setdefault("PDF_TEXTRACT_ASYNC_MULTIPAGE", "true")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reindex_all_domain")

ALL_DOMAIN = ROOT.parent / "data" / "all_Domain"
XOLAIR_PDF = ROOT / "data" / "sample_subsets" / "xolair_prescribing.pdf"
XOLAIR_DOC_ID = "43ae024f-3c0d-40f5-aa49-83e9f3682641"
XOLAIR_USER_ID = "f3d989fb-79af-4fe2-9360-12b06772348c"
DOMAIN_USER_ID = "all-domain-textract-reindex"


def _clear_settings() -> Any:
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    # Belt-and-suspenders for process-local overrides.
    settings.PDF_DOCLING_TABLES_ENABLED = False
    settings.PDF_TEXTRACT_FALLBACK_ENABLED = True
    settings.PDF_TEXTRACT_TABLES_ENABLED = True
    settings.PDF_TEXTRACT_ASYNC_MULTIPAGE = True
    settings.SEMANTIC_CHUNKING_ENABLED = True
    settings.CHUNK_STRATEGY = "semantic_llm"
    settings.EMBEDDING_PROVIDER = "bedrock"
    if not (settings.EMBEDDING_MODEL or "").strip():
        settings.EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
    return settings


def _existing_docs_snapshot() -> List[Dict[str, Any]]:
    from app.database import SessionLocal
    from app.models import Document

    db = SessionLocal()
    try:
        rows = (
            db.query(Document)
            .filter(Document.is_deleted.is_(False))
            .order_by(Document.uploaded_at.desc())
            .all()
        )
        return [
            {
                "id": d.id,
                "original_name": d.original_name,
                "status": d.status,
                "chunk_count": d.chunk_count,
                "page_count": d.page_count,
                "user_id": d.user_id,
            }
            for d in rows
        ]
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


def _upsert_document(
    *,
    user_id: str,
    document_id: Optional[str],
    original_name: str,
    filename: str,
    file_size: int,
) -> str:
    from app.database import SessionLocal
    from app.models import Document

    db = SessionLocal()
    try:
        doc = None
        if document_id:
            doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            # Prefer reusing an active row with the same user + original name.
            doc = (
                db.query(Document)
                .filter(
                    Document.user_id == user_id,
                    Document.original_name == original_name,
                    Document.is_deleted.is_(False),
                )
                .order_by(Document.uploaded_at.desc())
                .first()
            )
        if doc is None:
            doc = Document(
                id=document_id or str(uuid.uuid4()),
                user_id=user_id,
                filename=filename,
                original_name=original_name,
                file_size=file_size,
                status="pending",
                storage_backend="local",
                chunk_strategy="semantic_llm",
            )
            db.add(doc)
        else:
            doc.filename = filename
            doc.original_name = original_name
            doc.file_size = file_size
            doc.status = "pending"
            doc.is_deleted = False
            doc.error_message = None
            doc.storage_backend = "local"
            doc.s3_key = None
            doc.chunk_strategy = "semantic_llm"
        db.commit()
        db.refresh(doc)
        return str(doc.id)
    finally:
        db.close()


def _ingest_one(
    *,
    pdf: Path,
    user_id: str,
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    from app.database import SessionLocal
    from app.models import Document
    from app.rag.ingestion.pdf_layout.textract_tables import (
        get_textract_api_usage,
        reset_textract_api_usage,
    )
    from app.services.document_ingestion import ingest_document

    reset_textract_api_usage()
    local = _ensure_local_copy(user_id, pdf)
    doc_id = _upsert_document(
        user_id=user_id,
        document_id=document_id,
        original_name=pdf.name,
        filename=local.name,
        file_size=local.stat().st_size,
    )
    t0 = time.perf_counter()
    ingest_document(doc_id, str(local), pdf.name, user_id)
    elapsed = time.perf_counter() - t0
    usage = get_textract_api_usage()

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        result = {
            "document_id": doc_id,
            "original_name": pdf.name,
            "user_id": user_id,
            "status": getattr(doc, "status", None),
            "page_count": getattr(doc, "page_count", None),
            "chunk_count": getattr(doc, "chunk_count", None),
            "chunk_strategy": getattr(doc, "chunk_strategy", None),
            "index_version": getattr(doc, "index_version", None),
            "error_message": getattr(doc, "error_message", None),
            "elapsed_s": round(elapsed, 2),
            "textract_api_usage": usage,
            "path": str(pdf),
        }
    finally:
        db.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-xolair", action="store_true")
    parser.add_argument("--only-xolair", action="store_true")
    args = parser.parse_args()

    settings = _clear_settings()
    pdfs = sorted(ALL_DOMAIN.rglob("*.pdf")) if ALL_DOMAIN.exists() else []
    before = _existing_docs_snapshot()

    report: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "all_domain_dir": str(ALL_DOMAIN),
        "pdfs_discovered": [p.name for p in pdfs],
        "pdf_count": len(pdfs),
        "settings": {
            "PDF_DOCLING_TABLES_ENABLED": settings.PDF_DOCLING_TABLES_ENABLED,
            "PDF_TEXTRACT_FALLBACK_ENABLED": settings.PDF_TEXTRACT_FALLBACK_ENABLED,
            "PDF_TEXTRACT_ASYNC_MULTIPAGE": settings.PDF_TEXTRACT_ASYNC_MULTIPAGE,
            "EMBEDDING_PROVIDER": settings.EMBEDDING_PROVIDER,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
            "CHUNK_STRATEGY": settings.CHUNK_STRATEGY,
            "SEMANTIC_CHUNKING_ENABLED": settings.SEMANTIC_CHUNKING_ENABLED,
            "analyze_document_tables": False,
        },
        "before_documents": before,
        "results": [],
        "failed": [],
        "totals": {},
    }

    jobs: List[Dict[str, Any]] = []
    if not args.only_xolair:
        for pdf in pdfs:
            jobs.append({"pdf": pdf, "user_id": DOMAIN_USER_ID, "document_id": None})
    if not args.skip_xolair:
        if not XOLAIR_PDF.exists():
            logger.error("Xolair PDF missing: %s", XOLAIR_PDF)
            report["failed"].append({"pdf": str(XOLAIR_PDF), "error": "missing"})
        else:
            jobs.append(
                {
                    "pdf": XOLAIR_PDF,
                    "user_id": XOLAIR_USER_ID,
                    "document_id": XOLAIR_DOC_ID,
                }
            )

    logger.info(
        "Reindex jobs=%s docling=%s textract=%s async=%s embed=%s chunk=%s",
        len(jobs),
        settings.PDF_DOCLING_TABLES_ENABLED,
        settings.PDF_TEXTRACT_FALLBACK_ENABLED,
        settings.PDF_TEXTRACT_ASYNC_MULTIPAGE,
        settings.EMBEDDING_MODEL,
        settings.CHUNK_STRATEGY,
    )

    pages_total = 0
    chunks_total = 0
    detect_pages = 0
    analyze_pages = 0
    async_jobs = 0

    for job in jobs:
        pdf: Path = job["pdf"]
        logger.info("=== Ingesting %s ===", pdf.name)
        try:
            result = _ingest_one(
                pdf=pdf,
                user_id=job["user_id"],
                document_id=job.get("document_id"),
            )
            report["results"].append(result)
            if result.get("status") != "ready":
                report["failed"].append(result)
            pages_total += int(result.get("page_count") or 0)
            chunks_total += int(result.get("chunk_count") or 0)
            usage = result.get("textract_api_usage") or {}
            detect_pages += int(usage.get("detect_document_text_pages") or 0)
            analyze_pages += int(usage.get("analyze_document_tables_pages") or 0)
            async_jobs += int(usage.get("start_document_text_detection_jobs") or 0)
            logger.info(
                "Done %s status=%s pages=%s chunks=%s usage=%s",
                pdf.name,
                result.get("status"),
                result.get("page_count"),
                result.get("chunk_count"),
                usage,
            )
        except Exception as exc:
            logger.exception("Failed %s: %s", pdf.name, exc)
            entry = {
                "original_name": pdf.name,
                "path": str(pdf),
                "status": "failed",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
            report["results"].append(entry)
            report["failed"].append(entry)

    # DetectDocumentText pricing (public list): $1.50 / 1,000 pages.
    estimated_usd = round((detect_pages / 1000.0) * 1.50, 4)
    report["totals"] = {
        "jobs": len(jobs),
        "succeeded": sum(1 for r in report["results"] if r.get("status") == "ready"),
        "failed": len(report["failed"]),
        "pages_indexed": pages_total,
        "chunks_indexed": chunks_total,
        "detect_document_text_pages": detect_pages,
        "start_document_text_detection_jobs": async_jobs,
        "analyze_document_tables_pages": analyze_pages,
        "estimated_textract_cost_usd": estimated_usd,
        "pricing_note": (
            "Estimate only at $1.50 per 1,000 DetectDocumentText pages. "
            "Does not reverse prior AnalyzeDocument TABLES charges."
        ),
    }
    report["after_documents"] = _existing_docs_snapshot()
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out_dir = ROOT / "data" / "layout_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"reindex_all_domain_textract_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote report %s", out)
    logger.info("TOTALS %s", json.dumps(report["totals"]))
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

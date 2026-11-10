"""Reusable document ingestion pipeline."""
import logging
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from app.models import Document
from app.rag.llm.agent import persist_document_keywords
from app.rag.chunking.chunker import chunk_document, get_page_count
from app.rag.vectordb.vectorstore import store_chunks
from app.config import get_settings
from app.services import s3_storage

logger = logging.getLogger(__name__)
settings = get_settings()


def _update_progress(document_id: str, progress: int, stage: str, error: str = None):
    """Update document progress fields in the database."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.processing_progress = progress
            doc.processing_stage = stage
            if error:
                doc.error_message = error
            db.commit()
    except Exception as e:
        logger.warning("Failed to update progress for %s: %s", document_id, e)
    finally:
        db.close()


def materialize_document_file(
    *,
    user_id: str,
    filename: str,
    s3_key: str | None = None,
    preferred_path: str | None = None,
) -> tuple[str, bool]:
    """Return a local filesystem path for ingestion/view helpers.

    Prefers AWS S3 when the document is stored there. Returns
    ``(path, should_delete)`` so callers can clean up ephemeral downloads.
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path, False

    if s3_storage.is_configured() and (s3_key or filename):
        suffix = Path(filename or s3_key or ".bin").suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.close()
        try:
            ok = s3_storage.download_pdf_to_path(
                str(user_id),
                filename,
                tmp_path,
                key=s3_key,
            )
        except s3_storage.S3StorageError:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        if ok:
            return tmp_path, True
        Path(tmp_path).unlink(missing_ok=True)

    legacy = os.path.join(settings.UPLOAD_DIR, str(user_id), filename)
    if os.path.exists(legacy):
        return legacy, False

    raise FileNotFoundError(
        f"Document file not found in S3 or local storage: {filename}"
    )


def ingest_document(document_id: str, filepath: str, original_name: str, user_id: str):
    """
    Process a document: chunk it, generate embeddings, store vectors, summarize,
    and update the database record.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    cleanup_path: str | None = None
    try:
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted.is_(False),
        ).first()
        if not doc:
            logger.error("Document %s not found for ingestion", document_id)
            return

        doc.status = "processing"
        doc.processing_stage = "extracting"
        doc.processing_progress = 10
        doc.error_message = None
        doc.last_error_traceback = None
        db.commit()

        try:
            working_path, should_cleanup = materialize_document_file(
                user_id=user_id,
                filename=doc.filename,
                s3_key=getattr(doc, "s3_key", None),
                preferred_path=filepath or None,
            )
        except Exception as exc:
            doc.status = "failed"
            doc.processing_progress = 0
            doc.error_message = f"Could not load document bytes for ingestion: {exc}"[:500]
            db.commit()
            logger.error("Materialize failed for %s: %s", document_id, exc)
            return

        if should_cleanup:
            cleanup_path = working_path
        filepath = working_path

        page_count = get_page_count(filepath)
        doc.page_count = page_count
        doc.processing_progress = 20
        db.commit()

        try:
            chunk_kwargs = {}
            if doc.chunk_size is not None:
                chunk_kwargs["chunk_size"] = doc.chunk_size
            if doc.chunk_overlap is not None:
                chunk_kwargs["chunk_overlap"] = doc.chunk_overlap
            doc.processing_stage = "chunking"
            doc.processing_progress = 30
            db.commit()
            chunk_kwargs["document_id"] = document_id
            chunk_kwargs["document_name"] = original_name
            chunks = chunk_document(filepath, **chunk_kwargs)
        except TypeError:
            chunks = chunk_document(filepath)

        # ── Proximity caption pass (PDF only) ────────────────────────────────
        # Write bounding-box-derived captions into image chunks BEFORE store_chunks()
        # so generate_captions_for_chunks() in vectorstore.py only needs to handle
        # the OCR / placeholder fallback for any images without adjacent text.
        ext = filepath.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            try:
                from app.rag.ingestion.vision import extract_captions_from_pdf

                pdf_captions = extract_captions_from_pdf(filepath)
                # Build lookup: page -> [captions in figure_index order]
                caption_map: dict = {}
                for cap in pdf_captions:
                    caption_map.setdefault(cap["page"], []).append(cap)

                fig_counters: dict = {}
                for chunk in chunks:
                    if not chunk.get("image_bytes"):
                        continue
                    page = chunk.get("page", 1)
                    idx = fig_counters.get(page, 0)
                    page_caps = caption_map.get(page, [])
                    if idx < len(page_caps) and page_caps[idx]["caption"]:
                        chunk["image_caption"] = page_caps[idx]["caption"]
                        chunk["bbox"] = str(page_caps[idx]["bbox"])
                    fig_counters[page] = idx + 1
            except Exception as exc:
                logger.warning(
                    "Proximity caption extraction failed for %s: %s", document_id, exc
                )
        # ── End proximity caption pass ────────────────────────────────────────

        if not chunks:
            doc.status = "failed"
            doc.processing_progress = 0
            doc.error_message = "No text could be extracted from the document"
            db.commit()
            return

        doc.processing_progress = 55
        doc.processing_stage = "graph"
        db.commit()

        # ── Knowledge-graph indexing (incremental, per document) ──────────
        # The graph service owns its own session and records an explicit
        # indexing status. A graph failure is logged and marked on the status
        # row, never silently swallowed or allowed to fail the upload.
        try:
            from app.graph.service import index_document as index_document_graph

            graph_result = index_document_graph(document_id, user_id, chunks)
            if not graph_result.get("ok") and graph_result.get("status") != "disabled":
                logger.warning(
                    "Knowledge-graph indexing failed for document %s: %s",
                    document_id,
                    graph_result.get("error") or graph_result.get("status"),
                )
        except Exception as e:
            logger.warning(
                "Could not run knowledge-graph indexing for document %s: %s",
                document_id,
                e,
            )

        doc.processing_progress = 70
        doc.processing_stage = "embedding"
        db.commit()

        def _on_embed_progress(stored: int, total: int) -> None:
            if total <= 0:
                return
            # Map embedding batches into overall progress 70% → 90%.
            pct = 70 + int((stored / total) * 20)
            _update_progress(document_id, min(pct, 90), "embedding")

        chunk_count = store_chunks(
            chunks=chunks,
            document_id=document_id,
            filename=original_name,
            user_id=user_id,
            on_progress=_on_embed_progress,
        )

        persist_document_keywords(doc, chunks, db)

        doc.processing_progress = 92
        db.commit()

        try:
            from app.rag.utils.summarizer import generate_document_summary

            summary = generate_document_summary(filepath, max_sentences=2)
            if summary:
                doc.summary = summary
                db.commit()
        except Exception as e:
            logger.warning("Could not generate summary for document %s: %s", document_id, e)
            doc.summary = None

        # Automatic domain classification for dynamic prompt selection
        try:
            from app.services.prompt_management.detector import classify_document_domain

            sample = " ".join(
                (c.get("text") or "")[:200] for c in (chunks or [])[:8]
            )
            detection = classify_document_domain(
                db,
                filename=original_name,
                summary=doc.summary,
                keywords=doc.keywords,
                sample_text=sample,
            )
            doc.detected_domain = detection.domain
            doc.domain_confidence = float(detection.confidence)
            db.commit()
            logger.info(
                "Document %s domain=%s confidence=%.2f",
                document_id,
                detection.domain,
                detection.confidence,
            )
        except Exception as e:
            logger.warning(
                "Domain classification skipped for document %s: %s",
                document_id,
                e,
            )

        # ── URL extraction pass (PDF only) ────────────────────────────────
        ext = filepath.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            try:
                from app.rag.ingestion.url_extractor import extract_urls_from_pdf
                import json

                urls = extract_urls_from_pdf(filepath)
                doc.extracted_urls = json.dumps(urls) if urls else None
                db.commit()
                logger.info(
                    "Extracted %s URLs from document %s",
                    len(urls),
                    document_id,
                )
            except Exception as exc:
                logger.warning(
                    "URL extraction failed for document %s: %s",
                    document_id,
                    exc,
                )
        # ── End URL extraction pass ───────────────────────────────────────

        doc.chunk_count = chunk_count
        doc.status = "ready"
        doc.processing_progress = 100
        doc.processing_stage = "completed"
        doc.completed_at = datetime.now(timezone.utc)
        doc.error_message = None
        db.commit()

        logger.info(
            "Document %s ingested: %s pages, %s chunks",
            document_id,
            page_count,
            chunk_count,
        )

    except Exception as e:
        logger.error("Ingestion error for %s: %s", document_id, e)
        db.rollback()
        try:
            doc = db.query(Document).filter(
                Document.id == document_id,
                Document.is_deleted.is_(False),
            ).first()
            if doc:
                doc.status = "failed"
                doc.processing_progress = 0
                doc.error_message = str(e)[:500]
                doc.last_error_traceback = traceback.format_exc()[:2000]
                db.commit()
        except Exception:
            logger.exception("Failed to mark document %s as failed", document_id)
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not remove temp ingest file %s", cleanup_path)
        db.close()

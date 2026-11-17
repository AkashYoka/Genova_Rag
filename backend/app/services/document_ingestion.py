"""Reusable document ingestion pipeline."""
import json
import logging
import os
import tempfile
import threading
import time
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

# Debounce SQLite progress writes (reduces lock contention on long STE runs).
_progress_lock = threading.Lock()
_progress_last_write: dict[str, float] = {}
_progress_pending: dict[str, tuple[int, str, str | None]] = {}


def _update_progress(document_id: str, progress: int, stage: str, error: str = None):
    """Update document progress fields in the database (debounced)."""
    from app.database import SessionLocal

    min_interval_ms = int(
        getattr(settings, "PDF_INGEST_PROGRESS_MIN_INTERVAL_MS", 750) or 0
    )
    force = error is not None or int(progress) >= 100 or int(progress) <= 5
    now = time.monotonic()
    with _progress_lock:
        last = _progress_last_write.get(document_id, 0.0)
        if (
            not force
            and min_interval_ms > 0
            and (now - last) * 1000.0 < float(min_interval_ms)
        ):
            _progress_pending[document_id] = (int(progress), stage, error)
            return
        _progress_pending.pop(document_id, None)
        _progress_last_write[document_id] = now

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
    """Unified staged ingestion:

    materialize → layout_extract → semantic_chunk → image_captions
    → table_rag → embed_index → hierarchy → enrichment → completed

    PDF tables (STE AnalyzeDocument + DetectDocumentText) run inside
    ``layout_extract`` via the table extraction coordinator.
    """
    from app.database import SessionLocal
    from app.rag.ingestion.stages import (
        STAGE_COMPLETED,
        STAGE_EMBED_INDEX,
        STAGE_ENRICHMENT,
        STAGE_HIERARCHY,
        STAGE_IMAGE_CAPTIONS,
        STAGE_LAYOUT_EXTRACT,
        STAGE_MATERIALIZE,
        STAGE_SEMANTIC_CHUNK,
        STAGE_TABLE_RAG,
        IngestContext,
        IngestPipeline,
    )

    db = SessionLocal()
    cleanup_path: str | None = None
    ctx: IngestContext | None = None
    try:
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted.is_(False),
        ).first()
        if not doc:
            logger.error("Document %s not found for ingestion", document_id)
            return

        doc.status = "processing"
        doc.processing_stage = STAGE_MATERIALIZE
        doc.processing_progress = 5
        doc.error_message = None
        doc.last_error_traceback = None
        db.commit()

        requested_strategy = (
            getattr(doc, "chunk_strategy", None)
            or getattr(settings, "CHUNK_STRATEGY", "semantic_llm")
            or "semantic_llm"
        ).lower()
        chunk_size = doc.chunk_size
        chunk_overlap = doc.chunk_overlap

        ctx = IngestContext(
            document_id=document_id,
            user_id=user_id,
            original_name=original_name,
            filepath=filepath or "",
            requested_strategy=requested_strategy,
        )

        def _on_progress(stage: str, pct: int) -> None:
            _update_progress(document_id, int(pct), stage)

        pipeline = IngestPipeline(ctx, on_progress=_on_progress)

        # ── materialize ────────────────────────────────────────────────────
        def _stage_materialize() -> None:
            nonlocal cleanup_path
            working_path, should_cleanup = materialize_document_file(
                user_id=user_id,
                filename=doc.filename,
                s3_key=getattr(doc, "s3_key", None),
                preferred_path=filepath or None,
            )
            if should_cleanup:
                cleanup_path = working_path
                ctx.cleanup_path = working_path
            ctx.filepath = working_path
            ctx.page_count = get_page_count(working_path)
            doc.page_count = ctx.page_count
            db.commit()

        try:
            pipeline.run_stage(STAGE_MATERIALIZE, _stage_materialize)
        except Exception as exc:
            doc.status = "failed"
            doc.processing_progress = 0
            doc.processing_stage = "failed"
            doc.error_message = f"Could not load document bytes for ingestion: {exc}"[:500]
            db.commit()
            logger.error("Materialize failed for %s: %s", document_id, exc)
            return

        ext = ctx.filepath.rsplit(".", 1)[-1].lower()

        # ── layout_extract (PDF only; includes unified table plan) ─────────
        def _stage_layout_extract() -> None:
            from app.rag.ingestion.pdf_layout.parser import parse_pdf

            def _layout_progress(stage: str, pct: int) -> None:
                mapped = pipeline.map_nested(STAGE_LAYOUT_EXTRACT, pct)
                pipeline.progress(STAGE_LAYOUT_EXTRACT, mapped)

            parsed = parse_pdf(
                ctx.filepath,
                document_id=document_id,
                document_name=original_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress=_layout_progress,
            )
            ctx.parsed = parsed
            ctx.layout_report = dict(getattr(parsed, "report", None) or {})
            ctx.page_count = int(getattr(parsed, "page_count", 0) or ctx.page_count)
            doc.page_count = ctx.page_count
            db.commit()

        if ext == "pdf":
            pipeline.run_stage(STAGE_LAYOUT_EXTRACT, _stage_layout_extract)
        else:
            # Non-PDF: skip layout stage record for clarity.
            logger.info(
                "INGEST_STAGE skip document_id=%s stage=%s reason=non_pdf",
                document_id,
                STAGE_LAYOUT_EXTRACT,
            )

        # ── semantic_chunk ─────────────────────────────────────────────────
        def _stage_semantic_chunk() -> None:
            chunk_kwargs = {
                "document_id": document_id,
                "document_name": original_name,
                "strategy": requested_strategy,
            }
            if chunk_size is not None:
                chunk_kwargs["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                chunk_kwargs["chunk_overlap"] = chunk_overlap

            def _chunk_progress(stage: str, pct: int) -> None:
                mapped = pipeline.map_nested(STAGE_SEMANTIC_CHUNK, pct)
                pipeline.progress(STAGE_SEMANTIC_CHUNK, mapped)

            if ext == "pdf" and ctx.parsed is not None:
                from app.rag.chunking.semantic.service import (
                    semantic_chunk_document,
                    should_use_semantic_chunking,
                )

                if should_use_semantic_chunking(
                    ctx.filepath, strategy=requested_strategy
                ):
                    chunks = semantic_chunk_document(
                        ctx.filepath,
                        progress=_chunk_progress,
                        parsed=ctx.parsed,
                        **chunk_kwargs,
                    )
                else:
                    chunk_kwargs["progress"] = _chunk_progress
                    chunks = chunk_document(ctx.filepath, **chunk_kwargs)
            else:
                try:
                    chunk_kwargs["progress"] = _chunk_progress
                    chunks = chunk_document(ctx.filepath, **chunk_kwargs)
                except TypeError:
                    chunks = chunk_document(ctx.filepath)

            ctx.chunks = chunks
            ctx.chunk_hierarchy = getattr(chunks, "hierarchy", None) or []
            ctx.chunk_report = dict(getattr(chunks, "report", None) or {})
            ctx.chunk_strategy = (
                getattr(chunks, "strategy", None) or requested_strategy
            ).lower()
            ctx.chunk_index_version = getattr(chunks, "index_version", None) or ""

            # Attach layout table plan into chunk report for UI/audit.
            if ctx.layout_report:
                te = ctx.layout_report.get("table_extraction") or {}
                ctx.chunk_report.setdefault("layout_extract", {
                    "table_engine": ctx.layout_report.get("table_engine") or te.get("engine"),
                    "table_extraction": te,
                    "elapsed_s": ctx.layout_report.get("elapsed_s"),
                })
                ste_info = ((te.get("engine_info") or {}).get("ste")) or {}
                if ste_info and "ste_tables" not in ctx.chunk_report:
                    ctx.chunk_report["ste_tables"] = {
                        "table_count": ste_info.get("table_count"),
                        "bedrock_usage": ste_info.get("bedrock_usage"),
                        "elapsed_s": ste_info.get("elapsed_s"),
                        "extraction_engine": ste_info.get("extraction_engine"),
                        "output_dir": ste_info.get("output_dir"),
                    }

            ste_usage = (
                (ctx.chunk_report.get("ste_tables") or {}).get("bedrock_usage") or {}
            )
            if ste_usage and (
                ste_usage.get("input_tokens") is not None
                or ste_usage.get("total_tokens") is not None
            ):
                try:
                    from app.services.usage_tracker import record_llm_usage

                    record_llm_usage(
                        db,
                        user_id=user_id,
                        usage=ste_usage,
                        usage_type="ste_table_reconstruction",
                        provider="bedrock",
                        model_id=str(
                            ste_usage.get("model_id")
                            or getattr(settings, "PDF_STE_BEDROCK_MODEL_ID", "")
                            or getattr(settings, "BEDROCK_MODEL_ID", "")
                            or ""
                        )
                        or None,
                        document_id=document_id,
                        commit=False,
                    )
                except Exception as usage_exc:
                    logger.warning(
                        "Failed to record STE Bedrock usage for %s: %s",
                        document_id,
                        usage_exc,
                    )

            if not ctx.chunk_index_version:
                from app.rag.chunking.semantic.version import resolve_index_version

                ctx.chunk_index_version = resolve_index_version(
                    ctx.chunk_strategy,
                    used_fallback=not bool(ctx.chunk_hierarchy),
                )

        pipeline.run_stage(STAGE_SEMANTIC_CHUNK, _stage_semantic_chunk)

        if not ctx.chunks:
            doc.status = "failed"
            doc.processing_progress = 0
            doc.processing_stage = "failed"
            doc.error_message = "No text could be extracted from the document"
            db.commit()
            return

        # ── image_captions ─────────────────────────────────────────────────
        def _stage_image_captions() -> None:
            if ext != "pdf":
                return
            from app.rag.ingestion.vision import extract_captions_from_pdf

            pdf_captions = extract_captions_from_pdf(ctx.filepath)
            caption_map: dict = {}
            for cap in pdf_captions:
                caption_map.setdefault(cap["page"], []).append(cap)
            fig_counters: dict = {}
            for chunk in ctx.chunks:
                if not chunk.get("image_bytes"):
                    continue
                page = chunk.get("page", 1)
                idx = fig_counters.get(page, 0)
                page_caps = caption_map.get(page, [])
                if idx < len(page_caps) and page_caps[idx]["caption"]:
                    chunk["image_caption"] = page_caps[idx]["caption"]
                    chunk["bbox"] = str(page_caps[idx]["bbox"])
                fig_counters[page] = idx + 1

        pipeline.run_stage(
            STAGE_IMAGE_CAPTIONS, _stage_image_captions, fatal=False
        )

        # ── table_rag ──────────────────────────────────────────────────────
        def _stage_table_rag() -> None:
            from app.rag.table_rag import expand_table_rag_chunks, persist_document_tables

            table_index_version = str(
                getattr(settings, "TABLE_RAG_INDEX_VERSION", "") or ""
            ).strip() or (ctx.chunk_index_version or "")
            ctx.table_index_version = table_index_version
            store_chunk_list, table_records = expand_table_rag_chunks(
                ctx.chunks,
                document_id=document_id,
                filename=original_name,
                index_version=table_index_version,
                enabled=bool(getattr(settings, "TABLE_RAG_ENABLED", True)),
                max_rows_per_table=int(
                    getattr(settings, "TABLE_RAG_MAX_ROWS_PER_TABLE", 200) or 200
                ),
            )
            persist_document_tables(
                document_id=document_id,
                user_id=user_id,
                filename=original_name,
                tables=table_records or [],
                index_version=table_index_version,
                db=db,
            )
            ctx.store_chunk_list = list(store_chunk_list)
            ctx.table_records = list(table_records or [])
            logger.info(
                "TableRAG %s for document %s (tables=%s)",
                "stored" if ctx.table_records else "cleared",
                document_id,
                len(ctx.table_records),
            )

        try:
            pipeline.run_stage(STAGE_TABLE_RAG, _stage_table_rag)
        except Exception as exc:
            logger.warning(
                "TableRAG indexing failed for document %s (continuing text RAG): %s",
                document_id,
                exc,
            )
            ctx.store_chunk_list = list(ctx.chunks)
            ctx.table_records = []
            try:
                from app.rag.table_rag.store import delete_document_tables

                delete_document_tables(document_id, db=db)
            except Exception as clear_exc:
                logger.warning(
                    "Failed to clear stale document_tables for %s: %s",
                    document_id,
                    clear_exc,
                )

        if not ctx.store_chunk_list:
            ctx.store_chunk_list = list(ctx.chunks)

        # ── embed_index ────────────────────────────────────────────────────
        def _stage_embed_index() -> None:
            def _on_embed_progress(stored: int, total: int) -> None:
                if total <= 0:
                    return
                local = int((stored / total) * 100)
                pipeline.progress(
                    STAGE_EMBED_INDEX, pipeline.map_nested(STAGE_EMBED_INDEX, local)
                )

            ctx.indexed_chunk_count = store_chunks(
                chunks=ctx.store_chunk_list,
                document_id=document_id,
                filename=original_name,
                user_id=user_id,
                on_progress=_on_embed_progress,
            )

        pipeline.run_stage(STAGE_EMBED_INDEX, _stage_embed_index)

        hierarchy_chunks = [
            c
            for c in ctx.store_chunk_list
            if str(c.get("chunk_type") or "")
            not in (
                "table_summary",
                "table_interpretation",
                "table_row",
                "table_card",
                "table_row_window",
            )
        ]
        ctx.document_chunk_count = len(hierarchy_chunks)

        # ── hierarchy ──────────────────────────────────────────────────────
        def _stage_hierarchy() -> None:
            if not ctx.chunk_hierarchy:
                return
            from app.rag.chunking.semantic.hierarchy_store import persist_hierarchy

            hierarchy_counts = persist_hierarchy(
                document_id=document_id,
                user_id=user_id,
                sections=ctx.chunk_hierarchy,
                chunks=hierarchy_chunks,
                index_version=ctx.chunk_index_version or "",
                db=db,
            )
            logger.info(
                "HIERARCHY_FLOW %s",
                {
                    "event": "hierarchy_persisted",
                    "document_id": document_id,
                    "sections": hierarchy_counts.get("sections", 0),
                    "chunks": hierarchy_counts.get("chunks", 0),
                    "index_version": ctx.chunk_index_version,
                },
            )

        pipeline.run_stage(STAGE_HIERARCHY, _stage_hierarchy, fatal=False)

        doc.chunk_strategy = ctx.chunk_strategy
        doc.index_version = ctx.chunk_index_version
        report_payload = dict(ctx.chunk_report or {})
        report_payload["ingest_pipeline"] = ctx.report()
        doc.chunking_report = json.dumps(report_payload, ensure_ascii=False)
        db.commit()

        # ── enrichment ─────────────────────────────────────────────────────
        def _stage_enrichment() -> None:
            persist_document_keywords(doc, ctx.chunks, db)
            try:
                from app.rag.utils.summarizer import generate_document_summary

                summary = generate_document_summary(
                    ctx.filepath, max_sentences=2, chunks=list(ctx.chunks or [])
                )
                if summary:
                    doc.summary = summary
                    db.commit()
            except Exception as e:
                logger.warning(
                    "Could not generate summary for document %s: %s", document_id, e
                )
                doc.summary = None

            try:
                from app.services.prompt_management.detector import (
                    classify_document_domain,
                )

                sample = " ".join(
                    (c.get("text") or "")[:200] for c in (ctx.chunks or [])[:8]
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

            if ext == "pdf":
                try:
                    from app.rag.ingestion.url_extractor import extract_urls_from_pdf

                    urls = extract_urls_from_pdf(ctx.filepath)
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

        pipeline.run_stage(STAGE_ENRICHMENT, _stage_enrichment, fatal=False)

        doc.chunk_count = ctx.document_chunk_count
        doc.status = "ready"
        doc.processing_progress = 100
        doc.processing_stage = STAGE_COMPLETED
        doc.completed_at = datetime.now(timezone.utc)
        doc.error_message = None
        # Refresh report with completed stages.
        report_payload = dict(ctx.chunk_report or {})
        report_payload["ingest_pipeline"] = ctx.report()
        doc.chunking_report = json.dumps(report_payload, ensure_ascii=False)
        db.commit()

        logger.info(
            "Document %s ingested: %s pages, %s document chunks (%s indexed) stages=%s",
            document_id,
            ctx.page_count,
            ctx.document_chunk_count,
            ctx.indexed_chunk_count,
            [s.name for s in ctx.stages],
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
                doc.processing_stage = "failed"
                doc.error_message = str(e)[:500]
                doc.last_error_traceback = traceback.format_exc()[:2000]
                if ctx is not None:
                    try:
                        report_payload = dict(ctx.chunk_report or {})
                        report_payload["ingest_pipeline"] = ctx.report()
                        doc.chunking_report = json.dumps(
                            report_payload, ensure_ascii=False
                        )
                    except Exception:
                        pass
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

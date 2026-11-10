"""Celery tasks for document processing and Google Drive sync."""
import logging
import traceback
from datetime import datetime, timezone

from app.celery_app import celery_app
from app.database import get_db_session
from app.models import Document, DriveConnection
from app.services.document_ingestion import ingest_document as _ingest_document

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.sync_drive_folder_task",
    max_retries=2,
    default_retry_delay=120,
    autoretry_for=(IOError, TimeoutError, ConnectionError, OSError),
    retry_backoff=True,
    retry_backoff_max=900,
    retry_jitter=True,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=60 * 30,
    time_limit=60 * 35,
)
def sync_drive_folder_task(self, connection_id: str, user_id: str) -> dict:
    """
    Synchronize one Drive folder connection in a Celery worker.

    Opens its own DB session, loads the user refresh token securely from the
    database, lists/downloads PDFs, and reuses ``ingest_document``.
    """
    from app.services.drive_sync import run_drive_sync_job

    logger.info(
        "Drive sync task started connection=%s user=%s attempt=%s",
        connection_id,
        user_id,
        self.request.retries + 1,
    )
    try:
        result = run_drive_sync_job(connection_id, user_id)
    except Exception as exc:
        logger.exception(
            "Drive sync task failed connection=%s: %s", connection_id, exc
        )
        with get_db_session() as db:
            conn = (
                db.query(DriveConnection)
                .filter(DriveConnection.id == connection_id)
                .first()
            )
            if conn and self.request.retries >= (self.max_retries or 2) - 1:
                conn.sync_status = "failed"
                conn.last_error = str(exc)[:500]
                db.commit()
        raise

    logger.info(
        "Drive sync task finished connection=%s stats=%s",
        connection_id,
        {k: result.get(k) for k in ("discovered", "ingested", "updated", "failed")},
    )
    return result


@celery_app.task(
    bind=True,
    name="app.tasks.sync_dropbox_connection_task",
    max_retries=2,
    default_retry_delay=120,
    autoretry_for=(IOError, TimeoutError, ConnectionError, OSError),
    retry_backoff=True,
    retry_backoff_max=900,
    retry_jitter=True,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=60 * 30,
    time_limit=60 * 35,
)
def sync_dropbox_connection_task(self, connection_id: str, user_id: str) -> dict:
    """Synchronize one Dropbox connection in a Celery worker."""
    from app.models import DropboxConnection
    from app.services.dropbox_sync import run_dropbox_sync_job

    logger.info(
        "Dropbox sync task started connection=%s user=%s attempt=%s",
        connection_id,
        user_id,
        self.request.retries + 1,
    )
    try:
        result = run_dropbox_sync_job(connection_id, user_id)
    except Exception as exc:
        logger.exception(
            "Dropbox sync task failed connection=%s: %s", connection_id, exc
        )
        with get_db_session() as db:
            conn = (
                db.query(DropboxConnection)
                .filter(DropboxConnection.id == connection_id)
                .first()
            )
            if conn and self.request.retries >= (self.max_retries or 2) - 1:
                conn.sync_status = "failed"
                conn.last_error = str(exc)[:500]
                db.commit()
        raise

    logger.info(
        "Dropbox sync task finished connection=%s stats=%s",
        connection_id,
        {
            key: result.get(key)
            for key in ("discovered", "ingested", "updated", "failed")
        },
    )
    return result


@celery_app.task(
    bind=True,
    name="app.tasks.import_dropbox_files_task",
    max_retries=2,
    default_retry_delay=60,
    autoretry_for=(IOError, TimeoutError, ConnectionError, OSError),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    acks_late=True,
)
def import_dropbox_files_task(
    self, connection_id: str, user_id: str, file_ids: list[str]
) -> dict:
    """Re-import Dropbox files the user explicitly selected after deletion."""
    from app.services.dropbox_sync import run_dropbox_import_job

    return run_dropbox_import_job(connection_id, user_id, list(file_ids or []))


@celery_app.task(
    bind=True,
    name="app.tasks.process_document",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(IOError, TimeoutError, ConnectionError, OSError),
    retry_backoff=True, 
    retry_backoff_max=600,
    retry_jitter=True,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_document(
    self,
    document_id: str,
    filepath: str,
    original_name: str,
    user_id: str,
) -> dict[str, str]:
    """Run the RAG ingestion pipeline for a stored document.

    This task is a thin dispatch wrapper around
    ``app.services.document_ingestion.ingest_document``, which is the single
    source of truth for the ingestion state machine (status, progress,
    chunk_count, page_count, summary, URL extraction, knowledge graph, and
    vector storage). The task itself only records retry bookkeeping before
    delegating; it does not open a second DB session that writes the same
    Document row, since ingest_document manages its own SessionLocal()
    session end-to-end and commits/rolls back independently.
    """
    with get_db_session() as db:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.processing_started_at = datetime.now(timezone.utc)
            doc.retry_count = (doc.retry_count or 0) + 1
            db.commit()

    logger.info("Dispatching ingestion pipeline for document: %s", original_name)

    try:
        _ingest_document(
            document_id=document_id,
            filepath=filepath,
            original_name=original_name,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error(
            "Document %s processing failed (attempt %s): %s",
            document_id,
            self.request.retries + 1,
            exc,
        )
        with get_db_session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc and self.request.retries >= (self.max_retries or 3) - 1:
                doc.status = "failed"
                doc.last_error_traceback = traceback.format_exc()[:2000]
                doc.processing_progress = 0
                db.commit()
        raise

    with get_db_session() as db:
        doc = db.query(Document).filter(Document.id == document_id).first()
        final_status = doc.status if doc else "unknown"

    if final_status == "failed":
        raise RuntimeError(
            f"Ingestion pipeline marked document {document_id} as failed"
        )

    return {"document_id": document_id, "status": final_status}
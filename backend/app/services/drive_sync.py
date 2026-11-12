"""Google Drive PDF discovery and ingestion service.

Bridges ``User.google_refresh_token`` (OAuth) with ``DriveConnection`` folder
rows, downloads PDFs to a working path (temp → UPLOAD_DIR / S3, same pattern as
manual uploads — PDF extractors require a filesystem path), and reuses
``ingest_document`` for the RAG pipeline.
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Document, DriveConnection, User
from app.services.document_ingestion import ingest_document
from app.services.drive_public import (
    DriveAccessError,
    DriveInvalidFolder,
    DriveNetworkError,
    DriveNotFound,
    DrivePermissionDenied,
    download_public_file,
    fetch_public_folder_metadata,
    list_public_pdfs_recursive,
    probe_public_folder,
)

logger = logging.getLogger(__name__)
settings = get_settings()
_sync_lock = threading.Lock()

PDF_MIME_TYPE = "application/pdf"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Statuses that mean a Celery (or local fallback) job is already in flight.
ACTIVE_SYNC_STATUSES = frozenset({"queued", "processing", "syncing"})

# Folder link patterns accepted from the Settings UI.
_FOLDER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{10,}$")
_FOLDER_PATH_RE = re.compile(
    r"/folders/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)


def extract_folder_id(url_or_id: str) -> str:
    """Extract a Drive folder ID from a URL or raw ID string."""
    raw = (url_or_id or "").strip()
    if not raw:
        raise ValueError("Folder link is empty")

    if _FOLDER_ID_RE.fullmatch(raw):
        return raw

    parsed = urlparse(raw)
    path_match = _FOLDER_PATH_RE.search(parsed.path or "")
    if path_match:
        return path_match.group(1)

    query = parse_qs(parsed.query or "")
    for key in ("id", "folderid", "folderId"):
        values = query.get(key) or []
        if values and _FOLDER_ID_RE.fullmatch(values[0]):
            return values[0]

    raise ValueError(
        "Could not parse a Google Drive folder ID. "
        "Paste a folder URL like https://drive.google.com/drive/folders/…"
    )


def _require_oauth_config() -> None:
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        raise RuntimeError("Google Drive OAuth is not configured")


def build_user_drive_credentials(user: User):
    """Build refreshable OAuth credentials from the user's stored token."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not user.google_refresh_token:
        raise RuntimeError("Google Drive is not connected for this user")

    _require_oauth_config()
    credentials = Credentials(
        token=None,
        refresh_token=user.google_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        scopes=DRIVE_SCOPES,
    )
    # Force a refresh so expired/revoked tokens fail early with a clear error.
    credentials.refresh(Request())
    return credentials


def _build_drive_service_for_user(user: User):
    from googleapiclient.discovery import build

    credentials = build_user_drive_credentials(user)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _build_drive_service(connection: DriveConnection, user: Optional[User] = None):
    """Prefer user OAuth; fall back to legacy connection credentials only."""
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    owner = user
    if owner is None and connection.user_id:
        # Caller should pass user when available; lazy load via relationship if set.
        owner = getattr(connection, "user", None)

    if owner is not None and owner.google_refresh_token:
        return _build_drive_service_for_user(owner)

    if connection.credentials_json:
        import json

        credentials = Credentials.from_authorized_user_info(
            json.loads(connection.credentials_json),
            scopes=DRIVE_SCOPES,
        )
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    service_account_file = (
        connection.service_account_file or settings.GOOGLE_SERVICE_ACCOUNT_FILE
    )
    if service_account_file:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=DRIVE_SCOPES,
        )
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    raise RuntimeError(
        "Google Drive is not connected. Connect Drive in Settings, then try again."
    )


def fetch_folder_metadata(service, folder_id: str) -> dict:
    """Validate folder access and return name/id."""
    meta = (
        service.files()
        .get(
            fileId=folder_id,
            fields="id, name, mimeType, trashed",
            supportsAllDrives=True,
        )
        .execute()
    )
    if meta.get("trashed"):
        raise DriveInvalidFolder("That Drive folder is in the trash")
    if meta.get("mimeType") != FOLDER_MIME_TYPE:
        raise DriveInvalidFolder("The link does not point to a Google Drive folder")
    return meta


def resolve_folder_access(
    folder_id: str,
    user: Optional[User] = None,
) -> tuple[str, dict]:
    """
    Lightweight access check for the API: public metadata first, then user OAuth.

    Does **not** list or download PDFs — that work belongs in Celery workers.
    Returns ``(access_mode, metadata)`` where access_mode is ``public`` or ``oauth``.

    Raises:
        DrivePermissionDenied: private folder and user has no OAuth
        DriveNotFound / DriveInvalidFolder / DriveNetworkError
    """
    # 1) Public "Anyone with the link" path — no user OAuth
    try:
        meta = fetch_public_folder_metadata(folder_id)
        return "public", meta
    except DriveNetworkError:
        raise
    except DrivePermissionDenied:
        pass  # fall through to OAuth
    except DriveNotFound:
        # Anonymous callers sometimes see "not found" for private folders.
        pass
    except DriveAccessError:
        pass

    # 2) Authenticated user OAuth
    if user is not None and user.google_refresh_token:
        try:
            service = _build_drive_service_for_user(user)
            meta = fetch_folder_metadata(service, folder_id)
            return "oauth", meta
        except DriveInvalidFolder:
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "404" in message or "not found" in message:
                raise DriveNotFound(
                    "Google Drive folder not found (or you do not have access)"
                ) from exc
            if "403" in message or "permission" in message:
                raise DrivePermissionDenied(
                    "You do not have permission to access this Drive folder"
                ) from exc
            if "invalid_grant" in message or "revoked" in message:
                raise DrivePermissionDenied(
                    "Google Drive access was revoked. Reconnect Drive in Settings."
                ) from exc
            raise DriveNetworkError(f"Drive API error: {exc}") from exc

    raise DrivePermissionDenied(
        "This Drive folder is private. Connect Google Drive in Settings, then retry."
    )


def _list_children(service, parent_id: str) -> list[dict]:
    files: list[dict] = []
    page_token = None
    query = f"'{parent_id}' in parents and trashed = false"
    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                pageToken=page_token,
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return files


def list_pdf_files_recursive(service, folder_id: str) -> list[dict]:
    """Return PDFs under a folder, including nested subfolders (BFS)."""
    pdfs: list[dict] = []
    queue = [folder_id]
    seen_folders: set[str] = set()

    while queue:
        current = queue.pop(0)
        if current in seen_folders:
            continue
        seen_folders.add(current)
        try:
            children = _list_children(service, current)
        except Exception:
            logger.exception("Failed listing Drive folder %s", current)
            raise

        for item in children:
            mime = item.get("mimeType")
            if mime == FOLDER_MIME_TYPE:
                queue.append(item["id"])
            elif mime == PDF_MIME_TYPE:
                item = dict(item)
                item["parent_folder_id"] = current
                pdfs.append(item)
    return pdfs


def _list_pdf_files(service, folder_id: str) -> list[dict]:
    """Compatibility wrapper used by tests / shallow listings."""
    return list_pdf_files_recursive(service, folder_id)


def _download_drive_file(service, file_id: str, destination: str) -> None:
    """Download a Drive file to a local path (temp or UPLOAD_DIR working copy)."""
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with io.FileIO(destination, "wb") as file_handle:
        downloader = MediaIoBaseDownload(file_handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _mirror_to_s3(user_id: str, stored_filename: str, filepath: str) -> None:
    try:
        from app.services import s3_storage

        if s3_storage.is_configured():
            s3_storage.upload_pdf(str(user_id), stored_filename, filepath)
    except Exception as exc:
        logger.warning("Drive PDF S3 mirror failed for %s: %s", stored_filename, exc)


def _soft_delete_document(db: Session, document: Document) -> None:
    if document.is_deleted:
        return
    document.is_deleted = True
    document.deleted_at = datetime.now(timezone.utc)
    db.commit()
    try:
        from app.rag.vectordb.vectorstore import delete_document_chunks

        delete_document_chunks(str(document.id), str(document.user_id))
    except Exception as exc:
        logger.warning(
            "Vector cleanup failed for Drive-removed document %s: %s",
            document.id,
            exc,
        )


def _reingest_document(
    db: Session,
    document: Document,
    filepath: str,
) -> None:
    """Wipe vectors and re-run the shared ingestion pipeline."""
    try:
        from app.rag.vectordb.vectorstore import delete_document_chunks

        delete_document_chunks(str(document.id), str(document.user_id))
    except Exception as exc:
        logger.warning("Could not clear chunks before re-ingest %s: %s", document.id, exc)

    document.status = "pending"
    document.error_message = None
    document.processing_progress = 0
    document.processing_stage = "queued"
    db.commit()

    ingest_document(
        document_id=document.id,
        filepath=filepath,
        original_name=document.original_name,
        user_id=document.user_id,
    )


def _prepare_working_copy(user_id: str, source_tmp: str) -> tuple[str, str]:
    """
    Persist a working PDF under UPLOAD_DIR (viewer + ingest need a path).

    Download happens into a NamedTemporaryFile first; this copies into the
    same layout used by manual uploads. S3 remains the system of record when
    configured.
    """
    user_dir = os.path.join(settings.UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    stored_filename = f"{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(user_dir, stored_filename)
    shutil.copyfile(source_tmp, filepath)
    return stored_filename, filepath


def sync_connection(
    db: Session,
    connection: DriveConnection,
    user: User,
    *,
    recursive: bool = True,
) -> dict[str, Any]:
    """Synchronize one Drive folder connection for a user."""
    stats = {
        "connection_id": str(connection.id),
        "folder_id": connection.folder_id,
        "discovered": 0,
        "ingested": 0,
        "updated": 0,
        "skipped": 0,
        "deleted": 0,
        "failed": 0,
    }

    connection.sync_status = "processing"
    connection.last_error = None
    connection.updated_at = datetime.now(timezone.utc)
    db.commit()

    try:
        access_mode = (connection.access_mode or "oauth").lower()
        service = None

        if access_mode == "public":
            try:
                meta, files = probe_public_folder(connection.folder_id)
                connection.folder_name = meta.get("name") or connection.folder_name
                db.commit()
            except DrivePermissionDenied:
                # Folder may have been made private — upgrade to OAuth if possible
                if user.google_refresh_token:
                    access_mode = "oauth"
                    connection.access_mode = "oauth"
                    db.commit()
                else:
                    raise
            except DriveAccessError:
                raise

        if access_mode != "public":
            service = _build_drive_service(connection, user=user)
            try:
                meta = fetch_folder_metadata(service, connection.folder_id)
                connection.folder_name = meta.get("name") or connection.folder_name
                db.commit()
            except Exception as exc:
                logger.warning("Could not refresh folder metadata: %s", exc)

            files = (
                list_pdf_files_recursive(service, connection.folder_id)
                if recursive
                else [
                    f
                    for f in _list_children(service, connection.folder_id)
                    if f.get("mimeType") == PDF_MIME_TYPE
                ]
            )

        stats["discovered"] = len(files)
        stats["access_mode"] = access_mode
        remote_ids = {f["id"] for f in files}

        def _fetch_file(file_id: str, dest: str) -> None:
            if access_mode == "public":
                download_public_file(file_id, dest)
            else:
                _download_drive_file(service, file_id, dest)

        for drive_file in files:
            file_id = drive_file["id"]
            modified = drive_file.get("modifiedTime") or ""
            existing = (
                db.query(Document)
                .filter(
                    Document.user_id == user.id,
                    Document.drive_file_id == file_id,
                )
                .first()
            )

            try:
                if existing and not existing.is_deleted:
                    # Public listings often omit modifiedTime — skip re-ingest when
                    # already ready and times are both empty/equal.
                    if (
                        existing.status == "ready"
                        and (existing.drive_modified_time or "") == (modified or "")
                        and existing.drive_modified_time is not None
                    ):
                        stats["skipped"] += 1
                        continue
                    if (
                        existing.status == "ready"
                        and access_mode == "public"
                        and not modified
                        and existing.original_name == (drive_file.get("name") or existing.original_name)
                    ):
                        stats["skipped"] += 1
                        continue
                    if existing.drive_modified_time and existing.drive_modified_time == modified:
                        stats["skipped"] += 1
                        continue
                    # Modified (or first time we store modifiedTime) → re-download
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        _fetch_file(file_id, tmp_path)
                        # Replace working copy filename in place when possible
                        user_dir = os.path.join(settings.UPLOAD_DIR, str(user.id))
                        os.makedirs(user_dir, exist_ok=True)
                        filepath = os.path.join(user_dir, existing.filename)
                        shutil.copyfile(tmp_path, filepath)
                        existing.file_size = Path(filepath).stat().st_size
                        existing.original_name = drive_file.get("name") or existing.original_name
                        existing.drive_folder_id = connection.folder_id
                        existing.drive_modified_time = modified or existing.drive_modified_time
                        existing.drive_synced_at = datetime.now(timezone.utc)
                        db.commit()
                        _mirror_to_s3(str(user.id), existing.filename, filepath)
                        _reingest_document(db, existing, filepath)
                        stats["updated"] += 1
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                    continue

                if existing and existing.is_deleted:
                    # Restore + re-ingest if the file reappeared on Drive
                    existing.is_deleted = False
                    existing.deleted_at = None
                    db.commit()

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    _fetch_file(file_id, tmp_path)
                    stored_filename, filepath = _prepare_working_copy(str(user.id), tmp_path)
                    file_size = Path(filepath).stat().st_size

                    if existing:
                        # Soft-deleted row reused
                        if existing.filename and existing.filename != stored_filename:
                            old_path = os.path.join(
                                settings.UPLOAD_DIR, str(user.id), existing.filename
                            )
                            Path(old_path).unlink(missing_ok=True)
                        existing.filename = stored_filename
                        existing.original_name = drive_file.get("name") or stored_filename
                        existing.file_size = file_size
                        existing.status = "pending"
                        existing.drive_folder_id = connection.folder_id
                        existing.drive_modified_time = modified or None
                        existing.drive_synced_at = datetime.now(timezone.utc)
                        existing.error_message = None
                        db.commit()
                        document = existing
                    else:
                        document = Document(
                            user_id=user.id,
                            filename=stored_filename,
                            original_name=drive_file.get("name") or stored_filename,
                            file_size=file_size,
                            status="pending",
                            drive_file_id=file_id,
                            drive_folder_id=connection.folder_id,
                            drive_modified_time=modified or None,
                            drive_synced_at=datetime.now(timezone.utc),
                        )
                        db.add(document)
                        db.commit()
                        db.refresh(document)

                    _mirror_to_s3(str(user.id), document.filename, filepath)
                    ingest_document(
                        document_id=document.id,
                        filepath=filepath,
                        original_name=document.original_name,
                        user_id=user.id,
                    )
                    stats["ingested"] += 1
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            except Exception as exc:
                logger.exception("Drive ingestion failed for file %s", file_id)
                stats["failed"] += 1
                if existing:
                    db.refresh(existing)
                    existing.status = "failed"
                    existing.error_message = str(exc)[:500]
                    db.commit()

        # Soft-delete local Drive docs no longer present remotely (this folder)
        local_docs = (
            db.query(Document)
            .filter(
                Document.user_id == user.id,
                Document.drive_folder_id == connection.folder_id,
                Document.drive_file_id.isnot(None),
                Document.is_deleted.is_(False),
            )
            .all()
        )
        for doc in local_docs:
            if doc.drive_file_id not in remote_ids:
                _soft_delete_document(db, doc)
                stats["deleted"] += 1

        connection.sync_status = "completed" if stats["failed"] == 0 else "failed"
        if stats["failed"]:
            connection.last_error = f"{stats['failed']} file(s) failed during sync"
        connection.last_synced_at = datetime.now(timezone.utc)
        connection.updated_at = datetime.now(timezone.utc)
        db.commit()
        return stats

    except Exception as exc:
        logger.exception("Drive sync failed for connection %s", connection.id)
        connection.sync_status = "failed"
        connection.last_error = str(exc)[:500]
        connection.updated_at = datetime.now(timezone.utc)
        db.commit()
        stats["failed"] += 1
        stats["error"] = str(exc)[:500]
        return stats


def run_drive_sync_job(connection_id: str, user_id: str) -> dict[str, Any]:
    """
    Worker entrypoint: open a fresh DB session and sync one Drive connection.

    Called from Celery (and from FastAPI BackgroundTasks when Celery is eager).
    Never reuse an API request session here.
    """
    from app.database import get_db_session

    with get_db_session() as db:
        connection = (
            db.query(DriveConnection)
            .filter(
                DriveConnection.id == connection_id,
                DriveConnection.user_id == user_id,
            )
            .first()
        )
        if not connection:
            logger.warning(
                "Drive sync job skipped — connection %s not found for user %s",
                connection_id,
                user_id,
            )
            return {"connection_id": connection_id, "failed": 1, "error": "not_found"}

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            connection.sync_status = "failed"
            connection.last_error = "User not found"
            db.commit()
            return {"connection_id": connection_id, "failed": 1, "error": "user_not_found"}

        mode = (connection.access_mode or "public").lower()
        if mode == "oauth" and not user.google_refresh_token:
            connection.sync_status = "failed"
            connection.last_error = "Connect Google Drive to sync this private folder"
            db.commit()
            return {
                "connection_id": connection_id,
                "failed": 1,
                "error": connection.last_error,
            }

        return sync_connection(db, connection, user)


def enqueue_connection_sync(
    db: Session,
    connection: DriveConnection,
    user_id: str,
    *,
    background_tasks: Any = None,
    force: bool = False,
) -> tuple[str, bool]:
    """
    Mark a connection queued and dispatch Celery (or local BackgroundTasks).

    Returns ``(task_id, already_running)``.
    """
    status = (connection.sync_status or "").lower()
    if not force and status in ACTIVE_SYNC_STATUSES:
        return (connection.celery_task_id or ""), True

    connection.sync_status = "queued"
    connection.last_error = None
    connection.updated_at = datetime.now(timezone.utc)
    db.commit()

    connection_id = str(connection.id)
    uid = str(user_id)

    from app.celery_app import celery_app
    from app.tasks import sync_drive_folder_task

    use_local_bg = (
        bool(getattr(celery_app.conf, "task_always_eager", False))
        and background_tasks is not None
    )

    if use_local_bg:
        task_id = f"local_{uuid.uuid4().hex}"
        connection.celery_task_id = task_id
        db.commit()
        background_tasks.add_task(run_drive_sync_job, connection_id, uid)
        return task_id, False

    try:
        async_result = sync_drive_folder_task.delay(connection_id, uid)
        task_id = async_result.id
        connection.celery_task_id = task_id
        db.commit()
        return task_id, False
    except Exception as exc:
        logger.warning("Celery Drive enqueue failed, falling back: %s", exc)
        if background_tasks is not None:
            task_id = f"local_{uuid.uuid4().hex}"
            connection.celery_task_id = task_id
            db.commit()
            background_tasks.add_task(run_drive_sync_job, connection_id, uid)
            return task_id, False
        connection.sync_status = "failed"
        connection.last_error = f"Could not enqueue Drive sync: {exc}"[:500]
        db.commit()
        raise


def sync_user_folders(
    db: Session,
    user: User,
    connection_id: Optional[str] = None,
) -> dict[str, Any]:
    """Synchronously sync one or all folders (used by workers / tests)."""
    query = db.query(DriveConnection).filter(
        DriveConnection.user_id == user.id,
        DriveConnection.enabled.is_(True),
    )
    if connection_id:
        query = query.filter(DriveConnection.id == connection_id)
    connections = query.all()

    summary = {
        "connections": len(connections),
        "discovered": 0,
        "ingested": 0,
        "updated": 0,
        "skipped": 0,
        "deleted": 0,
        "failed": 0,
        "results": [],
    }

    needs_oauth = any(
        (c.access_mode or "").lower() == "oauth" for c in connections
    )
    if needs_oauth and not user.google_refresh_token:
        oauth_only = all((c.access_mode or "").lower() == "oauth" for c in connections)
        if oauth_only:
            summary["error"] = "Google Drive is not connected"
            return summary

    for connection in connections:
        mode = (connection.access_mode or "public").lower()
        if mode == "oauth" and not user.google_refresh_token:
            connection.sync_status = "failed"
            connection.last_error = "Connect Google Drive to sync this private folder"
            db.commit()
            summary["failed"] += 1
            summary["results"].append(
                {
                    "connection_id": str(connection.id),
                    "failed": 1,
                    "error": connection.last_error,
                }
            )
            continue
        result = sync_connection(db, connection, user)
        summary["results"].append(result)
        for key in ("discovered", "ingested", "updated", "skipped", "deleted", "failed"):
            summary[key] += int(result.get(key) or 0)
    return summary


def sync_drive_pdfs(db: Session) -> dict:
    """
    Scheduler helper: enqueue a Celery job per enabled Drive connection.

    Does not list/download PDFs in-process — workers own that work.
    """
    if not _sync_lock.acquire(blocking=False):
        logger.info("Drive sync enqueue skipped — previous scheduler tick still active")
        return {
            "connections": 0,
            "enqueued": 0,
            "skipped": 0,
            "failed": 0,
        }

    stats = {
        "connections": 0,
        "enqueued": 0,
        "skipped": 0,
        "failed": 0,
    }
    try:
        connections = (
            db.query(DriveConnection)
            .filter(DriveConnection.enabled.is_(True))
            .all()
        )
        stats["connections"] = len(connections)
        for connection in connections:
            user = db.query(User).filter(User.id == connection.user_id).first()
            if not user:
                stats["failed"] += 1
                continue
            mode = (connection.access_mode or "public").lower()
            if mode == "oauth" and not user.google_refresh_token:
                connection.sync_status = "failed"
                connection.last_error = "Owner has not connected Google Drive"
                db.commit()
                stats["failed"] += 1
                continue
            try:
                _task_id, already = enqueue_connection_sync(
                    db, connection, str(user.id), background_tasks=None
                )
                if already:
                    stats["skipped"] += 1
                else:
                    stats["enqueued"] += 1
            except Exception:
                logger.exception(
                    "Failed to enqueue scheduled Drive sync for %s", connection.id
                )
                stats["failed"] += 1
        logger.info("Drive sync enqueue complete: %s", stats)
        return stats
    finally:
        _sync_lock.release()


def sync_drive_pdfs_with_session() -> dict:
    """Scheduler entry: open a session and enqueue Drive sync jobs."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        return sync_drive_pdfs(db)
    finally:
        db.close()

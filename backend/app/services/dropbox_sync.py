"""Dropbox PDF discovery, lifecycle tracking and ingestion.

Bridges ``User.dropbox_refresh_token`` (OAuth) and public shared links with
``DropboxConnection`` rows, downloads PDFs to the same working directory used by
manual uploads, and reuses ``ingest_document`` for extraction, chunking,
embeddings, ChromaDB, BM25 and the knowledge graph.

Lifecycle rules
---------------
* New PDFs are ingested through the shared pipeline.
* Unchanged PDFs (same Dropbox ``rev``) are skipped.
* Modified PDFs are re-downloaded, their vectors are cleared, and re-ingested.
* A user deletion in Genova sets ``lifecycle_status=USER_DELETED`` and writes a
  ``DropboxDocumentTombstone`` so autosync can never restore the file. Only an
  explicit manual re-import clears the tombstone.
* A file missing from the source is marked ``SOURCE_DELETED`` (vectors removed)
  and may be restored automatically if it reappears.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    DOCUMENT_SOURCE_DROPBOX,
    DOCUMENT_SOURCE_DROPBOX_LINK,
    DROPBOX_SOURCE_TYPES,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SOURCE_DELETED,
    LIFECYCLE_USER_DELETED,
    Document,
    DropboxConnection,
    DropboxDocumentTombstone,
    User,
)
from app.services import dropbox_client as dbx
from app.services.document_ingestion import ingest_document
from app.services.dropbox_client import (
    DropboxAuthError,
    DropboxClient,
    DropboxError,
    DropboxInvalidLink,
)

logger = logging.getLogger(__name__)
settings = get_settings()
_sync_lock = threading.Lock()

PDF_EXTENSION = ".pdf"
ACTIVE_SYNC_STATUSES = frozenset({"queued", "processing", "syncing"})
MAX_FOLDERS = 500
MAX_DEPTH = 25
MAX_FILES = 2000


# ── Small helpers ────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _join_path(base: str, name: str) -> str:
    base = (base or "").rstrip("/")
    name = (name or "").strip("/")
    if not base:
        return f"/{name}" if name else ""
    return f"{base}/{name}" if name else base


def _is_pdf_name(name: str) -> bool:
    return bool(name) and name.lower().endswith(PDF_EXTENSION)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _pending_ids(connection: DropboxConnection) -> set[str]:
    raw = connection.pending_file_ids
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return set()
    return {str(item) for item in parsed} if isinstance(parsed, list) else set()


def _set_pending_ids(connection: DropboxConnection, file_ids: list[str]) -> None:
    connection.pending_file_ids = json.dumps(sorted(set(file_ids))) if file_ids else None


def _source_type(connection: DropboxConnection) -> str:
    return (
        DOCUMENT_SOURCE_DROPBOX_LINK
        if (connection.connection_kind or "oauth") == "link"
        else DOCUMENT_SOURCE_DROPBOX
    )


def _link_display_name(url: str) -> str:
    parsed = urlparse(url)
    segments = [part for part in parsed.path.split("/") if part]
    if "fi" in segments:
        return unquote(segments[-1]) or "Dropbox file"
    return "Dropbox shared folder"


# ── OAuth tokens ─────────────────────────────────────────────────────────────


def get_user_access_token(db: Session, user: User) -> str:
    """Return a valid Dropbox access token, refreshing it when needed."""
    if not user.dropbox_refresh_token:
        raise DropboxAuthError("Dropbox is not connected for this user")

    expires_at = _as_utc(user.dropbox_token_expires_at)
    if user.dropbox_access_token and expires_at and expires_at > _now() + timedelta(
        minutes=2
    ):
        return user.dropbox_access_token

    data = dbx.refresh_access_token(user.dropbox_refresh_token)
    access_token = data.get("access_token")
    if not access_token:
        raise DropboxAuthError("Dropbox did not return an access token")

    user.dropbox_access_token = access_token
    user.dropbox_token_expires_at = _now() + timedelta(
        seconds=int(data.get("expires_in") or 14400)
    )
    db.commit()
    return access_token


def _link_access_token(db: Session, user: User) -> Optional[str]:
    """Token for shared-link listing: user OAuth first, app auth second."""
    if user.dropbox_refresh_token:
        try:
            return get_user_access_token(db, user)
        except DropboxError as exc:
            logger.warning("Dropbox user token unavailable for link listing: %s", exc)

    if dbx.is_configured():
        try:
            return dbx.app_access_token()
        except DropboxError as exc:
            logger.warning("Dropbox app token unavailable for link listing: %s", exc)
    return None


# ── Discovery ────────────────────────────────────────────────────────────────


def _entry_path(entry: dict, parent: str, *, shared_link: bool) -> str:
    if shared_link:
        return _join_path(parent, entry.get("name") or "")
    return (
        entry.get("path_lower")
        or entry.get("path_display")
        or _join_path(parent, entry.get("name") or "")
    )


def _entry_revision(entry: dict) -> str:
    return str(
        entry.get("rev")
        or entry.get("content_hash")
        or entry.get("server_modified")
        or ""
    )


def _list_api_pdfs(
    client: DropboxClient,
    *,
    path: str = "",
    shared_link: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Breadth-first PDF discovery (shared links only list non-recursively)."""
    pdfs: list[dict[str, Any]] = []
    queue: list[tuple[str, int]] = [(path or "", 0)]
    seen: set[str] = set()

    while queue and len(seen) < MAX_FOLDERS and len(pdfs) < MAX_FILES:
        current, depth = queue.pop(0)
        if current in seen or depth > MAX_DEPTH:
            continue
        seen.add(current)
        entries = client.list_folder(
            current,
            shared_link=shared_link,
            recursive=False,
        )
        for entry in entries:
            tag = entry.get(".tag")
            name = entry.get("name") or ""
            if tag == "folder":
                queue.append((_join_path(current, name), depth + 1))
            elif tag == "file" and _is_pdf_name(name):
                file_path = _entry_path(entry, current, shared_link=bool(shared_link))
                pdfs.append(
                    {
                        "id": entry.get("id") or f"path:{_short_hash(file_path)}",
                        "name": name,
                        "path": file_path,
                        "rev": _entry_revision(entry),
                        "size": entry.get("size"),
                        "modified": entry.get("server_modified") or "",
                    }
                )
    return pdfs


def list_folder_pdfs(
    client: DropboxClient, folder_path: str
) -> list[dict[str, Any]]:
    return _list_api_pdfs(client, path=folder_path or "")


def list_shared_link_pdfs(
    client: DropboxClient, shared_link: str, path: str = ""
) -> list[dict[str, Any]]:
    return _list_api_pdfs(client, path=path or "", shared_link=shared_link)


def _looks_like_zip(name: str, data: bytes, content_type: str) -> bool:
    return (
        data[:2] == b"PK"
        or name.lower().endswith(".zip")
        or "zip" in (content_type or "").lower()
    )


def _pdfs_from_archive(
    *,
    filename: str,
    data: bytes,
    content_type: str,
    connection: DropboxConnection,
) -> tuple[list[dict[str, Any]], io.BytesIO]:
    """Build file entries from a public shared-link download (ZIP or PDF)."""
    files: list[dict[str, Any]] = []
    buffer = io.BytesIO(data)

    if not _looks_like_zip(filename, data, content_type):
        if _is_pdf_name(filename) or data[:4] == b"%PDF":
            display = filename or "Dropbox file.pdf"
            files.append(
                {
                    "id": f"link:{connection.remote_folder_id}:{_short_hash(filename)}",
                    "name": display,
                    "path": "",
                    "rev": f"sha256:{hashlib.sha256(data).hexdigest()[:32]}",
                    "size": len(data),
                    "modified": "",
                    "inline_data": data,
                }
            )
        return files, buffer

    try:
        with zipfile.ZipFile(buffer) as archive:
            for info in archive.infolist():
                if info.is_dir() or not _is_pdf_name(info.filename):
                    continue
                relative = "/" + info.filename.lstrip("/")
                files.append(
                    {
                        "id": (
                            f"link:{connection.remote_folder_id}:"
                            f"{_short_hash(relative)}"
                        ),
                        "name": Path(info.filename).name,
                        "path": relative,
                        "rev": f"crc:{info.CRC:08x}:{info.file_size}",
                        "size": info.file_size,
                        "modified": "",
                        "archive_name": info.filename,
                    }
                )
                if len(files) >= MAX_FILES:
                    break
    except zipfile.BadZipFile as exc:
        raise DropboxError("Dropbox returned a corrupt folder archive") from exc
    return files, buffer


def _bytes_from_public_archive(
    data: bytes,
    content_type: str,
    entry: dict[str, Any],
) -> bytes:
    """Read one PDF out of a public shared-link download (ZIP or raw PDF)."""
    if data[:4] == b"%PDF":
        return data

    target = (entry.get("path") or "").lstrip("/")
    name = entry.get("name") or ""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [
                member for member in archive.namelist() if not member.endswith("/")
            ]
            candidate = target if target and target in members else None
            if candidate is None:
                for member in members:
                    if Path(member).name == name or (
                        target and member.rstrip("/").endswith("/" + target)
                    ):
                        candidate = member
                        break
            if candidate is None:
                raise DropboxError(
                    f"Could not find {name or target} in the Dropbox archive"
                )
            return archive.read(candidate)
    except zipfile.BadZipFile as exc:
        raise DropboxError("Dropbox returned a corrupt folder archive") from exc


def resolve_connection_files(
    db: Session,
    connection: DropboxConnection,
    user: User,
) -> tuple[list[dict[str, Any]], Callable[[dict[str, Any]], bytes]]:
    """List PDFs for a connection and return a per-file byte fetcher."""
    if (connection.connection_kind or "oauth") == "link":
        shared_link = connection.shared_link
        if not shared_link:
            raise DropboxInvalidLink("Dropbox connection is missing its shared link")

        token = _link_access_token(db, user)
        if token:
            client = DropboxClient(token)
            try:
                files = list_shared_link_pdfs(
                    client, shared_link, connection.folder_path or ""
                )
            except DropboxError as exc:
                logger.warning(
                    "Dropbox shared-link API listing failed, falling back to "
                    "public archive: %s",
                    exc,
                )
            else:
                archive_cache: dict[str, Any] = {
                    "loaded": False,
                    "data": b"",
                    "content_type": "",
                }

                def fetch_api(entry: dict[str, Any]) -> bytes:
                    try:
                        return client.download_shared_link(
                            shared_link, path=entry.get("path") or None
                        )
                    except DropboxError as exc:
                        # Apps without the optional sharing.read scope can still
                        # read the public archive served by the shared link.
                        logger.info(
                            "Shared-link API download unavailable (%s); using "
                            "public archive fallback",
                            exc,
                        )
                        if not archive_cache["loaded"]:
                            _name, data, content_type = (
                                dbx.download_public_shared_link(shared_link)
                            )
                            archive_cache.update(
                                loaded=True, data=data, content_type=content_type
                            )
                        return _bytes_from_public_archive(
                            archive_cache["data"],
                            archive_cache["content_type"],
                            entry,
                        )

                return files, fetch_api

        filename, data, content_type = dbx.download_public_shared_link(shared_link)
        files, buffer = _pdfs_from_archive(
            filename=filename,
            data=data,
            content_type=content_type,
            connection=connection,
        )

        def fetch_public(entry: dict[str, Any]) -> bytes:
            inline = entry.get("inline_data")
            if inline is not None:
                return inline
            with zipfile.ZipFile(buffer) as archive:
                return archive.read(entry["archive_name"])

        return files, fetch_public

    client = DropboxClient(get_user_access_token(db, user))
    files = list_folder_pdfs(client, connection.folder_path or "")

    def fetch_oauth(entry: dict[str, Any]) -> bytes:
        return client.download(entry.get("path") or entry["id"])

    return files, fetch_oauth


def describe_folder(db: Session, user: User, folder_path: str) -> dict[str, Any]:
    """Validate an OAuth folder path and return connection metadata."""
    client = DropboxClient(get_user_access_token(db, user))
    meta = client.get_metadata(folder_path or "")
    if meta.get(".tag") not in (None, "folder"):
        raise DropboxInvalidLink("That Dropbox path is a file, not a folder")

    path = folder_path or ""
    return {
        "remote_folder_id": meta.get("id") or (path or "root"),
        "folder_name": meta.get("name") or "Dropbox",
        "folder_url": f"https://www.dropbox.com/home{path}" if path else "https://www.dropbox.com/home",
        "folder_path": path,
        "is_folder": True,
    }


def describe_shared_link(db: Session, user: User, url: str) -> dict[str, Any]:
    """Resolve a shared link's stable connection key and display name.

    The connection key is always derived from the canonical link so re-adding
    the same link is idempotent even when ``sharing.read`` is unavailable.
    """
    normalized = dbx.normalize_shared_link(url)
    key = dbx.shared_link_key(normalized)
    fallback = {
        "remote_folder_id": key,
        "folder_name": _link_display_name(normalized),
        "folder_url": normalized,
        "folder_path": "",
        "is_folder": True,
    }

    token = _link_access_token(db, user)
    if not token:
        return fallback
    try:
        meta = DropboxClient(token).shared_link_metadata(normalized)
    except DropboxError as exc:
        logger.info("Dropbox shared-link metadata unavailable: %s", exc)
        return fallback

    return {
        "remote_folder_id": key,
        "folder_name": meta.get("name") or fallback["folder_name"],
        "folder_url": normalized,
        "folder_path": "",
        "is_folder": meta.get(".tag") != "file",
    }


# ── Browsing ─────────────────────────────────────────────────────────────────


def browse_folder(
    db: Session,
    user: User,
    *,
    path: str = "",
    connection: Optional[DropboxConnection] = None,
) -> list[dict[str, Any]]:
    """List folders/PDFs for the Settings browser and annotate import state."""
    if connection is not None and (connection.connection_kind or "oauth") == "link":
        files, _fetch = resolve_connection_files(db, connection, user)
        rows = [
            {
                "id": entry["id"],
                "name": entry["name"],
                "path": entry.get("path") or "",
                "is_folder": False,
                "is_pdf": True,
                "size": entry.get("size"),
                "revision": entry.get("rev"),
                "modified_at": entry.get("modified") or None,
            }
            for entry in files
        ]
        return annotate_entries(db, user, rows)

    client = DropboxClient(get_user_access_token(db, user))
    target = path or (connection.folder_path if connection else "") or ""
    rows: list[dict[str, Any]] = []
    for entry in client.list_folder(target, recursive=False):
        tag = entry.get(".tag")
        if tag not in ("file", "folder"):
            continue
        name = entry.get("name") or ""
        is_pdf = tag == "file" and _is_pdf_name(name)
        if tag == "file" and not is_pdf:
            continue
        rows.append(
            {
                "id": entry.get("id"),
                "name": name,
                "path": entry.get("path_display") or entry.get("path_lower") or "",
                "is_folder": tag == "folder",
                "is_pdf": is_pdf,
                "size": entry.get("size"),
                "revision": _entry_revision(entry),
                "modified_at": entry.get("server_modified") or None,
            }
        )
    return annotate_entries(db, user, rows)


def annotate_entries(
    db: Session, user: User, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach existing document IDs / lifecycle state to listing rows."""
    ids = [row["id"] for row in rows if row.get("id")]
    if not ids:
        return rows

    documents = {
        doc.external_file_id: doc
        for doc in db.query(Document)
        .filter(
            Document.user_id == user.id,
            Document.external_file_id.in_(ids),
            Document.source_type.in_(DROPBOX_SOURCE_TYPES),
        )
        .all()
    }
    tombstones = {
        row.external_file_id
        for row in db.query(DropboxDocumentTombstone)
        .filter(
            DropboxDocumentTombstone.user_id == user.id,
            DropboxDocumentTombstone.external_file_id.in_(ids),
        )
        .all()
    }

    for row in rows:
        doc = documents.get(row.get("id"))
        row["document_id"] = str(doc.id) if doc else None
        if doc is not None:
            row["lifecycle_status"] = doc.lifecycle_status
        elif row.get("id") in tombstones:
            row["lifecycle_status"] = LIFECYCLE_USER_DELETED
        else:
            row["lifecycle_status"] = None
    return rows


# ── Ingestion helpers ────────────────────────────────────────────────────────


def _mirror_to_s3(user_id: str, stored_filename: str, filepath: str) -> None:
    try:
        from app.services import s3_storage

        if s3_storage.is_configured():
            s3_storage.upload_pdf(str(user_id), stored_filename, filepath)
    except Exception as exc:
        logger.warning("Dropbox PDF S3 mirror failed for %s: %s", stored_filename, exc)


def _prepare_working_copy(user_id: str, data: bytes) -> tuple[str, str]:
    user_dir = os.path.join(settings.UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    stored_filename = f"{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(user_dir, stored_filename)
    with open(filepath, "wb") as handle:
        handle.write(data)
    return stored_filename, filepath


def _clear_vectors_and_graph(document: Document) -> None:
    try:
        from app.rag.vectordb.vectorstore import delete_document_chunks

        delete_document_chunks(str(document.id), str(document.user_id))
    except Exception as exc:
        logger.warning(
            "Could not clear vectors for Dropbox document %s: %s", document.id, exc
        )
    try:
        from app.graph.service import remove_document as remove_document_graph

        remove_document_graph(str(document.id), str(document.user_id))
    except Exception as exc:
        logger.warning(
            "Could not clear graph for Dropbox document %s: %s", document.id, exc
        )


def _tombstone_exists(db: Session, user_id: Any, external_file_id: str) -> bool:
    return (
        db.query(DropboxDocumentTombstone.id)
        .filter(
            DropboxDocumentTombstone.user_id == user_id,
            DropboxDocumentTombstone.external_file_id == external_file_id,
        )
        .first()
        is not None
    )


def _remove_tombstone(db: Session, user_id: Any, external_file_id: str) -> None:
    (
        db.query(DropboxDocumentTombstone)
        .filter(
            DropboxDocumentTombstone.user_id == user_id,
            DropboxDocumentTombstone.external_file_id == external_file_id,
        )
        .delete(synchronize_session=False)
    )


def _apply_file(
    db: Session,
    document: Document,
    *,
    entry: dict[str, Any],
    data: bytes,
    connection: DropboxConnection,
) -> None:
    """Write bytes, reset state, clear vectors and re-run ingestion."""
    user_dir = os.path.join(settings.UPLOAD_DIR, str(document.user_id))
    os.makedirs(user_dir, exist_ok=True)

    if document.filename:
        filepath = os.path.join(user_dir, document.filename)
    else:
        document.filename, filepath = _prepare_working_copy(document.user_id, data)

    with open(filepath, "wb") as handle:
        handle.write(data)

    document.file_size = len(data)
    document.original_name = entry.get("name") or document.original_name
    document.source_type = _source_type(connection)
    document.external_file_id = entry["id"]
    document.external_folder_id = connection.remote_folder_id
    document.external_revision = entry.get("rev") or None
    document.external_synced_at = _now()
    document.lifecycle_status = LIFECYCLE_ACTIVE
    document.source_deleted_at = None
    document.is_deleted = False
    document.deleted_at = None
    document.status = "pending"
    document.error_message = None
    document.processing_progress = 0
    document.processing_stage = "queued"
    db.commit()

    _clear_vectors_and_graph(document)
    _mirror_to_s3(str(document.user_id), document.filename, filepath)
    # Queue per-document ingest so multiple Dropbox files can process in parallel
    # (solo workers still serialize; use --pool=threads --concurrency=N for overlap).
    _enqueue_document_ingest(
        document_id=str(document.id),
        filepath=filepath,
        original_name=document.original_name,
        user_id=str(document.user_id),
    )


def _enqueue_document_ingest(
    *,
    document_id: str,
    filepath: str,
    original_name: str,
    user_id: str,
) -> None:
    """Dispatch ``process_document`` so Dropbox imports don't block each other."""
    from app.tasks import process_document

    try:
        process_document.delay(
            document_id=document_id,
            filepath=filepath,
            original_name=original_name,
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning(
            "Celery queue failed for Dropbox document %s (%s); ingesting inline",
            document_id,
            exc,
        )
        ingest_document(
            document_id=document_id,
            filepath=filepath,
            original_name=original_name,
            user_id=user_id,
        )


def _create_file(
    db: Session,
    user: User,
    connection: DropboxConnection,
    entry: dict[str, Any],
    data: bytes,
) -> Document:
    stored_filename, filepath = _prepare_working_copy(user.id, data)
    document = Document(
        user_id=user.id,
        filename=stored_filename,
        original_name=entry.get("name") or stored_filename,
        file_size=len(data),
        status="pending",
        source_type=_source_type(connection),
        external_file_id=entry["id"],
        external_folder_id=connection.remote_folder_id,
        external_revision=entry.get("rev") or None,
        external_synced_at=_now(),
        lifecycle_status=LIFECYCLE_ACTIVE,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    _mirror_to_s3(str(user.id), document.filename, filepath)
    _enqueue_document_ingest(
        document_id=str(document.id),
        filepath=filepath,
        original_name=document.original_name,
        user_id=str(user.id),
    )
    return document


def _mark_source_deleted(db: Session, document: Document) -> None:
    document.is_deleted = True
    document.deleted_at = _now()
    document.lifecycle_status = LIFECYCLE_SOURCE_DELETED
    document.source_deleted_at = _now()
    db.commit()
    _clear_vectors_and_graph(document)


def _sync_file_list(
    db: Session,
    connection: DropboxConnection,
    user: User,
    files: list[dict[str, Any]],
    fetch_bytes: Callable[[dict[str, Any]], bytes],
    stats: dict[str, Any],
) -> None:
    for entry in files:
        file_id = entry["id"]
        revision = entry.get("rev") or ""
        existing = (
            db.query(Document)
            .filter(
                Document.user_id == user.id,
                Document.external_file_id == file_id,
                Document.source_type.in_(DROPBOX_SOURCE_TYPES),
            )
            .first()
        )

        # USER_DELETED is final until the user manually re-imports the file.
        if _tombstone_exists(db, user.id, file_id) or (
            existing is not None
            and existing.lifecycle_status == LIFECYCLE_USER_DELETED
        ):
            stats["skipped"] += 1
            continue

        is_unchanged = (
            existing is not None
            and existing.status == "ready"
            and existing.lifecycle_status == LIFECYCLE_ACTIVE
            and bool(revision)
            and (existing.external_revision or "") == revision
        )
        is_in_flight = (
            existing is not None
            and existing.lifecycle_status == LIFECYCLE_ACTIVE
            and existing.status in {"pending", "processing"}
        )
        if is_unchanged or is_in_flight:
            stats["skipped"] += 1
            continue

        try:
            data = fetch_bytes(entry)
            if not data or data[:4] != b"%PDF":
                raise ValueError("Dropbox file is not a valid PDF")

            if existing is not None:
                _apply_file(
                    db, existing, entry=entry, data=data, connection=connection
                )
                stats["updated"] += 1
            else:
                _create_file(db, user, connection, entry, data)
                stats["ingested"] += 1
        except Exception as exc:
            logger.exception(
                "Dropbox ingestion failed for file %s (%s)", file_id, entry.get("name")
            )
            stats["failed"] += 1
            if existing is not None:
                db.refresh(existing)
                existing.status = "failed"
                existing.error_message = str(exc)[:500]
                db.commit()


def _mark_missing_as_source_deleted(
    db: Session,
    connection: DropboxConnection,
    user: User,
    remote_ids: set[str],
    stats: dict[str, Any],
) -> None:
    local_documents = (
        db.query(Document)
        .filter(
            Document.user_id == user.id,
            Document.external_folder_id == connection.remote_folder_id,
            Document.source_type.in_(DROPBOX_SOURCE_TYPES),
            Document.is_deleted.is_(False),
        )
        .all()
    )
    for document in local_documents:
        if document.external_file_id not in remote_ids:
            _mark_source_deleted(db, document)
            stats["deleted"] += 1


# ── Sync entrypoints ─────────────────────────────────────────────────────────


def sync_connection(
    db: Session,
    connection: DropboxConnection,
    user: User,
) -> dict[str, Any]:
    """Synchronize one Dropbox connection (folder or shared link)."""
    stats: dict[str, Any] = {
        "connection_id": str(connection.id),
        "remote_folder_id": connection.remote_folder_id,
        "discovered": 0,
        "ingested": 0,
        "updated": 0,
        "skipped": 0,
        "deleted": 0,
        "failed": 0,
    }

    connection.sync_status = "processing"
    connection.last_error = None
    connection.updated_at = _now()
    db.commit()

    try:
        files, fetch_bytes = resolve_connection_files(db, connection, user)
        pending = _pending_ids(connection)
        if pending:
            files = [entry for entry in files if entry["id"] in pending]

        stats["discovered"] = len(files)
        stats["access_mode"] = (
            "link" if (connection.connection_kind or "oauth") == "link" else "oauth"
        )

        _sync_file_list(db, connection, user, files, fetch_bytes, stats)

        if pending:
            # A selection sync must not treat unselected files as remote deletions.
            _set_pending_ids(connection, [])
        else:
            _mark_missing_as_source_deleted(
                db, connection, user, {entry["id"] for entry in files}, stats
            )

        connection.sync_status = "completed" if stats["failed"] == 0 else "failed"
        if stats["failed"]:
            connection.last_error = f"{stats['failed']} file(s) failed during sync"
        connection.last_synced_at = _now()
        connection.updated_at = _now()
        db.commit()
        return stats

    except DropboxError as exc:
        logger.warning("Dropbox sync failed for %s: %s", connection.id, exc)
        connection.sync_status = "failed"
        connection.last_error = str(exc)[:500]
        connection.updated_at = _now()
        db.commit()
        stats["failed"] += 1
        stats["error"] = str(exc)[:500]
        return stats
    except Exception as exc:
        logger.exception("Unexpected Dropbox sync failure for %s", connection.id)
        connection.sync_status = "failed"
        connection.last_error = str(exc)[:500]
        connection.updated_at = _now()
        db.commit()
        stats["failed"] += 1
        stats["error"] = str(exc)[:500]
        return stats


def import_selected_files(
    db: Session,
    connection: DropboxConnection,
    user: User,
    file_ids: list[str],
) -> dict[str, Any]:
    """Manually re-import files the user previously deleted in Genova."""
    wanted = {str(file_id) for file_id in file_ids}
    stats: dict[str, Any] = {
        "connection_id": str(connection.id),
        "requested": len(wanted),
        "imported": 0,
        "skipped": 0,
        "failed": 0,
    }
    if not wanted:
        return stats

    files, fetch_bytes = resolve_connection_files(db, connection, user)
    selected = [entry for entry in files if entry["id"] in wanted]

    for entry in selected:
        file_id = entry["id"]
        existing = (
            db.query(Document)
            .filter(
                Document.user_id == user.id,
                Document.external_file_id == file_id,
                Document.source_type.in_(DROPBOX_SOURCE_TYPES),
            )
            .first()
        )
        was_deleted = _tombstone_exists(db, user.id, file_id) or (
            existing is not None and existing.lifecycle_status == LIFECYCLE_USER_DELETED
        )
        if not was_deleted:
            # Only explicitly deleted files can be re-imported through this path.
            stats["skipped"] += 1
            continue

        try:
            data = fetch_bytes(entry)
            if not data or data[:4] != b"%PDF":
                raise ValueError("Dropbox file is not a valid PDF")

            _remove_tombstone(db, user.id, file_id)
            if existing is not None:
                _apply_file(
                    db, existing, entry=entry, data=data, connection=connection
                )
            else:
                _create_file(db, user, connection, entry, data)
            db.commit()
            stats["imported"] += 1
        except Exception as exc:
            logger.exception("Dropbox manual re-import failed for %s", file_id)
            stats["failed"] += 1
            if existing is not None:
                db.refresh(existing)
                existing.status = "failed"
                existing.error_message = str(exc)[:500]
                db.commit()

    if selected:
        connection.last_synced_at = _now()
        connection.updated_at = _now()
        db.commit()
    return stats


def _get_connection(
    db: Session, connection_id: str, user_id: str
) -> Optional[DropboxConnection]:
    return (
        db.query(DropboxConnection)
        .filter(
            DropboxConnection.id == connection_id,
            DropboxConnection.user_id == user_id,
        )
        .first()
    )


def run_dropbox_sync_job(connection_id: str, user_id: str) -> dict[str, Any]:
    """Worker entrypoint: fresh session, one Dropbox connection."""
    from app.database import get_db_session

    with get_db_session() as db:
        connection = _get_connection(db, connection_id, user_id)
        if not connection:
            logger.warning(
                "Dropbox sync job skipped — connection %s not found for user %s",
                connection_id,
                user_id,
            )
            return {"connection_id": connection_id, "failed": 1, "error": "not_found"}

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            connection.sync_status = "failed"
            connection.last_error = "User not found"
            db.commit()
            return {
                "connection_id": connection_id,
                "failed": 1,
                "error": "user_not_found",
            }
        return sync_connection(db, connection, user)


def run_dropbox_import_job(
    connection_id: str, user_id: str, file_ids: list[str]
) -> dict[str, Any]:
    """Worker entrypoint for manual re-import of selected files."""
    from app.database import get_db_session

    with get_db_session() as db:
        connection = _get_connection(db, connection_id, user_id)
        if not connection:
            return {"connection_id": connection_id, "failed": 1, "error": "not_found"}
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"connection_id": connection_id, "failed": 1, "error": "user_not_found"}
        return import_selected_files(db, connection, user, file_ids)


def enqueue_connection_sync(
    db: Session,
    connection: DropboxConnection,
    user_id: str,
    *,
    background_tasks: Any = None,
    force: bool = False,
) -> tuple[str, bool]:
    """Mark a connection queued and dispatch Celery (or BackgroundTasks)."""
    status = (connection.sync_status or "").lower()
    if not force and status in ACTIVE_SYNC_STATUSES:
        return (connection.celery_task_id or ""), True

    connection.sync_status = "queued"
    connection.last_error = None
    connection.updated_at = _now()
    db.commit()

    connection_id = str(connection.id)
    uid = str(user_id)

    from app.celery_app import celery_app
    from app.tasks import sync_dropbox_connection_task

    use_local_bg = (
        bool(getattr(celery_app.conf, "task_always_eager", False))
        and background_tasks is not None
    )

    if use_local_bg:
        task_id = f"local_{uuid.uuid4().hex}"
        connection.celery_task_id = task_id
        db.commit()
        background_tasks.add_task(run_dropbox_sync_job, connection_id, uid)
        return task_id, False

    try:
        async_result = sync_dropbox_connection_task.delay(connection_id, uid)
        task_id = async_result.id
        connection.celery_task_id = task_id
        db.commit()
        return task_id, False
    except Exception as exc:
        logger.warning("Celery Dropbox enqueue failed, falling back: %s", exc)
        if background_tasks is not None:
            task_id = f"local_{uuid.uuid4().hex}"
            connection.celery_task_id = task_id
            db.commit()
            background_tasks.add_task(run_dropbox_sync_job, connection_id, uid)
            return task_id, False
        connection.sync_status = "failed"
        connection.last_error = f"Could not enqueue Dropbox sync: {exc}"[:500]
        db.commit()
        raise


def enqueue_selected_import(
    db: Session,
    connection: DropboxConnection,
    user_id: str,
    file_ids: list[str],
    *,
    background_tasks: Any = None,
) -> str:
    """Dispatch a background manual re-import task and return its ID."""
    from app.celery_app import celery_app
    from app.tasks import import_dropbox_files_task

    connection_id = str(connection.id)
    uid = str(user_id)
    ids = [str(file_id) for file_id in file_ids]

    use_local_bg = (
        bool(getattr(celery_app.conf, "task_always_eager", False))
        and background_tasks is not None
    )
    if use_local_bg:
        task_id = f"local_{uuid.uuid4().hex}"
        background_tasks.add_task(run_dropbox_import_job, connection_id, uid, ids)
        return task_id

    try:
        return import_dropbox_files_task.delay(connection_id, uid, ids).id
    except Exception as exc:
        logger.warning("Celery Dropbox import enqueue failed, falling back: %s", exc)
        if background_tasks is not None:
            task_id = f"local_{uuid.uuid4().hex}"
            background_tasks.add_task(run_dropbox_import_job, connection_id, uid, ids)
            return task_id
        raise


def sync_dropbox_connections(db: Session) -> dict:
    """Scheduler helper: enqueue a job per enabled Dropbox connection."""
    if not _sync_lock.acquire(blocking=False):
        logger.info("Dropbox enqueue skipped — previous scheduler tick still active")
        return {"connections": 0, "enqueued": 0, "skipped": 0, "failed": 0}

    stats = {"connections": 0, "enqueued": 0, "skipped": 0, "failed": 0}
    try:
        connections = (
            db.query(DropboxConnection)
            .filter(DropboxConnection.enabled.is_(True))
            .all()
        )
        stats["connections"] = len(connections)
        for connection in connections:
            user = db.query(User).filter(User.id == connection.user_id).first()
            if not user:
                stats["failed"] += 1
                continue
            if (
                (connection.connection_kind or "oauth") == "oauth"
                and not user.dropbox_refresh_token
            ):
                connection.sync_status = "failed"
                connection.last_error = "Owner has not connected Dropbox"
                db.commit()
                stats["failed"] += 1
                continue
            try:
                _task_id, already = enqueue_connection_sync(
                    db, connection, str(user.id), background_tasks=None
                )
                stats["skipped" if already else "enqueued"] += 1
            except Exception:
                logger.exception(
                    "Failed to enqueue scheduled Dropbox sync for %s", connection.id
                )
                stats["failed"] += 1
        logger.info("Dropbox sync enqueue complete: %s", stats)
        return stats
    finally:
        _sync_lock.release()


def sync_dropbox_connections_with_session() -> dict:
    """Scheduler entry: open a session and enqueue Dropbox sync jobs."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        return sync_dropbox_connections(db)
    finally:
        db.close()

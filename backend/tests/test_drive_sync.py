"""Tests for Google Drive folder sync and RAG ingestion bridge."""
from pathlib import Path

import pytest

from app.models import Document, DriveConnection
from app.services import drive_sync
from app.services.drive_public import (
    DrivePermissionDenied,
    parse_public_folder_entries,
)


def test_extract_folder_id_from_urls():
    assert (
        drive_sync.extract_folder_id(
            "https://drive.google.com/drive/folders/1AbCdefghijkLMNOPQrsTUvwxYZ012345"
        )
        == "1AbCdefghijkLMNOPQrsTUvwxYZ012345"
    )
    assert (
        drive_sync.extract_folder_id(
            "https://drive.google.com/drive/u/0/folders/1AbCdefghijkLMNOPQrsTUvwxYZ012345?usp=sharing"
        )
        == "1AbCdefghijkLMNOPQrsTUvwxYZ012345"
    )
    assert (
        drive_sync.extract_folder_id("1AbCdefghijkLMNOPQrsTUvwxYZ012345")
        == "1AbCdefghijkLMNOPQrsTUvwxYZ012345"
    )
    with pytest.raises(ValueError):
        drive_sync.extract_folder_id("https://example.com/not-a-folder")


def test_parse_public_folder_entries_detects_pdfs():
    html = """
    <div class="flip-entries">
      <div class="flip-entry" id="entry-fileAAA111">
        <a href="https://drive.google.com/file/d/fileAAA111/view">
          <img alt="PDF" src="type/application/pdf"/>
        </a>
        <div class="flip-entry-title">Guide.pdf</div>
      </div>
      <div class="flip-entry-last-modified">Jan 1</div>
      <div class="flip-entry" id="entry-folderBBB222">
        <a href="https://drive.google.com/drive/folders/folderBBB222">
        </a>
        <div class="flip-entry-title">Subfolder</div>
      </div>
      <div class="flip-entry-last-modified">Jan 2</div>
    </div>
    """
    entries = parse_public_folder_entries(html)
    assert len(entries) == 2
    assert entries[0]["id"] == "fileAAA111"
    assert entries[0]["mimeType"] == "application/pdf"
    assert entries[1]["mimeType"] == drive_sync.FOLDER_MIME_TYPE


def test_resolve_folder_access_public_without_oauth(user, monkeypatch):
    user.google_refresh_token = None

    monkeypatch.setattr(
        drive_sync,
        "fetch_public_folder_metadata",
        lambda folder_id: {
            "id": folder_id,
            "name": "Open Docs",
            "mimeType": drive_sync.FOLDER_MIME_TYPE,
        },
    )

    mode, meta = drive_sync.resolve_folder_access("publicFolder123", user)
    assert mode == "public"
    assert meta["name"] == "Open Docs"


def test_resolve_folder_access_private_requires_oauth(user, monkeypatch):
    user.google_refresh_token = None

    def deny(_folder_id):
        raise DrivePermissionDenied("private")

    monkeypatch.setattr(drive_sync, "fetch_public_folder_metadata", deny)

    with pytest.raises(DrivePermissionDenied):
        drive_sync.resolve_folder_access("privateFolder123", user)


def test_resolve_folder_access_private_with_oauth(user, monkeypatch):
    user.google_refresh_token = "refresh-token"

    def deny(_folder_id):
        raise DrivePermissionDenied("private")

    monkeypatch.setattr(drive_sync, "fetch_public_folder_metadata", deny)
    monkeypatch.setattr(
        drive_sync, "_build_drive_service_for_user", lambda u: object()
    )
    monkeypatch.setattr(
        drive_sync,
        "fetch_folder_metadata",
        lambda service, folder_id: {
            "id": folder_id,
            "name": "Private Specs",
            "mimeType": drive_sync.FOLDER_MIME_TYPE,
        },
    )

    mode, meta = drive_sync.resolve_folder_access("privateFolder123", user)
    assert mode == "oauth"
    assert meta["name"] == "Private Specs"


def test_drive_sync_ingests_new_pdf_and_skips_existing(db_session, user, tmp_path, monkeypatch):
    monkeypatch.setattr(drive_sync.settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(drive_sync.settings, "GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setattr(drive_sync.settings, "GOOGLE_CLIENT_SECRET", "csecret")
    user.google_refresh_token = "refresh-token"
    db_session.commit()

    monkeypatch.setattr(
        drive_sync,
        "_build_drive_service",
        lambda connection, user=None: object(),
    )
    monkeypatch.setattr(
        drive_sync,
        "list_pdf_files_recursive",
        lambda service, folder_id: [
            {
                "id": "drive-file-1",
                "name": "Guide.pdf",
                "size": "12",
                "modifiedTime": "2026-01-01T00:00:00.000Z",
            },
        ],
    )
    monkeypatch.setattr(
        drive_sync,
        "fetch_folder_metadata",
        lambda service, folder_id: {
            "id": folder_id,
            "name": "Research",
            "mimeType": drive_sync.FOLDER_MIME_TYPE,
        },
    )

    def fake_download(service, file_id, destination):
        Path(destination).write_bytes(b"%PDF-1.4\n")

    def fake_ingest(document_id, filepath, original_name, user_id):
        document = db_session.query(Document).filter(Document.id == document_id).one()
        document.status = "ready"
        document.page_count = 1
        document.chunk_count = 1
        db_session.commit()

    monkeypatch.setattr(drive_sync, "_download_drive_file", fake_download)
    monkeypatch.setattr(drive_sync, "ingest_document", fake_ingest)

    connection = DriveConnection(
        user_id=user.id,
        folder_id="folder-1",
        folder_name="Research",
        access_mode="oauth",
    )
    db_session.add(connection)
    db_session.commit()

    first = drive_sync.sync_connection(db_session, connection, user)
    second = drive_sync.sync_connection(db_session, connection, user)

    document = (
        db_session.query(Document)
        .filter(Document.drive_file_id == "drive-file-1")
        .one()
    )
    assert first["ingested"] == 1
    assert second["skipped"] == 1
    assert document.status == "ready"
    assert document.original_name == "Guide.pdf"
    assert document.drive_folder_id == "folder-1"
    assert document.drive_modified_time == "2026-01-01T00:00:00.000Z"
    db_session.refresh(connection)
    assert connection.sync_status == "completed"


def test_drive_sync_public_folder_without_oauth(db_session, user, tmp_path, monkeypatch):
    monkeypatch.setattr(drive_sync.settings, "UPLOAD_DIR", str(tmp_path))
    user.google_refresh_token = None
    db_session.commit()

    monkeypatch.setattr(
        drive_sync,
        "probe_public_folder",
        lambda folder_id: (
            {"id": folder_id, "name": "Shared", "mimeType": drive_sync.FOLDER_MIME_TYPE},
            [
                {
                    "id": "pub-file-1",
                    "name": "Public.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "",
                }
            ],
        ),
    )

    def fake_public_download(file_id, destination):
        Path(destination).write_bytes(b"%PDF-1.4\npublic\n")

    def fake_ingest(document_id, filepath, original_name, user_id):
        document = db_session.query(Document).filter(Document.id == document_id).one()
        document.status = "ready"
        document.chunk_count = 1
        db_session.commit()

    monkeypatch.setattr(drive_sync, "download_public_file", fake_public_download)
    monkeypatch.setattr(drive_sync, "ingest_document", fake_ingest)

    connection = DriveConnection(
        user_id=user.id,
        folder_id="public-folder-1",
        folder_name="Shared",
        access_mode="public",
    )
    db_session.add(connection)
    db_session.commit()

    result = drive_sync.sync_connection(db_session, connection, user)
    assert result["ingested"] == 1
    assert result["access_mode"] == "public"

    document = (
        db_session.query(Document)
        .filter(Document.drive_file_id == "pub-file-1")
        .one()
    )
    assert document.original_name == "Public.pdf"
    assert document.status == "ready"
    db_session.refresh(connection)
    assert connection.sync_status == "completed"


def test_drive_sync_updates_modified_pdf(db_session, user, tmp_path, monkeypatch):
    monkeypatch.setattr(drive_sync.settings, "UPLOAD_DIR", str(tmp_path))
    user.google_refresh_token = "refresh-token"
    db_session.commit()

    monkeypatch.setattr(
        drive_sync, "_build_drive_service", lambda connection, user=None: object()
    )
    monkeypatch.setattr(
        drive_sync,
        "fetch_folder_metadata",
        lambda service, folder_id: {
            "id": folder_id,
            "name": "Research",
            "mimeType": drive_sync.FOLDER_MIME_TYPE,
        },
    )

    files = [
        {
            "id": "drive-file-2",
            "name": "Notes.pdf",
            "size": "10",
            "modifiedTime": "2026-01-01T00:00:00.000Z",
        }
    ]
    monkeypatch.setattr(
        drive_sync, "list_pdf_files_recursive", lambda service, folder_id: files
    )
    monkeypatch.setattr(
        drive_sync,
        "_download_drive_file",
        lambda service, file_id, destination: Path(destination).write_bytes(b"%PDF-1.4\nv1\n"),
    )

    ingest_calls = {"n": 0}

    def fake_ingest(document_id, filepath, original_name, user_id):
        ingest_calls["n"] += 1
        document = db_session.query(Document).filter(Document.id == document_id).one()
        document.status = "ready"
        document.chunk_count = ingest_calls["n"]
        db_session.commit()

    def fake_reingest(db, document, filepath):
        fake_ingest(document.id, filepath, document.original_name, document.user_id)

    monkeypatch.setattr(drive_sync, "ingest_document", fake_ingest)
    monkeypatch.setattr(drive_sync, "_reingest_document", fake_reingest)

    connection = DriveConnection(
        user_id=user.id, folder_id="folder-2", access_mode="oauth"
    )
    db_session.add(connection)
    db_session.commit()

    drive_sync.sync_connection(db_session, connection, user)
    files[0]["modifiedTime"] = "2026-02-01T00:00:00.000Z"
    files[0]["name"] = "Notes-v2.pdf"
    result = drive_sync.sync_connection(db_session, connection, user)

    document = (
        db_session.query(Document)
        .filter(Document.drive_file_id == "drive-file-2")
        .one()
    )
    assert result["updated"] == 1
    assert document.drive_modified_time == "2026-02-01T00:00:00.000Z"
    assert document.original_name == "Notes-v2.pdf"


def test_drive_sync_soft_deletes_removed_pdf(db_session, user, tmp_path, monkeypatch):
    monkeypatch.setattr(drive_sync.settings, "UPLOAD_DIR", str(tmp_path))
    user.google_refresh_token = "refresh-token"
    db_session.commit()

    monkeypatch.setattr(
        drive_sync, "_build_drive_service", lambda connection, user=None: object()
    )
    monkeypatch.setattr(
        drive_sync,
        "fetch_folder_metadata",
        lambda service, folder_id: {
            "id": folder_id,
            "name": "Research",
            "mimeType": drive_sync.FOLDER_MIME_TYPE,
        },
    )

    remote = [
        {
            "id": "drive-file-3",
            "name": "Temp.pdf",
            "size": "8",
            "modifiedTime": "2026-01-01T00:00:00.000Z",
        }
    ]
    monkeypatch.setattr(
        drive_sync, "list_pdf_files_recursive", lambda service, folder_id: list(remote)
    )
    monkeypatch.setattr(
        drive_sync,
        "_download_drive_file",
        lambda service, file_id, destination: Path(destination).write_bytes(b"%PDF-1.4\n"),
    )

    def fake_ingest(document_id, filepath, original_name, user_id):
        document = db_session.query(Document).filter(Document.id == document_id).one()
        document.status = "ready"
        db_session.commit()

    monkeypatch.setattr(drive_sync, "ingest_document", fake_ingest)

    connection = DriveConnection(
        user_id=user.id, folder_id="folder-3", access_mode="oauth"
    )
    db_session.add(connection)
    db_session.commit()

    drive_sync.sync_connection(db_session, connection, user)
    remote.clear()
    result = drive_sync.sync_connection(db_session, connection, user)

    document = (
        db_session.query(Document)
        .filter(Document.drive_file_id == "drive-file-3")
        .one()
    )
    assert result["deleted"] == 1
    assert document.is_deleted is True


def test_add_public_drive_folder_without_oauth(
    client, auth_headers, user, db_session, monkeypatch
):
    user.google_refresh_token = None
    db_session.commit()

    monkeypatch.setattr(
        drive_sync,
        "resolve_folder_access",
        lambda folder_id, user=None: (
            "public",
            {
                "id": folder_id,
                "name": "Open Specs",
                "mimeType": drive_sync.FOLDER_MIME_TYPE,
            },
        ),
    )
    enqueued = []

    def fake_enqueue(db, connection, user_id, background_tasks=None, force=False):
        connection.sync_status = "queued"
        connection.celery_task_id = "task-public-1"
        db.commit()
        enqueued.append(str(connection.id))
        return "task-public-1", False

    monkeypatch.setattr(drive_sync, "enqueue_connection_sync", fake_enqueue)

    response = client.post(
        "/api/v1/drive/folders",
        headers=auth_headers,
        json={
            "url": "https://drive.google.com/drive/folders/publicFolderABC12345"
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["folder_id"] == "publicFolderABC12345"
    assert body["folder_name"] == "Open Specs"
    assert body["access_mode"] == "public"
    assert body["sync_status"] == "queued"
    assert body["task_id"] == "task-public-1"
    assert enqueued  # queued without running sync_connection

    listed = client.get("/api/v1/drive/folders", headers=auth_headers)
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["connected"] is False
    assert payload["public_access_available"] is True
    assert len(payload["folders"]) == 1
    assert payload["folders"][0]["access_mode"] == "public"


def test_add_drive_folder_api(client, auth_headers, user, db_session, monkeypatch):
    user.google_refresh_token = "refresh-token"
    db_session.commit()

    monkeypatch.setattr(
        drive_sync,
        "resolve_folder_access",
        lambda folder_id, user=None: (
            "oauth",
            {
                "id": folder_id,
                "name": "Specs",
                "mimeType": drive_sync.FOLDER_MIME_TYPE,
            },
        ),
    )

    def fake_enqueue(db, connection, user_id, background_tasks=None, force=False):
        connection.sync_status = "queued"
        connection.celery_task_id = "task-oauth-1"
        db.commit()
        return "task-oauth-1", False

    monkeypatch.setattr(drive_sync, "enqueue_connection_sync", fake_enqueue)

    response = client.post(
        "/api/v1/drive/folders",
        headers=auth_headers,
        json={
            "url": "https://drive.google.com/drive/folders/folderABC1234567890"
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["folder_id"] == "folderABC1234567890"
    assert body["folder_name"] == "Specs"
    assert body["access_mode"] == "oauth"
    assert body["task_id"] == "task-oauth-1"

    listed = client.get("/api/v1/drive/folders", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()["connected"] is True
    assert len(listed.json()["folders"]) == 1


def test_drive_folder_requires_oauth_when_private(
    client, auth_headers, user, db_session, monkeypatch
):
    user.google_refresh_token = None
    db_session.commit()

    def deny(folder_id, user=None):
        raise DrivePermissionDenied(
            "This Drive folder is private. Connect Google Drive in Settings, then retry."
        )

    monkeypatch.setattr(drive_sync, "resolve_folder_access", deny)

    response = client.post(
        "/api/v1/drive/folders",
        headers=auth_headers,
        json={"url": "https://drive.google.com/drive/folders/folderABC1234567890"},
    )
    assert response.status_code == 403
    body = response.json()
    assert body["error"]["code"] == "DRIVE_OAUTH_REQUIRED"
    assert "private" in body["error"]["message"].lower() or "connect" in body[
        "error"
    ]["message"].lower()


def test_drive_folder_isolation(client, auth_headers, user, other_user, db_session):
    other_conn = DriveConnection(
        user_id=other_user.id,
        folder_id="other-folder-id-xx",
        folder_name="Secret",
        access_mode="public",
    )
    db_session.add(other_conn)
    db_session.commit()

    listed = client.get("/api/v1/drive/folders", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()["folders"] == []


def test_sync_endpoint_returns_202_without_running_sync(
    client, auth_headers, user, db_session, monkeypatch
):
    """API must enqueue and return immediately — no Drive I/O in the request."""
    connection = DriveConnection(
        user_id=user.id,
        folder_id="folder-sync-async",
        folder_name="Async",
        access_mode="public",
        sync_status="idle",
    )
    db_session.add(connection)
    db_session.commit()

    sync_calls = []

    def boom(*args, **kwargs):
        sync_calls.append(1)
        raise AssertionError("sync_connection must not run inside the API request")

    monkeypatch.setattr(drive_sync, "sync_connection", boom)

    def fake_enqueue(db, conn, user_id, background_tasks=None, force=False):
        conn.sync_status = "queued"
        conn.celery_task_id = "celery-task-xyz"
        db.commit()
        return "celery-task-xyz", False

    monkeypatch.setattr(drive_sync, "enqueue_connection_sync", fake_enqueue)

    response = client.post(
        f"/api/v1/drive/folders/{connection.id}/sync",
        headers=auth_headers,
    )
    assert response.status_code == 202
    body = response.json()
    assert body["sync_status"] == "queued"
    assert body["task_id"] == "celery-task-xyz"
    assert body["already_running"] is False
    assert sync_calls == []

    status = client.get(
        f"/api/v1/drive/folders/{connection.id}/status",
        headers=auth_headers,
    )
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["sync_status"] == "queued"
    assert status_body["task_id"] == "celery-task-xyz"


def test_sync_all_endpoint_returns_202(client, auth_headers, user, db_session, monkeypatch):
    for folder_id in ("folder-a-1111111111", "folder-b-2222222222"):
        db_session.add(
            DriveConnection(
                user_id=user.id,
                folder_id=folder_id,
                access_mode="public",
                sync_status="idle",
                enabled=True,
            )
        )
    db_session.commit()

    def fake_enqueue(db, conn, user_id, background_tasks=None, force=False):
        tid = f"t-{conn.folder_id}"
        conn.sync_status = "queued"
        conn.celery_task_id = tid
        db.commit()
        return tid, False

    monkeypatch.setattr(drive_sync, "enqueue_connection_sync", fake_enqueue)

    response = client.post("/api/v1/drive/sync", headers=auth_headers)
    assert response.status_code == 202
    body = response.json()
    assert body["connections"] == 2
    assert len(body["task_ids"]) == 2


def test_celery_drive_task_runs_sync_job(db_session, user, monkeypatch):
    from app.tasks import sync_drive_folder_task

    connection = DriveConnection(
        user_id=user.id,
        folder_id="folder-worker-1",
        access_mode="public",
        sync_status="queued",
    )
    db_session.add(connection)
    db_session.commit()

    seen = {}

    def fake_job(connection_id, user_id):
        seen["connection_id"] = connection_id
        seen["user_id"] = user_id
        return {"connection_id": connection_id, "ingested": 2, "failed": 0}

    monkeypatch.setattr(drive_sync, "run_drive_sync_job", fake_job)

    result = sync_drive_folder_task.apply(
        args=(str(connection.id), str(user.id))
    ).get()
    assert result["ingested"] == 2
    assert seen["connection_id"] == str(connection.id)
    assert seen["user_id"] == str(user.id)


def test_enqueue_skips_duplicate_in_flight_sync(db_session, user):
    connection = DriveConnection(
        user_id=user.id,
        folder_id="folder-dup-1",
        access_mode="public",
        sync_status="processing",
        celery_task_id="existing-task",
    )
    db_session.add(connection)
    db_session.commit()

    task_id, already = drive_sync.enqueue_connection_sync(
        db_session, connection, str(user.id)
    )
    assert already is True
    assert task_id == "existing-task"

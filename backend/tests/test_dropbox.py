"""Tests for the Dropbox document source (OAuth + shared links + lifecycle)."""
import io
import zipfile

import pytest

from app.models import (
    DOCUMENT_SOURCE_DROPBOX,
    DOCUMENT_SOURCE_DROPBOX_LINK,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_SOURCE_DELETED,
    LIFECYCLE_USER_DELETED,
    Document,
    DropboxConnection,
    DropboxDocumentTombstone,
)
from app.services import dropbox_client, dropbox_sync

TEST_LINK = (
    "https://www.dropbox.com/scl/fo/bc9imczz3bxl5ya8l7cil/"
    "AN5l1yDnWejG9jYIR_fD6A0?rlkey=rp14xvckn7zin1bw4axe50bjt"
    "&st=tzxfo2y1&e=1&dl=0"
)


def _make_connection(db_session, user, *, kind="oauth", remote_id="folder-1"):
    connection = DropboxConnection(
        user_id=user.id,
        connection_kind=kind,
        remote_folder_id=remote_id,
        folder_path="/Docs" if kind == "oauth" else "",
        shared_link=TEST_LINK if kind == "link" else None,
        folder_name="Docs",
        sync_status="idle",
    )
    db_session.add(connection)
    db_session.commit()
    db_session.refresh(connection)
    return connection


def _patch_files(db_session, monkeypatch, files, data_by_id, ingest_calls=None):
    def fake_resolve(_db, _connection, _user):
        return list(files), lambda entry: data_by_id[entry["id"]]

    def fake_ingest(document_id, filepath, original_name, user_id):
        document = db_session.query(Document).filter(Document.id == document_id).one()
        document.status = "ready"
        document.chunk_count = 1
        document.page_count = 1
        db_session.commit()
        if ingest_calls is not None:
            ingest_calls.append(document_id)

    monkeypatch.setattr(dropbox_sync, "resolve_connection_files", fake_resolve)
    monkeypatch.setattr(dropbox_sync, "ingest_document", fake_ingest)


def _file(rev="rev-1", name="Guide.pdf", file_id="dbx-file-1"):
    return {
        "id": file_id,
        "name": name,
        "path": f"/Docs/{name}",
        "rev": rev,
        "size": 12,
        "modified": "2026-01-01T00:00:00Z",
    }


# ── Client / OAuth ───────────────────────────────────────────────────────────


def test_normalize_shared_link_strips_web_only_params():
    normalized = dropbox_client.normalize_shared_link(TEST_LINK)
    assert "rlkey=rp14xvckn7zin1bw4axe50bjt" in normalized
    assert "dl=" not in normalized
    assert "raw=" not in normalized
    assert dropbox_client.shared_link_key(TEST_LINK).startswith("link:")


def test_normalize_shared_link_rejects_non_dropbox_urls():
    with pytest.raises(dropbox_client.DropboxInvalidLink):
        dropbox_client.normalize_shared_link("https://example.com/file.pdf")
    with pytest.raises(dropbox_client.DropboxInvalidLink):
        dropbox_client.normalize_shared_link("https://www.dropbox.com/home")


def test_build_authorize_url_uses_env_config(monkeypatch):
    monkeypatch.setattr(dropbox_client.settings, "DROPBOX_APP_KEY", "app-key")
    monkeypatch.setattr(dropbox_client.settings, "DROPBOX_APP_SECRET", "app-secret")
    monkeypatch.setattr(
        dropbox_client.settings, "DROPBOX_REDIRECT_URI", "http://localhost/cb"
    )

    url = dropbox_client.build_authorize_url("state-123")
    assert "client_id=app-key" in url
    assert "token_access_type=offline" in url
    assert "state=state-123" in url
    assert "files.content.read" in url


def test_refresh_access_token_sends_refresh_grant(monkeypatch):
    seen = {}

    def fake_token_request(data):
        seen.update(data)
        return {"access_token": "fresh-token", "expires_in": 14400}

    monkeypatch.setattr(dropbox_client, "_token_request", fake_token_request)
    result = dropbox_client.refresh_access_token("refresh-abc")

    assert result["access_token"] == "fresh-token"
    assert seen["grant_type"] == "refresh_token"
    assert seen["refresh_token"] == "refresh-abc"


# ── Lifecycle ────────────────────────────────────────────────────────────────


def test_sync_ingests_new_then_skips_unchanged(
    db_session, user, tmp_path, monkeypatch
):
    monkeypatch.setattr(dropbox_sync.settings, "UPLOAD_DIR", str(tmp_path))
    files = [_file()]
    _patch_files(db_session, monkeypatch, files, {"dbx-file-1": b"%PDF-1.4\nnew\n"})

    connection = _make_connection(db_session, user)
    first = dropbox_sync.sync_connection(db_session, connection, user)
    second = dropbox_sync.sync_connection(db_session, connection, user)

    document = (
        db_session.query(Document)
        .filter(Document.external_file_id == "dbx-file-1")
        .one()
    )
    assert first["ingested"] == 1
    assert second["skipped"] == 1
    assert document.status == "ready"
    assert document.source_type == DOCUMENT_SOURCE_DROPBOX
    assert document.external_revision == "rev-1"
    assert document.external_folder_id == "folder-1"
    assert document.lifecycle_status == LIFECYCLE_ACTIVE
    db_session.refresh(connection)
    assert connection.sync_status == "completed"


def test_sync_reingests_modified_pdf(db_session, user, tmp_path, monkeypatch):
    monkeypatch.setattr(dropbox_sync.settings, "UPLOAD_DIR", str(tmp_path))
    files = [_file()]
    data = {"dbx-file-1": b"%PDF-1.4\nv1\n"}
    _patch_files(db_session, monkeypatch, files, data)
    connection = _make_connection(db_session, user)

    dropbox_sync.sync_connection(db_session, connection, user)

    files[0] = _file(rev="rev-2", name="Guide-v2.pdf")
    data["dbx-file-1"] = b"%PDF-1.4\nv2\n"
    result = dropbox_sync.sync_connection(db_session, connection, user)

    document = (
        db_session.query(Document)
        .filter(Document.external_file_id == "dbx-file-1")
        .one()
    )
    assert result["updated"] == 1
    assert document.external_revision == "rev-2"
    assert document.original_name == "Guide-v2.pdf"
    assert document.status == "ready"


def test_sync_marks_missing_pdf_as_source_deleted(
    db_session, user, tmp_path, monkeypatch
):
    monkeypatch.setattr(dropbox_sync.settings, "UPLOAD_DIR", str(tmp_path))
    files = [_file()]
    _patch_files(db_session, monkeypatch, files, {"dbx-file-1": b"%PDF-1.4\n"})
    connection = _make_connection(db_session, user)

    dropbox_sync.sync_connection(db_session, connection, user)
    files.clear()
    result = dropbox_sync.sync_connection(db_session, connection, user)

    document = (
        db_session.query(Document)
        .filter(Document.external_file_id == "dbx-file-1")
        .one()
    )
    assert result["deleted"] == 1
    assert document.is_deleted is True
    assert document.lifecycle_status == LIFECYCLE_SOURCE_DELETED


def test_user_deleted_pdf_is_not_restored_and_can_be_reimported(
    db_session, user, tmp_path, monkeypatch
):
    monkeypatch.setattr(dropbox_sync.settings, "UPLOAD_DIR", str(tmp_path))
    files = [_file()]
    _patch_files(db_session, monkeypatch, files, {"dbx-file-1": b"%PDF-1.4\n"})
    connection = _make_connection(db_session, user)

    dropbox_sync.sync_connection(db_session, connection, user)
    document = (
        db_session.query(Document)
        .filter(Document.external_file_id == "dbx-file-1")
        .one()
    )

    # Simulate the DELETE /documents endpoint.
    document.is_deleted = True
    document.deleted_at = dropbox_sync._now()
    document.lifecycle_status = LIFECYCLE_USER_DELETED
    db_session.add(
        DropboxDocumentTombstone(
            user_id=user.id, external_file_id=document.external_file_id
        )
    )
    db_session.commit()

    result = dropbox_sync.sync_connection(db_session, connection, user)
    db_session.refresh(document)
    assert result["skipped"] == 1
    assert document.is_deleted is True
    assert document.lifecycle_status == LIFECYCLE_USER_DELETED

    # Manual selection is the only way back.
    reimport = dropbox_sync.import_selected_files(
        db_session, connection, user, ["dbx-file-1"]
    )
    db_session.refresh(document)
    assert reimport["imported"] == 1
    assert document.is_deleted is False
    assert document.lifecycle_status == LIFECYCLE_ACTIVE
    assert (
        db_session.query(DropboxDocumentTombstone)
        .filter(DropboxDocumentTombstone.external_file_id == "dbx-file-1")
        .first()
        is None
    )


def test_reimport_skips_files_that_were_not_deleted(
    db_session, user, tmp_path, monkeypatch
):
    monkeypatch.setattr(dropbox_sync.settings, "UPLOAD_DIR", str(tmp_path))
    files = [_file()]
    _patch_files(db_session, monkeypatch, files, {"dbx-file-1": b"%PDF-1.4\n"})
    connection = _make_connection(db_session, user)

    dropbox_sync.sync_connection(db_session, connection, user)
    result = dropbox_sync.import_selected_files(
        db_session, connection, user, ["dbx-file-1"]
    )
    assert result["imported"] == 0
    assert result["skipped"] == 1


def test_shared_link_archive_import_without_oauth(
    db_session, user, tmp_path, monkeypatch
):
    monkeypatch.setattr(dropbox_sync.settings, "UPLOAD_DIR", str(tmp_path))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Payment/invoice.pdf", b"%PDF-1.4\ninvoice\n")
        archive.writestr("Payment/readme.txt", b"not a pdf")
    archive_bytes = buffer.getvalue()

    monkeypatch.setattr(dropbox_sync, "_link_access_token", lambda _db, _user: None)
    monkeypatch.setattr(
        dropbox_client,
        "download_public_shared_link",
        lambda _url: ("Payment.zip", archive_bytes, "application/zip"),
    )
    _patch_files_called = {}

    def fake_ingest(document_id, filepath, original_name, user_id):
        document = db_session.query(Document).filter(Document.id == document_id).one()
        document.status = "ready"
        db_session.commit()
        _patch_files_called["name"] = original_name

    monkeypatch.setattr(dropbox_sync, "ingest_document", fake_ingest)

    connection = _make_connection(
        db_session, user, kind="link", remote_id="link-folder-1"
    )
    first = dropbox_sync.sync_connection(db_session, connection, user)
    second = dropbox_sync.sync_connection(db_session, connection, user)

    document = (
        db_session.query(Document)
        .filter(Document.source_type == DOCUMENT_SOURCE_DROPBOX_LINK)
        .one()
    )
    assert first["ingested"] == 1
    assert second["skipped"] == 1
    assert document.original_name == "invoice.pdf"
    assert document.external_file_id.startswith("link:link-folder-1:")
    assert _patch_files_called["name"] == "invoice.pdf"


# ── API ──────────────────────────────────────────────────────────────────────


def test_status_endpoint_reports_configuration(
    client, auth_headers, monkeypatch
):
    monkeypatch.setattr(dropbox_client, "is_configured", lambda: True)
    monkeypatch.setattr(
        dropbox_client.settings, "DROPBOX_TEST_SHARED_LINK", TEST_LINK
    )

    response = client.get("/api/v1/dropbox/status", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is False
    assert body["configured"] is True
    assert body["test_shared_link"] == TEST_LINK


def test_add_shared_link_connection_queues_sync(
    client, auth_headers, user, db_session, monkeypatch
):
    monkeypatch.setattr(
        dropbox_sync,
        "describe_shared_link",
        lambda _db, _user, url: {
            "remote_folder_id": "link-folder-1",
            "folder_name": "Payment",
            "folder_url": url,
            "folder_path": "",
            "is_folder": True,
        },
    )

    def fake_enqueue(db, connection, user_id, background_tasks=None, force=False):
        connection.sync_status = "queued"
        connection.celery_task_id = "task-link-1"
        db.commit()
        return "task-link-1", False

    monkeypatch.setattr(dropbox_sync, "enqueue_connection_sync", fake_enqueue)

    response = client.post(
        "/api/v1/dropbox/connections",
        headers=auth_headers,
        json={"shared_link": TEST_LINK},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["connection_kind"] == "link"
    assert body["remote_folder_id"] == "link-folder-1"
    assert body["folder_name"] == "Payment"
    assert body["task_id"] == "task-link-1"

    listed = client.get("/api/v1/dropbox/connections", headers=auth_headers)
    assert listed.status_code == 200
    assert len(listed.json()["connections"]) == 1


def test_add_oauth_folder_requires_connection(
    client, auth_headers, user, db_session
):
    user.dropbox_refresh_token = None
    db_session.commit()

    response = client.post(
        "/api/v1/dropbox/connections",
        headers=auth_headers,
        json={"folder_path": "/Docs"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "DROPBOX_OAUTH_REQUIRED"


def test_dropbox_connection_isolation(
    client, auth_headers, user, other_user, db_session
):
    db_session.add(
        DropboxConnection(
            user_id=other_user.id,
            connection_kind="link",
            remote_folder_id="other-link-1",
            shared_link=TEST_LINK,
            folder_name="Secret",
        )
    )
    db_session.commit()

    response = client.get("/api/v1/dropbox/connections", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["connections"] == []


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/dropbox/callback",
        "/api/v1/auth/dropbox/callback",
        "/api/dropbox/callback",
    ],
)
def test_callback_paths_reject_invalid_state(client, path):
    response = client.get(f"{path}?code=abc&state=not-a-token")
    assert response.status_code == 400
    assert "not connected" in response.text.lower()


def test_shared_link_download_falls_back_to_public_archive(monkeypatch):
    import types

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Payment/invoice.pdf", b"%PDF-1.4\ninvoice\n")
    archive_bytes = buffer.getvalue()

    files = [
        {
            "id": "id:shared-1",
            "name": "invoice.pdf",
            "path": "/invoice.pdf",
            "rev": "rev-1",
            "size": 20,
            "modified": "",
        }
    ]

    class FakeClient:
        def __init__(self, _token):
            pass

        def download_shared_link(self, _url, path=None):
            raise dropbox_client.DropboxPermissionDenied("missing sharing.read")

    monkeypatch.setattr(dropbox_sync, "DropboxClient", FakeClient)
    monkeypatch.setattr(
        dropbox_sync,
        "list_shared_link_pdfs",
        lambda _client, _link, _path="": list(files),
    )
    monkeypatch.setattr(
        dropbox_sync, "_link_access_token", lambda _db, _user: "app-token"
    )
    monkeypatch.setattr(
        dropbox_client,
        "download_public_shared_link",
        lambda _url: ("Payment.zip", archive_bytes, "application/zip"),
    )

    connection = DropboxConnection(
        connection_kind="link",
        remote_folder_id="link-fallback",
        shared_link=TEST_LINK,
    )
    user = types.SimpleNamespace(dropbox_refresh_token=None)
    listed, fetch = dropbox_sync.resolve_connection_files(None, connection, user)

    data = fetch(listed[0])
    assert data.startswith(b"%PDF")
    assert b"invoice" in data


def test_deleting_dropbox_document_marks_tombstone(
    client, auth_headers, user, db_session, ready_document
):
    ready_document.source_type = DOCUMENT_SOURCE_DROPBOX
    ready_document.external_file_id = "dbx-delete-1"
    db_session.commit()

    response = client.delete(
        f"/api/v1/documents/{ready_document.id}", headers=auth_headers
    )
    assert response.status_code == 200

    db_session.refresh(ready_document)
    assert ready_document.is_deleted is True
    assert ready_document.lifecycle_status == LIFECYCLE_USER_DELETED
    tombstone = (
        db_session.query(DropboxDocumentTombstone)
        .filter(
            DropboxDocumentTombstone.user_id == user.id,
            DropboxDocumentTombstone.external_file_id == "dbx-delete-1",
        )
        .first()
    )
    assert tombstone is not None


def test_trash_restore_requires_dropbox_reimport(
    client, auth_headers, user, db_session, ready_document
):
    ready_document.source_type = DOCUMENT_SOURCE_DROPBOX
    ready_document.external_file_id = "dbx-trash-1"
    ready_document.is_deleted = True
    ready_document.deleted_at = dropbox_sync._now()
    ready_document.lifecycle_status = LIFECYCLE_USER_DELETED
    db_session.commit()

    response = client.post(
        f"/api/v1/documents/{ready_document.id}/restore", headers=auth_headers
    )
    assert response.status_code == 400
    assert "Dropbox" in response.text

    db_session.refresh(ready_document)
    assert ready_document.is_deleted is True


def test_enqueue_skips_duplicate_in_flight_sync(db_session, user):
    connection = _make_connection(db_session, user, remote_id="folder-dup")
    connection.sync_status = "processing"
    connection.celery_task_id = "existing-task"
    db_session.commit()

    task_id, already = dropbox_sync.enqueue_connection_sync(
        db_session, connection, str(user.id)
    )
    assert already is True
    assert task_id == "existing-task"

"""Tests for per-user S3 storage (key layout, notes JSON, PDF mirror/stream)."""
import io
import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from app.models import Document
from app.services import s3_storage


def _missing_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
        "GetObject",
    )


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def iter_chunks(self, chunk_size: int = 65536):
        yield self._data


class _FakeStoredPdf:
    """Mirrors the contract of ``s3_storage.open_pdf()`` (a stream proxy)."""

    def __init__(self, data: bytes):
        self._body = _FakeBody(data)
        self.content_length = len(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_chunks(self, chunk_size: int = 65536):
        yield from self._body.iter_chunks(chunk_size)


class _FakeS3:
    def __init__(self, objects=None, fail_get: bool = False):
        self.objects = objects or {}
        self.fail_get = fail_get
        self.puts: list[dict] = []
        self.uploads: list[dict] = []
        self.deletes: list[dict] = []

    def get_object(self, Bucket, Key):
        if self.fail_get or Key not in self.objects:
            raise _missing_error()
        return self.objects[Key]

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {}

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.uploads.append({"filename": filename, "bucket": bucket, "key": key, "extra": ExtraArgs})

    def download_file(self, bucket, key, destination):
        if key not in self.objects:
            raise _missing_error()
        with open(destination, "wb") as handle:
            handle.write(self.objects[key]["Body"].read())

    def delete_object(self, Bucket, Key):
        self.deletes.append({"Bucket": Bucket, "Key": Key})


@pytest.fixture()
def fake_bucket(monkeypatch):
    monkeypatch.setattr(s3_storage.settings, "S3_BUCKET", "genova-test-bucket")
    monkeypatch.setattr(s3_storage.settings, "AWS_S3_BUCKET", "genova-test-bucket")
    monkeypatch.setattr(s3_storage.settings, "S3_BASE_PREFIX", "NoteBase")
    fake = _FakeS3()
    monkeypatch.setattr(s3_storage, "_client", lambda: fake)
    return fake


# ── key layout ───────────────────────────────────────────────────────────────

def test_pdf_key_is_scoped_to_user():
    assert s3_storage.pdf_key("user-1", "abc.pdf") == "NoteBase/user-1/pdfs/abc.pdf"
    assert s3_storage.pdf_key("user-2", "abc.pdf") == "NoteBase/user-2/pdfs/abc.pdf"


def test_pdf_key_strips_path_traversal():
    assert (
        s3_storage.pdf_key("user-1", "../../other/secret.pdf")
        == "NoteBase/user-1/pdfs/secret.pdf"
    )


def test_notes_key_is_scoped_to_user():
    assert s3_storage.notes_key("user-1") == "NoteBase/user-1/sticky_notes/notes.json"


# ── notes JSON ───────────────────────────────────────────────────────────────

def test_load_notes_missing_object_returns_none(fake_bucket):
    assert s3_storage.load_notes("user-1") == (None, None)


def test_load_notes_returns_structured_json(fake_bucket):
    document = {"main": [{"name": "paragraph", "children": [{"data": "Hello"}]}]}
    fake_bucket.objects["NoteBase/user-1/sticky_notes/notes.json"] = {
        "Body": _FakeBody(json.dumps(document).encode()),
        "ContentLength": 10,
    }

    loaded, _ = s3_storage.load_notes("user-1")

    assert loaded == document


def test_load_notes_corrupt_json_is_treated_as_missing(fake_bucket):
    fake_bucket.objects["NoteBase/user-1/sticky_notes/notes.json"] = {
        "Body": _FakeBody(b"{not json"),
        "ContentLength": 9,
    }

    assert s3_storage.load_notes("user-1")[0] is None


def test_save_notes_writes_json_to_user_folder(fake_bucket):
    document = {"main": [{"name": "paragraph", "children": [{"data": "Notes ✓"}]}]}

    saved_at = s3_storage.save_notes("user-1", document)

    assert saved_at
    assert len(fake_bucket.puts) == 1
    put = fake_bucket.puts[0]
    assert put["Bucket"] == "genova-test-bucket"
    assert put["Key"] == "NoteBase/user-1/sticky_notes/notes.json"
    assert put["ContentType"].startswith("application/json")
    assert json.loads(put["Body"].decode()) == document
    assert put["Metadata"]["user-id"] == "user-1"


def test_save_notes_creates_object_when_absent(fake_bucket):
    s3_storage.save_notes("user-1", {"main": []})
    assert fake_bucket.puts[0]["Key"] == "NoteBase/user-1/sticky_notes/notes.json"


def test_delete_notes_targets_user_object(fake_bucket):
    s3_storage.delete_notes("user-2")
    assert fake_bucket.deletes[0]["Key"] == "NoteBase/user-2/sticky_notes/notes.json"


def test_create_folder_under_notebase(fake_bucket):
    key = s3_storage.create_folder("shared/exports")
    assert key == "NoteBase/shared/exports/"
    assert fake_bucket.puts[0]["Key"] == key
    assert fake_bucket.puts[0]["Body"] == b""


def test_create_folder_rejects_unsafe_segments(fake_bucket):
    with pytest.raises(s3_storage.S3StorageError):
        s3_storage.create_folder("../escape")


# ── PDFs ─────────────────────────────────────────────────────────────────────

def test_upload_pdf_uses_user_folder(fake_bucket, tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    key = s3_storage.upload_pdf("user-9", "report.pdf", str(pdf))

    assert key == "NoteBase/user-9/pdfs/report.pdf"
    assert fake_bucket.uploads[0]["key"] == "NoteBase/user-9/pdfs/report.pdf"
    assert fake_bucket.uploads[0]["extra"]["ContentType"] == "application/pdf"


def test_download_pdf_missing_returns_false(fake_bucket, tmp_path):
    assert s3_storage.download_pdf_to_path("user-9", "gone.pdf", str(tmp_path / "gone.pdf")) is False


def test_open_pdf_returns_streaming_proxy(fake_bucket):
    fake_bucket.objects["NoteBase/user-9/pdfs/report.pdf"] = {
        "Body": _FakeBody(b"%PDF-1.4 body"),
        "ContentLength": 13,
    }

    stored = s3_storage.open_pdf("user-9", "report.pdf")

    assert stored is not None
    assert stored.content_length == 13
    with stored:
        assert b"".join(stored.iter_chunks(chunk_size=4)) == b"%PDF-1.4 body"


def test_open_pdf_missing_returns_none(fake_bucket):
    assert s3_storage.open_pdf("user-9", "gone.pdf") is None


# ── route integration ───────────────────────────────────────────────────────

def test_upload_mirrors_document_into_user_folder(client, auth_headers, user, monkeypatch, tmp_path):
    import tempfile

    async def fake_validate(file):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", dir=tmp_path)
        tmp.write(b"hello s3")
        tmp.close()
        return tmp.name

    monkeypatch.setattr("app.routes.documents.validate_upload", fake_validate)
    monkeypatch.setattr(
        "app.routes.documents._queue_ingest",
        lambda **kwargs: "celery-1",
    )

    calls = []
    expected_user_id = str(user.id)
    monkeypatch.setattr("app.routes.documents.s3_storage.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.routes.documents.s3_storage.upload_pdf",
        lambda user_id, filename, path: calls.append((user_id, filename, path)) or "key",
    )

    response = client.post(
        "/api/v1/documents/upload",
        headers=auth_headers,
        files={"file": ("notes.txt", io.BytesIO(b"hello s3"), "text/plain")},
        data={"chunk_size": "1000", "chunk_overlap": "200"},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    # Storage location is server-side only — never returned to the browser.
    assert "storage_backend" not in body
    assert "s3_key" not in body
    assert len(calls) == 1
    assert calls[0][0] == expected_user_id
    assert calls[0][1].endswith(".txt")
    # Upload reads a short-lived temp path, then deletes it (no UPLOAD_DIR copy).
    assert isinstance(calls[0][2], str) and calls[0][2].endswith(".txt")
    assert not (tmp_path / expected_user_id / calls[0][1]).exists()


def test_serve_pdf_streams_from_s3(client, auth_headers, user, db_session, monkeypatch):
    doc = Document(
        user_id=user.id,
        filename="stored.pdf",
        original_name="Quarterly Report.pdf",
        file_size=13,
        status="ready",
        s3_key="NoteBase/user/pdfs/stored.pdf",
        storage_backend="s3",
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    monkeypatch.setattr("app.routes.documents.s3_storage.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.routes.documents.s3_storage.open_pdf",
        lambda user_id, filename, key=None: _FakeStoredPdf(b"%PDF-1.4 test"),
    )

    response = client.get(f"/api/v1/documents/{doc.id}/pdf", headers=auth_headers)

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 test"
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-length"] == "13"


def test_retry_remirrors_local_file_to_s3(
    client, auth_headers, user, db_session, monkeypatch, tmp_path
):
    monkeypatch.setattr("app.routes.documents.settings.UPLOAD_DIR", str(tmp_path))

    user_id = str(user.id)
    user_dir = tmp_path / user_id
    user_dir.mkdir(parents=True)
    (user_dir / "retry-me.pdf").write_bytes(b"%PDF-1.4 retry")

    doc = Document(
        user_id=user.id,
        filename="retry-me.pdf",
        original_name="Retry Me.pdf",
        file_size=15,
        status="failed",
        error_message="previous run failed",
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    doc_id = str(doc.id)

    calls = []
    monkeypatch.setattr("app.routes.documents.s3_storage.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.routes.documents.s3_storage.upload_pdf",
        lambda uid, filename, path: calls.append((uid, filename, path)) or "key",
    )
    monkeypatch.setattr(
        "app.routes.documents._queue_ingest",
        lambda **kwargs: "celery-retry",
    )

    response = client.post(f"/api/v1/documents/{doc_id}/retry", headers=auth_headers)

    assert response.status_code == 200, response.text
    assert calls == [(user_id, "retry-me.pdf", str(user_dir / "retry-me.pdf"))]
    body = response.json()
    assert "s3_key" not in body
    assert "storage_backend" not in body
    db_session.refresh(doc)
    assert doc.s3_key == "key"
    assert doc.storage_backend == "s3"
    assert not (user_dir / "retry-me.pdf").exists()

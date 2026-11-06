"""Live E2E: sticky notes CRUD + PDF S3 upload/view/download/persistence.

Uses the real FastAPI app, real DB, and real AWS S3 (gen-research/NoteBase/).
Never prints credentials.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfWriter

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env", override=True)

# Ensure S3 settings are visible before Settings is cached.
os.environ.setdefault("AWS_S3_BUCKET", os.getenv("AWS_S3_BUCKET") or "gen-research")
os.environ.setdefault("S3_BASE_PREFIX", os.getenv("S3_BASE_PREFIX") or "NoteBase")

from app.config import get_settings

get_settings.cache_clear()
settings = get_settings()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import SessionLocal, init_db
from app.models import Document, User
from app.routes.auth import router as auth_router
from app.routes.documents import router as documents_router
from app.routes.notes import router as notes_router
from app.services import s3_storage


def _build_app() -> FastAPI:
    """Slim ASGI app with the real auth/documents/notes routers (no heavy lifespan)."""
    application = FastAPI(title="NoteBase S3 E2E")
    application.include_router(auth_router, prefix="/api/v1")
    application.include_router(documents_router, prefix="/api/v1")
    application.include_router(notes_router, prefix="/api/v1")
    return application


def _pdf_bytes(label: str = "e2e") -> bytes:
    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": label})
    writer.write(buf)
    return buf.getvalue()


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def main() -> None:
    init_db()
    s3_storage.reset_client_cache()
    s3_storage.settings = get_settings()

    cfg = s3_storage.validate_configuration(probe=True)
    if not cfg.get("configured") or not cfg.get("reachable"):
        _fail(f"S3 not reachable: {cfg}")
    _ok(f"S3 reachable bucket={cfg['bucket']} prefix={cfg['base_prefix']}")

    client = TestClient(_build_app())
    suffix = uuid.uuid4().hex[:8]
    email = f"e2e.notes.pdf.{suffix}@example.com"
    password = "E2eTestPass!234"
    username = f"e2e_{suffix}"

    reg = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    if reg.status_code not in (200, 201):
        # Dev bypass / already verified paths vary; try login after ensure.
        print(f"register status={reg.status_code} body={reg.text[:200]}")

    # Mark verified for local SQLite when email verification is required.
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            # Some builds auto-create on register failure; create manually.
            from app.auth import hash_password

            user = User(
                username=username,
                email=email,
                hashed_password=hash_password(password),
                is_verified=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            user.is_verified = True
            db.commit()
            db.refresh(user)
        user_id = str(user.id)
    finally:
        db.close()

    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    if login.status_code != 200:
        _fail(f"login failed: {login.status_code} {login.text[:300]}")

    token = login.json().get("access_token") or login.json().get("token")
    if not token:
        _fail(f"no access token in login response: {login.json()}")
    headers = {"Authorization": f"Bearer {token}"}
    _ok("authenticated")

    # ── Sticky notes CRUD ────────────────────────────────────────────────
    empty = client.get("/api/v1/notes", headers=headers)
    if empty.status_code != 200:
        _fail(f"GET notes: {empty.status_code} {empty.text[:200]}")
    _ok(f"notes GET storage={empty.json().get('storage')}")

    note_id = f"note-{suffix}"
    payload = {
        "notes": {
            "format": "genova-sticky-notes",
            "version": 1,
            "activeId": note_id,
            "notes": [
                {
                    "id": note_id,
                    "title": f"PDF Notes {suffix}",
                    "document": {
                        "format": "ckeditor5-model",
                        "version": 1,
                        "main": [
                            {
                                "name": "paragraph",
                                "children": [{"data": f"Associated with PDF run {suffix}"}],
                            }
                        ],
                    },
                    "html": f"<p>Associated with PDF run {suffix}</p>",
                }
            ],
        }
    }
    put = client.put("/api/v1/notes", headers=headers, json=payload)
    if put.status_code != 200:
        _fail(f"PUT notes: {put.status_code} {put.text[:300]}")
    _ok("notes created/saved")

    got = client.get("/api/v1/notes", headers=headers)
    body = got.json()
    if not body.get("exists") or not body.get("notes"):
        _fail(f"notes not retrieved after save: {body}")
    raw = json.dumps(body["notes"])
    if f"Associated with PDF run {suffix}" not in raw:
        _fail("saved note content missing on retrieve")
    _ok("notes retrieved with content")

    # Edit
    payload["notes"]["notes"][0]["html"] = f"<p>Edited sticky note {suffix}</p>"
    payload["notes"]["notes"][0]["document"]["main"][0]["children"][0]["data"] = (
        f"Edited sticky note {suffix}"
    )
    edit = client.put("/api/v1/notes", headers=headers, json=payload)
    if edit.status_code != 200:
        _fail(f"edit notes: {edit.status_code}")
    got2 = client.get("/api/v1/notes", headers=headers).json()
    if f"Edited sticky note {suffix}" not in json.dumps(got2.get("notes")):
        _fail("edited note content not persisted")
    _ok("notes edited and persisted")

    # ── PDF upload directly to S3 ────────────────────────────────────────
    pdf_name = f"e2e-s3-{suffix}.pdf"
    pdf_bytes = _pdf_bytes(pdf_name)
    up = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": (pdf_name, pdf_bytes, "application/pdf")},
        data={"chunk_size": "1000", "chunk_overlap": "200"},
    )
    if up.status_code != 202:
        _fail(f"PDF upload: {up.status_code} {up.text[:400]}")
    doc = up.json()
    doc_id = doc["id"]
    s3_key = doc.get("s3_key")
    storage = doc.get("storage_backend")
    if storage != "s3" or not s3_key:
        _fail(f"upload did not store S3 metadata: {doc}")
    if not str(s3_key).startswith("NoteBase/"):
        _fail(f"s3_key missing NoteBase prefix: {s3_key}")
    _ok(f"PDF uploaded id={doc_id} key={s3_key}")

    # No local permanent file
    local_path = Path(settings.UPLOAD_DIR) / user_id / Path(s3_key).name
    if local_path.exists():
        _fail(f"PDF still present locally at {local_path}")
    _ok("no permanent local PDF copy")

    # Verify object exists in S3
    client_s3 = s3_storage._client()
    head = client_s3.head_object(Bucket=s3_storage._bucket(), Key=s3_key)
    if int(head.get("ContentLength") or 0) < 10:
        _fail(f"S3 object too small: {head}")
    _ok(f"S3 object verified size={head.get('ContentLength')}")

    # DB metadata
    db = SessionLocal()
    try:
        row = db.query(Document).filter(Document.id == doc_id).first()
        if row is None or row.s3_key != s3_key or row.storage_backend != "s3":
            _fail(f"DB metadata mismatch: s3_key={getattr(row,'s3_key',None)} backend={getattr(row,'storage_backend',None)}")
        filename = row.filename
    finally:
        db.close()
    _ok("DB stores s3_key + metadata")

    # View / download via API
    view = client.get(f"/api/v1/documents/{doc_id}/pdf", headers=headers)
    if view.status_code != 200:
        _fail(f"PDF view: {view.status_code} {view.text[:200]}")
    if not view.content.startswith(b"%PDF"):
        _fail("PDF view body is not a PDF")
    if len(view.content) < 10:
        _fail("PDF view body too small")
    _ok(f"PDF view/download OK bytes={len(view.content)}")

    # Persistence after "refresh" (re-list + re-fetch)
    listed = client.get("/api/v1/documents", headers=headers)
    if listed.status_code != 200:
        _fail(f"list documents: {listed.status_code}")
    items = listed.json().get("items") or listed.json().get("documents") or listed.json()
    if isinstance(items, dict):
        items = items.get("items") or []
    ids = {str(i.get("id")) for i in items if isinstance(i, dict)}
    if doc_id not in ids:
        _fail(f"uploaded doc missing after refresh/list; ids={list(ids)[:5]}")
    again = client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    if again.status_code != 200 or again.json().get("s3_key") != s3_key:
        _fail(f"document metadata lost after refresh: {again.text[:200]}")
    view2 = client.get(f"/api/v1/documents/{doc_id}/pdf", headers=headers)
    if view2.status_code != 200 or not view2.content.startswith(b"%PDF"):
        _fail("PDF not retrievable after refresh")
    _ok("PDF persists after refresh (list+get+view)")

    # Notes remain associated at user workspace level (same user, same PDF run marker)
    notes_after = client.get("/api/v1/notes", headers=headers).json()
    notes_blob = json.dumps(notes_after.get("notes"))
    if f"Edited sticky note {suffix}" not in notes_blob:
        _fail("sticky notes lost after PDF upload")
    if note_id not in notes_blob:
        _fail("sticky note id lost after PDF upload")
    _ok("sticky notes remain for the same user after PDF upload")

    # Delete notes
    deleted = client.delete("/api/v1/notes", headers=headers)
    if deleted.status_code != 200:
        _fail(f"DELETE notes: {deleted.status_code}")
    after_del = client.get("/api/v1/notes", headers=headers).json()
    if after_del.get("exists"):
        _fail("notes still exist after delete")
    _ok("notes deleted")

    # Soft-delete document cleanup of S3 object is not required (recycle bin),
    # but remove the smoke object to avoid clutter.
    try:
        s3_storage.delete_pdf(user_id, filename)
        _ok("cleaned S3 smoke PDF")
    except Exception as exc:
        print(f"WARN: S3 cleanup: {exc}")

    print("E2E_NOTES_PDF_S3_OK")


if __name__ == "__main__":
    main()

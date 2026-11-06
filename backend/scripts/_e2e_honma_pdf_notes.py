"""Live E2E: Honma PDF + sticky notes against gen-research/{user_id}/…

Verifies physical S3 objects, DB metadata sync, notes CRUD with full CKEditor
structure + PDF association, and persistence across re-login.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
PDF_PATH = ROOT / "Honma - The fountain of Gold (The Three Monkey Record of Money).pdf"

sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env", override=True)
os.environ["S3_BASE_PREFIX"] = "NoteBase"
os.environ.setdefault("AWS_S3_BUCKET", "gen-research")

from app.config import get_settings

get_settings.cache_clear()

from app.auth import hash_password
from app.database import SessionLocal, init_db
from app.models import Document, User
from app.routes.auth import router as auth_router
from app.routes.notes import router as notes_router
from app.services import s3_storage


def _ok(msg: str) -> None:
    print(f"OK: {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def _notes_app() -> FastAPI:
    application = FastAPI(title="Honma Notes E2E")
    application.include_router(auth_router, prefix="/api/v1")
    application.include_router(notes_router, prefix="/api/v1")
    return application


def main() -> None:
    if not PDF_PATH.is_file():
        _fail(f"PDF not found: {PDF_PATH}")

    init_db()
    s3_storage.reset_client_cache()
    s3_storage.settings = get_settings()
    cfg = s3_storage.validate_configuration(probe=True)
    if not cfg.get("reachable"):
        _fail(f"S3 unreachable: {cfg}")
    if (cfg.get("base_prefix") or "").rstrip("/") != "NoteBase":
        _fail(f"expected NoteBase prefix, got {cfg.get('base_prefix')!r}")
    _ok(f"S3 reachable bucket={cfg['bucket']} layout=NoteBase/{{user_id}}/…")

    client = TestClient(_notes_app())
    suffix = uuid.uuid4().hex[:8]
    email = f"honma.e2e.{suffix}@example.com"
    password = "E2eTestPass!234"
    username = f"honma_{suffix}"
    original_name = PDF_PATH.name
    pdf_bytes = PDF_PATH.read_bytes()
    stored_filename = f"{uuid.uuid4().hex}.pdf"

    db = SessionLocal()
    try:
        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            is_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        user_id = str(user.id)

        # Upload PDF directly to the required layout (same helper the upload route uses).
        tmp = BACKEND / "data" / f"_honma_{suffix}.pdf"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(pdf_bytes)
        s3_key = s3_storage.upload_pdf(user_id, stored_filename, str(tmp))
        tmp.unlink(missing_ok=True)

        expected_prefix = f"NoteBase/{user_id}/pdfs/"
        if not s3_key.startswith(expected_prefix):
            _fail(f"bad S3 key layout: {s3_key}")

        doc = Document(
            user_id=user.id,
            filename=stored_filename,
            original_name=original_name,
            file_size=len(pdf_bytes),
            status="ready",
            s3_key=s3_key,
            storage_backend="s3",
            page_count=1,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = str(doc.id)
    finally:
        db.close()

    _ok(f"PDF uploaded user_id={user_id}")
    _ok(f"S3 key={s3_key}")

    head = s3_storage._client().head_object(Bucket=s3_storage._bucket(), Key=s3_key)
    size = int(head.get("ContentLength") or 0)
    if size != len(pdf_bytes):
        _fail(f"S3 size mismatch local={len(pdf_bytes)} remote={size}")
    _ok(f"S3 object exists in pdfs/ size={size}")

    listed = s3_storage.list_prefix(f"{user_id}/pdfs")
    keys = [i["key"] for i in listed if i.get("type") == "file"]
    if s3_key not in keys:
        _fail(f"PDF not listed under NoteBase/{user_id}/pdfs/; found={keys[:5]}")
    _ok(f"listed under NoteBase/{user_id}/pdfs/")

    # Retrieve bytes via S3 helper (same path serve_pdf uses).
    stream = s3_storage.open_pdf(user_id, stored_filename, key=s3_key)
    if stream is None:
        _fail("open_pdf returned None")
    with stream:
        retrieved = b"".join(stream.iter_chunks())
    if retrieved != pdf_bytes:
        _fail("retrieved PDF bytes differ from source file")
    _ok("PDF retrieval matches source file")

    local = Path(get_settings().UPLOAD_DIR) / user_id / stored_filename
    if local.exists():
        _fail(f"unexpected local PDF: {local}")
    _ok("no permanent local PDF copy")

    # Auth for notes API
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if login.status_code != 200:
        _fail(f"login failed: {login.status_code} {login.text[:200]}")
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    _ok("authenticated")

    note_id = f"honma-note-{suffix}"
    ck_document = {
        "format": "ckeditor5-model",
        "version": 1,
        "main": [
            {
                "name": "heading",
                "attributes": {"headingLevel": 2},
                "children": [{"data": "Honma — Fountain of Gold"}],
            },
            {
                "name": "paragraph",
                "children": [
                    {"data": "Key idea: ", "attributes": {"bold": True}},
                    {
                        "data": "patience compounds like interest.",
                        "attributes": {"italic": True},
                    },
                ],
            },
            {
                "name": "paragraph",
                "children": [
                    {
                        "data": "Linked to source PDF",
                        "attributes": {"linkHref": f"genova://documents/{doc_id}"},
                    }
                ],
            },
            {
                "name": "listItem",
                "attributes": {
                    "listType": "bulleted",
                    "listIndent": 0,
                    "listItemId": "li1",
                },
                "children": [{"data": "Three monkeys: see / hear / speak"}],
            },
        ],
    }
    html = (
        "<h2>Honma — Fountain of Gold</h2>"
        "<p><strong>Key idea: </strong><i>patience compounds like interest.</i></p>"
        f'<p><a href="genova://documents/{doc_id}">Linked to source PDF</a></p>'
        "<ul><li>Three monkeys: see / hear / speak</li></ul>"
    )
    envelope = {
        "format": "genova-sticky-notes",
        "version": 1,
        "activeId": note_id,
        "notes": [
            {
                "id": note_id,
                "title": "Honma notes",
                "document": ck_document,
                "html": html,
                "documentId": doc_id,
                "documentName": original_name,
            }
        ],
    }

    put = client.put("/api/v1/notes", headers=headers, json={"notes": envelope})
    if put.status_code != 200:
        _fail(f"notes save failed: {put.status_code} {put.text[:300]}")
    notes_key = s3_storage.notes_key(user_id)
    if notes_key != f"NoteBase/{user_id}/sticky_notes/notes.json":
        _fail(f"unexpected notes key: {notes_key}")
    _ok(f"notes saved key={notes_key}")

    raw_obj = s3_storage._client().get_object(
        Bucket=s3_storage._bucket(), Key=notes_key
    )
    stored_notes = json.loads(raw_obj["Body"].read().decode("utf-8"))
    note0 = stored_notes["notes"][0]
    if note0.get("documentId") != doc_id or note0.get("documentName") != original_name:
        _fail("notes.json missing PDF association")
    if json.dumps(note0.get("document"), sort_keys=True) != json.dumps(
        ck_document, sort_keys=True
    ):
        _fail("CKEditor structure altered in notes.json")
    _ok("notes.json keeps full CKEditor structure + PDF association")

    # Refresh (GET)
    got = client.get("/api/v1/notes", headers=headers).json()
    rel = (got.get("notes") or {}).get("notes") or []
    if not rel or rel[0].get("documentId") != doc_id:
        _fail("reload lost PDF association")
    if json.dumps(rel[0].get("document"), sort_keys=True) != json.dumps(
        ck_document, sort_keys=True
    ):
        _fail("reload changed CKEditor document")
    _ok("notes reload reconstructs exact structure")

    # Edit
    envelope["notes"][0]["document"]["main"].append(
        {"name": "paragraph", "children": [{"data": "Edited after refresh."}]}
    )
    envelope["notes"][0]["html"] = html + "<p>Edited after refresh.</p>"
    if client.put("/api/v1/notes", headers=headers, json={"notes": envelope}).status_code != 200:
        _fail("notes edit failed")
    got2 = client.get("/api/v1/notes", headers=headers).json()
    if "Edited after refresh." not in json.dumps(got2.get("notes")):
        _fail("edit not persisted")
    _ok("notes edit persisted")

    # Logout / login + reopen PDF association
    login2 = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    headers2 = {"Authorization": f"Bearer {login2.json()['access_token']}"}
    notes3 = client.get("/api/v1/notes", headers=headers2).json()
    n3 = (notes3.get("notes") or {}).get("notes") or []
    if not n3 or n3[0].get("documentId") != doc_id:
        _fail("association lost after re-login")
    if "Edited after refresh." not in json.dumps(n3[0].get("document")):
        _fail("content lost after re-login")

    db = SessionLocal()
    try:
        row = db.query(Document).filter(Document.id == doc_id).first()
        if not row or row.s3_key != s3_key or row.storage_backend != "s3":
            _fail("DB out of sync after re-login")
    finally:
        db.close()

    stream2 = s3_storage.open_pdf(user_id, stored_filename, key=s3_key)
    with stream2:
        again = b"".join(stream2.iter_chunks())
    if again != pdf_bytes:
        _fail("PDF reopen after login mismatch")
    _ok("persistence after logout/login + PDF reopen")

    # Keep objects in S3 by default so they can be inspected in AWS CLI/Console.
    # Set HONMA_E2E_CLEANUP=1 to delete notes + PDF after the run.
    if os.getenv("HONMA_E2E_CLEANUP", "").strip() in {"1", "true", "True"}:
        if client.delete("/api/v1/notes", headers=headers2).status_code != 200:
            _fail("notes delete failed")
        if client.get("/api/v1/notes", headers=headers2).json().get("exists"):
            _fail("notes still exist after delete")
        _ok("notes deleted")
        try:
            s3_storage.delete_pdf(user_id, stored_filename, key=s3_key)
            _ok("cleaned S3 PDF")
        except Exception as exc:
            print(f"WARN: cleanup {exc}", flush=True)
    else:
        _ok("left PDF + notes.json in S3 for inspection (set HONMA_E2E_CLEANUP=1 to remove)")

    print("E2E_HONMA_PDF_NOTES_OK", flush=True)
    print(f"PATHS pdf=s3://gen-research/{s3_key}", flush=True)
    print(f"PATHS notes=s3://gen-research/{notes_key}", flush=True)
    print(f"USER_ID={user_id}", flush=True)


if __name__ == "__main__":
    main()

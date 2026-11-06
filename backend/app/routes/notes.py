"""Per-user sticky notes persisted as CKEditor JSON in AWS S3.

Every request resolves the user from the JWT, so the S3 prefix is always
``NoteBase/{authenticated_user_id}/sticky_notes/notes.json`` — a caller can
never read or write another user's notes.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.exceptions import AppException, ExternalServiceException, ValidationException
from app.models import User
from app.services import s3_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notes", tags=["Sticky Notes"])

# A notes document is text; 2 MB is far above any realistic editor payload and
# keeps a single user from writing an unbounded object.
MAX_NOTES_BYTES = 2 * 1024 * 1024
MAX_EXPORT_HTML_BYTES = 2 * 1024 * 1024

# Multi-note envelopes written by the current client:
#   {"format": "genova-sticky-notes", "version": 1, "activeId": "…",
#    "notes": [{"id": "…", "document": {...CKEditor model...}, "html": "…"}]}
# Older clients stored a single CKEditor document; both are accepted.
NOTES_COLLECTION_FORMAT = "genova-sticky-notes"


class NotesPayload(BaseModel):
    notes: Dict[str, Any] = Field(..., description="CKEditor 5 JSON document")


class NotesPdfExportPayload(BaseModel):
    html: str = Field(..., description="CKEditor HTML snapshot to typeset")
    title: Optional[str] = Field(None, description="Optional document title")
    note_id: Optional[str] = Field(None, description="Optional note identifier")


def _require_storage() -> None:
    if not s3_storage.is_configured():
        raise AppException(
            "S3_NOT_CONFIGURED",
            "Sticky-note storage is not configured. Set S3_BUCKET and AWS credentials.",
            503,
        )


def _validate_collection(document: Dict[str, Any]) -> None:
    """Validate the multi-note envelope (each note is an independent document)."""
    notes = document.get("notes")
    if not isinstance(notes, list):
        raise ValidationException("Sticky notes collection 'notes' must be a list")

    active_id = document.get("activeId")
    if active_id is not None and not isinstance(active_id, str):
        raise ValidationException("Sticky notes collection 'activeId' must be a string")

    for index, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ValidationException(f"Sticky note {index + 1} must be an object")

        note_id = note.get("id")
        if note_id is not None and not isinstance(note_id, str):
            raise ValidationException(f"Sticky note {index + 1} 'id' must be a string")

        note_document = note.get("document")
        if note_document is not None and not isinstance(note_document, dict):
            raise ValidationException(
                f"Sticky note {index + 1} 'document' must be a JSON object"
            )

        note_html = note.get("html")
        if note_html is not None and not isinstance(note_html, str):
            raise ValidationException(
                f"Sticky note {index + 1} 'html' must be a string"
            )

        document_id = note.get("documentId")
        if document_id is not None and not isinstance(document_id, str):
            raise ValidationException(
                f"Sticky note {index + 1} 'documentId' must be a string"
            )
        document_name = note.get("documentName")
        if document_name is not None and not isinstance(document_name, str):
            raise ValidationException(
                f"Sticky note {index + 1} 'documentName' must be a string"
            )


def _validate_document(document: Dict[str, Any]) -> None:
    """Reject payloads that cannot be a CKEditor data document."""
    if not isinstance(document, dict):
        raise ValidationException("Sticky notes must be a JSON object")

    if document.get("format") == NOTES_COLLECTION_FORMAT:
        _validate_collection(document)
    else:
        if "blocks" in document and not isinstance(document["blocks"], list):
            raise ValidationException("Sticky notes 'blocks' must be a list")
        if "main" in document and not isinstance(document["main"], (list, dict, str)):
            raise ValidationException(
                "Sticky notes 'main' must be a list of model nodes or an HTML string"
            )
        if "content" in document and not isinstance(document["content"], list):
            raise ValidationException("Sticky notes 'content' must be a list")

    try:
        encoded = json.dumps(document, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationException("Sticky notes must be JSON-serialisable") from exc

    if len(encoded) > MAX_NOTES_BYTES:
        raise ValidationException(
            f"Sticky notes exceed the {MAX_NOTES_BYTES // (1024 * 1024)} MB limit"
        )


@router.get(
    "",
    summary="Load the authenticated user's sticky notes",
    description=(
        "Returns the CKEditor JSON document stored at "
        "`{user_id}/sticky_notes/notes.json`. Missing files are reported as "
        "`notes: null` so the client can start with an empty editor."
    ),
)
def get_notes(current_user: User = Depends(get_current_user)):
    """Fetch this user's saved notes (never another user's)."""
    if not s3_storage.is_configured():
        return {
            "notes": None,
            "exists": False,
            "updated_at": None,
            "storage": "unavailable",
        }

    try:
        document, updated_at = s3_storage.load_notes(str(current_user.id))
    except s3_storage.S3StorageError as exc:
        logger.error("Sticky-note load failed for user %s: %s", current_user.id, exc)
        raise ExternalServiceException(
            "s3", "Could not load your sticky notes. Please try again."
        ) from exc

    return {
        "notes": document,
        "exists": document is not None,
        "updated_at": updated_at,
        "storage": "s3",
    }


@router.put(
    "",
    summary="Create or update the authenticated user's sticky notes",
    description=(
        "Writes the CKEditor JSON document to the user's `sticky_notes/notes.json` "
        "object, creating it on the first save."
    ),
)
def put_notes(payload: NotesPayload, current_user: User = Depends(get_current_user)):
    """Persist this user's notes as structured JSON."""
    _require_storage()
    _validate_document(payload.notes)

    try:
        saved_at = s3_storage.save_notes(str(current_user.id), payload.notes)
    except s3_storage.S3StorageError as exc:
        logger.error("Sticky-note save failed for user %s: %s", current_user.id, exc)
        raise ExternalServiceException(
            "s3", "Could not save your sticky notes. Your changes are kept locally."
        ) from exc

    return {"saved": True, "updated_at": saved_at, "storage": "s3"}


@router.delete(
    "",
    summary="Delete the authenticated user's sticky notes",
)
def delete_notes(current_user: User = Depends(get_current_user)):
    """Remove this user's notes object (idempotent)."""
    _require_storage()

    try:
        s3_storage.delete_notes(str(current_user.id))
    except s3_storage.S3StorageError as exc:
        logger.error("Sticky-note delete failed for user %s: %s", current_user.id, exc)
        raise ExternalServiceException(
            "s3", "Could not clear your sticky notes. Please try again."
        ) from exc

    return {"deleted": True, "storage": "s3"}


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s\-]+", "", (name or "notes").strip(), flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._") or "notes"
    return cleaned[:80]


@router.post(
    "/export/pdf",
    summary="Export CKEditor note HTML as a professionally typeset PDF",
    description=(
        "Accepts the editor HTML snapshot and returns a Genova publishing-system "
        "PDF (cover, TOC when applicable, editorial typography). Does not read or "
        "write S3 — the client supplies the content to export."
    ),
)
def export_notes_pdf(
    payload: NotesPdfExportPayload,
    current_user: User = Depends(get_current_user),
):
    """Typeset notes HTML with the shared PDF publishing design system."""
    html = payload.html or ""
    if not html.strip():
        raise ValidationException("Notes are empty — nothing to export")
    if len(html.encode("utf-8")) > MAX_EXPORT_HTML_BYTES:
        raise ValidationException(
            f"Export HTML exceeds the {MAX_EXPORT_HTML_BYTES // (1024 * 1024)} MB limit"
        )

    try:
        from app.publishing.notes import format_notes_pdf

        pdf_bytes = format_notes_pdf(
            html,
            title=payload.title,
            note_id=payload.note_id,
        )
    except Exception as exc:
        logger.exception(
            "Notes PDF export failed for user %s: %s", current_user.id, exc
        )
        raise AppException(
            "NOTES_PDF_EXPORT_FAILED",
            "Could not generate the notes PDF. Please try again.",
            500,
        ) from exc

    filename = f"{_safe_filename(payload.title or 'genova-notes')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )

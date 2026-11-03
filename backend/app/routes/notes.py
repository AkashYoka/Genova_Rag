"""Per-user sticky notes persisted as CKEditor JSON in AWS S3.

Every request resolves the user from the JWT, so the S3 prefix is always
``{authenticated_user_id}/sticky_notes/notes.json`` — a caller can never read
or write another user's notes.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends
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


class NotesPayload(BaseModel):
    notes: Dict[str, Any] = Field(..., description="CKEditor 5 JSON document")


def _require_storage() -> None:
    if not s3_storage.is_configured():
        raise AppException(
            "S3_NOT_CONFIGURED",
            "Sticky-note storage is not configured. Set S3_BUCKET and AWS credentials.",
            503,
        )


def _validate_document(document: Dict[str, Any]) -> None:
    """Reject payloads that cannot be a CKEditor data document."""
    if not isinstance(document, dict):
        raise ValidationException("Sticky notes must be a JSON object")

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

"""Google Drive folder connection APIs — Settings UI + async RAG sync."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.exceptions import AppException, NotFoundException, ValidationException
from app.models import Document, DriveConnection, User
from app.schemas import (
    DriveFolderCreateRequest,
    DriveFolderListResponse,
    DriveFolderResponse,
    DriveSyncAcceptedResponse,
    DriveSyncStatusResponse,
    GoogleDriveStatusResponse,
)
from app.services import drive_sync
from app.services.drive_public import (
    DriveInvalidFolder,
    DriveNetworkError,
    DriveNotFound,
    DrivePermissionDenied,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drive", tags=["Google Drive"])


def _folder_url(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


def _document_count(db: Session, user_id, folder_id: str) -> int:
    return (
        db.query(Document)
        .filter(
            Document.user_id == user_id,
            Document.drive_folder_id == folder_id,
            Document.is_deleted.is_(False),
        )
        .count()
    )


def _serialize_connection(db: Session, conn: DriveConnection) -> DriveFolderResponse:
    return DriveFolderResponse(
        id=str(conn.id),
        folder_id=conn.folder_id,
        folder_name=conn.folder_name,
        folder_url=conn.folder_url or _folder_url(conn.folder_id),
        access_mode=(conn.access_mode or "public"),
        enabled=bool(conn.enabled),
        sync_status=conn.sync_status or "idle",
        task_id=conn.celery_task_id,
        last_error=conn.last_error,
        last_synced_at=conn.last_synced_at,
        created_at=conn.created_at,
        document_count=_document_count(db, conn.user_id, conn.folder_id),
    )


def _validate_folder_access(folder_id: str, user: User) -> tuple[str, dict]:
    """API-only access check — never lists or downloads PDFs."""
    try:
        return drive_sync.resolve_folder_access(folder_id, user)
    except DrivePermissionDenied as exc:
        raise AppException(
            "DRIVE_OAUTH_REQUIRED",
            str(exc) or "Connect Google Drive to access this private folder.",
            403,
        ) from exc
    except DriveNotFound as exc:
        raise ValidationException(str(exc)) from exc
    except DriveInvalidFolder as exc:
        raise ValidationException(str(exc)) from exc
    except DriveNetworkError as exc:
        raise AppException(
            "DRIVE_NETWORK_ERROR",
            str(exc) or "Network error talking to Google Drive. Try again.",
            502,
        ) from exc
    except Exception as exc:
        logger.warning("Drive folder validation failed: %s", exc)
        message = str(exc)
        if "invalid_grant" in message.lower() or "revoked" in message.lower():
            raise AppException(
                "DRIVE_AUTH_REVOKED",
                "Google Drive access was revoked. Disconnect and reconnect Drive.",
                401,
            ) from exc
        raise ValidationException(
            "Could not access that Drive folder. Check the link and sharing permissions."
        ) from exc


@router.get("/status", response_model=GoogleDriveStatusResponse)
def drive_oauth_status(user: User = Depends(get_current_user)):
    """Whether the user has granted Drive OAuth (needed only for private folders)."""
    return GoogleDriveStatusResponse(connected=bool(user.google_refresh_token))


@router.get("/folders", response_model=DriveFolderListResponse)
def list_drive_folders(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List Drive folders the user has attached for sync."""
    connections = (
        db.query(DriveConnection)
        .filter(DriveConnection.user_id == user.id)
        .order_by(DriveConnection.created_at.desc())
        .all()
    )
    return DriveFolderListResponse(
        connected=bool(user.google_refresh_token),
        public_access_available=True,
        folders=[_serialize_connection(db, c) for c in connections],
    )


@router.post(
    "/folders",
    response_model=DriveFolderResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def add_drive_folder(
    payload: DriveFolderCreateRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Attach a Google Drive folder and enqueue an initial Celery sync.

    Validates access (public first, then OAuth) without listing/downloading PDFs,
    then returns 202 immediately with ``task_id`` / ``sync_status=queued``.
    """
    try:
        folder_id = drive_sync.extract_folder_id(payload.url)
    except ValueError as exc:
        raise ValidationException(str(exc)) from exc

    existing = (
        db.query(DriveConnection)
        .filter(
            DriveConnection.user_id == user.id,
            DriveConnection.folder_id == folder_id,
        )
        .first()
    )
    if existing:
        raise ValidationException("That Drive folder is already attached")

    access_mode, meta = _validate_folder_access(folder_id, user)
    folder_name: Optional[str] = meta.get("name") if isinstance(meta, dict) else None

    connection = DriveConnection(
        user_id=user.id,
        folder_id=folder_id,
        folder_name=folder_name,
        folder_url=_folder_url(folder_id),
        access_mode=access_mode,
        enabled=True,
        sync_status="idle",
    )
    db.add(connection)
    db.commit()
    db.refresh(connection)

    task_id, _already = drive_sync.enqueue_connection_sync(
        db,
        connection,
        str(user.id),
        background_tasks=background_tasks,
    )
    db.refresh(connection)
    payload_out = _serialize_connection(db, connection)
    payload_out.task_id = task_id or payload_out.task_id
    return payload_out


@router.delete("/folders/{connection_id}", response_model=DriveFolderResponse)
def remove_drive_folder(
    connection_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Detach a folder connection.

    Existing synced documents are left in the library (they remain searchable).
    """
    connection = (
        db.query(DriveConnection)
        .filter(
            DriveConnection.id == connection_id,
            DriveConnection.user_id == user.id,
        )
        .first()
    )
    if not connection:
        raise NotFoundException("Drive folder connection")

    payload = _serialize_connection(db, connection)
    db.delete(connection)
    db.commit()
    payload.enabled = False
    payload.sync_status = "idle"
    return payload


@router.get(
    "/folders/{connection_id}/status",
    response_model=DriveSyncStatusResponse,
)
def get_drive_folder_sync_status(
    connection_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lightweight poll endpoint for Drive sync progress (no Drive I/O)."""
    connection = (
        db.query(DriveConnection)
        .filter(
            DriveConnection.id == connection_id,
            DriveConnection.user_id == user.id,
        )
        .first()
    )
    if not connection:
        raise NotFoundException("Drive folder connection")

    return DriveSyncStatusResponse(
        connection_id=str(connection.id),
        folder_id=connection.folder_id,
        folder_name=connection.folder_name,
        access_mode=(connection.access_mode or "public"),
        sync_status=connection.sync_status or "idle",
        task_id=connection.celery_task_id,
        last_error=connection.last_error,
        last_synced_at=connection.last_synced_at,
        document_count=_document_count(db, connection.user_id, connection.folder_id),
    )


@router.post(
    "/folders/{connection_id}/sync",
    response_model=DriveSyncAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def sync_drive_folder(
    connection_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue a Celery sync for one folder — returns 202 immediately."""
    connection = (
        db.query(DriveConnection)
        .filter(
            DriveConnection.id == connection_id,
            DriveConnection.user_id == user.id,
        )
        .first()
    )
    if not connection:
        raise NotFoundException("Drive folder connection")

    mode = (connection.access_mode or "public").lower()
    if mode == "oauth" and not user.google_refresh_token:
        raise AppException(
            "DRIVE_OAUTH_REQUIRED",
            "Connect Google Drive in Settings before syncing this private folder.",
            403,
        )

    task_id, already = drive_sync.enqueue_connection_sync(
        db,
        connection,
        str(user.id),
        background_tasks=background_tasks,
    )
    db.refresh(connection)
    return DriveSyncAcceptedResponse(
        message=(
            "Drive sync already in progress"
            if already
            else "Drive sync queued"
        ),
        sync_status=connection.sync_status or "queued",
        connection_id=str(connection.id),
        task_id=task_id or connection.celery_task_id,
        task_ids=[task_id] if task_id else [],
        connections=1,
        already_running=already,
        folders=[_serialize_connection(db, connection)],
    )


@router.post(
    "/sync",
    response_model=DriveSyncAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def sync_all_drive_folders(
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue Celery sync jobs for every attached folder — returns 202 immediately."""
    connections = (
        db.query(DriveConnection)
        .filter(
            DriveConnection.user_id == user.id,
            DriveConnection.enabled.is_(True),
        )
        .all()
    )
    if not connections:
        return DriveSyncAcceptedResponse(
            message="No Drive folders attached",
            sync_status="idle",
            connections=0,
        )

    task_ids: list[str] = []
    folders: list[DriveFolderResponse] = []
    newly_queued = 0
    already_running = 0

    for connection in connections:
        mode = (connection.access_mode or "public").lower()
        if mode == "oauth" and not user.google_refresh_token:
            connection.sync_status = "failed"
            connection.last_error = (
                "Connect Google Drive to sync this private folder"
            )
            db.commit()
            folders.append(_serialize_connection(db, connection))
            continue

        task_id, already = drive_sync.enqueue_connection_sync(
            db,
            connection,
            str(user.id),
            background_tasks=background_tasks,
        )
        db.refresh(connection)
        if task_id:
            task_ids.append(task_id)
        if already:
            already_running += 1
        else:
            newly_queued += 1
        folders.append(_serialize_connection(db, connection))

    if newly_queued == 0 and already_running == 0:
        return DriveSyncAcceptedResponse(
            message="Connect Google Drive to sync private folders",
            sync_status="failed",
            connections=len(connections),
            folders=folders,
        )

    all_already = newly_queued == 0 and already_running > 0
    return DriveSyncAcceptedResponse(
        message=(
            "Drive sync already in progress"
            if all_already
            else "Drive sync queued"
        ),
        sync_status="queued",
        task_ids=task_ids,
        task_id=task_ids[0] if task_ids else None,
        connections=len(connections),
        already_running=all_already,
        folders=folders,
    )

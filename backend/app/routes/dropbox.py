"""Dropbox document source APIs — Settings UI, OAuth and RAG sync."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Optional

import jwt
from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import get_settings
from app.database import get_db
from app.exceptions import AppException, NotFoundException, ValidationException
from app.models import (
    DROPBOX_SOURCE_TYPES,
    Document,
    DropboxConnection,
    User,
)
from app.schemas import (
    DropboxAuthUrlResponse,
    DropboxBrowseResponse,
    DropboxConnectionCreateRequest,
    DropboxConnectionListResponse,
    DropboxConnectionResponse,
    DropboxEntryResponse,
    DropboxImportSelectedRequest,
    DropboxStatusResponse,
    DropboxSyncAcceptedResponse,
    DropboxSyncStatusResponse,
)
from app.services import dropbox_client as dbx
from app.services import dropbox_sync
from app.services.dropbox_client import (
    DropboxAuthError,
    DropboxError,
    DropboxInvalidLink,
    DropboxNetworkError,
    DropboxNotFound,
    DropboxNotConfigured,
    DropboxPermissionDenied,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/dropbox", tags=["Dropbox"])
# Alias so DROPBOX_REDIRECT_URI can mirror the existing /auth/<provider>/callback
# convention used by Google Drive.
auth_callback_router = APIRouter(prefix="/auth", tags=["Dropbox"])

DROPBOX_STATE_TOKEN_TYPE = "dropbox_oauth_state"


def _create_state(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": DROPBOX_STATE_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=10),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def _decode_state(state_token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            state_token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.InvalidTokenError:
        return None
    if payload.get("type") != DROPBOX_STATE_TOKEN_TYPE:
        return None
    return payload.get("sub")


def _callback_page(title: str, message: str, success: bool) -> HTMLResponse:
    color = "#16a34a" if success else "#dc2626"
    safe_title = escape(title)
    safe_message = escape(message)
    return HTMLResponse(
        f"""
        <!doctype html>
        <html>
          <head>
            <title>{safe_title}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1" />
          </head>
          <body style="font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5;">
            <h1 style="color: {color};">{safe_title}</h1>
            <p>{safe_message}</p>
            <p>You can close this tab and return to Genova.</p>
          </body>
        </html>
        """,
        status_code=status.HTTP_200_OK if success else status.HTTP_400_BAD_REQUEST,
    )


def _raise_dropbox_error(exc: DropboxError) -> None:
    """Map service errors onto the app's error envelope."""
    if isinstance(exc, DropboxNotConfigured):
        raise AppException("DROPBOX_NOT_CONFIGURED", str(exc), 503) from exc
    if isinstance(exc, DropboxAuthError):
        raise AppException("DROPBOX_OAUTH_REQUIRED", str(exc), 401) from exc
    if isinstance(exc, DropboxPermissionDenied):
        raise AppException("DROPBOX_PERMISSION_DENIED", str(exc), 403) from exc
    if isinstance(exc, DropboxNotFound):
        raise AppException("DROPBOX_NOT_FOUND", str(exc), 404) from exc
    if isinstance(exc, DropboxInvalidLink):
        raise ValidationException(str(exc)) from exc
    if isinstance(exc, DropboxNetworkError):
        raise AppException("DROPBOX_NETWORK_ERROR", str(exc), 502) from exc
    raise AppException("DROPBOX_ERROR", str(exc), 502) from exc


def _document_count(db: Session, user_id, connection: DropboxConnection) -> int:
    return (
        db.query(Document)
        .filter(
            Document.user_id == user_id,
            Document.external_folder_id == connection.remote_folder_id,
            Document.source_type.in_(DROPBOX_SOURCE_TYPES),
            Document.is_deleted.is_(False),
        )
        .count()
    )


def _serialize_connection(
    db: Session, connection: DropboxConnection
) -> DropboxConnectionResponse:
    kind = connection.connection_kind or "oauth"
    return DropboxConnectionResponse(
        id=str(connection.id),
        connection_kind=kind,
        remote_folder_id=connection.remote_folder_id,
        folder_path=connection.folder_path,
        folder_name=connection.folder_name,
        folder_url=connection.folder_url,
        access_mode=kind,
        enabled=bool(connection.enabled),
        sync_status=connection.sync_status or "idle",
        task_id=connection.celery_task_id,
        last_error=connection.last_error,
        last_synced_at=connection.last_synced_at,
        created_at=connection.created_at,
        document_count=_document_count(db, connection.user_id, connection),
    )


def _get_connection(
    db: Session, connection_id: str, user: User
) -> DropboxConnection:
    connection = (
        db.query(DropboxConnection)
        .filter(
            DropboxConnection.id == connection_id,
            DropboxConnection.user_id == user.id,
        )
        .first()
    )
    if not connection:
        raise NotFoundException("Dropbox connection")
    return connection


def _status_payload(user: User) -> DropboxStatusResponse:
    return DropboxStatusResponse(
        connected=bool(user.dropbox_refresh_token),
        configured=dbx.is_configured(),
        account_id=user.dropbox_account_id,
        display_name=user.dropbox_display_name,
        test_shared_link=settings.DROPBOX_TEST_SHARED_LINK or None,
    )


# ── Connection status / OAuth ────────────────────────────────────────────────


@router.get("/status", response_model=DropboxStatusResponse)
def dropbox_status(user: User = Depends(get_current_user)):
    """Whether Dropbox is configured and connected for this user."""
    return _status_payload(user)


@router.get("/connect", response_model=DropboxAuthUrlResponse)
def connect_dropbox(user: User = Depends(get_current_user)):
    """Create a Dropbox OAuth consent URL with offline (refresh) access."""
    try:
        auth_url = dbx.build_authorize_url(_create_state(user.id))
    except DropboxNotConfigured as exc:
        raise AppException("DROPBOX_NOT_CONFIGURED", str(exc), 503) from exc
    return DropboxAuthUrlResponse(auth_url=auth_url)


@router.get("/callback", include_in_schema=False)
@auth_callback_router.get("/dropbox/callback", include_in_schema=False)
def dropbox_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Handle Dropbox's OAuth callback and store encrypted tokens."""
    if error:
        return _callback_page("Dropbox not connected", error, False)
    if not code or not state:
        return _callback_page(
            "Dropbox not connected",
            "Missing Dropbox OAuth callback data.",
            False,
        )

    user_id = _decode_state(state)
    if not user_id:
        return _callback_page(
            "Dropbox not connected",
            "Invalid or expired Dropbox OAuth state.",
            False,
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return _callback_page("Dropbox not connected", "User not found.", False)

    try:
        tokens = dbx.exchange_code(code)
    except DropboxError as exc:
        logger.warning("Dropbox OAuth exchange failed: %s", exc)
        return _callback_page("Dropbox not connected", str(exc), False)

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return _callback_page(
            "Dropbox not connected",
            "Dropbox did not return a refresh token. Remove Genova from your "
            "Dropbox connected apps and try again.",
            False,
        )

    access_token = tokens.get("access_token")
    expires_in = int(tokens.get("expires_in") or 14400)
    account_id = tokens.get("account_id") or tokens.get("uid")

    display_name = None
    if access_token:
        try:
            account = dbx.DropboxClient(access_token).get_current_account()
            account_id = account.get("account_id") or account_id
            display_name = account.get("display_name") or account.get("email")
        except DropboxError as exc:
            logger.info("Could not fetch Dropbox account profile: %s", exc)

    user.dropbox_refresh_token = refresh_token
    user.dropbox_access_token = access_token
    user.dropbox_token_expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=expires_in
    )
    user.dropbox_account_id = account_id
    user.dropbox_display_name = display_name

    # Reconnecting clears the "disconnected" failure state on OAuth folders.
    db.query(DropboxConnection).filter(
        DropboxConnection.user_id == user.id,
        DropboxConnection.connection_kind == "oauth",
        DropboxConnection.sync_status == "failed",
    ).update(
        {"sync_status": "idle", "last_error": None},
        synchronize_session=False,
    )
    db.commit()

    return _callback_page(
        "Dropbox connected",
        "Read-only Dropbox access was granted successfully.",
        True,
    )


@router.delete("/disconnect", response_model=DropboxStatusResponse)
def disconnect_dropbox(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Forget the stored Dropbox tokens; link connections stay usable."""
    user.dropbox_refresh_token = None
    user.dropbox_access_token = None
    user.dropbox_token_expires_at = None
    user.dropbox_account_id = None
    user.dropbox_display_name = None

    db.query(DropboxConnection).filter(
        DropboxConnection.user_id == user.id,
        DropboxConnection.connection_kind == "oauth",
    ).update(
        {
            "sync_status": "failed",
            "last_error": "Dropbox disconnected — reconnect to sync",
            "updated_at": datetime.now(timezone.utc),
        },
        synchronize_session=False,
    )
    db.commit()
    db.refresh(user)
    return _status_payload(user)


# ── Browsing / attaching ─────────────────────────────────────────────────────


@router.get("/folders", response_model=DropboxBrowseResponse)
def browse_dropbox(
    path: str = Query(default="", max_length=1024),
    connection_id: Optional[str] = Query(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Browse Dropbox folders/PDFs and annotate imported documents."""
    connection: Optional[DropboxConnection] = None
    if connection_id:
        connection = _get_connection(db, connection_id, user)
    elif not user.dropbox_refresh_token:
        raise AppException(
            "DROPBOX_OAUTH_REQUIRED",
            "Connect Dropbox to browse folders.",
            403,
        )

    try:
        entries = dropbox_sync.browse_folder(
            db, user, path=path, connection=connection
        )
    except DropboxError as exc:
        _raise_dropbox_error(exc)

    target_path = path or (
        connection.folder_path if connection is not None else ""
    ) or ""
    return DropboxBrowseResponse(
        connected=bool(user.dropbox_refresh_token),
        connection_id=str(connection.id) if connection else None,
        path=target_path,
        entries=[DropboxEntryResponse(**entry) for entry in entries],
    )


@router.post(
    "/connections",
    response_model=DropboxConnectionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def add_dropbox_connection(
    payload: DropboxConnectionCreateRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Attach an OAuth folder or a public shared link and queue ingestion."""
    if payload.shared_link:
        try:
            normalized = dbx.normalize_shared_link(payload.shared_link)
            meta = dropbox_sync.describe_shared_link(db, user, normalized)
        except DropboxInvalidLink as exc:
            raise ValidationException(str(exc)) from exc
        except DropboxError as exc:
            _raise_dropbox_error(exc)

        kind = "link"
        shared_link: Optional[str] = normalized
        folder_path = ""
    else:
        if not user.dropbox_refresh_token:
            raise AppException(
                "DROPBOX_OAUTH_REQUIRED",
                "Connect Dropbox before importing a folder.",
                403,
            )
        try:
            meta = dropbox_sync.describe_folder(db, user, payload.folder_path)
        except DropboxError as exc:
            _raise_dropbox_error(exc)
        kind = "oauth"
        shared_link = None
        folder_path = meta.get("folder_path") or ""

    existing = (
        db.query(DropboxConnection)
        .filter(
            DropboxConnection.user_id == user.id,
            DropboxConnection.connection_kind == kind,
            DropboxConnection.remote_folder_id == meta["remote_folder_id"],
        )
        .first()
    )
    if existing:
        raise ValidationException("That Dropbox source is already attached")

    connection = DropboxConnection(
        user_id=user.id,
        connection_kind=kind,
        remote_folder_id=meta["remote_folder_id"],
        folder_path=folder_path,
        shared_link=shared_link,
        folder_name=meta.get("folder_name"),
        folder_url=meta.get("folder_url"),
        enabled=True,
        sync_status="idle",
        pending_file_ids=(
            json.dumps(sorted(set(payload.file_ids))) if payload.file_ids else None
        ),
    )
    db.add(connection)
    db.commit()
    db.refresh(connection)

    task_id, _already = dropbox_sync.enqueue_connection_sync(
        db,
        connection,
        str(user.id),
        background_tasks=background_tasks,
    )
    db.refresh(connection)
    response = _serialize_connection(db, connection)
    response.task_id = task_id or response.task_id
    return response


@router.get("/connections", response_model=DropboxConnectionListResponse)
def list_dropbox_connections(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    connections = (
        db.query(DropboxConnection)
        .filter(DropboxConnection.user_id == user.id)
        .order_by(DropboxConnection.created_at.desc())
        .all()
    )
    return DropboxConnectionListResponse(
        connected=bool(user.dropbox_refresh_token),
        configured=dbx.is_configured(),
        connections=[_serialize_connection(db, c) for c in connections],
    )


@router.delete(
    "/connections/{connection_id}",
    response_model=DropboxConnectionResponse,
)
def remove_dropbox_connection(
    connection_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Detach a Dropbox source; already-imported documents stay searchable."""
    connection = _get_connection(db, connection_id, user)
    payload = _serialize_connection(db, connection)
    db.delete(connection)
    db.commit()
    payload.enabled = False
    payload.sync_status = "idle"
    return payload


@router.get(
    "/connections/{connection_id}/status",
    response_model=DropboxSyncStatusResponse,
)
def dropbox_connection_status(
    connection_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lightweight poll endpoint for ingestion progress (no Dropbox I/O)."""
    connection = _get_connection(db, connection_id, user)
    return DropboxSyncStatusResponse(
        connection_id=str(connection.id),
        sync_status=connection.sync_status or "idle",
        task_id=connection.celery_task_id,
        last_error=connection.last_error,
        last_synced_at=connection.last_synced_at,
        document_count=_document_count(db, connection.user_id, connection),
        folder_name=connection.folder_name,
        connection_kind=connection.connection_kind or "oauth",
    )


@router.post(
    "/connections/{connection_id}/sync",
    response_model=DropboxSyncAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def sync_dropbox_connection(
    connection_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue a sync for one Dropbox source and return immediately."""
    connection = _get_connection(db, connection_id, user)
    if (
        (connection.connection_kind or "oauth") == "oauth"
        and not user.dropbox_refresh_token
    ):
        raise AppException(
            "DROPBOX_OAUTH_REQUIRED",
            "Connect Dropbox in Settings before syncing this folder.",
            403,
        )

    task_id, already = dropbox_sync.enqueue_connection_sync(
        db,
        connection,
        str(user.id),
        background_tasks=background_tasks,
    )
    db.refresh(connection)
    return DropboxSyncAcceptedResponse(
        message="Dropbox sync already in progress" if already else "Dropbox sync queued",
        sync_status=connection.sync_status or "queued",
        connection_id=str(connection.id),
        task_id=task_id or connection.celery_task_id,
        task_ids=[task_id] if task_id else [],
        connection_count=1,
        already_running=already,
        items=[_serialize_connection(db, connection)],
    )


@router.post(
    "/connections/{connection_id}/import",
    response_model=DropboxSyncAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def reimport_dropbox_files(
    connection_id: str,
    payload: DropboxImportSelectedRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Re-import PDFs the user explicitly deleted, selected from Dropbox."""
    connection = _get_connection(db, connection_id, user)
    if (
        (connection.connection_kind or "oauth") == "oauth"
        and not user.dropbox_refresh_token
    ):
        raise AppException(
            "DROPBOX_OAUTH_REQUIRED",
            "Connect Dropbox before re-importing files.",
            403,
        )

    task_id = dropbox_sync.enqueue_selected_import(
        db,
        connection,
        str(user.id),
        payload.file_ids,
        background_tasks=background_tasks,
    )
    return DropboxSyncAcceptedResponse(
        message="Dropbox re-import queued",
        sync_status="queued",
        connection_id=str(connection.id),
        task_id=task_id,
        task_ids=[task_id] if task_id else [],
        connection_count=1,
        items=[_serialize_connection(db, connection)],
    )


@router.post(
    "/sync",
    response_model=DropboxSyncAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def sync_all_dropbox_connections(
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue sync jobs for every attached Dropbox source."""
    connections = (
        db.query(DropboxConnection)
        .filter(
            DropboxConnection.user_id == user.id,
            DropboxConnection.enabled.is_(True),
        )
        .all()
    )
    if not connections:
        return DropboxSyncAcceptedResponse(
            message="No Dropbox sources attached",
            sync_status="idle",
            connection_count=0,
        )

    task_ids: list[str] = []
    items: list[DropboxConnectionResponse] = []
    newly_queued = 0
    already_running = 0

    for connection in connections:
        if (
            (connection.connection_kind or "oauth") == "oauth"
            and not user.dropbox_refresh_token
        ):
            connection.sync_status = "failed"
            connection.last_error = "Connect Dropbox to sync this folder"
            db.commit()
            items.append(_serialize_connection(db, connection))
            continue

        task_id, already = dropbox_sync.enqueue_connection_sync(
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
        items.append(_serialize_connection(db, connection))

    if newly_queued == 0 and already_running == 0:
        return DropboxSyncAcceptedResponse(
            message="Connect Dropbox to sync private folders",
            sync_status="failed",
            connection_count=len(connections),
            items=items,
        )

    all_already = newly_queued == 0 and already_running > 0
    return DropboxSyncAcceptedResponse(
        message=(
            "Dropbox sync already in progress" if all_already else "Dropbox sync queued"
        ),
        sync_status="queued",
        task_id=task_ids[0] if task_ids else None,
        task_ids=task_ids,
        connection_count=len(connections),
        already_running=all_already,
        items=items,
    )

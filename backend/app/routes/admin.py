"""
Admin-only operational statistics and database maintenance routes.
"""
import json
import re
import shutil
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, or_, text, inspect
from sqlalchemy.orm import Session

from app.auth import get_current_admin
from app.config import get_settings
from app.database import get_db
from app.metrics import get_query_metrics
from app.models import (
    ApiKey,
    ChatMessage,
    Document,
    DriveConnection,
    User,
    Workspace,
    WorkspaceMember,
    ResponseFeedback,
)
from app.schemas import (
    AdminFeedbackItem,
    AdminFeedbackResponse,
    AdminFeedbackSummary,
    AdminApiKeySummaryResponse,
    AdminDocumentItem,
    AdminDocumentsResponse,
    AdminDriveConnectionItem,
    AdminDriveConnectionsResponse,
    AdminStatsResponse,
    AdminWorkspaceItem,
    AdminWorkspacesResponse,
    DiskUsageResponse,
    UsageDailyResponse,
    UsageEventsResponse,
    UsageSummaryResponse,
    UserResponse,
)
from app.services.usage_tracker import (
    daily_usage,
    list_usage_events,
    summarize_usage,
    usage_query_bounds,
)

router = APIRouter(prefix="/admin", tags=["Admin"])
settings = get_settings()

# Never dump these columns in admin DB exports.
_EXPORT_REDACT_COLUMNS = {
    "hashed_password",
    "hashed_key",
    "google_refresh_token",
    "credentials_json",
    "verification_token_hash",
    "token_hash",
    "access_token",
    "refresh_token",
    "secret_key",
    "api_key",
    "private_key",
}

_EXPORT_REDACT_NAME_NEEDLES = (
    "password",
    "secret",
    "credential",
    "api_key",
    "apikey",
    "private_key",
    "hashed_key",
    "refresh_token",
    "access_token",
    "token_hash",
)


def _should_redact_export_column(col: str) -> bool:
    """True when a DB column must never appear in admin backups."""
    name = (col or "").strip().lower()
    if not name:
        return False
    if name in {c.lower() for c in _EXPORT_REDACT_COLUMNS}:
        return True
    return any(needle in name for needle in _EXPORT_REDACT_NAME_NEEDLES)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


@router.get(
    "/stats",
    response_model=AdminStatsResponse,
    summary="Get admin dashboard statistics",
    description=(
        "Returns aggregate user, document, message, query-latency, and disk "
        "usage metrics for authenticated administrators."
    ),
)
def get_admin_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """Return aggregate operational statistics for the admin dashboard."""
    upload_dir = Path(settings.UPLOAD_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    disk_usage = shutil.disk_usage(upload_dir)
    used_percent = (
        round((disk_usage.used / disk_usage.total) * 100, 2)
        if disk_usage.total
        else 0.0
    )
    query_metrics = get_query_metrics()

    total_pdfs_uploaded = (
        db.query(Document)
        .filter(func.lower(Document.original_name).like("%.pdf"))
        .count()
    )

    return AdminStatsResponse(
        total_users=db.query(User).count(),
        total_pdfs_uploaded=total_pdfs_uploaded,
        total_documents=db.query(Document).count(),
        total_messages=db.query(ChatMessage).count(),
        average_query_response_time_ms=float(
            query_metrics["average_query_response_time_ms"]
        ),
        query_count=int(query_metrics["query_count"]),
        disk_space_usage=DiskUsageResponse(
            total_bytes=disk_usage.total,
            used_bytes=disk_usage.used,
            free_bytes=disk_usage.free,
            usage_percent=used_percent,
            upload_dir_bytes=_directory_size(upload_dir),
        ),
        users=db.query(User).all()
    )


@router.get(
    "/users",
    response_model=List[UserResponse],
    summary="List all registered users",
    description="Returns the registered user inventory for authenticated administrators.",
)
def list_all_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """List all registered users."""
    return db.query(User).all()


@router.get(
    "/export-db",
    summary="Export database backup",
    description="Dumps all database tables securely into either a JSON document or a SQL injection script.",
)
def export_database(
    format: str = "json",
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """Securely export all database table records for offsite backup support."""
    format_type = format.lower()
    if format_type not in ["json", "sql"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid export format specified. Supported variants: json, sql"
        )

    inspector = inspect(db.get_bind())
    table_names = inspector.get_table_names()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _assert_safe_ident(name: str) -> str:
        # Table names come from the inspector, but still reject anything that
        # is not a plain SQL identifier before interpolating into text().
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
            raise HTTPException(status_code=500, detail="Unexpected table name")
        return name

    if format_type == "json":
        backup_data = {}
        for table in table_names:
            safe_table = _assert_safe_ident(table)
            result = db.execute(text(f"SELECT * FROM {safe_table}"))
            columns = list(result.keys())
            rows = []
            for row in result.fetchall():
                row_dict = {}
                for col, val in zip(columns, row):
                    if _should_redact_export_column(col):
                        row_dict[col] = "[REDACTED]"
                    elif isinstance(val, (datetime, date)):
                        row_dict[col] = val.isoformat()
                    elif isinstance(val, bytes):
                        row_dict[col] = val.decode("utf-8", errors="ignore")
                    else:
                        row_dict[col] = val
                rows.append(row_dict)
            backup_data[table] = rows

        content = json.dumps(backup_data, indent=2, default=str)
        filename = f"db_backup_{timestamp}.json"
        media_type = "application/json"

    else:
        sql_lines = [
            "-- Enterprise Agentic RAG System Database Backup",
            f"-- Generated at: {datetime.now(timezone.utc).isoformat()}",
            "-- Format: cross-compatible SQL script",
            "-- Sensitive credential columns are redacted.\n",
        ]
        for table in table_names:
            safe_table = _assert_safe_ident(table)
            result = db.execute(text(f"SELECT * FROM {safe_table}"))
            columns = list(result.keys())
            rows = result.fetchall()
            if rows:
                sql_lines.append(f"-- Data records for table: {safe_table}")
                cols_str = ", ".join([f'"{c}"' for c in columns])
                for row in rows:
                    vals = []
                    for col, val in zip(columns, row):
                        if _should_redact_export_column(col):
                            vals.append("'[REDACTED]'")
                        elif val is None:
                            vals.append("NULL")
                        elif isinstance(val, (int, float)):
                            vals.append(str(val))
                        elif isinstance(val, bool):
                            vals.append("1" if val else "0")
                        elif isinstance(val, (datetime, date)):
                            vals.append(f"'{val.isoformat()}'")
                        else:
                            escaped_val = str(val).replace("'", "''")
                            vals.append(f"'{escaped_val}'")
                    vals_str = ", ".join(vals)
                    sql_lines.append(
                        f'INSERT INTO "{table}" ({cols_str}) VALUES ({vals_str});'
                    )
                sql_lines.append("")

        content = "\n".join(sql_lines)
        filename = f"db_backup_{timestamp}.sql"
        media_type = "application/sql"

    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }
    return Response(content=content, media_type=media_type, headers=headers)


@router.get(
    "/usage/summary",
    response_model=UsageSummaryResponse,
    summary="Platform-wide AI usage summary",
)
def admin_usage_summary(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    start_dt, end_dt = usage_query_bounds(start, end)
    return summarize_usage(db, user_id=None, start_dt=start_dt, end_dt=end_dt)


@router.get(
    "/usage/daily",
    response_model=UsageDailyResponse,
    summary="Platform-wide daily AI usage",
)
def admin_usage_daily(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    start_dt, end_dt = usage_query_bounds(start, end)
    points = daily_usage(db, user_id=None, start_dt=start_dt, end_dt=end_dt)
    return {
        "start": start_dt.isoformat().replace("+00:00", "Z"),
        "end": end_dt.isoformat().replace("+00:00", "Z"),
        "points": points,
    }


@router.get(
    "/usage/events",
    response_model=UsageEventsResponse,
    summary="Platform-wide AI usage transactions",
)
def admin_usage_events(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    start_dt, end_dt = usage_query_bounds(start, end)
    return list_usage_events(
        db,
        user_id=None,
        start_dt=start_dt,
        end_dt=end_dt,
        page=page,
        limit=limit,
    )


@router.get(
    "/documents",
    response_model=AdminDocumentsResponse,
    summary="List documents across all users",
)
def admin_list_documents(
    status: Optional[str] = Query(
        None, description="Filter by status: ready|failed|processing|pending|..."
    ),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    query = db.query(Document, User).outerjoin(User, Document.user_id == User.id)
    if status:
        query = query.filter(Document.status == status.strip().lower())
    rows = query.order_by(Document.uploaded_at.desc()).limit(limit).all()
    status_counts = dict(
        db.query(Document.status, func.count(Document.id))
        .group_by(Document.status)
        .all()
    )
    items = [
        AdminDocumentItem(
            id=str(doc.id),
            user_id=str(doc.user_id) if doc.user_id else None,
            username=user.username if user else None,
            email=user.email if user else None,
            original_name=doc.original_name,
            status=doc.status or "unknown",
            page_count=doc.page_count,
            chunk_count=doc.chunk_count,
            error_message=(doc.error_message or None),
            processing_stage=getattr(doc, "processing_stage", None),
            retry_count=getattr(doc, "retry_count", None),
            uploaded_at=doc.uploaded_at,
            completed_at=getattr(doc, "completed_at", None),
        )
        for doc, user in rows
    ]
    return AdminDocumentsResponse(
        total=db.query(func.count(Document.id)).scalar() or 0,
        status_counts={str(k): int(v) for k, v in status_counts.items()},
        items=items,
    )


@router.get(
    "/drive/connections",
    response_model=AdminDriveConnectionsResponse,
    summary="List Drive connections across users (no credentials)",
)
def admin_list_drive_connections(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    rows = (
        db.query(DriveConnection, User)
        .outerjoin(User, DriveConnection.user_id == User.id)
        .order_by(DriveConnection.updated_at.desc())
        .limit(500)
        .all()
    )
    items = [
        AdminDriveConnectionItem(
            id=str(conn.id),
            user_id=str(conn.user_id),
            username=user.username if user else None,
            email=user.email if user else None,
            folder_id=conn.folder_id,
            folder_name=conn.folder_name,
            access_mode=conn.access_mode,
            enabled=bool(conn.enabled),
            sync_status=conn.sync_status,
            last_error=(conn.last_error[:300] if conn.last_error else None),
            last_synced_at=conn.last_synced_at,
            updated_at=conn.updated_at,
        )
        for conn, user in rows
    ]
    return AdminDriveConnectionsResponse(total=len(items), items=items)


@router.get(
    "/workspaces",
    response_model=AdminWorkspacesResponse,
    summary="List all workspaces with member counts",
)
def admin_list_workspaces(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    member_counts = dict(
        db.query(WorkspaceMember.workspace_id, func.count(WorkspaceMember.id))
        .group_by(WorkspaceMember.workspace_id)
        .all()
    )
    rows = (
        db.query(Workspace, User)
        .outerjoin(User, Workspace.created_by == User.id)
        .order_by(Workspace.created_at.desc())
        .all()
    )
    items = [
        AdminWorkspaceItem(
            id=str(ws.id),
            name=ws.name,
            created_by=str(ws.created_by),
            creator_username=creator.username if creator else None,
            member_count=int(member_counts.get(ws.id, 0)),
            created_at=ws.created_at,
        )
        for ws, creator in rows
    ]
    return AdminWorkspacesResponse(total=len(items), items=items)


@router.get(
    "/api-keys/summary",
    response_model=AdminApiKeySummaryResponse,
    summary="API key activity summary (no secrets)",
)
def admin_api_keys_summary(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    total = db.query(func.count(ApiKey.id)).scalar() or 0
    active = (
        db.query(func.count(ApiKey.id)).filter(ApiKey.is_active.is_(True)).scalar()
        or 0
    )
    user_ids = [row[0] for row in db.query(ApiKey.user_id).distinct().all()]
    by_user = []
    for user_id in user_ids:
        user = db.query(User).filter(User.id == user_id).first()
        total_keys = (
            db.query(func.count(ApiKey.id)).filter(ApiKey.user_id == user_id).scalar()
            or 0
        )
        active_keys = (
            db.query(func.count(ApiKey.id))
            .filter(ApiKey.user_id == user_id, ApiKey.is_active.is_(True))
            .scalar()
            or 0
        )
        last_used = (
            db.query(func.max(ApiKey.last_used_at))
            .filter(ApiKey.user_id == user_id)
            .scalar()
        )
        by_user.append(
            {
                "user_id": str(user_id),
                "username": user.username if user else None,
                "email": user.email if user else None,
                "total_keys": int(total_keys),
                "active_keys": int(active_keys),
                "last_used_at": last_used,
            }
        )
    by_user.sort(key=lambda row: row["total_keys"], reverse=True)
    return {
        "total_keys": int(total),
        "active_keys": int(active),
        "by_user": by_user,
    }


def _parse_feedback_bound(value: Optional[str], *, end_of_day: bool) -> Optional[datetime]:
    """Parse a YYYY-MM-DD or ISO-8601 bound into an aware UTC datetime."""
    if not value or not value.strip():
        return None
    cleaned = value.strip()
    try:
        if len(cleaned) == 10:
            day = date.fromisoformat(cleaned)
            parsed = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            if end_of_day:
                parsed = parsed + timedelta(days=1) - timedelta(microseconds=1)
            return parsed
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid date. Use YYYY-MM-DD or an ISO-8601 timestamp.",
        ) from exc


@router.get(
    "/feedback",
    response_model=AdminFeedbackResponse,
    summary="List user response feedback",
    description=(
        "Returns Like/Dislike ratings with the exact question and response, "
        "filterable by type, user/question/response text, and date range."
    ),
)
def admin_list_feedback(
    feedback: Optional[str] = Query(None, pattern="^(up|down)$"),
    search: Optional[str] = Query(None, max_length=200),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """Admin-only feedback ledger with summary, filters, and pagination."""
    query = db.query(ResponseFeedback, User).outerjoin(
        User, ResponseFeedback.user_id == User.id
    )

    if feedback:
        query = query.filter(ResponseFeedback.feedback == feedback)

    if search and search.strip():
        like = f"%{search.strip()}%"
        query = query.filter(
            or_(
                User.username.ilike(like),
                User.email.ilike(like),
                ResponseFeedback.question.ilike(like),
                ResponseFeedback.response.ilike(like),
            )
        )

    start_dt = _parse_feedback_bound(start, end_of_day=False)
    end_dt = _parse_feedback_bound(end, end_of_day=True)
    if start_dt:
        query = query.filter(ResponseFeedback.created_at >= start_dt)
    if end_dt:
        query = query.filter(ResponseFeedback.created_at <= end_dt)

    total = query.count()
    rows = (
        query.order_by(ResponseFeedback.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    # Summary reflects all feedback, not just the current filter/page.
    counts = dict(
        db.query(ResponseFeedback.feedback, func.count(ResponseFeedback.id))
        .group_by(ResponseFeedback.feedback)
        .all()
    )
    likes = int(counts.get("up", 0))
    dislikes = int(counts.get("down", 0))
    rated = likes + dislikes
    satisfaction = round((likes / rated) * 100, 1) if rated else 0.0

    items = [
        AdminFeedbackItem(
            id=str(row.id),
            user_id=str(row.user_id),
            username=user.username if user else None,
            email=user.email if user else None,
            question=row.question or "",
            response=row.response or "",
            feedback=row.feedback,
            reason=row.reason,
            comment=row.comment,
            session_id=str(row.session_id) if row.session_id else None,
            message_id=str(row.message_id),
            question_message_id=(
                str(row.question_message_id) if row.question_message_id else None
            ),
            created_at=row.created_at,
        )
        for row, user in rows
    ]

    return AdminFeedbackResponse(
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
        summary=AdminFeedbackSummary(
            total=rated,
            likes=likes,
            dislikes=dislikes,
            satisfaction_percentage=satisfaction,
        ),
        items=items,
    )

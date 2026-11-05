"""Authenticated user AI usage endpoints (own data only)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.schemas import (
    UsageDailyResponse,
    UsageEventsResponse,
    UsageSummaryResponse,
)
from app.services.usage_tracker import (
    daily_usage,
    list_usage_events,
    summarize_usage,
    usage_query_bounds,
)

router = APIRouter(prefix="/usage", tags=["Usage"])


@router.get(
    "/summary",
    response_model=UsageSummaryResponse,
    summary="Get AI usage summary for the current user",
)
def get_usage_summary(
    start: str | None = Query(None, description="UTC start (YYYY-MM-DD or ISO)"),
    end: str | None = Query(None, description="UTC end (YYYY-MM-DD or ISO)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start_dt, end_dt = usage_query_bounds(start, end)
    return summarize_usage(
        db, user_id=str(current_user.id), start_dt=start_dt, end_dt=end_dt
    )


@router.get(
    "/daily",
    response_model=UsageDailyResponse,
    summary="Get daily AI usage series for the current user",
)
def get_usage_daily(
    start: str | None = Query(None),
    end: str | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start_dt, end_dt = usage_query_bounds(start, end)
    points = daily_usage(
        db, user_id=str(current_user.id), start_dt=start_dt, end_dt=end_dt
    )
    return {
        "start": start_dt.isoformat().replace("+00:00", "Z"),
        "end": end_dt.isoformat().replace("+00:00", "Z"),
        "points": points,
    }


@router.get(
    "/events",
    response_model=UsageEventsResponse,
    summary="List AI usage transactions for the current user",
)
def get_usage_events(
    start: str | None = Query(None),
    end: str | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start_dt, end_dt = usage_query_bounds(start, end)
    return list_usage_events(
        db,
        user_id=str(current_user.id),
        start_dt=start_dt,
        end_dt=end_dt,
        page=page,
        limit=limit,
    )

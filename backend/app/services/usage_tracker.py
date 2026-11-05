"""Persist and query per-user AI usage records."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import LLMUsage
from app.pricing import enrich_usage_with_pricing

logger = logging.getLogger(__name__)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def record_llm_usage(
    db: Session,
    *,
    user_id: str,
    usage: dict[str, Any] | None,
    usage_type: str = "chat",
    provider: str | None = None,
    model_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
    document_id: str | None = None,
    commit: bool = True,
) -> LLMUsage | None:
    """Insert one usage row for the authenticated user.

    Does not invent token counts. When metadata is missing, stores nulls with
    ``cost_type=unavailable`` so request volume is still visible.
    """
    if not user_id:
        return None

    enriched = enrich_usage_with_pricing(
        usage,
        provider=provider,
        model_id=model_id,
    )
    prov = (enriched.get("provider") or provider or "unknown").strip().lower()
    mid = enriched.get("model_id") or model_id

    row = LLMUsage(
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
        document_id=document_id,
        provider=prov,
        model_id=mid,
        usage_type=(usage_type or "chat").strip().lower()[:64],
        input_tokens=_as_int(enriched.get("input_tokens")),
        output_tokens=_as_int(enriched.get("output_tokens")),
        cache_read_tokens=_as_int(enriched.get("cache_read_tokens")),
        cache_write_tokens=_as_int(enriched.get("cache_write_tokens")),
        total_tokens=_as_int(enriched.get("total_tokens")),
        estimated_cost_usd=(
            float(enriched["estimated_cost_usd"])
            if isinstance(enriched.get("estimated_cost_usd"), (int, float))
            else None
        ),
        cost_type=str(enriched.get("cost_type") or "unavailable")[:32],
        pricing_note=(
            str(enriched["pricing_note"])[:512]
            if enriched.get("pricing_note")
            else None
        ),
    )
    try:
        db.add(row)
        if commit:
            db.commit()
            db.refresh(row)
        else:
            db.flush()
        return row
    except Exception:
        logger.exception("Failed to record LLM usage for user %s", user_id)
        if commit:
            db.rollback()
        return None


def _parse_bound(value: Optional[str], *, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
            d = date.fromisoformat(raw)
            if end_of_day:
                return datetime(
                    d.year, d.month, d.day, 23, 59, 59, 999999, tzinfo=timezone.utc
                )
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def usage_query_bounds(
    start: Optional[str],
    end: Optional[str],
    *,
    default_days: int = 30,
) -> tuple[datetime, datetime]:
    """Resolve inclusive UTC window; default last ``default_days`` days."""
    now = datetime.now(timezone.utc)
    end_dt = _parse_bound(end, end_of_day=True) or now
    start_dt = _parse_bound(start) or (end_dt - timedelta(days=default_days))
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
    return start_dt, end_dt


def _base_filter(
    start_dt: datetime,
    end_dt: datetime,
    *,
    user_id: str | None = None,
):
    filters = (
        LLMUsage.created_at >= start_dt,
        LLMUsage.created_at <= end_dt,
    )
    if user_id:
        return (LLMUsage.user_id == user_id, *filters)
    return filters


def summarize_usage(
    db: Session,
    *,
    user_id: str | None,
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, Any]:
    filters = _base_filter(start_dt, end_dt, user_id=user_id)
    totals = (
        db.query(
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cache_read_tokens), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.estimated_cost_usd), 0.0),
        )
        .filter(*filters)
        .one()
    )

    by_provider_rows = (
        db.query(
            LLMUsage.provider,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.estimated_cost_usd), 0.0),
        )
        .filter(*filters)
        .group_by(LLMUsage.provider)
        .all()
    )
    by_model_rows = (
        db.query(
            LLMUsage.provider,
            LLMUsage.model_id,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.estimated_cost_usd), 0.0),
        )
        .filter(*filters)
        .group_by(LLMUsage.provider, LLMUsage.model_id)
        .all()
    )
    by_type_rows = (
        db.query(
            LLMUsage.usage_type,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.estimated_cost_usd), 0.0),
        )
        .filter(*filters)
        .group_by(LLMUsage.usage_type)
        .all()
    )
    by_user_rows = []
    if user_id is None:
        by_user_rows = (
            db.query(
                LLMUsage.user_id,
                func.count(LLMUsage.id),
                func.coalesce(func.sum(LLMUsage.total_tokens), 0),
                func.coalesce(func.sum(LLMUsage.estimated_cost_usd), 0.0),
            )
            .filter(*filters)
            .group_by(LLMUsage.user_id)
            .order_by(func.coalesce(func.sum(LLMUsage.total_tokens), 0).desc())
            .limit(50)
            .all()
        )

    # Month-to-date and today rollups (UTC)
    now = datetime.now(timezone.utc)
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    def _cost_and_tokens(since: datetime) -> dict[str, Any]:
        period_filters = [
            LLMUsage.created_at >= since,
            LLMUsage.created_at <= now,
        ]
        if user_id:
            period_filters.insert(0, LLMUsage.user_id == user_id)
        row = (
            db.query(
                func.count(LLMUsage.id),
                func.coalesce(func.sum(LLMUsage.total_tokens), 0),
                func.coalesce(func.sum(LLMUsage.estimated_cost_usd), 0.0),
            )
            .filter(*period_filters)
            .one()
        )
        return {
            "request_count": int(row[0] or 0),
            "total_tokens": int(row[1] or 0),
            "estimated_cost_usd": float(row[2] or 0.0),
        }

    payload = {
        "start": start_dt.isoformat().replace("+00:00", "Z"),
        "end": end_dt.isoformat().replace("+00:00", "Z"),
        "request_count": int(totals[0] or 0),
        "input_tokens": int(totals[1] or 0),
        "output_tokens": int(totals[2] or 0),
        "cache_read_tokens": int(totals[3] or 0),
        "total_tokens": int(totals[4] or 0),
        "estimated_cost_usd": float(totals[5] or 0.0),
        "today": _cost_and_tokens(day_start),
        "month_to_date": _cost_and_tokens(month_start),
        "by_provider": [
            {
                "provider": r[0],
                "request_count": int(r[1] or 0),
                "total_tokens": int(r[2] or 0),
                "estimated_cost_usd": float(r[3] or 0.0),
            }
            for r in by_provider_rows
        ],
        "by_model": [
            {
                "provider": r[0],
                "model_id": r[1],
                "request_count": int(r[2] or 0),
                "total_tokens": int(r[3] or 0),
                "estimated_cost_usd": float(r[4] or 0.0),
            }
            for r in by_model_rows
        ],
        "by_usage_type": [
            {
                "usage_type": r[0],
                "request_count": int(r[1] or 0),
                "total_tokens": int(r[2] or 0),
                "estimated_cost_usd": float(r[3] or 0.0),
            }
            for r in by_type_rows
        ],
        "by_user": [
            {
                "user_id": str(r[0]),
                "request_count": int(r[1] or 0),
                "total_tokens": int(r[2] or 0),
                "estimated_cost_usd": float(r[3] or 0.0),
            }
            for r in by_user_rows
        ],
        "cost_disclaimer": (
            "Costs are estimated from the model pricing registry unless marked "
            "provider_reported. They are not provider invoices."
        ),
    }
    return payload


def daily_usage(
    db: Session,
    *,
    user_id: str | None,
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict[str, Any]]:
    filters = _base_filter(start_dt, end_dt, user_id=user_id)
    # Dialect-aware UTC day bucket
    try:
        from app.database import is_mysql

        if is_mysql:
            day_expr = func.date(LLMUsage.created_at)
        else:
            day_expr = func.substr(func.datetime(LLMUsage.created_at), 1, 10)
    except Exception:
        day_expr = func.substr(func.datetime(LLMUsage.created_at), 1, 10)
    rows = (
        db.query(
            day_expr.label("day"),
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
            func.coalesce(func.sum(LLMUsage.estimated_cost_usd), 0.0),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
        )
        .filter(*filters)
        .group_by(day_expr)
        .order_by(day_expr.asc())
        .all()
    )

    by_day = {
        str(r[0]): {
            "date": str(r[0]),
            "request_count": int(r[1] or 0),
            "total_tokens": int(r[2] or 0),
            "estimated_cost_usd": float(r[3] or 0.0),
            "input_tokens": int(r[4] or 0),
            "output_tokens": int(r[5] or 0),
        }
        for r in rows
        if r[0]
    }

    # Fill missing calendar days for a continuous chart
    out: list[dict[str, Any]] = []
    cursor = start_dt.date()
    end_day = end_dt.date()
    while cursor <= end_day:
        key = cursor.isoformat()
        out.append(
            by_day.get(
                key,
                {
                    "date": key,
                    "request_count": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            )
        )
        cursor += timedelta(days=1)
    return out


def list_usage_events(
    db: Session,
    *,
    user_id: str | None,
    start_dt: datetime,
    end_dt: datetime,
    page: int = 1,
    limit: int = 50,
) -> dict[str, Any]:
    page = max(1, page)
    limit = min(max(1, limit), 200)
    filters = _base_filter(start_dt, end_dt, user_id=user_id)
    total = db.query(func.count(LLMUsage.id)).filter(*filters).scalar() or 0
    rows = (
        db.query(LLMUsage)
        .filter(*filters)
        .order_by(LLMUsage.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    return {
        "total": int(total),
        "page": page,
        "limit": limit,
        "pages": max(1, (int(total) + limit - 1) // limit),
        "items": [
            {
                "id": str(r.id),
                "created_at": r.created_at.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
                if r.created_at.tzinfo
                else r.created_at.replace(tzinfo=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "provider": r.provider,
                "model_id": r.model_id,
                "usage_type": r.usage_type,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cache_read_tokens": r.cache_read_tokens,
                "cache_write_tokens": r.cache_write_tokens,
                "total_tokens": r.total_tokens,
                "estimated_cost_usd": r.estimated_cost_usd,
                "cost_type": r.cost_type,
                "pricing_note": r.pricing_note,
                "user_id": str(r.user_id) if r.user_id else None,
                "session_id": str(r.session_id) if r.session_id else None,
                "message_id": str(r.message_id) if r.message_id else None,
                "document_id": str(r.document_id) if r.document_id else None,
            }
            for r in rows
        ],
    }

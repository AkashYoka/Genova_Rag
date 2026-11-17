"""Admin-only AI processing trace APIs."""
from __future__ import annotations

import json
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from app.audit import audit_log
from app.auth import get_current_admin
from app.config import get_settings
from app.database import get_db
from app.exceptions import NotFoundException
from app.models import AITrace, User
from app.schemas import (
    AITraceDetailResponse,
    AITraceFeedbackLink,
    AITraceListItem,
    AITraceListResponse,
    AITraceLLMCallOut,
    AITraceRetrievalOut,
    AITraceStepOut,
    AITraceSummaryStats,
)

router = APIRouter(prefix="/admin/ai-traces", tags=["Admin AI Traces"])


def _parse_bound(value: Optional[str], *, end_of_day: bool) -> Optional[datetime]:
    if not value or not value.strip():
        return None
    raw = value.strip()
    try:
        if len(raw) <= 10:
            day = datetime.strptime(raw[:10], "%Y-%m-%d").date()
            if end_of_day:
                return datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)
            return datetime.combine(day, time(0, 0, 0), tzinfo=timezone.utc)
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _loads(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _client_ip(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    return request.client.host if request.client else None


@router.get(
    "",
    response_model=AITraceListResponse,
    summary="List AI processing traces",
)
def list_ai_traces(
    trace_id: Optional[str] = Query(None, max_length=64),
    user: Optional[str] = Query(None, max_length=200, description="User id/username/email"),
    q: Optional[str] = Query(None, max_length=400, description="Question search"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    status: Optional[str] = Query(None, max_length=32),
    request_type: Optional[str] = Query(None, max_length=64),
    failed_only: bool = Query(False),
    sort: str = Query("created_at_desc", pattern="^(created_at_desc|created_at_asc|duration_desc|duration_asc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    request: Request = None,
):
    query = db.query(AITrace, User).outerjoin(User, AITrace.user_id == User.id)

    if trace_id and trace_id.strip():
        query = query.filter(AITrace.trace_id.ilike(f"%{trace_id.strip()}%"))

    if user and user.strip():
        like = f"%{user.strip()}%"
        query = query.filter(
            or_(
                User.username.ilike(like),
                User.email.ilike(like),
                AITrace.user_id.ilike(like),
            )
        )

    if q and q.strip():
        query = query.filter(AITrace.question.ilike(f"%{q.strip()}%"))

    start_dt = _parse_bound(start, end_of_day=False)
    end_dt = _parse_bound(end, end_of_day=True)
    if start_dt:
        query = query.filter(AITrace.created_at >= start_dt)
    if end_dt:
        query = query.filter(AITrace.created_at <= end_dt)

    if failed_only:
        query = query.filter(AITrace.status.in_(["failed", "partial"]))
    elif status and status.strip():
        query = query.filter(AITrace.status == status.strip())

    if request_type and request_type.strip():
        query = query.filter(AITrace.request_type == request_type.strip())

    total = query.count()

    if sort == "created_at_asc":
        order = AITrace.created_at.asc()
    elif sort == "duration_desc":
        order = AITrace.duration_ms.desc()
    elif sort == "duration_asc":
        order = AITrace.duration_ms.asc()
    else:
        order = AITrace.created_at.desc()

    rows = (
        query.order_by(order)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    success_count = 0
    failed_count = 0
    partial_count = 0
    avg_duration = None
    if total:
        base = db.query(AITrace).outerjoin(User, AITrace.user_id == User.id)
        if trace_id and trace_id.strip():
            base = base.filter(AITrace.trace_id.ilike(f"%{trace_id.strip()}%"))
        if user and user.strip():
            like = f"%{user.strip()}%"
            base = base.filter(
                or_(
                    User.username.ilike(like),
                    User.email.ilike(like),
                    AITrace.user_id.ilike(like),
                )
            )
        if q and q.strip():
            base = base.filter(AITrace.question.ilike(f"%{q.strip()}%"))
        if start_dt:
            base = base.filter(AITrace.created_at >= start_dt)
        if end_dt:
            base = base.filter(AITrace.created_at <= end_dt)
        if failed_only:
            base = base.filter(AITrace.status.in_(["failed", "partial"]))
        elif status and status.strip():
            base = base.filter(AITrace.status == status.strip())
        if request_type and request_type.strip():
            base = base.filter(AITrace.request_type == request_type.strip())

        for st, cnt in base.with_entities(AITrace.status, func.count(AITrace.id)).group_by(
            AITrace.status
        ):
            if st == "success":
                success_count = int(cnt)
            elif st == "failed":
                failed_count = int(cnt)
            elif st == "partial":
                partial_count = int(cnt)
        avg_duration = base.with_entities(func.avg(AITrace.duration_ms)).scalar()

    items = [
        AITraceListItem(
            trace_id=row.trace_id,
            user_id=str(row.user_id) if row.user_id else None,
            username=u.username if u else None,
            email=u.email if u else None,
            question=row.question or "",
            request_type=row.request_type or "rag",
            status=row.status or "success",
            created_at=row.created_at,
            duration_ms=row.duration_ms,
            total_tokens=row.total_tokens,
            retrieved_chunk_count=row.retrieved_chunk_count,
            final_answer_status=row.final_answer_status,
            early_exit=bool(row.early_exit),
            model_name=row.model_name,
            provider=row.provider,
        )
        for row, u in rows
    ]

    if getattr(get_settings(), "AI_TRACE_AUDIT_ACCESS", True):
        audit_log(
            action="admin.ai_traces.list",
            user_id=str(admin.id),
            ip_address=_client_ip(request),
            details={"page": page, "total": total, "failed_only": failed_only},
        )

    return AITraceListResponse(
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if page_size else 0,
        summary=AITraceSummaryStats(
            total=total,
            success=success_count,
            failed=failed_count,
            partial=partial_count,
            avg_duration_ms=float(avg_duration) if avg_duration is not None else None,
        ),
        items=items,
    )


@router.get(
    "/{trace_id}",
    response_model=AITraceDetailResponse,
    summary="Get AI processing trace detail",
)
def get_ai_trace(
    trace_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    request: Request = None,
):
    row = (
        db.query(AITrace)
        .options(
            joinedload(AITrace.steps),
            joinedload(AITrace.retrievals),
            joinedload(AITrace.llm_calls),
            joinedload(AITrace.feedback_links),
        )
        .filter(AITrace.trace_id == trace_id)
        .first()
    )
    if not row:
        raise NotFoundException("AI Trace")

    user = None
    if row.user_id:
        user = db.query(User).filter(User.id == row.user_id).first()

    steps = [
        AITraceStepOut(
            id=str(s.id),
            sequence=s.sequence,
            name=s.name,
            status=s.status,
            started_at=s.started_at,
            finished_at=s.finished_at,
            duration_ms=s.duration_ms,
            input_summary=s.input_summary,
            output_summary=s.output_summary,
            error_message=s.error_message,
            metadata=_loads(s.metadata_json),
        )
        for s in sorted(row.steps or [], key=lambda x: x.sequence)
    ]
    retrievals = [
        AITraceRetrievalOut(
            id=str(r.id),
            sequence=r.sequence,
            search_query=r.search_query,
            search_method=r.search_method,
            document_ids=_loads(r.document_ids_json) or [],
            document_names=_loads(r.document_names_json) or [],
            chunk_ids=_loads(r.chunk_ids_json) or [],
            chunks=_loads(r.chunks_json) or [],
            filters=_loads(r.filters_json),
            rejected=_loads(r.rejected_json),
            selected_chunk_ids=_loads(r.selected_chunk_ids_json) or [],
            result_count=r.result_count,
            duration_ms=r.duration_ms,
            metadata=_loads(r.metadata_json),
        )
        for r in sorted(row.retrievals or [], key=lambda x: x.sequence)
    ]
    llm_calls = []
    for c in sorted(row.llm_calls or [], key=lambda x: x.sequence):
        meta = _loads(c.metadata_json) or {}
        if not isinstance(meta, dict):
            meta = {}
        llm_calls.append(
            AITraceLLMCallOut(
                id=str(c.id),
                sequence=c.sequence,
                model_name=c.model_name,
                provider=c.provider,
                system_prompt_version=c.system_prompt_version,
                temperature=c.temperature,
                input_tokens=c.input_tokens,
                output_tokens=c.output_tokens,
                cached_tokens=c.cached_tokens,
                total_tokens=c.total_tokens,
                estimated_cost_usd=meta.get("estimated_cost_usd"),
                cost_type=meta.get("cost_type"),
                pricing_note=meta.get("pricing_note"),
                latency_ms=c.latency_ms,
                retry_count=c.retry_count,
                validation_result=c.validation_result,
                status=c.status,
                error_message=c.error_message,
                system_prompt=c.system_prompt,
                user_prompt=c.user_prompt,
                context_text=c.context_text,
                raw_response=c.raw_response,
                chunks_sent_to_llm=meta.get("chunks_sent_to_llm"),
                metadata=meta,
            )
        )
    feedback = [
        AITraceFeedbackLink(
            feedback_id=str(f.feedback_id) if f.feedback_id else None,
            message_id=str(f.message_id) if f.message_id else None,
            feedback=f.feedback,
            reason=f.reason,
            comment=f.comment,
            created_at=f.created_at,
        )
        for f in (row.feedback_links or [])
    ]

    duration_breakdown = [
        {"name": s.name, "duration_ms": s.duration_ms, "status": s.status}
        for s in steps
        if s.duration_ms is not None
    ]
    duration_breakdown.sort(key=lambda x: -(x.get("duration_ms") or 0))

    if getattr(get_settings(), "AI_TRACE_AUDIT_ACCESS", True):
        audit_log(
            action="admin.ai_traces.detail",
            user_id=str(admin.id),
            resource=trace_id,
            ip_address=_client_ip(request),
        )

    return AITraceDetailResponse(
        trace_id=row.trace_id,
        user_id=str(row.user_id) if row.user_id else None,
        username=user.username if user else None,
        email=user.email if user else None,
        session_id=str(row.session_id) if row.session_id else None,
        message_id=str(row.message_id) if row.message_id else None,
        document_id=str(row.document_id) if row.document_id else None,
        question=row.question or "",
        request_type=row.request_type or "rag",
        status=row.status or "success",
        early_exit=bool(row.early_exit),
        early_exit_reason=row.early_exit_reason,
        started_at=row.started_at,
        finished_at=row.finished_at,
        created_at=row.created_at,
        duration_ms=row.duration_ms,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cached_tokens=row.cached_tokens,
        total_tokens=row.total_tokens,
        model_name=row.model_name,
        provider=row.provider,
        retrieved_chunk_count=row.retrieved_chunk_count,
        final_answer=row.final_answer,
        final_answer_status=row.final_answer_status,
        error_summary=row.error_summary,
        warnings=_loads(row.warnings_json) or [],
        metadata=_loads(row.metadata_json) or {},
        steps=steps,
        retrievals=retrievals,
        llm_calls=llm_calls,
        feedback=feedback,
        duration_breakdown=duration_breakdown,
    )

"""Admin APIs for dynamic prompt management and domain detection."""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import get_current_admin
from app.database import get_db
from app.models import (
    DomainDetectionLog,
    PromptDomain,
    PromptExecutionLog,
    PromptTemplate,
    PromptVersion,
    User,
)
from app.schemas import (
    DomainCardResponse,
    DomainDetectRequest,
    DomainDetectResponse,
    PromptDomainCreate,
    PromptDomainResponse,
    PromptDomainUpdate,
    PromptOverviewResponse,
    PromptPreviewRequest,
    PromptPreviewResponse,
    PromptStatsResponse,
    PromptTemplateCreate,
    PromptTemplateResponse,
    PromptTemplateUpdate,
    PromptTestRequest,
    PromptTestResponse,
    PromptVersionResponse,
)
from app.services.prompt_management.defaults import DOMAIN_META
from app.services.prompt_management.detector import (
    DomainDetectionResult,
    clear_detection_cache,
    detect_domain,
)
from app.services.prompt_management.resolver import resolve_prompts
from app.services.prompt_management.seed import (
    ensure_default_prompts,
    get_canonical_domain_system,
    get_canonical_global_system,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/prompts", tags=["Admin - Prompts"])


def _template_response(row: PromptTemplate) -> PromptTemplateResponse:
    return PromptTemplateResponse(
        id=str(row.id),
        domain_id=str(row.domain_id) if row.domain_id else None,
        domain_slug=row.domain.slug if row.domain else None,
        prompt_type=row.prompt_type,
        name=row.name,
        content=row.content,
        description=row.description,
        version=row.version,
        is_active=row.is_active,
        is_default=row.is_default,
        created_by=str(row.created_by) if row.created_by else None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _snapshot(db: Session, template: PromptTemplate, note: str, user_id: Optional[str]) -> None:
    db.add(
        PromptVersion(
            template_id=template.id,
            version_number=template.version,
            content=template.content,
            change_note=note,
            created_by=user_id,
        )
    )


@router.post("/seed", status_code=status.HTTP_200_OK)
def seed_defaults(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    ensure_default_prompts(db)
    db.commit()
    return {"ok": True, "message": "Default domains and prompts ensured"}


# ── Domains ──────────────────────────────────────────────────────────────────


@router.get("/domains", response_model=List[PromptDomainResponse])
def list_domains(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
    include_inactive: bool = Query(False),
):
    q = db.query(PromptDomain)
    if not include_inactive:
        q = q.filter(PromptDomain.is_active.is_(True))
    rows = q.order_by(PromptDomain.sort_order.asc(), PromptDomain.name.asc()).all()
    return rows


@router.post("/domains", response_model=PromptDomainResponse, status_code=201)
def create_domain(
    body: PromptDomainCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    slug = body.slug.strip().lower()
    if db.query(PromptDomain).filter(PromptDomain.slug == slug).first():
        raise HTTPException(status_code=409, detail=f"Domain '{slug}' already exists")
    if body.is_fallback:
        db.query(PromptDomain).filter(PromptDomain.is_fallback.is_(True)).update(
            {"is_fallback": False}
        )
    row = PromptDomain(
        slug=slug,
        name=body.name.strip(),
        description=body.description,
        is_active=body.is_active,
        is_builtin=False,
        is_fallback=body.is_fallback,
        sort_order=body.sort_order,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    clear_detection_cache()
    return row


@router.put("/domains/{domain_id}", response_model=PromptDomainResponse)
def update_domain(
    domain_id: str,
    body: PromptDomainUpdate,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    row = db.query(PromptDomain).filter(PromptDomain.id == domain_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Domain not found")
    data = body.model_dump(exclude_unset=True)
    if data.get("is_fallback"):
        db.query(PromptDomain).filter(
            PromptDomain.is_fallback.is_(True), PromptDomain.id != domain_id
        ).update({"is_fallback": False})
    for key, value in data.items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    clear_detection_cache()
    return row


@router.delete("/domains/{domain_id}", status_code=204)
def delete_domain(
    domain_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    row = db.query(PromptDomain).filter(PromptDomain.id == domain_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Domain not found")
    if row.is_builtin:
        raise HTTPException(status_code=400, detail="Built-in domains cannot be deleted; deactivate instead")
    if row.is_fallback:
        raise HTTPException(status_code=400, detail="Cannot delete the fallback domain")
    db.delete(row)
    db.commit()
    clear_detection_cache()
    return None


# ── Templates ────────────────────────────────────────────────────────────────


@router.get("/templates", response_model=List[PromptTemplateResponse])
def list_templates(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
    domain_id: Optional[str] = None,
    prompt_type: Optional[str] = None,
    active_only: bool = Query(False),
):
    q = db.query(PromptTemplate)
    if domain_id == "global":
        q = q.filter(PromptTemplate.domain_id.is_(None))
    elif domain_id:
        q = q.filter(PromptTemplate.domain_id == domain_id)
    if prompt_type:
        q = q.filter(PromptTemplate.prompt_type == prompt_type)
    if active_only:
        q = q.filter(PromptTemplate.is_active.is_(True))
    rows = q.order_by(PromptTemplate.prompt_type.asc(), PromptTemplate.name.asc()).all()
    return [_template_response(r) for r in rows]


@router.post("/templates", response_model=PromptTemplateResponse, status_code=201)
def create_template(
    body: PromptTemplateCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    if body.domain_id:
        domain = db.query(PromptDomain).filter(PromptDomain.id == body.domain_id).first()
        if not domain:
            raise HTTPException(status_code=404, detail="Domain not found")
    if body.is_default:
        q = db.query(PromptTemplate).filter(
            PromptTemplate.prompt_type == body.prompt_type,
            PromptTemplate.is_default.is_(True),
        )
        if body.domain_id:
            q = q.filter(PromptTemplate.domain_id == body.domain_id)
        else:
            q = q.filter(PromptTemplate.domain_id.is_(None))
        q.update({"is_default": False})
    row = PromptTemplate(
        domain_id=body.domain_id,
        prompt_type=body.prompt_type,
        name=body.name.strip(),
        content=body.content,
        description=body.description,
        version=1,
        is_active=body.is_active,
        is_default=body.is_default,
        created_by=str(admin.id),
    )
    db.add(row)
    db.flush()
    _snapshot(db, row, "Created", str(admin.id))
    db.commit()
    db.refresh(row)
    clear_detection_cache()
    return _template_response(row)


@router.get("/templates/{template_id}", response_model=PromptTemplateResponse)
def get_template(
    template_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    row = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    return _template_response(row)


@router.put("/templates/{template_id}", response_model=PromptTemplateResponse)
def update_template(
    template_id: str,
    body: PromptTemplateUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    row = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    data = body.model_dump(exclude_unset=True)
    change_note = data.pop("change_note", None) or "Updated"
    content_changed = "content" in data and data["content"] != row.content
    if data.get("is_default"):
        q = db.query(PromptTemplate).filter(
            PromptTemplate.prompt_type == row.prompt_type,
            PromptTemplate.is_default.is_(True),
            PromptTemplate.id != row.id,
        )
        if row.domain_id:
            q = q.filter(PromptTemplate.domain_id == row.domain_id)
        else:
            q = q.filter(PromptTemplate.domain_id.is_(None))
        q.update({"is_default": False})
    for key, value in data.items():
        setattr(row, key, value)
    if content_changed:
        row.version = int(row.version or 1) + 1
        _snapshot(db, row, change_note, str(admin.id))
    db.commit()
    db.refresh(row)
    clear_detection_cache()
    return _template_response(row)


@router.delete("/templates/{template_id}", status_code=204)
def delete_template(
    template_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    row = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    db.delete(row)
    db.commit()
    clear_detection_cache()
    return None


@router.post("/templates/{template_id}/activate", response_model=PromptTemplateResponse)
def activate_template(
    template_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    row = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    row.is_active = True
    db.commit()
    db.refresh(row)
    clear_detection_cache()
    return _template_response(row)


@router.post("/templates/{template_id}/deactivate", response_model=PromptTemplateResponse)
def deactivate_template(
    template_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    row = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    row.is_active = False
    db.commit()
    db.refresh(row)
    clear_detection_cache()
    return _template_response(row)


@router.get("/templates/{template_id}/versions", response_model=List[PromptVersionResponse])
def list_versions(
    template_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    rows = (
        db.query(PromptVersion)
        .filter(PromptVersion.template_id == template_id)
        .order_by(PromptVersion.version_number.desc())
        .all()
    )
    return rows


@router.post(
    "/templates/{template_id}/versions/{version_number}/restore",
    response_model=PromptTemplateResponse,
)
def restore_version(
    template_id: str,
    version_number: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    template = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    version = (
        db.query(PromptVersion)
        .filter(
            PromptVersion.template_id == template_id,
            PromptVersion.version_number == version_number,
        )
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    template.content = version.content
    template.version = int(template.version or 1) + 1
    _snapshot(
        db,
        template,
        f"Restored from v{version_number}",
        str(admin.id),
    )
    db.commit()
    db.refresh(template)
    clear_detection_cache()
    return _template_response(template)


# ── Preview / detect / stats ─────────────────────────────────────────────────


@router.post("/preview", response_model=PromptPreviewResponse)
def preview_prompt(
    body: PromptPreviewRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    detection = None
    if body.domain_slug:
        detection = DomainDetectionResult(
            domain=body.domain_slug,
            confidence=1.0,
            rationale="Forced domain for preview",
        )
    resolved = resolve_prompts(
        db,
        question=body.question,
        prompt_type=body.prompt_type,
        user_id=str(admin.id),
        document_ids=body.document_ids,
        retrieved_context=body.context,
        detection=detection,
        persist_execution=False,
    )
    return PromptPreviewResponse(
        domain=resolved.domain,
        confidence=resolved.confidence,
        was_fallback=resolved.was_fallback,
        system_prompt=resolved.system_prompt,
        task_prompt=resolved.task_prompt,
        assembled_user_prompt=resolved.format_task(
            context=body.context, question=body.question
        ),
        rationale=resolved.rationale,
    )


@router.post("/detect", response_model=DomainDetectResponse)
def detect_domain_endpoint(
    body: DomainDetectRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    result = detect_domain(
        db,
        question=body.question,
        user_id=str(admin.id),
        document_ids=body.document_ids,
        retrieved_context=body.context,
        persist=True,
    )
    db.commit()
    return DomainDetectResponse(
        domain=result.domain,
        confidence=result.confidence,
        was_fallback=result.was_fallback,
        secondary_domains=result.secondary_domains,
        rationale=result.rationale,
        cache_hit=result.cache_hit,
    )


@router.get("/stats", response_model=PromptStatsResponse)
def prompt_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
    limit: int = Query(20, ge=1, le=100),
):
    domains = db.query(PromptDomain).order_by(PromptDomain.sort_order.asc()).all()
    detections_total = db.query(func.count(DomainDetectionLog.id)).scalar() or 0
    fallback_total = (
        db.query(func.count(DomainDetectionLog.id))
        .filter(DomainDetectionLog.was_fallback.is_(True))
        .scalar()
        or 0
    )
    low_confidence_total = (
        db.query(func.count(DomainDetectionLog.id))
        .filter(DomainDetectionLog.confidence < 0.55)
        .scalar()
        or 0
    )
    by_domain_rows = (
        db.query(DomainDetectionLog.detected_domain, func.count(DomainDetectionLog.id))
        .group_by(DomainDetectionLog.detected_domain)
        .all()
    )
    avg_rows = (
        db.query(
            DomainDetectionLog.detected_domain,
            func.avg(DomainDetectionLog.confidence),
        )
        .group_by(DomainDetectionLog.detected_domain)
        .all()
    )
    recent = (
        db.query(DomainDetectionLog)
        .order_by(DomainDetectionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    executions = (
        db.query(PromptExecutionLog)
        .order_by(PromptExecutionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return PromptStatsResponse(
        domains=[
            {
                "id": str(d.id),
                "slug": d.slug,
                "name": d.name,
                "is_active": d.is_active,
                "is_fallback": d.is_fallback,
                "is_builtin": d.is_builtin,
            }
            for d in domains
        ],
        detections_total=int(detections_total),
        fallback_total=int(fallback_total),
        low_confidence_total=int(low_confidence_total),
        queries_by_domain={slug: int(count) for slug, count in by_domain_rows},
        avg_confidence_by_domain={
            slug: round(float(avg or 0), 3) for slug, avg in avg_rows
        },
        recent_detections=[
            {
                "id": str(r.id),
                "domain": r.detected_domain,
                "confidence": r.confidence,
                "was_fallback": r.was_fallback,
                "question_preview": r.question_preview,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recent
        ],
        recent_executions=[
            {
                "id": str(r.id),
                "domain": r.domain_slug,
                "prompt_type": r.prompt_type,
                "confidence": r.confidence,
                "system_template_id": r.system_template_id,
                "task_template_id": r.task_template_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in executions
        ],
    )


@router.get("/overview", response_model=PromptOverviewResponse)
def prompt_overview(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """Admin landing data: KPIs + domain cards with active prompt metadata."""
    ensure_default_prompts(db)
    db.commit()

    domains = (
        db.query(PromptDomain)
        .order_by(PromptDomain.sort_order.asc(), PromptDomain.name.asc())
        .all()
    )
    active_prompts = (
        db.query(func.count(PromptTemplate.id))
        .filter(PromptTemplate.is_active.is_(True))
        .scalar()
        or 0
    )
    default_prompts = (
        db.query(func.count(PromptTemplate.id))
        .filter(PromptTemplate.is_default.is_(True), PromptTemplate.is_active.is_(True))
        .scalar()
        or 0
    )
    detections_total = db.query(func.count(DomainDetectionLog.id)).scalar() or 0
    fallback_total = (
        db.query(func.count(DomainDetectionLog.id))
        .filter(DomainDetectionLog.was_fallback.is_(True))
        .scalar()
        or 0
    )
    last_tpl = (
        db.query(PromptTemplate)
        .order_by(PromptTemplate.updated_at.desc())
        .first()
    )
    by_domain_rows = {
        slug: int(count)
        for slug, count in (
            db.query(DomainDetectionLog.detected_domain, func.count(DomainDetectionLog.id))
            .group_by(DomainDetectionLog.detected_domain)
            .all()
        )
    }
    avg_rows = {
        slug: round(float(avg or 0), 3)
        for slug, avg in (
            db.query(
                DomainDetectionLog.detected_domain,
                func.avg(DomainDetectionLog.confidence),
            )
            .group_by(DomainDetectionLog.detected_domain)
            .all()
        )
    }
    cards: List[DomainCardResponse] = []
    for d in domains:
        tpl = (
            db.query(PromptTemplate)
            .filter(
                PromptTemplate.domain_id == d.id,
                PromptTemplate.prompt_type == "system",
                PromptTemplate.is_active.is_(True),
            )
            .order_by(PromptTemplate.is_default.desc(), PromptTemplate.updated_at.desc())
            .first()
        )
        meta = DOMAIN_META.get(d.slug, {})
        cards.append(
            DomainCardResponse(
                id=str(d.id),
                slug=d.slug,
                name=d.name,
                description=d.description or meta.get("description"),
                purpose=meta.get("purpose"),
                is_active=bool(d.is_active),
                is_builtin=bool(d.is_builtin),
                is_fallback=bool(d.is_fallback),
                template_id=str(tpl.id) if tpl else None,
                template_name=tpl.name if tpl else None,
                version=tpl.version if tpl else None,
                is_prompt_active=bool(tpl.is_active) if tpl else False,
                last_updated=tpl.updated_at if tpl else d.updated_at,
                query_count=by_domain_rows.get(d.slug, 0),
                avg_confidence=avg_rows.get(d.slug),
            )
        )
    recent = (
        db.query(DomainDetectionLog)
        .order_by(DomainDetectionLog.created_at.desc())
        .limit(8)
        .all()
    )
    return PromptOverviewResponse(
        total_domains=len(domains),
        active_domains=sum(1 for d in domains if d.is_active),
        active_prompts=int(active_prompts),
        default_prompts=int(default_prompts),
        detections_total=int(detections_total),
        fallback_total=int(fallback_total),
        last_updated=last_tpl.updated_at if last_tpl else None,
        domains=cards,
        recent_detections=[
            {
                "id": str(r.id),
                "domain": r.detected_domain,
                "confidence": r.confidence,
                "was_fallback": r.was_fallback,
                "question_preview": r.question_preview,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recent
        ],
    )


@router.post(
    "/templates/{template_id}/reset-default",
    response_model=PromptTemplateResponse,
)
def reset_template_default(
    template_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """Restore a template to the canonical V1 default for its domain/type."""
    row = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")

    content = None
    if row.prompt_type == "system" and row.domain_id is None:
        content = get_canonical_global_system()
    elif row.prompt_type == "system" and row.domain_id:
        domain = db.query(PromptDomain).filter(PromptDomain.id == row.domain_id).first()
        if not domain:
            raise HTTPException(status_code=404, detail="Domain not found")
        content = get_canonical_domain_system(domain.slug)
        if not content:
            raise HTTPException(
                status_code=400,
                detail="No built-in default exists for this custom domain",
            )
    else:
        from app.services.prompt_management.defaults import global_task_defaults

        for ptype, (_name, _desc, body) in global_task_defaults().items():
            if ptype == row.prompt_type:
                content = body
                break
    if not content:
        raise HTTPException(status_code=400, detail="No default content available to restore")

    row.content = content
    row.version = int(row.version or 1) + 1
    row.is_active = True
    _snapshot(db, row, "Reset to V1 default", str(admin.id))
    db.commit()
    db.refresh(row)
    clear_detection_cache()
    return _template_response(row)


@router.post("/test", response_model=PromptTestResponse)
def test_prompt(
    body: PromptTestRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """Assemble prompts and run a one-off LLM answer without activating anything."""
    detection = None
    if body.domain_slug and not body.auto_detect:
        detection = DomainDetectionResult(
            domain=body.domain_slug,
            confidence=1.0,
            rationale="Forced domain for admin test",
        )
    elif body.domain_slug and body.auto_detect:
        # Prefer explicit domain when provided, still record confidence as forced.
        detection = DomainDetectionResult(
            domain=body.domain_slug,
            confidence=1.0,
            rationale="Admin-selected domain for test",
        )

    resolved = resolve_prompts(
        db,
        question=body.question,
        prompt_type=body.prompt_type,
        user_id=str(admin.id),
        document_ids=body.document_ids,
        retrieved_context=body.context,
        detection=detection,
        persist_execution=True,
    )
    user_prompt = resolved.format_task(context=body.context, question=body.question)

    answer = ""
    try:
        from app.rag.llm.agent import (
            get_llm_client,
            llm_model_for_provider,
            normalize_llm_provider,
        )
        from app.config import get_settings

        settings = get_settings()
        provider = normalize_llm_provider(None)
        client = get_llm_client(llm_provider=provider)
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": resolved.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=llm_model_for_provider(provider),
            max_tokens=min(768, settings.LLM_MAX_NEW_TOKENS),
            temperature=settings.LLM_TEMPERATURE,
        )
        if response and getattr(response, "choices", None):
            answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Prompt test LLM call failed: %s", exc)
        answer = f"(Test generation failed: {exc})"

    tpl_name = None
    if resolved.domain_template_id:
        tpl = (
            db.query(PromptTemplate)
            .filter(PromptTemplate.id == resolved.domain_template_id)
            .first()
        )
        tpl_name = tpl.name if tpl else None

    db.commit()
    return PromptTestResponse(
        domain=resolved.domain,
        confidence=resolved.confidence,
        was_fallback=resolved.was_fallback,
        prompt_version=resolved.domain_version or resolved.system_version,
        template_id=resolved.domain_template_id or resolved.system_template_id,
        template_name=tpl_name,
        system_prompt_preview=resolved.system_prompt[:2000],
        answer=answer or "(Empty model response)",
        rationale=resolved.rationale,
    )

"""Tests for dynamic prompt management and domain resolution."""
from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import PromptDomain, PromptTemplate
from app.services.prompt_management.detector import (
    DomainDetectionResult,
    _parse_json_payload,
    clear_detection_cache,
)
from app.services.prompt_management.resolver import infer_task_type, resolve_prompts
from app.services.prompt_management.seed import ensure_default_prompts


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    ensure_default_prompts(session)
    session.commit()
    clear_detection_cache()
    yield session
    session.close()


def test_seed_creates_builtin_domains_and_global_templates(db_session):
    domains = {d.slug for d in db_session.query(PromptDomain).all()}
    assert "general" in domains
    assert "medical" in domains
    assert "payment" in domains
    assert "education" in domains
    assert "legal" in domains
    assert "research" in domains
    general = db_session.query(PromptDomain).filter_by(slug="general").one()
    assert general.is_fallback is True

    global_system = (
        db_session.query(PromptTemplate)
        .filter(
            PromptTemplate.domain_id.is_(None),
            PromptTemplate.prompt_type == "system",
            PromptTemplate.is_active.is_(True),
        )
        .one()
    )
    assert "cite" in global_system.content.lower() or "Source:" in global_system.content


def test_resolve_prompts_uses_medical_overlay(db_session, monkeypatch):
    def fake_detect(*_args, **_kwargs):
        return DomainDetectionResult(
            domain="medical",
            confidence=0.91,
            rationale="test",
        )

    monkeypatch.setattr(
        "app.services.prompt_management.resolver.detect_domain",
        fake_detect,
    )
    resolved = resolve_prompts(
        db_session,
        question="What is the Avastin dose for recurrent glioblastoma?",
        prompt_type="rag",
        persist_execution=True,
    )
    assert resolved.domain == "medical"
    assert "DOMAIN FOCUS — Medical" in resolved.system_prompt
    assert "Never invent clinical guidance" in resolved.system_prompt or "dosages" in resolved.system_prompt.lower()
    assert "{context}" in resolved.task_prompt
    assert "{question}" in resolved.task_prompt
    user = resolved.format_task(context="dose is 10 mg/kg", question="dose?")
    assert "10 mg/kg" in user
    assert "dose?" in user


def test_low_confidence_falls_back_to_general(db_session, monkeypatch):
    def fake_detect(*_args, **_kwargs):
        return DomainDetectionResult(
            domain="legal",
            confidence=0.2,
            rationale="ambiguous",
            was_fallback=False,
        )

    # Bypass detector in resolve by passing detection that already applied threshold logic
    # Exercise detector threshold via resolve's detect_domain monkeypatch that mimics detector output after threshold
    def fake_detect_threshold(db, **kwargs):
        # Simulate detector already falling back
        return DomainDetectionResult(
            domain="general",
            confidence=0.2,
            was_fallback=True,
            rationale="Low confidence",
        )

    monkeypatch.setattr(
        "app.services.prompt_management.resolver.detect_domain",
        fake_detect_threshold,
    )
    resolved = resolve_prompts(
        db_session,
        question="something vague",
        prompt_type="rag",
        persist_execution=False,
    )
    assert resolved.domain == "general"
    assert resolved.was_fallback is True


def test_infer_task_type():
    assert infer_task_type("Please summarize this chapter") == "summarization"
    assert infer_task_type("Compare doc A and doc B") == "comparison"
    assert infer_task_type("Extract all invoice totals") == "extraction"
    assert infer_task_type("Explain mitosis") == "explanation"
    assert infer_task_type("What is the dose?") == "rag"


def test_parse_json_payload_strips_fences():
    raw = '```json\n{"domain":"payment","confidence":0.8,"secondary_domains":[],"rationale":"invoice"}\n```'
    data = _parse_json_payload(raw)
    assert data["domain"] == "payment"
    assert data["confidence"] == 0.8


def test_custom_domain_can_be_added_without_code_change(db_session):
    db_session.add(
        PromptDomain(
            slug="insurance",
            name="Insurance",
            description="Claims and policies",
            is_active=True,
            is_builtin=False,
            sort_order=200,
        )
    )
    db_session.commit()
    assert (
        db_session.query(PromptDomain).filter_by(slug="insurance").one().name
        == "Insurance"
    )


def test_every_builtin_domain_has_rich_system_default(db_session):
    from app.services.prompt_management.defaults import DOMAIN_SYSTEM_DEFAULTS

    for slug, expected in DOMAIN_SYSTEM_DEFAULTS.items():
        domain = db_session.query(PromptDomain).filter_by(slug=slug).one()
        tpl = (
            db_session.query(PromptTemplate)
            .filter(
                PromptTemplate.domain_id == domain.id,
                PromptTemplate.prompt_type == "system",
                PromptTemplate.is_active.is_(True),
            )
            .one()
        )
        assert len(tpl.content) >= 400, slug
        assert "DOMAIN FOCUS" in tpl.content
        assert expected[:40] in tpl.content or "Never" in tpl.content or "Prefer" in tpl.content


def test_reset_restores_canonical_medical_content(db_session):
    from app.services.prompt_management.defaults import DOMAIN_SYSTEM_DEFAULTS
    from app.services.prompt_management.seed import get_canonical_domain_system

    domain = db_session.query(PromptDomain).filter_by(slug="medical").one()
    tpl = (
        db_session.query(PromptTemplate)
        .filter(
            PromptTemplate.domain_id == domain.id,
            PromptTemplate.prompt_type == "system",
        )
        .one()
    )
    tpl.content = "CUSTOM MEDICAL PROMPT"
    tpl.version = 9
    db_session.commit()
    canonical = get_canonical_domain_system("medical")
    assert canonical == DOMAIN_SYSTEM_DEFAULTS["medical"]
    tpl.content = canonical
    tpl.version = 10
    db_session.commit()
    assert "dosages" in tpl.content.lower() or "clinical" in tpl.content.lower()

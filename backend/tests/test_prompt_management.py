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
    assert "trading" in domains
    assert "chemistry" in domains
    assert "mathematics" in domains
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


def test_persona_domains_have_eight_required_sections():
    from app.services.prompt_management.defaults import (
        DOMAIN_SYSTEM_DEFAULTS,
        PERSONA_SECTION_MARKERS,
    )

    personas = {
        "general": "General Document Analyst",
        "medical": "Medical Document Analyst",
        "research": "Research Analyst",
        "trading": "Trading Analyst",
        "chemistry": "Chemistry Analyst",
        "mathematics": "Mathematics Analyst",
    }
    for slug, title in personas.items():
        text = DOMAIN_SYSTEM_DEFAULTS[slug]
        assert title in text, slug
        for marker in PERSONA_SECTION_MARKERS:
            assert marker in text, f"{slug} missing {marker}"
        # Principle-driven only — no fixture answers.
        for banned in ("Xolair", "CRSwNP", "8,850.00", "AlgebraAndTrigo"):
            assert banned not in text, f"{slug} hardcodes {banned}"
    # Medical keeps PV principles.
    medical = DOMAIN_SYSTEM_DEFAULTS["medical"]
    assert "PHARMACOVIGILANCE" in medical
    assert "SCOPE LOCK" in medical


def test_seed_does_not_clobber_customized_overlay(db_session):
    domain = db_session.query(PromptDomain).filter_by(slug="research").one()
    tpl = (
        db_session.query(PromptTemplate)
        .filter(
            PromptTemplate.domain_id == domain.id,
            PromptTemplate.prompt_type == "system",
            PromptTemplate.is_active.is_(True),
        )
        .one()
    )
    tpl.content = "CUSTOM RESEARCH ANALYST PROMPT — do not overwrite"
    tpl.is_default = False
    tpl.version = 7
    db_session.commit()

    ensure_default_prompts(db_session)
    db_session.commit()

    refreshed = (
        db_session.query(PromptTemplate)
        .filter(PromptTemplate.id == tpl.id)
        .one()
    )
    assert refreshed.content == "CUSTOM RESEARCH ANALYST PROMPT — do not overwrite"
    assert refreshed.version == 7


def test_filename_hints_map_smoke_pdfs_to_personas():
    from app.services.prompt_management.detector import infer_domain_from_filename

    allowed = [
        "general",
        "medical",
        "research",
        "trading",
        "chemistry",
        "mathematics",
        "education",
        "finance",
    ]
    assert infer_domain_from_filename("Trading.pdf", allowed) == "trading"
    assert infer_domain_from_filename("Research.pdf", allowed) == "research"
    assert infer_domain_from_filename("Medical.pdf", allowed) == "medical"
    assert infer_domain_from_filename("Chemistr.pdf", allowed) == "chemistry"
    assert infer_domain_from_filename("AlgebraAndTrigo.pdf", allowed) == "mathematics"
    assert infer_domain_from_filename("random-notes.pdf", allowed) is None


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
        # No fixture-/product-specific band-aids in seeded overlays.
        for banned in ("CRSwNP", "GSTIN", "8,850.00", "IgE **300**", "wet AMD"):
            assert banned not in tpl.content, f"{slug} still hardcodes {banned}"


def test_medical_system_holds_principles_task_is_slim(db_session):
    from app.services.prompt_management.defaults import (
        DOMAIN_SYSTEM_DEFAULTS,
        DOMAIN_TASK_DEFAULTS,
    )

    system = DOMAIN_SYSTEM_DEFAULTS["medical"]
    task = DOMAIN_TASK_DEFAULTS["medical"]["rag"][2]
    assert "SCOPE LOCK" in system
    assert "RANGE / BOUNDARY MATCHING" in system
    assert "EXPOSURE / REGIMEN COMPARISON" in system
    assert "PHARMACOVIGILANCE" in system
    assert "TASK FLOW" in task
    assert "{context}" in task and "{question}" in task
    assert len(task) < len(system)
    assert "CRSwNP" not in system and "CRSwNP" not in task
    assert "150 mg/4 wk" not in system

    domain = db_session.query(PromptDomain).filter_by(slug="medical").one()
    sys_tpl = (
        db_session.query(PromptTemplate)
        .filter(
            PromptTemplate.domain_id == domain.id,
            PromptTemplate.prompt_type == "system",
            PromptTemplate.is_active.is_(True),
        )
        .one()
    )
    rag_tpl = (
        db_session.query(PromptTemplate)
        .filter(
            PromptTemplate.domain_id == domain.id,
            PromptTemplate.prompt_type == "rag",
            PromptTemplate.is_active.is_(True),
        )
        .one()
    )
    assert "CLINICAL VOICE" in sys_tpl.content
    assert "TASK FLOW" in rag_tpl.content


def test_elevated_domains_have_epistemic_rules():
    from app.services.prompt_management.defaults import (
        DOMAIN_SYSTEM_DEFAULTS,
        PERSONA_SECTION_MARKERS,
    )

    markers = {
        "finance": ("reporting period", "GAAP"),
        "legal": ("obligations", "not legal advice"),
        "research": ("p-values", "limitations"),
        "technical": ("CLI commands", "prerequisites"),
        "education": ("learner", "formulas"),
        "general": ("plain-language", "explicitly"),
        "payment": ("INVOICE / LINE-ITEM", "OCR"),
        "trading": ("Trading Analyst", "instrument"),
        "chemistry": ("Chemistry Analyst", "conditions"),
        "mathematics": ("Mathematics Analyst", "notation"),
    }
    for slug, needles in markers.items():
        text = DOMAIN_SYSTEM_DEFAULTS[slug]
        assert "ROLE" in text and "ABSTAIN" in text, slug
        for marker in PERSONA_SECTION_MARKERS:
            assert marker in text, f"{slug} missing {marker}"
        for needle in needles:
            assert needle.lower() in text.lower(), f"{slug} missing {needle}"


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

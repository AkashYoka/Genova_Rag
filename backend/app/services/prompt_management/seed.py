"""Seed built-in domains and default prompt layers (idempotent; never clobber custom)."""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.models import PromptDomain, PromptTemplate, PromptVersion
from app.services.prompt_management.defaults import (
    DOMAIN_META,
    DOMAIN_SYSTEM_DEFAULTS,
    DOMAIN_TASK_DEFAULTS,
    PERSONA_SECTION_MARKERS,
    domain_overlay_name,
    global_system_default,
    global_task_defaults,
)

logger = logging.getLogger(__name__)

BUILTIN_ORDER = [
    ("general", True, 0),
    ("medical", False, 10),
    ("payment", False, 20),
    ("education", False, 30),
    ("legal", False, 40),
    ("research", False, 50),
    ("finance", False, 60),
    ("trading", False, 65),
    ("technical", False, 70),
    ("chemistry", False, 80),
    ("mathematics", False, 90),
]

# Domains whose stock overlays should elevate to the eight-section persona layout.
_PERSONA_UPGRADE_SLUGS = frozenset(
    {
        "general",
        "medical",
        "payment",
        "education",
        "legal",
        "research",
        "finance",
        "trading",
        "technical",
        "chemistry",
        "mathematics",
    }
)


def _snapshot(db: Session, template: PromptTemplate, note: str, user_id: Optional[str] = None) -> None:
    db.add(
        PromptVersion(
            template_id=template.id,
            version_number=template.version,
            content=template.content,
            change_note=note,
            created_by=user_id,
        )
    )


_MEDICAL_PV_MARKER = "PHARMACOVIGILANCE / PRESCRIBING CASE ASSESSMENT"
_MEDICAL_VOICE_MARKER = "CLINICAL VOICE"
_MEDICAL_SIBLING_MARKER = "SIBLING TABLE / INDICATION DISAMBIGUATION"
_MEDICAL_BOUNDARY_MARKER = "RANGE / BOUNDARY MATCHING"
_PAYMENT_INVOICE_MARKER = "INVOICE / LINE-ITEM ACCURACY"


def _looks_like_stock_short_overlay(content: str) -> bool:
    """True for original brief DOMAIN FOCUS stubs (safe to upgrade once).

    Also upgrades medium-length pre-principle stubs that still lack epistemic
    framing (finance/legal/research/etc. elevations).
    """
    text = (content or "").strip()
    if not text.startswith("DOMAIN FOCUS"):
        return False
    if len(text) < 450:
        return True
    if "Epistemic rules:" not in text and "CLINICAL VOICE" not in text:
        return len(text) < 1600
    return False


def _should_upgrade_medical_overlay(content: str) -> bool:
    """Upgrade stock medical overlay when it lacks PV or clinician-voice guidance.

    Only upgrades content that still looks like the built-in DOMAIN FOCUS medical
    stub (starts with the stock header and was not clearly customized).
    """
    text = (content or "").strip()
    if not text.startswith("DOMAIN FOCUS — Medical"):
        return False
    # Heuristic: customized prompts usually remove or rewrite the stock header
    # behaviour bullets; require the stock "Never invent clinical guidance" line.
    if "Never invent clinical guidance" not in text:
        return False
    if _MEDICAL_PV_MARKER not in text:
        return True
    if _MEDICAL_VOICE_MARKER not in text:
        return True
    if _MEDICAL_SIBLING_MARKER not in text:
        return True
    if _MEDICAL_BOUNDARY_MARKER not in text:
        return True
    # Drop fixture-hardcoded overlays (product/band examples) in favor of principles.
    if any(
        marker in text
        for marker in (
            "CRSwNP",
            "IgE **300**",
            "150 mg/4 wk",
            "wet AMD",
            "Table 1 style",
        )
    ):
        return True
    # Upgrade when PV taxonomy still softens regimen off-label into mere "deviation".
    if (
        "Off-label use: Yes when" not in text
        and "prefer “intentional deviation from labeled dosing”" in text
    ):
        return True
    if "non-recommended dose" not in text and "PHARMACOVIGILANCE" in text:
        return True
    return False


def _should_upgrade_payment_overlay(content: str) -> bool:
    text = (content or "").strip()
    if not text.startswith("DOMAIN FOCUS — Payment"):
        return False
    if _PAYMENT_INVOICE_MARKER not in text:
        return True
    # Drop country-/fixture-specific invoice examples.
    if any(marker in text for marker in ("GSTIN", "CGST/SGST", "8,850.00", "HSN")):
        return True
    if _missing_persona_sections(text):
        return True
    return False


def _missing_persona_sections(content: str) -> bool:
    """True when most of the eight persona section headers are absent."""
    text = content or ""
    present = sum(1 for marker in PERSONA_SECTION_MARKERS if marker in text)
    return present < 6


def _should_upgrade_persona_overlay(slug: str, content: str) -> bool:
    """Elevate stock DOMAIN FOCUS overlays to the eight-section persona layout.

    Never upgrades content that no longer looks like a built-in DOMAIN FOCUS stub.
    """
    if slug not in _PERSONA_UPGRADE_SLUGS:
        return False
    text = (content or "").strip()
    if not text.startswith("DOMAIN FOCUS"):
        return False
    # Customized prompts usually drop the stock header framing.
    if slug == "medical" and "Never invent clinical guidance" not in text:
        return False
    return _missing_persona_sections(text)


def ensure_default_prompts(db: Session, *, upgrade_stock_overlays: bool = True) -> None:
    """Idempotently create built-in domains and default active prompts."""
    domain_by_slug: dict[str, PromptDomain] = {}
    for slug, is_fallback, sort_order in BUILTIN_ORDER:
        meta = DOMAIN_META[slug]
        domain = db.query(PromptDomain).filter(PromptDomain.slug == slug).first()
        if not domain:
            domain = PromptDomain(
                slug=slug,
                name=meta["name"],
                description=meta["description"],
                is_active=True,
                is_builtin=True,
                is_fallback=is_fallback,
                sort_order=sort_order,
            )
            db.add(domain)
            db.flush()
            logger.info("Seeded prompt domain '%s'", slug)
        else:
            if is_fallback and not domain.is_fallback:
                domain.is_fallback = True
            # Refresh marketing copy for builtins if description empty.
            if domain.is_builtin and not (domain.description or "").strip():
                domain.description = meta["description"]
            if domain.is_builtin and domain.name != meta["name"] and not domain.description:
                domain.name = meta["name"]
        domain_by_slug[slug] = domain

    def upsert_template(
        *,
        domain: Optional[PromptDomain],
        prompt_type: str,
        name: str,
        content: str,
        description: str = "",
        is_default: bool = True,
        upgrade_if_short: bool = False,
    ) -> PromptTemplate:
        q = db.query(PromptTemplate).filter(
            PromptTemplate.prompt_type == prompt_type,
            PromptTemplate.name == name,
        )
        if domain is None:
            q = q.filter(PromptTemplate.domain_id.is_(None))
        else:
            q = q.filter(PromptTemplate.domain_id == domain.id)
        existing = q.first()
        if existing:
            if (
                upgrade_if_short
                and upgrade_stock_overlays
                and _looks_like_stock_short_overlay(existing.content)
                and content.strip()
                and existing.content.strip() != content.strip()
            ):
                existing.content = content
                existing.description = description or existing.description
                existing.version = int(existing.version or 1) + 1
                existing.is_active = True
                existing.is_default = True
                _snapshot(db, existing, "V1 default template upgrade")
                logger.info("Upgraded stock prompt '%s' (%s)", name, prompt_type)
            return existing
        template = PromptTemplate(
            domain_id=domain.id if domain else None,
            prompt_type=prompt_type,
            name=name,
            content=content,
            description=description,
            version=1,
            is_active=True,
            is_default=is_default,
        )
        db.add(template)
        db.flush()
        _snapshot(db, template, "Initial V1 default")
        return template

    # Also match legacy overlay names from the first seed pass.
    for slug, domain in domain_by_slug.items():
        content = DOMAIN_SYSTEM_DEFAULTS[slug]
        modern_name = domain_overlay_name(domain.name)
        legacy_name = f"{domain.name} Domain Overlay"
        legacy = (
            db.query(PromptTemplate)
            .filter(
                PromptTemplate.domain_id == domain.id,
                PromptTemplate.prompt_type == "system",
                PromptTemplate.name == legacy_name,
            )
            .first()
        )
        if legacy and legacy.name != modern_name:
            legacy.name = modern_name
            if _looks_like_stock_short_overlay(legacy.content):
                legacy.content = content
                legacy.description = DOMAIN_META[slug]["purpose"]
                legacy.version = int(legacy.version or 1) + 1
                _snapshot(db, legacy, "V1 default template upgrade")
            continue
        existing_overlay = (
            db.query(PromptTemplate)
            .filter(
                PromptTemplate.domain_id == domain.id,
                PromptTemplate.prompt_type == "system",
                PromptTemplate.name == modern_name,
            )
            .first()
        )
        if (
            slug == "medical"
            and existing_overlay
            and (
                _should_upgrade_medical_overlay(existing_overlay.content)
                or _should_upgrade_persona_overlay(slug, existing_overlay.content)
            )
            and content.strip()
            and existing_overlay.content.strip() != content.strip()
        ):
            existing_overlay.content = content
            existing_overlay.description = DOMAIN_META[slug]["purpose"]
            existing_overlay.version = int(existing_overlay.version or 1) + 1
            existing_overlay.is_active = True
            existing_overlay.is_default = True
            _snapshot(db, existing_overlay, "Medical clinician-voice overlay upgrade")
            logger.info("Upgraded medical domain overlay with clinician voice / PV assessment")
            continue
        if (
            slug == "payment"
            and existing_overlay
            and existing_overlay.is_default
            and _should_upgrade_payment_overlay(existing_overlay.content)
            and content.strip()
            and existing_overlay.content.strip() != content.strip()
        ):
            existing_overlay.content = content
            existing_overlay.description = DOMAIN_META[slug]["purpose"]
            existing_overlay.version = int(existing_overlay.version or 1) + 1
            existing_overlay.is_active = True
            existing_overlay.is_default = True
            _snapshot(db, existing_overlay, "Payment invoice line-item overlay upgrade")
            logger.info("Upgraded payment domain overlay with invoice line-item rules")
            continue
        if (
            existing_overlay
            and existing_overlay.is_default
            and _should_upgrade_persona_overlay(slug, existing_overlay.content)
            and content.strip()
            and existing_overlay.content.strip() != content.strip()
        ):
            existing_overlay.content = content
            existing_overlay.description = DOMAIN_META[slug]["purpose"]
            existing_overlay.version = int(existing_overlay.version or 1) + 1
            existing_overlay.is_active = True
            existing_overlay.is_default = True
            _snapshot(db, existing_overlay, "Persona analyst overlay upgrade")
            logger.info("Upgraded %s domain overlay to persona analyst layout", slug)
            continue
        upsert_template(
            domain=domain,
            prompt_type="system",
            name=modern_name,
            content=content,
            description=DOMAIN_META[slug]["purpose"],
            upgrade_if_short=True,
        )

    upsert_template(
        domain=None,
        prompt_type="system",
        name="Global Base System",
        content=global_system_default(),
        description="Shared citation, grounding, security, and formatting rules for every domain.",
    )
    for prompt_type, (name, description, content) in global_task_defaults().items():
        upsert_template(
            domain=None,
            prompt_type=prompt_type,
            name=name,
            content=content,
            description=description,
        )

    # Domain-specific task prompts (e.g. medical PV case RAG).
    for slug, tasks in DOMAIN_TASK_DEFAULTS.items():
        domain = domain_by_slug.get(slug)
        if not domain:
            continue
        for prompt_type, (name, description, content) in tasks.items():
            if not (content or "").strip():
                continue
            existing = (
                db.query(PromptTemplate)
                .filter(
                    PromptTemplate.domain_id == domain.id,
                    PromptTemplate.prompt_type == prompt_type,
                    PromptTemplate.name == name,
                )
                .first()
            )
            if existing:
                # Upgrade stock medical RAG/QA when missing PV block or clinician voice.
                content_text = existing.content or ""
                needs_medical_task_upgrade = (
                    slug == "medical"
                    and prompt_type in {"rag", "qa"}
                    and existing.is_default
                    and (
                        "TASK FLOW" not in content_text
                        or "CRSwNP" in content_text
                        or "IgE 300" in content_text
                        or "150 vs 600 mg" in content_text
                        or "Off-label use = wrong indication" not in content_text
                    )
                )
                if needs_medical_task_upgrade:
                    existing.content = content
                    existing.description = description
                    existing.version = int(existing.version or 1) + 1
                    existing.is_active = True
                    existing.is_default = True
                    _snapshot(db, existing, "Medical clinician-voice RAG task upgrade")
                    logger.info("Upgraded medical domain %s task prompt", prompt_type)
                continue
            upsert_template(
                domain=domain,
                prompt_type=prompt_type,
                name=name,
                content=content,
                description=description,
                is_default=True,
            )

    db.flush()


def get_canonical_domain_system(slug: str) -> Optional[str]:
    return DOMAIN_SYSTEM_DEFAULTS.get(slug)


def get_canonical_global_system() -> str:
    return global_system_default()

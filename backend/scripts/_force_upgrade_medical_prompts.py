"""Force-upgrade medical overlay + RAG/QA task prompts from defaults."""
from __future__ import annotations

from app.database import SessionLocal
from app.services.prompt_management.seed import ensure_default_prompts


def main() -> None:
    db = SessionLocal()
    try:
        ensure_default_prompts(db, upgrade_stock_overlays=True)
        db.commit()
        print("ensure_default_prompts committed")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # Verify markers landed.
    from app.models import PromptDomain, PromptTemplate

    db = SessionLocal()
    try:
        medical = db.query(PromptDomain).filter(PromptDomain.slug == "medical").one()
        rows = (
            db.query(PromptTemplate)
            .filter(
                PromptTemplate.domain_id == medical.id,
                PromptTemplate.is_active.is_(True),
            )
            .all()
        )
        for t in rows:
            c = t.content or ""
            print(
                t.prompt_type,
                t.name,
                "v",
                t.version,
                "SIBLING",
                "SIBLING TABLE / INDICATION DISAMBIGUATION" in c
                or "SCOPE LOCK" in c,
                "PV",
                "PHARMACOVIGILANCE" in c,
                "BOUNDARY",
                "RANGE / BOUNDARY MATCHING" in c,
                "TASK_FLOW",
                "TASK FLOW" in c,
                "HARDCODE",
                any(x in c for x in ("CRSwNP", "GSTIN", "IgE 300", "8,850")),
                "len",
                len(c),
            )

        # Spot-check elevated overlays for principle markers / no hardcoding.
        for slug in (
            "payment",
            "finance",
            "legal",
            "research",
            "technical",
            "education",
            "general",
        ):
            d = db.query(PromptDomain).filter(PromptDomain.slug == slug).one()
            ov = (
                db.query(PromptTemplate)
                .filter(
                    PromptTemplate.domain_id == d.id,
                    PromptTemplate.prompt_type == "system",
                    PromptTemplate.is_active.is_(True),
                )
                .one()
            )
            c = ov.content or ""
            print(
                "overlay",
                slug,
                "v",
                ov.version,
                "EPISTEMIC",
                "Epistemic rules:" in c or "INVOICE / LINE-ITEM" in c,
                "HARDCODE",
                any(x in c for x in ("CRSwNP", "GSTIN", "8,850", "HSN")),
                "len",
                len(c),
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()

"""Push updated medical domain prompts from defaults into the active DB templates."""
from __future__ import annotations

from app.database import SessionLocal
from app.models import PromptDomain, PromptTemplate
from app.services.prompt_management.defaults import (
    DOMAIN_SYSTEM_DEFAULTS,
    DOMAIN_TASK_DEFAULTS,
)


def main() -> None:
    db = SessionLocal()
    try:
        domain = (
            db.query(PromptDomain)
            .filter(PromptDomain.slug == "medical", PromptDomain.is_active.is_(True))
            .one()
        )
        # Medical domain overlay is stored as prompt_type="system".
        overlay = (
            db.query(PromptTemplate)
            .filter(
                PromptTemplate.domain_id == domain.id,
                PromptTemplate.prompt_type == "system",
                PromptTemplate.is_active.is_(True),
            )
            .order_by(PromptTemplate.is_default.desc(), PromptTemplate.updated_at.desc())
            .first()
        )
        medical_system = DOMAIN_SYSTEM_DEFAULTS["medical"]
        if overlay:
            overlay.content = medical_system
            print("updated medical system overlay", overlay.id, "v", overlay.version)
        else:
            print("WARNING: no medical system template found")

        medical_rag = DOMAIN_TASK_DEFAULTS["medical"]["rag"][2]
        for ptype in ("rag", "qa"):
            tmpl = (
                db.query(PromptTemplate)
                .filter(
                    PromptTemplate.domain_id == domain.id,
                    PromptTemplate.prompt_type == ptype,
                    PromptTemplate.is_active.is_(True),
                )
                .order_by(PromptTemplate.is_default.desc(), PromptTemplate.updated_at.desc())
                .first()
            )
            if tmpl:
                tmpl.content = medical_rag
                print(f"updated medical {ptype}", tmpl.id, "v", tmpl.version)
            else:
                print(f"WARNING: no medical {ptype} template found")

        # Global system fallback (non-domain)
        global_system = (
            db.query(PromptTemplate)
            .filter(
                PromptTemplate.domain_id.is_(None),
                PromptTemplate.prompt_type == "system",
                PromptTemplate.is_active.is_(True),
            )
            .order_by(PromptTemplate.is_default.desc(), PromptTemplate.updated_at.desc())
            .first()
        )
        if global_system:
            from app.rag.prompts import prompts as fallback

            global_system.content = fallback.SYSTEM_PROMPT
            print("updated global system", global_system.id)

        global_rag = (
            db.query(PromptTemplate)
            .filter(
                PromptTemplate.domain_id.is_(None),
                PromptTemplate.prompt_type == "rag",
                PromptTemplate.is_active.is_(True),
            )
            .order_by(PromptTemplate.is_default.desc(), PromptTemplate.updated_at.desc())
            .first()
        )
        if global_rag:
            from app.rag.prompts import prompts as fallback

            global_rag.content = fallback.RAG_PROMPT_TEMPLATE
            print("updated global rag", global_rag.id)

        db.commit()
        print("committed")
    finally:
        db.close()


if __name__ == "__main__":
    main()

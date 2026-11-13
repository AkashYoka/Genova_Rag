"""Dump active medical prompts and document domains."""
from __future__ import annotations

from sqlalchemy.orm import joinedload

from app.database import SessionLocal
from app.models import Document, PromptDomain, PromptTemplate


def main() -> None:
    db = SessionLocal()
    try:
        print("=== DOMAINS ===")
        for d in db.query(PromptDomain).all():
            print(d.id, d.slug, d.name, "active", d.is_active, "builtin", d.is_builtin)

        print("\n=== TEMPLATES ===")
        templates = (
            db.query(PromptTemplate)
            .options(joinedload(PromptTemplate.versions))
            .all()
        )
        for t in templates:
            active = next((v for v in (t.versions or []) if v.is_active), None)
            content = (active.content if active else "") or ""
            print(
                f"type={t.prompt_type} name={t.name!r} domain_id={t.domain_id} "
                f"active_ver={getattr(active, 'version_number', None)} len={len(content)}"
            )
            markers = [
                "PHARMACOVIGILANCE",
                "CLINICAL VOICE",
                "DOMAIN FOCUS — Medical",
                "Based on the provided",
                "TABLE LOOKUP",
                "CURRENT-QUESTION ONLY",
                "SIBLING TABLE",
                "indication-matching",
            ]
            print(" markers", {m: (m in content) for m in markers})
            if t.prompt_type in ("rag", "qa", "overlay", "system") and (
                t.domain_id or "medical" in (t.name or "").lower()
            ):
                print(" HEAD:", content[:320].replace("\n", " | "))

        print("\n=== DOC DOMAINS ===")
        for name in ("xolair", "lucentis"):
            rows = (
                db.query(Document)
                .filter(
                    Document.original_name.ilike(f"%{name}%"),
                    Document.is_deleted.is_(False),
                )
                .all()
            )
            for r in rows:
                print(r.id, r.original_name, "domain=", r.domain, "status=", r.status)
    finally:
        db.close()


if __name__ == "__main__":
    main()

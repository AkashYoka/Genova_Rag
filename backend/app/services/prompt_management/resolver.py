"""Resolve and compose Global Base + Domain Overlay + Task prompts."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.models import PromptDomain, PromptExecutionLog, PromptTemplate
from app.rag.prompts import prompts as fallback
from app.services.prompt_management.detector import (
    DomainDetectionResult,
    detect_domain,
)

logger = logging.getLogger(__name__)

TASK_ALIASES = {
    "question_answering": "rag",
    "qa": "rag",
    "ask": "rag",
}


@dataclass
class ResolvedPrompts:
    domain: str
    confidence: float
    was_fallback: bool
    system_prompt: str
    task_prompt: str
    prompt_type: str
    system_template_id: Optional[str] = None
    system_version: Optional[int] = None
    domain_template_id: Optional[str] = None
    domain_version: Optional[int] = None
    task_template_id: Optional[str] = None
    task_version: Optional[int] = None
    secondary_domains: Optional[List[str]] = None
    rationale: str = ""

    def format_task(self, *, context: str, question: str) -> str:
        try:
            return self.task_prompt.format(context=context, question=question)
        except Exception:
            # Templates may omit placeholders (e.g. agent / comparison).
            return (
                self.task_prompt.replace("{context}", context).replace(
                    "{question}", question
                )
            )


def _pick_template(
    db: Session,
    *,
    prompt_type: str,
    domain_id: Optional[str] = None,
) -> Optional[PromptTemplate]:
    q = db.query(PromptTemplate).filter(
        PromptTemplate.prompt_type == prompt_type,
        PromptTemplate.is_active.is_(True),
    )
    if domain_id is None:
        q = q.filter(PromptTemplate.domain_id.is_(None))
    else:
        q = q.filter(PromptTemplate.domain_id == domain_id)
    return (
        q.order_by(PromptTemplate.is_default.desc(), PromptTemplate.updated_at.desc())
        .first()
    )


def _fallback_content(prompt_type: str) -> str:
    mapping = {
        "system": fallback.SYSTEM_PROMPT,
        "rag": fallback.RAG_PROMPT_TEMPLATE,
        "qa": fallback.RAG_PROMPT_TEMPLATE,
        "greeting": fallback.GREETING_PROMPT,
        "agent": fallback.AGENT_SYSTEM_PROMPT,
        "comparison": fallback.MULTI_DOC_COMPARISON_GUIDANCE.strip(),
    }
    return mapping.get(prompt_type) or fallback.RAG_PROMPT_TEMPLATE


def resolve_prompts(
    db: Session,
    *,
    question: str,
    prompt_type: str = "rag",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    document_ids: Optional[Sequence[str]] = None,
    retrieved_context: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    llm_provider: Optional[str] = None,
    detection: Optional[DomainDetectionResult] = None,
    persist_execution: bool = True,
) -> ResolvedPrompts:
    """Detect domain (unless provided) and assemble layered prompts."""
    task = TASK_ALIASES.get((prompt_type or "rag").lower(), (prompt_type or "rag").lower())

    detection = detection or detect_domain(
        db,
        question=question,
        user_id=user_id,
        session_id=session_id,
        document_ids=document_ids,
        retrieved_context=retrieved_context,
        chat_history=chat_history,
        llm_provider=llm_provider,
        persist=True,
    )

    domain_row = (
        db.query(PromptDomain)
        .filter(PromptDomain.slug == detection.domain, PromptDomain.is_active.is_(True))
        .first()
    )

    global_system = _pick_template(db, prompt_type="system", domain_id=None)
    domain_overlay = (
        _pick_template(db, prompt_type="system", domain_id=domain_row.id)
        if domain_row
        else None
    )
    task_tpl = _pick_template(db, prompt_type=task, domain_id=domain_row.id if domain_row else None)
    if task_tpl is None:
        task_tpl = _pick_template(db, prompt_type=task, domain_id=None)

    system_parts = []
    if global_system:
        system_parts.append(global_system.content)
    else:
        system_parts.append(_fallback_content("system"))
    if domain_overlay and domain_overlay.content.strip():
        system_parts.append(domain_overlay.content.strip())
    # Mixed-domain hint when secondary domains reported with decent confidence.
    if detection.secondary_domains and detection.confidence < 0.85:
        system_parts.append(
            "MIXED-DOMAIN NOTE: Secondary signals were "
            + ", ".join(detection.secondary_domains)
            + ". Prefer evidence from the retrieved context; do not force a single-domain framing."
        )

    task_content = task_tpl.content if task_tpl else _fallback_content(task)

    resolved = ResolvedPrompts(
        domain=detection.domain,
        confidence=float(detection.confidence),
        was_fallback=bool(detection.was_fallback),
        system_prompt="\n\n".join(system_parts),
        task_prompt=task_content,
        prompt_type=task,
        system_template_id=str(global_system.id) if global_system else None,
        system_version=global_system.version if global_system else None,
        domain_template_id=str(domain_overlay.id) if domain_overlay else None,
        domain_version=domain_overlay.version if domain_overlay else None,
        task_template_id=str(task_tpl.id) if task_tpl else None,
        task_version=task_tpl.version if task_tpl else None,
        secondary_domains=list(detection.secondary_domains or []),
        rationale=detection.rationale or "",
    )

    if persist_execution:
        try:
            db.add(
                PromptExecutionLog(
                    user_id=user_id,
                    session_id=session_id,
                    domain_slug=resolved.domain,
                    confidence=resolved.confidence,
                    prompt_type=resolved.prompt_type,
                    system_template_id=resolved.system_template_id,
                    system_version=resolved.system_version,
                    task_template_id=resolved.task_template_id,
                    task_version=resolved.task_version,
                    domain_template_id=resolved.domain_template_id,
                    domain_version=resolved.domain_version,
                )
            )
            db.flush()
        except Exception as exc:
            logger.debug("Could not persist prompt execution log: %s", exc)

    return resolved


def infer_task_type(question: str, default: str = "rag") -> str:
    """Lightweight task routing from the user question (no LLM)."""
    q = (question or "").lower()
    if any(w in q for w in ("summarize", "summary", "tldr", "tl;dr")):
        return "summarization"
    if any(w in q for w in ("compare", "difference", "vs ", "versus")):
        return "comparison"
    if any(w in q for w in ("extract", "list all", "pull out", "what are the fields")):
        return "extraction"
    if any(w in q for w in ("explain", "what does", "how does", "eli5", "like i'm 5")):
        return "explanation"
    return default

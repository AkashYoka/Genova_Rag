"""
Graph retrieval for GraphRAG: match the question to entities, expand a bounded
neighbourhood, and turn evidence-backed relationships into prompt context plus
citable sources.

Traversal is always bounded (depth ≤ 3, capped node/edge counts) and never
leaves the user's own graph or the requested document scope.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.config import get_settings
from app.graph import store

logger = logging.getLogger(__name__)
settings = get_settings()

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]+")
_STOPWORDS = {
    "the", "and", "for", "with", "what", "which", "who", "whom", "how", "why",
    "does", "did", "are", "was", "were", "has", "have", "had", "from", "that",
    "this", "these", "those", "between", "into", "about", "can", "could",
    "should", "would", "will", "you", "your", "document", "documents", "pdf",
    "according", "tell", "explain", "describe", "give", "show", "list", "there",
}


@dataclass
class GraphEvidence:
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    seed_entity_ids: List[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.relationships and not self.entities


def query_terms(query: str) -> List[str]:
    terms = []
    for token in _TOKEN.findall((query or "").casefold()):
        cleaned = token.strip("-._")
        if len(cleaned) < 3 or cleaned in _STOPWORDS or cleaned.isdigit():
            continue
        if cleaned not in terms:
            terms.append(cleaned)
    return terms


def _score_entity(entity: Dict[str, Any], terms: Sequence[str], normalized_query: str) -> int:
    name = (entity.get("canonical_name") or "").casefold()
    normalized_name = (entity.get("normalized_name") or name).strip()
    aliases = [alias.casefold() for alias in entity.get("aliases") or []]
    description = (entity.get("description") or "").casefold()
    entity_type = (entity.get("entity_type") or "").casefold()

    score = 0
    if normalized_name and normalized_name in normalized_query:
        score += 12
    if entity_type and entity_type.replace("_", " ") in normalized_query:
        score += 3
    for term in terms:
        if term in normalized_name.split():
            score += 6
        elif term in normalized_name:
            score += 4
        if any(term in alias for alias in aliases):
            score += 5
        if term in description:
            score += 1
    if entity.get("mention_count"):
        score += min(int(entity["mention_count"]), 3)
    return score


def match_query_entities(
    db: Session,
    user_id: Any,
    query: str,
    *,
    document_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return stored entities that the question is plausibly about."""
    limit = limit or settings.GRAPH_MAX_QUERY_ENTITIES
    terms = query_terms(query)
    if not terms:
        return []

    candidates = store.search_entities(db, user_id, limit=400)
    if document_ids:
        allowed = {
            entity["id"]
            for entity in store.search_entities(
                db, user_id, document_ids=document_ids, limit=400
            )
        }
        candidates = [entity for entity in candidates if entity["id"] in allowed]

    normalized_query = " ".join(terms)
    scored = [
        (_score_entity(entity, terms, normalized_query), entity)
        for entity in candidates
    ]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: (-item[0], -int(item[1].get("mention_count") or 0)))
    return [entity for _, entity in scored[: max(1, limit)]]


def _source_from_evidence(evidence: Dict[str, Any], relationship: Dict[str, Any]) -> Dict[str, Any]:
    text = evidence.get("evidence") or ""
    confidence = float(evidence.get("confidence") or relationship.get("confidence") or 0.0)
    return {
        "text": text[:300] + ("..." if len(text) > 300 else ""),
        "filename": evidence.get("document_name") or "document",
        "page": int(evidence.get("page") or 0),
        "score": round(confidence, 4),
        "confidence": round(min(1.0, max(0.0, confidence)) * 100, 1),
        "bbox": "",
    }


def _format_relationship_line(relationship: Dict[str, Any]) -> str:
    source = relationship.get("source") or {}
    target = relationship.get("target") or {}
    label = (relationship.get("relation_type") or "related_to").replace("_", " ")
    return (
        f"- {source.get('name') or '?'} —{label}→ {target.get('name') or '?'} "
        f"(confidence: {float(relationship.get('confidence') or 0.0):.2f})"
    )


def format_context(evidence: GraphEvidence) -> str:
    """Render graph evidence as a compact, citation-bearing prompt block."""
    if evidence.is_empty:
        return ""

    lines = [
        "## Knowledge Graph Context",
        "Relationships extracted from the user's documents. Use them only as "
        "supporting evidence; cite the document and page given with each item.",
        "",
    ]

    if evidence.relationships:
        lines.append("### Evidence-backed relationships")
        for relationship in evidence.relationships:
            lines.append(_format_relationship_line(relationship))
            description = (relationship.get("description") or "").strip()
            if description:
                lines.append(f"  Description: {description}")
            for item in (relationship.get("evidence") or [])[:2]:
                quote = (item.get("evidence") or "").strip()
                if not quote:
                    continue
                page = item.get("page")
                location = f"{item.get('document_name') or 'document'}"
                if page:
                    location += f", page {page}"
                lines.append(f'  Evidence: "{quote}" ({location})')
        lines.append("")

    entity_lines = []
    for entity in evidence.entities[:12]:
        name = entity.get("canonical_name")
        if not name:
            continue
        label = (entity.get("entity_type") or "").replace("_", " ").title()
        description = (entity.get("description") or "").strip()
        entry = f"- {name} ({label})"
        if description:
            entry += f": {description}"
        entity_lines.append(entry)
    if entity_lines:
        lines.append("### Related entities")
        lines.extend(entity_lines)

    return "\n".join(lines).strip()


def retrieve_graph_evidence(
    db: Session,
    user_id: Any,
    query: str,
    *,
    document_ids: Optional[Sequence[str]] = None,
    depth: Optional[int] = None,
    max_entities: Optional[int] = None,
    max_relationships: Optional[int] = None,
    max_sources: Optional[int] = None,
) -> GraphEvidence:
    """Retrieve bounded graph evidence relevant to ``query``."""
    depth = settings.GRAPH_TRAVERSAL_DEPTH if depth is None else depth
    max_relationships = (
        settings.GRAPH_MAX_GRAPH_RELATIONSHIPS
        if max_relationships is None
        else max_relationships
    )
    max_sources = settings.GRAPH_MAX_EVIDENCE_SOURCES if max_sources is None else max_sources

    try:
        seeds = match_query_entities(
            db,
            user_id,
            query,
            document_ids=document_ids,
            limit=max_entities,
        )
    except Exception as exc:
        logger.warning("Graph entity matching failed: %s", exc)
        return GraphEvidence()

    if not seeds:
        return GraphEvidence()

    try:
        subgraph = store.neighborhood(
            db,
            user_id,
            entity_ids=[seed["id"] for seed in seeds],
            depth=depth,
            document_ids=document_ids,
        )
    except Exception as exc:
        logger.warning("Graph neighbourhood retrieval failed: %s", exc)
        return GraphEvidence()

    relationships = list(subgraph.get("edges") or [])[:max_relationships]
    # Rank relationships by confidence, then by whether they are in-document scope.
    relationships.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)

    sources: List[Dict[str, Any]] = []
    seen_sources = set()
    for relationship in relationships:
        for item in relationship.get("evidence") or []:
            key = (
                item.get("document_name"),
                item.get("page"),
                (item.get("evidence") or "")[:80],
            )
            if key in seen_sources:
                continue
            seen_sources.add(key)
            sources.append(_source_from_evidence(item, relationship))
            if len(sources) >= max_sources:
                break
        if len(sources) >= max_sources:
            break

    evidence = GraphEvidence(
        entities=list(subgraph.get("nodes") or []),
        relationships=relationships,
        sources=sources,
        seed_entity_ids=[seed["id"] for seed in seeds],
        truncated=bool(subgraph.get("truncated")),
    )
    evidence.context = format_context(evidence)
    return evidence


def get_entity_context(
    query: str,
    user_id: str,
    document_ids: Optional[Sequence[str]] = None,
    *,
    db: Optional[Session] = None,
) -> str:
    """Convenience wrapper returning only prompt text (empty when unavailable)."""
    owns_session = db is None
    if owns_session:
        from app.database import SessionLocal

        db = SessionLocal()
    try:
        return retrieve_graph_evidence(db, user_id, query, document_ids=document_ids).context
    finally:
        if owns_session and db is not None:
            db.close()


__all__ = [
    "GraphEvidence",
    "format_context",
    "get_entity_context",
    "match_query_entities",
    "query_terms",
    "retrieve_graph_evidence",
]

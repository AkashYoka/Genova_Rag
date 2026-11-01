"""
Relational persistence for the knowledge graph.

Storage choice: the existing SQLAlchemy catalog. The graph needs transactional
writes, provenance rows and cross-document queries — all of which the existing
database already provides — so no external graph database is required. Every
function here takes an explicit ``Session`` so the storage layer stays
replaceable behind the same primitives.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    Document,
    GraphEntity,
    GraphEntityMention,
    GraphIndexStatus,
    GraphRelationship,
    GraphRelationshipEvidence,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ── identifiers ───────────────────────────────────────────────────────────────

def _digest(prefix: str, *parts: str) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:20]}"


def entity_id(user_id: Any, entity_type: str, normalized_name: str) -> str:
    """Stable, backend-generated entity id (never trusted from the model)."""
    return _digest("e", str(user_id), entity_type, normalized_name)


def relationship_id(user_id: Any, source_id: str, target_id: str, relation_type: str) -> str:
    return _digest("r", str(user_id), source_id, target_id, relation_type)


def mention_id(entity_id: str, document_id: Any, chunk_id: str, evidence: str) -> str:
    return _digest("m", entity_id, str(document_id), chunk_id or "", evidence[:120])


def evidence_id(relationship_id: str, document_id: Any, chunk_id: str, evidence: str) -> str:
    return _digest("ev", relationship_id, str(document_id), chunk_id or "", evidence[:120])


# ── serialisation helpers ─────────────────────────────────────────────────────

def entity_aliases(entity: GraphEntity) -> List[str]:
    if not entity.aliases_json:
        return []
    try:
        parsed = json.loads(entity.aliases_json)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]


def entity_row(entity: GraphEntity) -> Dict[str, Any]:
    aliases = entity_aliases(entity)
    return {
        "id": entity.id,
        "canonical_name": entity.canonical_name,
        "normalized_name": entity.normalized_name,
        "entity_type": entity.entity_type,
        "description": entity.description or "",
        "aliases": aliases,
        "mention_count": entity.mention_count or 0,
        "document_count": entity.document_count or 0,
        "confidence": float(entity.confidence or 0.0),
        "possible_duplicate_of": entity.possible_duplicate_of,
    }


def _document_names(db: Session, user_id: Any, document_ids: Iterable[str]) -> Dict[str, str]:
    ids = [str(doc_id) for doc_id in document_ids if doc_id]
    if not ids:
        return {}
    rows = (
        db.query(Document.id, Document.original_name)
        .filter(Document.user_id == user_id, Document.id.in_(ids))
        .all()
    )
    return {str(row[0]): row[1] for row in rows}


# ── entity reads ──────────────────────────────────────────────────────────────

def list_entities(db: Session, user_id: Any, *, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
    query = db.query(GraphEntity).filter(GraphEntity.user_id == user_id)
    if entity_type:
        query = query.filter(GraphEntity.entity_type == entity_type)
    return [entity_row(row) for row in query.all()]


def get_entity(db: Session, user_id: Any, target_entity_id: str) -> Optional[GraphEntity]:
    return (
        db.query(GraphEntity)
        .filter(GraphEntity.id == target_entity_id, GraphEntity.user_id == user_id)
        .first()
    )


def find_entity_by_normalized(
    db: Session, user_id: Any, entity_type: str, normalized_name: str
) -> Optional[GraphEntity]:
    return (
        db.query(GraphEntity)
        .filter(
            GraphEntity.user_id == user_id,
            GraphEntity.entity_type == entity_type,
            GraphEntity.normalized_name == normalized_name,
        )
        .first()
    )


def load_entity_lookup(db: Session, user_id: Any, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Rows shaped for :func:`app.graph.resolution.build_lookup`."""
    return list_entities(db, user_id, entity_type=entity_type)


# ── entity writes ─────────────────────────────────────────────────────────────

def create_entity(
    db: Session,
    user_id: Any,
    *,
    name: str,
    entity_type: str,
    normalized_name: str,
    description: str = "",
    aliases: Optional[Sequence[str]] = None,
    confidence: float = 0.0,
    possible_duplicate_of: Optional[str] = None,
) -> GraphEntity:
    entity = GraphEntity(
        id=entity_id(user_id, entity_type, normalized_name),
        user_id=user_id,
        canonical_name=name,
        normalized_name=normalized_name,
        entity_type=entity_type,
        description=description or None,
        aliases_json=json.dumps(list(aliases or [])) if aliases else None,
        mention_count=0,
        document_count=0,
        confidence=confidence,
        possible_duplicate_of=possible_duplicate_of,
    )
    db.add(entity)
    db.flush()
    return entity


def merge_entity_metadata(
    db: Session,
    entity: GraphEntity,
    *,
    description: str = "",
    aliases: Optional[Sequence[str]] = None,
    confidence: float = 0.0,
) -> None:
    """Enrich an existing entity without overwriting better evidence."""
    if description and len(description) > len(entity.description or ""):
        entity.description = description
    if aliases:
        current = entity_aliases(entity)
        lowered = {alias.casefold() for alias in current}
        if entity.canonical_name.casefold() in lowered:
            lowered.discard(entity.canonical_name.casefold())
        added = False
        for alias in aliases:
            alias = (alias or "").strip()
            if alias and alias.casefold() not in lowered:
                current.append(alias)
                lowered.add(alias.casefold())
                added = True
        if added:
            entity.aliases_json = json.dumps(current)
    if confidence > float(entity.confidence or 0.0):
        entity.confidence = confidence
    entity.updated_at = datetime.now(timezone.utc)


def add_mention(
    db: Session,
    *,
    target_entity_id: str,
    user_id: Any,
    document_id: Any,
    chunk_id: str = "",
    page: Optional[int] = None,
    section: str = "",
    evidence_text: str = "",
    confidence: float = 0.0,
    extraction_method: str = "llm",
) -> bool:
    """Record one evidence mention; returns False when it already exists."""
    new_id = mention_id(target_entity_id, document_id, chunk_id, evidence_text)
    exists = db.query(GraphEntityMention.id).filter(GraphEntityMention.id == new_id).first()
    if exists:
        return False
    db.add(
        GraphEntityMention(
            id=new_id,
            entity_id=target_entity_id,
            user_id=user_id,
            document_id=document_id,
            chunk_id=chunk_id or None,
            page=page,
            section=section or None,
            evidence_text=evidence_text or None,
            confidence=confidence,
            extraction_method=extraction_method,
        )
    )
    return True


# ── relationship writes ───────────────────────────────────────────────────────

def upsert_relationship(
    db: Session,
    user_id: Any,
    *,
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    description: str = "",
    confidence: float = 0.0,
    extraction_method: str = "llm",
) -> Tuple[GraphRelationship, bool]:
    rid = relationship_id(user_id, source_entity_id, target_entity_id, relation_type)
    existing = db.query(GraphRelationship).filter(GraphRelationship.id == rid).first()
    if existing is not None:
        if confidence > float(existing.confidence or 0.0):
            existing.confidence = confidence
        if description and not existing.description:
            existing.description = description
        return existing, False

    relationship = GraphRelationship(
        id=rid,
        user_id=user_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        description=description or None,
        confidence=confidence,
        extraction_method=extraction_method,
    )
    db.add(relationship)
    db.flush()
    return relationship, True


def add_relationship_evidence(
    db: Session,
    *,
    relationship_id: str,
    user_id: Any,
    document_id: Any,
    chunk_id: str = "",
    page: Optional[int] = None,
    section: str = "",
    evidence_text: str = "",
    confidence: float = 0.0,
) -> bool:
    new_id = evidence_id(relationship_id, document_id, chunk_id, evidence_text)
    exists = (
        db.query(GraphRelationshipEvidence.id)
        .filter(GraphRelationshipEvidence.id == new_id)
        .first()
    )
    if exists:
        return False
    db.add(
        GraphRelationshipEvidence(
            id=new_id,
            relationship_id=relationship_id,
            user_id=user_id,
            document_id=document_id,
            chunk_id=chunk_id or None,
            page=page,
            section=section or None,
            evidence_text=evidence_text or None,
            confidence=confidence,
        )
    )
    return True


# ── lifecycle: replace / delete per document ─────────────────────────────────

def replace_document_evidence(db: Session, user_id: Any, document_id: Any) -> Dict[str, int]:
    """Remove every graph fact sourced only from ``document_id``.

    Shared entities and relationships supported by other documents survive:
    the rows themselves are pruned only when no evidence remains.
    """
    removed_mentions = (
        db.query(GraphEntityMention)
        .filter(
            GraphEntityMention.user_id == user_id,
            GraphEntityMention.document_id == document_id,
        )
        .delete(synchronize_session=False)
    )
    removed_evidence = (
        db.query(GraphRelationshipEvidence)
        .filter(
            GraphRelationshipEvidence.user_id == user_id,
            GraphRelationshipEvidence.document_id == document_id,
        )
        .delete(synchronize_session=False)
    )
    removed_relationships = prune_orphan_relationships(db, user_id)
    removed_entities = prune_orphan_entities(db, user_id)
    refresh_counts(db, user_id)
    db.flush()
    return {
        "mentions": int(removed_mentions or 0),
        "evidence": int(removed_evidence or 0),
        "relationships": removed_relationships,
        "entities": removed_entities,
    }


def prune_orphan_relationships(db: Session, user_id: Any) -> int:
    """Delete relationships that no longer have any supporting evidence."""
    evidenced_ids = {
        row[0]
        for row in db.query(GraphRelationshipEvidence.relationship_id)
        .filter(GraphRelationshipEvidence.user_id == user_id)
        .distinct()
        .all()
    }
    orphan_ids = [
        row[0]
        for row in db.query(GraphRelationship.id)
        .filter(GraphRelationship.user_id == user_id)
        .all()
        if row[0] not in evidenced_ids
    ]
    if not orphan_ids:
        return 0
    db.query(GraphRelationship).filter(GraphRelationship.id.in_(orphan_ids)).delete(
        synchronize_session=False
    )
    return len(orphan_ids)


def prune_orphan_entities(db: Session, user_id: Any) -> int:
    """Delete entities no longer mentioned by any document or used by an edge."""
    mentioned = {
        row[0]
        for row in db.query(GraphEntityMention.entity_id)
        .filter(GraphEntityMention.user_id == user_id)
        .distinct()
        .all()
    }
    connected = set()
    for source, target in (
        db.query(GraphRelationship.source_entity_id, GraphRelationship.target_entity_id)
        .filter(GraphRelationship.user_id == user_id)
        .all()
    ):
        connected.add(source)
        connected.add(target)
    keep = mentioned | connected

    orphan_ids = [
        row[0]
        for row in db.query(GraphEntity.id).filter(GraphEntity.user_id == user_id).all()
        if row[0] not in keep
    ]
    if not orphan_ids:
        return 0
    db.query(GraphEntity).filter(GraphEntity.id.in_(orphan_ids)).delete(
        synchronize_session=False
    )
    return len(orphan_ids)


def refresh_counts(db: Session, user_id: Any) -> None:
    """Recompute per-entity mention/document counts after a change."""
    entities = db.query(GraphEntity).filter(GraphEntity.user_id == user_id).all()
    if not entities:
        return
    rows = (
        db.query(
            GraphEntityMention.entity_id,
            GraphEntityMention.document_id,
        )
        .filter(GraphEntityMention.user_id == user_id)
        .all()
    )
    counts: Dict[str, Dict[str, Any]] = {}
    for entity_id_value, document_id_value in rows:
        bucket = counts.setdefault(entity_id_value, {"mentions": 0, "docs": set()})
        bucket["mentions"] += 1
        bucket["docs"].add(str(document_id_value))
    for entity in entities:
        bucket = counts.get(entity.id, {"mentions": 0, "docs": set()})
        entity.mention_count = bucket["mentions"]
        entity.document_count = len(bucket["docs"])


def delete_user_graph(db: Session, user_id: Any) -> None:
    """Drop every graph row for a user (used by account deletion)."""
    entity_ids = [
        row[0] for row in db.query(GraphEntity.id).filter(GraphEntity.user_id == user_id).all()
    ]
    relationship_ids = [
        row[0]
        for row in db.query(GraphRelationship.id).filter(GraphRelationship.user_id == user_id).all()
    ]
    if relationship_ids:
        db.query(GraphRelationshipEvidence).filter(
            GraphRelationshipEvidence.relationship_id.in_(relationship_ids)
        ).delete(synchronize_session=False)
        db.query(GraphRelationship).filter(GraphRelationship.id.in_(relationship_ids)).delete(
            synchronize_session=False
        )
    if entity_ids:
        db.query(GraphEntityMention).filter(
            GraphEntityMention.entity_id.in_(entity_ids)
        ).delete(synchronize_session=False)
        db.query(GraphEntity).filter(GraphEntity.id.in_(entity_ids)).delete(
            synchronize_session=False
        )
    db.query(GraphIndexStatus).filter(GraphIndexStatus.user_id == user_id).delete(
        synchronize_session=False
    )


# ── indexing status ──────────────────────────────────────────────────────────

def set_status(
    db: Session,
    *,
    document_id: Any,
    user_id: Any,
    status: str,
    entity_count: Optional[int] = None,
    relationship_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    error: Optional[str] = None,
    increment_attempts: bool = False,
) -> GraphIndexStatus:
    row = (
        db.query(GraphIndexStatus)
        .filter(GraphIndexStatus.document_id == document_id)
        .first()
    )
    if row is None:
        row = GraphIndexStatus(document_id=document_id, user_id=user_id, status=status)
        db.add(row)
    row.status = status
    row.user_id = user_id
    if entity_count is not None:
        row.entity_count = entity_count
    if relationship_count is not None:
        row.relationship_count = relationship_count
    if chunk_count is not None:
        row.chunk_count = chunk_count
    if increment_attempts:
        row.attempts = (row.attempts or 0) + 1
    row.error = (error or None) if error else None
    row.updated_at = datetime.now(timezone.utc)
    db.flush()
    return row


def get_status(db: Session, user_id: Any, document_id: Any) -> Optional[GraphIndexStatus]:
    return (
        db.query(GraphIndexStatus)
        .filter(
            GraphIndexStatus.document_id == document_id,
            GraphIndexStatus.user_id == user_id,
        )
        .first()
    )


def status_row(row: Optional[GraphIndexStatus], document_id: Any) -> Dict[str, Any]:
    if row is None:
        return {
            "document_id": str(document_id),
            "status": "not_indexed",
            "entity_count": 0,
            "relationship_count": 0,
            "chunk_count": 0,
            "attempts": 0,
            "error": None,
            "updated_at": None,
        }
    return {
        "document_id": str(row.document_id),
        "status": row.status,
        "entity_count": row.entity_count or 0,
        "relationship_count": row.relationship_count or 0,
        "chunk_count": row.chunk_count or 0,
        "attempts": row.attempts or 0,
        "error": row.error,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


# ── query helpers ────────────────────────────────────────────────────────────

def _document_scoped_entity_ids(
    db: Session, user_id: Any, document_ids: Sequence[str]
) -> set:
    if not document_ids:
        return set()
    ids = [str(doc_id) for doc_id in document_ids]
    rows = (
        db.query(GraphEntityMention.entity_id)
        .filter(
            GraphEntityMention.user_id == user_id,
            GraphEntityMention.document_id.in_(ids),
        )
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def _document_scoped_relationship_ids(
    db: Session, user_id: Any, document_ids: Sequence[str]
) -> set:
    if not document_ids:
        return set()
    ids = [str(doc_id) for doc_id in document_ids]
    rows = (
        db.query(GraphRelationshipEvidence.relationship_id)
        .filter(
            GraphRelationshipEvidence.user_id == user_id,
            GraphRelationshipEvidence.document_id.in_(ids),
        )
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def stats(db: Session, user_id: Any, document_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    entity_query = db.query(GraphEntity).filter(GraphEntity.user_id == user_id)
    relationship_query = db.query(GraphRelationship).filter(GraphRelationship.user_id == user_id)

    if document_ids:
        allowed_entities = _document_scoped_entity_ids(db, user_id, document_ids)
        allowed_relationships = _document_scoped_relationship_ids(db, user_id, document_ids)
        entity_query = entity_query.filter(GraphEntity.id.in_(allowed_entities or {""}))
        relationship_query = relationship_query.filter(
            GraphRelationship.id.in_(allowed_relationships or {""})
        )

    entities = entity_query.all()
    relationships = relationship_query.all()

    type_counts: Dict[str, int] = {}
    for entity in entities:
        type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
    relation_counts: Dict[str, int] = {}
    for relationship in relationships:
        relation_counts[relationship.relation_type] = (
            relation_counts.get(relationship.relation_type, 0) + 1
        )

    document_ids_touched = {
        str(row[0])
        for row in db.query(GraphEntityMention.document_id)
        .filter(GraphEntityMention.user_id == user_id)
        .distinct()
        .all()
    }

    return {
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "document_count": len(document_ids_touched),
        "entity_types": [
            {"type": key, "count": value}
            for key, value in sorted(type_counts.items(), key=lambda item: -item[1])
        ],
        "relationship_types": [
            {"type": key, "count": value}
            for key, value in sorted(relation_counts.items(), key=lambda item: -item[1])
        ],
        "top_entities": [
            entity_row(entity)
            for entity in sorted(
                entities,
                key=lambda item: (item.mention_count or 0, item.document_count or 0),
                reverse=True,
            )[:12]
        ],
    }


def search_entities(
    db: Session,
    user_id: Any,
    *,
    query_text: str = "",
    entity_type: Optional[str] = None,
    document_ids: Optional[Sequence[str]] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    rows = db.query(GraphEntity).filter(GraphEntity.user_id == user_id)
    if entity_type:
        rows = rows.filter(GraphEntity.entity_type == entity_type)
    if document_ids:
        allowed = _document_scoped_entity_ids(db, user_id, document_ids)
        rows = rows.filter(GraphEntity.id.in_(allowed or {""}))

    candidates = rows.all()
    term = (query_text or "").strip().casefold()
    if term:
        def score(entity: GraphEntity) -> int:
            name = (entity.canonical_name or "").casefold()
            description = (entity.description or "").casefold()
            aliases = [alias.casefold() for alias in entity_aliases(entity)]
            if name == term:
                return 100
            if name.startswith(term):
                return 80
            if term in name:
                return 60
            if any(term in alias for alias in aliases):
                return 50
            if term in description:
                return 20
            return 0

        scored = [(score(entity), entity) for entity in candidates]
        candidates = [entity for value, entity in sorted(scored, key=lambda item: -item[0]) if value > 0]

    candidates.sort(key=lambda item: (item.mention_count or 0, item.document_count or 0), reverse=True)
    return [entity_row(entity) for entity in candidates[: max(1, limit)]]


def entity_mentions(db: Session, user_id: Any, target_entity_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    rows = (
        db.query(GraphEntityMention)
        .filter(
            GraphEntityMention.user_id == user_id,
            GraphEntityMention.entity_id == target_entity_id,
        )
        .order_by(GraphEntityMention.created_at.desc())
        .limit(limit)
        .all()
    )
    names = _document_names(db, user_id, [row.document_id for row in rows])
    return [
        {
            "document_id": str(row.document_id),
            "document_name": names.get(str(row.document_id), "document"),
            "chunk_id": row.chunk_id,
            "page": row.page,
            "section": row.section,
            "evidence": row.evidence_text,
            "confidence": float(row.confidence or 0.0),
        }
        for row in rows
    ]


def relationships_for_entity(
    db: Session, user_id: Any, target_entity_id: str, limit: int = 30
) -> List[GraphRelationship]:
    return (
        db.query(GraphRelationship)
        .filter(
            GraphRelationship.user_id == user_id,
            or_(
                GraphRelationship.source_entity_id == target_entity_id,
                GraphRelationship.target_entity_id == target_entity_id,
            ),
        )
        .order_by(GraphRelationship.confidence.desc())
        .limit(limit)
        .all()
    )


def relationship_evidence(
    db: Session, user_id: Any, target_relationship_id: str, limit: int = 8
) -> List[Dict[str, Any]]:
    rows = (
        db.query(GraphRelationshipEvidence)
        .filter(
            GraphRelationshipEvidence.user_id == user_id,
            GraphRelationshipEvidence.relationship_id == target_relationship_id,
        )
        .order_by(GraphRelationshipEvidence.confidence.desc())
        .limit(limit)
        .all()
    )
    names = _document_names(db, user_id, [row.document_id for row in rows])
    return [
        {
            "document_id": str(row.document_id),
            "document_name": names.get(str(row.document_id), "document"),
            "chunk_id": row.chunk_id,
            "page": row.page,
            "section": row.section,
            "evidence": row.evidence_text,
            "confidence": float(row.confidence or 0.0),
        }
        for row in rows
    ]


def relationship_row(db: Session, user_id: Any, relationship: GraphRelationship) -> Dict[str, Any]:
    source = get_entity(db, user_id, relationship.source_entity_id)
    target = get_entity(db, user_id, relationship.target_entity_id)
    evidence = relationship_evidence(db, user_id, relationship.id)
    return {
        "id": relationship.id,
        "source": {
            "id": relationship.source_entity_id,
            "name": source.canonical_name if source else "",
            "type": source.entity_type if source else "",
        },
        "target": {
            "id": relationship.target_entity_id,
            "name": target.canonical_name if target else "",
            "type": target.entity_type if target else "",
        },
        "relation_type": relationship.relation_type,
        "description": relationship.description or "",
        "confidence": float(relationship.confidence or 0.0),
        "evidence_count": len(evidence),
        "evidence": evidence,
    }


def graph_snapshot(
    db: Session,
    user_id: Any,
    *,
    document_ids: Optional[Sequence[str]] = None,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
) -> Dict[str, Any]:
    """Whole-workspace graph: the busiest entities plus every edge between them.

    The graph is user-scoped and entities are merged across documents during
    extraction, so an entity mentioned in two PDFs is a single node and its
    edges join those documents instead of being duplicated per file.
    """
    max_nodes = max_nodes or settings.GRAPH_MAX_NEIGHBORHOOD_NODES
    max_edges = max_edges or max(max_nodes * 3, 120)

    nodes = search_entities(db, user_id, document_ids=document_ids, limit=max_nodes)
    node_ids = [entity["id"] for entity in nodes]
    if not node_ids:
        return {
            "nodes": [],
            "edges": [],
            "truncated": False,
            "seed_entity_ids": [],
            "depth": 1,
        }

    allowed_relationships = (
        _document_scoped_relationship_ids(db, user_id, document_ids) if document_ids else None
    )
    rows = (
        db.query(GraphRelationship)
        .filter(
            GraphRelationship.user_id == user_id,
            GraphRelationship.source_entity_id.in_(node_ids),
            GraphRelationship.target_entity_id.in_(node_ids),
        )
        .order_by(GraphRelationship.confidence.desc())
        .all()
    )
    if allowed_relationships is not None:
        rows = [row for row in rows if row.id in allowed_relationships]

    edges = [relationship_row(db, user_id, row) for row in rows[:max_edges]]
    totals = stats(db, user_id, document_ids)
    return {
        "nodes": nodes,
        "edges": edges,
        "truncated": totals["entity_count"] > len(nodes)
        or totals["relationship_count"] > len(edges),
        "seed_entity_ids": node_ids[:5],
        "depth": 1,
    }


def neighborhood(
    db: Session,
    user_id: Any,
    *,
    entity_ids: Sequence[str],
    depth: int = 1,
    document_ids: Optional[Sequence[str]] = None,
    max_nodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Bounded breadth-first subgraph around one or more starting entities."""
    max_nodes = max_nodes or settings.GRAPH_MAX_NEIGHBORHOOD_NODES
    allowed_relationships = (
        _document_scoped_relationship_ids(db, user_id, document_ids) if document_ids else None
    )

    node_ids: List[str] = []
    seen_nodes: set = set()
    seen_edges: set = set()
    edges: List[Dict[str, Any]] = []

    frontier = [entity_id_value for entity_id_value in entity_ids if entity_id_value]
    for entity_id_value in frontier:
        if entity_id_value not in seen_nodes:
            seen_nodes.add(entity_id_value)
            node_ids.append(entity_id_value)

    depth = max(1, min(int(depth or 1), 3))
    for _ in range(depth):
        if not frontier or len(node_ids) >= max_nodes:
            break
        next_frontier: List[str] = []
        rows = (
            db.query(GraphRelationship)
            .filter(
                GraphRelationship.user_id == user_id,
                or_(
                    GraphRelationship.source_entity_id.in_(frontier),
                    GraphRelationship.target_entity_id.in_(frontier),
                ),
            )
            .order_by(GraphRelationship.confidence.desc())
            .all()
        )
        for relationship in rows:
            if allowed_relationships is not None and relationship.id not in allowed_relationships:
                continue
            if relationship.id in seen_edges:
                continue
            if len(node_ids) >= max_nodes:
                break
            seen_edges.add(relationship.id)
            edges.append(relationship_row(db, user_id, relationship))
            for endpoint in (relationship.source_entity_id, relationship.target_entity_id):
                if endpoint not in seen_nodes:
                    seen_nodes.add(endpoint)
                    node_ids.append(endpoint)
                    next_frontier.append(endpoint)
        frontier = next_frontier

    nodes: List[Dict[str, Any]] = []
    for entity_id_value in node_ids:
        entity = get_entity(db, user_id, entity_id_value)
        if entity is None:
            continue
        nodes.append(entity_row(entity))

    return {
        "nodes": nodes,
        "edges": edges,
        "truncated": len(node_ids) >= max_nodes,
        "seed_entity_ids": list(entity_ids),
        "depth": depth,
    }


def entity_document_ids(db: Session, user_id: Any, target_entity_id: str) -> List[str]:
    rows = (
        db.query(GraphEntityMention.document_id)
        .filter(
            GraphEntityMention.user_id == user_id,
            GraphEntityMention.entity_id == target_entity_id,
        )
        .distinct()
        .all()
    )
    return [str(row[0]) for row in rows]


def documents_with_graph(db: Session, user_id: Any) -> List[Dict[str, Any]]:
    rows = (
        db.query(GraphIndexStatus, Document.original_name)
        .outerjoin(Document, Document.id == GraphIndexStatus.document_id)
        .filter(GraphIndexStatus.user_id == user_id)
        .all()
    )
    return [
        {
            **(status_row(status, status.document_id)),
            "document_name": name or "document",
        }
        for status, name in rows
    ]

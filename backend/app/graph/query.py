"""
Read-only graph queries behind the graph APIs.

All functions take ``user_id`` and scope every query to it, so a graph request
can never leak another account's documents.
"""
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.config import get_settings
from app.graph import store

settings = get_settings()


def overview(
    db: Session,
    user_id: Any,
    *,
    document_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    stats = store.stats(db, user_id, document_ids)
    return {
        "available": stats["entity_count"] > 0,
        "scope": "documents" if document_ids else "workspace",
        "document_ids": [str(doc_id) for doc_id in document_ids] if document_ids else [],
        **stats,
    }


def search(
    db: Session,
    user_id: Any,
    *,
    query_text: str = "",
    entity_type: Optional[str] = None,
    document_ids: Optional[Sequence[str]] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    return store.search_entities(
        db,
        user_id,
        query_text=query_text,
        entity_type=entity_type,
        document_ids=document_ids,
        limit=min(max(limit, 1), 100),
    )


def entity_detail(db: Session, user_id: Any, target_entity_id: str) -> Optional[Dict[str, Any]]:
    entity = store.get_entity(db, user_id, target_entity_id)
    if entity is None:
        return None

    mentions = store.entity_mentions(db, user_id, target_entity_id)
    relationships = store.relationships_for_entity(db, user_id, target_entity_id)

    neighbours: List[Dict[str, Any]] = []
    seen = set()
    for relationship in relationships:
        other_id = (
            relationship.target_entity_id
            if relationship.source_entity_id == target_entity_id
            else relationship.source_entity_id
        )
        if other_id in seen:
            continue
        seen.add(other_id)
        other = store.get_entity(db, user_id, other_id)
        if other is None:
            continue
        neighbours.append(
            {
                "id": other.id,
                "name": other.canonical_name,
                "type": other.entity_type,
                "relation_type": relationship.relation_type,
                "direction": (
                    "outgoing"
                    if relationship.source_entity_id == target_entity_id
                    else "incoming"
                ),
                "confidence": float(relationship.confidence or 0.0),
            }
        )

    detail = store.entity_row(entity)
    detail.update(
        {
            "mentions": mentions,
            "documents": _documents(mentions),
            "connected": neighbours,
            "relationships": [
                store.relationship_row(db, user_id, relationship)
                for relationship in relationships
            ],
        }
    )
    return detail


def _documents(mentions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    seen = set()
    for mention in mentions:
        key = mention.get("document_id")
        if not key or key in seen:
            continue
        seen.add(key)
        documents.append(
            {
                "id": key,
                "name": mention.get("document_name") or "document",
            }
        )
    return documents


def relationship_detail(db: Session, user_id: Any, target_relationship_id: str) -> Optional[Dict[str, Any]]:
    from app.models import GraphRelationship

    relationship = (
        db.query(GraphRelationship)
        .filter(
            GraphRelationship.id == target_relationship_id,
            GraphRelationship.user_id == user_id,
        )
        .first()
    )
    if relationship is None:
        return None
    detail = store.relationship_row(db, user_id, relationship)
    detail["documents"] = _documents(detail.get("evidence") or [])
    return detail


def neighborhood(
    db: Session,
    user_id: Any,
    *,
    entity_id: Optional[str] = None,
    depth: int = 1,
    document_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Bounded subgraph around an entity, or around the busiest entities."""
    limit = min(limit or settings.GRAPH_MAX_NEIGHBORHOOD_NODES, 200)

    if not entity_id:
        # No seed means "show me the whole knowledge base": every document's
        # entities plus the cross-document edges between them.
        return store.graph_snapshot(
            db,
            user_id,
            document_ids=document_ids,
            max_nodes=limit,
        )

    entity = store.get_entity(db, user_id, entity_id)
    if entity is None:
        return {"nodes": [], "edges": [], "truncated": False, "seed_entity_ids": []}

    return store.neighborhood(
        db,
        user_id,
        entity_ids=[entity.id],
        depth=depth,
        document_ids=document_ids,
        max_nodes=limit,
    )

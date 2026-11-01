"""
Graph indexing service: extraction → resolution → persistence.

Called from the ingestion pipeline (incremental, per document) and from the
reindex API. The service owns transactions and always records an explicit
indexing status, so a failed extraction is visible and retryable instead of
silently marking the document as indexed.
"""
import logging
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.graph import store
from app.graph.extraction import extract_from_chunks
from app.graph.resolution import (
    ExistingEntity,
    build_lookup,
    normalize_name,
    resolve,
)
from app.models import Document, GraphEntity, GraphRelationship

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_db():
    from app.database import SessionLocal

    return SessionLocal()


def _lookup_rows(db, user_id: Any) -> List[ExistingEntity]:
    return build_lookup(store.list_entities(db, user_id))


def _persist_extraction(
    db,
    user_id: Any,
    document_id: Any,
    extraction,
) -> Dict[str, int]:
    """Resolve extracted entities against the stored graph and persist them."""
    lookup = _lookup_rows(db, user_id)
    entity_by_key: Dict[str, str] = {}
    created_entities = 0
    merged_entities = 0
    ambiguous = 0

    for candidate in extraction.entities:
        normalized = normalize_name(candidate.name)
        if not normalized:
            continue

        decision = resolve(
            name=candidate.name,
            entity_type=candidate.entity_type,
            aliases=candidate.aliases,
            existing=lookup,
            threshold=settings.GRAPH_ENTITY_RESOLUTION_THRESHOLD,
        )

        if decision.action == "match" and decision.matched_entity_id:
            entity = store.get_entity(db, user_id, decision.matched_entity_id)
            if entity is None:
                continue
            store.merge_entity_metadata(
                db,
                entity,
                description=candidate.description,
                aliases=list(candidate.aliases) + decision.new_aliases,
                confidence=candidate.confidence,
            )
            merged_entities += 1
        else:
            entity = store.create_entity(
                db,
                user_id,
                name=candidate.name,
                entity_type=candidate.entity_type,
                normalized_name=normalized,
                description=candidate.description,
                aliases=candidate.aliases,
                confidence=candidate.confidence,
                possible_duplicate_of=decision.possible_duplicate_of,
            )
            created_entities += 1
            if decision.possible_duplicate_of:
                ambiguous += 1
            lookup.append(
                ExistingEntity(
                    id=entity.id,
                    name=entity.canonical_name,
                    entity_type=entity.entity_type,
                    aliases=list(candidate.aliases),
                )
            )

        entity_by_key[candidate.key] = entity.id
        store.add_mention(
            db,
            target_entity_id=entity.id,
            user_id=user_id,
            document_id=document_id,
            chunk_id=candidate.chunk_id,
            page=candidate.page,
            section=candidate.section,
            evidence_text=candidate.evidence or candidate.description,
            confidence=candidate.confidence,
            extraction_method=candidate.extraction_method,
        )

    created_relationships = 0
    skipped_relationships = 0
    for candidate in extraction.relationships:
        source_id = entity_by_key.get(candidate.source_key)
        target_id = entity_by_key.get(candidate.target_key)
        if not source_id or not target_id or source_id == target_id:
            skipped_relationships += 1
            continue

        relationship, created = store.upsert_relationship(
            db,
            user_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=candidate.relation_type,
            description=candidate.description,
            confidence=candidate.confidence,
            extraction_method=candidate.extraction_method,
        )
        if created:
            created_relationships += 1
        store.add_relationship_evidence(
            db,
            relationship_id=relationship.id,
            user_id=user_id,
            document_id=document_id,
            chunk_id=candidate.chunk_id,
            page=candidate.page,
            section=candidate.section,
            evidence_text=candidate.evidence,
            confidence=candidate.confidence,
        )

    store.refresh_counts(db, user_id)
    return {
        "entities_created": created_entities,
        "entities_merged": merged_entities,
        "entities_ambiguous": ambiguous,
        "relationships_created": created_relationships,
        "relationships_skipped": skipped_relationships,
    }


def index_document(
    document_id: str,
    user_id: str,
    chunks: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    llm_provider: Optional[str] = None,
    hf_token: Optional[str] = None,
    completer: Any = None,
    db: Any = None,
) -> Dict[str, Any]:
    """Build/refresh the graph contribution of a single document.

    Returns a status dict; never raises for extraction problems so that a graph
    failure cannot break document ingestion. Callers that need hard failure can
    inspect ``ok`` in the result.
    """
    owns_session = db is None
    session = db or _get_db()

    try:
        document = (
            session.query(Document)
            .filter(Document.id == document_id, Document.is_deleted.is_(False))
            .first()
        )
        if document is None:
            return {"ok": False, "status": "missing_document", "document_id": str(document_id)}

        if not settings.GRAPH_INDEXING_ENABLED:
            store.set_status(
                session,
                document_id=document_id,
                user_id=user_id,
                status="disabled",
                error=None,
            )
            session.commit()
            return {"ok": True, "status": "disabled", "document_id": str(document_id)}

        store.set_status(
            session,
            document_id=document_id,
            user_id=user_id,
            status="indexing",
            increment_attempts=True,
        )
        session.commit()

        if chunks is None:
            chunks = load_document_chunks(str(document_id), str(user_id))
        chunks = list(chunks or [])

        if not chunks:
            store.replace_document_evidence(session, user_id, document_id)
            store.set_status(
                session,
                document_id=document_id,
                user_id=user_id,
                status="ready",
                entity_count=0,
                relationship_count=0,
                chunk_count=0,
            )
            session.commit()
            return {
                "ok": True,
                "status": "ready",
                "document_id": str(document_id),
                "entities": 0,
                "relationships": 0,
                "reason": "no_chunks",
            }

        try:
            extraction = extract_from_chunks(
                chunks,
                str(document_id),
                llm_provider=llm_provider,
                hf_token=hf_token,
                completer=completer,
            )
        except Exception as exc:
            logger.warning("Graph extraction failed for %s: %s", document_id, exc)
            store.set_status(
                session,
                document_id=document_id,
                user_id=user_id,
                status="failed",
                chunk_count=len(chunks),
                error=str(exc)[:500],
            )
            session.commit()
            return {
                "ok": False,
                "status": "failed",
                "document_id": str(document_id),
                "error": str(exc)[:500],
            }

        try:
            # Replace this document's previous contribution only after a
            # successful extraction, so a failure leaves the graph intact.
            store.replace_document_evidence(session, user_id, document_id)
            persisted = _persist_extraction(
                session, user_id, document_id, extraction
            )
            entity_count = (
                session.query(GraphEntity).filter(GraphEntity.user_id == user_id).count()
            )
            relationship_count = (
                session.query(GraphRelationship)
                .filter(GraphRelationship.user_id == user_id)
                .count()
            )
            store.set_status(
                session,
                document_id=document_id,
                user_id=user_id,
                status="ready",
                entity_count=entity_count,
                relationship_count=relationship_count,
                chunk_count=extraction.chunks_sent,
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.exception("Graph persistence failed for %s", document_id)
            store.set_status(
                session,
                document_id=document_id,
                user_id=user_id,
                status="failed",
                error=str(exc)[:500],
            )
            session.commit()
            return {
                "ok": False,
                "status": "failed",
                "document_id": str(document_id),
                "error": str(exc)[:500],
            }

        return {
            "ok": True,
            "status": "ready",
            "document_id": str(document_id),
            "entities": len(extraction.entities),
            "relationships": len(extraction.relationships),
            "chunks_sent": extraction.chunks_sent,
            "rejected_entities": extraction.rejected_entities,
            "rejected_relationships": extraction.rejected_relationships,
            **persisted,
        }
    finally:
        if owns_session:
            session.close()


def remove_document(document_id: str, user_id: str, *, db: Any = None) -> Dict[str, int]:
    """Remove one document's graph contribution, preserving shared facts."""
    owns_session = db is None
    session = db or _get_db()
    try:
        removed = store.replace_document_evidence(session, user_id, document_id)
        status = store.get_status(session, user_id, document_id)
        if status is not None:
            session.delete(status)
        session.commit()
        return removed
    finally:
        if owns_session:
            session.close()


def load_document_chunks(document_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Read a document's indexed chunks back out of the vector store.

    Used by reindexing, where the original chunks are no longer in memory.
    """
    try:
        from app.rag.vectordb.vectorstore import get_chroma_client, get_collection_name

        client = get_chroma_client()
        collection = client.get_or_create_collection(name=get_collection_name(user_id))
        payload = collection.get(
            where={"document_id": str(document_id)},
            include=["metadatas"],
        )
    except Exception as exc:
        logger.warning("Could not read chunks for graph reindex of %s: %s", document_id, exc)
        return []

    metadatas = payload.get("metadatas") or []
    chunks: List[Dict[str, Any]] = []
    for metadata in metadatas:
        if not isinstance(metadata, dict) or not metadata.get("text"):
            continue
        chunks.append(
            {
                "text": metadata.get("text"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "section": metadata.get("section") or "",
            }
        )
    chunks.sort(key=lambda chunk: chunk.get("chunk_index") or 0)
    return chunks


def reindex_document(
    document_id: str,
    user_id: str,
    *,
    llm_provider: Optional[str] = None,
    hf_token: Optional[str] = None,
    completer: Any = None,
) -> Dict[str, Any]:
    """Rebuild a document's graph contribution from its indexed chunks."""
    chunks = load_document_chunks(document_id, user_id)
    return index_document(
        document_id,
        user_id,
        chunks,
        llm_provider=llm_provider,
        hf_token=hf_token,
        completer=completer,
    )


def status(document_id: str, user_id: str, *, db: Any = None) -> Dict[str, Any]:
    owns_session = db is None
    session = db or _get_db()
    try:
        row = store.get_status(session, user_id, document_id)
        return store.status_row(row, document_id)
    finally:
        if owns_session:
            session.close()

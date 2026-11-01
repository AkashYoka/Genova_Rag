"""
Knowledge graph (GraphRAG) API.

Two generations of endpoint live here:

* ``/graph/overview``, ``/graph/search``, ``/graph/entities/...`` and friends
  serve the workspace graph stored in the relational database (cross-document,
  evidence-backed).
* ``/graph/{document_id}`` and ``/graph/{document_id}/summary`` are the original
  per-document spaCy co-occurrence graph and are kept for backward
  compatibility.

Static paths are declared before ``/{document_id}`` so route matching stays
unambiguous.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import get_settings
from app.database import get_db
from app.graph import query as graph_query
from app.graph import service as graph_service
from app.graph import store as graph_store
from app.models import Document, User
from app.rag.utils.graph_builder import load_graph

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Response schemas ──────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    name: str
    label: str          # NER type / entity type
    mentions: int
    pages: List[int]


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: int
    pages: List[int]


class GraphResponse(BaseModel):
    document_id: str
    document_name: str
    node_count: int
    edge_count: int
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class GraphSummaryResponse(BaseModel):
    document_id: str
    document_name: str
    node_count: int
    edge_count: int
    top_entities: List[Dict[str, Any]]
    graph_available: bool


class GraphIndexStatusResponse(BaseModel):
    document_id: str
    status: str
    entity_count: int = 0
    relationship_count: int = 0
    chunk_count: int = 0
    attempts: int = 0
    error: Optional[str] = None
    updated_at: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_owned_document(document_id: str, user: User, db: Session) -> Document:
    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.user_id == user.id,
            Document.is_deleted.is_(False),
        )
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return doc


def _document_ids_param(document_ids: Optional[str]) -> List[str]:
    """Parse a comma-separated document filter."""
    if not document_ids:
        return []
    return [item.strip() for item in document_ids.split(",") if item.strip()]


# ── Workspace graph (relational store) ───────────────────────────────────────

@router.get("/overview", summary="Graph overview for a workspace or document selection")
def graph_overview(
    document_ids: Optional[str] = Query(None, description="Comma-separated document ids"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return graph_query.overview(
        db, current_user.id, document_ids=_document_ids_param(document_ids) or None
    )


@router.get("/search", summary="Search graph entities by name, alias or description")
def graph_search(
    q: str = Query("", description="Search text"),
    entity_type: Optional[str] = Query(None),
    document_ids: Optional[str] = Query(None),
    limit: int = Query(25, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {
        "query": q,
        "results": graph_query.search(
            db,
            current_user.id,
            query_text=q,
            entity_type=entity_type,
            document_ids=_document_ids_param(document_ids) or None,
            limit=limit,
        ),
    }


@router.get("/neighborhood", summary="Bounded subgraph around an entity")
def graph_neighborhood(
    entity_id: Optional[str] = Query(None, description="Seed entity id"),
    depth: int = Query(1, ge=1, le=3),
    limit: int = Query(60, ge=1, le=200),
    document_ids: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return graph_query.neighborhood(
        db,
        current_user.id,
        entity_id=entity_id,
        depth=depth,
        document_ids=_document_ids_param(document_ids) or None,
        limit=limit,
    )


@router.get("/entities/{entity_id}", summary="Entity details with evidence and connections")
def graph_entity(
    entity_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    detail = graph_query.entity_detail(db, current_user.id, entity_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entity not found")
    return detail


@router.get("/relationships/{relationship_id}", summary="Relationship details with evidence")
def graph_relationship(
    relationship_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    detail = graph_query.relationship_detail(db, current_user.id, relationship_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Relationship not found"
        )
    return detail


@router.get("/documents", summary="Graph indexing status for every document")
def graph_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {"documents": graph_store.documents_with_graph(db, current_user.id)}


@router.get(
    "/documents/{document_id}/status",
    response_model=GraphIndexStatusResponse,
    summary="Graph indexing status for one document",
)
def graph_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_owned_document(document_id, current_user, db)
    return graph_service.status(document_id, current_user.id, db=db)


@router.post(
    "/documents/{document_id}/reindex",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run graph extraction for one document",
)
def graph_reindex(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not settings.GRAPH_INDEXING_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Graph indexing is disabled",
        )
    _get_owned_document(document_id, current_user, db)

    user_id = str(current_user.id)
    background_tasks.add_task(graph_service.reindex_document, document_id, user_id)
    return {
        "document_id": document_id,
        "status": "queued",
        "message": "Graph reindexing started",
    }


@router.delete(
    "/documents/{document_id}",
    summary="Remove one document's contribution to the graph",
)
def graph_forget_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_owned_document(document_id, current_user, db)
    removed = graph_service.remove_document(document_id, str(current_user.id), db=db)
    return {"document_id": document_id, "removed": removed}


# ── Legacy per-document co-occurrence graph ──────────────────────────────────

@router.get("/{document_id}", response_model=GraphResponse)
def get_graph(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Full per-document co-occurrence graph (legacy visualisation payload)."""
    doc = _get_owned_document(document_id, current_user, db)

    graph = load_graph(str(current_user.id), document_id)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge graph not available for this document yet. "
                   "The document may still be processing.",
        )

    nodes: List[GraphNode] = []
    for node_id, data in graph.nodes(data=True):
        nodes.append(
            GraphNode(
                id=node_id,
                name=data.get("name", node_id),
                label=data.get("label", "UNKNOWN"),
                mentions=data.get("mentions", 1),
                pages=sorted(data.get("pages", [])),
            )
        )

    edges: List[GraphEdge] = []
    for source, target, data in graph.edges(data=True):
        edges.append(
            GraphEdge(
                source=source,
                target=target,
                weight=data.get("weight", 1),
                pages=sorted(data.get("pages", [])),
            )
        )

    return GraphResponse(
        document_id=document_id,
        document_name=doc.original_name,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        nodes=nodes,
        edges=edges,
    )


@router.get("/{document_id}/summary", response_model=GraphSummaryResponse)
def get_graph_summary(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lightweight statistics for the legacy per-document graph."""
    doc = _get_owned_document(document_id, current_user, db)

    graph = load_graph(str(current_user.id), document_id)
    if graph is None:
        return GraphSummaryResponse(
            document_id=document_id,
            document_name=doc.original_name,
            node_count=0,
            edge_count=0,
            top_entities=[],
            graph_available=False,
        )

    top_entities = sorted(
        [
            {
                "id": node_id,
                "name": data.get("name", node_id),
                "label": data.get("label", "UNKNOWN"),
                "mentions": data.get("mentions", 1),
            }
            for node_id, data in graph.nodes(data=True)
        ],
        key=lambda x: x["mentions"],
        reverse=True,
    )[:10]

    return GraphSummaryResponse(
        document_id=document_id,
        document_name=doc.original_name,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        top_entities=top_entities,
        graph_available=True,
    )

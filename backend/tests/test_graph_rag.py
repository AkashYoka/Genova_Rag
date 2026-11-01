"""Cross-document GraphRAG tests: extraction → resolution → store → retrieval.

The fixtures use three domain-neutral documents (a system architecture note, an
API note and a data note) so the tests exercise arbitrary entity/relationship
types rather than any fixed vocabulary.
"""
import json

import pytest

from app.graph import query as graph_query
from app.graph import service as graph_service
from app.graph import store
from app.graph.extraction import EXTRACTION_SYSTEM_PROMPT
from app.graph.resolution import ExistingEntity, resolve
from app.graph.retrieval import get_entity_context, retrieve_graph_evidence
from app.graph.schemas import parse_extraction_payload
from app.models import Document


# ── Extraction validation ────────────────────────────────────────────────────

TECH_PAYLOAD = {
    "entities": [
        {
            "key": "backend",
            "name": "Application Backend",
            "type": "Software Component",
            "description": "The service that serves requests.",
            "aliases": ["Backend"],
            "confidence": 0.9,
            "evidence": "The Application Backend uses the Primary Database.",
        },
        {
            "key": "db",
            "name": "Primary Database",
            "type": "data store",
            "confidence": 0.85,
            "evidence": "The Application Backend uses the Primary Database.",
        },
    ],
    "relationships": [
        {
            "source": "backend",
            "target": "db",
            "type": "uses",
            "description": "The backend reads and writes rows.",
            "confidence": 0.8,
            "evidence": "The Application Backend uses the Primary Database.",
        },
        {
            "source": "backend",
            "target": "db",
            "type": "depends on",
            "description": "No supporting quote.",
            "confidence": 0.9,
        },
        {
            "source": "backend",
            "target": "missing",
            "type": "owns",
            "confidence": 0.9,
            "evidence": "Unknown endpoint.",
        },
    ],
}


def test_extraction_accepts_arbitrary_entity_and_relationship_types():
    result = parse_extraction_payload(
        TECH_PAYLOAD,
        {"doc-1_0": {"page": 2, "section": "Architecture"}},
        default_chunk_id="doc-1_0",
    )

    assert {entity.name for entity in result.entities} == {
        "Application Backend",
        "Primary Database",
    }
    types = {entity.entity_type for entity in result.entities}
    assert types == {"SOFTWARE_COMPONENT", "DATA_STORE"}
    assert [rel.relation_type for rel in result.relationships] == ["uses"]


def test_extraction_drops_relationships_without_evidence_or_known_endpoints():
    result = parse_extraction_payload(TECH_PAYLOAD, {}, default_chunk_id="doc-1_0")

    # The evidenced relationship survives; the unevidenced and dangling ones do not.
    assert len(result.relationships) == 1
    assert result.rejected_relationships == 2
    relationship = result.relationships[0]
    assert relationship.evidence == "The Application Backend uses the Primary Database."
    assert relationship.page is None  # no chunk metadata supplied


@pytest.mark.parametrize(
    "payload, expected_type",
    [
        (
            {
                "entities": [
                    {"name": "Net Revenue", "type": "financial metric", "confidence": 0.8}
                ],
                "relationships": [],
            },
            "FINANCIAL_METRIC",
        ),
        (
            {
                "entities": [
                    {"name": "Randomised Trial", "type": "study design", "confidence": 0.7}
                ],
                "relationships": [],
            },
            "STUDY_DESIGN",
        ),
        (
            {
                "entities": [{"name": "Vendor Invoice", "type": "Record Type"}],
                "relationships": [],
            },
            "RECORD_TYPE",
        ),
    ],
)
def test_extraction_normalises_model_defined_types_from_any_domain(payload, expected_type):
    result = parse_extraction_payload(payload, {}, default_chunk_id="c0")

    assert result.entities[0].entity_type == expected_type


def test_extraction_prompt_is_domain_neutral():
    """The extraction prompt must not ship a closed entity vocabulary."""
    lowered = " ".join(EXTRACTION_SYSTEM_PROMPT.lower().split())
    for forbidden in ("patient", "diagnosis", "medication", "drug", "disease"):
        assert forbidden not in lowered
    assert "free text" in lowered
    assert "do not force a fixed vocabulary" in lowered


# ── Entity resolution ────────────────────────────────────────────────────────

def _existing():
    return [
        ExistingEntity(
            id="e-postgres",
            name="PostgreSQL",
            entity_type="DATABASE",
            aliases=["Postgres"],
        ),
        ExistingEntity(
            id="e-pipeline",
            name="Data Pipeline",
            entity_type="PROCESS",
        ),
    ]


def test_resolution_merges_exact_and_alias_matches():
    exact = resolve(name="postgresql", entity_type="DATABASE", existing=_existing())
    alias = resolve(name="Postgres", entity_type="DATABASE", existing=_existing())
    acronym = resolve(name="DP", entity_type="PROCESS", existing=_existing())

    assert exact.action == "match" and exact.matched_entity_id == "e-postgres"
    assert alias.action == "match" and alias.matched_entity_id == "e-postgres"
    assert acronym.matched_entity_id == "e-pipeline"


def test_resolution_keeps_ambiguous_names_separate():
    decision = resolve(
        name="Data Pipelines",
        entity_type="PROCESS",
        existing=_existing(),
        threshold=0.9,
    )

    assert decision.action == "new"
    assert decision.possible_duplicate_of == "e-pipeline"


def test_resolution_never_merges_across_entity_types():
    decision = resolve(name="PostgreSQL", entity_type="PRODUCT", existing=_existing())

    assert decision.action == "new"
    assert decision.matched_entity_id is None


# ── Cross-document graph construction ───────────────────────────────────────

def _completer(responses):
    def complete(messages):
        prompt = messages[-1]["content"]
        for marker, payload in responses.items():
            if marker in prompt:
                return json.dumps(payload)
        return json.dumps({"entities": [], "relationships": []})

    return complete


DOC_A = {
    "entities": [
        {
            "key": "backend",
            "name": "Application Backend",
            "type": "SYSTEM",
            "aliases": ["Backend"],
            "confidence": 0.9,
            "evidence": "The Application Backend uses the Primary Database for storage.",
        },
        {
            "key": "db",
            "name": "Primary Database",
            "type": "DATA_STORE",
            "confidence": 0.85,
            "evidence": "The Application Backend uses the Primary Database for storage.",
        },
    ],
    "relationships": [
        {
            "source": "backend",
            "target": "db",
            "type": "uses",
            "confidence": 0.82,
            "evidence": "The Application Backend uses the Primary Database for storage.",
        }
    ],
}

DOC_B = {
    "entities": [
        {
            "key": "backend",
            "name": "application backend",
            "type": "SYSTEM",
            "confidence": 0.7,
            "evidence": "The Application Backend exposes the Public API over HTTPS.",
        },
        {
            "key": "api",
            "name": "Public API",
            "type": "INTERFACE",
            "confidence": 0.8,
            "evidence": "The Application Backend exposes the Public API over HTTPS.",
        },
    ],
    "relationships": [
        {
            "source": "backend",
            "target": "api",
            "type": "exposes",
            "confidence": 0.75,
            "evidence": "The Application Backend exposes the Public API over HTTPS.",
        }
    ],
}

DOC_C = {
    "entities": [
        {
            "key": "db",
            "name": "Primary Database",
            "type": "DATA_STORE",
            "confidence": 0.8,
            "evidence": "The Primary Database stores Customer Records.",
        },
        {
            "key": "records",
            "name": "Customer Records",
            "type": "DATA_ENTITY",
            "confidence": 0.8,
            "evidence": "The Primary Database stores Customer Records.",
        },
    ],
    "relationships": [
        {
            "source": "db",
            "target": "records",
            "type": "stores",
            "confidence": 0.7,
            "evidence": "The Primary Database stores Customer Records.",
        }
    ],
}


@pytest.fixture()
def graph_documents(db_session, user):
    """Three documents whose text carries a distinct extraction marker."""
    docs = {}
    for key, text in (
        ("A", "ARCH-A The Application Backend uses the Primary Database for storage."),
        ("B", "ARCH-B The Application Backend exposes the Public API over HTTPS."),
        ("C", "ARCH-C The Primary Database stores Customer Records."),
    ):
        document = Document(
            user_id=user.id,
            filename=f"doc-{key}.txt",
            original_name=f"doc-{key}.txt",
            file_size=64,
            page_count=1,
            status="ready",
        )
        db_session.add(document)
        db_session.commit()
        db_session.refresh(document)
        docs[key] = {"document": document, "text": text}
    return docs


def _index(docs, db_session, user, *, keys=("A", "B", "C")):
    responses = {
        "ARCH-A": DOC_A,
        "ARCH-B": DOC_B,
        "ARCH-C": DOC_C,
    }
    results = {}
    for key in keys:
        entry = docs[key]
        results[key] = graph_service.index_document(
            str(entry["document"].id),
            str(user.id),
            [{"text": entry["text"], "page": 1, "chunk_index": 0}],
            completer=_completer(responses),
            db=db_session,
        )
    return results


def test_indexing_merges_shared_entities_across_documents(db_session, user, graph_documents):
    results = _index(graph_documents, db_session, user)

    assert all(result["ok"] for result in results.values())
    entities = {row["canonical_name"]: row for row in store.list_entities(db_session, user.id)}

    backend = entities["Application Backend"]
    assert backend["mention_count"] == 2
    assert backend["document_count"] == 2
    # "application backend" from doc B resolves to the same canonical entity.
    assert "application backend" not in entities

    relationships = store.relationships_for_entity(db_session, user.id, backend["id"])
    labels = {(rel.source_entity_id, rel.target_entity_id, rel.relation_type) for rel in relationships}
    assert len(labels) == 2
    assert {label[2] for label in labels} == {"uses", "exposes"}


def test_neighborhood_traverses_across_documents(db_session, user, graph_documents):
    _index(graph_documents, db_session, user)
    entities = {row["canonical_name"]: row for row in store.list_entities(db_session, user.id)}

    one_hop = graph_query.neighborhood(
        db_session, user.id, entity_id=entities["Application Backend"]["id"], depth=1
    )
    names = {node["canonical_name"] for node in one_hop["nodes"]}
    assert names == {"Application Backend", "Primary Database", "Public API"}

    two_hops = graph_query.neighborhood(
        db_session, user.id, entity_id=entities["Application Backend"]["id"], depth=2
    )
    names = {node["canonical_name"] for node in two_hops["nodes"]}
    assert "Customer Records" in names
    assert two_hops["truncated"] is False


def test_workspace_graph_shows_every_document_and_cross_document_edges(
    db_session, user, graph_documents
):
    _index(graph_documents, db_session, user)

    workspace = graph_query.neighborhood(db_session, user.id)

    names = {node["canonical_name"] for node in workspace["nodes"]}
    assert {"Application Backend", "Primary Database", "Public API"} <= names
    assert workspace["edges"]
    assert workspace["truncated"] is False
    # A shared entity is a single node whose edges join the two documents.
    shared = [n for n in workspace["nodes"] if n["canonical_name"] == "Application Backend"]
    assert len(shared) == 1
    edge_pairs = {
        (edge["source"]["name"], edge["target"]["name"]) for edge in workspace["edges"]
    }
    assert ("Application Backend", "Primary Database") in edge_pairs


def test_entity_detail_reports_documents_and_evidence(db_session, user, graph_documents):
    _index(graph_documents, db_session, user)
    backend_id = next(
        row["id"]
        for row in store.list_entities(db_session, user.id)
        if row["canonical_name"] == "Application Backend"
    )

    detail = graph_query.entity_detail(db_session, user.id, backend_id)

    assert detail is not None
    assert len(detail["documents"]) == 2
    assert len(detail["mentions"]) == 2
    assert all(mention["evidence"] for mention in detail["mentions"])
    assert {document["name"] for document in detail["documents"]} == {
        "doc-A.txt",
        "doc-B.txt",
    }


def test_removing_one_document_preserves_shared_entities(db_session, user, graph_documents):
    _index(graph_documents, db_session, user)
    doc_a = str(graph_documents["A"]["document"].id)

    graph_service.remove_document(doc_a, str(user.id), db=db_session)

    entities = {row["canonical_name"]: row for row in store.list_entities(db_session, user.id)}
    assert "Application Backend" in entities  # still supported by doc B
    assert entities["Application Backend"]["mention_count"] == 1
    assert "Primary Database" in entities  # still supported by doc C

    backend_id = entities["Application Backend"]["id"]
    relation_types = {
        rel.relation_type
        for rel in store.relationships_for_entity(db_session, user.id, backend_id)
    }
    # The doc-A-only relationship is gone; doc B's relationship remains.
    assert relation_types == {"exposes"}


def test_removing_all_documents_clears_the_graph(db_session, user, graph_documents):
    _index(graph_documents, db_session, user)
    for entry in graph_documents.values():
        graph_service.remove_document(str(entry["document"].id), str(user.id), db=db_session)

    assert store.list_entities(db_session, user.id) == []


def test_graph_is_isolated_between_users(db_session, user, other_user, graph_documents):
    _index(graph_documents, db_session, user)
    backend_id = next(
        row["id"]
        for row in store.list_entities(db_session, user.id)
        if row["canonical_name"] == "Application Backend"
    )

    assert store.list_entities(db_session, other_user.id) == []
    assert graph_query.entity_detail(db_session, other_user.id, backend_id) is None
    assert (
        graph_query.neighborhood(db_session, other_user.id, entity_id=backend_id)["nodes"]
        == []
    )


def test_failed_extraction_is_recorded_and_leaves_graph_intact(db_session, user, graph_documents):
    _index(graph_documents, db_session, user)
    doc_a = str(graph_documents["A"]["document"].id)

    def failing_completer(_messages):
        raise RuntimeError("model timeout")

    result = graph_service.index_document(
        doc_a,
        str(user.id),
        [{"text": "ARCH-A retry", "page": 1, "chunk_index": 0}],
        completer=failing_completer,
        db=db_session,
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    status = graph_service.status(doc_a, str(user.id), db=db_session)
    assert status["status"] == "failed"
    assert "model timeout" in (status["error"] or "")

    # Previous evidence from that document was not destroyed by the failure.
    entities = {row["canonical_name"] for row in store.list_entities(db_session, user.id)}
    assert "Application Backend" in entities


# ── Graph retrieval feeding the RAG pipeline ────────────────────────────────

def test_graph_retrieval_returns_evidence_and_citations(db_session, user, graph_documents):
    _index(graph_documents, db_session, user)

    evidence = retrieve_graph_evidence(
        db_session,
        user.id,
        "How does the Application Backend use the Primary Database?",
    )

    assert evidence.is_empty is False
    assert "Knowledge Graph Context" in evidence.context
    assert "Application Backend" in evidence.context
    assert evidence.sources
    assert all(source["filename"].startswith("doc-") for source in evidence.sources)
    assert all(source["text"] for source in evidence.sources)


def test_graph_retrieval_respects_document_scope(db_session, user, graph_documents):
    _index(graph_documents, db_session, user)
    doc_c = str(graph_documents["C"]["document"].id)

    scoped = retrieve_graph_evidence(
        db_session,
        user.id,
        "How does the Application Backend use the Primary Database?",
        document_ids=[doc_c],
    )
    unscoped = retrieve_graph_evidence(
        db_session,
        user.id,
        "How does the Application Backend use the Primary Database?",
    )

    # Doc C never mentions the Application Backend and holds no evidence for
    # the "uses" relationship, so that fact must not leak into a doc-C scope.
    assert "Application Backend" in unscoped.context
    assert "Application Backend" not in scoped.context
    assert "Primary Database" in scoped.context


def test_graph_api_serves_the_authenticated_users_graph(
    client, auth_headers, db_session, user, graph_documents
):
    _index(graph_documents, db_session, user)

    overview = client.get("/api/v1/graph/overview", headers=auth_headers)
    assert overview.status_code == 200
    assert overview.json()["entity_count"] == 4

    search = client.get("/api/v1/graph/search?q=public", headers=auth_headers)
    assert search.status_code == 200
    results = search.json()["results"]
    assert [item["canonical_name"] for item in results] == ["Public API"]

    entity_id = results[0]["id"]
    detail = client.get(f"/api/v1/graph/entities/{entity_id}", headers=auth_headers)
    assert detail.status_code == 200
    body = detail.json()
    assert body["canonical_name"] == "Public API"
    assert body["mentions"][0]["evidence"]

    neighborhood = client.get(
        "/api/v1/graph/neighborhood", params={"depth": 2}, headers=auth_headers
    )
    assert neighborhood.status_code == 200
    assert len(neighborhood.json()["nodes"]) >= 3

    documents = client.get("/api/v1/graph/documents", headers=auth_headers)
    assert documents.status_code == 200
    assert len(documents.json()["documents"]) == 3


@pytest.mark.parametrize("path", ["/api/v1/graph/overview", "/api/v1/graph/search"])
def test_graph_api_requires_authentication(client, db_session, user, graph_documents, path):
    response = client.get(path)
    assert response.status_code in (401, 403)


def test_graph_api_does_not_expose_another_users_graph(
    client, db_session, user, other_user, graph_documents
):
    from app.auth import create_access_token

    _index(graph_documents, db_session, user)
    entity_id = next(
        row["id"]
        for row in store.list_entities(db_session, user.id)
        if row["canonical_name"] == "Public API"
    )

    other_headers = {"Authorization": f"Bearer {create_access_token(other_user.id)}"}
    response = client.get(f"/api/v1/graph/entities/{entity_id}", headers=other_headers)
    assert response.status_code == 404

    overview = client.get("/api/v1/graph/overview", headers=other_headers)
    assert overview.status_code == 200
    assert overview.json()["entity_count"] == 0


def test_chat_history_round_trip_preserves_graph_metadata(db_session, user):
    from app.models import ChatMessage
    from app.routes.chat import _enriched_sources_payload, _message_response_from_row

    graph = {
        "entities": ["Application Backend", "Primary Database"],
        "relationship_count": 2,
        "source_count": 1,
        "truncated": False,
    }
    saved = _enriched_sources_payload([], graph=graph, llm_provider="ollama")
    message = ChatMessage(
        user_id=user.id,
        role="assistant",
        content="Answer grounded in the graph.",
        sources_json=json.dumps(saved),
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)

    response = _message_response_from_row(message)

    assert response.graph is not None
    assert response.graph.relationship_count == 2
    assert response.graph.entities == ["Application Backend", "Primary Database"]


def test_entity_context_wrapper_returns_prompt_text(db_session, user, graph_documents):
    _index(graph_documents, db_session, user)

    context = get_entity_context(
        "What does the Public API connect to?",
        str(user.id),
        db=db_session,
    )

    assert "Public API" in context

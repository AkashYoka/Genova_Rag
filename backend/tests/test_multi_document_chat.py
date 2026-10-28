from unittest.mock import MagicMock

from app.rag.retrieval import retriever
from app.models import Document


# ── Retrieval: document_ids reaches both vector and BM25 (dev's direct-call retrieve) ──

def _mock_db(monkeypatch, doc_rows):
    mock_db = MagicMock()
    mock_db.__enter__.return_value = mock_db
    mock_query = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value.all.return_value = doc_rows
    monkeypatch.setattr("app.database.SessionLocal", lambda: mock_db)


def test_retrieve_forwards_document_ids_to_vector_and_bm25(monkeypatch):
    """Single document_ids list (len==1) still forwards as a filter list."""
    _mock_db(monkeypatch, [("doc-a",)])

    seen = {"vector": "unset", "bm25": "unset"}

    monkeypatch.setattr(retriever, "transform_query", lambda _q: ["q"])
    monkeypatch.setattr(retriever, "embed_query", lambda q: f"embedding:{q}")
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)

    def fake_query_chunks(query_embedding, user_id, document_id=None, document_ids=None, top_k=10):
        seen["vector"] = document_ids
        return [{"id": "v1", "text": "vec", "filename": "a.pdf", "page": 1, "score": 0.5}]

    def fake_query_bm25(query, user_id, document_id=None, document_ids=None, top_k=10):
        seen["bm25"] = document_ids
        return [{"id": "b1", "text": "bm", "filename": "b.pdf", "page": 1, "score": 0.5}]

    monkeypatch.setattr(retriever, "query_chunks", fake_query_chunks)
    monkeypatch.setattr("app.rag.retrieval.bm25.query_bm25", fake_query_bm25)

    retriever.retrieve("question", user_id="user-1", document_ids=["doc-a"])

    assert seen["vector"] == ["doc-a"]
    if seen["bm25"] != "unset":
        assert seen["bm25"] == ["doc-a"]


def test_retrieve_multi_doc_fetches_per_document(monkeypatch):
    """2+ document_ids → per-doc retrieve so each PDF can contribute."""
    _mock_db(monkeypatch, [("doc-a",), ("doc-b",)])

    seen_ids = []

    monkeypatch.setattr(retriever, "transform_query", lambda _q: ["q"])
    monkeypatch.setattr(retriever, "embed_query", lambda q: f"embedding:{q}")
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)

    def fake_query_chunks(query_embedding, user_id, document_id=None, document_ids=None, top_k=10):
        seen_ids.append(document_id)
        return [
            {
                "id": f"chunk-{document_id}",
                "text": f"text for {document_id}",
                "filename": f"{document_id}.pdf",
                "page": 1,
                "score": 0.9 if document_id == "doc-a" else 0.1,
                "document_id": document_id,
            }
        ]

    monkeypatch.setattr(retriever, "query_chunks", fake_query_chunks)
    monkeypatch.setattr(retriever.settings, "USE_HYBRID_SEARCH", False)

    results = retriever.retrieve(
        "summarize",
        user_id="user-1",
        document_ids=["doc-a", "doc-b"],
    )

    assert seen_ids == ["doc-a", "doc-b"]
    result_docs = {c.get("document_id") for c in results}
    assert result_docs == {"doc-a", "doc-b"}


def test_retrieve_single_document_leaves_document_ids_none(monkeypatch):
    _mock_db(monkeypatch, [("doc-a",)])

    seen = {"vector_id": "unset", "vector_ids": "unset"}

    monkeypatch.setattr(retriever, "transform_query", lambda _q: ["q"])
    monkeypatch.setattr(retriever, "embed_query", lambda q: f"embedding:{q}")
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)

    def fake_query_chunks(query_embedding, user_id, document_id=None, document_ids=None, top_k=10):
        seen["vector_id"] = document_id
        seen["vector_ids"] = document_ids
        return [{"id": "v1", "text": "vec", "filename": "a.pdf", "page": 1, "score": 0.5}]

    monkeypatch.setattr(retriever, "query_chunks", fake_query_chunks)

    retriever.retrieve("question", user_id="user-1", document_id="doc-a")

    assert seen["vector_id"] == "doc-a"
    assert seen["vector_ids"] is None


# ── Summarize-attached intent uses Document.summary ──

def test_summarize_attached_uses_document_summaries(monkeypatch):
    from app.rag.llm import agent

    docs = [
        {"id": "inv", "filename": "invoice.pdf", "summary": "Invoice for $42 from Acme."},
        {"id": "chem", "filename": "chemistry-text.pdf", "summary": "OpenStax chemistry overview."},
        {"id": "math", "filename": "math-calculus.pdf", "summary": "Calculus textbook front matter."},
    ]
    monkeypatch.setattr(agent, "_load_attached_document_rows", lambda user_id, ids: docs)

    called_retrieve = {"n": 0}

    def boom(*a, **k):
        called_retrieve["n"] += 1
        raise AssertionError("retrieve should not run when summaries exist")

    monkeypatch.setattr(agent, "retrieve", boom)

    class FakeChoice:
        def __init__(self, content):
            self.message = MagicMock(content=content)

    class FakeResp:
        choices = [FakeChoice(
            "1. invoice.pdf — Invoice for $42 from Acme.\n"
            "2. chemistry-text.pdf — OpenStax chemistry overview.\n"
            "3. math-calculus.pdf — Calculus textbook front matter."
        )]

    class FakeClient:
        last_usage = None

        def chat_completion(self, **kwargs):
            prompt = kwargs["messages"][1]["content"]
            assert "invoice.pdf" in prompt
            assert "chemistry-text.pdf" in prompt
            assert "math-calculus.pdf" in prompt
            assert "Invoice for $42" in prompt
            return FakeResp()

    monkeypatch.setattr(agent, "get_llm_client", lambda llm_provider=None: FakeClient())

    result = agent._generate_answer_direct(
        question="summarize all attached",
        user_id="user-1",
        document_ids=["inv", "chem", "math"],
    )

    assert called_retrieve["n"] == 0
    assert "invoice.pdf" in result["answer"]
    assert "chemistry-text.pdf" in result["answer"]
    assert "math-calculus.pdf" in result["answer"]
    filenames = {s["filename"] for s in result["sources"]}
    assert filenames == {"invoice.pdf", "chemistry-text.pdf", "math-calculus.pdf"}


def test_is_summarize_attached_intent_markers():
    from app.rag.llm.agent import _is_summarize_attached_intent

    assert _is_summarize_attached_intent("Please summarize all attached documents")
    assert _is_summarize_attached_intent("what are these documents?")
    assert not _is_summarize_attached_intent("What dose does the invoice list?")


# ── Prompt: comparison guidance only when more than one document ──

def test_comparison_guidance_present_only_for_multiple_documents(monkeypatch):
    from app.rag.llm import agent
    from app.rag.prompts.prompts import MULTI_DOC_COMPARISON_GUIDANCE

    captured = {}

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

    def capture_prompt(llm, tools, prompt):
        captured["template"] = prompt.template
        return "agent"

    monkeypatch.setattr(agent, "get_llm_client", lambda hf_token=None: FakeLLM())
    monkeypatch.setattr(agent, "create_react_agent", capture_prompt)
    monkeypatch.setattr(agent, "AgentExecutor", lambda **kwargs: kwargs)

    agent.get_agent_executor(user_id="user-1", document_ids=["doc-a", "doc-b"])
    assert MULTI_DOC_COMPARISON_GUIDANCE.strip() in captured["template"]

    captured.clear()
    agent.get_agent_executor(user_id="user-1", document_id="doc-a")
    assert MULTI_DOC_COMPARISON_GUIDANCE.strip() not in captured["template"]


# ── Route guard: ownership + readiness for document_ids ──

def test_chat_ask_multi_doc_success(client, auth_headers, ready_document, db_session, user, monkeypatch):
    second = Document(
        user_id=user.id,
        filename="second.txt",
        original_name="second.txt",
        file_size=128,
        status="ready",
    )
    db_session.add(second)
    db_session.commit()
    db_session.refresh(second)

    monkeypatch.setattr(
        "app.routes.chat.generate_answer",
        lambda question, user_id, document_id=None, document_ids=None, **kwargs: {
            "answer": "Across both docs",
            "sources": [],
        },
    )

    response = client.post(
        "/api/v1/chat/ask",
        headers=auth_headers,
        json={"question": "Compare them", "document_ids": [ready_document.id, second.id]},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Across both docs"


def test_chat_ask_multi_doc_rejects_missing_document(client, auth_headers, ready_document):
    response = client.post(
        "/api/v1/chat/ask",
        headers=auth_headers,
        json={"question": "Compare", "document_ids": [ready_document.id, "missing-doc-id"]},
    )
    assert response.status_code == 404


def test_chat_ask_multi_doc_rejects_not_ready_document(client, auth_headers, ready_document, pending_document):
    response = client.post(
        "/api/v1/chat/ask",
        headers=auth_headers,
        json={"question": "Compare", "document_ids": [ready_document.id, pending_document.id]},
    )
    assert response.status_code == 400


def test_chat_ask_multi_doc_rejects_other_users_document(client, auth_headers, ready_document, db_session, other_user):
    other_doc = Document(
        user_id=other_user.id,
        filename="other.txt",
        original_name="other.txt",
        file_size=64,
        status="ready",
    )
    db_session.add(other_doc)
    db_session.commit()
    db_session.refresh(other_doc)

    response = client.post(
        "/api/v1/chat/ask",
        headers=auth_headers,
        json={"question": "Compare", "document_ids": [ready_document.id, other_doc.id]},
    )
    # not owned -> treated as missing
    assert response.status_code == 404
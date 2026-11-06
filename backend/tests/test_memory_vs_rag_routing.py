"""
Regression tests: conversation-memory vs document RAG routing.

Verifies intent classification and that /chat/ask uses chat history (not corpus)
for memory questions — including when a document is attached.
"""
from __future__ import annotations

import json

import pytest

from app.services.conversation_memory import (
    QueryIntent,
    classify_query_intent,
    history_likely_answers,
    is_capability_question,
    is_chat_memory_question,
    is_personal_fact_statement,
    resolve_deterministic_memory_answer,
)


HISTORY_WITH_PROJECT = [
    {"role": "user", "content": "My project is named PersistenceTest and I am testing conversation memory."},
    {"role": "assistant", "content": "Got it."},
]

HISTORY_WITH_INVOICE = [
    {
        "role": "user",
        "content": "What is the total invoice amount including taxes?",
    },
    {
        "role": "assistant",
        "content": "The total is Rs 6,41,920.00.",
        "sources": [
            {
                "chunks": [
                    {
                        "text": "Grand Total 641920",
                        "filename": "02_Vendor_Invoice_Services_IGST.pdf",
                        "page": 1,
                        "score": 0.9,
                        "confidence": 90,
                    }
                ]
            }
        ],
    },
]


# ── Unit: intent classifier ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,has_attach,expected",
    [
        ("Hello", True, QueryIntent.CAPABILITY),
        ("What can you do?", True, QueryIntent.CAPABILITY),
        ("What is my project name?", False, QueryIntent.CHAT_MEMORY),
        ("What is my project name?", True, QueryIntent.CHAT_MEMORY),
        ("What was my first question in this conversation?", False, QueryIntent.CHAT_MEMORY),
        ("Summarize everything we discussed so far.", False, QueryIntent.CHAT_MEMORY),
        ("What is the IGST amount on that invoice?", True, QueryIntent.DOCUMENT),
        ("What does the uploaded document say about taxable value?", True, QueryIntent.DOCUMENT),
        ("My project is named PersistenceTest.", False, QueryIntent.CHAT_MEMORY),
        ("My project is named PersistenceTest.", True, QueryIntent.CHAT_MEMORY),
    ],
)
def test_classify_query_intent_cases(question, has_attach, expected):
    intent = classify_query_intent(
        question, HISTORY_WITH_PROJECT + HISTORY_WITH_INVOICE, has_attach=has_attach
    )
    assert intent is expected


def test_history_likely_answers_project_name():
    assert history_likely_answers("What is my project name?", HISTORY_WITH_PROJECT)
    assert is_chat_memory_question("What was my first question in this conversation?")
    assert is_chat_memory_question("Summarize everything we discussed so far.")
    assert is_capability_question("Hello!")
    assert is_capability_question("What can you do?")
    assert is_personal_fact_statement("My project is named PersistenceTest.")
    first = resolve_deterministic_memory_answer(
        "What was my first question in this conversation?",
        HISTORY_WITH_PROJECT,
    )
    assert first is not None
    assert "PersistenceTest" in first


# ── Integration: ask path uses memory without calling retrieve ────────────────


def _create_session(client, auth_headers, title):
    resp = client.post(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        json={"title": title},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _ask(client, auth_headers, question, session_id, **extra):
    return client.post(
        "/api/v1/chat/ask",
        headers=auth_headers,
        json={"question": question, "session_id": session_id, **extra},
    )


def _force_ollama_direct(monkeypatch):
    """Unit tests must not hit the HuggingFace agent path."""
    monkeypatch.setattr(
        "app.rag.llm.agent.normalize_llm_provider", lambda provider=None: "ollama"
    )


def test_project_name_memory_without_and_with_document(
    client, auth_headers, ready_document, monkeypatch
):
    """Scenarios 1 and 4: PersistenceTest recalled even with a PDF attached."""
    _force_ollama_direct(monkeypatch)
    retrieve_calls = {"n": 0}

    def boom_retrieve(*_a, **_k):
        retrieve_calls["n"] += 1
        return []

    monkeypatch.setattr("app.rag.llm.agent.retrieve", boom_retrieve)
    monkeypatch.setattr(
        "app.rag.llm.agent._collect_graph_evidence", lambda *a, **k: None
    )

    def memory_aware_generate(question, user_id, document_id=None, chat_history=None, **kwargs):
        # Route through the real agent entry so intent logic runs.
        from app.rag.llm.agent import generate_answer as real_generate

        return real_generate(
            question,
            user_id,
            document_id=document_id,
            document_ids=kwargs.get("document_ids"),
            top_k=kwargs.get("top_k"),
            chat_history=chat_history,
            llm_provider=kwargs.get("llm_provider"),
        )

    # Stub LLM client used by capability/memory answers
    class _FakeChoice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _FakeResp:
        def __init__(self, text):
            self.choices = [_FakeChoice(text)]

    class _FakeClient:
        last_usage = {"provider": "ollama", "prompt_tokens": 1, "completion_tokens": 1}

        def chat_completion(self, messages, **kwargs):
            user = messages[-1]["content"]
            if "project" in user.lower() and "PersistenceTest" in user:
                return _FakeResp("Your project is named PersistenceTest.")
            if "PersistenceTest" in user:
                return _FakeResp("PersistenceTest")
            if "first question" in user.lower():
                return _FakeResp(
                    "Your first question was about the PersistenceTest project statement."
                )
            if "summarize" in user.lower() or "discussed" in user.lower():
                return _FakeResp(
                    "We discussed that your project is named PersistenceTest."
                )
            if "capability" in user.lower() or "what can you do" in user.lower() or "Hello" in user:
                return _FakeResp(
                    "I can analyze PDFs, answer with citations, and remember chat facts."
                )
            # Default: echo that history was provided
            return _FakeResp("Answered from chat history.")

    monkeypatch.setattr("app.rag.llm.agent.get_llm_client", lambda **k: _FakeClient())
    # Use route-level generate_answer which delegates to agent
    monkeypatch.setattr("app.routes.chat.generate_answer", memory_aware_generate)

    session_id = _create_session(client, auth_headers, "MemoryRoute")

    r0 = _ask(
        client,
        auth_headers,
        "My project is named PersistenceTest and I am testing conversation memory.",
        session_id,
    )
    assert r0.status_code == 200
    # Seed may hit GENERAL retrieve; memory asks must not.
    retrieve_calls["n"] = 0

    # Scenario 1
    r1 = _ask(client, auth_headers, "What is my project name?", session_id)
    assert r1.status_code == 200
    assert "PersistenceTest" in r1.json()["answer"]
    assert retrieve_calls["n"] == 0

    # Scenario 2
    r2 = _ask(
        client,
        auth_headers,
        "What was my first question in this conversation?",
        session_id,
    )
    assert r2.status_code == 200
    # First user turn was the PersistenceTest statement (not later questions).
    assert "PersistenceTest" in r2.json()["answer"]
    assert retrieve_calls["n"] == 0

    # Scenario 3
    r3 = _ask(
        client,
        auth_headers,
        "Summarize everything we discussed so far.",
        session_id,
    )
    assert r3.status_code == 200
    assert "PersistenceTest" in r3.json()["answer"]
    assert retrieve_calls["n"] == 0

    # Scenario 4 — PDF attached must NOT disable memory
    r4 = _ask(
        client,
        auth_headers,
        "What is my project name?",
        session_id,
        document_id=ready_document.id,
    )
    assert r4.status_code == 200
    assert "PersistenceTest" in r4.json()["answer"]
    assert retrieve_calls["n"] == 0


def test_document_question_still_uses_rag_with_attach(
    client, auth_headers, ready_document, monkeypatch
):
    """Scenario 5: invoice question with PDF attached must retrieve."""
    _force_ollama_direct(monkeypatch)
    retrieve_calls = {"n": 0}

    def fake_retrieve(*_a, **_k):
        retrieve_calls["n"] += 1
        return [
            {
                "text": "IGST Rs 97,920.00 on taxable value Rs 5,44,000.00",
                "filename": "ready.txt",
                "page": 1,
                "score": 0.95,
                "confidence": 95.0,
            }
        ]

    class _FakeChoice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _FakeResp:
        def __init__(self, text):
            self.choices = [_FakeChoice(text)]

    class _FakeClient:
        last_usage = None

        def chat_completion(self, messages, **kwargs):
            return _FakeResp(
                "The IGST amount is Rs 97,920.00 [Source: ready.txt, Page 1]."
            )

    monkeypatch.setattr("app.rag.llm.agent.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "app.rag.llm.agent._collect_graph_evidence", lambda *a, **k: None
    )
    monkeypatch.setattr("app.rag.llm.agent.get_llm_client", lambda **k: _FakeClient())

    def via_agent(question, user_id, document_id=None, chat_history=None, **kwargs):
        from app.rag.llm.agent import generate_answer as real_generate

        return real_generate(
            question,
            user_id,
            document_id=document_id,
            document_ids=kwargs.get("document_ids"),
            top_k=kwargs.get("top_k"),
            chat_history=chat_history,
            llm_provider=kwargs.get("llm_provider"),
        )

    monkeypatch.setattr("app.routes.chat.generate_answer", via_agent)

    session_id = _create_session(client, auth_headers, "DocRoute")
    _ask(client, auth_headers, "We are looking at the vendor invoice.", session_id)

    r = _ask(
        client,
        auth_headers,
        "What is the IGST amount on that invoice?",
        session_id,
        document_id=ready_document.id,
    )
    assert r.status_code == 200
    assert retrieve_calls["n"] >= 1
    assert "97,920" in r.json()["answer"] or "97920" in r.json()["answer"].replace(",", "")
    assert r.json()["sources"]


def test_capability_not_hijacked_by_attachment(
    client, auth_headers, ready_document, monkeypatch
):
    """Scenarios 6–7: Hello / What can you do ignore attached PDF."""
    _force_ollama_direct(monkeypatch)
    retrieve_calls = {"n": 0}
    monkeypatch.setattr(
        "app.rag.llm.agent.retrieve",
        lambda *a, **k: retrieve_calls.__setitem__("n", retrieve_calls["n"] + 1) or [],
    )
    monkeypatch.setattr(
        "app.rag.llm.agent._collect_graph_evidence", lambda *a, **k: None
    )

    class _FakeChoice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _FakeResp:
        def __init__(self, text):
            self.choices = [_FakeChoice(text)]

    class _FakeClient:
        last_usage = None

        def chat_completion(self, messages, **kwargs):
            return _FakeResp(
                "I can help you upload and analyze PDFs with citations."
            )

    monkeypatch.setattr("app.rag.llm.agent.get_llm_client", lambda **k: _FakeClient())

    def via_agent(question, user_id, document_id=None, chat_history=None, **kwargs):
        from app.rag.llm.agent import generate_answer as real_generate

        return real_generate(
            question,
            user_id,
            document_id=document_id,
            chat_history=chat_history,
            llm_provider=kwargs.get("llm_provider"),
        )

    monkeypatch.setattr("app.routes.chat.generate_answer", via_agent)
    session_id = _create_session(client, auth_headers, "CapabilityRoute")

    for q in ("Hello", "What can you do?"):
        r = _ask(
            client,
            auth_headers,
            q,
            session_id,
            document_id=ready_document.id,
        )
        assert r.status_code == 200
        body = r.json()["answer"].lower()
        assert "pdf" in body or "citation" in body or "document" in body
        assert "igst" not in body
        assert retrieve_calls["n"] == 0


def test_session_isolation_and_stream_memory(client, auth_headers, monkeypatch):
    """Scenarios 9 and 12: isolation + streaming memory path."""
    _force_ollama_direct(monkeypatch)
    retrieve_calls = {"n": 0}

    def soft_retrieve(*_a, **_k):
        retrieve_calls["n"] += 1
        return []

    monkeypatch.setattr("app.rag.llm.agent.retrieve", soft_retrieve)
    monkeypatch.setattr(
        "app.rag.llm.agent._collect_graph_evidence", lambda *a, **k: None
    )

    class _FakeChoice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _FakeResp:
        def __init__(self, text):
            self.choices = [_FakeChoice(text)]

    class _FakeClient:
        last_usage = None

        def chat_completion(self, messages, **kwargs):
            user = messages[-1]["content"].lower()
            if "postgresql" in user and "mysql" not in user.split("current question")[-1]:
                # history may contain postgres; answer based on current q + history
                pass
            hist = messages[-1]["content"]
            if "mysql" in hist.lower() and "which database" in hist.lower():
                if "postgresql" in hist.lower().split("current question")[0] and "mysql" in hist.lower().split("current question")[0]:
                    # both shouldn't be in same session history
                    pass
            cur = hist.lower().split("current question:")[-1]
            prior = hist.lower().split("current question:")[0]
            if "database" in cur:
                if "postgresql" in prior and "mysql" not in prior:
                    return _FakeResp("PostgreSQL")
                if "mysql" in prior and "postgresql" not in prior:
                    return _FakeResp("MySQL")
            return _FakeResp("Unknown")

    monkeypatch.setattr("app.rag.llm.agent.get_llm_client", lambda **k: _FakeClient())

    def via_agent(question, user_id, document_id=None, chat_history=None, **kwargs):
        from app.rag.llm.agent import generate_answer as real_generate

        return real_generate(
            question, user_id, chat_history=chat_history, llm_provider=kwargs.get("llm_provider")
        )

    def via_stream(question, user_id, document_id=None, chat_history=None, **kwargs):
        from app.rag.llm.agent import generate_answer_stream as real_stream

        yield from real_stream(
            question, user_id, chat_history=chat_history, llm_provider=kwargs.get("llm_provider")
        )

    monkeypatch.setattr("app.routes.chat.generate_answer", via_agent)
    monkeypatch.setattr("app.routes.chat.generate_answer_stream", via_stream)

    a = _create_session(client, auth_headers, "IsoA")
    b = _create_session(client, auth_headers, "IsoB")
    assert _ask(client, auth_headers, "My application database is PostgreSQL.", a).status_code == 200
    assert _ask(client, auth_headers, "My application database is MySQL.", b).status_code == 200

    before_memory = retrieve_calls["n"]
    ra = _ask(client, auth_headers, "What is my application database?", a)
    rb = _ask(client, auth_headers, "What is my application database?", b)
    assert ra.status_code == 200 and rb.status_code == 200
    assert "PostgreSQL" in ra.json()["answer"]
    assert "MySQL" in rb.json()["answer"]
    assert "MySQL" not in ra.json()["answer"]
    assert "PostgreSQL" not in rb.json()["answer"]
    # Memory asks must not retrieve
    assert retrieve_calls["n"] == before_memory

    # Streaming memory
    stream = client.post(
        "/api/v1/chat/ask/stream",
        headers=auth_headers,
        json={
            "question": "What database did I mention earlier?",
            "session_id": a,
        },
    )
    assert stream.status_code == 200
    assert "PostgreSQL" in stream.text
    assert retrieve_calls["n"] == before_memory

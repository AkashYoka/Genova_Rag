"""
Integration tests for database-backed conversation persistence.

Scenarios cover memory, isolation, streaming, keywords, failures, and concurrency.
LLM generation is mocked; persistence and history loading use the real DB.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import sessionmaker

from app.models import ChatMessage, ChatSession, Document
from app.rag.llm.agent import persist_document_keywords
from app.services.conversation_memory import (
    build_prompt_history,
    is_chat_memory_question,
    is_conversation_followup,
    load_prompt_history,
    normalize_role,
    persist_chat_message,
)


def _create_session(client, auth_headers, title="Test Chat"):
    resp = client.post(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        json={"title": title},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _history(client, auth_headers, session_id):
    resp = client.get(
        f"/api/v1/chat/history/session/{session_id}",
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _ask(client, auth_headers, question, session_id, **extra):
    body = {"question": question, "session_id": session_id, **extra}
    resp = client.post("/api/v1/chat/ask", headers=auth_headers, json=body)
    return resp


def _memory_aware_generate(question, user_id, document_id=None, chat_history=None, **kwargs):
    """Mock LLM that answers from chat_history when facts are present."""
    history = chat_history or []
    q = (question or "").lower()
    joined = "\n".join(
        f"{m.get('role')}: {m.get('content')}" for m in history if m.get("content")
    ).lower()

    if "what database" in q or "which database" in q:
        if "postgresql" in joined:
            return {"answer": "Your project uses PostgreSQL.", "sources": []}
        if "mysql" in joined:
            return {"answer": "Your project uses MySQL.", "sources": []}
        return {"answer": "I do not know which database from this chat.", "sources": []}

    if "retrieval stage" in q or "retrieval" in q and "detail" in q:
        if "rag" in joined:
            return {
                "answer": (
                    "In the RAG discussion, the retrieval stage fetches the most "
                    "relevant document chunks for the query before generation."
                ),
                "sources": [],
            }
        return {"answer": "No prior RAG context found.", "sources": []}

    if "what did i ask previously" in q or "what did i ask" in q:
        users = [m.get("content") for m in history if m.get("role") == "user"]
        prev = users[-1] if users else "(none)"
        return {"answer": f"You previously asked: {prev}", "sources": []}

    if "previous answer" in q:
        assistants = [m.get("content") for m in history if m.get("role") == "assistant"]
        prev = assistants[-1] if assistants else "(none)"
        return {"answer": f"My previous answer was: {prev}", "sources": []}

    if "technology" in q and ("mention" in q or "earlier" in q):
        if "postgresql" in joined:
            return {"answer": "You mentioned PostgreSQL earlier.", "sources": []}
        return {"answer": "No technology found in history.", "sources": []}

    if "summarize our conversation" in q or "summarise our conversation" in q:
        return {
            "answer": f"Summary of {len(history)} prior turns: {joined[:400]}",
            "sources": [],
        }

    if "what can you do" in q:
        return {
            "answer": "I can analyze PDFs, answer with citations, and keep chat memory.",
            "sources": [],
        }

    if "fact-0" in q or "beginning fact" in q or "first secret" in q:
        if "secret-alpha-42" in joined:
            return {"answer": "The early fact is secret-alpha-42.", "sources": []}
        return {"answer": "Early fact not found in history.", "sources": []}

    if "follow-up about the document" in q or "about that document" in q:
        sources = []
        for m in reversed(history):
            if m.get("role") == "assistant" and m.get("sources"):
                src = m["sources"]
                if isinstance(src, dict):
                    sources = src.get("chunks") or src.get("sources") or []
                elif isinstance(src, list):
                    sources = src
                break
        return {
            "answer": "Continuing from the prior document discussion.",
            "sources": sources if isinstance(sources, list) else [],
        }

    # Echo acknowledgements for statements / first turns
    if document_id and "invoice" in q:
        return {
            "answer": "The invoice total is $1200.",
            "sources": [
                {
                    "text": "Invoice total $1200",
                    "filename": "invoice.pdf",
                    "page": 1,
                    "score": 0.9,
                    "confidence": 90.0,
                }
            ],
        }

    return {"answer": f"Acknowledged: {question}", "sources": []}


# ── Unit helpers ──────────────────────────────────────────────────────────────


def test_normalize_role_mapping():
    assert normalize_role("USER") == "user"
    assert normalize_role("human") == "user"
    assert normalize_role("AI") == "assistant"
    assert normalize_role("bot") == "assistant"
    assert normalize_role("weird") == "assistant"


def test_chat_memory_and_followup_detection():
    assert is_chat_memory_question("What did I ask previously?")
    assert is_chat_memory_question("Summarize our conversation")
    assert is_conversation_followup("Explain the retrieval stage in more detail")
    assert is_conversation_followup("Tell me more about that")
    assert not is_conversation_followup("What is photosynthesis?")


def test_should_inject_prompt_history_gates_self_contained_cases():
    from app.services.conversation_memory import (
        is_self_contained_grounded_question,
        should_inject_prompt_history,
    )

    table_q = (
        "According to Table 1 for patients 12 years and older with asthma, "
        "what dose and frequency apply for pretreatment IgE 350 IU/mL and "
        "body weight 78 kg?"
    )
    case_q = (
        "Asthma patient >=12 years, IgE 300 IU/mL, weight 70 kg. "
        "What is the label dose and frequency from the dosing table?"
    )
    label_q = (
        "According to the prescribing information, what is the recommended "
        "dose and frequency for neovascular (wet) AMD?"
    )
    follow_q = "What about that dose?"

    assert is_self_contained_grounded_question(table_q)
    assert is_self_contained_grounded_question(case_q)
    assert is_self_contained_grounded_question(label_q)
    assert not is_self_contained_grounded_question(follow_q)

    assert should_inject_prompt_history(table_q, has_attach=True) is False
    assert should_inject_prompt_history(case_q, has_attach=True) is False
    assert should_inject_prompt_history(label_q, has_attach=True) is False
    assert should_inject_prompt_history(follow_q, has_attach=True) is True
    assert should_inject_prompt_history(
        "What did I ask previously?", has_attach=True
    ) is True


def test_build_prompt_history_keeps_early_facts():
    messages = []
    for i in range(40):
        messages.append({"role": "user", "content": f"filler question {i}?"})
        messages.append({"role": "assistant", "content": f"filler answer {i}"})
    messages.insert(0, {"role": "user", "content": "My codename is secret-alpha-42"})
    messages.insert(1, {"role": "assistant", "content": "Noted."})
    block = build_prompt_history(messages, recent_limit=10)
    assert "secret-alpha-42" in block
    assert "Earlier facts" in block


# ── Scenario 1 – Basic conversation memory ────────────────────────────────────


def test_scenario1_basic_conversation_memory(client, auth_headers, monkeypatch, db_session, user):
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_id = _create_session(client, auth_headers, "S1")

    r1 = _ask(client, auth_headers, "My project uses PostgreSQL.", session_id)
    assert r1.status_code == 200

    r2 = _ask(client, auth_headers, "What database does my project use?", session_id)
    assert r2.status_code == 200
    assert "PostgreSQL" in r2.json()["answer"]

    hist = _history(client, auth_headers, session_id)
    messages = hist["messages"] if isinstance(hist, dict) else hist
    roles = [m["role"] for m in messages]
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 2
    db_rows = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id, ChatMessage.user_id == user.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    assert len(db_rows) >= 4


# ── Scenario 2 – Follow-up context ────────────────────────────────────────────


def test_scenario2_followup_context(client, auth_headers, monkeypatch):
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_id = _create_session(client, auth_headers, "S2")
    assert _ask(client, auth_headers, "Explain how RAG works.", session_id).status_code == 200
    r = _ask(
        client,
        auth_headers,
        "Explain the retrieval stage in more detail.",
        session_id,
    )
    assert r.status_code == 200
    assert "retrieval" in r.json()["answer"].lower()
    assert "rag" in r.json()["answer"].lower()


# ── Scenario 3 – Persistence after "restart" ──────────────────────────────────


def test_scenario3_persistence_after_restart(client, auth_headers, monkeypatch, db_session, user):
    """Simulate restart by relying only on DB rows (no process memory)."""
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_id = _create_session(client, auth_headers, "S3")
    _ask(client, auth_headers, "My project uses PostgreSQL.", session_id)
    _ask(client, auth_headers, "Remember that for later.", session_id)

    # "Restart": drop any ORM identity map and reload history purely from DB
    db_session.expire_all()
    dicts, formatted = load_prompt_history(
        db_session, user_id=str(user.id), session_id=str(session_id)
    )
    assert any("PostgreSQL" in (m.get("content") or "") for m in dicts)
    assert "PostgreSQL" in formatted

    r = _ask(client, auth_headers, "What database does my project use?", session_id)
    assert "PostgreSQL" in r.json()["answer"]


# ── Scenario 4 – Multiple independent conversations ───────────────────────────


def test_scenario4_conversation_isolation(client, auth_headers, monkeypatch):
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_a = _create_session(client, auth_headers, "Conv A")
    session_b = _create_session(client, auth_headers, "Conv B")

    assert _ask(client, auth_headers, "My project uses PostgreSQL.", session_a).status_code == 200
    assert _ask(client, auth_headers, "My project uses MySQL.", session_b).status_code == 200

    a = _ask(client, auth_headers, "What database does my project use?", session_a)
    b = _ask(client, auth_headers, "What database does my project use?", session_b)
    assert "PostgreSQL" in a.json()["answer"]
    assert "MySQL" in b.json()["answer"]
    assert "MySQL" not in a.json()["answer"]
    assert "PostgreSQL" not in b.json()["answer"]


# ── Scenario 5 – Long conversation keeps early facts ──────────────────────────


def test_scenario5_long_conversation_early_facts(client, auth_headers, monkeypatch, db_session, user):
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_id = _create_session(client, auth_headers, "S5")

    base = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = []
    rows.append(
        ChatMessage(
            user_id=user.id,
            session_id=session_id,
            role="user",
            content="My first secret is secret-alpha-42",
            created_at=base,
        )
    )
    rows.append(
        ChatMessage(
            user_id=user.id,
            session_id=session_id,
            role="assistant",
            content="Noted the secret.",
            created_at=base + timedelta(seconds=1),
        )
    )
    for i in range(30):
        rows.append(
            ChatMessage(
                user_id=user.id,
                session_id=session_id,
                role="user",
                content=f"Filler question number {i}?",
                created_at=base + timedelta(seconds=2 + i * 2),
            )
        )
        rows.append(
            ChatMessage(
                user_id=user.id,
                session_id=session_id,
                role="assistant",
                content=f"Filler answer number {i}.",
                created_at=base + timedelta(seconds=3 + i * 2),
            )
        )
    db_session.add_all(rows)
    db_session.commit()

    dicts, formatted = load_prompt_history(
        db_session, user_id=str(user.id), session_id=str(session_id)
    )
    assert "secret-alpha-42" in formatted
    assert len(dicts) >= 60

    r = _ask(
        client,
        auth_headers,
        "What was my first secret mentioned at the beginning?",
        session_id,
    )
    assert r.status_code == 200
    assert "secret-alpha-42" in r.json()["answer"]


# ── Scenario 6 – Chat memory questions ────────────────────────────────────────


def test_scenario6_chat_memory_questions(client, auth_headers, monkeypatch):
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_id = _create_session(client, auth_headers, "S6")
    _ask(client, auth_headers, "My project uses PostgreSQL.", session_id)
    _ask(client, auth_headers, "What database does my project use?", session_id)

    q1 = _ask(client, auth_headers, "What did I ask previously?", session_id)
    assert "database" in q1.json()["answer"].lower()

    q2 = _ask(client, auth_headers, "What was your previous answer?", session_id)
    assert "postgresql" in q2.json()["answer"].lower() or "previous answer" in q2.json()["answer"].lower()

    q3 = _ask(client, auth_headers, "What technology did I mention earlier?", session_id)
    assert "PostgreSQL" in q3.json()["answer"]

    q4 = _ask(client, auth_headers, "Summarize our conversation.", session_id)
    assert "postgresql" in q4.json()["answer"].lower()


# ── Scenario 7 – Greeting / capability keeps context ──────────────────────────


def test_scenario7_capability_preserves_context(
    client, auth_headers, monkeypatch, ready_document
):
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_id = _create_session(client, auth_headers, "S7")
    r1 = _ask(
        client,
        auth_headers,
        "What is the invoice total?",
        session_id,
        document_id=ready_document.id,
    )
    assert r1.status_code == 200
    assert r1.json()["sources"]

    r2 = _ask(client, auth_headers, "What can you do?", session_id)
    assert r2.status_code == 200
    assert "pdf" in r2.json()["answer"].lower() or "citation" in r2.json()["answer"].lower()

    r3 = _ask(
        client,
        auth_headers,
        "Follow-up about the document we discussed",
        session_id,
    )
    assert r3.status_code == 200
    assert "document" in r3.json()["answer"].lower()


# ── Scenario 8 – Streaming persistence ────────────────────────────────────────


def test_scenario8_streaming_persistence(client, auth_headers, monkeypatch, db_session, user):
    session_id = _create_session(client, auth_headers, "S8")

    def ok_stream(*_a, **_k):
        yield f"data: {json.dumps({'type': 'sources', 'data': []})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': 'Hello '})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': 'stream'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    monkeypatch.setattr("app.routes.chat.generate_answer_stream", ok_stream)
    resp = client.post(
        "/api/v1/chat/ask/stream",
        headers=auth_headers,
        json={"question": "Say hello via stream", "session_id": session_id},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "Hello" in body
    assert "stream" in body

    rows = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    assert any(m.role == "user" and "stream" in m.content for m in rows)
    assert any(m.role == "assistant" and "Hello stream" in m.content for m in rows)


def test_scenario8_failed_stream_does_not_persist_assistant(
    client, auth_headers, monkeypatch, db_session, user
):
    session_id = _create_session(client, auth_headers, "S8-fail")

    def bad_stream(*_a, **_k):
        yield f"data: {json.dumps({'type': 'token', 'data': 'partial'})}\n\n"
        yield f"data: {json.dumps({'type': 'error', 'data': 'boom'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    monkeypatch.setattr("app.routes.chat.generate_answer_stream", bad_stream)
    resp = client.post(
        "/api/v1/chat/ask/stream",
        headers=auth_headers,
        json={"question": "This will fail mid-stream", "session_id": session_id},
    )
    assert resp.status_code == 200
    rows = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .all()
    )
    assert any(m.role == "user" for m in rows)
    assert not any(m.role == "assistant" for m in rows)


def test_scenario8_retry_does_not_duplicate_assistant_from_one_stream(
    client, auth_headers, monkeypatch, db_session
):
    """A single completed stream must create exactly one assistant row."""
    session_id = _create_session(client, auth_headers, "S8-once")
    calls = {"n": 0}

    def once_stream(*_a, **_k):
        calls["n"] += 1
        yield f"data: {json.dumps({'type': 'token', 'data': 'only-once'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    monkeypatch.setattr("app.routes.chat.generate_answer_stream", once_stream)
    resp = client.post(
        "/api/v1/chat/ask/stream",
        headers=auth_headers,
        json={"question": "Unique stream answer", "session_id": session_id},
    )
    assert resp.status_code == 200
    assistants = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id, ChatMessage.role == "assistant")
        .all()
    )
    assert len(assistants) == 1
    assert assistants[0].content == "only-once"
    assert calls["n"] == 1


# ── Scenario 9 – Document context and citations ───────────────────────────────


def test_scenario9_document_citations_retained(
    client, auth_headers, monkeypatch, ready_document, db_session
):
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_id = _create_session(client, auth_headers, "S9")
    r1 = _ask(
        client,
        auth_headers,
        "What is the invoice total?",
        session_id,
        document_id=ready_document.id,
    )
    assert r1.status_code == 200
    assert r1.json()["sources"]
    assert r1.json()["sources"][0]["filename"] == "invoice.pdf"

    assistant = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at.desc())
        .first()
    )
    assert assistant is not None
    assert assistant.sources_json
    parsed = json.loads(assistant.sources_json)
    chunks = parsed.get("chunks") if isinstance(parsed, dict) else parsed
    assert chunks and chunks[0]["filename"] == "invoice.pdf"

    r2 = _ask(
        client,
        auth_headers,
        "Follow-up about the document we discussed",
        session_id,
    )
    assert r2.status_code == 200


# ── Scenario 10 – Database failure handling ───────────────────────────────────


def test_scenario10_db_commit_failure_rolls_back(
    client, auth_headers, monkeypatch, db_session, user
):
    session_id = _create_session(client, auth_headers, "S10")
    before = db_session.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()

    def boom_persist(*_a, **_k):
        raise RuntimeError("simulated commit failure")

    monkeypatch.setattr(
        "app.services.conversation_memory.persist_chat_message", boom_persist
    )
    resp = _ask(client, auth_headers, "This should not persist", session_id)
    assert resp.status_code == 500
    body = resp.json()
    assert "not saved" in (body.get("error", {}).get("message") or body.get("detail") or "").lower() or (
        body.get("error", {}).get("code") == "DATABASE_ERROR"
    ) or "DATABASE_ERROR" in str(body)

    db_session.expire_all()
    after = db_session.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
    assert after == before
    # Ensure we never claimed a successful assistant turn
    assert not any(
        m.role == "assistant"
        for m in db_session.query(ChatMessage).filter(ChatMessage.session_id == session_id)
    )


# ── Scenario 11 – Keyword persistence ─────────────────────────────────────────


def test_scenario11_keyword_persistence(db_session, user):
    doc = Document(
        user_id=user.id,
        filename="kw.txt",
        original_name="kw.txt",
        file_size=10,
        page_count=1,
        chunk_count=1,
        status="ready",
        keywords=json.dumps(["keep-me", "important"]),
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    # Empty chunks must not wipe existing keywords
    persist_document_keywords(doc, [], db_session)
    db_session.commit()
    db_session.refresh(doc)
    assert json.loads(doc.keywords) == ["keep-me", "important"]

    # Malformed / empty text chunks skipped
    persist_document_keywords(
        doc,
        [{"text": ""}, {"content": "   "}, None, 123],
        db_session,
    )
    db_session.commit()
    db_session.refresh(doc)
    assert json.loads(doc.keywords) == ["keep-me", "important"]

    # Normal chunks update keywords
    persist_document_keywords(
        doc,
        [
            {"text": "postgresql database indexing retrieval embeddings vector search"},
            {"content": "postgresql retrieval hybrid search ranking rerank"},
        ],
        db_session,
    )
    db_session.commit()
    db_session.refresh(doc)
    kws = json.loads(doc.keywords)
    assert isinstance(kws, list) and len(kws) > 0


# ── Scenario 12 – Concurrent requests ─────────────────────────────────────────


def test_scenario12_concurrent_requests(client, auth_headers, monkeypatch, db_session, user):
    """
    Concurrent writers against the same session via separate DB sessions.
    (Shared TestClient SQLAlchemy session is not thread-safe; we stress the
    persistence layer directly, then verify API ordering.)
    """
    monkeypatch.setattr("app.routes.chat.generate_answer", _memory_aware_generate)
    session_id = _create_session(client, auth_headers, "S12")
    user_id = str(user.id)
    session_id_s = str(session_id)
    SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=db_session.get_bind())

    def write_pair(i: int):
        s = SessionFactory()
        try:
            persist_chat_message(
                s,
                user_id=user_id,
                session_id=session_id_s,
                role="user",
                content=f"Concurrent fact number {i}",
            )
            persist_chat_message(
                s,
                user_id=user_id,
                session_id=session_id_s,
                role="assistant",
                content=f"Ack concurrent {i}",
            )
            return i
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write_pair, i) for i in range(12)]
        done = [fut.result() for fut in as_completed(futures)]
    assert sorted(done) == list(range(12))

    db_session.expire_all()
    rows = (
        db_session.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id, ChatMessage.user_id == user.id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )
    users = [m for m in rows if m.role == "user"]
    assistants = [m for m in rows if m.role == "assistant"]
    assert len(users) == 12
    assert len(assistants) == 12
    for a, b in zip(rows, rows[1:]):
        assert (a.created_at, str(a.id)) <= (b.created_at, str(b.id))

    # API path still loads ordered history without duplicates for a follow-up
    r = _ask(client, auth_headers, "What did I ask previously?", session_id)
    assert r.status_code == 200


def test_persist_chat_message_skips_empty_assistant(db_session, user):
    session = ChatSession(user_id=user.id, title="empty")
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)

    skipped = persist_chat_message(
        db_session,
        user_id=str(user.id),
        session_id=str(session.id),
        role="assistant",
        content="   ",
    )
    assert skipped is None
    assert db_session.query(ChatMessage).filter(ChatMessage.session_id == session.id).count() == 0

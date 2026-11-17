"""AI processing trace creation, persistence, and admin access control."""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from app.auth import create_access_token, hash_password
from app.models import AITrace, AITraceStep, User, UserRole
from app.services import ai_tracing


def _admin_headers(db_session):
    admin = User(
        username="trace_admin",
        email="trace_admin@example.com",
        hashed_password=hash_password("password123"),
        role=UserRole.admin,
        is_admin=True,
    )
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return admin, {"Authorization": f"Bearer {create_access_token(admin.id)}"}


def test_normalize_trace_id_accepts_uuid_and_generates():
    tid = str(uuid.uuid4())
    assert ai_tracing.normalize_trace_id(tid) == tid
    generated = ai_tracing.normalize_trace_id("not-a-uuid")
    uuid.UUID(generated)


def test_trace_context_steps_and_soft_fail_persist(db_session, monkeypatch):
    monkeypatch.setenv("AI_TRACE_ENABLED", "true")
    monkeypatch.setattr("app.database.SessionLocal", lambda: db_session)
    from app.config import get_settings

    get_settings.cache_clear()

    ctx = ai_tracing.begin_ai_trace(
        question="What is the labeled dose?",
        user_id=str(uuid.uuid4()),
        request_type="rag",
    )
    assert ctx is not None
    tid = ctx.trace_id

    with ai_tracing.trace_step("vector_search_hybrid_retrieval"):
        ai_tracing.record_retrieval(
            search_query="dose",
            search_method="hybrid_rrf",
            chunks=[
                {
                    "id": "c1",
                    "filename": "doc.pdf",
                    "document_id": str(uuid.uuid4()),
                    "score": 0.02,
                    "text": "300 mg every 2 weeks",
                }
            ],
        )
    ai_tracing.record_llm_call(
        model_name="test-model",
        provider="ollama",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        latency_ms=12.5,
        raw_response="300 mg",
    )
    ai_tracing.set_final_answer("300 mg", status="success")
    out = ai_tracing.finish_ai_trace(status="success")
    assert out == tid

    row = db_session.query(AITrace).filter(AITrace.trace_id == tid).first()
    assert row is not None
    assert row.question.startswith("What is the labeled")
    assert row.total_tokens == 15
    steps = (
        db_session.query(AITraceStep)
        .filter(AITraceStep.trace_pk == row.id)
        .all()
    )
    assert any(s.name == "vector_search_hybrid_retrieval" for s in steps)


def test_finish_soft_fails_when_persist_raises(monkeypatch):
    monkeypatch.setenv("AI_TRACE_ENABLED", "true")
    from app.config import get_settings

    get_settings.cache_clear()

    ai_tracing.begin_ai_trace(question="q", user_id=str(uuid.uuid4()))
    with patch.object(ai_tracing, "_persist_trace", side_effect=RuntimeError("db down")):
        tid = ai_tracing.finish_ai_trace(status="success")
    assert tid  # still returns id; does not raise


def test_contextvar_isolation_between_traces():
    a = ai_tracing.begin_ai_trace(question="a", user_id="u1")
    assert ai_tracing.current_trace_id() == a.trace_id
    ai_tracing.clear_ai_trace()
    b = ai_tracing.begin_ai_trace(question="b", user_id="u2")
    assert ai_tracing.current_trace_id() == b.trace_id
    assert a.trace_id != b.trace_id
    ai_tracing.clear_ai_trace()


def test_admin_ai_traces_forbid_regular_user(client, auth_headers):
    r = client.get("/api/v1/admin/ai-traces", headers=auth_headers)
    assert r.status_code == 403


def test_admin_ai_traces_list_and_detail(client, db_session, user):
    _, headers = _admin_headers(db_session)
    tid = str(uuid.uuid4())
    row = AITrace(
        trace_id=tid,
        user_id=user.id,
        question="IgE 350 weight 78 dose?",
        request_type="rag",
        status="success",
        duration_ms=123.4,
        total_tokens=42,
        retrieved_chunk_count=3,
        final_answer="300 mg every 2 weeks",
        final_answer_status="success",
    )
    db_session.add(row)
    db_session.flush()
    db_session.add(
        AITraceStep(
            trace_pk=row.id,
            sequence=1,
            name="llm_request",
            status="success",
            duration_ms=80.0,
        )
    )
    db_session.commit()

    listed = client.get("/api/v1/admin/ai-traces?q=IgE", headers=headers)
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] >= 1
    assert any(item["trace_id"] == tid for item in body["items"])

    detail = client.get(f"/api/v1/admin/ai-traces/{tid}", headers=headers)
    assert detail.status_code == 200
    d = detail.json()
    assert d["trace_id"] == tid
    assert d["question"].startswith("IgE")
    assert len(d["steps"]) == 1
    assert d["steps"][0]["name"] == "llm_request"


def test_ask_returns_trace_id(client, auth_headers, user, monkeypatch):
    monkeypatch.setenv("AI_TRACE_ENABLED", "true")
    from app.config import get_settings

    get_settings.cache_clear()

    def fake_generate_answer(**kwargs):
        return {
            "answer": "Hello",
            "sources": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    with patch("app.routes.chat.generate_answer", side_effect=fake_generate_answer):
        response = client.post(
            "/api/v1/chat/ask",
            headers=auth_headers,
            json={"question": "hi there for tracing"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data.get("trace_id")
    uuid.UUID(data["trace_id"])

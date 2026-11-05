"""Tests for per-user AI usage ledger and APIs."""
from __future__ import annotations

from datetime import datetime, timezone

from app.auth import create_access_token
from app.models import LLMUsage
from app.services.usage_tracker import record_llm_usage, summarize_usage, usage_query_bounds


def test_record_and_summarize_usage(db_session, user):
    row = record_llm_usage(
        db_session,
        user_id=str(user.id),
        usage={
            "input_tokens": 1000,
            "output_tokens": 250,
            "provider": "bedrock",
            "model_id": "us.amazon.nova-2-lite-v1:0",
        },
        usage_type="chat",
        commit=True,
    )
    assert row is not None
    assert row.total_tokens == 1250
    assert row.estimated_cost_usd is not None
    assert row.cost_type == "estimated"
    assert row.user_id == user.id

    start_dt, end_dt = usage_query_bounds(None, None, default_days=7)
    summary = summarize_usage(
        db_session, user_id=str(user.id), start_dt=start_dt, end_dt=end_dt
    )
    assert summary["request_count"] == 1
    assert summary["total_tokens"] == 1250
    assert summary["estimated_cost_usd"] > 0
    assert summary["by_provider"][0]["provider"] == "bedrock"


def test_ollama_usage_is_zero_cost(db_session, user):
    row = record_llm_usage(
        db_session,
        user_id=str(user.id),
        usage={
            "input_tokens": 500,
            "output_tokens": 100,
            "provider": "ollama",
            "model_id": "qwen2.5:3b",
        },
        usage_type="chat",
    )
    assert row is not None
    assert row.estimated_cost_usd == 0.0
    assert row.cost_type == "local_zero"


def test_usage_api_requires_auth(client):
    response = client.get("/api/v1/usage/summary")
    assert response.status_code in (401, 403)


def test_usage_api_returns_only_own_rows(
    client, db_session, user, other_user, auth_headers
):
    record_llm_usage(
        db_session,
        user_id=str(user.id),
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "provider": "ollama",
            "model_id": "qwen2.5:3b",
        },
        usage_type="chat",
    )
    record_llm_usage(
        db_session,
        user_id=str(other_user.id),
        usage={
            "input_tokens": 9999,
            "output_tokens": 9999,
            "provider": "bedrock",
            "model_id": "us.amazon.nova-2-lite-v1:0",
        },
        usage_type="chat",
    )

    summary = client.get("/api/v1/usage/summary", headers=auth_headers)
    assert summary.status_code == 200
    body = summary.json()
    assert body["request_count"] == 1
    assert body["total_tokens"] == 120
    assert body["estimated_cost_usd"] == 0.0

    events = client.get("/api/v1/usage/events", headers=auth_headers)
    assert events.status_code == 200
    payload = events.json()
    assert payload["total"] == 1
    assert payload["items"][0]["provider"] == "ollama"
    assert all(item["provider"] != "bedrock" or item["input_tokens"] != 9999 for item in payload["items"])

    other_headers = {
        "Authorization": f"Bearer {create_access_token(other_user.id)}"
    }
    other_summary = client.get("/api/v1/usage/summary", headers=other_headers)
    assert other_summary.status_code == 200
    assert other_summary.json()["request_count"] == 1
    assert other_summary.json()["total_tokens"] == 19998


def test_usage_daily_fills_calendar(client, db_session, user, auth_headers):
    db_session.add(
        LLMUsage(
            user_id=user.id,
            provider="ollama",
            model_id="qwen2.5:3b",
            usage_type="chat",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.0,
            cost_type="local_zero",
            created_at=datetime.now(timezone.utc),
        )
    )
    db_session.commit()

    start = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    response = client.get(
        f"/api/v1/usage/daily?start={start}&end={start}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    points = response.json()["points"]
    assert len(points) == 1
    assert points[0]["total_tokens"] == 15


def test_usage_from_ollama_helper():
    from app.rag.llm.ollama_client import usage_from_ollama

    usage = usage_from_ollama(
        {"prompt_eval_count": 42, "eval_count": 7},
        model_id="qwen2.5:3b",
    )
    assert usage["input_tokens"] == 42
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 49
    assert usage["estimated_cost_usd"] == 0.0
    assert usage["cost_type"] == "local_zero"

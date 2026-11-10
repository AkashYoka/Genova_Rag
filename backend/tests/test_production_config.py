"""Production configuration fail-closed tests."""
from __future__ import annotations

import pytest

from app.config import Settings


def test_production_rejects_dev_bypass_and_defaults():
    s = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="change-me-in-production-please",
        FIELD_ENCRYPTION_KEY="change-me-in-production-field-encryption-key",
        DEV_AUTH_BYPASS=True,
        ALLOWED_ORIGINS="*",
        ALLOWED_HOSTS="",
    )
    with pytest.raises(ValueError) as exc:
        s.validate_production()
    msg = str(exc.value)
    assert "DEV_AUTH_BYPASS" in msg
    assert "SECRET_KEY" in msg or "insecure default" in msg
    assert "ALLOWED_ORIGINS" in msg or "ALLOWED_HOSTS" in msg


def test_production_accepts_strong_config():
    s = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="x" * 48,
        FIELD_ENCRYPTION_KEY="y" * 48,
        DEV_AUTH_BYPASS=False,
        ALLOWED_ORIGINS="https://app.example.com",
        ALLOWED_HOSTS="api.example.com",
        COOKIE_SECURE=True,
        COOKIE_SAMESITE="lax",
    )
    s.validate_production()  # must not raise


def test_development_allows_missing_hosts():
    s = Settings(
        ENVIRONMENT="development",
        SECRET_KEY="change-me-in-production-please",
        FIELD_ENCRYPTION_KEY="change-me-in-production-field-encryption-key",
        ALLOWED_HOSTS="",
    )
    s.validate_production()  # warns only


def test_production_hides_ollama_llm_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    s = Settings(
        ENVIRONMENT="production",
        LLM_PROVIDER="ollama",
        EMBEDDING_PROVIDER="ollama",
        SECRET_KEY="x" * 48,
        FIELD_ENCRYPTION_KEY="y" * 48,
        ALLOWED_ORIGINS="https://rag.genova-ai.yokaiq.in",
        ALLOWED_HOSTS="rag.genova-ai.yokaiq.in",
    )
    assert s.LLM_PROVIDER == "bedrock"
    assert s.EMBEDDING_PROVIDER == "bedrock"
    assert s.available_llm_providers() == ["bedrock"]
    assert "ollama" not in s.available_llm_providers()

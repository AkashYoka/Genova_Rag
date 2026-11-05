"""Tests for the configurable model pricing registry."""
from __future__ import annotations

import os

import pytest

from app.pricing import (
    enrich_usage_with_pricing,
    estimate_cost_usd,
    lookup_pricing,
    reset_pricing_registry_cache,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_pricing_registry_cache()
    yield
    reset_pricing_registry_cache()
    os.environ.pop("MODEL_PRICING_JSON", None)


def test_ollama_is_local_zero():
    result = estimate_cost_usd(
        provider="ollama",
        model_id="qwen2.5:3b",
        input_tokens=1000,
        output_tokens=200,
    )
    assert result["estimated_cost_usd"] == 0.0
    assert result["cost_type"] == "local_zero"


def test_nova_us_profile_estimate():
    result = estimate_cost_usd(
        provider="bedrock",
        model_id="us.amazon.nova-2-lite-v1:0",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert result["estimated_cost_usd"] == pytest.approx(0.33 + 2.75)
    assert result["cost_type"] == "estimated"
    assert "Not an AWS bill" in (result["pricing_note"] or "")


def test_unknown_model_does_not_invent_cost():
    result = estimate_cost_usd(
        provider="huggingface",
        model_id="some-mystery-model",
        input_tokens=100,
        output_tokens=50,
    )
    assert result["estimated_cost_usd"] is None
    assert result["cost_type"] == "unavailable"


def test_missing_tokens_does_not_invent_cost_for_cloud():
    result = estimate_cost_usd(
        provider="bedrock",
        model_id="us.amazon.nova-2-lite-v1:0",
        input_tokens=None,
        output_tokens=None,
    )
    assert result["estimated_cost_usd"] is None
    assert result["cost_type"] == "unavailable"


def test_env_override_extends_registry(monkeypatch):
    monkeypatch.setenv(
        "MODEL_PRICING_JSON",
        '{"openai":{"gpt-test":{"input_usd_per_1m":1.0,"output_usd_per_1m":2.0,'
        '"cost_type":"estimated","pricing_note":"test rate"}}}',
    )
    reset_pricing_registry_cache()
    rates = lookup_pricing("openai", "gpt-test")
    assert rates is not None
    assert rates["input_usd_per_1m"] == 1.0
    result = estimate_cost_usd(
        provider="openai",
        model_id="gpt-test",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert result["estimated_cost_usd"] == pytest.approx(3.0)


def test_enrich_preserves_provider_reported_cost():
    enriched = enrich_usage_with_pricing(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "estimated_cost_usd": 1.23,
            "cost_type": "provider_reported",
            "provider": "bedrock",
            "model_id": "us.amazon.nova-2-lite-v1:0",
        }
    )
    assert enriched["estimated_cost_usd"] == 1.23
    assert enriched["cost_type"] == "provider_reported"


def test_enrich_adds_cache_tokens_to_total():
    enriched = enrich_usage_with_pricing(
        {
            "inputTokens": 100,
            "outputTokens": 20,
            "cacheReadInputTokens": 50,
            "provider": "bedrock",
            "model_id": "us.amazon.nova-2-lite-v1:0",
        }
    )
    assert enriched["input_tokens"] == 100
    assert enriched["cache_read_tokens"] == 50
    assert enriched["total_tokens"] == 170
    assert enriched["estimated_cost_usd"] is not None

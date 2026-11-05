"""Configurable model pricing registry for estimated AI usage costs.

Rates are USD per 1M tokens. Estimates are never billed amounts.
Override or extend via MODEL_PRICING_JSON (env) without code changes.
"""
from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Official / published on-demand list rates (USD per 1M tokens).
# Nova US cross-region profile matches the project's default BEDROCK_MODEL_ID.
_DEFAULT_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "ollama": {
        "*": {
            "input_usd_per_1m": 0.0,
            "output_usd_per_1m": 0.0,
            "cache_read_usd_per_1m": 0.0,
            "cache_write_usd_per_1m": 0.0,
            "cost_type": "local_zero",
            "pricing_note": "Local Ollama inference — $0 API cost.",
        }
    },
    "bedrock": {
        "us.amazon.nova-2-lite-v1:0": {
            "input_usd_per_1m": 0.33,
            "output_usd_per_1m": 2.75,
            "cache_read_usd_per_1m": 0.0825,
            "cache_write_usd_per_1m": 0.33,
            "cost_type": "estimated",
            "pricing_note": (
                "Estimated using Amazon Bedrock US on-demand list rates for "
                "us.amazon.nova-2-lite-v1:0 ($0.33/1M input + $2.75/1M output). "
                "Not an AWS bill."
            ),
        },
        "amazon.nova-2-lite-v1:0": {
            "input_usd_per_1m": 0.33,
            "output_usd_per_1m": 2.75,
            "cache_read_usd_per_1m": 0.0825,
            "cache_write_usd_per_1m": 0.33,
            "cost_type": "estimated",
            "pricing_note": (
                "Estimated using Amazon Bedrock on-demand list rates for "
                "amazon.nova-2-lite-v1:0. Not an AWS bill."
            ),
        },
        "global.amazon.nova-2-lite-v1:0": {
            "input_usd_per_1m": 0.30,
            "output_usd_per_1m": 2.50,
            "cache_read_usd_per_1m": 0.075,
            "cache_write_usd_per_1m": 0.30,
            "cost_type": "estimated",
            "pricing_note": (
                "Estimated using Amazon Bedrock global on-demand list rates for "
                "global.amazon.nova-2-lite-v1:0 ($0.30/1M input + $2.50/1M output). "
                "Not an AWS bill."
            ),
        },
    },
    "openai": {
        "gpt-4o-mini": {
            "input_usd_per_1m": 0.15,
            "output_usd_per_1m": 0.60,
            "cache_read_usd_per_1m": 0.075,
            "cost_type": "estimated",
            "pricing_note": (
                "Estimated using OpenAI list rates for gpt-4o-mini. Not an OpenAI bill."
            ),
        }
    },
    "huggingface": {
        # Inference Endpoint / Hub pricing varies by deployment — leave unset.
    },
}

_registry_cache: Optional[dict[str, dict[str, dict[str, Any]]]] = None


def _merge_registry(
    base: dict[str, dict[str, dict[str, Any]]],
    override: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    merged = deepcopy(base)
    for provider, models in (override or {}).items():
        if not isinstance(models, dict):
            continue
        bucket = merged.setdefault(str(provider).lower(), {})
        for model_id, rates in models.items():
            if isinstance(rates, dict):
                bucket[str(model_id)] = {**bucket.get(str(model_id), {}), **rates}
    return merged


def get_pricing_registry() -> dict[str, dict[str, dict[str, Any]]]:
    """Return the active pricing registry (defaults + MODEL_PRICING_JSON overrides)."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache

    registry = deepcopy(_DEFAULT_REGISTRY)
    raw = (os.environ.get("MODEL_PRICING_JSON") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                registry = _merge_registry(registry, parsed)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid MODEL_PRICING_JSON ignored: %s", exc)

    _registry_cache = registry
    return registry


def reset_pricing_registry_cache() -> None:
    """Clear cached registry (tests / hot-reload of env overrides)."""
    global _registry_cache
    _registry_cache = None


def lookup_pricing(provider: str | None, model_id: str | None) -> dict[str, Any] | None:
    """Find pricing rates for a provider/model. Supports ``*`` wildcard per provider."""
    if not provider:
        return None
    registry = get_pricing_registry()
    provider_key = provider.strip().lower()
    models = registry.get(provider_key) or {}
    if model_id and model_id in models:
        return models[model_id]
    # Soft match: strip inference-profile region prefix variants already exact-keyed.
    if model_id:
        lowered = model_id.lower()
        for key, rates in models.items():
            if key != "*" and key.lower() == lowered:
                return rates
    if "*" in models:
        return models["*"]
    return None


def estimate_cost_usd(
    *,
    provider: str | None,
    model_id: str | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> dict[str, Any]:
    """Estimate cost from the registry.

    Returns a dict with ``estimated_cost_usd``, ``cost_type``, ``pricing_note``.
    When rates or token counts are unavailable, cost is ``None`` and
    ``cost_type`` is ``unavailable`` (never invents numbers).
    """
    rates = lookup_pricing(provider, model_id)
    if not rates:
        return {
            "estimated_cost_usd": None,
            "cost_type": "unavailable",
            "pricing_note": (
                "No pricing entry for this provider/model; cost not estimated."
            ),
        }

    cost_type = rates.get("cost_type") or "estimated"
    note = rates.get("pricing_note")

    # Local / free providers: $0 when we have a rate entry, even if tokens missing.
    if cost_type == "local_zero":
        return {
            "estimated_cost_usd": 0.0,
            "cost_type": "local_zero",
            "pricing_note": note or "Local inference — $0 API cost.",
        }

    # Need at least input or output token counts to estimate.
    has_tokens = any(
        isinstance(v, int)
        for v in (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
    )
    if not has_tokens:
        return {
            "estimated_cost_usd": None,
            "cost_type": "unavailable",
            "pricing_note": note
            or "Token counts unavailable; cost not estimated.",
        }

    total = 0.0
    if isinstance(input_tokens, int):
        total += input_tokens / 1_000_000 * float(rates.get("input_usd_per_1m") or 0.0)
    if isinstance(output_tokens, int):
        total += output_tokens / 1_000_000 * float(rates.get("output_usd_per_1m") or 0.0)
    if isinstance(cache_read_tokens, int):
        total += cache_read_tokens / 1_000_000 * float(
            rates.get("cache_read_usd_per_1m") or 0.0
        )
    if isinstance(cache_write_tokens, int):
        total += cache_write_tokens / 1_000_000 * float(
            rates.get("cache_write_usd_per_1m")
            or rates.get("input_usd_per_1m")
            or 0.0
        )

    return {
        "estimated_cost_usd": round(total, 8),
        "cost_type": cost_type,
        "pricing_note": note,
    }


def enrich_usage_with_pricing(
    usage: dict[str, Any] | None,
    *,
    provider: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Normalize a provider usage dict and attach estimated cost from the registry.

    Preserves provider-reported costs when ``cost_type == "provider_reported"``.
    Does not invent token counts.
    """
    data = dict(usage or {})
    prov = (provider or data.get("provider") or "").strip().lower() or None
    mid = model_id or data.get("model_id")
    if prov:
        data["provider"] = prov
    if mid:
        data["model_id"] = mid

    # Normalize common aliases
    if "input_tokens" not in data and "inputTokens" in data:
        data["input_tokens"] = data.get("inputTokens")
    if "output_tokens" not in data and "outputTokens" in data:
        data["output_tokens"] = data.get("outputTokens")
    if "total_tokens" not in data and "totalTokens" in data:
        data["total_tokens"] = data.get("totalTokens")
    if "cache_read_tokens" not in data:
        data["cache_read_tokens"] = data.get(
            "cacheReadInputTokens", data.get("cache_read_input_tokens")
        )
    if "cache_write_tokens" not in data:
        data["cache_write_tokens"] = data.get(
            "cacheWriteInputTokens", data.get("cache_write_input_tokens")
        )

    input_tokens = data.get("input_tokens")
    output_tokens = data.get("output_tokens")
    if (
        data.get("total_tokens") is None
        and isinstance(input_tokens, int)
        and isinstance(output_tokens, int)
    ):
        cache_read = data.get("cache_read_tokens")
        extra = cache_read if isinstance(cache_read, int) else 0
        data["total_tokens"] = input_tokens + output_tokens + extra

    if data.get("cost_type") == "provider_reported" and data.get("estimated_cost_usd") is not None:
        return data

    estimate = estimate_cost_usd(
        provider=prov,
        model_id=mid,
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        cache_read_tokens=data.get("cache_read_tokens")
        if isinstance(data.get("cache_read_tokens"), int)
        else None,
        cache_write_tokens=data.get("cache_write_tokens")
        if isinstance(data.get("cache_write_tokens"), int)
        else None,
    )
    data["estimated_cost_usd"] = estimate["estimated_cost_usd"]
    data["cost_type"] = estimate["cost_type"]
    if estimate.get("pricing_note"):
        data["pricing_note"] = estimate["pricing_note"]
    return data

"""Index version resolution for safe reindexing.

``index_version`` is written only after a successful ingestion run. Old
vectors are removed by ``store_chunks`` before the new ones are inserted, so a
version change never leaves duplicate or stale chunks behind.
"""
from __future__ import annotations

from app.config import get_settings


def resolve_index_version(strategy: str | None = None, *, used_fallback: bool = False) -> str:
    settings = get_settings()
    strategy = (strategy or getattr(settings, "CHUNK_STRATEGY", "semantic_llm") or "").lower()
    if strategy == "legacy":
        return "legacy-v1"
    if used_fallback or strategy == "deterministic":
        return str(
            getattr(settings, "SEMANTIC_FALLBACK_VERSION", "semantic-fallback-v1")
            or "semantic-fallback-v1"
        )
    return str(
        getattr(settings, "SEMANTIC_CHUNKING_VERSION", "semantic-llm-v1")
        or "semantic-llm-v1"
    )


def strategy_label() -> str:
    settings = get_settings()
    return str(getattr(settings, "CHUNK_STRATEGY", "semantic_llm") or "semantic_llm").lower()

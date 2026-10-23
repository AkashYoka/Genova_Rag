"""
Document/query embeddings with optional Redis/LRU caching.

Providers:
- ollama (default): nomic-embed-text via local Ollama HTTP API
- huggingface: sentence-transformers via langchain_huggingface
"""
import hashlib
import json
import logging
import os
from typing import List, Optional, Protocol, runtime_checkable

from app.config import get_settings
from app.rag.tracing import trace_call

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Singleton embedding model ─────────────────────────────────────────────────
_embedding_model = None

# ── Embedding cache configuration ────────────────────────────────────────────
_EMBEDDING_CACHE_TTL: int = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))  # 24 h
_EMBEDDING_LRU_MAX_SIZE: int = int(os.getenv("EMBEDDING_LRU_MAX_SIZE", "512"))

# ── LRU in-memory fallback (mirrors cache.py pattern) ────────────────────────
_emb_lru_store: dict = {}
_emb_lru_order: list = []


@runtime_checkable
class _Embedder(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


def _lru_get(key: str) -> Optional[List[float]]:
    raw = _emb_lru_store.get(key)
    return json.loads(raw) if raw is not None else None


def _lru_set(key: str, vector: List[float]) -> None:
    if key in _emb_lru_store:
        _emb_lru_order.remove(key)
    elif len(_emb_lru_store) >= _EMBEDDING_LRU_MAX_SIZE:
        oldest = _emb_lru_order.pop(0)
        del _emb_lru_store[oldest]
    _emb_lru_store[key] = json.dumps(vector)
    _emb_lru_order.append(key)


def _get_redis():
    """Reuse the Redis client already initialised by cache.py."""
    try:
        from app.cache import _get_redis as _cache_get_redis
        return _cache_get_redis()
    except Exception:
        return None


def _make_embedding_key(text: str) -> str:
    """SHA-256 hash of (model_name, text) so keys are model-scoped."""
    raw = f"{settings.EMBEDDING_PROVIDER}:{settings.EMBEDDING_MODEL}:{text.strip()}"
    return "emb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[List[float]]:
    r = _get_redis()
    if r is not None:
        try:
            value = r.get(key)
            if value:
                logger.debug("Embedding cache HIT (Redis) %s", key[:16])
                return json.loads(value)
        except Exception as exc:
            logger.warning("Redis embedding GET failed: %s", exc)

    vector = _lru_get(key)
    if vector is not None:
        logger.debug("Embedding cache HIT (LRU) %s", key[:16])
    return vector


def _cache_set(key: str, vector: List[float]) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.setex(key, _EMBEDDING_CACHE_TTL, json.dumps(vector))
            logger.debug("Embedding cache SET (Redis) %s TTL %ds", key[:16], _EMBEDDING_CACHE_TTL)
            return
        except Exception as exc:
            logger.warning("Redis embedding SET failed: %s", exc)
    _lru_set(key, vector)
    logger.debug("Embedding cache SET (LRU) %s", key[:16])


def get_embedding_model() -> _Embedder:
    """
    Get or create the embedding model (singleton).

    Ollama: talks to OLLAMA_BASE_URL (no large local download).
    HuggingFace: loads sentence-transformers on CPU.
    """
    global _embedding_model

    if _embedding_model is None:
        provider = settings.EMBEDDING_PROVIDER
        logger.info(
            "Loading embedding model: provider=%s model=%s",
            provider,
            settings.EMBEDDING_MODEL,
        )
        if provider == "ollama":
            from app.rag.ollama_client import (
                OllamaEmbeddings,
                check_ollama_model,
                check_ollama_reachable,
            )

            check_ollama_reachable(settings.OLLAMA_BASE_URL)
            check_ollama_model(settings.OLLAMA_BASE_URL, settings.EMBEDDING_MODEL)
            _embedding_model = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
            )
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            _embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER={provider!r}. "
                "Use 'ollama' or 'huggingface'."
            )
        logger.info("Embedding model ready (%s)", provider)

    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts, serving cached vectors where available."""
    if not texts:
        return []

    keys = [_make_embedding_key(t) for t in texts]
    results: List[Optional[List[float]]] = [None] * len(texts)
    miss_indices: List[int] = []
    miss_texts: List[str] = []

    for i, key in enumerate(keys):
        cached = _cache_get(key)
        if cached is not None:
            results[i] = cached
        else:
            miss_indices.append(i)
            miss_texts.append(texts[i])

    cache_hits = len(texts) - len(miss_indices)
    logger.debug(
        "embed_texts: %d/%d served from cache, %d to embed",
        cache_hits, len(texts), len(miss_indices),
    )

    if miss_texts:
        model = get_embedding_model()
        new_vectors: List[List[float]] = trace_call(
            "embed_texts",
            lambda: model.embed_documents(miss_texts),
            run_type="embedding",
            metadata={
                "embedding_provider": settings.EMBEDDING_PROVIDER,
                "embedding_model": settings.EMBEDDING_MODEL,
                "text_count": len(miss_texts),
                "cache_hits": cache_hits,
            },
        )
        for idx, vector in zip(miss_indices, new_vectors):
            _cache_set(keys[idx], vector)
            results[idx] = vector

    return results  # type: ignore[return-value]


def embed_query(query: str) -> List[float]:
    """Embed a single query string, using cache when available."""
    key = _make_embedding_key(query)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    model = get_embedding_model()
    vector: List[float] = trace_call(
        "embed_query",
        lambda: model.embed_query(query),
        run_type="embedding",
        metadata={
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "query_length": len(query),
        },
    )
    _cache_set(key, vector)
    return vector

"""
Response caching utility for PDF-Assistant-RAG.

Supports two backends:
- Redis (preferred, for production)
- LRU in-memory cache (fallback for development or when Redis is unavailable)

Cache key is a SHA-256 hash of (user_id, document_id, question) to ensure
keys are short, stable, and unique across all user/question/document
combinations — never shared between different users.
"""

import hashlib
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis client (lazy init — only created when REDIS_URL is set)
# ---------------------------------------------------------------------------

_redis_client = None
_redis_available = False
_redis_init_attempted = False
_redis_last_error: Optional[str] = None


def _cache_settings():
    """Load Redis/cache settings from pydantic Settings (.env-aware)."""
    from app.config import get_settings

    settings = get_settings()
    return (
        (settings.REDIS_URL or "").strip() or None,
        int(settings.CACHE_TTL),
        int(settings.CACHE_LRU_MAX_SIZE),
    )


# Mutable module knobs (tests may override LRU_MAX_SIZE).
REDIS_URL: Optional[str] = None
CACHE_TTL: int = 3600
LRU_MAX_SIZE: int = 128
_config_loaded = False


def _ensure_config() -> None:
    """Pull REDIS_URL / TTL / LRU size from Settings once (.env-aware)."""
    global REDIS_URL, CACHE_TTL, LRU_MAX_SIZE, _config_loaded
    if _config_loaded:
        return
    redis_url, cache_ttl, lru_max = _cache_settings()
    REDIS_URL = redis_url
    CACHE_TTL = cache_ttl
    LRU_MAX_SIZE = lru_max
    _config_loaded = True


def _get_redis():
    """
    Lazily initialise the Redis client.
    Returns the client if Redis is reachable, otherwise returns None.
    After one failure, stops retrying for the process lifetime.
    """
    global _redis_client, _redis_available, _redis_init_attempted, _redis_last_error

    if _redis_client is not None:
        return _redis_client if _redis_available else None

    if _redis_init_attempted:
        return None

    _redis_init_attempted = True
    _ensure_config()
    redis_url = REDIS_URL

    if not redis_url:
        logger.info("REDIS_URL not set — using LRU in-memory cache.")
        _redis_available = False
        _redis_last_error = "REDIS_URL not configured"
        return None

    try:
        import redis  # soft import — redis package is optional

        client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        client.ping()
        _redis_client = client
        _redis_available = True
        _redis_last_error = None
        logger.info("Redis cache connected at %s", redis_url)
        return _redis_client
    except Exception as exc:  # noqa: BLE001
        _redis_last_error = str(exc)[:200]
        logger.warning("Redis unavailable (%s) — falling back to LRU cache.", exc)
        _redis_available = False
        return None


# ---------------------------------------------------------------------------
# LRU in-memory fallback
# ---------------------------------------------------------------------------

_lru_store: dict = {}
_lru_order: list = []


def _lru_get(key: str) -> Optional[str]:
    return _lru_store.get(key)


def _lru_set(key: str, value: str) -> None:
    _ensure_config()
    if key in _lru_store:
        _lru_order.remove(key)
    elif len(_lru_store) >= LRU_MAX_SIZE:
        oldest = _lru_order.pop(0)
        del _lru_store[oldest]
    _lru_store[key] = value
    _lru_order.append(key)


def _lru_delete(key: str) -> None:
    if key in _lru_store:
        del _lru_store[key]
        _lru_order.remove(key)


# ---------------------------------------------------------------------------
# Public API — these are the only functions chat.py needs
# ---------------------------------------------------------------------------


def make_cache_key(
    user_id: str,
    document_id: str,
    question: str,
    session_id: str = "",
) -> str:
    """
    Generate a stable, short cache key from user_id + session + document + question.

    SHA-256 gives us a 64-char hex string that is:
    - Always the same length regardless of question length
    - Unique per (user_id, session_id, document_id, question) — session_id
      prevents cross-conversation cache leakage for identical questions
    - Safe for Redis keys and dict keys
    """
    raw = (
        f"{user_id}:{session_id or ''}:{document_id}:"
        f"{question.strip().lower()}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decode_cache_payload(raw: str):
    """Decode cache JSON. Supports legacy plain-string answers and {answer, sources}."""
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, str):
        return {"answer": data, "sources": []}
    if isinstance(data, dict) and "answer" in data:
        sources = data.get("sources")
        if not isinstance(sources, list):
            sources = []
        return {"answer": data.get("answer") or "", "sources": sources}
    return None


def get_cached_payload(
    user_id: str,
    document_id: str,
    question: str,
    session_id: str = "",
) -> Optional[dict]:
    """
    Look up a cached answer (+ sources) for a scoped question key.
    Returns ``{"answer": str, "sources": list}`` on hit, None on miss.
    """
    key = make_cache_key(user_id, document_id, question, session_id=session_id)
    r = _get_redis()

    if r is not None:
        try:
            value = r.get(key)
            if value:
                logger.debug("Cache HIT (Redis) for key %s", key[:12])
                return _decode_cache_payload(value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis GET failed (%s) — checking LRU.", exc)

    value = _lru_get(key)
    if value:
        logger.debug("Cache HIT (LRU) for key %s", key[:12])
        return _decode_cache_payload(value)

    logger.debug("Cache MISS for key %s", key[:12])
    return None


def get_cached_response(
    user_id: str,
    document_id: str,
    question: str,
    session_id: str = "",
) -> Optional[str]:
    """
    Look up a cached answer for a scoped question key.
    Returns the answer string on hit, None on miss.
    """
    payload = get_cached_payload(
        user_id, document_id, question, session_id=session_id
    )
    if payload is None:
        return None
    return payload.get("answer")


def set_cached_response(
    user_id: str,
    document_id: str,
    question: str,
    answer: str,
    sources: Optional[list] = None,
    session_id: str = "",
) -> None:
    """
    Store an answer (and optional citation sources). Tries Redis first; falls back to LRU.
    TTL is controlled by the CACHE_TTL environment variable.
    """
    _ensure_config()
    key = make_cache_key(user_id, document_id, question, session_id=session_id)
    serialised = json.dumps({"answer": answer, "sources": sources or []})
    r = _get_redis()

    if r is not None:
        try:
            r.setex(key, CACHE_TTL, serialised)
            logger.debug("Cache SET (Redis) key %s TTL %ds", key[:12], CACHE_TTL)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis SET failed (%s) — storing in LRU.", exc)

    _lru_set(key, serialised)
    logger.debug("Cache SET (LRU) key %s", key[:12])


def invalidate_cache(user_id: str, document_id: str, question: str) -> None:
    """Remove one cache entry — useful when a document is re-indexed."""
    key = make_cache_key(user_id, document_id, question)
    r = _get_redis()
    if r is not None:
        try:
            r.delete(key)
        except Exception:  # noqa: BLE001
            pass
    _lru_delete(key)

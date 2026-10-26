"""
Document/query embeddings with optional Redis/LRU caching.

Providers (from ``app.config.Settings``, verified defaults):
- ollama (default): ``EMBEDDING_MODEL=nomic-embed-text``, ``EMBEDDING_DIMENSION=768``
- huggingface: typically MiniLM 384-d when switched via env

Ask-time caller: ``retriever.retrieve`` → ``embed_query(search_query)``.
Index-time caller: ``vectorstore.store_chunks`` → ``get_embedding_model().embed_documents``.

Common example question (used in all six RAG files):
  "Does the XOLAIR prescribing label recommend every-2-week dosing for
   chronic spontaneous urticaria (CSU)?"
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

# Singleton: one embedder per process (OllamaEmbeddings or HuggingFaceEmbeddings).
_embedding_model = None

# Cache knobs (env-overridable). TTL only applies when Redis is used.
_EMBEDDING_CACHE_TTL: int = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))  # 24 h
_EMBEDDING_LRU_MAX_SIZE: int = int(os.getenv("EMBEDDING_LRU_MAX_SIZE", "512"))

# In-memory LRU used when REDIS_URL is unset (typical local/dev).
_emb_lru_store: dict = {}
_emb_lru_order: list = []

# Shared sample question for docstring examples in this module.
_EX_Q = (
    "Does the XOLAIR prescribing label recommend every-2-week dosing for "
    "chronic spontaneous urticaria (CSU)?"
)


@runtime_checkable
class _Embedder(Protocol):
    """Duck-typed embedder API shared by Ollama and HuggingFace wrappers.

    Exact methods callers rely on
    -----------------------------
    embed_documents(texts: List[str]) -> List[List[float]]
        Index path. For nomic via Ollama, each text is prefixed
        ``search_document: `` inside ``OllamaEmbeddings._prefix`` (ollama_client.py).
    embed_query(text: str) -> List[float]
        Ask path. For nomic, prefixed ``search_query: `` before POST /api/embed.

    Exact example (ask path, defaults)
    ----------------------------------
    model.embed_query(_EX_Q)
      → List[float] with len == settings.EMBEDDING_DIMENSION (768 for nomic)
      ILLUSTRATIVE values: [0.0123, -0.4410, 0.0881, ...]  # not a live run
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


def _lru_get(key: str) -> Optional[List[float]]:
    """Read one cached vector from the in-memory LRU.

    Args:
        key: Cache key from ``_make_embedding_key``, e.g. ``emb:<64-hex>``.

    Returns:
        Decoded ``List[float]`` on hit; ``None`` on miss.

    Exact example
    -------------
    BEFORE: _emb_lru_store has no entry for key
      _lru_get("emb:aaaaaaaa...") → None

    AFTER a prior _lru_set stored a 768-d vector as JSON:
      _lru_get("emb:aaaaaaaa...") → [0.01, -0.2, ...]   # ILLUSTRATIVE floats

    Next: returned by ``_cache_get`` when Redis is absent or misses.
    """
    raw = _emb_lru_store.get(key)
    return json.loads(raw) if raw is not None else None


def _lru_set(key: str, vector: List[float]) -> None:
    """Insert/update an LRU entry; evict oldest when size >= _EMBEDDING_LRU_MAX_SIZE.

    Args:
        key: ``emb:...`` key.
        vector: Embedding floats (expected length 768 for default nomic).

    Side effects:
        Mutates module globals ``_emb_lru_store`` and ``_emb_lru_order``.

    Exact example
    -------------
    BEFORE: len(_emb_lru_store) == 512 (full), oldest key = "emb:old..."
    Call:   _lru_set("emb:new...", [0.0] * 768)
    AFTER:  "emb:old..." removed; "emb:new..." present; order ends with new key.

    Next: only called from ``_cache_set`` when Redis SET is skipped/fails.
    """
    if key in _emb_lru_store:
        _emb_lru_order.remove(key)
    elif len(_emb_lru_store) >= _EMBEDDING_LRU_MAX_SIZE:
        oldest = _emb_lru_order.pop(0)
        del _emb_lru_store[oldest]
    _emb_lru_store[key] = json.dumps(vector)
    _emb_lru_order.append(key)


def _get_redis():
    """Return the shared Redis client from ``app.cache._get_redis``, or None.

    Exact example
    -------------
    Local/dev with REDIS_URL unset (matches prior app.log: "REDIS_URL not set"):
      _get_redis() → None
      → callers use LRU only.

    Next: ``_cache_get`` / ``_cache_set``.
    """
    try:
        from app.cache import _get_redis as _cache_get_redis
        return _cache_get_redis()
    except Exception:
        return None


def _make_embedding_key(text: str) -> str:
    """SHA-256 cache key scoped by provider + model + stripped text.

    Args:
        text: Raw query or document string (whitespace stripped inside).

    Returns:
        ``"emb:" + hexdigest`` (always starts with ``emb:``).

    Exact example (defaults: ollama + nomic-embed-text)
    --------------------------------------------------
    text = _EX_Q
    raw  = "ollama:nomic-embed-text:" + _EX_Q.strip()
    key  = "emb:" + sha256(raw.encode("utf-8")).hexdigest()
         # e.g. "emb:3f8c1a..."  (exact hex depends on text; not hardcoded here)

    Why provider:model is in the key: switching to MiniLM must not reuse nomic vectors.

    Next: ``_cache_get(key)`` / ``_cache_set(key, vector)``.
    """
    raw = f"{settings.EMBEDDING_PROVIDER}:{settings.EMBEDDING_MODEL}:{text.strip()}"
    return "emb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[List[float]]:
    """Try Redis GET, then LRU. Never raises to the caller for Redis errors.

    Exact example (ask path, first time for _EX_Q)
    ----------------------------------------------
    key = _make_embedding_key(_EX_Q)
    Redis None → _lru_get miss → return None  (CACHE MISS)
    Caller ``embed_query`` then calls the model.

    Exact example (second ask of same string)
    -----------------------------------------
    LRU hit → return List[float] len 768 without calling Ollama.

    Next: ``embed_query`` / ``embed_texts`` branch on None vs vector.
    """
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
    """Persist vector: Redis SETEX(TTL) if available, else ``_lru_set``.

    Exact example
    -------------
    After first embed of _EX_Q with Redis down:
      _cache_set(key, vector_768)
      → _lru_set(...); log "Embedding cache SET (LRU) emb:...."

    Verified TTL default: _EMBEDDING_CACHE_TTL == 86400 seconds when env unset.
    """
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
    """Lazy singleton factory for the configured embedding backend.

    Config read (verified defaults in config.py):
        EMBEDDING_PROVIDER = "ollama"
        EMBEDDING_MODEL    = "nomic-embed-text"
        OLLAMA_BASE_URL    = "http://localhost:11434"
        EMBEDDING_DIMENSION = 768

    Branch:
        provider == "ollama" → OllamaEmbeddings after check_ollama_reachable/model
        provider == "huggingface" → HuggingFaceEmbeddings(device=cpu, normalize=True)
        else → ValueError

    Exact example (default branch)
    ------------------------------
    First call:
      check_ollama_reachable("http://localhost:11434")
      check_ollama_model(..., "nomic-embed-text")
      _embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url=...)
    Later calls: return same instance.

    Side effects: process-global ``_embedding_model``; may raise if Ollama down.

    Next: ``embed_query`` / ``embed_texts`` / ``vectorstore.store_chunks``.
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
    """Batch-embed document strings with per-item cache (index helpers / tests).

    Note: ``vectorstore.store_chunks`` currently calls
    ``embedding_model.embed_documents`` directly, not this function. This API
    still shows the project's caching pattern for batch embedding.

    Args:
        texts: Chunk strings in order.

    Returns:
        Same-length list of vectors. Empty input → ``[]``.

    Exact example
    -------------
    texts = [
      "The recommended dose for CSU is 150 mg or 300 mg by subcutaneous "
      "injection every 4 weeks.",                           # ILLUSTRATIVE label text
    ]
    keys = [_make_embedding_key(t) for t in texts]
    First run: miss_texts == texts → model.embed_documents(miss_texts)
      → Ollama prefixes each with "search_document: " (nomic family)
      → results[0] length 768; _cache_set(keys[0], results[0])
    Second run: cache hit → no model call.

    Next stage (when used): Chroma ``collection.add(embeddings=...)``.
    """
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
    """Embed one search/user string for Chroma nearest-neighbor search.

    Args:
        query: Usually one string from ``retriever.transform_query`` output
               (original question or an LLM paraphrase).

    Returns:
        ``List[float]`` — verified expected length 768 when using default nomic.

    What the vector represents
    --------------------------
    A point in the same embedding space as indexed chunk vectors. Cosine
    distance in Chroma (``hnsw:space=cosine``) becomes similarity
    ``score = 1 - distance`` in ``vectorstore.query_chunks``.

    Exact call from retriever.py
    ----------------------------
    for search_query in transform_query(user_question):
        query_vector = embed_query(search_query)   # ← this function
        vector_results = query_chunks(query_embedding=query_vector, ...)

    Exact example (first miss)
    --------------------------
    query = _EX_Q
    key = _make_embedding_key(query)           # "emb:" + sha256(...)
    _cache_get(key) → None
    model = get_embedding_model()              # OllamaEmbeddings
    # Inside OllamaEmbeddings.embed_query → _embed(for_query=True):
    #   POST http://localhost:11434/api/embed
    #   input item = "search_query: " + query   (nomic prefix; ollama_client.py)
    vector = [... 768 floats ...]              # ILLUSTRATIVE contents
    _cache_set(key, vector)
    return vector

    Exact example (cache hit)
    -------------------------
    Same query again → _cache_get returns vector; model not called.

    Fallback: none inside this function — Ollama failures raise RuntimeError
    from ``OllamaEmbeddings._embed`` (propagates to retrieve/agent).
    """
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


# =============================================================================
# PIPELINE FLOW WITH EXACT EXAMPLE
# =============================================================================
#
# A. Actual input (type + sample)
#    query: str =
#      "Does the XOLAIR prescribing label recommend every-2-week dosing for
#       chronic spontaneous urticaria (CSU)?"
#
# B. Function-by-function (ask path used by retriever.retrieve)
#
#    1) embed_query(query)
#         args: query=_EX_Q
#         ops:  key=_make_embedding_key(query)
#               cached=_cache_get(key)  → None on first ask (ILLUSTRATIVE)
#               model=get_embedding_model()  → OllamaEmbeddings(nomic-embed-text)
#               vector=model.embed_query(query) via trace_call
#                 (nomic: prefixes "search_query: " before /api/embed)
#               _cache_set(key, vector)
#         returns: List[float] length 768 (settings.EMBEDDING_DIMENSION)
#         next:    vectorstore.query_chunks(query_embedding=vector, ...)
#
#    2) Supporting: _make_embedding_key → _cache_get/_cache_set → _get_redis/_lru_*
#
# C. Before → after
#    BEFORE: Python str (question)
#    AFTER:  List[float] dim=768  representing semantic location of the question
#            ILLUSTRATIVE: [0.02, -0.11, 0.07, ...]  # not from a live embed call
#
# D. Verified config vs illustrative
#    VERIFIED from config.py: provider=ollama, model=nomic-embed-text, dim=768,
#      OLLAMA_BASE_URL=http://localhost:11434, LRU max=512, TTL=86400.
#    ILLUSTRATIVE: exact float values, exact sha256 hex of the sample question.
#
# Index-time sibling path (vectorstore.store_chunks):
#    embedding_model.embed_documents(batch_texts)
#      → each text prefixed "search_document: " for nomic
#      → collection.add(embeddings=...)
#
# Next file: vectorstore.py (query_chunks) and bm25.py (parallel lexical path).
# =============================================================================

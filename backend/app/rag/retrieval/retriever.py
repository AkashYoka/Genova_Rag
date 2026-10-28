"""
Two-stage retrieval: hybrid (vector + BM25 via RRF) + optional cross-encoder.

Verified settings (config.py):
  TOP_K_RETRIEVAL = 20
  TOP_K_RERANK = 8
  USE_HYBRID_SEARCH = True
  RRF_K = 60
  RERANKER_ENABLED = False  (default → sort+slice, no BGE)

Caller: ``agent._generate_answer_direct`` / ``_generate_answer_stream_direct``
  retrieve(query=question, user_id=..., document_id=..., top_k=TOP_K_RERANK or None)

Common example question:
  "Does the XOLAIR prescribing label recommend every-2-week dosing for
   chronic spontaneous urticaria (CSU)?"
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

from app.config import get_settings
from app.rag.embeddings.embeddings import embed_query
from app.rag.utils.tracing import trace_function
from app.rag.vectordb.vectorstore import query_chunks
from app.rag.retrieval.reranker import get_reranker

logger = logging.getLogger(__name__)
settings = get_settings()
MAX_QUERY_VARIANTS = 4

_EX_Q = (
    "Does the XOLAIR prescribing label recommend every-2-week dosing for "
    "chronic spontaneous urticaria (CSU)?"
)
_EX_USER = "f3d989fb-79af-4fe2-9360-12b06772348c"
_EX_DOC = "8c57d30c-9568-4437-b51f-b52d9ac495d1"


# ── RRF core ─────────────────────────────────────────────────────────────────

def rrf_merge(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse two ranked lists with Reciprocal Rank Fusion.

    Formula (exact in loop): ``rrf_scores[key] += 1.0 / (k + rank)`` with rank
    1-based. Same chunk in both lists accumulates both terms.

    Dedup key order: ``id`` or ``chunk_id`` if set; else
    ``document_id|page|text[:200]``.

    Exact example (ILLUSTRATIVE ranks; k=60 verified default)
    ---------------------------------------------------------
    vector_results[0] = CSU every-4-weeks chunk (id=..._42)  # rank 1
    bm25_results[2]   = same id                               # rank 3
    rrf = 1/(60+1) + 1/(60+3) ≈ 0.016393 + 0.015873 ≈ 0.032266
    OUT chunk includes ``rrf_score`` rounded to 6 decimals.

    retrieve then does: ``chunk["score"] = chunk.pop("rrf_score")``.

    Why RRF: vector similarity ~0–1 vs BM25 raw ~1–10 — ranks are comparable.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_store: Dict[str, Dict[str, Any]] = {}

    def _key(chunk: Dict[str, Any]) -> str:
        """Stable deduplication key — prefer explicit IDs, fall back to content hash."""
        for field in ("id", "chunk_id"):
            if chunk.get(field):
                return str(chunk[field])
        text = str(chunk.get("text", ""))
        return "|".join([
            str(chunk.get("document_id", "")),
            str(chunk.get("page", "")),
            text[:200],
        ])

    def _accumulate(results: List[Dict[str, Any]]) -> None:
        for rank, chunk in enumerate(results, start=1):
            key = _key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chunk_store or chunk.get("score", 0) > chunk_store[key].get("score", 0):
                chunk_store[key] = chunk

    _accumulate(vector_results)
    _accumulate(bm25_results)

    merged = []
    for key, rrf_score in sorted(rrf_scores.items(), key=lambda t: t[1], reverse=True):
        chunk = chunk_store[key].copy()
        chunk["rrf_score"] = round(rrf_score, 6)
        merged.append(chunk)

    return merged


# ── Query helpers ─────────────────────────────────────────────────────────────

def transform_query(query: str) -> List[str]:
    """Expand one question into ≤ MAX_QUERY_VARIANTS (4) search strings.

    Exact example
    -------------
    IN:  query = _EX_Q
    Ops: generated = _generate_query_variants(_EX_Q)  # up to 3 paraphrases
         return _dedupe_queries([_EX_Q, *generated])[:4]

    ILLUSTRATIVE OUT (paraphrases depend on live LLM; shapes only):
      [
        _EX_Q,
        "XOLAIR CSU dosing interval every two weeks vs label",
        "chronic spontaneous urticaria omalizumab recommended frequency",
        "Does prescribing information allow biweekly XOLAIR for CSU",
      ]

    Fallback: LLM exception → log warning; generated=[]; return [original] only.
    Blank strip → ``[]``.

    Next: retrieve loops ``for search_query in transform_query(query)``.
    """
    original_query = query.strip()
    if not original_query:
        return []

    try:
        generated_queries = _generate_query_variants(original_query)
    except Exception as e:
        logger.warning("Query transformation failed, using original query only: %s", e)
        generated_queries = []

    return _dedupe_queries([original_query, *generated_queries])[:MAX_QUERY_VARIANTS]


def _generate_query_variants(query: str) -> List[str]:
    """Ask configured chat LLM for exactly 3 JSON-array paraphrases.

    Exact client selection
    ----------------------
    if settings.uses_ollama_llm:
        client = get_llm_client()  # OllamaChatClient, model settings.LLM_MODEL
                                  # default "qwen2.5:3b"
      except → log "Query rewrite skipped"; return []
    else:
        needs HF_TOKEN; else return []

    Exact call
    ----------
    client.chat_completion(
      messages=[system rewriter, user prompt with query],
      model=settings.LLM_MODEL,
      max_tokens=256,
      temperature=0.3,
    )
    → _parse_query_variants(content)[:3]

    ILLUSTRATIVE model output string:
      '["XOLAIR CSU every 2 weeks dosing?", "...", "..."]'

    Fallback: no choices / bad parse → fewer or empty variants (original still kept).
    """
    # HuggingFace path needs a token; Ollama path needs a reachable server.
    if settings.uses_ollama_llm:
        try:
            from app.rag.llm.agent import get_llm_client
            client = get_llm_client()
        except Exception as exc:
            logger.warning("Query rewrite skipped (Ollama unavailable): %s", exc)
            return []
    else:
        if not settings.HF_TOKEN:
            return []
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=settings.HF_TOKEN)

    prompt = (
        "Generate exactly 3 semantic variations of the user question below. "
        "Each variation must preserve the original meaning but use different "
        "vocabulary, phrasing, or sentence structure to improve document retrieval coverage. "
        "Do NOT add new topics or change the intent. "
        "Return ONLY a JSON array of 3 strings, with no extra text, markdown, or explanation.\n\n"
        f"User question: {query}\n\n"
        'Example output: ["variation one", "variation two", "variation three"]'
    )

    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query rewriter for a RAG retrieval system. "
                    "You output only valid JSON arrays of strings, nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model=settings.LLM_MODEL,
        max_tokens=256,
        temperature=0.3,
    )

    if not response.choices:
        return []

    content = response.choices[0].message.content or ""
    variants = _parse_query_variants(content)

    # Cap at 3 variants as requested — the original is added by transform_query
    return variants[:3]


def _parse_query_variants(content: str) -> List[str]:
    """Parse rewrite LLM output: JSON list, embedded ``[...]``, or bullet lines.

    Exact example
    -------------
    IN:  '["alpha", "beta", "gamma"]' → ["alpha","beta","gamma"]
    IN:  'Sure:\\n["alpha", "beta"]' → regex extract → ["alpha","beta"]
    IN:  '- alpha\\n- beta' → ["alpha","beta"]
    """
    content = content.strip()
    if not content:
        return []

    parsed = _try_parse_query_json(content)
    if parsed is not None:
        return parsed

    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        parsed = _try_parse_query_json(match.group(0))
        if parsed is not None:
            return parsed

    queries = []
    for line in content.splitlines():
        cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip().strip('"')
        if cleaned:
            queries.append(cleaned)
    return queries


def _try_parse_query_json(content: str) -> Optional[List[str]]:
    """json.loads helper; accepts list or ``{"queries": [...]}``.

    Exact example
    -------------
    '["a","b"]' → ["a","b"]
    '{"queries":["a"]}' → ["a"]
    'not-json' → None
    '[1,2]' → []  (non-strings dropped → empty list, not None)
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        parsed = parsed.get("queries", [])

    if not isinstance(parsed, list):
        return []

    return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]


def _dedupe_queries(queries: List[str]) -> List[str]:
    """Case-insensitive dedupe after collapsing internal whitespace.

    Exact example
    -------------
    IN:  [_EX_Q, _EX_Q.upper(), "  other  q  "]
    OUT: [_EX_Q, "other q"]  # first casing kept
    """
    deduped = []
    seen = set()
    for query in queries:
        normalized = " ".join(query.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _merge_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe across query variants; key = document_id|page|text[:200]; keep max score.

    Exact example
    -------------
    Variant A and B both retrieve id ..._42 → one entry with higher score kept.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        candidate_copy = dict(candidate)
        key = "|".join([
            str(candidate_copy.get("document_id", "")),
            str(candidate_copy.get("page", "")),
            str(candidate_copy.get("text", ""))[:200],
        ])
        existing = merged.get(key)
        if existing is None or candidate_copy.get("score", 0) > existing.get("score", 0):
            merged[key] = candidate_copy
    return list(merged.values())


# ── Main retrieval pipeline ───────────────────────────────────────────────────

@trace_function(
    "retrieve",
    metadata_factory=lambda query, user_id, document_id=None, document_ids=None, top_k=None: {
        "user_id": user_id,
        "document_id": document_id,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "top_k_retrieval": settings.TOP_K_RETRIEVAL,
        "top_k_rerank": settings.TOP_K_RERANK,
        "hybrid_search": settings.USE_HYBRID_SEARCH,
        "rrf_k": settings.RRF_K,
    },
)
def retrieve(
    query: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Full ask-time retrieval for one question.

    Exact default call from agent (Ollama direct)
    ---------------------------------------------
    retrieve(
      query=_EX_Q,
      user_id=_EX_USER,
      document_id=_EX_DOC,
      document_ids=None,
      top_k=settings.TOP_K_RERANK,  # 8 passed as retrieval width from agent
    )
    Note: agent passes ``top_k or settings.TOP_K_RERANK``. If top_k is 8,
    ``effective_top_k`` for Chroma/BM25 is 8 (not 20). If caller omits top_k
    (None), effective_top_k = TOP_K_RETRIEVAL (20).

    Stage 1 per search_query in transform_query(query):
      query_vector = embed_query(search_query)
      vector_results = query_chunks(..., top_k=effective_top_k)
      if USE_HYBRID_SEARCH:
        bm25_results = query_bm25(...)  # except → [] + warning
        merged = rrf_merge(..., k=RRF_K); score = pop rrf_score
      else: use vector_results only

    Stage 2:
      candidates = _merge_candidates(all_candidates)
      reranker = get_reranker()
      if reranker: rerank(query=ORIGINAL query, ..., top_k=TOP_K_RERANK)
      else: sort by score; [:TOP_K_RERANK]

    Confidence (exact):
      max_score = max(rerank_score or score); floor 0.001
      confidence = round((raw/max_score)*100, 1)
      if rerank_score present → score=round(rerank_score,4); del rerank_score

    Exact OUT shape (≤8 items; contents ILLUSTRATIVE)
    -------------------------------------------------
    [{
      "id": "..._42",
      "text": "... every 4 weeks ...",
      "filename": "xolair_prescribing.pdf",
      "document_id": "...",
      "page": 9,
      "score": 0.0323,       # RRF or rerank
      "confidence": 100.0,   # best chunk
      ...
    }, ...]

    Fallbacks summary:
      rewrite fail → original only
      BM25 fail → vector-only merge for that variant
      USE_HYBRID_SEARCH False → vector only
      RERANKER_ENABLED False → sort+slice
      no candidates → []

    Next: agent._sources_from_chunks / _format_context.
    """
    # Multi-attach: retrieve per document so one dominant PDF cannot crowd out others.
    unique_ids: List[str] = []
    if document_ids and not document_id:
        seen = set()
        for raw_id in document_ids:
            doc_id = str(raw_id) if raw_id is not None else ""
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                unique_ids.append(doc_id)

    if len(unique_ids) > 1:
        final_cap = settings.TOP_K_RERANK
        per_doc_k = max(2, (final_cap + len(unique_ids) - 1) // len(unique_ids))
        pooled: List[Dict[str, Any]] = []
        for doc_id in unique_ids:
            pooled.extend(
                retrieve(
                    query=query,
                    user_id=user_id,
                    document_id=doc_id,
                    document_ids=None,
                    top_k=per_doc_k,
                )
            )

        if not pooled:
            return []

        # Fair merge: keep best chunk per doc first, then fill by score.
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in pooled:
            key = str(chunk.get("document_id") or "")
            by_doc.setdefault(key, []).append(chunk)
        for doc_chunks in by_doc.values():
            doc_chunks.sort(
                key=lambda x: x.get("rerank_score", x.get("score", 0)),
                reverse=True,
            )

        selected: List[Dict[str, Any]] = []
        seen_ids = set()
        for doc_id in unique_ids:
            for chunk in by_doc.get(doc_id, []):
                cid = chunk.get("id")
                if cid in seen_ids:
                    continue
                selected.append(chunk)
                seen_ids.add(cid)
                break

        remainder = [
            c
            for doc_chunks in by_doc.values()
            for c in doc_chunks
            if c.get("id") not in seen_ids
        ]
        remainder.sort(
            key=lambda x: x.get("rerank_score", x.get("score", 0)),
            reverse=True,
        )
        for chunk in remainder:
            if len(selected) >= final_cap:
                break
            cid = chunk.get("id")
            if cid in seen_ids:
                continue
            selected.append(chunk)
            seen_ids.add(cid)

        if selected:
            max_score = max(
                chunk.get("rerank_score", chunk.get("score", 0)) for chunk in selected
            )
            max_score = max(max_score, 0.001)
            for chunk in selected:
                raw = chunk.get("rerank_score", chunk.get("score", 0))
                chunk["confidence"] = round((raw / max_score) * 100, 1)
                if "rerank_score" in chunk:
                    chunk["score"] = round(chunk["rerank_score"], 4)
                    del chunk["rerank_score"]

        from app.observability import chunks_retrieved_var

        chunks_retrieved_var.set(len(selected))
        logger.info(
            "Retrieved %d chunks across %d attached docs for query: '%s'",
            len(selected),
            len(unique_ids),
            query,
        )
        return selected

    effective_top_k = top_k if top_k is not None else settings.TOP_K_RETRIEVAL

    # ── Stage 1: Hybrid retrieval with query transformation ───────────────────
    all_candidates: List[Dict[str, Any]] = []

    for search_query in transform_query(query):
        query_vector = embed_query(search_query)

        # Vector results (always)
        vector_results = query_chunks(
            query_embedding=query_vector,
            user_id=user_id,
            document_id=document_id,
            document_ids=document_ids,
            top_k=effective_top_k,
        )

        if settings.USE_HYBRID_SEARCH:
            try:
                from app.rag.retrieval.bm25 import query_bm25
                bm25_results = query_bm25(
                    query=search_query,
                    user_id=user_id,
                    document_id=document_id,
                    document_ids=document_ids,
                    top_k=effective_top_k,
                )
            except Exception as exc:
                logger.warning("BM25 retrieval failed, using vector-only: %s", exc)
                bm25_results = []

            merged = rrf_merge(
                vector_results=vector_results,
                bm25_results=bm25_results,
                k=settings.RRF_K,
            )

            for chunk in merged:
                chunk["score"] = chunk.pop("rrf_score")

            all_candidates.extend(merged)
        else:
            all_candidates.extend(vector_results)

    if not all_candidates:
        return []

    candidates = _merge_candidates(all_candidates)

    # ── Stage 2: Cross-encoder reranking ─────────────────────────────────────
    final_k = top_k if top_k is not None else settings.TOP_K_RERANK
    reranker = get_reranker()

    if reranker is not None:
        top_chunks = reranker.rerank(
            query=query,
            documents=candidates,
            top_k=final_k,
        )
    else:
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_chunks = candidates[:final_k]

    # ── Confidence normalisation ──────────────────────────────────────────────
    if top_chunks:
        max_score = max(
            chunk.get("rerank_score", chunk.get("score", 0))
            for chunk in top_chunks
        )
        max_score = max(max_score, 0.001)

        for chunk in top_chunks:
            raw = chunk.get("rerank_score", chunk.get("score", 0))
            chunk["confidence"] = round((raw / max_score) * 100, 1)
            if "rerank_score" in chunk:
                chunk["score"] = round(chunk["rerank_score"], 4)
                del chunk["rerank_score"]

    from app.observability import chunks_retrieved_var
    chunks_retrieved_var.set(len(top_chunks))
    logger.info(
        "Retrieved %d relevant chunks for query: '%s'",
        len(top_chunks),
        query,
    )

    return top_chunks


# =============================================================================
# PIPELINE FLOW WITH EXACT EXAMPLE
# =============================================================================
#
# A. Actual input
#    query: str = _EX_Q
#    user_id: str = _EX_USER          # ILLUSTRATIVE
#    document_id: str = _EX_DOC       # ILLUSTRATIVE Xolair id from prior UI test
#    top_k: Optional[int] = 8         # as passed by agent Ollama path
#
# B. Stage chain (default config: hybrid ON, rerank OFF)
#
#    transform_query(_EX_Q)
#      → [original, ≤3 paraphrases]   # paraphrases ILLUSTRATIVE / live LLM
#
#    for each search_query:
#      embed_query(search_query)           → List[float] dim 768
#      query_chunks(..., top_k=8)          → semantic hits
#      query_bm25(..., top_k=8)            → lexical hits
#      rrf_merge(..., k=60)                → rrf_score → renamed score
#
#    _merge_candidates(all_candidates)
#    get_reranker() → None
#    sort by score; top_chunks = candidates[:8]
#    confidence % relative to max score
#
# C. Before → after (compact)
#    Original question
#      → query variants (≤4 strings)
#      → query embedding (768-d)
#      → vector search results (list[dict] score=1-distance)
#      → BM25 results (list[dict] raw BM25 score)
#      → RRF merge (rrf_score / score)
#      → deduplication (_merge_candidates)
#      → reranking SKIPPED (RERANKER_ENABLED=False)
#      → final chunks ≤8 with confidence
#
# D. Verified vs illustrative
#    VERIFIED: TOP_K_*, RRF_K, USE_HYBRID_SEARCH, RERANKER_ENABLED=False,
#      models nomic-embed-text / qwen2.5:3b for rewrite.
#    ILLUSTRATIVE: chunk text, pages, numeric scores, rewrite strings, UUIDs.
#
# Next: agent.py formats these chunks into the LLM prompt.
# =============================================================================

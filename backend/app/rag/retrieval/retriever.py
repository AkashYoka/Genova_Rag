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
  "According to the dosing schedule table, what value applies for the
   stated patient parameters?"
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Set

from app.config import get_settings
from app.rag.embeddings.embeddings import embed_query
from app.rag.utils.tracing import trace_function
from app.rag.vectordb.vectorstore import query_chunks
from app.rag.retrieval.reranker import get_reranker

logger = logging.getLogger(__name__)
settings = get_settings()
MAX_QUERY_VARIANTS = 4


def _exclude_soft_deleted_chunks(
    chunks: List[Dict[str, Any]],
    user_id: str,
) -> List[Dict[str, Any]]:
    """Drop chunks whose document is in Trash (soft-deleted).

    Only documents explicitly marked ``is_deleted=True`` are excluded. Missing
    SQL rows (e.g. ephemeral demo indexes) are kept so retrieval still works
    when vectors exist but the metadata row is absent.
    """
    if not chunks:
        return chunks
    doc_ids = {
        str(chunk.get("document_id"))
        for chunk in chunks
        if chunk.get("document_id")
    }
    if not doc_ids:
        return chunks
    try:
        from app.database import SessionLocal
        from app.models import Document

        db = SessionLocal()
        try:
            deleted = {
                str(row[0])
                for row in db.query(Document.id)
                .filter(
                    Document.user_id == user_id,
                    Document.id.in_(list(doc_ids)),
                    Document.is_deleted.is_(True),
                )
                .all()
            }
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Could not filter soft-deleted chunks: %s", exc)
        return chunks
    if not deleted:
        return chunks
    return [
        chunk
        for chunk in chunks
        if str(chunk.get("document_id") or "") not in deleted
    ]

_EX_Q = (
    "According to the schedule table, what value applies for the stated parameters?"
)
_EX_USER = "user-id"
_EX_DOC = "document-id"


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
    vector_results[0] = schedule-table chunk (id=..._42)  # rank 1
    bm25_results[2]   = same id                           # rank 3
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

# Domain-agnostic: questions that typically need a table / schedule / matrix.
_STRUCTURED_LOOKUP_RE = re.compile(
    r"(?i)\b("
    r"table|chart|grid|matrix|schedule|frequency|rate|amount|total|"
    r"dose|dosing|dosage|parameter|threshold|limit|range|"
    r"off[- ]?label|special\s+situation|adverse\s+event|prescrib|"
    r"medication\s+error|indication|body\s*weight|"
    r"every\s+\d+\s+(?:weeks?|days?|hours?|months?)"
    r")\b"
)
_STRUCTURED_UNIT_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*(?:mg|kg|g|ml|l|%|\$|usd|eur|inr|"
    r"weeks?|days?|hours?|hrs?|months?|iu/?mL|ppm|mhz|ghz|gb|mb)\b"
)
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{3,}", re.I)
_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "with", "from", "that",
    "this", "these", "those", "what", "which", "when", "where", "how",
    "why", "does", "did", "have", "has", "was", "were", "please", "check",
    "below", "content", "according", "using", "into", "about", "your",
    "you", "any", "some", "all", "based", "should", "would", "could",
}


def _query_content_tokens(text: str) -> Set[str]:
    return {
        t.lower()
        for t in _QUERY_TOKEN_RE.findall(text or "")
        if t.lower() not in _QUERY_STOPWORDS
    }


def _structured_table_query_variants(query: str) -> List[str]:
    """Deterministic variants that prioritize tables/schedules for lookup asks.

    Domain-agnostic: no product names, indications, or lab assays. Surfaces
    numeric+unit tokens from the question so BM25 can hit matching cells.
    """
    q = query.strip()
    if not q or not _STRUCTURED_LOOKUP_RE.search(q):
        return []
    variants: List[str] = [
        "recommended values table chart schedule parameters",
        "lookup table row column matching criteria",
    ]
    nums = _STRUCTURED_UNIT_RE.findall(q)
    if nums:
        variants.append("table " + " ".join(nums[:6]))
    topic_tokens = sorted(_query_content_tokens(q))[:8]
    if topic_tokens:
        variants.append("table " + " ".join(topic_tokens))
    return variants


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
        "recommended values table chart schedule parameters",
        "lookup table row column matching criteria",
        "<LLM paraphrase>",
      ]

    Fallback: LLM exception → log warning; generated=[]; return [original] only.
    Blank strip → ``[]``.

    Next: retrieve loops ``for search_query in transform_query(query)``.
    """
    original_query = query.strip()
    if not original_query:
        return []

    structured_variants = _structured_table_query_variants(original_query)

    try:
        generated_queries = _generate_query_variants(original_query)
    except Exception as e:
        logger.warning("Query transformation failed, using original query only: %s", e)
        generated_queries = []

    # Prefer deterministic table-oriented variants ahead of LLM paraphrases so
    # narrative chunks do not crowd out the matching schedule/matrix.
    return _dedupe_queries(
        [original_query, *structured_variants, *generated_queries]
    )[:MAX_QUERY_VARIANTS]


def _generate_query_variants(query: str) -> List[str]:
    """Ask configured chat LLM for exactly 3 JSON-array paraphrases.

    Exact client selection
    ----------------------
    if settings.uses_ollama_llm:
        client = get_llm_client()  # OllamaChatClient, model settings.LLM_MODEL
                                  # default "qwen2.5:3b"
      except → log "Query rewrite skipped"; return []
    Other providers (bedrock, huggingface) use the same provider-aware client.

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
      '["schedule table lookup?", "...", "..."]'

    Fallback: no choices / bad parse → fewer or empty variants (original still kept).
    """
    # The provider-aware client handles Ollama, Bedrock, and HuggingFace.
    try:
        from app.rag.llm.agent import get_llm_client
        client = get_llm_client()
    except Exception as exc:
        logger.warning("Query rewrite skipped (%s): %s", settings.LLM_PROVIDER, exc)
        return []

    prompt = (
        "Generate exactly 3 semantic variations of the user question below. "
        "Each variation must preserve the original meaning but use different "
        "vocabulary, phrasing, or sentence structure to improve document retrieval coverage. "
        "Do NOT add new topics or change the intent. "
        "Return ONLY a JSON array of 3 strings, with no extra text, markdown, or explanation.\n\n"
        f"User question: {query}\n\n"
        'Example output: ["variation one", "variation two", "variation three"]'
    )

    # Use the provider-correct model id. Passing settings.LLM_MODEL (Ollama id)
    # into Bedrock Converse causes ValidationException and aborts rewrite.
    from app.rag.llm.agent import llm_model_for_provider

    rewrite_model = llm_model_for_provider()
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
        model=rewrite_model,
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


def _chunk_index_from_id(chunk_id: Any) -> Optional[int]:
    if chunk_id is None:
        return None
    text = str(chunk_id)
    if "_" not in text:
        return None
    tail = text.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return None


def _needs_adjacent_expansion(chunk: Dict[str, Any]) -> bool:
    chunk_type = str(chunk.get("chunk_type") or chunk.get("content_type") or "").lower()
    if chunk_type in ("table", "table_summary", "table_interpretation", "table_row"):
        return True
    text = str(chunk.get("text") or "")
    if re.search(r"(?i)\btable\s+\d+\b", text):
        return True
    if "table facts:" in text.lower():
        return True
    return False


def _boost_table_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    modality: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Prefer table / Table-facts / TableRAG index chunks for lookup questions.

    Domain-agnostic: when the query looks like a table/schedule/parameter
    lookup, boost any table (or linearized ``Table facts:`` block). Extra
    weight when the table text shares content tokens with the query so the
    relevant matrix outranks unrelated narrative paragraphs.

    ``modality`` comes from TableRAG query routing:
      - table: stronger table boost + mild paragraph downrank
      - both: moderate table boost
      - paragraph: no boost (preserve text RAG)
    """
    if not candidates:
        return candidates

    modality_l = (modality or "").lower()
    structured_query = bool(_STRUCTURED_LOOKUP_RE.search(query or ""))
    # Boost only when the question looks tabular/numeric or routing says table.
    # Plain "both"/paragraph questions keep existing text-RAG ranking.
    if modality_l == "paragraph" and not structured_query:
        return candidates
    if modality_l != "table" and not structured_query:
        return candidates

    q_tokens = _query_content_tokens(query or "")
    boosted: List[Dict[str, Any]] = []
    for chunk in candidates:
        item = dict(chunk)
        text = str(item.get("text") or "")
        text_l = text.lower()
        chunk_type = str(
            item.get("chunk_type") or item.get("content_type") or ""
        ).lower()
        index_kind = str(item.get("index_kind") or "").lower()
        score = float(item.get("score") or 0)
        factor = 1.0
        is_table = (
            chunk_type
            in ("table", "table_summary", "table_interpretation", "table_row")
            or index_kind
            in ("table_summary", "table_interpretation", "table_row")
            or "table facts:" in text_l
            or "table interpretation" in text_l
            or bool(item.get("table_json"))
        )
        is_paragraph = not is_table and chunk_type in ("text", "paragraph", "list", "")

        if is_table:
            factor += 0.35
            if chunk_type == "table_row" or index_kind == "table_row":
                # Exact-value / multi-constraint lookups land on row texts.
                factor += 0.25 if modality_l == "table" else 0.15
            elif chunk_type == "table_summary" or index_kind == "table_summary":
                factor += 0.2 if modality_l == "table" else 0.1
            elif (
                chunk_type == "table_interpretation"
                or index_kind == "table_interpretation"
            ):
                # Semantic purpose / axis phrasing — useful for how/why questions.
                factor += 0.22 if modality_l == "table" else 0.12
        if re.search(r"(?i)\btable\s+\d+\b", text) and re.search(
            r"(?i)dose|dosing|schedule|frequency|amount|rate|weight|"
            r"every\s+\d+\s+(?:weeks?|days?|hours?|months?)",
            text,
        ):
            factor += 0.25
        if is_table and q_tokens:
            overlap = len(q_tokens & _query_content_tokens(text))
            if overlap >= 2:
                factor += 0.2
            elif overlap == 1:
                factor += 0.1
        if modality_l == "table" and is_paragraph:
            factor *= 0.85
        if factor != 1.0:
            item["score"] = score * factor
            item["boost_reason"] = "structured_table"
            if modality_l:
                item["query_modality"] = modality_l
        boosted.append(item)
    return boosted


def _expand_adjacent_chunks(
    selected: List[Dict[str, Any]],
    *,
    user_id: str,
    window: int,
    final_cap: int,
) -> List[Dict[str, Any]]:
    """Include neighboring chunk_index hits for tables / Table-N references.

    Uses already-retrieved candidate metadata when present; otherwise fetches
    adjacent ids from the vector store by reconstructed chunk id.
    """
    if not selected or window <= 0:
        return selected

    from app.rag.vectordb.vectorstore import get_chunks_by_ids

    existing_ids = {str(c.get("id")) for c in selected if c.get("id")}
    to_fetch: List[str] = []
    for chunk in selected:
        if not _needs_adjacent_expansion(chunk):
            continue
        doc_id = chunk.get("document_id")
        idx = chunk.get("chunk_index")
        if idx is None:
            idx = _chunk_index_from_id(chunk.get("id"))
        if doc_id is None or idx is None:
            continue
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            continue
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbor_id = f"{doc_id}_{idx_i + offset}"
            if neighbor_id not in existing_ids:
                to_fetch.append(neighbor_id)
                existing_ids.add(neighbor_id)

    if not to_fetch:
        return selected

    try:
        neighbors = get_chunks_by_ids(user_id=user_id, chunk_ids=to_fetch)
    except Exception as exc:
        logger.debug("Adjacent chunk expansion skipped: %s", exc)
        return selected

    # Keep original ranking; append neighbors until cap.
    out = list(selected)
    seen = {str(c.get("id")) for c in out if c.get("id")}
    for neighbor in neighbors:
        nid = str(neighbor.get("id") or "")
        if not nid or nid in seen:
            continue
        # Slightly down-rank neighbors so primary hits stay first.
        neighbor = dict(neighbor)
        neighbor["score"] = float(neighbor.get("score") or 0) * 0.85
        neighbor.setdefault("confidence", 0)
        out.append(neighbor)
        seen.add(nid)
        if len(out) >= max(final_cap, len(selected)):
            break
    return out[: max(final_cap, len(selected))]


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
      top_k=settings.TOP_K_RERANK,  # final slice only; stage-1 width is TOP_K_RETRIEVAL
    )
    Note: agent passes ``top_k or settings.TOP_K_RERANK`` as the *final* count.
    Stage-1 Chroma/BM25 always uses ``TOP_K_RETRIEVAL`` (20) so dosing tables
    can enter the candidate pool before prescribing-table boost + final slice.

    Stage 1 per search_query in transform_query(query):
      query_vector = embed_query(search_query)
      vector_results = query_chunks(..., top_k=TOP_K_RETRIEVAL)
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
      "filename": "document.pdf",
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

        selected = _exclude_soft_deleted_chunks(selected, user_id)

        from app.observability import chunks_retrieved_var

        chunks_retrieved_var.set(len(selected))
        logger.info(
            "Retrieved %d chunks across %d attached docs for query: '%s'",
            len(selected),
            len(unique_ids),
            query,
        )
        return selected

    # Stage 1 always gathers a wide candidate pool. ``top_k`` only controls the
    # final slice after boost/rerank (agent typically passes TOP_K_RERANK=8).
    stage1_top_k = max(int(settings.TOP_K_RETRIEVAL or 20), int(top_k or 0))

    # ── Stage 1: Hybrid retrieval with query transformation ───────────────────
    all_candidates: List[Dict[str, Any]] = []
    search_queries = transform_query(query)

    query_modality = "both"
    try:
        from app.rag.table_rag.routing import route_query

        route = route_query(query)
        query_modality = route.modality.value
        logger.info(
            "TableRAG route=%s reason=%s structured=%s for query='%s'",
            route.modality.value,
            route.reason,
            route.wants_structured_lookup,
            query[:120],
        )
    except Exception as exc:
        logger.debug("TableRAG routing unavailable: %s", exc)

    try:
        from app.rag.retrieval.rag_debug import record_rewrites

        record_rewrites(search_queries)
    except Exception:
        pass

    for search_query in search_queries:
        query_vector = embed_query(search_query)

        # Vector results (always)
        vector_results = query_chunks(
            query_embedding=query_vector,
            user_id=user_id,
            document_id=document_id,
            document_ids=document_ids,
            top_k=stage1_top_k,
        )

        if settings.USE_HYBRID_SEARCH:
            try:
                from app.rag.retrieval.bm25 import query_bm25
                bm25_results = query_bm25(
                    query=search_query,
                    user_id=user_id,
                    document_id=document_id,
                    document_ids=document_ids,
                    top_k=stage1_top_k,
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
    candidates = _boost_table_candidates(
        query, candidates, modality=query_modality
    )

    # ── Stage 2: Cross-encoder reranking ─────────────────────────────────────
    final_k = top_k if top_k is not None else settings.TOP_K_RERANK
    # Table-heavy routes keep a slightly wider final window so row hits survive.
    if query_modality == "table":
        final_k = max(final_k, min(final_k + 2, stage1_top_k))
    reranker = get_reranker()

    rerank_enabled = reranker is not None
    if reranker is not None:
        top_chunks = reranker.rerank(
            query=query,
            documents=candidates,
            top_k=final_k,
        )
    else:
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_chunks = candidates[:final_k]

    # Expand table / Table-N hits with neighboring chunks for complete grids.
    if getattr(settings, "RAG_EXPAND_ADJACENT_CHUNKS", True):
        top_chunks = _expand_adjacent_chunks(
            top_chunks,
            user_id=user_id,
            window=int(getattr(settings, "RAG_ADJACENT_CHUNK_WINDOW", 1) or 1),
            final_cap=max(final_k + 2, final_k),
        )

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
            chunk.setdefault("query_modality", query_modality)

    top_chunks = _exclude_soft_deleted_chunks(top_chunks, user_id)

    # Hydrate full structured tables from SQL for answer generation.
    if getattr(settings, "TABLE_RAG_HYDRATE_ON_RETRIEVE", True) and getattr(
        settings, "TABLE_RAG_ENABLED", True
    ):
        try:
            from app.rag.table_rag import hydrate_retrieved_tables

            top_chunks = hydrate_retrieved_tables(top_chunks, user_id=user_id)
        except Exception as exc:
            logger.warning("TableRAG hydrate skipped: %s", exc)

    from app.observability import chunks_retrieved_var
    chunks_retrieved_var.set(len(top_chunks))
    try:
        from app.rag.retrieval.rag_debug import record_retrieval

        method = "hybrid_rrf" if settings.USE_HYBRID_SEARCH else "vector"
        if rerank_enabled:
            method = f"{method}+rerank"
        record_retrieval(
            method=method,
            chunks=top_chunks,
            rerank_enabled=rerank_enabled,
        )
    except Exception:
        pass
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
#      query_chunks(..., top_k=TOP_K_RETRIEVAL)  → semantic hits
#      query_bm25(..., top_k=TOP_K_RETRIEVAL)    → lexical hits
#      rrf_merge(..., k=60)                → rrf_score → renamed score
#
#    _merge_candidates(all_candidates)
#    _boost_table_candidates(...)
#    get_reranker() → None
#    sort by score; top_chunks = candidates[:final_k]  # final_k = top_k or 8
#    optional adjacent expand for table hits
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
#      → structured-table boost (domain-agnostic lookup queries)
#      → reranking SKIPPED (RERANKER_ENABLED=False)
#      → final chunks ≤8 with confidence (+ adjacent expand)
#
# D. Verified vs illustrative
#    VERIFIED: TOP_K_*, RRF_K, USE_HYBRID_SEARCH, RERANKER_ENABLED=False,
#      models nomic-embed-text / qwen2.5:3b for rewrite.
#    ILLUSTRATIVE: chunk text, pages, numeric scores, rewrite strings, UUIDs.
#
# Next: agent.py formats these chunks into the LLM prompt.
# =============================================================================

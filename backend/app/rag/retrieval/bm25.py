"""
BM25 keyword search via rank_bm25 (one pickle per document).

Storage path pattern (exact):
  {CHROMA_PERSIST_DIR}/bm25/{user_id_with_underscores}/{document_id}.pkl
Default CHROMA_PERSIST_DIR = "./data/chroma_db_ollama_nomic"

Ask-time: ``retriever.retrieve`` imports ``query_bm25`` when USE_HYBRID_SEARCH=True.
Differs from vector search: scores raw token overlap, not embedding similarity.

Common example question:
  "Does the XOLAIR prescribing label recommend every-2-week dosing for
   chronic spontaneous urticaria (CSU)?"
"""
import os
import glob
import pickle
import logging
import re
from typing import List, Dict, Any, Optional

from app.config import get_settings

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logger = logging.getLogger(__name__)
settings = get_settings()

_EX_Q = (
    "Does the XOLAIR prescribing label recommend every-2-week dosing for "
    "chronic spontaneous urticaria (CSU)?"
)
_EX_DOC = "8c57d30c-9568-4437-b51f-b52d9ac495d1"
_EX_USER = "f3d989fb-79af-4fe2-9360-12b06772348c"
_EX_FILE = "xolair_prescribing.pdf"


def get_bm25_dir(user_id: str) -> str:
    """Ensure and return ``.../bm25/{clean_user_id}/``.

    Exact example
    -------------
    user_id=_EX_USER
    → "./data/chroma_db_ollama_nomic/bm25/f3d989fb_79af_4fe2_9360_12b06772348c"
    (creates directory if missing)
    """
    clean_id = user_id.replace("-", "_")
    path = os.path.join(settings.CHROMA_PERSIST_DIR, "bm25", clean_id)
    os.makedirs(path, exist_ok=True)
    return path

def get_bm25_path(user_id: str, document_id: str) -> str:
    """Return ``{get_bm25_dir(user_id)}/{document_id}.pkl``.

    Exact example
    -------------
    → ".../bm25/f3d989fb_.../8c57d30c-9568-4437-b51f-b52d9ac495d1.pkl"
    """
    return os.path.join(get_bm25_dir(user_id), f"{document_id}.pkl")

def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens via ``re.findall(r'\\w+', text.lower())``.

    Exact example (the shared question)
    -----------------------------------
    IN:  _EX_Q
    OUT: [
      "does", "the", "xolair", "prescribing", "label", "recommend",
      "every", "2", "week", "dosing", "for", "chronic", "spontaneous",
      "urticaria", "csu",
    ]
    Note: "every-2-week" splits into every, 2, week because ``\\w+`` breaks on hyphens.

    Exact example (label chunk text, ILLUSTRATIVE)
    ----------------------------------------------
    IN:  "150 mg or 300 mg every 4 weeks"
    OUT: ["150", "mg", "or", "300", "mg", "every", "4", "weeks"]

    Next: BM25Okapi corpus tokens at index time; query tokens in ``query_bm25``.
    """
    return re.findall(r'\w+', text.lower())

def store_bm25_index(chunks: List[Dict[str, Any]], document_id: str, filename: str, user_id: str):
    """Build BM25Okapi over chunk texts and pickle ``{bm25, chunks}``.

    Exact example
    -------------
    chunks = [{
      "text": "Recommended dosage for CSU: 150 mg or 300 mg SC every 4 weeks.",
      "page": 9, "chunk_index": 42,
    }]  # ILLUSTRATIVE
    tokenized_texts = [tokenize(chunks[0]["text"]), ...]
    bm25 = BM25Okapi(tokenized_texts)
    formatted_chunks[0]["id"] = f"{document_id}_42"
    pickle → get_bm25_path(user_id, document_id)

    Fallback: if BM25Okapi is None (package missing) → log warning and return.
    Empty chunks → return without writing.

    Called from: vectorstore.store_chunks.
    """
    if BM25Okapi is None:
        logger.warning("rank_bm25 is not installed; skipping BM25 index storage")
        return

    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    tokenized_texts = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    # Format chunks to match vectorstore output
    formatted_chunks = []
    for chunk in chunks:
        chunk_idx = chunk.get("chunk_index")
        chunk_id = f"{document_id}_{chunk_idx}" if chunk_idx is not None else None
        formatted_chunks.append({
            "id": chunk_id,
            "text": chunk["text"],
            "filename": filename,
            "document_id": document_id,
            "page": chunk.get("page", 1),
        })

    data = {
        "bm25": bm25,
        "chunks": formatted_chunks
    }

    path = get_bm25_path(user_id, document_id)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Stored BM25 index for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to store BM25 index for {document_id}: {e}")

def _query_single_index(path: str, tokenized_query: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Score one pickle; return up to top_k chunks with score > 0.

    Exact example
    -------------
    path = get_bm25_path(_EX_USER, _EX_DOC)
    tokenized_query = tokenize(_EX_Q)
    scores = bm25.get_scores(tokenized_query)   # one float per chunk in corpus
    top_indices = sorted(..., reverse=True)[:top_k]

    OUT item (ILLUSTRATIVE scores/order):
      {
        "id": "8c57d30c-..._42",
        "text": "... every 4 weeks ...",
        "filename": "xolair_prescribing.pdf",
        "document_id": "8c57d30c-...",
        "page": 9,
        "score": 5.21,   # RAW BM25 — not 0..1; RRF only uses rank
      }

    Why raw scores: ``rrf_merge`` uses list position, not magnitude.
    Missing/corrupt pickle → ``[]``.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load BM25 index from {path}: {e}")
        return []

    bm25 = data["bm25"]
    chunks = data["chunks"]

    scores = bm25.get_scores(tokenized_query)
    
    # Get top_k indices sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in top_indices:
        if scores[i] > 0:
            chunk = chunks[i].copy()
            # Normalize BM25 score to 0-1 range roughly, or just keep raw.
            # BM25 scores are usually > 0, often 1-10.
            # We keep the raw score for now, RRF will handle the ranking.
            chunk["score"] = float(scores[i])
            results.append(chunk)

    return results

def query_bm25(
    query: str,
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Keyword search scoped like vectorstore filters.

    Branches (exact):
      1) document_id set → one pickle via get_bm25_path
      2) document_ids set → query each, merge, sort by score, [:top_k]
      3) neither → glob all ``*.pkl`` for user, merge, sort, [:top_k]
      BM25Okapi is None → ``[]``

    Exact ask example (hybrid retrieve)
    -----------------------------------
    IN from retriever:
      query = _EX_Q  (or a rewrite variant)
      user_id = _EX_USER
      document_id = _EX_DOC
      top_k = 20  (TOP_K_RETRIEVAL)

    Ops:
      tokenized_query = tokenize(query)
      return _query_single_index(get_bm25_path(user_id, document_id), tokenized_query, 20)

    OUT: up to 20 dicts sorted by raw BM25 score (ILLUSTRATIVE ordering favors
    chunks containing "xolair", "csu", "week", "dosing", etc.)

    Vs vector search: BM25 needs shared tokens; "urticaria schedule" without
    "CSU" may rank lower even if semantically related.

    Next: rrf_merge(vector_results, bm25_results, k=60).
    """
    if BM25Okapi is None:
        return []

    tokenized_query = tokenize(query)
    
    if document_id:
        path = get_bm25_path(user_id, document_id)
        return _query_single_index(path, tokenized_query, top_k)
    
    if document_ids:
        all_results = []
        for doc_id in document_ids:
            path = get_bm25_path(user_id, doc_id)
            if os.path.exists(path):
                results = _query_single_index(path, tokenized_query, top_k)
                all_results.extend(results)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
    
    # If no document_id and no document_ids, query all documents for this user
    user_dir = get_bm25_dir(user_id)
    all_results = []
    
    for path in glob.glob(os.path.join(user_dir, "*.pkl")):
        results = _query_single_index(path, tokenized_query, top_k)
        all_results.extend(results)
        
    # Sort all results by score and take top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]

def delete_bm25_index(document_id: str, user_id: str):
    """Remove one document pickle if present."""
    path = get_bm25_path(user_id, document_id)
    if os.path.exists(path):
        try:
            os.remove(path)
            logger.info(f"Deleted BM25 index for document {document_id}")
        except Exception as e:
            logger.warning(f"Error deleting BM25 index: {e}")

def delete_user_bm25_indexes(user_id: str):
    """Remove all pickles and the user BM25 directory."""
    user_dir = get_bm25_dir(user_id)
    if os.path.exists(user_dir):
        try:
            for path in glob.glob(os.path.join(user_dir, "*.pkl")):
                os.remove(path)
            os.rmdir(user_dir)
            logger.info(f"Deleted BM25 directory for user {user_id}")
        except Exception as e:
            logger.warning(f"Error deleting BM25 directory for user {user_id}: {e}")


# =============================================================================
# PIPELINE FLOW WITH EXACT EXAMPLE
# =============================================================================
#
# A. Actual input
#    query: str = _EX_Q
#    user_id / document_id as above; top_k=20 from retrieve
#
# B. Function chain
#    tokenize(query)
#      → ["does","the","xolair",...,"csu"]
#    query_bm25 → get_bm25_path → _query_single_index
#      → bm25.get_scores(tokens) → top indices with score>0
#
# C. Before → after
#    BEFORE: natural-language question string
#    AFTER:  lexical hit list with RAW BM25 scores (often ~1–10, ILLUSTRATIVE)
#
# D. Code truth
#    Scores are NOT normalized to 0–1 (comment in _query_single_index).
#    Hybrid fusion ignores score magnitude and uses rank via rrf_merge.
#
# Next: retriever.rrf_merge(vector_results, bm25_results, k=settings.RRF_K)
# =============================================================================

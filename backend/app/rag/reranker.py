"""
Cross-encoder reranker (BAAI/bge-reranker-v2-m3).

Verified config (config.py):
  RERANKER_ENABLED = False   # default — get_reranker() returns None
  RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"
  TOP_K_RERANK     = 8       # applied by retriever when calling rerank

When disabled (default Ollama setup), ``retriever.retrieve`` sorts by RRF
``score`` and slices ``[:TOP_K_RERANK]`` — this file is not invoked.

Common example question:
  "Does the XOLAIR prescribing label recommend every-2-week dosing for
   chronic spontaneous urticaria (CSU)?"
"""

import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder

from app.config import get_settings

logger = logging.getLogger(__name__)

_EX_Q = (
    "Does the XOLAIR prescribing label recommend every-2-week dosing for "
    "chronic spontaneous urticaria (CSU)?"
)


# ── Reranker Class ─────────────────────────────────────
class Reranker:
    """Lazy CrossEncoder wrapper used only when RERANKER_ENABLED is true."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Store model id; do not load weights yet.

        Exact example
        -------------
        Reranker() → model_name="BAAI/bge-reranker-v2-m3", _model=None
        """
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = device
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        """First-use load of sentence_transformers.CrossEncoder(max_length=512).

        Exact example
        -------------
        First rerank() call → log "Loading reranker: BAAI/bge-reranker-v2-m3"
        → self._model = CrossEncoder(...); subsequent calls reuse it.
        """
        if self._model is None:
            logger.info(f"Loading reranker: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device
            )
            logger.info("Reranker loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """Score (query, doc[text]) pairs; return top_k docs with ``rerank_score``.

        Exact call from retriever.py (when get_reranker() is not None)
        --------------------------------------------------------------
        top_chunks = reranker.rerank(
            query=query,              # ORIGINAL user question (not a rewrite)
            documents=candidates,     # deduped hybrid candidates
            top_k=settings.TOP_K_RERANK,  # 8
        )

        Exact example of pairs (ILLUSTRATIVE documents)
        -----------------------------------------------
        query = _EX_Q
        documents = [
          {"text": "... CSU ... every 4 weeks ...", "page": 9, "score": 0.031},
          {"text": "... asthma dosing table ...", "page": 5, "score": 0.028},
        ]
        pairs = [
          (_EX_Q, documents[0]["text"]),
          (_EX_Q, documents[1]["text"]),
        ]
        scores = model.predict(pairs)   # ILLUSTRATIVE e.g. [0.91, 0.22]
        sort descending → keep top_k
        each kept doc gets doc["rerank_score"] = float(score)

        Empty documents → ``[]``.

        Next in retrieve: confidence = (rerank_score / max) * 100; score replaced
        by rounded rerank_score; key ``rerank_score`` deleted.
        """
        if not documents:
            return []

        model = self._load_model()

        # Prepare query-document pairs
        pairs = [(query, doc[text_key]) for doc in documents]

        # Get relevance scores
        scores = model.predict(pairs)

        # Pair scores with documents and sort in descending order
        scored = list(zip(scores, documents))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top_k documents
        reranked = [doc for _, doc in scored[:top_k]]

        # Attach rerank_score to each returned document
        for (score, doc) in scored:
            if doc in reranked:
                doc["rerank_score"] = float(score)

        return reranked


_reranker_instance: Optional[Reranker] = None

def get_reranker(model_name: Optional[str] = None) -> Optional[Reranker]:
    """Return singleton Reranker, or None when RERANKER_ENABLED is false.

    Exact behavior (verified default)
    ---------------------------------
    settings.RERANKER_ENABLED == False
      → return None immediately (no model download)

    If enabled:
      first call creates Reranker(model_name=...); later calls reuse it.

    Exact retrieve branch when None (default)
    -----------------------------------------
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_chunks = candidates[:settings.TOP_K_RERANK]  # 8

    There is no try/except around get_reranker itself; disable is the soft
    "fallback". Load/predict failures would raise if enabled.
    """
    global _reranker_instance
    settings = get_settings()
    if not getattr(settings, "RERANKER_ENABLED", False):
        return None
    if _reranker_instance is None:
        _reranker_instance = Reranker(model_name=model_name)
    return _reranker_instance


# =============================================================================
# PIPELINE FLOW WITH EXACT EXAMPLE
# =============================================================================
#
# A. Gate
#    get_reranker()
#      RERANKER_ENABLED False (VERIFIED default) → None → skip this file
#      True → Reranker.rerank(_EX_Q, candidates, top_k=8)
#
# B. When enabled — pairs and ordering
#    pairs[i] = (original_question, candidates[i]["text"])
#    model.predict(pairs) → scores
#    sort by score desc → [:8] → attach rerank_score
#
# C. Before → after (enabled)
#    BEFORE: hybrid candidates ordered by RRF score
#    AFTER:  ≤8 chunks ordered by cross-encoder relevance
#
# D. When disabled (actual default path for this repo)
#    Input candidates → sort by score → slice 8 → confidence normalisation
#    in retriever.retrieve (no CrossEncoder call)
#
# Next: agent._sources_from_chunks / _format_context
# =============================================================================

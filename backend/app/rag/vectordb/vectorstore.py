"""
ChromaDB vector store operations (per-user collections).

Verified defaults (config.py):
  CHROMA_PERSIST_DIR = "./data/chroma_db_ollama_nomic"
  Collection metadata uses hnsw:space = "cosine"

Ask-time: ``retriever.retrieve`` → ``query_chunks(query_embedding=..., top_k=20)``.
Index-time: ingestion → ``store_chunks`` → BM25 + embed_documents + collection.add.

Common example question:
  "Does the XOLAIR prescribing label recommend every-2-week dosing for
   chronic spontaneous urticaria (CSU)?"
"""
import logging
from typing import List, Dict, Any, Optional, Callable
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_chroma_client = None

# ILLUSTRATIVE ids used only in comments (from a prior UI test session shape;
# not read from the DB by this documentation).
_EX_USER = "f3d989fb-79af-4fe2-9360-12b06772348c"
_EX_DOC = "8c57d30c-9568-4437-b51f-b52d9ac495d1"
_EX_FILE = "xolair_prescribing.pdf"
_EX_Q = (
    "Does the XOLAIR prescribing label recommend every-2-week dosing for "
    "chronic spontaneous urticaria (CSU)?"
)


def get_chroma_client():
    """Return process-wide PersistentClient under settings.CHROMA_PERSIST_DIR.

    Exact example
    -------------
    First call:
      os.makedirs("./data/chroma_db_ollama_nomic", exist_ok=True)
      _chroma_client = chromadb.PersistentClient(path=..., anonymized_telemetry=False)
      log: "ChromaDB initialized at ./data/chroma_db_ollama_nomic"
    Later: return same client.

    Next: get_or_create_collection / get_collection in store/query/delete.
    """
    global _chroma_client

    if _chroma_client is None:
        import os
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB initialized at {settings.CHROMA_PERSIST_DIR}")

    return _chroma_client


def get_collection_name(user_id: str) -> str:
    """Map user UUID → valid Chroma name (hyphens → underscores, max 63 chars).

    Exact example
    -------------
    IN:  user_id = "f3d989fb-79af-4fe2-9360-12b06772348c"
    OUT: "user_f3d989fb_79af_4fe2_9360_12b06772348c"

    Isolates every user's vectors so query_chunks never crosses accounts.
    """
    # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
    clean_id = user_id.replace("-", "_")
    name = f"user_{clean_id}"
    # Truncate if too long
    return name[:63]


def store_chunks(
    chunks: List[Dict[str, Any]],
    document_id: str,
    filename: str,
    user_id: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Embed + upsert chunks into Chroma; rebuild BM25 for the same document.

    Args:
        chunks: Dicts with at least text, page, chunk_index (from chunker).
        document_id / filename / user_id: ownership + metadata.
        on_progress: Optional callback ``(stored_count, total_count)`` after each batch.

    Returns:
        Number of chunks added (0 if chunks empty).

    Exact example (index path)
    --------------------------
    chunks = [{
      "text": "Recommended dosage for chronic spontaneous urticaria: "
              "150 mg or 300 mg SC every 4 weeks.",  # ILLUSTRATIVE label excerpt
      "page": 9,
      "chunk_index": 42,
      "chunk_type": "text",
    }]
    document_id=_EX_DOC, filename=_EX_FILE, user_id=_EX_USER

    Steps in code order:
      1) delete_document_chunks(document_id, user_id)  # wipe old Chroma+BM25
      2) store_bm25_index(chunks, ...)                 # bm25.py
      3) generate_captions_for_chunks(chunks)          # optional; warn on fail
      4) get_embedding_model().embed_documents(batch)  # search_document: prefix
      5) collection.add(ids=[f"{document_id}_{chunk_index}"], embeddings, metadatas, documents)
    OUT: 1 (or N for full PDF; prior session noted ~217 chunks for Xolair — ILLUSTRATIVE count)

    Fallbacks: BM25/vision failures are logged; Chroma add still proceeds if BM25 fails.
    """
    if not chunks:
        return 0

    # Delete existing chunks for this document and user before inserting new ones
    # to avoid stale/orphaned chunks in ChromaDB and BM25.
    delete_document_chunks(document_id, user_id)

    # Build and store BM25 index
    from app.rag.retrieval.bm25 import store_bm25_index
    try:
        store_bm25_index(chunks, document_id, filename, user_id)
    except Exception as e:
        logger.error(f"Could not build BM25 index: {e}")

    # Generate captions for any extracted images before embedding
    try:
        from app.rag.ingestion.vision import generate_captions_for_chunks

        generate_captions_for_chunks(chunks)
    except Exception as e:
        logger.warning(f"Could not generate image captions: {e}")

    client = get_chroma_client()
    from app.rag.embeddings.embeddings import get_embedding_model

    embedding_model = get_embedding_model()

    collection_name = get_collection_name(user_id)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Prepare batch data ───────────────────────────
    texts = [chunk["text"] for chunk in chunks]
    ids = [f"{document_id}_{chunk['chunk_index']}" for chunk in chunks]
    metadatas = [
        {
            "text": chunk["text"],
            "filename": filename,
            "document_id": document_id,
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "chunk_type": chunk.get("chunk_type", "text"),
            **({"bbox": chunk.get("bbox", "")} if chunk.get("bbox") else {}),
            **({"table_index": chunk.get("table_index", 0)} if chunk.get("chunk_type") == "table" else {}),
            # Indicate whether this chunk was originally an image and include a short caption
            **({"is_image": True, "image_caption": chunk.get("image_caption", "")}
               if chunk.get("is_image") else {}),
        }
        for chunk in chunks
    ]

    # ── Embed and upsert in batches ──────────────────
    batch_size = 50
    total_stored = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        # Generate embeddings
        embeddings = embedding_model.embed_documents(batch_texts)

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            documents=batch_texts,
        )
        total_stored += len(batch_texts)
        if on_progress:
            on_progress(total_stored, len(texts))

    logger.info(f"Stored {total_stored} chunks for document {document_id}")
    return total_stored


def query_chunks(
    query_embedding: List[float],
    user_id: str,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Semantic search: nearest neighbors to ``query_embedding`` in the user collection.

    Filtering (exact code):
        if document_id:   where = {"document_id": {"$eq": document_id}}
        elif document_ids: where = {"document_id": {"$in": document_ids}}
        else:             where = None  (all docs for that user)

    Score transform (exact):
        Chroma returns cosine ``distance``; this function sets
        ``score = round(1 - distance, 4)`` (higher = more similar).

    Exact ask example (from retrieve with defaults)
    -----------------------------------------------
    IN:
      query_embedding = embed_query(_EX_Q)   # List[float] len 768
      user_id = _EX_USER
      document_id = _EX_DOC                  # dashboard selected Xolair only
      document_ids = None
      top_k = settings.TOP_K_RETRIEVAL (= 20 when retrieve uses default)

    Ops:
      collection = client.get_collection("user_f3d989fb_79af_...")
      results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where={"document_id": {"$eq": _EX_DOC}},
        include=["documents", "metadatas", "distances"],
      )

    OUT (ILLUSTRATIVE one hit; shape is real):
      [{
        "id": "8c57d30c-9568-4437-b51f-b52d9ac495d1_42",
        "text": "Recommended dosage for CSU is 150 mg or 300 mg ... every 4 weeks.",
        "filename": "xolair_prescribing.pdf",
        "document_id": "8c57d30c-...",
        "page": 9,
        "chunk_type": "text",
        "bbox": "",
        "score": 0.8123,   # ILLUSTRATIVE similarity
      }, ... up to 20 items]

    Fallback: missing collection → warning + ``[]``.

    Next: ``retriever.rrf_merge(vector_results, bm25_results)``.
    """
    client = get_chroma_client()
    collection_name = get_collection_name(user_id)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        logger.warning(f"Collection {collection_name} not found")
        return []

    # ── Build filter ─────────────────────────────────
    where_filter = None
    if document_id:
        where_filter = {"document_id": {"$eq": document_id}}
    elif document_ids:
        where_filter = {"document_id": {"$in": document_ids}}

    # ── Query ────────────────────────────────────────
    # ChromaDB always returns ids; do not pass "ids" in include (1.x rejects it).
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # ── Format results ───────────────────────────────
    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0
            chunk_id = results["ids"][0][i] if results.get("ids") and len(results["ids"]) > 0 else None

            # Convert cosine distance to similarity score (0-1)
            similarity = 1 - distance

            chunks.append({
                "id": chunk_id,
                "text": doc,
                "filename": metadata.get("filename", ""),
                "document_id": metadata.get("document_id", ""),
                "page": metadata.get("page", 1),
                "chunk_type": metadata.get("chunk_type", "text"),
                "bbox": metadata.get("bbox", ""),
                "score": round(similarity, 4),
            })

    return chunks


def delete_document_chunks(document_id: str, user_id: str):
    """Delete BM25 pickle + all Chroma ids where metadata.document_id matches.

    Exact example
    -------------
    delete_document_chunks(_EX_DOC, _EX_USER)
      → delete_bm25_index(...)
      → collection.get(where={"document_id": {"$eq": _EX_DOC}}) → delete(ids=...)
    """
    client = get_chroma_client()
    collection_name = get_collection_name(user_id)

    try:
        from app.rag.retrieval.bm25 import delete_bm25_index
        delete_bm25_index(document_id, user_id)
    except Exception as e:
        logger.warning(f"Error deleting BM25 index: {e}")

    try:
        collection = client.get_collection(name=collection_name)
        # Get all IDs for this document
        results = collection.get(
            where={"document_id": {"$eq": document_id}},
            include=[],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
    except Exception as e:
        logger.warning(f"Error deleting chunks: {e}")


def delete_user_collection(user_id: str):
    """Wipe all BM25 indexes and the entire Chroma collection for one user."""
    client = get_chroma_client()
    collection_name = get_collection_name(user_id)

    try:
        from app.rag.retrieval.bm25 import delete_user_bm25_indexes
        delete_user_bm25_indexes(user_id)
    except Exception as e:
        logger.warning(f"Error deleting user BM25 indexes: {e}")

    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted collection {collection_name}")
    except Exception as e:
        logger.warning(f"Error deleting collection: {e}")


# =============================================================================
# PIPELINE FLOW WITH EXACT EXAMPLE
# =============================================================================
#
# A. Actual input into query_chunks (ask path)
#    query_embedding: List[float]  # len 768 from embeddings.embed_query(_EX_Q)
#    user_id: str = "f3d989fb-..."           # ILLUSTRATIVE session user
#    document_id: str | None = "8c57d30c-..." # single selected PDF
#    top_k: int = 20                         # VERIFIED TOP_K_RETRIEVAL default
#
# B. Execution
#    get_chroma_client()
#    get_collection_name(user_id) → "user_f3d989fb_79af_..."
#    where_filter = {"document_id": {"$eq": document_id}}
#    collection.query(query_embeddings=[query_embedding], n_results=20, where=...)
#    for each hit: score = round(1 - distance, 4)
#
# C. Before → after
#    BEFORE: dense query vector only
#    AFTER:  list[dict] with id, text, filename, document_id, page, chunk_type,
#            bbox, score  (semantic ranking — paraphrases can match)
#
# D. Contrast with BM25 (next parallel call in retrieve):
#    Vector: meaning similarity in embedding space
#    BM25:   exact token overlap after tokenize()
#
# Next: retriever.rrf_merge(vector_results, bm25_results, k=60)
# =============================================================================

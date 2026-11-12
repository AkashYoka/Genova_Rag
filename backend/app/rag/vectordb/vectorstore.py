"""
ChromaDB vector store operations (per-user collections).

Verified defaults (config.py):
  CHROMA_PERSIST_DIR = "./data/chroma_db_ollama_nomic"
  Collection metadata uses hnsw:space = "cosine"

Ask-time: ``retriever.retrieve`` → ``query_chunks(query_embedding=..., top_k=20)``.
Index-time: ingestion → ``store_chunks`` → BM25 + embed_documents + collection.add.

Common example question:
  "According to the schedule table, what value applies for the
   stated parameters?"
"""
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_chroma_client = None
# Serialize Chroma SQLite access inside each process. Concurrent query/add
# against PersistentClient frequently surfaces:
#   sqlite3.InternalError: Error executing plan: Internal error: Error finding id
_chroma_lock = threading.RLock()
_CHROMA_TRANSIENT_MARKERS = (
    "error finding id",
    "database is locked",
    "disk i/o error",
    "database disk image is malformed",
)
_CHROMA_MAX_ATTEMPTS = 5
_CHROMA_RETRY_BASE_SEC = 0.05

# ILLUSTRATIVE ids used only in comments (from a prior UI test session shape;
# not read from the DB by this documentation).
_EX_USER = "user-id"
_EX_DOC = "document-id"
_EX_FILE = "document.pdf"
_EX_Q = (
    "According to the schedule table, what value applies for the stated parameters?"
)


def _is_chroma_transient_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(marker in msg for marker in _CHROMA_TRANSIENT_MARKERS):
        return True
    name = type(exc).__name__.lower()
    return name in {"internalerror", "operationalerror"} and "sqlite" in msg


def _harden_chroma_sqlite(persist_dir: str) -> None:
    """Enable WAL + busy timeout on Chroma's SQLite file when present."""
    db_path = os.path.join(persist_dir, "chroma.sqlite3")
    if not os.path.exists(db_path):
        return
    try:
        import sqlite3

        with sqlite3.connect(db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.commit()
    except Exception as exc:
        logger.warning("Could not harden Chroma SQLite pragmas: %s", exc)


@contextmanager
def _chroma_exclusive() -> Iterator[None]:
    """Exclusive Chroma section (process lock + cross-process file lock)."""
    with _chroma_lock:
        lock_path = os.path.join(settings.CHROMA_PERSIST_DIR, ".chroma_access.lock")
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        # Cross-process lock so app + Celery worker do not hit SQLite together.
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except Exception:
                # Windows host / missing fcntl: process lock still applies.
                pass
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


def _chroma_retry(operation_name: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` under the Chroma lock with retries for transient SQLite faults."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _CHROMA_MAX_ATTEMPTS + 1):
        try:
            with _chroma_exclusive():
                return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= _CHROMA_MAX_ATTEMPTS or not _is_chroma_transient_error(exc):
                raise
            delay = _CHROMA_RETRY_BASE_SEC * (2 ** (attempt - 1))
            logger.warning(
                "Chroma %s failed (attempt %s/%s): %s — retrying in %.2fs",
                operation_name,
                attempt,
                _CHROMA_MAX_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


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
        with _chroma_lock:
            if _chroma_client is None:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
                _harden_chroma_sqlite(settings.CHROMA_PERSIST_DIR)

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


def _chunk_metadata(
    chunk: Dict[str, Any], filename: str, document_id: str
) -> Dict[str, Any]:
    """Flatten chunk metadata to Chroma-safe scalars (no nested dicts)."""
    metadata: Dict[str, Any] = {
        "text": chunk["text"],
        "filename": filename,
        "document_id": document_id,
        "page": chunk["page"],
        "chunk_index": chunk["chunk_index"],
        "chunk_type": chunk.get("chunk_type", "text"),
    }
    for key in (
        "page_start",
        "page_end",
        "section",
        "subsection",
        "content_type",
        "extraction_method",
        "ocr_confidence",
        "table_part",
        "table_index",
        "table_id",
        "section_id",
        "parent_section_id",
        "section_level",
        "section_path",
        "source_block_ids",
        "table_ids",
        "index_version",
        "token_count",
    ):
        value = chunk.get(key)
        if value not in (None, ""):
            metadata[key] = value
    for key in ("bbox", "table_json", "layout_metadata"):
        value = chunk.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        metadata[key] = value
    if chunk.get("is_image"):
        metadata["is_image"] = True
        metadata["image_caption"] = chunk.get("image_caption", "")
    return metadata


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
      1) generate_captions_for_chunks(chunks)          # optional; warn on fail
      2) get_embedding_model().embed_documents(batch)  # search_document: prefix
      3) delete_document_chunks(document_id, user_id)  # only after embeddings succeed
      4) store_bm25_index(chunks, ...)                 # bm25.py
      5) collection.add(ids=[f"{document_id}_{chunk_index}"], embeddings, metadatas, documents)
    OUT: 1 (or N for full PDF; prior session noted ~217 chunks for Xolair — ILLUSTRATIVE count)

    Fallbacks: BM25/vision failures are logged; Chroma add still proceeds if BM25 fails.
    """
    if not chunks:
        return 0

    # Generate captions for any extracted images before embedding.
    try:
        from app.rag.ingestion.vision import generate_captions_for_chunks

        generate_captions_for_chunks(chunks)
    except Exception as e:
        logger.warning(f"Could not generate image captions: {e}")

    from app.rag.embeddings.embeddings import get_embedding_model

    embedding_model = get_embedding_model()
    collection_name = get_collection_name(user_id)

    # ── Prepare batch data ───────────────────────────
    texts = [chunk["text"] for chunk in chunks]
    ids = [f"{document_id}_{chunk['chunk_index']}" for chunk in chunks]
    metadatas = [_chunk_metadata(chunk, filename, document_id) for chunk in chunks]

    # ── Embed in batches BEFORE touching the live index ──
    # A reindex must not destroy the previous vectors/BM25 index when embedding
    # fails (network error, provider outage). Deletion happens only after every
    # batch has been embedded successfully.
    batch_size = 50
    embedded_batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        # Generate embeddings outside the Chroma lock (network / Ollama).
        embeddings = embedding_model.embed_documents(batch_texts)
        embedded_batches.append(
            (batch_ids, embeddings, batch_metadatas, batch_texts)
        )

    # Replace the old document index only now that new embeddings are ready.
    delete_document_chunks(document_id, user_id)

    # Build and store BM25 index
    from app.rag.retrieval.bm25 import store_bm25_index
    try:
        store_bm25_index(chunks, document_id, filename, user_id)
    except Exception as e:
        logger.error(f"Could not build BM25 index: {e}")

    # ── Upsert embedded batches ──────────────────────
    total_stored = 0
    for batch_ids, embeddings, batch_metadatas, batch_texts in embedded_batches:
        def _add_batch(
            _ids=batch_ids,
            _embeddings=embeddings,
            _metadatas=batch_metadatas,
            _docs=batch_texts,
        ):
            client = get_chroma_client()
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            collection.add(
                ids=_ids,
                embeddings=_embeddings,
                metadatas=_metadatas,
                documents=_docs,
            )

        _chroma_retry("collection.add", _add_batch)
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
        "text": "Recommended schedule value is 300 ... every 4 weeks.",
        "filename": "document.pdf",
        "document_id": "8c57d30c-...",
        "page": 9,
        "chunk_type": "text",
        "bbox": "",
        "score": 0.8123,   # ILLUSTRATIVE similarity
      }, ... up to 20 items]

    Fallback: missing collection → warning + ``[]``.

    Next: ``retriever.rrf_merge(vector_results, bm25_results)``.
    """
    collection_name = get_collection_name(user_id)

    # ── Build filter ─────────────────────────────────
    where_filter = None
    if document_id:
        where_filter = {"document_id": {"$eq": document_id}}
    elif document_ids:
        where_filter = {"document_id": {"$in": document_ids}}

    def _query():
        client = get_chroma_client()
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            return None
        # ChromaDB always returns ids; do not pass "ids" in include (1.x rejects it).
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

    try:
        results = _chroma_retry("collection.query", _query)
    except Exception as exc:
        if "not found" in str(exc).lower():
            logger.warning(f"Collection {collection_name} not found")
            return []
        raise

    if results is None:
        logger.warning(f"Collection {collection_name} not found")
        return []

    # ── Format results ───────────────────────────────
    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0
            chunk_id = results["ids"][0][i] if results.get("ids") and len(results["ids"]) > 0 else None

            # Convert cosine distance to similarity score (0-1)
            similarity = 1 - distance

            chunk = {
                "id": chunk_id,
                "text": doc,
                "filename": metadata.get("filename", ""),
                "document_id": metadata.get("document_id", ""),
                "page": metadata.get("page", 1),
                "chunk_type": metadata.get("chunk_type", "text"),
                "bbox": metadata.get("bbox", ""),
                "score": round(similarity, 4),
            }
            for key in (
                "page_start",
                "page_end",
                "section",
                "subsection",
                "content_type",
                "extraction_method",
                "ocr_confidence",
                "table_part",
                "table_index",
                "table_id",
                "chunk_index",
                "table_json",
                "layout_metadata",
                "is_image",
                "image_caption",
                "section_id",
                "parent_section_id",
                "section_level",
                "section_path",
                "source_block_ids",
                "table_ids",
                "index_version",
                "token_count",
            ):
                if key in metadata and metadata[key] not in (None, ""):
                    chunk[key] = metadata[key]
            chunks.append(chunk)

    return chunks


def get_chunks_by_ids(user_id: str, chunk_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch stored chunks by Chroma ids (used for adjacent-table expansion)."""
    if not chunk_ids:
        return []
    collection_name = get_collection_name(user_id)
    unique_ids = []
    seen = set()
    for cid in chunk_ids:
        text = str(cid)
        if text and text not in seen:
            seen.add(text)
            unique_ids.append(text)

    def _get():
        client = get_chroma_client()
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            return None
        return collection.get(
            ids=unique_ids,
            include=["documents", "metadatas"],
        )

    try:
        results = _chroma_retry("collection.get_by_ids", _get)
    except Exception as exc:
        logger.debug("get_chunks_by_ids failed: %s", exc)
        return []
    if not results or not results.get("ids"):
        return []

    chunks: List[Dict[str, Any]] = []
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    for i, chunk_id in enumerate(ids):
        metadata = metas[i] if i < len(metas) and metas[i] else {}
        text = docs[i] if i < len(docs) else ""
        chunk = {
            "id": chunk_id,
            "text": text or "",
            "filename": metadata.get("filename", ""),
            "document_id": metadata.get("document_id", ""),
            "page": metadata.get("page", 1),
            "chunk_type": metadata.get("chunk_type", "text"),
            "bbox": metadata.get("bbox", ""),
            "score": 0.0,
        }
        for key in (
            "page_start",
            "page_end",
            "section",
            "subsection",
            "content_type",
            "extraction_method",
            "chunk_index",
            "table_id",
            "table_json",
            "layout_metadata",
            "section_id",
            "parent_section_id",
            "section_level",
            "section_path",
            "source_block_ids",
            "table_ids",
            "index_version",
            "token_count",
        ):
            if key in metadata and metadata[key] not in (None, ""):
                chunk[key] = metadata[key]
        chunks.append(chunk)
    return chunks


def delete_document_chunks(document_id: str, user_id: str):
    """Delete BM25 pickle + all Chroma ids where metadata.document_id matches.

    Exact example
    -------------
    delete_document_chunks(_EX_DOC, _EX_USER)
      → delete_bm25_index(...)
      → collection.get(where={"document_id": {"$eq": _EX_DOC}}) → delete(ids=...)
    """
    collection_name = get_collection_name(user_id)

    try:
        from app.rag.retrieval.bm25 import delete_bm25_index
        delete_bm25_index(document_id, user_id)
    except Exception as e:
        logger.warning(f"Error deleting BM25 index: {e}")

    try:
        from app.rag.tables.store import delete_document_tables

        delete_document_tables(user_id=user_id, document_id=document_id)
    except Exception as e:
        logger.debug(f"Error deleting table store: {e}")

    def _delete_doc_vectors():
        client = get_chroma_client()
        collection = client.get_collection(name=collection_name)
        results = collection.get(
            where={"document_id": {"$eq": document_id}},
            include=[],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    try:
        deleted = _chroma_retry("collection.delete", _delete_doc_vectors)
        if deleted:
            logger.info(f"Deleted {deleted} chunks for document {document_id}")
    except Exception as e:
        logger.warning(f"Error deleting chunks: {e}")


def delete_user_collection(user_id: str):
    """Wipe all BM25 indexes and the entire Chroma collection for one user."""
    collection_name = get_collection_name(user_id)

    try:
        from app.rag.retrieval.bm25 import delete_user_bm25_indexes
        delete_user_bm25_indexes(user_id)
    except Exception as e:
        logger.warning(f"Error deleting user BM25 indexes: {e}")

    def _delete_collection():
        client = get_chroma_client()
        client.delete_collection(name=collection_name)

    try:
        _chroma_retry("delete_collection", _delete_collection)
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

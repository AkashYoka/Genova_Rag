# Genova AI — GraphRAG Implementation Notes

What changed, why, and how to operate it.

---

## 1. Files added

| File | Purpose |
|---|---|
| `backend/app/graph/__init__.py` | Package marker / module index |
| `backend/app/graph/schemas.py` | Extraction dataclasses, payload validation, type normalisation, confidence clamping |
| `backend/app/graph/extraction.py` | Domain-neutral extraction prompt, chunk sampling, batching, JSON recovery, failure accounting |
| `backend/app/graph/resolution.py` | Entity resolution: normalisation, alias/acronym match, fuzzy duplicate flagging |
| `backend/app/graph/store.py` | SQL persistence and bounded graph reads |
| `backend/app/graph/service.py` | Index/reindex/remove orchestration + status |
| `backend/app/graph/retrieval.py` | GraphRAG evidence retrieval, prompt block formatting, citations |
| `backend/app/graph/query.py` | Read-only queries behind the API |
| `backend/tests/test_graph_rag.py` | Cross-document pipeline, deletion, isolation, retrieval and API tests |
| `docs/graph-rag-*.md` | This documentation set |

## 2. Files modified

| File | Change |
|---|---|
| `backend/app/models.py` | Added `GraphEntity`, `GraphEntityMention`, `GraphRelationship`, `GraphRelationshipEvidence`, `GraphIndexStatus` |
| `backend/app/config.py` | Added `GRAPH_INDEXING_ENABLED`, `GRAPH_RETRIEVAL_ENABLED` and the `GRAPH_*` tuning values |
| `backend/app/main.py` | Mounted `app.routes.graph.router` under `/api/v1` |
| `backend/app/routes/graph.py` | Workspace graph endpoints (overview, search, neighbourhood, entity, relationship, documents, status, reindex, forget); legacy per-document endpoints retained |
| `backend/app/services/document_ingestion.py` | Replaced the old per-document spaCy graph block with `graph.service.index_document` (stage `graph`, progress 55) |
| `backend/app/routes/documents.py` | Soft delete now removes that document's graph contribution (shared entities preserved) |
| `backend/app/services/cleanup.py` | Hard delete uses `graph.service.remove_document` instead of the legacy JSON graph delete |
| `backend/app/rag/llm/agent.py` | Added `_collect_graph_evidence`, `_graph_payload`, `_merge_sources`; both direct answer paths (non-stream and stream) now add graph context, graph citations and a `graph` metadata field/event |
| `backend/app/rag/utils/tools.py` | `PDFSearchTool` now reads graph context from `app.graph.retrieval` (the HuggingFace ReAct path) |
| `backend/app/schemas.py` | Added `GraphContextInfo`, and `graph` on `ChatResponse` / `ChatMessageResponse` |
| `backend/app/routes/chat.py` | Persist, parse and return graph metadata in chat history and responses; forward the streaming `graph` event over HTTP and WebSocket |
| `frontend/src/components/graph/KnowledgeGraph.tsx` | Rewritten against the workspace API: document selector, search, type/relationship filters, depth control, node/edge detail with evidence, lazy expansion, loading/empty/error states |
| `frontend/src/app/dashboard/page.tsx` | Renders the graph explorer in a Sheet, opened from the header |
| `frontend/src/components/layout/Header.tsx` | Added the knowledge-graph toggle button |
| `frontend/src/store/chat-store.ts` | `GraphContextInfo` type and `ChatMsg.graph` |
| `frontend/src/components/chat/ChatPanel.tsx` | Handles the `graph` stream event, restores it from history, renders a graph badge above citations |
| `backend/tests/test_documents.py` | The two tests that asserted the *old* spaCy graph contract now assert the graph service contract |
| `backend/tests/test_rag_tools.py` | Graph-context fake updated to the `document_ids` scope parameter |

Deleted: `backend/app/routes/patch_graph_context.py` (a patch script accidentally committed as a route module; it was never imported).

## 3. Pipeline changes in detail

### 3.1 Ingestion

```text
extract text → chunk → graph indexing → embeddings/Chroma/BM25 → keywords → summary → ready
```

`ingest_document` now calls:

```python
from app.graph.service import index_document as index_document_graph
graph_result = index_document_graph(document_id, user_id, chunks)
```

Why before embeddings: the graph service needs the same chunk text and can fail independently; a graph failure is logged and recorded in `graph_index_status` but does not abort embedding or the upload.

Why in-memory chunks: the Chroma ids are `{document_id}_{chunk_index}`, and `graph.extraction` derives the same ids, so graph evidence and vector citations line up without a second read.

### 3.2 Reindexing

`graph.service.reindex_document` reloads the document's chunks from Chroma (`load_document_chunks`) and re-runs extraction. Exposed at `POST /api/v1/graph/documents/{id}/reindex` (202, background task), and surfaced in the UI for failed documents.

### 3.3 Deletion semantics

| Event | Effect on graph |
|---|---|
| Document soft delete | That document's mentions and relationship evidence are removed; relationships/entities with no remaining evidence are pruned |
| Hard delete (cleanup job) | Same, plus the status row is deleted |
| Remove explicit graph contribution | `DELETE /api/v1/graph/documents/{id}` |

Shared facts survive: an entity mentioned in documents A and B stays when A is deleted, with `mention_count`/`document_count` recalculated.

### 3.4 Answer generation

Both `_generate_answer_direct` and `_generate_answer_stream_direct` now:

1. collect bounded graph evidence for the question and scope (`document_id` + `document_ids`);
2. append the formatted graph block to the RAG context;
3. merge graph citations into `sources`;
4. return/emit `graph` metadata (`entities`, `relationship_count`, `source_count`, `truncated`).

If graph retrieval is disabled, empty, or throws, the functions behave exactly as before. The HuggingFace ReAct path gets graph context through `PDFSearchTool`.

### 3.5 Chat persistence

`graph` is stored inside `sources_json` (enriched payload) and returned on history responses, so a reloaded conversation still shows which answers used the graph. Cached answers (`app/cache.py`) store answer + sources only, so a cache hit shows citations but not the graph badge.

## 4. Configuration

Add to `.env` only if changing defaults:

```dotenv
# Feature switches
GRAPH_INDEXING_ENABLED=true
GRAPH_RETRIEVAL_ENABLED=true

# Extraction cost
GRAPH_EXTRACTION_MODEL=            # blank → configured chat model
GRAPH_EXTRACTION_MAX_CHUNKS=40
GRAPH_EXTRACTION_BATCH_CHUNKS=4
GRAPH_MIN_CONFIDENCE=0.35

# Resolution
GRAPH_ENTITY_RESOLUTION_THRESHOLD=0.9

# Retrieval bounds
GRAPH_TRAVERSAL_DEPTH=2
GRAPH_MAX_QUERY_ENTITIES=8
GRAPH_MAX_NEIGHBORHOOD_NODES=60
GRAPH_MAX_GRAPH_RELATIONSHIPS=30
GRAPH_MAX_EVIDENCE_SOURCES=3
```

`GRAPH_INDEXING_ENABLED=false` writes a `disabled` status row and skips extraction, which is the fastest way to pause LLM spend without code changes.

## 5. Database

No migration is required: `init_db()` runs `Base.metadata.create_all` at startup, and the new tables are additive. To create them manually:

```sql
-- Equivalent DDL is generated by SQLAlchemy; the tables are:
graph_entities
graph_entity_mentions
graph_relationships
graph_relationship_evidence
graph_index_status
```

Indexes exist on `user_id` (all tables), `normalized_name`, `entity_type`, `entity_id`, `source_entity_id`, `target_entity_id`, `relationship_id`, `document_id`.

Backfill for documents indexed before this feature: `POST /api/v1/graph/documents/{id}/reindex` per document (the UI exposes a retry action on failed rows; for a bulk backfill iterate document ids and call the service).

## 6. Frontend

* **Entry point:** header graph button → right-side `Sheet` on the dashboard (width `min(1180px, 96vw)`).
* **Scope:** the sheet opens scoped to the active document when one is selected; the document list lets you widen to any set or the whole account.
* **Data loading:** `neighborhood` only, capped by the backend; double-clicking a node (or "Expand one more hop") merges a single hop.
* **Filters:** document checkboxes, entity-type chips, relationship-type select, label toggle — all applied client-side to the loaded subgraph, so filtering never triggers a full-graph load.
* **Details:** entity panel shows type, description, aliases, counts, documents, connections and quoted evidence; relationship panel shows direction, confidence, description and every supporting passage.
* **States:** loading spinner, "No graph yet" empty state with retry-failed option, error state with retry, truncated-neighbourhood banner.
* **Chat:** assistant messages show a "Knowledge graph · N relationships · M sources" badge when the answer used graph evidence.

## 7. Operations

| Symptom | Check |
|---|---|
| Graph empty after upload | `GET /api/v1/graph/documents/{id}/status` — `failed` carries the model error; `disabled` means indexing is off |
| All extraction batches fail | LLM provider unreachable or key rejected; the error is stored in the status row |
| Entities duplicated across documents | Same name but different `entity_type`, or below the fuzzy threshold — inspect `possible_duplicate_of` in the entity panel |
| Answers lost graph context | `GRAPH_RETRIEVAL_ENABLED`; also the question may not mention any stored entity |
| Graph endpoint 404 | Document not owned by the caller, or no stored graph for it |

## 8. Deliberate limitations

* No destructive auto-merge of near-duplicate entities; they are flagged for review instead.
* Graph traversal is bounded (depth ≤ 3, node/edge caps); the UI never requests the entire graph.
* The legacy per-document JSON graph is read-only legacy: nothing new writes it, and `/graph/{document_id}` returns 404 for documents indexed after this change.
* Workspace isolation is account isolation, matching `documents.user_id` in the existing schema.

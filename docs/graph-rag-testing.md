# Genova AI — GraphRAG Testing

How the feature is verified, what each test asserts, and the known pre-existing failures in this repository.

---

## 1. Running the suite

```bash
cd backend
./.venv/Scripts/python.exe -m pytest tests/test_graph_rag.py -q       # graph feature
./.venv/Scripts/python.exe -m pytest tests -q                          # whole backend
```

Frontend checks:

```bash
cd frontend
npx tsc --noEmit
npx eslint src/components/graph/KnowledgeGraph.tsx src/components/chat/ChatPanel.tsx \
            src/app/dashboard/page.tsx src/store/chat-store.ts
```

## 2. `backend/tests/test_graph_rag.py`

Fixtures are domain-neutral: `DOC_A` (system architecture), `DOC_B` (API surface) and `DOC_C` (data handling) share the entity "Application Backend" / "Primary Database" so cross-document behaviour is observable. Extraction is injected through `graph.service.index_document(..., completer=...)`, so no network or LLM is required.

### Entity and relationship extraction

| Test | Asserts |
|---|---|
| `test_extraction_accepts_arbitrary_entity_and_relationship_types` | Free-text types are normalised (`"Software Component"` → `SOFTWARE_COMPONENT`, `"data store"` → `DATA_STORE`); the evidenced relationship survives |
| `test_extraction_drops_relationships_without_evidence_or_known_endpoints` | A relationship with no quote and one pointing at an unknown entity are rejected and counted; direction and evidence of the kept edge are preserved |
| `test_extraction_normalises_model_defined_types_from_any_domain` | Financial, research and business payloads all pass — no hardcoded vocabulary |
| `test_extraction_prompt_is_domain_neutral` | The prompt contains no medical/clinical vocabulary and explicitly allows free-text types |

### Entity resolution

| Test | Asserts |
|---|---|
| `test_resolution_merges_exact_and_alias_matches` | Case-normalised exact match, stored alias and acronym all resolve to the existing entity |
| `test_resolution_keeps_ambiguous_names_separate` | A near-duplicate name creates a *new* entity with `possible_duplicate_of` set — never a silent merge |
| `test_resolution_never_merges_across_entity_types` | Same name, different type stays separate |

### Cross-document graph

| Test | Asserts |
|---|---|
| `test_indexing_merges_shared_entities_across_documents` | 3 documents → one canonical "Application Backend" with 2 mentions across 2 documents and 2 distinct outgoing relationships; the lowercase variant from doc B resolves into it |
| `test_neighborhood_traverses_across_documents` | Depth 1 returns the direct neighbours; depth 2 reaches `Customer Records` via doc C — a path that only exists across documents |
| `test_entity_detail_reports_documents_and_evidence` | Detail lists both documents, one mention per document, and non-empty quotations |
| `test_removing_one_document_preserves_shared_entities` | Deleting doc A keeps the shared entity (mention count 1, supported by doc B), drops the doc-A-only relationship, and keeps an entity supported by doc C |
| `test_removing_all_documents_clears_the_graph` | Removing every document prunes all entities |
| `test_graph_is_isolated_between_users` | Another account sees no entities, no detail, `404`-equivalent access and an empty neighbourhood |
| `test_failed_extraction_is_recorded_and_leaves_graph_intact` | A provider error marks `graph_index_status = failed` with the underlying message, and previously stored evidence is not destroyed |

### Retrieval feeding answers

| Test | Asserts |
|---|---|
| `test_graph_retrieval_returns_evidence_and_citations` | Non-empty context block containing the seed entity, with citations whose filenames and quotes are present |
| `test_graph_retrieval_respects_document_scope` | A doc-C-only scope cannot surface the doc-A "uses" fact, while the unscoped query can |
| `test_entity_context_wrapper_returns_prompt_text` | The convenience wrapper (used by the tool layer) returns prompt text |

### HTTP surface

| Test | Asserts |
|---|---|
| `test_graph_api_serves_the_authenticated_users_graph` | `/overview`, `/search`, `/entities/{id}`, `/neighborhood` and `/documents` all respond with the account's data |
| `test_graph_api_requires_authentication` | `/overview` and `/search` reject anonymous calls (401/403) |
| `test_graph_api_does_not_expose_another_users_graph` | Another account gets `404` for a known entity id and an empty overview |
| `test_chat_history_round_trip_preserves_graph_metadata` | `graph` survives `sources_json` persistence and is re-parsed into `ChatMessageResponse` |

## 3. Existing tests updated (not removed)

Two tests asserted the previous per-document spaCy graph contract. They now assert the new contract with the same intent:

| File | Test | Change |
|---|---|---|
| `tests/test_documents.py` | `test_ingest_document_builds_and_saves_graph` | Patches `app.graph.service.index_document` and asserts it receives the document id, user id and the document's chunks during ingestion |
| `tests/test_documents.py` | `test_delete_document_soft_deletes_and_hides_document` | Asserts the soft delete removes the document's graph contribution (previously asserted nothing was deleted) |
| `tests/test_rag_tools.py` | `test_pdf_search_tool_formats_chunks_and_graph_context` | The injected graph function now receives `document_ids=[document_id]` |

The legacy `tests/test_graph_builder.py`, `tests/test_graph_retriever.py` and `tests/test_graphrag_agent.py` are untouched and still pass; the legacy modules remain in the codebase for old data.

## 4. Observed results

Backend, on this machine (`./.venv/Scripts/python.exe -m pytest tests -q`):

* `tests/test_graph_rag.py` — 24 passed
* Full suite — 327+ passed; the failures are the pre-existing ones listed below.

Frontend:

* `npx tsc --noEmit` — clean.
* `npx eslint` on the graph panel, chat panel, dashboard page and chat store — clean (the dashboard `notesFocusToken` effect error and the `Header.tsx` unused-import warning exist on `HEAD` and are unrelated).

## 5. Known failures

None. The two tests that previously failed on `HEAD` — `tests/test_chat.py::test_agent_dynamic_token` (per-request HuggingFace token) and `tests/test_batch_upload.py::test_batch_upload_document_persisted_in_db` (eager-mode ingestion) — now pass; the suite is green apart from unrelated pre-existing lint findings.

## 6. Not covered by automated tests

* A live LLM extraction round trip (the suite injects the completion function by design so CI is deterministic and free).
* Browser interaction tests for pan/zoom/drag; the graph panel is exercised manually and type/lint-checked, but no Playwright/Vitest test was added.
* Prompt-injection behaviour of extracted text: evidence strings are stored and rendered as text, and the answer prompt labels graph context as supporting evidence, but this is not asserted by a test.

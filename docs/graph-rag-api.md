# Genova AI — GraphRAG API Reference

Base path: `/api/v1/graph`
Authentication: `Authorization: Bearer <access token>` — identical to the rest of the API (`get_current_user`).
All responses are scoped to the authenticated account; another account's ids return `404`, never data.

---

## 1. Overview

`GET /api/v1/graph/overview`

| Query | Type | Default | Notes |
|---|---|---|---|
| `document_ids` | string | – | Comma-separated document ids to scope the graph |

```json
{
  "available": true,
  "scope": "workspace",
  "document_ids": [],
  "entity_count": 128,
  "relationship_count": 214,
  "document_count": 4,
  "entity_types": [{ "type": "SYSTEM", "count": 22 }],
  "relationship_types": [{ "type": "uses", "count": 31 }],
  "top_entities": [
    {
      "id": "1f8c…",
      "canonical_name": "Application Backend",
      "normalized_name": "application backend",
      "entity_type": "SYSTEM",
      "description": "The service that serves requests.",
      "aliases": ["Backend"],
      "mention_count": 12,
      "document_count": 3,
      "confidence": 0.91,
      "possible_duplicate_of": null
    }
  ]
}
```

`available` is `false` when the account has no entities — the UI uses this for its empty state.

## 2. Search entities

`GET /api/v1/graph/search`

| Query | Type | Default | Notes |
|---|---|---|---|
| `q` | string | `""` | Matched against name, aliases and description |
| `entity_type` | string | – | Exact normalised type, e.g. `SYSTEM` |
| `document_ids` | string | – | Comma-separated scope |
| `limit` | int | `25` | 1–100 |

```json
{ "query": "public", "results": [ { "id": "…", "canonical_name": "Public API", "…": "…" } ] }
```

Ordering: exact name, prefix, substring, alias, then description matches; ties broken by mention/document counts.

## 3. Bounded neighbourhood

`GET /api/v1/graph/neighborhood`

| Query | Type | Default | Notes |
|---|---|---|---|
| `entity_id` | string | – | Seed entity; omitted = the busiest entities |
| `depth` | int | `1` | 1–3 |
| `limit` | int | `60` | 1–200 nodes |
| `document_ids` | string | – | Restricts both seeds and traversed relationships |

```json
{
  "nodes": [ { "id": "…", "canonical_name": "Application Backend", "entity_type": "SYSTEM", "…": "…" } ],
  "edges": [
    {
      "id": "…",
      "source": { "id": "a", "name": "Application Backend", "type": "SYSTEM" },
      "target": { "id": "b", "name": "Primary Database", "type": "DATA_STORE" },
      "relation_type": "uses",
      "description": "The backend reads and writes rows.",
      "confidence": 0.82,
      "evidence_count": 2,
      "evidence": [
        {
          "document_id": "…",
          "document_name": "architecture.pdf",
          "chunk_id": "doc_42",
          "page": 7,
          "section": "Storage",
          "evidence": "The Application Backend uses the Primary Database for storage.",
          "confidence": 0.82
        }
      ]
    }
  ],
  "truncated": false,
  "seed_entity_ids": ["a"],
  "depth": 1
}
```

`truncated: true` means a cap was hit; narrow the scope rather than paging the whole graph.

## 4. Entity details

`GET /api/v1/graph/entities/{entity_id}` → `404` when not owned / unknown.

```json
{
  "id": "…",
  "canonical_name": "Application Backend",
  "normalized_name": "application backend",
  "entity_type": "SYSTEM",
  "description": "…",
  "aliases": ["Backend"],
  "mention_count": 12,
  "document_count": 3,
  "confidence": 0.91,
  "possible_duplicate_of": null,
  "mentions": [
    {
      "document_id": "…",
      "document_name": "architecture.pdf",
      "chunk_id": "doc_42",
      "page": 7,
      "section": "Storage",
      "evidence": "The Application Backend uses the Primary Database for storage.",
      "confidence": 0.9
    }
  ],
  "documents": [{ "id": "…", "name": "architecture.pdf" }],
  "connected": [
    {
      "id": "…",
      "name": "Primary Database",
      "type": "DATA_STORE",
      "relation_type": "uses",
      "direction": "outgoing",
      "confidence": 0.82
    }
  ],
  "relationships": [ { "id": "…", "source": {}, "target": {}, "relation_type": "uses", "evidence": [] } ]
}
```

## 5. Relationship details

`GET /api/v1/graph/relationships/{relationship_id}`

Shape matches the edge object in §3 plus `"documents": [{ "id", "name" }]`, aggregated from its evidence rows. This is the "why does this edge exist?" endpoint.

## 6. Documents and indexing status

`GET /api/v1/graph/documents`

```json
{
  "documents": [
    {
      "document_id": "…",
      "document_name": "architecture.pdf",
      "status": "ready",
      "entity_count": 42,
      "relationship_count": 61,
      "chunk_count": 40,
      "attempts": 1,
      "error": null,
      "updated_at": "2026-10-30T16:04:11"
    }
  ]
}
```

`status` ∈ `pending | indexing | ready | failed | disabled | not_indexed`.

`GET /api/v1/graph/documents/{document_id}/status` → single row (same shape). `404` if the document is not owned by the caller.

`POST /api/v1/graph/documents/{document_id}/reindex` → `202`

```json
{ "document_id": "…", "status": "queued", "message": "Graph reindexing started" }
```

Re-reads chunks from the vector store and re-runs extraction. `409` when `GRAPH_INDEXING_ENABLED=false`.

`DELETE /api/v1/graph/documents/{document_id}`

```json
{ "document_id": "…", "removed": { "mentions": 8, "evidence": 5, "relationships": 1, "entities": 0 } }
```

Removes only that document's contribution; shared entities/relationships survive.

## 7. Chat integration

The graph also appears in the chat surface — no separate call needed.

* `POST /api/v1/chat/ask` → `ChatResponse.graph`:

```json
{
  "answer": "…",
  "sources": [ … ],
  "graph": {
    "entities": ["Application Backend", "Primary Database"],
    "relationship_count": 3,
    "source_count": 2,
    "truncated": false
  }
}
```

* `POST /api/v1/chat/ask/stream` and `WS /api/v1/chat/ws` emit, before `sources`:

```text
data: {"type":"graph","data":{"entities":["Application Backend"],"relationship_count":2,"source_count":1,"truncated":false}}
```

* `GET /api/v1/chat/history/{document_id}` and `/chat/history/session/{id}` include `graph` on assistant messages when it was used.
* Graph-derived citations are merged into `sources`, so citation chips and PDF page links work unchanged.

## 8. Legacy endpoints (unchanged)

| Endpoint | Behaviour |
|---|---|
| `GET /api/v1/graph/{document_id}` | Per-document spaCy co-occurrence graph from `data/graphs/*.json`; `404` when no file exists (all documents indexed after this feature) |
| `GET /api/v1/graph/{document_id}/summary` | Counts for that legacy graph |

Kept only so previously generated graphs stay readable. New UI surfaces use the workspace endpoints.

## 9. Errors

| Status | Cause |
|---|---|
| `401 / 403` | Missing or invalid token |
| `404` | Unknown entity/relationship/document, or owned by another account |
| `409` | Reindex requested while graph indexing is disabled |
| `422` | Invalid query parameter (depth/limit out of range) |

Route declaration order matters: static paths (`/overview`, `/search`, `/neighborhood`, `/entities/...`, `/relationships/...`, `/documents`) are declared before `/{document_id}` so `/overview` is never parsed as a document id.

"""TableRAG: structured table storage, indexing, routing, and retrieval.

Builds on the Textract → ``TableData`` → ``table_json`` path without replacing
text RAG. Tables are persisted as complete structures in SQL, and table/row
textual representations are indexed through the same Chroma + BM25 stores used
for paragraph chunks.

LLM table understanding (during semantic chunking) attaches interpretation
metadata only; extracted cell values remain ground truth.
"""

from app.rag.table_rag.hydrate import hydrate_retrieved_tables
from app.rag.table_rag.indexing import expand_table_rag_chunks, extract_tables_from_chunks
from app.rag.table_rag.routing import QueryModality, route_query
from app.rag.table_rag.schema import TableLookupDecision, TablePayload, TableUnderstanding
from app.rag.table_rag.store import (
    delete_document_tables,
    get_document_table,
    list_document_tables,
    persist_document_tables,
)
from app.rag.table_rag.understanding import (
    apply_understanding_to_table,
    profile_full_table_grid,
    understand_document_tables,
)

__all__ = [
    "QueryModality",
    "TableLookupDecision",
    "TablePayload",
    "TableUnderstanding",
    "apply_understanding_to_table",
    "delete_document_tables",
    "expand_table_rag_chunks",
    "extract_tables_from_chunks",
    "get_document_table",
    "hydrate_retrieved_tables",
    "list_document_tables",
    "persist_document_tables",
    "profile_full_table_grid",
    "route_query",
    "understand_document_tables",
]

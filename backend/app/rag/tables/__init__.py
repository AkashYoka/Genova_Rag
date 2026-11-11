"""Application-facing wrapper around the typed table layer."""
from app.rag.tables.normalize import (
    linearize_facts,
    normalize_grid,
    normalize_table_data,
    record_from_table_data,
)
from app.rag.tables.schema import TableCell, TableRecord, make_table_id
from app.rag.tables.store import (
    attach_complete_tables,
    get_record,
    load_records,
    save_records,
    save_tables_from_chunks,
    tables_from_chunks,
)

__all__ = [
    "TableCell",
    "TableRecord",
    "make_table_id",
    "linearize_facts",
    "normalize_grid",
    "normalize_table_data",
    "record_from_table_data",
    "attach_complete_tables",
    "get_record",
    "load_records",
    "save_records",
    "save_tables_from_chunks",
    "tables_from_chunks",
]

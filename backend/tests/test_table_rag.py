"""TableRAG unit tests: indexing, routing, hydrate, merged cells, multi-page."""
from __future__ import annotations

import json

import pytest

from app.rag.table_rag.ids import ensure_table_id, make_table_id, parse_table_payload
from app.rag.table_rag.indexing import (
    CHUNK_TYPE_TABLE_ROW,
    CHUNK_TYPE_TABLE_SUMMARY,
    expand_table_rag_chunks,
    extract_tables_from_chunks,
)
from app.rag.table_rag.representations import (
    compact_table_prompt_block,
    table_row_text,
    table_summary_text,
)
from app.rag.table_rag.routing import QueryModality, route_query
from app.rag.generation.structured_lookup import lookup_cell_in_table, lookup_from_chunk
from app.rag.generation.table_interpretation import interpret_table_question
from app.rag.llm.agent import _format_context, _table_json_appendix
from app.rag.retrieval.retriever import _boost_table_candidates


def _dosing_table(*, page: int = 3, table_id: str = "t_dosing") -> dict:
    return {
        "table_id": table_id,
        "page": page,
        "headers": [
            "Assay (IU/mL)",
            "Schedule",
            "30-60 kg",
            ">60-70 kg",
            ">70-90 kg",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "150", "225", "300"],
            [">300-400", "Every 2 weeks", "225", "225", "300"],
            ["≥30-100", "Every 4 weeks", "150", "150", "150"],
        ],
        "n_rows": 4,
        "n_cols": 5,
        "caption": "Recommended doses",
        "extraction_method": "textract",
        "bbox": [72, 120, 540, 400],
        "merged_cells": [
            {
                "row_index": 0,
                "col_index": 0,
                "row_span": 1,
                "col_span": 2,
                "text": "Assay / Schedule",
            }
        ],
        "cells": [],
        "header_rows": [["Assay (IU/mL)", "Schedule", "30-60 kg", ">60-70 kg", ">70-90 kg"]],
        "low_confidence_cells": [],
        "linearized_facts": [
            "Assay >300-400, Schedule Every 2 weeks, >70-90 kg = 300",
        ],
    }


def _table_chunk(table: dict, *, chunk_index: int = 0) -> dict:
    return {
        "text": "TABLE\n" + "\n".join(
            " | ".join(str(c) for c in row) for row in ([table["headers"]] + table["rows"])
        ),
        "page": table["page"],
        "page_start": table["page"],
        "page_end": table["page"],
        "chunk_index": chunk_index,
        "chunk_type": "table",
        "content_type": "table",
        "section": "Dosage",
        "subsection": "Adults",
        "extraction_method": "textract",
        "table_id": table["table_id"],
        "table_json": json.dumps(table),
    }


# ── Routing ──────────────────────────────────────────────────────────────────


def test_route_numerical_table_question():
    route = route_query(
        "What dose applies for assay 350 IU/mL and body weight 78 kg every 2 weeks?"
    )
    assert route.modality == QueryModality.TABLE
    assert route.needs_tables
    assert route.wants_structured_lookup


def test_route_paragraph_question():
    route = route_query("What is the mechanism of action described in the overview?")
    assert route.modality == QueryModality.PARAGRAPH
    assert route.needs_paragraphs
    assert not route.needs_tables


def test_route_combined_text_and_table():
    route = route_query(
        "Explain the indication and what dose from the table applies for 70 kg and 250 IU/mL."
    )
    assert route.modality in (QueryModality.BOTH, QueryModality.TABLE)
    assert route.needs_tables


# ── Indexing ─────────────────────────────────────────────────────────────────


def test_expand_creates_summary_and_row_chunks():
    table = _dosing_table()
    chunks = [_table_chunk(table), {"text": "Prose paragraph.", "page": 2, "chunk_index": 1, "chunk_type": "text"}]
    expanded, records = expand_table_rag_chunks(
        chunks, document_id="doc-1", filename="label.pdf"
    )
    assert len(records) == 1
    assert records[0]["table_id"] == "t_dosing"
    summaries = [c for c in expanded if c["chunk_type"] == CHUNK_TYPE_TABLE_SUMMARY]
    rows = [c for c in expanded if c["chunk_type"] == CHUNK_TYPE_TABLE_ROW]
    assert len(summaries) == 1
    assert len(rows) == 3
    assert all(c.get("table_id") == "t_dosing" for c in summaries + rows)
    # Contiguous indexes for Chroma ids.
    assert [c["chunk_index"] for c in expanded] == list(range(len(expanded)))
    assert "TABLE SUMMARY" in summaries[0]["text"]
    assert "row_index 1" in rows[1]["text"]
    assert "300" in rows[1]["text"]


def test_extract_prefers_complete_table_over_split_part():
    full = _dosing_table()
    partial = dict(full)
    partial["rows"] = full["rows"][:1]
    partial["part"] = 1
    chunks = [
        _table_chunk(partial, chunk_index=0),
        _table_chunk(full, chunk_index=1),
    ]
    records = extract_tables_from_chunks(chunks, document_id="doc-1")
    assert len(records) == 1
    assert len(records[0]["table"]["rows"]) == 3


def test_multi_page_tables_get_distinct_ids():
    t1 = _dosing_table(page=3, table_id="")
    t2 = _dosing_table(page=4, table_id="")
    t2["headers"] = ["Event", "Rate"]
    t2["rows"] = [["Headache", "3%"], ["Rash", "1%"]]
    ensure_table_id(t1, ordinal=0, document_id="doc-x")
    ensure_table_id(t2, ordinal=0, document_id="doc-x")
    assert t1["table_id"] != t2["table_id"]
    chunks = [_table_chunk(t1, chunk_index=0), _table_chunk(t2, chunk_index=1)]
    records = extract_tables_from_chunks(chunks, document_id="doc-x")
    assert len(records) == 2
    pages = {r["page"] for r in records}
    assert pages == {3, 4}


def test_merged_cells_preserved_in_summary_and_prompt():
    table = _dosing_table()
    summary = table_summary_text(table, filename="x.pdf", section="Dosage")
    assert "merged_cells" in summary
    block = compact_table_prompt_block(table)
    assert "merged_cells" in block
    assert "span=1x2" in block


def test_low_confidence_flagged_explicitly():
    table = _dosing_table()
    table["low_confidence_cells"] = [
        {"row": 1, "col": 4, "text": "300", "confidence": 0.2}
    ]
    table["needs_verification"] = True
    summary = table_summary_text(table)
    assert "low-confidence" in summary.lower() or "extraction_warning" in summary
    block = compact_table_prompt_block(table)
    assert "WARNING" in block


# ── Exact value / multi-row / numerical ──────────────────────────────────────


def test_exact_value_lookup_via_row_representation():
    table = _dosing_table()
    row_text = table_row_text(table, 1, table["rows"][1], filename="x.pdf")
    assert "225" in row_text
    assert ">300-400" in row_text
    result = lookup_cell_in_table(
        table,
        "What value applies for assay 350 IU/mL and body weight 78 kg every 2 weeks?",
        page=3,
    )
    assert hasattr(result, "value")
    assert result.value == "300"


def test_multi_row_comparison_question_uses_table_chunks():
    table = _dosing_table()
    chunk = _table_chunk(table)
    q = (
        "Compare the recommended values for assay 250 IU/mL vs 350 IU/mL "
        "at body weight 78 kg from the dosing table."
    )
    interp = interpret_table_question(q, [chunk])
    assert interp.is_structured_lookup
    # Comparison may verify one intersecting cell; at least table path engaged.
    assert interp.verified is not None or interp.insufficient is not None


def test_lookup_from_hydrated_table_field():
    table = _dosing_table()
    chunk = {
        "filename": "x.pdf",
        "page": 3,
        "chunk_type": "table_row",
        "table": table,
        "table_id": table["table_id"],
    }
    result = lookup_from_chunk(
        chunk,
        "What value applies for assay 350 IU/mL and body weight 65 kg every 2 weeks?",
    )
    assert hasattr(result, "value")
    assert result.value == "225"


def test_combined_text_plus_table_context_formatting():
    table = _dosing_table()
    chunks = [
        {
            "filename": "xolair_prescribing.pdf",
            "page": 2,
            "chunk_type": "text",
            "section": "Indications",
            "text": "Indicated for moderate to severe persistent asthma.",
        },
        {
            "filename": "xolair_prescribing.pdf",
            "page": 3,
            "chunk_type": "table_summary",
            "section": "Dosage",
            "text": "TABLE SUMMARY dosing",
            "table_id": table["table_id"],
            "table": table,
            "table_prompt_block": compact_table_prompt_block(table),
            "table_json": json.dumps(table),
        },
    ]
    ctx = _format_context(chunks)
    assert "Indicated for moderate" in ctx
    assert "Structured table" in ctx or "headers:" in ctx
    assert ">300-400" in ctx
    appendix = _table_json_appendix(chunks[1])
    assert "merged_cells" in appendix


def test_boost_prefers_table_row_for_table_modality():
    candidates = [
        {"id": "a", "text": "narrative dosing discussion", "chunk_type": "text", "score": 1.0},
        {
            "id": "b",
            "text": "TABLE ROW assay 350 IU/mL 300 mg",
            "chunk_type": "table_row",
            "index_kind": "table_row",
            "score": 1.0,
            "table_json": "{}",
        },
    ]
    q = "What dose for 350 IU/mL and 78 kg every 2 weeks?"
    boosted = _boost_table_candidates(q, candidates, modality="table")
    by_id = {c["id"]: c for c in boosted}
    assert by_id["b"]["score"] > by_id["a"]["score"]


def test_make_table_id_stable():
    a = make_table_id(page=3, ordinal=0, bbox=[1, 2, 3, 4], headers=["A", "B"])
    b = make_table_id(page=3, ordinal=0, bbox=[1, 2, 3, 4], headers=["A", "B"])
    c = make_table_id(page=3, ordinal=1, bbox=[1, 2, 3, 4], headers=["A", "B"])
    assert a == b
    assert a != c


def test_parse_table_payload_roundtrip():
    table = _dosing_table()
    raw = json.dumps(table)
    parsed = parse_table_payload(raw)
    assert parsed["headers"] == table["headers"]
    assert parse_table_payload(None) is None


# ── SQL store (sqlite) ───────────────────────────────────────────────────────


def test_persist_and_load_document_tables(tmp_path, monkeypatch):
    # Use an isolated sqlite file so we do not touch the app test bootstrap DB.
    db_path = tmp_path / "table_rag.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app import models  # noqa: F401
    from app.rag.table_rag.store import (
        get_document_table,
        list_document_tables,
        persist_document_tables,
    )
    from app.rag.table_rag.hydrate import hydrate_retrieved_tables

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Minimal parent document row for FK.
    user_id = "11111111-1111-1111-1111-111111111111"
    doc_id = "22222222-2222-2222-2222-222222222222"
    db.add(
        models.User(
            id=user_id,
            username="table_rag_user",
            email="table-rag@example.com",
            hashed_password="x",
        )
    )
    db.add(
        models.Document(
            id=doc_id,
            user_id=user_id,
            filename="x.pdf",
            original_name="xolair_prescribing.pdf",
        )
    )
    db.commit()

    table = _dosing_table()
    records = [
        {
            "table_id": table["table_id"],
            "table": table,
            "page_start": 3,
            "page_end": 3,
            "section": "Dosage",
            "summary_text": table_summary_text(table),
            "ordinal": 0,
        }
    ]
    n = persist_document_tables(
        document_id=doc_id,
        user_id=user_id,
        filename="xolair_prescribing.pdf",
        tables=records,
        db=db,
    )
    assert n == 1
    listed = list_document_tables(document_id=doc_id, user_id=user_id, db=db)
    assert len(listed) == 1
    loaded = get_document_table(
        document_id=doc_id, table_id=table["table_id"], user_id=user_id, db=db
    )
    assert loaded is not None
    assert loaded["table"]["merged_cells"]
    assert len(loaded["table"]["rows"]) == 3

    # Hydrate path: monkeypatch loader is unnecessary when using same DB — but
    # hydrate opens SessionLocal from app.database. Patch load_tables_by_ids.
    from app.rag.table_rag import hydrate as hydrate_mod

    def _fake_load(*, user_id, pairs, db=None):
        return {f"{doc_id}:{table['table_id']}": loaded}

    monkeypatch.setattr(hydrate_mod, "load_tables_by_ids", _fake_load)
    hydrated = hydrate_retrieved_tables(
        [
            {
                "document_id": doc_id,
                "table_id": table["table_id"],
                "chunk_type": "table_row",
                "row_index": 1,
                "text": "row",
                "page": 3,
            }
        ],
        user_id=user_id,
    )
    assert hydrated[0]["table"]["headers"][0].startswith("Assay")
    assert "table_prompt_block" in hydrated[0]
    assert hydrated[0]["relevant_row_indices"] == [1]
    db.close()

"""Hardening tests: quality gate, provenance, chunk isolation, SQL confidence."""
from __future__ import annotations

from typing import Any, List

import pytest

from app.rag.chunking.semantic.generator import generate_chunks
from app.rag.chunking.semantic.types import SectionNode, SourceBlock
from app.rag.generation.table_calculator import compute_from_operands
from app.rag.ingestion.pdf_layout.docling_tables import (
    _caption_from_docling_table,
    _cell_text,
    _estimate_structure_prior,
)
from app.rag.ingestion.pdf_layout.extraction_quality import (
    evaluate_table_quality,
    filter_quality_tables,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData
from app.rag.table_rag.routing import QueryModality, route_query
from app.rag.table_rag.store import persist_document_tables


def _td(**kwargs) -> TableData:
    base = dict(
        page=1,
        bbox=BBox(0, 0, 100, 100),
        headers=["A", "B"],
        rows=[["1", "2"], ["3", "4"]],
        extraction_method="docling",
        extraction_engine="docling",
        structure_confidence=0.9,
    )
    base.update(kwargs)
    return TableData(**base)


def test_docling_caption_ignores_uncalled_caption_text_method():
    """Bound caption_text methods must not be stringified into chunk text."""

    class FakeTable:
        caption = None
        title = None

        def caption_text(self, doc=None):
            return "Invoice line items"

    caption = _caption_from_docling_table(FakeTable(), document=object())
    assert caption == "Invoice line items"
    assert _cell_text(FakeTable().caption_text) == ""
    # Uncalled method attribute must not dump the method repr.
    assert "<bound method" not in caption


def test_quality_hard_fails_majority_synthetic_headers():
    table = _td(
        headers=["Column 1", "Column 2", "Column 3", "Real"],
        rows=[["a", "b", "c", "d"], ["e", "f", "g", "h"]],
        structure_confidence=0.95,
    )
    result = evaluate_table_quality(table)
    # 3/4 synthetic → weak_headers hard fail
    assert "weak_headers" in result.reasons
    assert not result.ok


def test_quality_accepts_real_headers():
    result = evaluate_table_quality(
        _td(headers=["Year", "Revenue ($M)"], rows=[["2024", "12.4"]])
    )
    assert result.ok


def test_filter_quality_splits_accepted_rejected():
    good = _td(headers=["Material", "MPa"], rows=[["A36", "250"]])
    bad = _td(
        headers=["Column 1", "Column 2"],
        rows=[["x", "y"]],
        structure_confidence=0.2,
    )
    accepted, rejected, reports = filter_quality_tables([good, bad])
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert reports[0]["engine"] == "docling"
    assert rejected[0].provenance.get("quality")


def test_semantic_generator_isolates_table_from_prose():
    prose = SourceBlock(
        block_id="p1",
        text="Introductory paragraph about the schedule.",
        page_start=1,
        page_end=1,
        order=0,
        element_type="paragraph",
        content_type="paragraph",
        extraction_method="native",
    )
    table = SourceBlock(
        block_id="t1",
        text="| A | B |\n| --- | --- |\n| 1 | 2 |",
        page_start=1,
        page_end=1,
        order=1,
        element_type="table",
        content_type="table",
        extraction_method="native",
        table={
            "table_id": "t_p1_i0_abc",
            "headers": ["A", "B"],
            "rows": [["1", "2"]],
            "extraction_method": "docling",
            "extraction_engine": "docling",
            "page": 1,
            "pages": [1],
        },
    )
    node = SectionNode(
        section_id="s1",
        title="Section",
        level=1,
        path=["Section"],
        block_ids=["p1", "t1"],
        parent_section_id=None,
        ordinal=0,
    )
    chunks = generate_chunks(
        [node],
        [prose, table],
        document_id="doc1",
        document_name="demo.pdf",
        max_tokens=1800,
    )
    assert len(chunks) >= 2
    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
    text_chunks = [c for c in chunks if c.get("chunk_type") == "text"]
    assert table_chunks
    assert text_chunks
    # Table chunk must not swallow the prose paragraph.
    assert "Introductory paragraph" not in table_chunks[0]["text"]
    assert table_chunks[0]["extraction_method"] == "docling"


def test_persist_empty_tables_clears_document(monkeypatch):
    deleted = {"n": 0}
    added: List[Any] = []

    class FakeQuery:
        def filter(self, *_a, **_k):
            return self

        def delete(self, synchronize_session=False):
            deleted["n"] += 1
            return 1

    class FakeDB:
        def query(self, *_a, **_k):
            return FakeQuery()

        def add(self, row):
            added.append(row)

        def commit(self):
            return None

        def rollback(self):
            return None

        def close(self):
            return None

    n = persist_document_tables(
        document_id="doc-clear",
        user_id="u1",
        filename="x.pdf",
        tables=[],
        db=FakeDB(),
    )
    assert n == 0
    assert deleted["n"] == 1
    assert added == []


def test_persist_maps_structure_confidence(monkeypatch):
    added: List[Any] = []

    class FakeQuery:
        def filter(self, *_a, **_k):
            return self

        def delete(self, synchronize_session=False):
            return 0

    class FakeDB:
        def query(self, *_a, **_k):
            return FakeQuery()

        def add(self, row):
            added.append(row)

        def commit(self):
            return None

        def rollback(self):
            return None

        def close(self):
            return None

    n = persist_document_tables(
        document_id="doc-conf",
        user_id="u1",
        filename="x.pdf",
        tables=[
            {
                "table_id": "t1",
                "table": {
                    "table_id": "t1",
                    "headers": ["A", "B"],
                    "rows": [["1", "2"]],
                    "page": 1,
                    "pages": [1],
                    "extraction_engine": "docling",
                    "structure_confidence": 0.82,
                    "ocr_confidence": 0.0,
                },
            }
        ],
        db=FakeDB(),
    )
    assert n == 1
    assert abs(float(added[0].confidence) - 0.82) < 1e-6
    assert added[0].extraction_method == "docling"


def test_quality_hard_fails_empty_body():
    result = evaluate_table_quality(
        _td(headers=["A", "B"], rows=[], structure_confidence=0.9)
    )
    assert not result.ok
    assert "too_few_rows" in result.reasons


def test_docling_failure_propagates_for_fallback(monkeypatch):
    """Missing/failed Docling must raise so parser can trigger Textract."""
    from app.rag.ingestion.pdf_layout import docling_tables as dt

    def _boom():
        raise RuntimeError("docling is required for primary table extraction")

    monkeypatch.setattr(dt, "get_docling_converter", _boom)
    with pytest.raises(RuntimeError, match="docling is required"):
        dt.extract_tables_from_pdf("dummy.pdf")


def test_docling_page_batches_cover_all_pages(monkeypatch):
    """Large PDFs should be converted in page_range batches."""
    from app.rag.ingestion.pdf_layout import docling_tables as dt

    calls = []

    class FakeConverter:
        def convert(self, filepath, raises_on_error=False, page_range=None, **_k):
            calls.append(page_range)
            class Doc:
                tables = []
            class Result:
                document = Doc()
            return Result()

    monkeypatch.setattr(dt, "get_docling_converter", lambda: FakeConverter())
    monkeypatch.setattr(dt, "_pdf_page_count", lambda *_a, **_k: 10)
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.docling_tables.get_settings",
        lambda: type("S", (), {"PDF_DOCLING_TABLES_ENABLED": True, "PDF_DOCLING_PAGE_BATCH_SIZE": 4})(),
    )
    tables = dt.extract_tables_from_pdf("dummy.pdf", page_sizes={i: (612, 792) for i in range(1, 11)})
    assert tables == []
    assert calls == [(1, 4), (5, 8), (9, 10)]


def test_structure_prior_penalizes_synthetic_headers():
    weak = _estimate_structure_prior(
        headers=["Column 1", "Column 2"],
        rows=[["a", "b"]],
        merged=[],
    )
    strong = _estimate_structure_prior(
        headers=["Year", "Revenue"],
        rows=[["2024", "12.4"]],
        merged=[],
    )
    assert strong > weak
    assert weak < 0.55


def test_calculator_max_min_and_rejects_unsupported():
    table = {
        "table_id": "t_fin",
        "headers": ["Item", "Amount"],
        "rows": [["A", "10"], ["B", "40"], ["C", "25"]],
        "pages": [1],
    }
    mx = compute_from_operands(
        table, [(0, 1), (1, 1), (2, 1)], operation="max", question="maximum amount"
    )
    assert mx.value == 40
    mn = compute_from_operands(
        table, [(0, 1), (1, 1), (2, 1)], operation="min", question="minimum amount"
    )
    assert mn.value == 10
    with pytest.raises(ValueError, match="unsupported"):
        compute_from_operands(
            table, [(0, 1), (1, 1)], operation="median", question="median"
        )


def test_route_max_question_prefers_table():
    route = route_query("What is the maximum yield strength in the table?")
    assert route.modality in {QueryModality.TABLE, QueryModality.BOTH}


def test_expand_table_rag_avoids_duplicate_index_chunks():
    from app.rag.table_rag.indexing import expand_table_rag_chunks

    source = {
        "id": "c0",
        "text": "| Year | Rev |\n| --- | --- |\n| 2024 | 10 |",
        "chunk_type": "table",
        "content_type": "table",
        "page": 1,
        "page_start": 1,
        "page_end": 1,
        "section": "Financials",
        "table_json": {
            "table_id": "t_fin_1",
            "headers": ["Year", "Rev"],
            "rows": [["2024", "10"], ["2025", "12"]],
            "page": 1,
            "pages": [1],
            "extraction_engine": "docling",
            "structure_confidence": 0.8,
        },
    }
    chunks, records = expand_table_rag_chunks(
        [source],
        document_id="doc1",
        filename="fin.pdf",
        enabled=True,
        max_rows_per_table=200,
    )
    assert len(records) == 1
    cards = [c for c in chunks if c.get("chunk_type") == "table_card"]
    assert len(cards) == 1
    # Running expand twice on already-expanded list must not duplicate cards.
    again, records2 = expand_table_rag_chunks(
        chunks,
        document_id="doc1",
        filename="fin.pdf",
        enabled=True,
    )
    assert len(records2) == 1
    assert len([c for c in again if c.get("chunk_type") == "table_card"]) == 1

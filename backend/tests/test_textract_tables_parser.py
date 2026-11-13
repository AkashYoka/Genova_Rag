"""Parser integration tests for the Textract-only table extraction path."""
from __future__ import annotations

import json
import types

import pymupdf as fitz
import pytest

from app.rag.ingestion.pdf_layout import parser
from app.rag.ingestion.pdf_layout.types import BBox, TableData


def _make_text_pdf(path, text: str = "This is a normal paragraph for layout parsing. " * 4):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _fake_table() -> TableData:
    return TableData(
        page=1,
        bbox=BBox(60.0, 200.0, 300.0, 260.0),
        headers=["Metric", "Value"],
        rows=[["Alpha", "10"]],
        extraction_method="textract",
        caption="Table 1. Demo",
        header_rows=[["Metric", "Value"]],
        cells=[
            {
                "row_index": 0,
                "col_index": 0,
                "text": "Metric",
                "confidence": 99.0,
                "bbox": [60.0, 200.0, 180.0, 230.0],
                "row_span": 1,
                "col_span": 1,
                "is_header": True,
            },
            {
                "row_index": 0,
                "col_index": 1,
                "text": "Value",
                "confidence": 99.0,
                "bbox": [180.0, 200.0, 300.0, 230.0],
                "row_span": 1,
                "col_span": 1,
                "is_header": True,
            },
        ],
        merged_cells=[],
    )


@pytest.fixture(autouse=True)
def _textract_settings(monkeypatch):
    monkeypatch.setattr(
        parser,
        "get_settings",
        lambda: types.SimpleNamespace(
            PDF_LAYOUT_WRITE_DIAGNOSTICS=False,
            PDF_TEXTRACT_TABLES_ENABLED=True,
            PDF_TEXTRACT_TABULAR_ONLY=False,
            PDF_TEXTRACT_REGION="us-east-1",
            BEDROCK_REGION="us-east-1",
            PDF_OCR_ENABLED=True,
            PDF_OCR_DPI=150,
            PDF_CHUNK_SIZE=800,
            PDF_CHUNK_OVERLAP=100,
            PDF_TABLE_MAX_CHARS=4000,
            PDF_TABLE_SPLIT_ROWS=40,
        ),
    )


def test_parse_pdf_uses_textract_without_legacy_table_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "native.pdf"
    _make_text_pdf(pdf_path)

    calls = {"textract": 0}

    def fake_extract(page, page_number):
        calls["textract"] += 1
        return [_fake_table()]

    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.extract_tables_from_page",
        fake_extract,
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.usable_textract_tables",
        lambda tables: list(tables),
    )
    monkeypatch.setattr(parser, "_repair_page_tables", lambda tables, filepath="": tables)
    monkeypatch.setattr(parser, "write_diagnostics", lambda _parsed: {})

    parsed = parser.parse_pdf(str(pdf_path), document_name="native.pdf")

    report = parsed.report["table_extraction"]
    assert report["engine"] == "textract"
    assert report["table_count"] == 1
    assert report["failure_count"] == 0
    assert calls["textract"] == 1
    assert parsed.pages[0].metrics["table_source"] == "textract"
    assert parsed.pages[0].metrics["table_engine"] == "textract"
    assert parsed.pages[0].metrics["textract_invoked"] is True

    table_chunks = [
        chunk for chunk in parsed.chunks if chunk.get("chunk_type") == "table"
    ]
    assert table_chunks
    payload = json.loads(table_chunks[0]["table_json"])
    assert payload["extraction_method"] == "textract"
    assert payload["headers"] == ["Metric", "Value"]
    assert payload["rows"] == [["Alpha", "10"]]
    assert "merged_cells" in payload


def test_parse_pdf_reports_textract_failure_without_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "native.pdf"
    _make_text_pdf(pdf_path)

    def boom(page, page_number):
        raise RuntimeError("synthetic textract failure")

    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.extract_tables_from_page",
        boom,
    )
    monkeypatch.setattr(parser, "_repair_page_tables", lambda tables, filepath="": tables)
    monkeypatch.setattr(parser, "write_diagnostics", lambda _parsed: {})

    # Even if native tables exist as tabular hints, they must not become stored tables.
    monkeypatch.setattr(
        parser,
        "_native_tables",
        lambda page: [_fake_table()],
    )

    parsed = parser.parse_pdf(str(pdf_path), document_name="native.pdf")

    report = parsed.report["table_extraction"]
    assert report["table_count"] == 0
    assert report["failure_count"] == 1
    assert "synthetic textract failure" in report["failures"][0]["error"]
    assert parsed.pages[0].tables == []
    assert parsed.pages[0].metrics["table_source"] == "none"
    assert parsed.pages[0].metrics.get("textract_error")
    assert not [
        chunk for chunk in parsed.chunks if chunk.get("chunk_type") == "table"
    ]


def test_table_engine_is_always_textract():
    assert parser._table_engine() == "textract"

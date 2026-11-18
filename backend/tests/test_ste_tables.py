"""Unit tests for STE Textract AnalyzeDocument+Bedrock tables and parse_pdf flag."""
from __future__ import annotations

import json
import types
from typing import Any, Dict, List

import pymupdf as fitz
import pytest

from app.rag.ingestion.pdf_layout import parser
from app.rag.ingestion.pdf_layout.ste_tables import (
    EXTRACTION_ENGINE,
    merge_tables_prefer_ste,
    table_content_fingerprint,
    universal_dict_to_table_data,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData


def _make_text_pdf(path, text: str = "Paragraph for layout. " * 6):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _universal_sample() -> Dict[str, Any]:
    return {
        "document_id": "doc-1",
        "page_number": 2,
        "table_id": "table_001",
        "table_title": "Claims overview",
        "original_headers": ["Slide", "Claim"],
        "reconstructed_headers": ["Slide", "Claim in the deck"],
        "original_grid": [
            ["Slide", "Claim"],
            ["2", "Alpha works"],
            ["3", "Beta improves"],
        ],
        "independent_rows": [
            {
                "row_id": "table_001_row_1",
                "row_index": 0,
                "cells": {
                    "slide": {
                        "header": "Slide",
                        "original_value": "2",
                        "normalized_value": "2",
                        "unit": None,
                        "status": "present",
                        "source_coordinates": [10, 20, 40, 40],
                        "confidence": 0.95,
                        "reconstruction_method": "source_cell",
                    },
                    "claim_in_the_deck": {
                        "header": "Claim in the deck",
                        "original_value": "Alpha works",
                        "normalized_value": "Alpha works",
                        "unit": None,
                        "status": "present",
                        "source_coordinates": [40, 20, 200, 40],
                        "confidence": 0.95,
                        "reconstruction_method": "source_cell",
                    },
                },
                "source": {
                    "page": 2,
                    "pages": [2],
                    "bbox": [60.0, 180.0, 400.0, 320.0],
                    "caption": "Claims overview",
                },
            },
            {
                "row_id": "table_001_row_2",
                "row_index": 1,
                "cells": {
                    "slide": {
                        "header": "Slide",
                        "original_value": "3",
                        "normalized_value": "3",
                        "status": "present",
                        "confidence": 0.9,
                        "reconstruction_method": "source_cell",
                    },
                    "claim_in_the_deck": {
                        "header": "Claim in the deck",
                        "original_value": "Beta improves",
                        "normalized_value": "Beta improves",
                        "status": "present",
                        "confidence": 0.9,
                        "reconstruction_method": "source_cell",
                    },
                },
                "source": {
                    "page": 2,
                    "pages": [2],
                    "bbox": [60.0, 180.0, 400.0, 320.0],
                    "caption": "Claims overview",
                },
            },
        ],
        "unavailable_regions": [],
        "footnotes": [],
        "extraction_confidence": 0.95,
        "reconstruction_warnings": [],
        "source_provenance": {
            "extraction_method": "textract_analyze_tables",
            "bbox": [60.0, 180.0, 400.0, 320.0],
            "pages": [2],
            "backend": "textract",
            "bedrock_applied": True,
            "fully_cellular": True,
        },
    }


def test_universal_dict_to_table_data_preserves_cells_and_audit():
    td = universal_dict_to_table_data(_universal_sample())
    assert td.extraction_engine == EXTRACTION_ENGINE
    assert td.page == 2
    assert td.pages == [2]
    assert td.headers == ["Slide", "Claim in the deck"]
    assert td.rows == [["2", "Alpha works"], ["3", "Beta improves"]]
    assert td.caption == "Claims overview"
    assert td.bbox.as_tuple() == (60.0, 180.0, 400.0, 320.0)
    assert len(td.independent_rows) == 2
    assert td.provenance["reconstructed_headers"] == [
        "Slide",
        "Claim in the deck",
    ]
    assert td.provenance["original_extraction"]["original_headers"] == [
        "Slide",
        "Claim",
    ]
    assert td.provenance["original_extraction"]["original_grid"][1] == [
        "2",
        "Alpha works",
    ]
    payload = td.to_json()
    assert payload["extraction_engine"] == EXTRACTION_ENGINE
    assert payload["independent_rows"][0]["row_id"] == "table_001_row_1"
    # Never invent values — empty/null statuses stay empty strings in grid.
    assert "Insufficient Data" not in json.dumps(payload["rows"])


def test_merge_tables_prefer_ste_dedupes_by_iou():
    ste = TableData(
        page=1,
        bbox=BBox(50, 100, 300, 250),
        headers=["A", "B"],
        rows=[["1", "2"]],
        extraction_engine=EXTRACTION_ENGINE,
        extraction_method="textract_analyze_tables",
    )
    textract = TableData(
        page=1,
        bbox=BBox(55, 105, 295, 245),  # high IoU overlap
        headers=["A", "B"],
        rows=[["1", "x"]],  # different body — STE still wins on IoU
        extraction_engine="textract_detect",
        extraction_method="textract_detect",
    )
    other = TableData(
        page=1,
        bbox=BBox(400, 100, 550, 200),
        headers=["C"],
        rows=[["z"]],
        extraction_engine="textract_detect",
        extraction_method="textract_detect",
    )
    merged = merge_tables_prefer_ste([textract, other], [ste], iou_threshold=0.55)
    assert len(merged) == 2
    assert merged[0].extraction_engine == EXTRACTION_ENGINE
    assert merged[0].rows == [["1", "2"]]
    assert merged[1].headers == ["C"]


def test_merge_tables_prefer_ste_dedupes_by_fingerprint():
    ste = TableData(
        page=3,
        bbox=BBox(0, 0, 0, 0),
        headers=["Metric", "Value"],
        rows=[["Alpha", "10"]],
        extraction_engine=EXTRACTION_ENGINE,
    )
    textract = TableData(
        page=3,
        bbox=BBox(10, 10, 20, 20),
        headers=["Metric", "Value"],
        rows=[["Alpha", "10"]],
        extraction_engine="textract_detect",
    )
    assert table_content_fingerprint(ste) == table_content_fingerprint(textract)
    merged = merge_tables_prefer_ste([textract], [ste])
    assert len(merged) == 1
    assert merged[0].extraction_engine == EXTRACTION_ENGINE


def _settings(**overrides):
    base = dict(
        PDF_LAYOUT_WRITE_DIAGNOSTICS=False,
        PDF_DOCLING_TABLES_ENABLED=False,
        PDF_DOCLING_TABLEFORMER_MODE="accurate",
        PDF_STE_TABLES_ENABLED=False,
        PDF_STE_BEDROCK_ENABLED=True,
        PDF_STE_BEDROCK_MODEL_ID="us.amazon.nova-2-lite-v1:0",
        PDF_STE_BACKEND="textract",
        PDF_STE_TABLEFORMER_MODE="accurate",
        PDF_STE_OUTPUT_DIR="./data/ste_runs",
        PDF_STE_USE_CACHE=True,
        PDF_STE_AWS_REGION="",
        PDF_STE_IOU_DEDUPE=0.55,
        PDF_TEXTRACT_FALLBACK_ENABLED=True,
        PDF_TEXTRACT_TABLES_ENABLED=True,
        PDF_TEXTRACT_TABULAR_ONLY=False,
        PDF_TEXTRACT_ASYNC_MULTIPAGE=False,
        PDF_TEXTRACT_REGION="us-east-1",
        BEDROCK_REGION="us-east-1",
        BEDROCK_MODEL_ID="us.amazon.nova-2-lite-v1:0",
        PDF_OCR_ENABLED=True,
        PDF_OCR_DPI=150,
        PDF_CHUNK_SIZE=800,
        PDF_CHUNK_OVERLAP=100,
        PDF_TABLE_MAX_CHARS=4000,
        PDF_TABLE_SPLIT_ROWS=40,
        PDF_TABLE_QUALITY_MIN_COLS=2,
        PDF_TABLE_QUALITY_MIN_ROWS=1,
        PDF_TABLE_QUALITY_MAX_EMPTY_CELL_RATIO=0.65,
        PDF_TABLE_QUALITY_MIN_STRUCTURE_CONFIDENCE=0.35,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _fake_textract_table() -> TableData:
    return TableData(
        page=1,
        bbox=BBox(60.0, 200.0, 300.0, 260.0),
        headers=["Metric", "Value"],
        rows=[["Alpha", "10"]],
        extraction_method="textract_detect",
        extraction_engine="textract_detect",
        structure_confidence=0.9,
        ocr_confidence=99.0,
    )


def _fake_ste_table() -> TableData:
    return TableData(
        page=1,
        bbox=BBox(58.0, 198.0, 302.0, 262.0),
        headers=["Metric", "Value"],
        rows=[["Alpha", "10"]],
        extraction_method="textract_analyze_tables",
        extraction_engine=EXTRACTION_ENGINE,
        caption="STE table",
        structure_confidence=0.95,
        ocr_confidence=95.0,
        independent_rows=[
            {
                "row_id": "t1_row_1",
                "row_index": 0,
                "cells": {
                    "metric": {
                        "header": "Metric",
                        "normalized_value": "Alpha",
                        "original_value": "Alpha",
                        "status": "present",
                    },
                    "value": {
                        "header": "Value",
                        "normalized_value": "10",
                        "original_value": "10",
                        "status": "present",
                    },
                },
                "source": {"page": 1, "pages": [1], "bbox": [58, 198, 302, 262]},
            }
        ],
        provenance={
            "independent_rows": [],
            "reconstructed_headers": ["Metric", "Value"],
            "original_extraction": {
                "original_headers": ["Metric", "Value"],
                "original_grid": [["Metric", "Value"], ["Alpha", "10"]],
            },
            "bedrock": {
                "applied": True,
                "model_id": "us.amazon.nova-2-lite-v1:0",
                "total_tokens": 1200,
            },
        },
    )


def test_parse_pdf_flag_off_does_not_call_ste(monkeypatch, tmp_path):
    pdf_path = tmp_path / "off.pdf"
    _make_text_pdf(pdf_path)
    calls = {"ste": 0}

    monkeypatch.setattr(parser, "get_settings", lambda: _settings(PDF_STE_TABLES_ENABLED=False))

    def boom_ste(*_a, **_k):
        calls["ste"] += 1
        raise AssertionError("STE must not run when flag is off")

    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.ste_tables.extract_ste_tables_from_pdf",
        boom_ste,
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.extract_tables_from_pdf_detect",
        lambda *a, **k: ([_fake_textract_table()], {"api": "DetectDocumentText", "pages_processed": 1}),
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.usable_textract_tables",
        lambda tables: list(tables),
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.get_textract_api_usage",
        lambda: {"detect_document_text_pages": 1},
    )
    monkeypatch.setattr(parser, "_repair_page_tables", lambda tables, filepath="": list(tables))
    monkeypatch.setattr(parser, "write_diagnostics", lambda _p: {})

    parsed = parser.parse_pdf(str(pdf_path), document_name="off.pdf")
    assert calls["ste"] == 0
    assert parsed.report["table_extraction"]["engine"] == "textract_detect"
    assert parsed.pages[0].tables
    assert parsed.pages[0].tables[0].extraction_engine == "textract_detect"


def test_parse_pdf_flag_on_prefers_ste_and_dedupes(monkeypatch, tmp_path):
    pdf_path = tmp_path / "on.pdf"
    _make_text_pdf(pdf_path)
    calls = {"ste": 0}

    monkeypatch.setattr(parser, "get_settings", lambda: _settings(PDF_STE_TABLES_ENABLED=True))

    def fake_ste(*_a, **_k):
        calls["ste"] += 1
        return [_fake_ste_table()], {
            "ok": True,
            "extraction_engine": EXTRACTION_ENGINE,
            "table_count": 1,
            "bedrock_usage": {
                "input_tokens": 800,
                "output_tokens": 400,
                "total_tokens": 1200,
                "estimated_cost_usd": 0.001,
                "model_id": "us.amazon.nova-2-lite-v1:0",
            },
            "elapsed_s": 1.2,
        }

    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.ste_tables.extract_ste_tables_from_pdf",
        fake_ste,
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.extract_tables_from_pdf_detect",
        lambda *a, **k: ([_fake_textract_table()], {"api": "DetectDocumentText", "pages_processed": 1}),
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.usable_textract_tables",
        lambda tables: list(tables),
    )
    monkeypatch.setattr(
        "app.rag.ingestion.pdf_layout.textract_tables.get_textract_api_usage",
        lambda: {"detect_document_text_pages": 1},
    )
    monkeypatch.setattr(parser, "_repair_page_tables", lambda tables, filepath="": list(tables))
    monkeypatch.setattr(parser, "write_diagnostics", lambda _p: {})

    parsed = parser.parse_pdf(str(pdf_path), document_name="on.pdf")
    assert calls["ste"] == 1
    assert parsed.report["table_extraction"]["engine"] == EXTRACTION_ENGINE
    engine_info = parsed.report["table_extraction"]["engine_info"]
    assert engine_info["ste_tables_enabled"] is True
    assert engine_info["ste"]["bedrock_usage"]["total_tokens"] == 1200
    assert len(parsed.pages[0].tables) == 1
    assert parsed.pages[0].tables[0].extraction_engine == EXTRACTION_ENGINE
    assert parsed.pages[0].metrics["table_source"] == EXTRACTION_ENGINE
    table_chunks = [c for c in parsed.chunks if c.get("chunk_type") == "table"]
    assert table_chunks
    payload = json.loads(table_chunks[0]["table_json"])
    assert payload["extraction_engine"] == EXTRACTION_ENGINE
    assert payload["caption"] == "STE table"


def test_ste_independent_rows_do_not_fan_out_windows(monkeypatch):
    """STE independent_rows must not create per-row table_row_window chunks."""
    from app.rag.table_rag.indexing import (
        CHUNK_TYPE_TABLE_CARD,
        CHUNK_TYPE_TABLE_ROW_WINDOW,
        expand_table_rag_chunks,
    )

    table = _fake_ste_table()
    table.provenance["independent_rows"] = table.independent_rows
    chunk = {
        "text": table.markdown,
        "page": 1,
        "page_start": 1,
        "page_end": 1,
        "chunk_type": "table",
        "content_type": "table",
        "extraction_method": EXTRACTION_ENGINE,
        "table_json": json.dumps(table.to_json()),
        "filename": "demo.pdf",
    }
    monkeypatch.setattr(
        "app.rag.table_rag.indexing.get_settings",
        lambda: types.SimpleNamespace(
            TABLE_RAG_ROW_WINDOW_THRESHOLD=10,
            TABLE_RAG_ROW_WINDOW_SIZE=8,
            TABLE_RAG_MAX_ROW_WINDOWS=40,
        ),
    )
    expanded, records = expand_table_rag_chunks(
        [chunk],
        document_id="d1",
        filename="demo.pdf",
        index_version="v2_card_window",
        enabled=True,
    )
    assert len(records) == 1
    cards = [c for c in expanded if c.get("chunk_type") == CHUNK_TYPE_TABLE_CARD]
    windows = [
        c for c in expanded if c.get("chunk_type") == CHUNK_TYPE_TABLE_ROW_WINDOW
    ]
    assert len(cards) == 1
    assert windows == []

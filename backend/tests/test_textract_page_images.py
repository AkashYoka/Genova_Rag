"""DetectDocumentText must receive page images only — never the PDF."""
from __future__ import annotations

import types
from pathlib import Path

import pymupdf as fitz
import pytest

from app.rag.ingestion.pdf_layout import textract_tables as tt
from app.rag.ingestion.pdf_layout.types import BBox, TableData, TextWord


def _tiny_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello table world " * 8)
    doc.save(str(path))
    doc.close()


@pytest.fixture(autouse=True)
def _settings(monkeypatch, tmp_path_factory):
    report_dir = tmp_path_factory.mktemp("layout_reports")
    monkeypatch.setattr(
        tt,
        "get_settings",
        lambda: types.SimpleNamespace(
            PDF_TEXTRACT_FALLBACK_ENABLED=True,
            PDF_TEXTRACT_TABLES_ENABLED=True,
            PDF_TEXTRACT_DPI=100,
            PDF_OCR_DPI=100,
            PDF_TEXTRACT_MIN_CONFIDENCE=50.0,
            PDF_TEXTRACT_MAX_WORKERS=2,
            PDF_TEXTRACT_TIMEOUT_SECONDS=30,
            PDF_TEXTRACT_REGION="us-east-1",
            BEDROCK_REGION="us-east-1",
            PDF_TEXTRACT_ASYNC_MULTIPAGE=True,  # must be ignored
            PDF_TEXTRACT_SAVE_PAGE_IMAGES=False,
            PDF_TABLE_LLM_NORMALIZE_ENABLED=False,
            PDF_LAYOUT_REPORT_DIR=str(report_dir),
        ),
    )


def test_extract_never_calls_async_pdf_api(monkeypatch, tmp_path):
    pdf_path = tmp_path / "demo.pdf"
    _tiny_pdf(pdf_path)
    doc = fitz.open(str(pdf_path))
    page_sizes = {1: (float(doc[0].rect.width), float(doc[0].rect.height))}

    calls = {"detect_bytes": 0, "payloads": []}

    def fake_detect(image_bytes: bytes):
        calls["detect_bytes"] += 1
        calls["payloads"].append(image_bytes[:8])
        # Minimal WORD block so reconstruction has something to chew on.
        return {
            "Blocks": [
                {
                    "BlockType": "WORD",
                    "Text": "Hello",
                    "Confidence": 99.0,
                    "Geometry": {
                        "BoundingBox": {"Left": 0.1, "Top": 0.1, "Width": 0.1, "Height": 0.05}
                    },
                }
            ]
        }

    monkeypatch.setattr(tt, "_detect_document_text_bytes", fake_detect)
    monkeypatch.setattr(
        tt,
        "table_data_from_textract_blocks",
        lambda *a, **k: [
            TableData(
                page=1,
                bbox=BBox(10, 10, 100, 80),
                headers=["A", "B"],
                rows=[["1", "2"]],
                extraction_method="textract_detect",
                extraction_engine="textract_detect",
            )
        ],
    )
    monkeypatch.setattr(
        tt,
        "words_from_detect_blocks",
        lambda *a, **k: [
            TextWord(
                text="Hello",
                bbox=BBox(10, 10, 40, 20),
                page=1,
                confidence=99.0,
            )
        ],
    )

    assert not hasattr(tt, "extract_tables_from_pdf_async")
    tt.reset_textract_api_usage()

    tables, meta = tt.extract_tables_from_pdf_detect(
        str(pdf_path), page_sizes=page_sizes, doc=doc
    )
    doc.close()

    assert meta["api"] == "DetectDocumentText"
    assert meta["mode"] == "page_images"
    assert meta["pdf_uploaded_to_textract"] is False
    assert calls["detect_bytes"] == 1
    assert tables
    assert tables[0].provenance.get("input") == "page_image"
    usage = tt.get_textract_api_usage()
    assert usage["start_document_text_detection_jobs"] == 0
    assert usage["pages_via_async"] == 0
    assert usage["analyze_document_tables_pages"] == 0
    # PNG magic or JPEG SOI — never PDF "%PDF"
    head = calls["payloads"][0]
    assert head[:4] != b"%PDF"
    assert head[:4] == b"\x89PNG" or head[:2] == b"\xff\xd8"


def test_render_page_image_never_returns_pdf_bytes(tmp_path):
    pdf_path = tmp_path / "demo.pdf"
    _tiny_pdf(pdf_path)
    doc = fitz.open(str(pdf_path))
    image_bytes, fmt, img_w, img_h, used_dpi = tt._render_page_image_bytes(
        doc[0], dpi=100
    )
    doc.close()
    assert fmt in {"png", "jpeg"}
    assert image_bytes[:4] != b"%PDF"
    assert len(image_bytes) > 100
    assert img_w > 0 and img_h > 0
    assert used_dpi >= 72


def test_words_pack_polygon_px():
    blocks = [
        {
            "BlockType": "WORD",
            "Text": "Dose",
            "Confidence": 99.0,
            "Geometry": {
                "BoundingBox": {
                    "Left": 0.1,
                    "Top": 0.2,
                    "Width": 0.1,
                    "Height": 0.05,
                },
                "Polygon": [
                    {"X": 0.1, "Y": 0.2},
                    {"X": 0.2, "Y": 0.2},
                    {"X": 0.2, "Y": 0.25},
                    {"X": 0.1, "Y": 0.25},
                ],
            },
        }
    ]
    words = tt.words_from_detect_blocks(
        blocks,
        page_number=1,
        page_width=612.0,
        page_height=792.0,
        image_width=1000,
        image_height=2000,
    )
    assert len(words) == 1
    assert words[0].polygon_px == [100, 400, 200, 400, 200, 500, 100, 500]


def test_page_ocr_layout_from_blocks():
    blocks = [
        {
            "BlockType": "WORD",
            "Text": "A",
            "Confidence": 98.0,
            "Geometry": {
                "BoundingBox": {
                    "Left": 0.0,
                    "Top": 0.0,
                    "Width": 0.1,
                    "Height": 0.1,
                },
                "Polygon": [
                    {"X": 0.0, "Y": 0.0},
                    {"X": 0.1, "Y": 0.0},
                    {"X": 0.1, "Y": 0.1},
                    {"X": 0.0, "Y": 0.1},
                ],
            },
        }
    ]
    layout = tt.page_ocr_layout_from_blocks(
        blocks,
        page_number=3,
        page_width=612.0,
        page_height=792.0,
        image_width=800,
        image_height=1100,
        image_format="png",
        dpi=300,
    )
    payload = layout.to_dict()
    assert payload["page_index"] == 3
    assert payload["image_width"] == 800
    assert payload["image_height"] == 1100
    assert payload["dpi"] == 300
    assert len(payload["words"]) == 1
    assert payload["words"][0]["polygon_px"] == [0, 0, 80, 0, 80, 110, 0, 110]


def test_save_page_images_when_enabled(monkeypatch, tmp_path):
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(
        tt,
        "get_settings",
        lambda: types.SimpleNamespace(
            PDF_TEXTRACT_FALLBACK_ENABLED=True,
            PDF_TEXTRACT_TABLES_ENABLED=True,
            PDF_TEXTRACT_DPI=100,
            PDF_OCR_DPI=100,
            PDF_TEXTRACT_MIN_CONFIDENCE=50.0,
            PDF_TEXTRACT_MAX_WORKERS=1,
            PDF_TEXTRACT_TIMEOUT_SECONDS=30,
            PDF_TEXTRACT_REGION="us-east-1",
            BEDROCK_REGION="us-east-1",
            PDF_TEXTRACT_ASYNC_MULTIPAGE=False,
            PDF_TEXTRACT_SAVE_PAGE_IMAGES=True,
            PDF_TABLE_LLM_NORMALIZE_ENABLED=False,
            PDF_LAYOUT_REPORT_DIR=str(report_dir),
        ),
    )
    pdf_path = tmp_path / "demo.pdf"
    _tiny_pdf(pdf_path)
    doc = fitz.open(str(pdf_path))
    page_sizes = {1: (float(doc[0].rect.width), float(doc[0].rect.height))}

    monkeypatch.setattr(
        tt,
        "_detect_document_text_bytes",
        lambda _b: {
            "Blocks": [
                {
                    "BlockType": "WORD",
                    "Text": "Hello",
                    "Confidence": 99.0,
                    "Geometry": {
                        "BoundingBox": {
                            "Left": 0.1,
                            "Top": 0.1,
                            "Width": 0.1,
                            "Height": 0.05,
                        },
                        "Polygon": [
                            {"X": 0.1, "Y": 0.1},
                            {"X": 0.2, "Y": 0.1},
                            {"X": 0.2, "Y": 0.15},
                            {"X": 0.1, "Y": 0.15},
                        ],
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        tt,
        "table_data_from_textract_blocks",
        lambda *a, **k: [],
    )

    _tables, meta = tt.extract_tables_from_pdf_detect(
        str(pdf_path), page_sizes=page_sizes, doc=doc
    )
    doc.close()

    assert meta["workers"] <= 10
    assert meta["saved_page_images"]
    saved = Path(meta["saved_page_images"][0])
    assert saved.exists()
    assert saved.name.startswith("page-0001.")
    assert 1 in meta["page_layouts"]
    assert meta["page_layouts"][1]["words"]

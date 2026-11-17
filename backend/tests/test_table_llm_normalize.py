"""Domain-agnostic LLM table normalize (mocked client)."""
from __future__ import annotations

import json
import types

from app.rag.ingestion.pdf_layout.table_llm_normalize import (
    TABLE_NORMALIZE_SYSTEM_PROMPT,
    normalize_table_with_llm,
)
from app.rag.ingestion.pdf_layout.types import BBox, TableData


def test_system_prompt_has_cellular_contract():
    assert "Insufficient Data" in TABLE_NORMALIZE_SYSTEM_PROMPT
    assert "Zero merged cells" in TABLE_NORMALIZE_SYSTEM_PROMPT
    assert "No invented" in TABLE_NORMALIZE_SYSTEM_PROMPT


def test_normalize_rewrites_column_n_headers(monkeypatch):
    table = TableData(
        page=1,
        bbox=BBox(0, 0, 100, 100),
        headers=["IgE", "Freq", "Column 3", "Column 4"],
        rows=[[">100-200", "Every 4 Weeks", "75", "150"]],
        header_rows=[["IgE", "Freq", "≥10-12 >12-15", ""]],
        caption="Example",
        extraction_method="textract_detect",
    )

    class _Client:
        def chat_completion(self, **_kwargs):
            payload = {
                "headers": [
                    "Pretreatment Serum IgE (IU/mL)",
                    "Dosing Frequency",
                    "Body Weight ≥10-12 kg Dose (mg)",
                    "Body Weight >12-15 kg Dose (mg)",
                ],
                "header_rows": [
                    [
                        "Pretreatment Serum IgE (IU/mL)",
                        "Dosing Frequency",
                        "Body Weight ≥10-12 kg Dose (mg)",
                        "Body Weight >12-15 kg Dose (mg)",
                    ]
                ],
                "rows": [[">100-200", "Every 4 Weeks", "75", "150"]],
            }
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(payload))
                    )
                ]
            )

    settings = types.SimpleNamespace(
        PDF_TABLE_LLM_NORMALIZE_ENABLED=True,
        PDF_TABLE_LLM_NORMALIZE_MAX_TOKENS=2048,
        SEMANTIC_CHUNKING_MODEL_ID="test-model",
        BEDROCK_MODEL_ID="test-model",
        LLM_MODEL="test-model",
    )
    out, stats = normalize_table_with_llm(
        table, client=_Client(), settings=settings
    )
    assert stats["used_llm"] is True
    assert not any(h.lower().startswith("column ") for h in out.headers)
    assert out.rows[0][2] == "75"
    assert out.provenance.get("llm_normalized") is True

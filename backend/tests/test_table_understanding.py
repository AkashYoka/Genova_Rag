"""LLM table understanding: schema validation, immutability, enrichment."""
from __future__ import annotations

import json
from types import SimpleNamespace

from app.rag.chunking.semantic.types import SourceBlock
from app.rag.table_rag.indexing import expand_table_rag_chunks
from app.rag.table_rag.representations import table_row_text, table_summary_text
from app.rag.table_rag.schema import TableUnderstanding
from app.rag.table_rag.understanding import (
    apply_understanding_to_table,
    deterministic_understanding,
    understand_document_tables,
    validate_understanding_against_table,
)


def _table() -> dict:
    return {
        "table_id": "t_demo",
        "page": 5,
        "caption": "Table 1. Dosing schedule",
        "headers": ["Assay (IU/mL)", "Schedule", "30-60 kg", ">70-90 kg"],
        "header_rows": [["Assay (IU/mL)", "Schedule", "30-60 kg", ">70-90 kg"]],
        "rows": [
            [">200-300", "Every 4 weeks", "150", "300"],
            [">300-400", "Every 2 weeks", "225", "300"],
        ],
        "merged_cells": [
            {
                "row_index": 0,
                "col_index": 2,
                "row_span": 1,
                "col_span": 2,
                "text": "Body weight",
            }
        ],
        "cells": [{"row_index": 1, "col_index": 3, "text": "300"}],
        "bbox": [10, 20, 30, 40],
        "n_rows": 3,
        "n_cols": 4,
        "extraction_method": "textract",
        "linearized_facts": ["Assay >300-400 / >70-90 kg = 300"],
    }


def test_apply_understanding_never_mutates_extracted_cells():
    table = _table()
    original = json.loads(json.dumps(table))
    understanding = TableUnderstanding(
        table_id="t_demo",
        description="Asthma dosing matrix",
        purpose="Lookup dose by assay and weight",
        subject="dosing",
        column_meanings=[],
        semantic_search_phrases=["dosing table asthma IgE weight"],
        confidence=0.9,
    )
    enriched = apply_understanding_to_table(table, understanding)
    assert enriched["headers"] == original["headers"]
    assert enriched["rows"] == original["rows"]
    assert enriched["merged_cells"] == original["merged_cells"]
    assert enriched["cells"] == original["cells"]
    assert enriched["bbox"] == original["bbox"]
    assert "llm_understanding" in enriched
    assert enriched["llm_understanding"]["description"] == "Asthma dosing matrix"


def test_validation_flags_unknown_headers_and_low_conf():
    table = _table()
    table["low_confidence_cells"] = [{"row": 0, "col": 3, "text": "300", "confidence": 0.1}]
    understanding = TableUnderstanding(
        table_id="t_demo",
        description="x",
        column_meanings=[{"header": "NotARealHeader", "meaning": "bogus"}],
        lookup_attributes=[
            {"name": "AlsoFake", "role": "row_axis", "header_match": "AlsoFake"}
        ],
        confidence=0.2,
    )
    validated = validate_understanding_against_table(understanding, table)
    assert validated.validated is True
    assert validated.uncertain is True
    assert any("NotARealHeader" in w for w in validated.validation_warnings)
    assert any("low-confidence" in w for w in validated.validation_warnings)


def test_summary_and_row_include_llm_phrases_without_changing_values():
    from app.rag.table_rag.representations import table_interpretation_text

    table = _table()
    understanding = TableUnderstanding(
        table_id="t_demo",
        description="Pretreatment assay by body-weight dosing grid",
        purpose="Find recommended dose",
        subject="Xolair dosing",
        semantic_search_phrases=["pretreatment serum assay dosing table"],
        lookup_attributes=[
            {
                "name": "Assay (IU/mL)",
                "role": "row_axis",
                "header_match": "Assay (IU/mL)",
            }
        ],
        confidence=0.8,
    )
    table = apply_understanding_to_table(table, understanding)
    # Separate embeddings: summary stays structural; interpretation is its own text.
    summary = table_summary_text(table, section="Dosage", include_llm=False)
    assert "TABLE SUMMARY" in summary
    assert ">300-400" in summary  # extracted fact/row still present
    assert "LLM TABLE INTERPRETATION" not in summary

    interp = table_interpretation_text(table, section="Dosage")
    assert "TABLE INTERPRETATION" in interp
    assert "LLM TABLE INTERPRETATION" in interp
    assert "not cell ground truth" in interp.lower()
    assert "Pretreatment assay" in interp

    # Legacy combined mode still available when flag is off.
    combined = table_summary_text(table, section="Dosage", include_llm=True)
    assert "LLM TABLE INTERPRETATION" in combined

    row_text = table_row_text(table, 1, table["rows"][1])
    assert "225" in row_text
    assert "300" in row_text
    assert "semantic_cues" in row_text


def test_understand_document_tables_uses_mocked_llm():
    table = _table()
    block = SourceBlock(
        block_id="b00003",
        order=3,
        element_type="table",
        text="table markdown",
        page_start=5,
        page_end=5,
        section="Dosage",
        subsection="Adults",
        table=table,
    )
    context = SourceBlock(
        block_id="b00002",
        order=2,
        element_type="paragraph",
        text="Determine dosage using serum assay and body weight.",
        page_start=5,
        page_end=5,
        section="Dosage",
    )
    node = SimpleNamespace(
        block_ids=["b00002", "b00003"],
        path=["Dosage and Administration", "Adults"],
        children=[],
    )

    class _FakeClient:
        def chat_completion(self, **kwargs):
            payload = {
                "tables": [
                    {
                        "table_id": "t_demo",
                        "block_id": "b00003",
                        "description": "Matrix of recommended doses by assay band and weight.",
                        "purpose": "Select subcutaneous dose and frequency",
                        "subject": "asthma dosing",
                        "column_meanings": [
                            {
                                "header": "Assay (IU/mL)",
                                "meaning": "Pretreatment assay band",
                                "unit": "IU/mL",
                                "is_axis": True,
                            },
                            {
                                "header": ">70-90 kg",
                                "meaning": "Dose for weight band",
                                "unit": "mg",
                                "is_lookup_value": True,
                            },
                        ],
                        "hierarchical_headers": [
                            {
                                "level": 1,
                                "text": "Body weight",
                                "spans_columns": ["30-60 kg", ">70-90 kg"],
                                "interpretation": "Weight-band dose columns",
                            }
                        ],
                        "row_column_relationships": [
                            {
                                "description": "Assay band row intersects weight column for dose",
                                "related_headers": ["Assay (IU/mL)", ">70-90 kg"],
                            }
                        ],
                        "units_and_measurements": ["IU/mL", "mg", "weeks"],
                        "footnotes": [],
                        "semantic_search_phrases": [
                            "recommended dose by assay and body weight"
                        ],
                        "suggested_qa_patterns": [
                            {
                                "question_pattern": "What dose for assay X and weight Y?",
                                "answer_strategy": "row/col intersection",
                                "required_axes": ["assay", "weight"],
                            }
                        ],
                        "lookup_attributes": [
                            {
                                "name": "assay",
                                "role": "row_axis",
                                "header_match": "Assay (IU/mL)",
                                "example_values": [">300-400"],
                            }
                        ],
                        "numerical_dimensions": ["body weight kg", "assay IU/mL", "dose mg"],
                        "confidence": 0.88,
                        "uncertain": False,
                        "uncertainty_notes": [],
                    }
                ]
            }
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload))
                    )
                ]
            )

    blocks, report = understand_document_tables(
        [context, block],
        [node],
        document_name="xolair_prescribing.pdf",
        client=_FakeClient(),
        settings=SimpleNamespace(
            TABLE_RAG_ENABLED=True,
            TABLE_RAG_LLM_UNDERSTANDING_ENABLED=True,
            TABLE_RAG_LLM_BATCH_SIZE=2,
            TABLE_RAG_LLM_MAX_TABLES=10,
            TABLE_RAG_LLM_PROMPT_ROWS=8,
            TABLE_RAG_LLM_MAX_OUTPUT_TOKENS=1024,
            SEMANTIC_CHUNKING_MODEL_ID="test-model",
            BEDROCK_MODEL_ID="test-model",
            LLM_MODEL="test-model",
        ),
        enabled=True,
    )
    assert report["understood"] == 1
    assert report["llm_calls"] == 1
    enriched = blocks[1].table
    assert enriched["rows"][1][3] == "300"  # original cell intact
    assert enriched["llm_understanding"]["purpose"].startswith("Select subcutaneous")
    assert "Body weight" in json.dumps(enriched["llm_understanding"])


def test_expand_index_includes_llm_interpretation_text():
    table = _table()
    table = apply_understanding_to_table(
        table,
        deterministic_understanding(table, section_path="Dosage"),
    )
    chunk = {
        "text": "table",
        "page": 5,
        "chunk_index": 0,
        "chunk_type": "table",
        "section": "Dosage",
        "table_id": "t_demo",
        "table_json": json.dumps(table),
    }
    expanded, records = expand_table_rag_chunks(
        [chunk], document_id="doc-1", filename="x.pdf"
    )
    summaries = [c for c in expanded if c["chunk_type"] == "table_summary"]
    interps = [c for c in expanded if c["chunk_type"] == "table_interpretation"]
    rows = [c for c in expanded if c["chunk_type"] == "table_row"]
    assert summaries
    assert "LLM TABLE INTERPRETATION" not in summaries[0]["text"]
    assert interps
    assert "LLM TABLE INTERPRETATION" in interps[0]["text"]
    assert "index_kind" in interps[0] and interps[0]["index_kind"] == "table_interpretation"
    assert rows
    assert records[0]["table"]["rows"] == table["rows"]

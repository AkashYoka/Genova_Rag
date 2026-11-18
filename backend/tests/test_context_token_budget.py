"""Regression: slim table prompts + context budget keep dosing RAG lean."""
from __future__ import annotations

from app.rag.llm.agent import _format_context
from app.rag.table_rag.representations import compact_table_prompt_block


def _xolair_like_table():
    headers = [
        "Pretreatment Serum IgE (IU/mL)",
        "Dosing Freq.",
        "Body Weight / 30-60 kg / Dose (mg)",
        "Body Weight / >60-70 kg / Dose (mg)",
        "Body Weight / >70-90 kg / Dose (mg)",
        "Body Weight / >90-150 kg / Dose (mg)",
    ]
    rows = [
        [">=30-100", "Every 4 weeks", "150", "150", "150", "300"],
        [">100-200", "Every 4 weeks", "300", "300", "300", "225"],
        [">200-300", "Every 4 weeks", "300", "225", "225", "300"],
        [">300-400", "Every 2 weeks", "225", "225", "300", "Insufficient Data"],
        [">400-500", "Every 2 weeks", "300", "300", "375", "Insufficient Data"],
        [">500-600", "Every 2 weeks", "300", "375", "Insufficient Data", "Insufficient Data"],
    ]
    return {
        "table_id": "t1",
        "page": 5,
        "caption": "Table 1. Subcutaneous XOLAIR Doses for Patients 12+ with Asthma",
        "headers": headers,
        "rows": rows,
    }


def test_slim_prompt_much_smaller_than_verbose():
    table = _xolair_like_table()
    verbose = compact_table_prompt_block(table, max_rows=20, slim=False, include_understanding=False)
    slim = compact_table_prompt_block(table, max_rows=8, slim=True, include_understanding=False)
    assert "LOOKUP RULE" in slim
    assert "[r3,c4|" not in slim
    assert "| r3 |" in slim or "r3" in slim
    assert len(slim) < len(verbose)
    assert len(slim) < len(verbose) * 0.55


def test_format_context_dedupes_table_id_and_skips_duplicate_text(monkeypatch):
    monkeypatch.setattr(
        "app.rag.llm.agent.settings.TABLE_RAG_SKIP_CHUNK_TEXT_WHEN_HYDRATED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "app.rag.llm.agent.settings.RAG_CONTEXT_MAX_CHARS",
        5000,
        raising=False,
    )
    block = compact_table_prompt_block(
        _xolair_like_table(),
        max_rows=4,
        relevant_row_indices=[3],
        relevant_col_indices=[0, 1, 4],
        slim=True,
    )
    chunks = [
        {
            "filename": "xolair.pdf",
            "page": 5,
            "section": "Dosing Information",
            "chunk_type": "table_row_window",
            "table_id": "t1",
            "text": "HUGE " * 2000 + "TABLE ROW WINDOW duplicate grid 225 300 150",
            "table_prompt_block": block,
        },
        {
            "filename": "xolair.pdf",
            "page": 5,
            "section": "Dosing Information",
            "chunk_type": "table_card",
            "table_id": "t1",
            "text": "second copy of same table should be skipped",
            "table_prompt_block": block + "\nEXTRA",
        },
        {
            "filename": "xolair.pdf",
            "page": 4,
            "section": "DOSAGE AND ADMINISTRATION",
            "chunk_type": "text",
            "text": "Overview of dosage determination.",
        },
    ]
    ctx = _format_context(chunks)
    assert ctx.count("Structured table") == 1
    assert "second copy" not in ctx
    assert "HUGE HUGE" not in ctx
    assert "Overview of dosage determination" in ctx
    assert len(ctx) < 2500

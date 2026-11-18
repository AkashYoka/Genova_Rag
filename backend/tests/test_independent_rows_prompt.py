"""Independent row facts for answer prompts (assessment-style)."""
from __future__ import annotations

from app.rag.table_rag.independent_rows import (
    independent_row_facts,
    independent_rows_prompt_block,
)


def test_independent_row_facts_bind_all_headers():
    table = {
        "caption": "Rate card",
        "headers": ["Tier", "Cadence", "USD"],
        "rows": [
            ["Starter", "Monthly", "9"],
            ["Growth", "Annual", "99"],
        ],
    }
    facts = independent_row_facts(table)
    assert facts[0] == "row_fact[r0]: Tier=Starter | Cadence=Monthly | USD=9"
    assert "Growth" in facts[1] and "Annual" in facts[1]


def test_prompt_block_instructs_single_fact_match():
    table = {
        "caption": "Table 1 dosing",
        "headers": ["IgE", "Freq", "70-90 kg"],
        "rows": [
            [">200-300", "Every 4 weeks", "225"],
            [">300-400", "Every 2 weeks", "300"],
        ],
    }
    block = independent_rows_prompt_block(table, include_grid=False)
    assert "INDEPENDENT ROW FACTS" in block
    assert "row_fact[r1]:" in block
    assert "Every 2 weeks" in block
    assert "300" in block
    assert "do not mix values across different row_fact lines" in block.lower()

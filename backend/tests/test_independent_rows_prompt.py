"""Independent cell facts for answer prompts (assessment-style)."""
from __future__ import annotations

from app.rag.table_rag.independent_rows import (
    independent_cell_facts,
    independent_row_facts,
    independent_rows_prompt_block,
)


def test_narrow_table_emits_one_fact_per_row():
    table = {
        "caption": "Rate card",
        "headers": ["Tier", "Cadence", "USD"],
        "rows": [
            ["Starter", "Monthly", "9"],
            ["Growth", "Annual", "99"],
        ],
    }
    facts = independent_cell_facts(table)
    assert len(facts) == 2
    assert facts[0] == "cell_fact[r0]: Tier=Starter | Cadence=Monthly | USD=9"
    assert "Growth" in facts[1] and "Annual" in facts[1]
    # Back-compat alias
    assert independent_row_facts(table) == facts


def test_parallel_columns_emit_one_fact_per_value_cell():
    table = {
        "caption": "Matrix",
        "headers": [
            "Band",
            "Freq",
            "Body Weight / 30-60 kg / Dose (mg)",
            "Body Weight / >70-90 kg / Dose (mg)",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "225", "225"],
            [">300-400", "Every 2 weeks", "225", "300"],
        ],
    }
    facts = independent_cell_facts(table)
    # 2 rows × 2 parallel weight columns
    assert len(facts) == 4
    target = [
        f
        for f in facts
        if "r1" in f and ">70-90" in f and "=300" in f and "Every 2 weeks" in f
    ]
    assert len(target) == 1
    # Must not jam the sibling 225 weight band into the same 300 fact
    assert "30-60" not in target[0]


def test_prompt_block_instructs_single_cell_fact_match():
    table = {
        "caption": "Table 1 dosing",
        "headers": ["IgE", "Freq", "70-90 kg", ">90-150 kg"],
        "rows": [
            [">200-300", "Every 4 weeks", "225", "300"],
            [">300-400", "Every 2 weeks", "300", "Insufficient Data"],
        ],
    }
    block = independent_rows_prompt_block(table, include_grid=False)
    assert "INDEPENDENT CELL FACTS" in block
    assert "cell_fact[r1," in block
    assert "Every 2 weeks" in block
    assert "do not mix values across different cell_fact lines" in block.lower()


def test_query_prunes_to_covering_bands():
    table = {
        "caption": "Table 1 matrix",
        "headers": [
            "Band",
            "Freq",
            "Body Weight / 30-60 kg / Dose (mg)",
            "Body Weight / >70-90 kg / Dose (mg)",
            "Body Weight / >90-150 kg / Dose (mg)",
        ],
        "rows": [
            [">200-300", "Every 4 weeks", "225", "225", "300"],
            [">300-400", "Every 2 weeks", "225", "300", "Insufficient Data"],
            [">400-500", "Every 2 weeks", "300", "375", "Insufficient Data"],
        ],
    }
    facts = independent_cell_facts(
        table, query="Table 1 values for 350 and 78"
    )
    # Only the covering IgE row + covering weight column should remain.
    assert len(facts) == 1
    assert ">300-400" in facts[0]
    assert ">70-90" in facts[0]
    assert "=300" in facts[0]
    assert "30-60" not in facts[0]
    assert ">400-500" not in facts[0]

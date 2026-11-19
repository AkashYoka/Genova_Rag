"""Query-near prune: joint row×col coverage with disjoint query numbers."""
from __future__ import annotations

from app.rag.table_rag.independent_rows import (
    independent_cell_facts,
    select_query_near_indices,
)


def _crswnp_like_table():
    # Minimal Table-3-shaped matrix: IgE rows × weight columns.
    headers = [
        "Pretreatment Serum IgE (IU/mL)",
        "Dosing Freq.",
        "Body Weight / >40-50 kg / Dose (mg)",
        "Body Weight / >50-60 kg / Dose (mg)",
        "Body Weight / >125-150 kg / Dose (mg)",
    ]
    rows = [
        ["30-100", "Every 4 Weeks", "150", "150", "300"],
        [">100-200", "Every 4 Weeks", "300", "300", "600"],
        [">200-300", "Every 4 Weeks", "300", "300", "600"],
    ]
    return {"caption": "Table 3. Adult CRSwNP", "headers": headers, "rows": rows}


def test_joint_prune_igE_150_weight_55_keeps_300_not_150():
    table = _crswnp_like_table()
    q = (
        "An adult with CRSwNP has pretreatment IgE of 150 IU/mL and weighs 55 kg. "
        "What is the labeled subcutaneous XOLAIR regimen?"
    )
    rows, cols = select_query_near_indices(table, q)
    assert rows == [1]  # >100-200, not 30-100
    assert cols == [3]  # >50-60 kg, not >125-150 kg

    facts = independent_cell_facts(table, query=q)
    blob = "\n".join(facts)
    assert "300" in blob
    assert "Dose (mg)=150" not in blob.replace(" ", "")
    # Explicit: the winning fact is 300 on the matching band.
    assert any(">100-200" in f and ">50-60" in f and "300" in f for f in facts)
    assert not any("30-100" in f for f in facts)

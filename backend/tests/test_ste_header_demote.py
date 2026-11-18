"""False Textract header bands must demote back to body rows."""
from __future__ import annotations

from ste.models import BBox
from ste.reconstructor import reconstruct_from_grid


def test_demote_first_data_row_marked_as_header():
    grid = [
        [
            "Country",
            "Model",
            "Who builds and submits",
            "What the buyer does",
            "What it means for Genova",
        ],
        [
            "UAE",
            "Peppol 5-corner, PINT-AE",
            "Supplier, through its provider. Both providers report to the FTA",
            "Receives through its own provider",
            "AR compliance, plus structured AP intake",
        ],
        [
            "Saudi Arabia",
            "Real-time clearance (Fatoora), UBL 2.1",
            "Seller must get ZATCA clearance before delivering the invoice",
            "Receives the cleared invoice",
            "AR only, and the ERP must hold delivery until clearance",
        ],
    ]
    table = reconstruct_from_grid(
        grid,
        table_id="t_demo",
        page=3,
        bbox=BBox(0, 0, 100, 100),
        header_row_count=2,  # Textract false positive
        confidence=0.95,
    )
    assert table is not None
    assert table.headers[:2] == ["Country", "Model"]
    assert not any(" / " in h for h in table.headers)
    assert table.rows[0][0] == "UAE"
    assert table.rows[1][0] == "Saudi Arabia"
    assert len(table.rows) == 2


def test_demote_slide_claim_table_false_header():
    grid = [
        ["Slide", "Claim in the deck", "Status", "Note"],
        [
            "1, 9",
            "Genova does not submit to the tax authority; the gateway does",
            "Holds",
            "UAE taxpayers must use an accredited provider and cannot connect to the FTA directly (Taxually)",
        ],
        [
            "2, 7",
            "$9.90 vs $2.67 per invoice; 9.2 vs 3.1 days",
            "Holds as a benchmark",
            "The deck already labels these as self-reported survey averages.",
        ],
    ]
    table = reconstruct_from_grid(
        grid,
        table_id="t_slide",
        page=2,
        bbox=BBox(0, 0, 100, 100),
        header_row_count=2,
        confidence=0.97,
    )
    assert table is not None
    assert table.headers == ["Slide", "Claim in the deck", "Status", "Note"]
    assert table.rows[0][0] == "1, 9"
    assert len(table.rows) == 2

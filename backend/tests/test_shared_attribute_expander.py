"""Shared categorical fill — structural mid-span vertical merges."""
from __future__ import annotations

from ste.grid_postprocess import stitch_wrapped_label_column
from ste.shared_attribute_expander import (
    body_row_positions_from_cells_meta,
    expand_shared_attributes,
)


def test_fill_propagates_lower_label_into_trailing_empties():
    rows = [
        ["tier-a", "Monthly", "9", "8"],
        ["tier-b", "Weekly", "19", "18"],
        ["tier-c", "Weekly", "29", "28"],
        ["tier-d", "", "39", "38"],
        ["tier-e", "", "49", "48"],
    ]
    warnings = expand_shared_attributes(rows)
    assert any("shared_attr_col_1_filled" in w for w in warnings)
    assert rows[3][1] == "Weekly"
    assert rows[4][1] == "Weekly"
    assert rows[0][1] == "Monthly"


def test_fill_equidistant_empty_prefers_earlier_block():
    rows = [
        ["a", "Monthly", "9"],
        ["b", "", "19"],
        ["c", "Annual", "99"],
        ["d", "", "199"],
    ]
    expand_shared_attributes(rows)
    assert rows[1][1] == "Monthly"
    assert rows[3][1] == "Annual"


def test_centered_label_blocks_split_gap_at_centroids():
    """Mid-span identical labels must not forward-fill into the next span.

    Two label blocks sit in the middle of their visual merges; blank rows
    above/between/below inherit the nearest block by centroid distance.
    """
    rows = [
        ["r0", ""],
        ["r1", ""],
        ["r2", "Region North"],
        ["r3", "Region North"],
        ["r4", ""],
        ["r5", ""],
        ["r6", ""],
        ["r7", ""],
        ["r8", "Region South"],
        ["r9", "Region South"],
        ["r10", ""],
        ["r11", ""],
    ]
    expand_shared_attributes(rows)

    north = {r[0] for r in rows if r[1] == "Region North"}
    south = {r[0] for r in rows if r[1] == "Region South"}
    assert north == {"r0", "r1", "r2", "r3", "r4", "r5"}
    assert south == {"r6", "r7", "r8", "r9", "r10", "r11"}


def test_wrapped_numeric_phrase_then_nearest_block_fill():
    """Wrap-stitch + block fill: fragments mid-span, blanks assigned structurally."""
    rows = [
        ["r0", ""],
        ["r1", ""],
        ["r2", "Every"],
        ["r3", "4"],
        ["r4", "weeks"],
        ["r5", ""],
        ["r6", ""],
        ["r7", ""],
        ["r8", ""],
        ["r9", "Every"],
        ["r10", "2"],
        ["r11", "weeks"],
        ["r12", ""],
        ["r13", ""],
    ]
    stitch_wrapped_label_column(rows, col=1)
    expand_shared_attributes(rows)

    four = {r[0] for r in rows if r[1] == "Every 4 weeks"}
    two = {r[0] for r in rows if r[1] == "Every 2 weeks"}
    assert four == {"r0", "r1", "r2", "r3", "r4", "r5", "r6"}
    assert two == {"r7", "r8", "r9", "r10", "r11", "r12", "r13"}


def test_geometry_row_positions_override_index_when_provided():
    """When mid-Y positions skew a label block, geometry wins over row index."""
    rows = [
        ["r0", ""],
        ["r1", "North"],
        ["r2", ""],
        ["r3", ""],
        ["r4", "South"],
        ["r5", ""],
    ]
    positions = [0.0, 10.0, 20.0, 55.0, 70.0, 80.0]
    expand_shared_attributes(rows, row_positions=positions)
    assert [r[1] for r in rows] == [
        "North",
        "North",
        "North",
        "South",
        "South",
        "South",
    ]


def test_body_row_positions_from_cells_meta_averages_mid_y():
    meta = [
        {"row": 2, "col": 0, "bbox": [0, 100, 10, 120]},
        {"row": 2, "col": 1, "bbox": [10, 100, 20, 120]},
        {"row": 3, "col": 0, "bbox": [0, 130, 10, 150]},
    ]
    pos = body_row_positions_from_cells_meta(meta, header_row_count=2, body_rows=2)
    assert pos is not None
    assert pos[0] == 110.0
    assert pos[1] == 140.0

"""Flat CSV with all header columns present on every row."""
from __future__ import annotations

import csv
from pathlib import Path

from ste.models import ExtractedTable


def write_csv(path: Path, table: ExtractedTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(len(table.headers or []), max((len(r) for r in table.rows or []), default=0), 1)
    headers = list(table.headers or []) + [f"Column {i+1}" for i in range(width)]
    headers = headers[:width]
    # Add status companion columns for insufficient_data visibility
    fieldnames = []
    for h in headers:
        fieldnames.append(h)
        fieldnames.append(f"{h}__status")

    cell_status = {(c.row, c.col): c.status for c in (table.cells or [])}

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ri, row in enumerate(table.rows or []):
            cells = list(row) + [""] * width
            cells = cells[:width]
            record = {}
            for ci, h in enumerate(headers):
                record[h] = cells[ci]
                record[f"{h}__status"] = cell_status.get((ri, ci), "present" if cells[ci] else "empty")
            writer.writerow(record)

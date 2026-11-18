"""Recover row labels that an engine merged into a multi-line neighbour cell.

Docling/TableFormer sometimes concatenates two visual label lines into the
upper row's cell and leaves the following row's label cell empty (the label
bounding box spans both rows). That destroys row independence: the lower row
loses its identity and shared-attribute backfill would copy the wrong label.

This module uses the PDF text layer (digital PDFs only) to split the merged
cell back into its visual lines. It never invents text: the concatenation of
the recovered lines must exactly match the engine's cell tokens, and recovery
only happens when the following row has values but no label.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from ste.backends.base import RawTable
from ste.models import normalize_ws

# bbox -> [(visual line text, line bbox in PDF points)]
LineProvider = Callable[
    [Sequence[float]], Sequence[Tuple[str, Sequence[float]]]
]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(normalize_ws(text).lower())


def split_merged_labels(
    table: RawTable,
    line_provider: LineProvider,
    *,
    page_size: Optional[Tuple[float, float]] = None,
    max_splits: int = 3,
) -> tuple[int, List[str]]:
    """Split merged multi-line labels using native PDF lines.

    Args:
        table: Raw table (grid + cells_meta) mutated in place.
        line_provider: ``bbox -> [(line text, line bbox)]`` from the PDF text
            layer. Fewer than two lines (e.g. a scanned page) disables
            recovery for that cell.
        page_size: Optional page width/height used to compute normalized cell
            bboxes for shading analysis.
        max_splits: Maximum labels recoverable from one merged cell.

    Returns:
        ``(recovered_cell_count, warnings)``.
    """
    warnings: List[str] = []
    recovered = 0
    if not table.grid or not table.cells_meta:
        return recovered, warnings

    snapshot = [
        dict(meta) for meta in table.cells_meta if int(meta.get("col", -1)) == 0
    ]
    existing = {
        (int(m.get("row", -1)), int(m.get("col", -1))): m
        for m in table.cells_meta
    }

    for meta in sorted(snapshot, key=lambda m: int(m.get("row", -1))):
        row = int(meta.get("row", -1))
        if row < 0 or row >= len(table.grid):
            continue
        if int(meta.get("row_span", 1)) > 1 or int(meta.get("col_span", 1)) > 1:
            continue
        cell_text = normalize_ws(table.grid[row][0])
        bbox = meta.get("bbox")
        if not cell_text or not bbox or len(bbox) < 4:
            continue

        raw_lines = list(line_provider(bbox) or [])
        lines: List[str] = []
        line_boxes: List[List[float]] = []
        for item in raw_lines:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            text = normalize_ws(str(item[0]))
            box = item[1]
            if not text or not box or len(box) < 4:
                continue
            lines.append(text)
            line_boxes.append([float(v) for v in box[:4]])
        if len(lines) < 2 or len(lines) > 1 + max_splits:
            continue
        if _tokens(" ".join(lines)) != _tokens(cell_text):
            continue

        targets: List[int] = []
        for offset in range(1, len(lines)):
            target = row + offset
            if target >= len(table.grid):
                break
            if normalize_ws(table.grid[target][0]):
                break
            if not any(
                normalize_ws(value)
                for index, value in enumerate(table.grid[target])
                if index != 0
            ):
                break
            target_meta = existing.get((target, 0))
            if target_meta and normalize_ws(str(target_meta.get("text") or "")):
                break
            targets.append(target)
        if len(targets) < len(lines) - 1:
            continue

        for offset, target in enumerate(targets, start=1):
            line = lines[offset]
            table.grid[target][0] = line
            clone = dict(meta)
            clone.update(
                {
                    "row": target,
                    "col": 0,
                    "origin_row": target,
                    "origin_col": 0,
                    "row_span": 1,
                    "col_span": 1,
                    "text": line,
                    "column_header": False,
                    "row_header": True,
                    "recovered_from_merged_label": True,
                    "bbox": list(line_boxes[offset]),
                    "bbox_scope": "native_text_line",
                }
            )
            if page_size and page_size[0] > 0 and page_size[1] > 0:
                x0, y0, x1, y1 = line_boxes[offset]
                clone["bbox_norm"] = [
                    x0 / page_size[0],
                    y0 / page_size[1],
                    (x1 - x0) / page_size[0],
                    (y1 - y0) / page_size[1],
                ]
            table.cells_meta.append(clone)
        table.grid[row][0] = lines[0]
        meta["text"] = lines[0]
        meta["bbox"] = list(line_boxes[0])
        meta["bbox_scope"] = "native_text_line"
        recovered += len(targets)
        warnings.append(
            f"merged_label_split:p{table.page}:row{row}:+{len(targets)}"
        )

    return recovered, warnings


__all__ = ["split_merged_labels"]

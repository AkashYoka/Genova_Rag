"""Repair Docling cells that merge adjacent column text or split merged values.

Two failure modes are handled using the PDF text layer (digital PDFs only):

1. A header cell contains labels from two visual columns and the neighbouring
   header cell is empty (e.g. ``"Freq. 20-25 kg"`` in the body-weight column,
   leaving the dosing-frequency column unnamed).
2. A vertically merged value is fragmented across rows/columns (e.g.
   ``"Every 2"`` in the frequency column, with ``"weeks"`` misassigned to a
   dose column).

Every change must be token-exact with native PDF lines: text is reassembled
or moved, never invented. Scanned pages without a text layer are untouched.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ste.backends.base import RawTable
from ste.models import normalize_ws

LineProvider = Callable[
    [Sequence[float]], Sequence[Tuple[str, Sequence[float]]]
]
Box = Tuple[float, float, float, float]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(normalize_ws(text).lower())


def _is_submultiset(small: Sequence[str], big: Sequence[str]) -> bool:
    if not small:
        return False
    return not (Counter(small) - Counter(big))


def _bbox(meta: Dict[str, object]) -> Optional[Box]:
    box = meta.get("bbox")
    if not box or len(box) < 4:
        return None
    try:
        values = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return (values[0], values[1], values[2], values[3])


def _center_x(box: Box) -> float:
    return (box[0] + box[2]) / 2.0


def _center_y(box: Box) -> float:
    return (box[1] + box[3]) / 2.0


def _union(boxes: Sequence[Box]) -> List[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _clean_lines(
    line_provider: LineProvider, bbox: Sequence[float]
) -> List[Tuple[str, Box]]:
    out: List[Tuple[str, Box]] = []
    try:
        raw = line_provider(bbox) or []
    except Exception:
        return out
    for item in raw:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        text = normalize_ws(str(item[0]))
        box = item[1]
        if not text or not box or len(box) < 4:
            continue
        try:
            b = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        except (TypeError, ValueError):
            continue
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        out.append((text, b))
    return out


def column_ranges_from_cells(
    cells_meta: Sequence[Dict[str, object]], header_row_count: int
) -> Dict[int, Box]:
    """Median x-span per column from single-column body cells."""
    per_col: Dict[int, List[Box]] = defaultdict(list)
    for meta in cells_meta:
        try:
            row = int(meta.get("row", -1))
            col = int(meta.get("col", -1))
            col_span = int(meta.get("col_span", 1) or 1)
        except (TypeError, ValueError):
            continue
        if row < header_row_count or col < 0 or col_span != 1:
            continue
        box = _bbox(meta)
        if box:
            per_col[col].append(box)
    ranges: Dict[int, Box] = {}
    for col, boxes in per_col.items():
        ranges[col] = (
            median(b[0] for b in boxes),
            median(b[1] for b in boxes),
            median(b[2] for b in boxes),
            median(b[3] for b in boxes),
        )
    return ranges


def _nearest_column(x: float, ranges: Dict[int, Box]) -> Optional[int]:
    best: Optional[int] = None
    best_distance = float("inf")
    for col, box in ranges.items():
        if box[0] - 1.0 <= x <= box[2] + 1.0:
            return col
        distance = min(abs(x - box[0]), abs(x - box[2]))
        if distance < best_distance:
            best, best_distance = col, distance
    return best


def split_header_labels(
    table: RawTable,
    line_provider: LineProvider,
    ranges: Dict[int, Box],
    *,
    header_row_count: int,
) -> Tuple[int, List[str]]:
    """Split header cells that contain labels from more than one column."""
    warnings: List[str] = []
    changed = 0
    if header_row_count <= 0 or not ranges or not table.grid:
        return changed, warnings

    origin_cells = [
        meta
        for meta in table.cells_meta
        if int(meta.get("row", -1)) < header_row_count
        and meta.get("row") == meta.get("origin_row")
        and meta.get("col") == meta.get("origin_col")
    ]

    for meta in sorted(
        origin_cells, key=lambda m: (int(m.get("row", 0)), int(m.get("col", 0)))
    ):
        try:
            row = int(meta.get("row", -1))
            col = int(meta.get("col", -1))
            span = max(1, int(meta.get("col_span", 1) or 1))
        except (TypeError, ValueError):
            continue
        if row < 0 or row >= len(table.grid) or col < 0 or col >= table.n_cols:
            continue
        text = normalize_ws(table.grid[row][col])
        box = _bbox(meta)
        if not text or not box:
            continue
        lines = _clean_lines(line_provider, box)
        if len(lines) < 2:
            continue
        if sorted(_tokens(" ".join(t for t, _ in lines))) != sorted(_tokens(text)):
            continue

        groups: Dict[int, List[str]] = defaultdict(list)
        group_boxes: Dict[int, List[Box]] = defaultdict(list)
        for line_text, line_box in lines:
            target = _nearest_column(_center_x(line_box), ranges)
            if target is None:
                continue
            groups[target].append(line_text)
            group_boxes[target].append(line_box)
        covered = set(range(col, col + span))
        outside = {c: " ".join(parts) for c, parts in groups.items() if c not in covered}
        if not outside:
            continue
        inside_cols = [c for c in groups if c in covered]
        if len(inside_cols) != 1:
            continue
        inside_text = " ".join(groups[inside_cols[0]])

        # Never overwrite a different existing header label.
        can_apply = True
        for target, label in outside.items():
            if target < 0 or target >= table.n_cols:
                can_apply = False
                break
            current = normalize_ws(table.grid[row][target])
            if current and current.lower() != label.lower():
                can_apply = False
                break
        if not can_apply:
            continue

        for target, label in outside.items():
            table.grid[row][target] = label
            info = {
                "row": row,
                "col": target,
                "origin_row": row,
                "origin_col": target,
                "row_span": 1,
                "col_span": 1,
                "text": label,
                "bbox": _union(group_boxes[target]),
                "confidence": 0.0,
                "column_header": True,
                "row_header": False,
                "page": table.page,
                "bbox_scope": "native_text_line",
                "recovered_from_header_merge": True,
            }
            existing = next(
                (
                    item
                    for item in table.cells_meta
                    if int(item.get("row", -1)) == row
                    and int(item.get("col", -1)) == target
                ),
                None,
            )
            if existing is not None:
                existing.update(info)
            else:
                table.cells_meta.append(info)

        for target in covered:
            if 0 <= target < table.n_cols:
                table.grid[row][target] = inside_text
        meta["text"] = inside_text
        changed += 1
        warnings.append(
            f"header_label_split:p{table.page}:r{row}:"
            f"cols{','.join(str(c) for c in sorted(outside))}"
        )
    return changed, warnings


def _group_line_blocks(
    lines: Sequence[Tuple[str, Box]],
) -> List[List[Tuple[str, Box]]]:
    ordered = sorted(lines, key=lambda item: (item[1][1], item[1][0]))
    blocks: List[List[Tuple[str, Box]]] = []
    current: List[Tuple[str, Box]] = []
    for text, box in ordered:
        if current:
            prev = current[-1][1]
            height = max(1.0, prev[3] - prev[1])
            gap = box[1] - prev[3]
            if gap > max(3.2, height * 0.45):
                blocks.append(current)
                current = []
        current.append((text, box))
    if current:
        blocks.append(current)
    return blocks


def repair_merged_column_values(
    table: RawTable,
    line_provider: LineProvider,
    ranges: Dict[int, Box],
    *,
    header_row_count: int,
) -> Tuple[int, List[str]]:
    """Reassemble fragmented vertically merged values inside their column."""
    warnings: List[str] = []
    changed = 0
    if not ranges or not table.grid:
        return changed, warnings

    # Only vertically merged cells (row_span > 1) can contain a value whose
    # text is fragmented across rows. Restrict native lines to the cell's own
    # bbox: grouping all lines in a column would swallow an entire dose column
    # and write it into the first data cell.
    pad = 10.0
    for meta in list(table.cells_meta):
        try:
            row = int(meta.get("row", -1))
            col = int(meta.get("col", -1))
            span = int(meta.get("row_span", 1) or 1)
        except (TypeError, ValueError):
            continue
        if meta.get("row") != meta.get("origin_row"):
            continue
        if row < header_row_count or row >= len(table.grid):
            continue
        if col < 1 or col >= table.n_cols or span < 2:
            continue
        band = ranges.get(col)
        box = _bbox(meta)
        if not band or not box:
            continue

        y_low = box[1] - pad
        y_high = box[3] + pad
        query = [band[0] - 1.0, y_low, band[2] + 1.0, y_high]
        lines = [
            (text, line_box)
            for text, line_box in _clean_lines(line_provider, query)
            if y_low <= _center_y(line_box) <= y_high
            and band[0] - 2.0 <= _center_x(line_box) <= band[2] + 2.0
        ]
        if len(lines) < 2:
            continue
        lines.sort(key=lambda item: (item[1][1], item[1][0]))

        block_text = normalize_ws(" ".join(text for text, _ in lines))
        block_tokens = _tokens(block_text)
        current = normalize_ws(table.grid[row][col])
        if not block_tokens or not current or current == block_text:
            continue
        if not _is_submultiset(_tokens(current), block_tokens):
            continue
        if Counter(_tokens(current)) == Counter(block_tokens):
            continue

        covered = range(row, min(row + span, len(table.grid)))
        conflict = False
        for covered_row in covered:
            value = normalize_ws(table.grid[covered_row][col])
            if value and not _is_submultiset(_tokens(value), block_tokens):
                conflict = True
                break
        if conflict:
            continue

        # A merged cell covers every row in its span; fill them all so no row
        # is left with a truncated fragment ("Every 2").
        for covered_row in covered:
            table.grid[covered_row][col] = block_text
        meta["text"] = block_text
        meta["bbox_scope"] = "native_text_block"
        meta["repaired_merged_value"] = True
        changed += 1
        warnings.append(f"merged_column_value_repaired:p{table.page}:r{row}:c{col}")

        # Remove fragments of this same value misassigned to another column
        # because Docling's cell bbox sits inside this column band.
        for other in table.cells_meta:
            try:
                other_row = int(other.get("row", -1))
                other_col = int(other.get("col", -1))
            except (TypeError, ValueError):
                continue
            if (
                other_col == col
                or other_col < 0
                or other_row < header_row_count
                or other_row >= len(table.grid)
                or other_col >= table.n_cols
            ):
                continue
            other_box = _bbox(other)
            if not other_box:
                continue
            if not (band[0] - 2.0 <= _center_x(other_box) <= band[2] + 2.0):
                continue
            if not (y_low <= _center_y(other_box) <= y_high):
                continue
            other_text = normalize_ws(table.grid[other_row][other_col])
            if not other_text or other_text == block_text:
                continue
            if _is_submultiset(_tokens(other_text), block_tokens):
                table.grid[other_row][other_col] = ""
                other["text"] = ""
                other["cleared_as_misassigned_fragment"] = True
                warnings.append(
                    f"misassigned_fragment_cleared:p{table.page}:"
                    f"r{other_row}:c{other_col}"
                )
    return changed, warnings


def repair_fragmented_column_values(
    table: RawTable,
    line_provider: LineProvider,
    ranges: Dict[int, Box],
    *,
    header_row_count: int,
) -> Tuple[int, List[str]]:
    """Rebuild values Docling split into stacked one-row fragments.

    Some engines report a vertically merged value as several adjacent cells
    with ``row_span = 1`` (``"Every" / "4" / "Weeks"`` on three different
    rows) instead of one merged cell. This groups such vertically contiguous
    fragments, rebuilds the phrase from native PDF lines, and repeats it over
    the blank rows up to the next value in the same column. Only columns that
    contain at least one fragmented group are touched.
    """
    warnings: List[str] = []
    changed = 0
    if not ranges or not table.grid:
        return changed, warnings

    last_row = len(table.grid) - 1
    for col in sorted(ranges):
        if col < 1 or col >= table.n_cols:
            continue
        band = ranges[col]
        entries: List[Tuple[int, str, Box]] = []
        for meta in table.cells_meta:
            if meta.get("col") != col or meta.get("row") != meta.get("origin_row"):
                continue
            try:
                row = int(meta.get("row", -1))
            except (TypeError, ValueError):
                continue
            if row < header_row_count or row > last_row:
                continue
            text = normalize_ws(table.grid[row][col])
            box = _bbox(meta)
            if text and box:
                entries.append((row, text, box))
        entries.sort(key=lambda item: item[0])
        if len(entries) < 2:
            continue

        groups: List[List[Tuple[int, str, Box]]] = []
        for item in entries:
            if groups:
                prev = groups[-1][-1][2]
                height = max(1.0, prev[3] - prev[1])
                gap = item[2][1] - prev[3]
                if (
                    band[0] - 2.0 <= _center_x(item[2]) <= band[2] + 2.0
                    and gap <= max(3.2, height * 0.45)
                ):
                    groups[-1].append(item)
                    continue
            groups.append([item])

        values: List[Tuple[int, str, Box]] = []
        qualified = False
        for group in groups:
            if len(group) == 1:
                values.append(group[0])
                continue
            y_low = min(item[2][1] for item in group) - 2.0
            y_high = max(item[2][3] for item in group) + 2.0
            lines = [
                (text, line_box)
                for text, line_box in _clean_lines(
                    line_provider, [band[0] - 1.0, y_low, band[2] + 1.0, y_high]
                )
                if y_low <= _center_y(line_box) <= y_high
                and band[0] - 2.0 <= _center_x(line_box) <= band[2] + 2.0
            ]
            if len(lines) < 2:
                values.extend(group)
                continue
            lines.sort(key=lambda item: (item[1][1], item[1][0]))
            block_text = normalize_ws(" ".join(text for text, _ in lines))
            block_tokens = _tokens(block_text)
            group_tokens = [_tokens(item[1]) for item in group]
            if (
                not block_tokens
                or max(len(tokens) for tokens in group_tokens) >= len(block_tokens)
                or not all(
                    _is_submultiset(tokens, block_tokens)
                    for tokens in group_tokens
                )
            ):
                values.extend(group)
                continue
            start_row = min(item[0] for item in group)
            values.append((start_row, block_text, _union([item[2] for item in group])))
            qualified = True

        if not qualified:
            continue

        values.sort(key=lambda item: item[0])
        seen_rows: set[int] = set()
        unique_values: List[Tuple[int, str, Box]] = []
        for value in values:
            if value[0] in seen_rows:
                continue
            seen_rows.add(value[0])
            unique_values.append(value)

        for index, (start, value_text, value_box) in enumerate(unique_values):
            end = (
                unique_values[index + 1][0] - 1
                if index + 1 < len(unique_values)
                else last_row
            )
            if end < start:
                continue
            value_tokens = _tokens(value_text)
            conflict = False
            for row in range(start, end + 1):
                current = normalize_ws(table.grid[row][col])
                if current and not _is_submultiset(_tokens(current), value_tokens):
                    conflict = True
                    break
            if conflict:
                continue

            rows_changed = 0
            for row in range(start, end + 1):
                if normalize_ws(table.grid[row][col]) != value_text:
                    rows_changed += 1
                table.grid[row][col] = value_text
                existing = next(
                    (
                        item
                        for item in table.cells_meta
                        if int(item.get("row", -1)) == row
                        and int(item.get("col", -1)) == col
                    ),
                    None,
                )
                info = {
                    "row": row,
                    "col": col,
                    "origin_row": start,
                    "origin_col": col,
                    "row_span": 1,
                    "col_span": 1,
                    "text": value_text,
                    "bbox": list(value_box),
                    "confidence": 0.0,
                    "column_header": False,
                    "row_header": False,
                    "page": table.page,
                    "bbox_scope": "native_text_block",
                    "repaired_fragmented_value": True,
                }
                if existing is not None:
                    existing.update(info)
                else:
                    table.cells_meta.append(info)
            if rows_changed:
                changed += rows_changed
                warnings.append(
                    f"fragmented_column_value_repaired:p{table.page}:"
                    f"r{start}-{end}:c{col}"
                )
    return changed, warnings


__all__ = [
    "column_ranges_from_cells",
    "repair_fragmented_column_values",
    "repair_merged_column_values",
    "split_header_labels",
]

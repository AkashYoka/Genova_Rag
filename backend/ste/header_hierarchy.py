"""Leaf headers and header paths from multi-row header bands (domain-agnostic)."""
from __future__ import annotations

import re
from typing import List, Sequence

from ste.models import normalize_ws, pad_row

_RANGE_RE = re.compile(
    r"(?i)(?:≥|>=|>|≤|<=|<)?\s*\d+(?:\.\d+)?\s*(?:–|-|—|to)\s*\d+(?:\.\d+)?"
)
_MULTI_RANGE_SPLIT_RE = re.compile(
    r"(?i)(?=(?:≥|>=|>|≤|<=|<)\s*\d)|(?<=\d)\s+(?=(?:≥|>=|>|≤|<=|<)?\s*\d)"
)
_COLUMN_N_RE = re.compile(r"(?i)^column\s+\d+$")
_SHORT_UNIT_RE = re.compile(r"(?i)^(?:\([^)]{0,12}\)|[a-z%]{1,8})$")


def split_jammed_ranges(text: str) -> List[str]:
    token = normalize_ws(text)
    if not token or not _RANGE_RE.search(token):
        return [token] if token else []
    ranges = _RANGE_RE.findall(token)
    if len(ranges) < 2:
        return [token]
    parts = [
        normalize_ws(p) for p in _MULTI_RANGE_SPLIT_RE.split(token) if normalize_ws(p)
    ]
    cleaned = [p for p in parts if _RANGE_RE.search(p)]
    return cleaned if len(cleaned) >= 2 else [token]


def expand_jammed_headers(
    headers: List[str],
    header_rows: List[List[str]],
    body: List[List[str]],
) -> tuple[List[str], List[List[str]], List[List[str]]]:
    """Spread multi-range headers into neighboring empty/Column-N leaves."""
    width = max(
        len(headers),
        max((len(r) for r in header_rows), default=0),
        max((len(r) for r in body), default=0),
        1,
    )
    headers = pad_row(headers, width)
    header_rows = [pad_row(r, width) for r in header_rows]
    body = [pad_row(r, width) for r in body]

    jam_col = -1
    jam_parts: List[str] = []
    for ci, h in enumerate(headers):
        parts = split_jammed_ranges(h)
        if len(parts) >= 3:
            jam_col = ci
            jam_parts = parts
            break
    if jam_col < 0:
        return headers, header_rows, body

    def weak(ci: int) -> bool:
        h = normalize_ws(headers[ci])
        return (
            not h
            or bool(_COLUMN_N_RE.match(h))
            or bool(_SHORT_UNIT_RE.match(h))
            or ci == jam_col
        )

    left = jam_col
    while left > 0 and weak(left - 1):
        left -= 1
    right = jam_col
    while right + 1 < width and weak(right + 1):
        right += 1
    span = list(range(left, right + 1))
    if len(span) < 2:
        return headers, header_rows, body

    assigned = list(jam_parts)[: len(span)]
    while len(assigned) < len(span):
        assigned.append(assigned[-1] if assigned else "")
    for slot, ci in enumerate(span):
        band = assigned[slot]
        if not band:
            continue
        headers[ci] = band
        if header_rows:
            header_rows[-1][ci] = band
    return headers, header_rows, body


def _is_range_leaf(text: str) -> bool:
    return bool(_RANGE_RE.search(normalize_ws(text)))


def _is_unit_fragment(text: str) -> bool:
    """Parenthetical unit shard such as (mg) or (kg)."""
    return bool(re.match(r"^\([^)]{0,16}\)$", normalize_ws(text)))


def _should_merge_horizontal(a: str, b: str) -> bool:
    """True when OCR split one header phrase across adjacent cells."""
    a, b = normalize_ws(a), normalize_ws(b)
    if not a or not b:
        return False
    if _is_range_leaf(a) or _is_range_leaf(b):
        return False
    # Duplicate span labels (Group|Group) are not word fragments.
    if a.lower() == b.lower():
        return False
    # Dose + (mg) / Body Weight + (kg) style
    if _is_unit_fragment(b) and len(a.split()) <= 4:
        return True
    # Spanning title fragments: Study|1 → Study 1
    if (
        re.search(r"[A-Za-z]", a)
        and re.fullmatch(r"\d{1,3}", b)
        and len(a.split()) <= 3
        and len(a) <= 24
    ):
        return True
    # Single short alphabetic tokens: Body|Weight, Dosing fragments, etc.
    # Refuse when both look like distinct numbered/complete group ids (Group1|Group2).
    if re.search(r"\d", a) and re.search(r"\d", b):
        return False
    if (
        len(a.split()) == 1
        and len(b.split()) == 1
        and re.match(r"(?i)^[A-Za-z(]", a)
        and re.match(r"(?i)^[A-Za-z(]", b)
        and len(a) <= 14
        and len(b) <= 14
    ):
        return True
    return False


def _leaf_row_index(rows: Sequence[Sequence[str]], col: int) -> int:
    """Row index of the most specific non-empty value in a column."""
    last = -1
    for ri, row in enumerate(rows):
        if col < len(row) and normalize_ws(row[col]):
            last = ri
    return last


def _next_nonempty(row: Sequence[str], start: int) -> int:
    for j in range(start, len(row)):
        if normalize_ws(row[j]):
            return j
    return -1


def stitch_split_header_phrases(header_rows: List[List[str]]) -> List[List[str]]:
    """Merge adjacent OCR-split header words into one phrase (same row).

    Does not merge sibling leaf labels (e.g. Low|High) — only fragments of one
    spanning phrase (Body|Weight|(kg)) or unit attachments (Dose|(mg)).
    """
    if not header_rows:
        return header_rows
    width = max(len(r) for r in header_rows)
    working = [pad_row(r, width) for r in header_rows]
    for ri, padded in enumerate(working):
        i = 0
        while i < width:
            a = normalize_ws(padded[i])
            if not a:
                i += 1
                continue
            j = _next_nonempty(padded, i + 1)
            if j < 0:
                break
            b = normalize_ws(padded[j])
            # Only skip empties when attaching a trailing unit fragment.
            gap = j - i - 1
            if gap > 0 and not _is_unit_fragment(b):
                i += 1
                continue
            if gap > 2:
                i += 1
                continue
            if a and b and _should_merge_horizontal(a, b):
                # Sibling leaves on this row must stay separate columns.
                if (
                    _leaf_row_index(working, i) == ri
                    and _leaf_row_index(working, j) == ri
                    and not _is_unit_fragment(b)
                ):
                    i += 1
                    continue
                padded[i] = normalize_ws(f"{a} {b}")
                padded[j] = ""
                continue
            i += 1
    return working


def _is_fragment_continuation(prev: str, nxt: str) -> bool:
    """Same-column fragments that are one label, not a hierarchy step."""
    a, b = normalize_ws(prev), normalize_ws(nxt)
    if not a or not b:
        return False
    if _is_range_leaf(a) or _is_range_leaf(b):
        return False
    # Attach parenthetical units: Dose + (mg)
    if re.match(r"^\([^)]{0,16}\)$", b):
        return True
    # Only single-token + single-token (Dosing + Freq.), not "Group A" + "Low"
    if (
        len(a.split()) == 1
        and len(b.split()) == 1
        and re.match(r"(?i)^[A-Za-z]", a)
        and re.match(r"(?i)^[A-Za-z.(]", b)
        and not re.search(r"\d", a + b)
    ):
        return True
    return False


def _join_header_parts(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    merged = [parts[0]]
    for p in parts[1:]:
        if _is_fragment_continuation(merged[-1], p):
            merged[-1] = normalize_ws(f"{merged[-1]} {p}")
        else:
            merged.append(p)
    return " / ".join(merged)


def _is_carryable_parent(val: str) -> bool:
    """Group/banner labels may span; terminal-looking titles must not.

    Structural only: long titles with many words, or values that look like
    complete leaf labels with dense digits+units, are not carried.
    """
    low = normalize_ws(val)
    if not low:
        return False
    words = low.split()
    # Dense measurement leaf like "150mg (n=175)" — do not span.
    if re.search(r"(?i)\d+\s*[a-z]{1,4}\b", low) and "(" in low:
        return False
    # Very long multi-word titles are usually terminal column names.
    if len(words) >= 5:
        return False
    return True


def _propagate_group_across_range_span(header_rows: List[List[str]]) -> List[List[str]]:
    """If a group label sits above a contiguous leaf-range band, span it fully.

    Textract often places a centered spanning title over only the middle cell of
    the leaf band. This copies that group label across the contiguous range
    columns using only text already present in the header rows.
    """
    if len(header_rows) < 2:
        return header_rows
    width = max(len(r) for r in header_rows)
    rows = [pad_row(r, width) for r in header_rows]
    # Prefer the densest range row as the leaf band.
    best_ri, best_count = -1, 0
    for ri, row in enumerate(rows):
        count = sum(1 for c in row if _is_range_leaf(c))
        if count > best_count:
            best_ri, best_count = ri, count
    if best_ri < 0 or best_count < 2:
        return rows
    leaf = rows[best_ri]
    # Contiguous spans of range leaves
    spans: List[tuple[int, int]] = []
    i = 0
    while i < width:
        if not _is_range_leaf(leaf[i]):
            i += 1
            continue
        j = i
        while j < width and _is_range_leaf(leaf[j]):
            j += 1
        if j - i >= 2:
            spans.append((i, j - 1))
        i = j
    for left, right in spans:
        for ri, row in enumerate(rows):
            if ri == best_ri:
                continue
            # Find a carryable group label inside/near this span on this row.
            label = ""
            for c in range(left, right + 1):
                val = normalize_ws(row[c])
                if (
                    val
                    and _is_carryable_parent(val)
                    and not _is_range_leaf(val)
                    and not _is_unit_fragment(val)
                ):
                    label = val
                    break
            if not label:
                # Allow one cell left of span (centered titles often drift)
                if left > 0:
                    val = normalize_ws(row[left - 1])
                    if (
                        val
                        and _is_carryable_parent(val)
                        and not _is_range_leaf(val)
                        and not _is_unit_fragment(val)
                    ):
                        label = val
            if not label:
                continue
            for c in range(left, right + 1):
                cur = normalize_ws(row[c])
                # Overwrite empty cells and stray unit shards left by OCR splits.
                if (
                    not cur
                    or cur.lower() == label.lower()
                    or _is_unit_fragment(cur)
                ):
                    row[c] = label
    return rows


def forward_fill_header_rows(header_rows: List[List[str]]) -> List[List[str]]:
    """Carry parent span labels into empty sibling cells (left → right).

    Does not fill gaps *between* two different parent banners — only trailing
    empties after a parent until the next distinct parent or end of row.
    """
    if not header_rows:
        return header_rows
    header_rows = stitch_split_header_phrases(
        [list(r) for r in header_rows]
    )
    header_rows = _propagate_group_across_range_span(header_rows)
    width = max(len(r) for r in header_rows)
    out: List[List[str]] = []
    for row in header_rows:
        padded = pad_row(row, width)
        empty_ratio = sum(1 for c in padded if not normalize_ws(c)) / max(1, width)
        if empty_ratio < 0.25:
            out.append(list(padded))
            continue

        def _next_nonempty(start: int) -> str:
            for j in range(start, width):
                v = normalize_ws(padded[j])
                if v:
                    return v
            return ""

        filled: List[str] = []
        prev = ""
        for i, cell in enumerate(padded):
            val = normalize_ws(cell)
            if val:
                prev = val if _is_carryable_parent(val) else ""
                filled.append(val)
                continue
            nxt = _next_nonempty(i + 1)
            # Gap between two distinct carryable parents → leave empty.
            if (
                prev
                and nxt
                and _is_carryable_parent(nxt)
                and nxt.lower() != prev.lower()
            ):
                filled.append("")
            else:
                filled.append(prev)
        out.append(filled)
    return out


def _column_parts(padded: List[List[str]], col: int) -> List[str]:
    parts: List[str] = []
    for r in padded:
        val = normalize_ws(r[col])
        if val and (not parts or parts[-1].lower() != val.lower()):
            parts.append(val)
    return parts


def leaf_headers(header_rows: Sequence[Sequence[str]]) -> List[str]:
    """Most specific leaf label per column (source text only)."""
    if not header_rows:
        return []
    width = max(len(r) for r in header_rows)
    padded = forward_fill_header_rows([pad_row(r, width) for r in header_rows])
    if len(padded) == 1:
        return list(padded[0])
    leaves: List[str] = []
    for c in range(width):
        parts = _column_parts(padded, c)
        if not parts:
            leaves.append(f"Column {c + 1}")
        elif len(parts) == 1:
            leaves.append(parts[0])
        else:
            # Prefer the range/band leaf when present; else last part.
            band = next((p for p in reversed(parts) if _is_range_leaf(p)), None)
            leaves.append(band or parts[-1])
    return leaves


def header_paths(header_rows: Sequence[Sequence[str]]) -> List[str]:
    """Full header paths from existing ancestor cells only.

    Hierarchy steps use `` / ``. OCR word-fragments of one label use spaces.
    """
    if not header_rows:
        return []
    width = max(len(r) for r in header_rows)
    padded = forward_fill_header_rows([pad_row(r, width) for r in header_rows])
    paths: List[str] = []
    for c in range(width):
        parts = _column_parts(padded, c)
        if not parts:
            paths.append(f"Column {c + 1}")
        else:
            paths.append(_join_header_parts(parts))
    return paths


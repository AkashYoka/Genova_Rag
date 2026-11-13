"""Geometry-aware header and body-cell reconstruction for Textract tables.

Deterministic only: uses CELL / MERGED_CELL row-col indices, spans, bboxes,
and extracted text. Never invents missing headers or values. When uncertain,
preserves the raw grid and emits warnings / needs_verification.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.ingestion.pdf_layout.types import normalize_ws

_DIGIT_RE = re.compile(r"\d")
# Parenthesized units like "(mg)" or explicit banner tokens — not free words.
_UNITISH_RE = re.compile(
    r"(?i)^(?:\(\s*[a-zA-Zμµ%/²³]{1,8}\s*\)|dose|value|amount|unit|rate|"
    r"freq(?:uency)?|total|count|%)$"
)
_FREQ_FRAGMENT_RE = re.compile(
    r"(?i)^(every|q|weeks?|days?|months?|hours?|hrs?|\d+)$"
)
_SPACE_RANGE_RE = re.compile(
    r"(?i)^(≥|>=|>|≤|<=|<)?\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*([a-zA-Z%/μµ²³]*)$"
)
_HYPHEN_RANGE_RE = re.compile(
    r"(?i)(≥|>=|>|≤|<=|<)?\s*-?\d+(?:\.\d+)?\s*(?:–|-|—|to)\s*-?\d+"
)


@dataclass
class HeaderReconstruction:
    """Normalized header paths derived from Textract geometry."""

    original_header_cells: List[Dict[str, Any]] = field(default_factory=list)
    normalized_header_paths: List[List[str]] = field(default_factory=list)
    leaf_headers: List[str] = field(default_factory=list)
    header_hierarchy: List[Dict[str, Any]] = field(default_factory=list)
    axis_candidates: List[Dict[str, Any]] = field(default_factory=list)
    reconstructed_header_rows: List[List[str]] = field(default_factory=list)
    reconstructed_rows: List[List[str]] = field(default_factory=list)
    normalization_confidence: float = 0.0
    content_hash: str = ""
    warnings: List[str] = field(default_factory=list)
    needs_verification: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _content_hash(
    header_rows: Sequence[Sequence[str]],
    rows: Sequence[Sequence[str]],
    merged_cells: Sequence[Dict[str, Any]],
) -> str:
    payload = {
        "header_rows": [list(r) for r in header_rows],
        "rows": [list(r) for r in rows[:24]],
        "merged": [
            {
                "r": m.get("row_index"),
                "c": m.get("col_index"),
                "rs": m.get("row_span"),
                "cs": m.get("col_span"),
                "t": normalize_ws(str(m.get("text") or "")),
            }
            for m in (merged_cells or [])[:48]
        ],
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def _pad_grid(rows: Sequence[Sequence[str]], width: int) -> List[List[str]]:
    out: List[List[str]] = []
    for row in rows:
        cells = [normalize_ws(c) for c in row]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        out.append(cells[:width])
    return out


def _join_fragments(parts: Sequence[str]) -> str:
    cleaned = [normalize_ws(p) for p in parts if normalize_ws(p)]
    if not cleaned:
        return ""
    # Avoid duplicating identical adjacent tokens.
    merged: List[str] = []
    for part in cleaned:
        if merged and merged[-1].lower() == part.lower():
            continue
        merged.append(part)
    return normalize_ws(" ".join(merged))


def _fill_merged_text(
    grid: List[List[str]],
    merged_cells: Sequence[Dict[str, Any]],
    *,
    row_offset: int = 0,
) -> None:
    """Propagate non-empty merged origin text across its span (in-place)."""
    height = len(grid)
    width = max((len(r) for r in grid), default=0)
    for m in merged_cells or []:
        text = normalize_ws(str(m.get("text") or ""))
        if not text:
            continue
        r0 = int(m.get("row_index") or 0) - row_offset
        c0 = int(m.get("col_index") or 0)
        rs = max(1, int(m.get("row_span") or 1))
        cs = max(1, int(m.get("col_span") or 1))
        if r0 < 0 or r0 >= height or c0 < 0 or c0 >= width:
            continue
        for r in range(r0, min(height, r0 + rs)):
            for c in range(c0, min(width, c0 + cs)):
                if not grid[r][c]:
                    grid[r][c] = text


def _stitch_horizontal_header_fragments(grid: List[List[str]]) -> List[str]:
    """Join adjacent non-numeric header tokens split across columns.

    Example: ``Body`` | ``Weight`` under an empty banner → ``Body Weight``.
    Only stitches when neither token contains digits (avoids joining weight bands).
    """
    warnings: List[str] = []
    if not grid:
        return warnings
    width = len(grid[0])
    for r, row in enumerate(grid):
        c = 0
        while c < width - 1:
            a = normalize_ws(row[c])
            b = normalize_ws(row[c + 1])
            if (
                a
                and b
                and not _DIGIT_RE.search(a)
                and not _DIGIT_RE.search(b)
                and not _UNITISH_RE.match(a)
                and not _UNITISH_RE.match(b)
                and len(a) <= 24
                and len(b) <= 24
            ):
                # Stitch only short single-token fragments (e.g. Body|Weight).
                # Avoid joining unrelated multi-word headers sitting side-by-side.
                a_tokens = a.split()
                b_tokens = b.split()
                if (
                    len(a_tokens) == 1
                    and len(b_tokens) == 1
                    and a.isalpha()
                    and b.isalpha()
                    and len(a) <= 12
                    and len(b) <= 12
                ):
                    joined = _join_fragments([a, b])
                    row[c] = joined
                    row[c + 1] = joined
                    warnings.append(
                        f"stitched_header_fragments:row{r}:col{c}-{c+1}:{joined!r}"
                    )
                    c += 2
                    continue
            c += 1
    return warnings


def _propagate_parent_labels(
    grid: List[List[str]],
    merged_cells: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """Fill blank parent header cells using repeated labels and merge spans."""
    if not grid:
        return
    height = len(grid)
    width = len(grid[0])
    for r in range(height):
        # Expand repeated identical non-numeric labels across intervening blanks.
        c = 0
        while c < width:
            val = normalize_ws(grid[r][c])
            if not val or _DIGIT_RE.search(val):
                c += 1
                continue
            j = c + 1
            while j < width:
                nxt = normalize_ws(grid[r][j])
                if not nxt:
                    j += 1
                    continue
                if nxt.lower() == val.lower():
                    j += 1
                    continue
                break
            saw_repeat = any(
                normalize_ws(grid[r][k]).lower() == val.lower()
                for k in range(c + 1, j)
            )
            if saw_repeat:
                for k in range(c, j):
                    if not normalize_ws(grid[r][k]):
                        grid[r][k] = val
            c = max(j, c + 1)

    # Empty merged header spans: fill from any non-empty cell already in the span.
    for m in merged_cells or []:
        r0 = int(m.get("row_index") or 0)
        c0 = int(m.get("col_index") or 0)
        rs = max(1, int(m.get("row_span") or 1))
        cs = max(1, int(m.get("col_span") or 1))
        if r0 < 0 or r0 >= height or c0 < 0 or c0 >= width:
            continue
        if r0 >= len(grid):  # body merges ignored here
            continue
        labels = []
        for r in range(r0, min(height, r0 + rs)):
            for c in range(c0, min(width, c0 + cs)):
                val = normalize_ws(grid[r][c])
                if val and not _DIGIT_RE.search(val):
                    labels.append(val)
        if not labels:
            continue
        # Prefer the longest categorical label (e.g. stitched "Body Weight").
        parent = max(labels, key=len)
        for r in range(r0, min(height, r0 + rs)):
            for c in range(c0, min(width, c0 + cs)):
                if not normalize_ws(grid[r][c]):
                    grid[r][c] = parent
                elif (
                    not _DIGIT_RE.search(grid[r][c])
                    and grid[r][c].lower() in parent.lower()
                    and grid[r][c].lower() != parent.lower()
                ):
                    grid[r][c] = parent


def _leaf_and_paths(
    grid: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not grid:
        return [], [], [], []
    width = len(grid[0])
    leaves: List[str] = []
    paths: List[List[str]] = []
    hierarchy: List[Dict[str, Any]] = []
    axes: List[Dict[str, Any]] = []

    for c in range(width):
        parts: List[str] = []
        for r in range(len(grid)):
            val = normalize_ws(grid[r][c])
            if not val:
                continue
            if not parts or parts[-1].lower() != val.lower():
                parts.append(val)
        if not parts:
            leaf = f"Column {c + 1}"
            path = [leaf]
            conf = 0.2
        else:
            child = parts[-1]
            # Prefer specific trailing label when parent is categorical.
            if len(parts) >= 2 and _DIGIT_RE.search(child) and not _DIGIT_RE.search(parts[0]):
                leaf = child
            elif len(parts) >= 2 and parts[0].lower() in {p.lower() for p in parts[1:]}:
                leaf = child
            else:
                leaf = " > ".join(parts) if len(parts) > 1 and not _DIGIT_RE.search(child) else child
            path = parts
            conf = 0.85 if len(parts) >= 1 else 0.4
        leaves.append(leaf)
        paths.append(path)
        hierarchy.append(
            {
                "column_index": c,
                "path": path,
                "leaf": leaf,
                "levels": [{"level": i + 1, "text": p} for i, p in enumerate(path)],
            }
        )
        axes.append(
            {
                "column_index": c,
                "header": leaf,
                "path": path,
                "has_numeric": bool(_DIGIT_RE.search(leaf)),
                "confidence": conf,
            }
        )
    return leaves, paths, hierarchy, axes


def _repair_space_separated_ranges(rows: List[List[str]]) -> List[str]:
    """Normalize ``30 100`` → ``30-100`` when hyphen was lost inside one cell.

    Only rewrites cells that already contain both endpoints as digits; does not
    invent missing bounds.
    """
    warnings: List[str] = []
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            text = normalize_ws(cell)
            if not text or _HYPHEN_RANGE_RE.search(text):
                continue
            m = _SPACE_RANGE_RE.match(text)
            if not m:
                continue
            op, lo, hi, unit = m.group(1) or "", m.group(2), m.group(3), m.group(4) or ""
            repaired = f"{op}{lo}-{hi}".strip()
            if unit:
                repaired = f"{repaired} {unit}".strip()
            row[ci] = repaired
            warnings.append(f"space_range_repaired:r{ri}c{ci}:{text!r}->{repaired!r}")
    return warnings


def _reconstruct_frequency_column(rows: List[List[str]]) -> List[str]:
    """Reassemble vertically fragmented schedule text using only source tokens.

    Textract often splits ``Every 4 weeks`` across three body rows. When a run of
    adjacent rows in one column are short frequency fragments, join the run and
    assign the combined phrase back to each row in the run.
    """
    warnings: List[str] = []
    if not rows:
        return warnings
    width = max(len(r) for r in rows)
    rows[:] = _pad_grid(rows, width)
    # Prefer column 1 (common schedule column); also scan other sparse columns.
    candidate_cols = [1] if width > 2 else list(range(1, width))
    for extra in range(2, min(width, 4)):
        if extra not in candidate_cols:
            candidate_cols.append(extra)

    for col in candidate_cols:
        fragments = [normalize_ws(r[col]) for r in rows]
        nonempty = sum(1 for f in fragments if f)
        if nonempty < 2:
            continue
        frag_like = sum(1 for f in fragments if f and _FREQ_FRAGMENT_RE.match(f))
        if frag_like < 2:
            continue
        # Walk runs of non-empty fragments.
        i = 0
        while i < len(fragments):
            if not fragments[i]:
                i += 1
                continue
            j = i
            parts: List[str] = []
            while j < len(fragments) and fragments[j]:
                # Stop run if we hit a complete phrase already.
                cur = fragments[j]
                parts.append(cur)
                joined = _join_fragments(parts)
                j += 1
                if re.search(
                    r"(?i)\bevery\s+\d+\s+(?:weeks?|days?|months?|hours?)\b",
                    joined,
                ):
                    break
                # Cap run length.
                if len(parts) >= 4:
                    break
            joined = _join_fragments(parts)
            if re.search(
                r"(?i)\bevery\s+\d+\s+(?:weeks?|days?|months?|hours?)\b",
                joined,
            ) or (
                len(parts) >= 2
                and any(p.lower() == "every" for p in parts)
                and any(p.isdigit() for p in parts)
            ):
                for k in range(i, j):
                    rows[k][col] = joined
                warnings.append(
                    f"freq_column_reconstructed:col{col}:rows{i}-{j-1}:{joined!r}"
                )
            i = max(j, i + 1)
    return warnings


def _reconstruct_split_phrase_cells(rows: List[List[str]]) -> List[str]:
    """Join horizontally adjacent unsupported-phrase fragments in a row.

    Example: ``Insufficient`` | ``Data`` | ``to Recommend`` | ``a Dose``.
    """
    warnings: List[str] = []
    phrase_token = re.compile(
        r"(?i)^(?:insufficient|data|to|recommend|a|dose|not|applicable|n/?a)$"
    )
    for ri, row in enumerate(rows):
        width = len(row)
        c = 0
        while c < width:
            if not phrase_token.match(normalize_ws(row[c])):
                c += 1
                continue
            j = c
            parts: List[str] = []
            while j < width and phrase_token.match(normalize_ws(row[j])):
                parts.append(normalize_ws(row[j]))
                j += 1
            if len(parts) >= 2:
                joined = _join_fragments(parts)
                for k in range(c, j):
                    row[k] = joined
                warnings.append(
                    f"phrase_cells_joined:r{ri}:c{c}-{j-1}:{joined!r}"
                )
            c = max(j, c + 1)
    return warnings


def _looks_like_header_continuation(row: Sequence[str], parent_row: Sequence[str]) -> bool:
    """True when a body row is actually a leaf header band (misclassified).

    Domain-agnostic: leading key cells empty, remaining cells are range/unit
    labels (contain digits or unitish tokens), and parent row is categorical.
    """
    cells = [normalize_ws(c) for c in row]
    if not cells or len(cells) < 3:
        return False
    nonempty = [c for c in cells if c]
    if len(nonempty) < 2:
        return False
    # Typical: first 1–2 cells empty, rest are axis labels.
    if cells[0]:
        return False
    digitish = sum(1 for c in nonempty if _DIGIT_RE.search(c) or _UNITISH_RE.match(c))
    if digitish < max(2, len(nonempty) - 1):
        return False
    # Reject if most cells look like measured values (pure integers without unit).
    pure_values = sum(
        1 for c in nonempty if re.fullmatch(r"-?\d+(?:\.\d+)?", c or "")
    )
    if pure_values >= max(2, len(nonempty) // 2):
        return False
    parent = [normalize_ws(c) for c in parent_row]
    parent_nonempty = [c for c in parent if c]
    if parent_nonempty and all(_DIGIT_RE.search(c) for c in parent_nonempty):
        return False
    return True


def reconstruct_table_headers(
    *,
    header_rows: Sequence[Sequence[str]],
    rows: Sequence[Sequence[str]],
    cells: Optional[Sequence[Dict[str, Any]]] = None,
    merged_cells: Optional[Sequence[Dict[str, Any]]] = None,
    existing_headers: Optional[Sequence[str]] = None,
) -> HeaderReconstruction:
    """Build normalized header paths and lightly repair fragmented body cells."""
    merged_cells = list(merged_cells or [])
    cells = list(cells or [])
    width = max(
        [len(r) for r in header_rows]
        + [len(r) for r in rows]
        + [len(existing_headers or [])]
        + [0]
    )
    if width == 0:
        return HeaderReconstruction(
            normalization_confidence=0.0,
            warnings=["empty_table"],
            needs_verification=True,
        )

    digest = _content_hash(header_rows, rows, merged_cells)
    header_grid = _pad_grid(header_rows, width) if header_rows else []
    body = _pad_grid(rows, width)
    warnings: List[str] = []

    # Promote a misclassified leaf-header band from body into header_rows.
    if header_grid and body and _looks_like_header_continuation(body[0], header_grid[-1]):
        header_grid.append(list(body[0]))
        body = body[1:]
        warnings.append("promoted_body_row_to_header_continuation")

    original_header_cells: List[Dict[str, Any]] = []
    for cell in cells:
        if not cell.get("is_header"):
            continue
        original_header_cells.append(
            {
                "row_index": cell.get("row_index"),
                "col_index": cell.get("col_index"),
                "text": normalize_ws(str(cell.get("text") or "")),
                "row_span": cell.get("row_span", 1),
                "col_span": cell.get("col_span", 1),
                "bbox": cell.get("bbox"),
                "confidence": cell.get("confidence"),
            }
        )

    if header_grid:
        _fill_merged_text(header_grid, merged_cells, row_offset=0)
        warnings.extend(_stitch_horizontal_header_fragments(header_grid))
        _propagate_parent_labels(header_grid, merged_cells)
        leaves, paths, hierarchy, axes = _leaf_and_paths(header_grid)
    else:
        leaves = [normalize_ws(h) for h in (existing_headers or [])] or [
            f"Column {i + 1}" for i in range(width)
        ]
        paths = [[h] for h in leaves]
        hierarchy = [
            {"column_index": i, "path": [h], "leaf": h, "levels": [{"level": 1, "text": h}]}
            for i, h in enumerate(leaves)
        ]
        axes = [
            {
                "column_index": i,
                "header": h,
                "path": [h],
                "has_numeric": bool(_DIGIT_RE.search(h)),
                "confidence": 0.5,
            }
            for i, h in enumerate(leaves)
        ]
        warnings.append("no_header_rows_raw_headers_used")

    warnings.extend(_repair_space_separated_ranges(body))
    warnings.extend(_reconstruct_frequency_column(body))
    warnings.extend(_reconstruct_split_phrase_cells(body))

    # Confidence: high when we have multi-row headers with leaf labels; lower
    # when many synthetic Column N labels or stitch warnings dominate.
    synthetic = sum(1 for h in leaves if h.lower().startswith("column "))
    conf = 0.9
    if synthetic:
        conf -= 0.15 * min(synthetic, 4)
    if any(w.startswith("stitched_header") for w in warnings):
        conf -= 0.05
    if any(w.startswith("no_header") for w in warnings):
        conf -= 0.25
    conf = max(0.15, min(1.0, conf))
    needs_verification = conf < 0.55 or synthetic > 0 or not header_grid

    return HeaderReconstruction(
        original_header_cells=original_header_cells,
        normalized_header_paths=paths,
        leaf_headers=leaves,
        header_hierarchy=hierarchy,
        axis_candidates=axes,
        reconstructed_header_rows=[list(r) for r in header_grid],
        reconstructed_rows=body,
        normalization_confidence=round(conf, 3),
        content_hash=digest,
        warnings=list(dict.fromkeys(warnings)),
        needs_verification=needs_verification,
    )


# Simple process-local cache keyed by content hash.
_RECON_CACHE: Dict[str, HeaderReconstruction] = {}


def reconstruct_table_headers_cached(
    *,
    header_rows: Sequence[Sequence[str]],
    rows: Sequence[Sequence[str]],
    cells: Optional[Sequence[Dict[str, Any]]] = None,
    merged_cells: Optional[Sequence[Dict[str, Any]]] = None,
    existing_headers: Optional[Sequence[str]] = None,
) -> HeaderReconstruction:
    digest = _content_hash(header_rows, rows, merged_cells or [])
    hit = _RECON_CACHE.get(digest)
    if hit is not None:
        return hit
    result = reconstruct_table_headers(
        header_rows=header_rows,
        rows=rows,
        cells=cells,
        merged_cells=merged_cells,
        existing_headers=existing_headers,
    )
    _RECON_CACHE[digest] = result
    return result

"""Expand visually/structurally connected unavailable-data regions.

Domain-agnostic: uses adjacency, fragment continuity, optional shading,
and generic unavailable phrasing — never document-specific keywords.
"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ste.models import normalize_ws, pad_row

Coord = Tuple[int, int]

_PURE_NUM_RE = re.compile(
    r"(?i)^[≥≤<>]?\s*-?\d+(?:[.,]\d+)?(?:\s*(?:%|mg|ml|kg|iu))?$"
)
_FULL_UNAVAIL_RE = re.compile(
    r"(?i)\b("
    r"insufficient\s+data|"
    r"not\s+available|"
    r"not\s+applicable|"
    r"not\s+reported|"
    r"none\s+reported|"
    r"data\s+not\s+available|"
    r"no\s+data"
    r")\b"
)
# Generic short tokens that appear when OCR splits an unavailable phrase.
_FRAG_UNAVAIL_RE = re.compile(
    r"(?i)^("
    r"insufficient|data|not|available|applicable|reported|none|"
    r"to|a|an|the|for|of|or|and|"
    r"recommend|recommended|dose|dosing|"  # common continuation words only
    r"n/?a|nil|unknown|undetermined"
    r")$"
)
_PRESENT_VALUE_RE = re.compile(
    r"(?i)^("
    r"[≥≤<>]?\d|"  # starts with digit/op
    r".*%$|"
    r".*\(\s*\d"  # rates like 2 (1.1%)
    r")"
)


def _is_present_value(text: str) -> bool:
    t = normalize_ws(text)
    if not t:
        return False
    if _FULL_UNAVAIL_RE.search(t):
        return False
    if _PURE_NUM_RE.match(t):
        return True
    if _PRESENT_VALUE_RE.match(t) and not _FRAG_UNAVAIL_RE.match(t):
        # Avoid treating "to" as present; already handled
        if re.search(r"\d", t):
            return True
    return False


def _is_unavail_seed(text: str) -> bool:
    t = normalize_ws(text)
    if not t:
        return False
    if _FULL_UNAVAIL_RE.search(t):
        return True
    if _FRAG_UNAVAIL_RE.match(t):
        return True
    # Multi-word cell that is mostly unavailable phrasing
    words = t.lower().split()
    if len(words) >= 2 and _FULL_UNAVAIL_RE.search(t):
        return True
    hit = sum(1 for w in words if _FRAG_UNAVAIL_RE.match(w))
    return hit >= 2 and hit >= len(words) - 1


def _is_expandable(text: str) -> bool:
    """Empty or unavailable fragment — never overwrite present numeric values."""
    t = normalize_ws(text)
    if not t:
        return True
    if _is_present_value(t):
        return False
    return _is_unavail_seed(t)


def _neighbors(r: int, c: int, n_rows: int, n_cols: int) -> List[Coord]:
    out: List[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < n_rows and 0 <= cc < n_cols:
            out.append((rr, cc))
    return out


def _compose_meaning(texts: Sequence[str]) -> str:
    parts: List[str] = []
    seen: Set[str] = set()
    for t in texts:
        tok = normalize_ws(t)
        if not tok:
            continue
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(tok)
    joined = normalize_ws(" ".join(parts))
    if not joined:
        return "Insufficient Data"
    # Prefer a clean canonical if the full phrase is present.
    if _FULL_UNAVAIL_RE.search(joined):
        # Keep original wording order when possible
        return joined
    return joined


def _mean_luminance(
    image_bytes: bytes,
    bbox_norm: Sequence[float],
) -> Optional[float]:
    """Return mean luminance 0..255 for a normalized Textract bbox, or None."""
    try:
        from io import BytesIO

        from PIL import Image
    except Exception:
        return None
    if len(bbox_norm) < 4:
        return None
    left, top, width, height = [float(x) for x in bbox_norm[:4]]
    if width <= 0 or height <= 0:
        return None
    try:
        im = Image.open(BytesIO(image_bytes)).convert("L")
        w, h = im.size
        x0 = max(0, int(left * w))
        y0 = max(0, int(top * h))
        x1 = min(w, int((left + width) * w))
        y1 = min(h, int((top + height) * h))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = im.crop((x0, y0, x1, y1))
        hist = crop.histogram()
        total = sum(hist) or 1
        mean = sum(i * v for i, v in enumerate(hist)) / total
        return float(mean)
    except Exception:
        return None


def _dark_cell(
    image_bytes: Optional[bytes],
    bbox_norm: Optional[Sequence[float]],
    *,
    threshold: float = 90.0,
) -> bool:
    if not image_bytes or not bbox_norm:
        return False
    lum = _mean_luminance(image_bytes, bbox_norm)
    return lum is not None and lum <= threshold


def expand_unavailable_regions(
    body: List[List[str]],
    *,
    cells_meta: Optional[List[Dict[str, Any]]] = None,
    image_bytes: Optional[bytes] = None,
    header_row_count: int = 0,
    max_fraction: float = 0.55,
) -> Tuple[List[List[str]], List[Dict[str, Any]], List[str], Dict[Coord, Dict[str, Any]]]:
    """Detect connected unavailable regions and expand into covered body cells.

    Returns:
      body (mutated copy applied),
      regions (region provenance list),
      warnings,
      cell_region_map: (row,col) -> {region_id, explanation, status}
    """
    warnings: List[str] = []
    if not body:
        return body, [], warnings, {}

    n_rows = len(body)
    n_cols = max(len(r) for r in body)
    body = [pad_row(r, n_cols) for r in body]

    # Optional shade map keyed by body-relative (r,c)
    shade_dark: Set[Coord] = set()
    meta_by_rc: Dict[Coord, Dict[str, Any]] = {}
    if cells_meta:
        for m in cells_meta:
            try:
                gr = int(m.get("row", -1))
                gc = int(m.get("col", -1))
            except Exception:
                continue
            br = gr - int(header_row_count or 0)
            if br < 0 or gc < 0:
                continue
            meta_by_rc[(br, gc)] = m
            if _dark_cell(image_bytes, m.get("bbox_norm")):
                shade_dark.add((br, gc))

    seeds: Set[Coord] = set()
    for r in range(n_rows):
        for c in range(n_cols):
            text = body[r][c]
            if _is_unavail_seed(text):
                seeds.add((r, c))
            elif (r, c) in shade_dark and not _is_present_value(text):
                # Dark shaded empty/non-numeric cell near unavailable content
                # becomes a seed only if an unavail neighbor exists (checked in grow)
                seeds.add((r, c))

    if not seeds:
        return body, [], warnings, {}

    # Restrict dark-only seeds: require adjacency to a text seed
    text_seeds = {
        (r, c)
        for r, c in seeds
        if _is_unavail_seed(body[r][c])
    }
    if text_seeds:
        pruned: Set[Coord] = set(text_seeds)
        for r, c in seeds:
            if (r, c) in text_seeds:
                continue
            if any(n in text_seeds for n in _neighbors(r, c, n_rows, n_cols)):
                pruned.add((r, c))
        seeds = pruned

    # Grow connected components among expandable cells
    visited: Set[Coord] = set()
    regions: List[Dict[str, Any]] = []
    cell_map: Dict[Coord, Dict[str, Any]] = {}
    region_idx = 0

    for start in sorted(seeds):
        if start in visited:
            continue
        if not _is_expandable(body[start[0]][start[1]]) and start not in shade_dark:
            continue
        q: deque[Coord] = deque([start])
        comp: List[Coord] = []
        local_visited: Set[Coord] = set()
        while q:
            r, c = q.popleft()
            if (r, c) in local_visited:
                continue
            local_visited.add((r, c))
            text = body[r][c]
            is_seedish = (r, c) in seeds or _is_unavail_seed(text)
            is_empty = not normalize_ws(text)
            dark = (r, c) in shade_dark
            if not (is_seedish or is_empty or dark):
                if _is_present_value(text):
                    continue
                if not _is_expandable(text):
                    continue
            if _is_present_value(text) and not _is_unavail_seed(text):
                continue
            comp.append((r, c))
            for nb in _neighbors(r, c, n_rows, n_cols):
                if nb in local_visited:
                    continue
                nr, nc = nb
                ntext = body[nr][nc]
                # Expand into neighbors that are empty, fragments, seeds, or dark
                if _is_present_value(ntext) and not _is_unavail_seed(ntext):
                    continue
                if (
                    _is_unavail_seed(ntext)
                    or not normalize_ws(ntext)
                    or nb in seeds
                    or nb in shade_dark
                    or _is_expandable(ntext)
                ):
                    # Require connectivity through seedish/empty/dark chain
                    q.append(nb)

        if not comp:
            continue
        # Must contain at least one textual unavailable cue (not shade-only blob)
        if not any(_is_unavail_seed(body[r][c]) for r, c in comp):
            continue
        # Safety: do not swallow most of the body
        if len(comp) > max(3, int(max_fraction * n_rows * n_cols)):
            warnings.append(
                f"unavailable_region_skipped_too_large:{len(comp)}"
            )
            visited |= local_visited
            continue

        region_idx += 1
        region_id = f"unavailable_region_{region_idx:03d}"
        meaning = _compose_meaning([body[r][c] for r, c in sorted(comp)])
        if not _FULL_UNAVAIL_RE.search(meaning):
            # Normalize generic incomplete fragments to a clear status label
            # while preserving fragment text in provenance.
            if any(_is_unavail_seed(body[r][c]) for r, c in comp):
                meaning = (
                    meaning
                    if meaning
                    else "Insufficient Data"
                )
                if "insufficient" not in meaning.lower() and "not " not in meaning.lower():
                    # Fragments only — keep composed text as explanation
                    pass

        covered = [{"row": r, "column": c} for r, c in sorted(comp)]
        regions.append(
            {
                "region_id": region_id,
                "meaning": meaning,
                "status": "insufficient_data",
                "covered_cells": covered,
                "reconstruction_method": "connected_unavailable_region",
            }
        )
        for r, c in comp:
            visited.add((r, c))
            body[r][c] = meaning
            cell_map[(r, c)] = {
                "region_id": region_id,
                "explanation": meaning,
                "status": "insufficient_data",
            }

    if regions:
        warnings.append(f"unavailable_regions_expanded:{len(regions)}")
        warnings.append(
            f"unavailable_region_cells:{sum(len(r['covered_cells']) for r in regions)}"
        )
    return body, regions, warnings, cell_map


def validate_unavailable_region_coverage(
    body: Sequence[Sequence[str]],
    regions: Sequence[Dict[str, Any]],
    cell_map: Dict[Coord, Dict[str, Any]],
) -> Dict[str, Any]:
    """Connected-region coverage check."""
    issues: List[str] = []
    checks: Dict[str, bool] = {}
    for region in regions:
        rid = region.get("region_id")
        covered = region.get("covered_cells") or []
        for cell in covered:
            r, c = int(cell["row"]), int(cell["column"])
            info = cell_map.get((r, c))
            if not info or info.get("status") != "insufficient_data":
                issues.append(f"missing_status:{rid}:{r},{c}")
                continue
            if not normalize_ws(info.get("explanation") or ""):
                issues.append(f"missing_explanation:{rid}:{r},{c}")
            if r < len(body) and c < len(body[r]):
                text = normalize_ws(body[r][c])
                if not text:
                    issues.append(f"empty_covered_cell:{rid}:{r},{c}")
                elif _is_present_value(text) and not _FULL_UNAVAIL_RE.search(text):
                    issues.append(f"present_value_inside_region:{rid}:{r},{c}")
        # Outside cells should not share this region_id incorrectly — checked via map
    checks["all_covered_cells_explicit"] = not any(
        i.startswith("missing_status") for i in issues
    )
    checks["no_empty_covered_cells"] = not any(
        i.startswith("empty_covered_cell") for i in issues
    )
    checks["no_present_values_inside_regions"] = not any(
        i.startswith("present_value_inside_region") for i in issues
    )
    return {
        "ok": not issues,
        "checks": checks,
        "issues": issues,
        "region_count": len(regions),
    }

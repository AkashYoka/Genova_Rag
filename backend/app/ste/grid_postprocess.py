"""Structural (domain-agnostic) cleanup of extracted TABLE grids.

Rules use layout/shape only — never invent domain column names or units.
Semantic disambiguation is handled by Bedrock (universal prompt), not here.
"""
from __future__ import annotations

import re
from typing import List, Optional, Pattern, Sequence, Tuple

from ste.models import normalize_ws, pad_row

_PURE_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_RANGE_SPACE_RE = re.compile(
    r"(?i)^(≥|>=|>|≤|<=|<)?\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)$"
)
_RANGE_SPACED_HYPHEN_RE = re.compile(
    r"(?i)((?:≥|>=|>|≤|<=|<)\s*)?(\d+(?:\.\d+)?)\s*[–\-—]\s*(\d+(?:\.\d+)?)"
)
# Short repeated unit / banner tokens (any domain).
_UNIT_BANNER_RE = re.compile(
    r"(?i)^(?:(?:amount|value|unit|qty|quantity|rate|score|count|n|total)"
    r"(?:\s*\([^)]{0,12}\))?|\([^)]{0,12}\))$"
)
_DEFAULT_UNAVAIL_FULL = re.compile(
    r"(?i)\b(insufficient\s+data|not\s+available|not\s+applicable|"
    r"not\s+reported|n/?a|nil|none\s+reported)\b"
)
_DEFAULT_UNAVAIL_FRAG = re.compile(
    r"(?i)^(insufficient|data|not|available|applicable|reported|n/?a|nil)$"
)


def is_unit_banner_row(row: Sequence[str]) -> bool:
    """True when value cells are mostly identical short unit/banner labels."""
    cells = [normalize_ws(c) for c in row]
    if len(cells) < 3:
        return False
    value_cells = cells[2:] if len(cells) > 2 else cells[1:]
    nonempty = [c for c in value_cells if c]
    if not nonempty:
        return False
    # Body measurements / p-notes are never unit banners.
    if any(
        "%" in c
        or _PURE_NUM_RE.match(c)
        or re.search(r"(?i)\bp\s*[<≥>=]", c)
        or re.search(r"[<≥>]=?\s*\d", c)
        for c in nonempty
    ):
        return False
    unitish = sum(1 for c in nonempty if _UNIT_BANNER_RE.match(c))
    if unitish >= 1 and unitish == len(nonempty):
        return True
    # Repeated identical short tokens also count (e.g. all "(mg)").
    if len(set(c.lower() for c in nonempty)) == 1 and len(nonempty[0]) <= 24:
        return unitish >= 1 or nonempty[0].startswith("(")
    return False


# Back-compat alias used by reconstructor / older imports
_is_unit_dose_row = is_unit_banner_row


def promote_unit_banner_rows(
    header_rows: List[List[str]],
    body: List[List[str]],
) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    warnings: List[str] = []
    if not body:
        return header_rows, body, warnings
    width = max(
        max((len(r) for r in header_rows), default=0),
        max(len(r) for r in body),
        1,
    )
    header_rows = [pad_row(r, width) for r in header_rows]
    body = [pad_row(r, width) for r in body]
    kept: List[List[str]] = []
    for row in body:
        if is_unit_banner_row(row):
            header_rows.append(list(row))
            warnings.append("promoted_unit_banner")
            continue
        kept.append(row)
    return header_rows, kept, warnings


def _normalize_range_token(text: str) -> Tuple[str, bool]:
    token = normalize_ws(text)
    if not token:
        return token, False
    changed = False
    m = _RANGE_SPACE_RE.match(token)
    if m:
        op, a, b = m.group(1) or "", m.group(2), m.group(3)
        token = f"{op}{a}-{b}"
        changed = True

    def _fix_spaced_hyphen(m: re.Match[str]) -> str:
        op = (m.group(1) or "").strip()
        return f"{op}{m.group(2)}-{m.group(3)}"

    cleaned, n = _RANGE_SPACED_HYPHEN_RE.subn(_fix_spaced_hyphen, token)
    if n:
        token = normalize_ws(cleaned)
        changed = True
    glued, g_n = re.subn(
        r"(?i)([A-Za-z])((?:≥|>=|>|≤|<=|<)?\d+(?:\.\d+)?-\d+(?:\.\d+)?)",
        r"\1 \2",
        token,
    )
    if g_n:
        token = normalize_ws(glued)
        changed = True
    return token, changed


def restore_range_hyphens(rows: List[List[str]]) -> List[str]:
    warnings: List[str] = []
    fixed = 0
    for row in rows:
        for i, cell in enumerate(row):
            new, ok = _normalize_range_token(cell)
            if ok:
                row[i] = new
                fixed += 1
    if fixed:
        warnings.append(f"range_hyphens_restored:{fixed}")
    return warnings


def restore_range_hyphens_in_headers(headers: List[str]) -> List[str]:
    warnings: List[str] = []
    fixed = 0
    out: List[str] = []
    for h in headers:
        new, ok = _normalize_range_token(h)
        out.append(new)
        if ok:
            fixed += 1
    if fixed:
        warnings.append(f"header_range_hyphens_restored:{fixed}")
    headers[:] = out
    return warnings


def stitch_wrapped_label_column(rows: List[List[str]], col: int = 1) -> List[str]:
    """Merge short non-numeric wrap fragments in a categorical column.

    Domain-agnostic: joins consecutive rows whose cells in ``col`` look like
    pieces of one label (short tokens, mostly letters), then back-fills the
    joined phrase into those rows. Does not invent vocabulary.
    """
    warnings: List[str] = []
    if not rows or col < 0:
        return warnings
    width = max(len(r) for r in rows)
    if col >= width:
        return warnings
    body = [pad_row(r, width) for r in rows]

    def _frag(tok: str) -> bool:
        t = normalize_ws(tok)
        if not t or len(t) > 24:
            return False
        if _PURE_NUM_RE.match(t):
            return True  # mid-phrase numbers allowed (e.g. "4")
        if re.search(r"[%$#=]", t):
            return False
        if re.search(r"\(\s*\d", t):
            return False
        # Prefer alphabetic / short alnum fragments
        return bool(re.search(r"[A-Za-z]", t)) and len(t.split()) <= 3

    def _standalone_heading(tok: str) -> bool:
        """Distinct row headings must not wrap-merge into each other."""
        t = normalize_ws(tok)
        if not t:
            return False
        # Short ALLCAPS codes (UAE, USA, KSA) are distinct row keys.
        if re.match(r"^[A-Z]{2,6}$", t):
            return True
        # Single Capitalized word
        if re.match(r"^[A-Z][A-Za-z]{3,}$", t):
            return True
        # Title-Case multi-word entities ("Saudi Arabia", "New Zealand")
        if re.match(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+$", t):
            return True
        return False

    def _complete(parts: List[str]) -> bool:
        joined = normalize_ws(" ".join(parts))
        words = joined.split()
        if len(words) < 2:
            return False
        # Completed when we have at least one letter-word and one number,
        # or 2+ letter words — typical wrap completions.
        has_alpha = any(re.search(r"[A-Za-z]{2,}", w) for w in words)
        has_num = any(_PURE_NUM_RE.match(w) for w in words)
        return has_alpha and (has_num or len(words) >= 2)

    i = 0
    stitched = 0
    while i < len(body):
        tok = normalize_ws(body[i][col])
        if not tok or not _frag(tok) or _PURE_NUM_RE.match(tok):
            i += 1
            continue
        # Only start a group on a short alphabetic fragment
        if not re.search(r"[A-Za-z]", tok):
            i += 1
            continue
        parts = [tok]
        j = i + 1
        empty_run = 0
        while j < len(body) and len(parts) < 4:
            nxt = normalize_ws(body[j][col])
            if not nxt:
                empty_run += 1
                # A wrapped label never spans more than one blank row;
                # larger gaps separate distinct merged values.
                if empty_run > 1:
                    break
                j += 1
                continue
            empty_run = 0
            if not _frag(nxt):
                break
            # New group if next looks like a full completed label already
            if len(nxt.split()) >= 3 and re.search(r"[A-Za-z]{3,}", nxt):
                break
            # Two standalone Capitalized headings are different rows, not a wrap.
            if _standalone_heading(parts[0]) and _standalone_heading(nxt):
                break
            parts.append(nxt)
            j += 1
            if _complete(parts):
                # Absorb one trailing short alpha token (e.g. "weeks" after "Every 4")
                if j < len(body):
                    peek = normalize_ws(body[j][col])
                    if (
                        peek
                        and _frag(peek)
                        and not _PURE_NUM_RE.match(peek)
                        and not _standalone_heading(peek)
                        and len(peek.split()) == 1
                        and len(peek) <= 12
                        and re.search(r"[A-Za-z]", peek)
                    ):
                        parts.append(peek)
                        j += 1
                break
        if len(parts) >= 2 and _complete(parts):
            phrase = normalize_ws(" ".join(parts))
            end = max(j, i + 1)
            for k in range(i, end):
                cur = normalize_ws(body[k][col])
                if not cur or _frag(cur):
                    body[k][col] = phrase
            stitched += 1
            i = end
            continue
        i += 1
    if stitched:
        warnings.append(f"label_wrap_stitched:{stitched}")
    rows[:] = body
    return warnings


def _paren_balance(text: str) -> int:
    return text.count("(") - text.count(")")


def _is_complete_independent_value(text: str) -> bool:
    """Standalone measurement that should not absorb a sibling value."""
    t = normalize_ws(text)
    if not t:
        return False
    if _PURE_NUM_RE.match(t):
        return True
    if re.fullmatch(r"-?\d+(?:\.\d+)?\s*%", t):
        return True
    if re.fullmatch(r"-?\d+(?:[.,]\d+)?\s*(?:%|mg|mcg|ml|mL|kg)?", t):
        return True
    return False


def _is_value_shard_pair(left: str, right: str) -> bool:
    """True when adjacent body cells are OCR shards of one visual value."""
    a, b = normalize_ws(left), normalize_ws(right)
    if not a or not b:
        return False
    # Two complete independent measurements stay separate.
    if _is_complete_independent_value(a) and _is_complete_independent_value(b):
        return False
    bal_a = _paren_balance(a)
    bal_b = _paren_balance(b)
    # Open paren on left, continuation/close on right.
    if bal_a > 0 and bal_b <= 0:
        return True
    # Value + trailing parenthetical note: 46% | (p<0.0001)
    if (
        _is_complete_independent_value(a)
        and b.startswith("(")
        and bal_b <= 0
        and len(b) <= 48
    ):
        return True
    # Trailing paren-only shard after any short left token.
    if (
        b.startswith("(")
        and ")" in b
        and bal_a == 0
        and not _is_complete_independent_value(b)
        and len(a) <= 32
        and not ( _is_complete_independent_value(a) and _is_complete_independent_value(b) )
    ):
        # Avoid merging two full phrases that just happen to have parens.
        if _is_complete_independent_value(a) or bal_a > 0 or a.endswith((":", ";")):
            return True
    # Left ends mid-token (open colon/paren/hyphen) and right continues.
    if a.endswith((":", ";", "-", "–", "—", "(")) and len(b) <= 64:
        return True
    return False


def stitch_split_value_shards(rows: List[List[str]], *, label_cols: int = 1) -> List[str]:
    """Merge OCR-split adjacent value shards within a body row.

    Domain-agnostic: uses only shape (parens, incomplete tokens). Never uses
    row-label vocabulary. Keeps independent sibling measurements separate.
    """
    warnings: List[str] = []
    if not rows:
        return warnings
    width = max(len(r) for r in rows)
    body = [pad_row(r, width) for r in rows]
    start = max(0, min(label_cols, width - 1))
    fixed = 0
    for row in body:
        i = start
        while i < width - 1:
            a = normalize_ws(row[i])
            if not a:
                i += 1
                continue
            j = i + 1
            while j < width and not normalize_ws(row[j]):
                j += 1
            if j >= width:
                break
            # Only absorb across a small empty gap (merged-cell OCR drift).
            if j > i + 2:
                i += 1
                continue
            b = normalize_ws(row[j])
            if _is_value_shard_pair(a, b):
                row[i] = normalize_ws(f"{a} {b}")
                row[j] = ""
                fixed += 1
                continue
            i += 1
    if fixed:
        warnings.append(f"value_shards_stitched:{fixed}")
    rows[:] = body
    return warnings


def _header_parent_key(path: str) -> str:
    """Parent prefix of a header path (everything before the last ' / ')."""
    p = normalize_ws(path)
    if " / " not in p:
        return p
    return p.rsplit(" / ", 1)[0]


def column_groups_from_headers(
    headers: Sequence[str],
    *,
    label_cols: int = 1,
) -> List[Tuple[int, int]]:
    """Consecutive value-column spans that share the same header parent."""
    width = len(headers)
    start = max(0, min(label_cols, width))
    if start >= width:
        return []
    groups: List[Tuple[int, int]] = []
    i = start
    while i < width:
        key = _header_parent_key(headers[i])
        j = i + 1
        while j < width and _header_parent_key(headers[j]) == key:
            j += 1
        groups.append((i, j - 1))
        i = j
    return groups


def expand_spanned_values(
    rows: List[List[str]],
    headers: Sequence[str],
    *,
    label_cols: int = 1,
) -> List[str]:
    """Copy a lone value across the columns of its visual header group.

    If a row has exactly one non-empty value cell in a group (or across the
    whole value band), fill every empty cell in that span. Fully cellular;
    no domain vocabulary.
    """
    warnings: List[str] = []
    if not rows or not headers:
        return warnings
    width = max(len(headers), max(len(r) for r in rows))
    body = [pad_row(r, width) for r in rows]
    headers = pad_row(list(headers), width)
    groups = column_groups_from_headers(headers, label_cols=label_cols)
    start = max(0, min(label_cols, width))
    filled = 0
    for row in body:
        # Per header-parent group first.
        for left, right in groups:
            nonempty = [
                c
                for c in range(left, right + 1)
                if normalize_ws(row[c])
            ]
            if len(nonempty) != 1:
                continue
            val = normalize_ws(row[nonempty[0]])
            for c in range(left, right + 1):
                if not normalize_ws(row[c]):
                    row[c] = val
                    filled += 1
        # Whole value band: one lone value with empties elsewhere.
        vals = [normalize_ws(row[c]) for c in range(start, width) if normalize_ws(row[c])]
        empties = [c for c in range(start, width) if not normalize_ws(row[c])]
        if len(set(vals)) == 1 and vals and empties:
            val = vals[0]
            for c in empties:
                row[c] = val
                filled += 1
    if filled:
        warnings.append(f"spanned_values_expanded:{filled}")
    rows[:] = body
    return warnings


def collapse_unavailable_fragments(
    rows: List[List[str]],
    *,
    fragment_re: Optional[Pattern[str]] = None,
    full_re: Optional[Pattern[str]] = None,
    canonical: str = "Insufficient Data",
) -> List[str]:
    """Collapse horizontal fragments of an unavailable-data phrase."""
    warnings: List[str] = []
    if not rows:
        return warnings
    frag_re = fragment_re or _DEFAULT_UNAVAIL_FRAG
    full = full_re or _DEFAULT_UNAVAIL_FULL
    width = max(len(r) for r in rows)
    body = [pad_row(r, width) for r in rows]
    fixed = 0
    for row in body:
        joined = " ".join(normalize_ws(c) for c in row if normalize_ws(c))
        if full.search(joined):
            for i, cell in enumerate(row):
                cur = normalize_ws(cell)
                if not cur:
                    continue
                if full.search(cur) or frag_re.match(cur):
                    if not _PURE_NUM_RE.match(cur):
                        row[i] = canonical
                        fixed += 1
            continue
        i = 0
        while i < width:
            cur = normalize_ws(row[i])
            if not cur or not frag_re.match(cur):
                i += 1
                continue
            j = i
            frags: List[str] = []
            while j < width:
                t = normalize_ws(row[j])
                if t and (frag_re.match(t) or full.search(t)):
                    frags.append(t)
                    j += 1
                    continue
                if not t:
                    j += 1
                    continue
                break
            blob = " ".join(frags).lower()
            if full.search(blob) or (
                "insufficient" in blob and "data" in blob
            ):
                for k in range(i, j):
                    if normalize_ws(row[k]) and not _PURE_NUM_RE.match(row[k]):
                        row[k] = canonical
                        fixed += 1
                i = j
                continue
            i += 1
    if fixed:
        warnings.append(f"unavailable_fragments_collapsed:{fixed}")
    rows[:] = body
    return warnings


def compose_header_paths(
    headers: List[str],
    header_rows: List[List[str]],
) -> Tuple[List[str], List[str]]:
    """Build reconstructed header paths from *existing* ancestor cell text only.

    Returns (original_leaf_headers, reconstructed_paths).
    Paths use `` / `` separators. Never invents domain tokens.
    """
    from ste.header_hierarchy import header_paths, leaf_headers

    width = max(len(headers), max((len(r) for r in header_rows), default=0), 1)
    headers = pad_row(headers, width)
    if header_rows:
        leaves = leaf_headers(header_rows)
        paths = header_paths(header_rows)
    else:
        leaves = list(headers)
        paths = list(headers)
    leaves = pad_row(leaves, width)
    paths = pad_row(paths, width)
    # Prefer path when it adds parent context; otherwise keep leaf.
    out: List[str] = []
    for leaf, path in zip(leaves, paths):
        if path and path != leaf and leaf in path:
            out.append(path)
        elif path:
            out.append(path)
        else:
            out.append(leaf)
    return leaves, out


def postprocess_grid(
    headers: List[str],
    header_rows: List[List[str]],
    body: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[List[str]], List[str]]:
    """Structural postprocess only (no domain rules)."""
    warnings: List[str] = []
    header_rows, body, w = promote_unit_banner_rows(header_rows, body)
    warnings.extend(w)
    warnings.extend(restore_range_hyphens(header_rows))
    warnings.extend(restore_range_hyphens(body))
    from ste.header_hierarchy import leaf_headers

    if header_rows:
        headers = leaf_headers(header_rows)
    warnings.extend(restore_range_hyphens_in_headers(headers))
    if body:
        width = max(len(r) for r in body)
        for col in range(min(3, width)):
            warnings.extend(stitch_wrapped_label_column(body, col=col))
    warnings.extend(collapse_unavailable_fragments(body))
    warnings.extend(stitch_split_value_shards(body, label_cols=1))
    _leaves, paths = compose_header_paths(headers, header_rows)
    # Use path form as working headers (still source-grounded).
    headers = paths
    warnings.extend(restore_range_hyphens_in_headers(headers))
    warnings.extend(expand_spanned_values(body, headers, label_cols=1))
    return headers, header_rows, body, warnings

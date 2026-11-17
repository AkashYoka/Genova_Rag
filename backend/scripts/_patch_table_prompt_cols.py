"""Patch compact_table_prompt_block to emit explicit [row,col] labels."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_RAG = ROOT / "app" / "rag" / "table_rag"

# Locate the module that owns compact_table_prompt_block (filename may vary).
target = None
for path in TABLE_RAG.glob("*.py"):
    text = path.read_text(encoding="utf-8")
    if "def compact_table_prompt_block" in text:
        target = path
        break
if target is None:
    raise SystemExit("compact_table_prompt_block not found")

text = target.read_text(encoding="utf-8")
old = '''    if headers:
        lines.append("headers: " + " | ".join(headers))

    if relevant_row_indices:
        keep = sorted({int(i) for i in relevant_row_indices if 0 <= int(i) < len(rows)})
        if keep and len(keep) < len(rows):
            for idx in keep:
                cells = [str(c or "") for c in rows[idx]]
                lines.append(f"row[{idx}]: " + " | ".join(cells))
            lines.append(f"(showing {len(keep)} relevant of {len(rows)} rows)")
        else:
            relevant_row_indices = None

    if not relevant_row_indices:
        shown = rows[:max_rows]
        for idx, row in enumerate(shown):
            cells = [str(c or "") for c in row]
            lines.append(f"row[{idx}]: " + " | ".join(cells))
        if len(rows) > max_rows:
            lines.append(f"... ({len(rows) - max_rows} more rows)")
'''

new = '''    if headers:
        # Explicit col indices prevent adjacent-band confusion
        # (e.g. body-weight >60-70 vs >70-90).
        col_bits = [f"[col{i}] {h}" for i, h in enumerate(headers)]
        lines.append("headers: " + " | ".join(col_bits))
        lines.append(
            "column_index_map: "
            + "; ".join(f"col{i}={h!r}" for i, h in enumerate(headers))
        )
        lines.append(
            "LOOKUP RULE: match the question numeric axes to the exact header "
            "band (honor exclusive/inclusive bounds). Read only the intersecting "
            "cell — never an adjacent weight or IgE column. State matched row "
            "label, column header, and [row_idx,col_idx] before quoting the value."
        )

    def _format_row(idx: int, row: list) -> str:
        cells = [str(c or "") for c in row]
        width = max(len(headers), len(cells))
        while len(cells) < width:
            cells.append("")
        cells = cells[:width]
        labeled = []
        for ci, val in enumerate(cells):
            header = headers[ci] if ci < len(headers) else f"col{ci}"
            labeled.append(f"[r{idx},c{ci}|{header}]={val}")
        return f"row[{idx}]: " + " | ".join(labeled)

    if relevant_row_indices:
        keep = sorted({int(i) for i in relevant_row_indices if 0 <= int(i) < len(rows)})
        if keep and len(keep) < len(rows):
            for idx in keep:
                lines.append(_format_row(idx, list(rows[idx])))
            lines.append(f"(showing {len(keep)} relevant of {len(rows)} rows)")
        else:
            relevant_row_indices = None

    if not relevant_row_indices:
        shown = rows[:max_rows]
        for idx, row in enumerate(shown):
            lines.append(_format_row(idx, list(row)))
        if len(rows) > max_rows:
            lines.append(f"... ({len(rows) - max_rows} more rows)")
'''

if old not in text:
    if "[r{idx},c{ci}|" in text or "[col{i}]" in text:
        print("already patched:", target.name)
    else:
        raise SystemExit(f"OLD BLOCK NOT FOUND in {target.name}")
else:
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("patched:", target.name)

# Ensure imports point at the real module stem.
stem = target.stem
for path in (ROOT / "app").rglob("*.py"):
    src = path.read_text(encoding="utf-8")
    # Historical module names that may still appear after renames.
    updated = src
    for legacy in ("txts", "texts"):
        updated = updated.replace(
            f"app.rag.table_rag.{legacy}",
            f"app.rag.table_rag.{stem}",
        )
    if updated != src:
        path.write_text(updated, encoding="utf-8")
        print("fixed imports in", path.relative_to(ROOT))

print("module_stem=", stem)

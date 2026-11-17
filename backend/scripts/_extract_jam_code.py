import json
from pathlib import Path

path = Path(
    r"C:\Users\sakth\.cursor\projects\c-PC-Yoka-Genova-Rag\agent-transcripts"
    r"\5d9983e0-37e4-48ff-b90c-2e661ba47ca6\5d9983e0-37e4-48ff-b90c-2e661ba47ca6.jsonl"
)
out_dir = Path(__file__).resolve().parent / "_jam_extract"
out_dir.mkdir(exist_ok=True)
want = (
    "_split_jammed_weight_header",
    "_expand_jammed_header_columns",
    "_compose_weight_dose_leaf",
    "_is_unit_dose_row",
)
lines = path.read_text(encoding="utf-8").splitlines()
for idx in [601, 602, 603, 604, 631, 632, 633, 634]:
    obj = json.loads(lines[idx])
    content = (obj.get("message") or {}).get("content") or []
    if isinstance(content, str):
        print("LINE", idx, "str", content[:120])
        continue
    for part in content:
        if part.get("type") != "tool_use":
            continue
        inp = part.get("input") or {}
        ns = inp.get("new_string") or inp.get("contents") or ""
        if any(f in ns for f in want) or "def _leaf_headers" in ns:
            name = part.get("name") or "tool"
            out = out_dir / f"line{idx}_{name}.txt"
            out.write_text(ns, encoding="utf-8")
            print("WROTE", out.name, "len", len(ns))

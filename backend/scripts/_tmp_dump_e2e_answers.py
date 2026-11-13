import json
from pathlib import Path

r = json.loads(Path("data/layout_reports/e2e_xolair_lucentis.json").read_text(encoding="utf-8"))
out = []
for name in ("xolair", "lucentis"):
    for q in r["docs"][name]["questions"]:
        lu = q.get("structured_lookup") or {}
        out.append(
            {
                "doc": name,
                "i": q["i"],
                "question": q["question"],
                "status": lu.get("status") or ("prose" if not lu else "other"),
                "value": lu.get("value"),
                "page": lu.get("page"),
                "candidates": lu.get("candidate_values"),
                "answer": (q.get("answer") or "")[:500],
            }
        )
Path("data/layout_reports/_e2e_answers_flat.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("wrote", len(out), "rows")
for row in out:
    print(f"{row['doc']} Q{row['i']}: {row['status']} val={row['value']} cands={row['candidates']}")
    print(" ", row["question"][:120])
    print(" ", row["answer"][:180].replace("\n", " "))
    print()

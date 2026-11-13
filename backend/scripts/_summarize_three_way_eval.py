"""Summarize three_way_table_eval.json into a readable report markdown."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "layout_reports" / "three_way_table_eval.json"
OUT = ROOT / "data" / "layout_reports" / "three_way_table_eval_report.md"


def main() -> int:
    r = json.loads(SRC.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("# Three-Way PDF Table Extraction + TableRAG Evaluation")
    lines.append("")
    lines.append(f"- **PDF:** `{r['pdf']}`")
    lines.append(f"- **Pages:** {r['pages']}")
    lines.append(
        f"- **Native Textract commit:** `{r['native_textract_commit']}` — "
        f"{r.get('native_textract_commit_subject')}"
    )
    lines.append("")
    lines.append("## Configurations")
    lines.append("")
    lines.append("| Case | Label | Key settings |")
    lines.append("|---|---|---|")
    for cid, c in r["cases"].items():
        cfg = c.get("config") or {}
        lines.append(
            f"| {cid} | {c.get('label')} | "
            f"`PDF_TABLE_ENGINE={cfg.get('PDF_TABLE_ENGINE')}`, "
            f"`OCR_BACKEND={cfg.get('PDF_IMG2TABLE_OCR_BACKEND')}`, "
            f"`FORCE_OCR={cfg.get('PDF_IMG2TABLE_FORCE_OCR')}`, "
            f"`SEPARATE_INTERP={cfg.get('TABLE_RAG_SEPARATE_INTERPRETATION_EMBEDDING')}` |"
        )
    lines.append("")
    lines.append("## Isolation")
    lines.append("")
    for cid, c in r["cases"].items():
        iso = c.get("isolation") or {}
        lines.append(
            f"- **{cid}:** user=`{c.get('user_id')}` doc=`{c.get('document_id')}` "
            f"chroma=`{iso.get('chroma_collection')}`"
        )
    lines.append("")
    lines.append("## Extraction comparison")
    lines.append("")
    lines.append(
        "| Evaluation | img2table + Tesseract | img2table + Textract | Native Textract |"
    )
    lines.append("|---|---|---|---|")
    labels = {"A": "A", "B": "B", "C": "C"}
    ct = r.get("comparison_table") or {}
    mapping = [
        ("Table detection", "table_detection"),
        ("Empty-cell rate", "cell_empty_rate"),
        ("Merged cells", "merged_cells"),
        ("Page 5 matrix", "page_5_tables"),
        ("Page 6 matrix", "page_6_tables"),
        ("Page 7 matrix", "page_7_tables"),
        ("Page 11 tables", "page_11_tables"),
        ("Retrieval accuracy", "retrieval_hit_rate"),
        ("Numerical answer accuracy (structured q1)", "numerical_lookup_hits"),
        ("Citation accuracy", "citation_pages_present"),
        ("Extraction latency (s)", "extraction_latency_s"),
        ("Total ingestion time (s)", "total_ingestion_s"),
        ("Cost per PDF (est. USD)", "cost_usd_est"),
    ]
    for title, key in mapping:
        vals = ct.get(key) or {}
        lines.append(
            f"| {title} | {vals.get('A')} | {vals.get('B')} | {vals.get('C')} |"
        )
    lines.append("")
    lines.append("### Per-table structure samples")
    lines.append("")
    for cid, c in r["cases"].items():
        lines.append(f"#### Case {cid}: {c.get('label')}")
        lines.append("")
        lines.append("| Page | Shape | Merged | Method | Headers (head) | Row0 (head) |")
        lines.append("|---|---|---|---|---|---|")
        for t in (c.get("extraction") or {}).get("tables") or []:
            lines.append(
                f"| {t.get('page')} | {t.get('n_rows')}x{t.get('n_cols')} | "
                f"{t.get('merged_cells')} | `{t.get('extraction_method')}` | "
                f"{t.get('headers')[:4]} | {t.get('sample_row0')[:6]} |"
            )
        lines.append("")
        lines.append(
            f"Index kinds: `{(c.get('extraction') or {}).get('index_kind_counts')}`"
        )
        lines.append("")

    lines.append("## Question answers (all three systems)")
    lines.append("")
    # Align questions by id
    qids = [q["id"] for q in r.get("questions") or []]
    for qid in qids:
        qmeta = next(q for q in r["questions"] if q["id"] == qid)
        lines.append(f"### {qid}")
        lines.append("")
        lines.append(f"**Question:** {qmeta['text']}")
        lines.append("")
        for cid, c in r["cases"].items():
            qq = next(
                (q for q in (c.get("questions") or []) if q["question_id"] == qid),
                None,
            )
            if not qq:
                lines.append(f"**{cid}:** _(missing)_")
                continue
            lookup = qq.get("structured_lookup")
            lines.append(f"**{cid} ({c.get('label')})**")
            lines.append(
                f"- Route: `{qq.get('route')}`"
            )
            lines.append(
                f"- Retrieved tables: `{qq.get('retrieved_table_ids')}`"
            )
            lines.append(
                f"- Retrieved rows: `{qq.get('retrieved_row_ids')}`"
            )
            lines.append(
                f"- Interpretation embedding hits: "
                f"{qq.get('interpretation_embedding_hits')} "
                f"(contributed={qq.get('interpretation_contributed')})"
            )
            lines.append(f"- Structured lookup: `{json.dumps(lookup, ensure_ascii=False)}`")
            lines.append(
                f"- Latency: {qq.get('latency_seconds')}s"
            )
            ans = (qq.get("answer") or "").strip() or "_(no answer)_"
            lines.append("")
            lines.append(ans)
            lines.append("")
            lines.append("<details><summary>Retrieval trace (RRF top)</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(qq.get("rrf_retrieved") or [], indent=2)[:4000])
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines.append("## 2-rep vs 3-rep ablation (Case A path)")
    lines.append("")
    abl = r.get("ablation_two_vs_three")
    if not abl:
        lines.append("_Not run._")
    elif abl.get("error"):
        lines.append(f"_Failed: {abl['error']}_")
    else:
        for tag in ("two_rep", "three_rep"):
            block = abl.get(tag) or {}
            lines.append(f"### {tag}")
            lines.append(f"- Index kinds: `{block.get('index_kind_counts')}`")
            for q in block.get("questions") or []:
                lines.append(
                    f"- {q.get('question_id')}: tables={q.get('retrieved_table_ids')} "
                    f"interp={q.get('interpretation_contributed')} "
                    f"lookup={q.get('structured_lookup')} "
                    f"ans_head={(q.get('answer_head') or '')[:120]!r}"
                )
            lines.append("")

    rec = r.get("recommendation") or {}
    lines.append("## Recommendation")
    lines.append("")
    hv = rec.get("human_verdict") or {}
    if hv:
        lines.append(f"**Winner: Case {hv.get('winner')} — {hv.get('title')}**")
        lines.append("")
        for w in hv.get("why") or []:
            lines.append(f"- {w}")
        lines.append("")
        lines.append("### Caveats")
        lines.append("")
        for w in hv.get("caveats") or []:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append(f"**Winner: Case {rec.get('winner')}**")
        lines.append("")
        lines.append(rec.get("rationale") or "")
        lines.append("")
    lines.append(f"Auto scores: `{rec.get('scores')}`")
    lines.append("")
    lines.append("Auto notes:")
    for cid, notes in (rec.get("notes") or {}).items():
        lines.append(f"- **{cid}:** {', '.join(notes)}")
    lines.append("")
    an = r.get("accuracy_notes") or {}
    if an:
        lines.append("### Q1 accuracy notes")
        lines.append("")
        for k, v in an.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for lim in r.get("limitations") or []:
        lines.append(f"- {lim}")
    lines.append("- Bedrock query-transformation model id warnings were observed; retrieval still used the original query.")
    lines.append("- LLM table understanding often fell back to deterministic interpretation (live LLM batches partially failed).")
    lines.append("- Page numbers inside table_ids reflect subset-PDF indices before remap; reported page metrics use remapped source pages.")
    lines.append("- Cost figures are public-list estimates, not AWS Cost Explorer actuals.")
    lines.append("- Cell accuracy vs ground-truth PDF was judged via structured lookup + sample rows, not a full cell-by-cell OCR audit.")
    lines.append("- FORCE_OCR was enabled for Cases A/B so OCR backends are comparable; production img2table-without-FORCE_OCR may differ.")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

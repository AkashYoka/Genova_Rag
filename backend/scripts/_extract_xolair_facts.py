"""Extract verifiable snippets from Xolair prescribing PDF for UI QA."""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf

CANDIDATES = [
    Path(r"C:\PC\Yoka\Misc\genova-rag\samples\Medcare\xolair_prescribing.pdf"),
    Path(r"C:\PC\Yoka\PDF-Assistant-RAG-dev\PDF-Assistant-RAG-dev\sample\xolair_prescribing.pdf"),
]
OUT = Path(__file__).with_name("_xolair_facts.json")


def main() -> None:
    path = next(p for p in CANDIDATES if p.exists())
    doc = pymupdf.open(path)
    pages = []
    for i in range(len(doc)):
        pages.append({"page": i + 1, "text": doc[i].get_text("text")})

    # Focus pages typically containing highlights / dosage / warnings
    focus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 19, 27, 43, 45, 46]
    focus_text = {p: next(x["text"] for x in pages if x["page"] == p) for p in focus if p <= len(doc)}

    queries = [
        "subcutaneous",
        "Chronic Spontaneous Urticaria",
        "150 mg or 300 mg",
        "every 4 weeks",
        "anaphylaxis",
        "CONTRAINDICATIONS",
        "status asthmaticus",
        "acute bronchospasm",
        "IgE-Mediated Food Allergy",
        "nasal polyps",
        "DRUG INTERACTIONS",
        "parasitic",
        "helminth",
        "Pregnancy",
        "Lactation",
        "self-administration",
        "prefilled syringe",
        "autoinjector",
        "serum total IgE",
        "inhaled corticosteroids",
        "H1 antihistamine",
        "emergency treatment",
        "75 mg to 375 mg",
        "every 2 or 4 weeks",
    ]
    hits = []
    for q in queries:
        for page in pages:
            if q.lower() in page["text"].lower():
                idx = page["text"].lower().find(q.lower())
                snippet = page["text"][max(0, idx - 120) : idx + 500].replace("\n", " ")
                hits.append({"query": q, "page": page["page"], "snippet": snippet[:650]})
                break

    OUT.write_text(
        json.dumps(
            {
                "path": str(path),
                "page_count": len(doc),
                "hits": hits,
                "focus_pages": {str(k): v[:3500] for k, v in focus_text.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT} hits={len(hits)}")
    for h in hits:
        print(f"p{h['page']:02d} [{h['query']}] {h['snippet'][:160]}...")


if __name__ == "__main__":
    main()

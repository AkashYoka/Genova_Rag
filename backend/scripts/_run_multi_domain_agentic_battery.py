"""Live /chat/ask E2E battery across Trading / Research / Medical / Chemistr / Algebra PDFs.

Questions and expected checks are grounded in the source PDFs under
``data/uploads/all-domain-textract-reindex/``. Document IDs match the smoke suite.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent / "_multi_domain_agentic_battery.json"
BASE = "http://127.0.0.1:7860"

# Fallback IDs from scripts/_smoke_nonmedical_iterative.py; runtime resolve by
# filename overrides these when the documents API is available.
DOCS = {
    "trading": "3e8616b1-baaa-4648-89d6-a3669c11d35f",
    "research": "14d09766-cb93-4f4e-9232-27c75b42a971",
    "medical": "37a6219e-4c06-456f-a35e-17adee8999c8",  # Medical.pdf = ROZLYTREK
    "chemistry": "59118577-e022-4b09-974f-764933e2a4cf",
    "algebra": "6df2ad7b-4944-4e55-96c9-f5f00374a932",
}

CASES = [
    # ---- Trading.pdf (Honma rice-trading text; mostly prose) ----
    {
        "doc": "trading",
        "id": "T1",
        "persona": "trading",
        "question": "According to the document, what should a trader not do during a good harvest year?",
        "expect": {
            "any": [r"do not sell|don't sell|not sell"],
            "any2": [r"rice"],
            "page_hint": True,
        },
    },
    {
        "doc": "trading",
        "id": "T2",
        "persona": "trading",
        "question": (
            "In section 2.1.1 'In trading the beginning is crucial', how many days "
            "should you wait from the moment you feel you could profit before buying or selling?"
        ),
        "expect": {
            "any": [r"three days|3 days|wait (for )?three"],
        },
    },
    {
        "doc": "trading",
        "id": "T3",
        "persona": "trading",
        "question": "Summarize the guidance about buying when everybody is weighed down and the market looks bearish.",
        "expect": {
            "any": [r"buy|buying"],
            "any2": [r"bear|drop|weighed|sluggish|determination"],
        },
    },
    {
        "doc": "trading",
        "id": "T4",
        "persona": "trading",
        "question": "What Xolair dose does this trading document recommend for asthma?",
        "expect": {"abstain": True},
    },
    # ---- Research.pdf (GraphRAG survey) ----
    {
        "doc": "research",
        "id": "R1",
        "persona": "research",
        "question": (
            "From the opening title block, what survey topic does this paper cover, "
            "and name author Qinggang Zhang's role on the author list?"
        ),
        "expect": {
            "any": [r"graph retrieval-augmented|graphrag|retrieval-augmented generation"],
            "any2": [r"qinggang zhang|first|lead|corresponding|\*"],
        },
    },
    {
        "doc": "research",
        "id": "R2",
        "persona": "research",
        "question": "According to the abstract, what three critical challenges do traditional RAG systems face?",
        "expect": {
            "any": [r"query understanding|complex query"],
            "any2": [r"knowledge integration|integrat"],
            "any3": [
                r"scalability|latency|knowledge base|efficiency|bottleneck|at scale"
            ],
        },
    },
    {
        "doc": "research",
        "id": "R3",
        "persona": "research",
        "question": "How does the paper categorize GraphRAG models based on graph structure utilization?",
        "expect": {
            "any": [r"knowledge-based|knowledge based|knowledge carriers"],
            "any2": [r"index-based|index|indexing"],
        },
    },
    {
        "doc": "research",
        "id": "R4",
        "persona": "research",
        "question": "What pediatric Xolair dosing table does this GraphRAG survey recommend?",
        "expect": {"abstain": True},
    },
    # ---- Medical.pdf (ROZLYTREK) — document id resolved at runtime ----
    {
        "doc": "medical",
        "id": "M1",
        "persona": "medical",
        "question": "What is ROZLYTREK (entrectinib) indicated for in ROS1-positive disease?",
        "expect": {
            "any": [r"ros1"],
            "any2": [r"nsclc|non-small cell|lung cancer"],
            "any3": [r"adult"],
        },
    },
    {
        "doc": "medical",
        "id": "M2",
        "persona": "medical",
        "question": (
            "According to the footnote under the NTRK pediatric BSA dosing guidance, "
            "what target dose in mg/m2 are the BSA categories based on?"
        ),
        "expect": {
            # Verifiable from extracted Medical.pdf footnote (BSA table grid
            # rows are not present in the indexed text; the 300 mg/m2 note is).
            "any": [r"300\s*mg\s*/\s*m|300\s*mg/m|300\s*mg\s*per\s*m"],
            "page_hint": True,
        },
    },
    {
        "doc": "medical",
        "id": "M3",
        "persona": "medical",
        "question": (
            "If a ROZLYTREK dose is missed, when should the patient make up that dose "
            "according to the administration overview?"
        ),
        "expect": {
            "any": [r"12\s*hours|within 12"],
        },
    },
    {
        "doc": "medical",
        "id": "M4",
        "persona": "medical",
        "question": "What Xolair asthma IgE/weight dose does the ROZLYTREK label recommend?",
        "expect": {"abstain": True},
    },
    # ---- Chemistr.pdf ----
    {
        "doc": "chemistry",
        "id": "C1",
        "persona": "chemistry",
        "question": "What three domains do chemists use to study and describe the behavior of matter and energy?",
        "expect": {
            "any": [r"macroscopic"],
            "any2": [r"microscopic"],
            "any3": [r"symbolic"],
        },
    },
    {
        "doc": "chemistry",
        "id": "C2",
        "persona": "chemistry",
        "question": "According to Table 1.1 on elemental composition of Earth, what is the percent mass of oxygen?",
        "expect": {
            "any": [r"49\.?2|49\.20|nearly one-half|one-half|almost half|~?49"],
        },
    },
    {
        "doc": "chemistry",
        "id": "C3",
        "persona": "chemistry",
        "question": "Define a molecule as stated in this document, and give one elemental example that consists of pairs of atoms.",
        "expect": {
            "any": [r"two or more atoms|joined|chemical bonds"],
            "any2": [r"hydrogen|nitrogen|oxygen|chlorine"],
        },
    },
    {
        "doc": "chemistry",
        "id": "C4",
        "persona": "chemistry",
        "question": "What is the recommended ROZLYTREK dose for BSA ≥1.51 m2 in this chemistry textbook?",
        "expect": {"abstain": True},
    },
    # ---- AlgebraAndTrigo.pdf ----
    {
        "doc": "algebra",
        "id": "A1",
        "persona": "mathematics",
        "question": "How does Example 1 write the integer 7 as a rational number?",
        "expect": {
            "any": [
                r"7\s*/\s*1",
                r"numerator.{0,40}1\s+in\s+the\s+denominator",
                r"numerator and 1",
                r"7\s+over\s+1",
                r"fraction.{0,80}7",
                r"integer 7 is written as a fraction",
            ],
        },
    },
    {
        "doc": "algebra",
        "id": "A2",
        "persona": "mathematics",
        "question": "According to the How To section, what are the four steps for simplifying an expression using the order of operations?",
        "expect": {
            "any": [r"grouping"],
            "any2": [r"exponent|radical"],
            "any3": [r"multiplication|division"],
            "any4": [r"addition|subtraction"],
        },
    },
    {
        "doc": "algebra",
        "id": "A3",
        "persona": "mathematics",
        "question": (
            "In the distributive property discussion, what is 4 · [12 + (−7)] equal to "
            "after distributing and simplifying?"
        ),
        "expect": {"any": [r"\b20\b"]},
    },
    {
        "doc": "algebra",
        "id": "A4",
        "persona": "mathematics",
        "question": "Does this algebra chapter contain a medical dosing table for omalizumab?",
        "expect": {"abstain": True},
    },
]


def _low(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    # Normalize common LaTeX fractions so 7/1 matches \\frac{7}{1}.
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"\1/\2",
        text,
    )
    text = text.replace(r"\(", " ").replace(r"\)", " ")
    return re.sub(r"\s+", " ", text)


def score(ans: str, expect: dict) -> dict:
    low = _low(ans)
    checks = {}
    ok_n = 0
    total = 0

    def mark(name: str, cond: bool) -> None:
        nonlocal ok_n, total
        total += 1
        checks[name] = bool(cond)
        if cond:
            ok_n += 1

    if expect.get("abstain"):
        mark(
            "abstain",
            bool(
                re.search(
                    r"insufficient|not (enough|present|found|contain)|no (evidence|mention)|"
                    r"does not (contain|discuss|include|recommend)|cannot (find|determine)|"
                    r"could(n't| not) find|not available|no .*table|unrelated|"
                    r"no mention|focuses (entirely )?on|do not contain",
                    low,
                )
            ),
        )
    for key in ("any", "any2", "any3", "any4"):
        patterns = expect.get(key)
        if not patterns:
            continue
        mark(key, any(re.search(p, low) for p in patterns))
    if expect.get("page_hint"):
        mark("page_or_source", bool(re.search(r"\bpage\b|\bsource\b", low)))

    return {
        "pass": ok_n,
        "total": total,
        "ok": ok_n == total and total > 0,
        "checks": checks,
    }


def _resolve_doc_ids(client: httpx.Client, headers: dict) -> dict:
    """Map logical names to live document IDs by filename when possible."""
    ids = dict(DOCS)
    try:
        rows = client.get("/api/v1/documents", headers=headers).json()
    except Exception:
        return ids
    items = rows if isinstance(rows, list) else (rows.get("items") or rows.get("documents") or [])
    by_name = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        name = str(row.get("filename") or row.get("original_filename") or "").lower()
        did = str(row.get("id") or "")
        if name and did:
            by_name[name] = did
    mapping = {
        "trading": ["trading.pdf"],
        "research": ["research.pdf"],
        "medical": ["medical.pdf"],
        "chemistry": ["chemistr.pdf", "chemistry.pdf"],
        "algebra": ["algebraandtrigo.pdf", "algebra.pdf"],
    }
    for key, names in mapping.items():
        for n in names:
            if n in by_name:
                ids[key] = by_name[n]
                break
    return ids


def main() -> None:
    results = []
    with httpx.Client(base_url=BASE, timeout=240.0) as client:
        tok = client.post(
            "/api/v1/auth/login",
            json={"email": "dev@genova.ai", "password": "localdev123"},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {tok}"}
        try:
            client.post("/api/v1/chat/cache/clear", headers=headers)
        except Exception:
            pass

        doc_ids = _resolve_doc_ids(client, headers)
        print("doc_ids", json.dumps(doc_ids, indent=2), flush=True)

        sessions = {}
        for key, did in doc_ids.items():
            sess = client.post(
                "/api/v1/chat/sessions",
                headers=headers,
                json={"title": f"Agentic E2E: {key}", "document_id": did},
            )
            sessions[key] = (sess.json() or {}).get("id")

        for case in CASES:
            did = doc_ids[case["doc"]]
            sid = sessions[case["doc"]]
            t0 = time.perf_counter()
            r = client.post(
                "/api/v1/chat/ask",
                headers=headers,
                json={
                    "question": case["question"],
                    "document_id": did,
                    "llm_provider": "bedrock",
                    "session_id": sid,
                },
            )
            elapsed = round(time.perf_counter() - t0, 2)
            data = r.json() if r.status_code < 500 else {"error": r.text[:500]}
            ans = data.get("answer") or ""
            sc = score(ans, case["expect"])
            # Soft persona check from answer voice / domain markers when present.
            persona_ok = True
            if case.get("persona") == "medical" and "rozlytrek" in case["question"].lower():
                persona_ok = "xolair" not in _low(ans) or "insufficient" in _low(ans)
            row = {
                "id": case["id"],
                "doc": case["doc"],
                "document_id": did,
                "persona_expected": case.get("persona"),
                "question": case["question"],
                "status": r.status_code,
                "elapsed_s": elapsed,
                "score": sc,
                "persona_ok": persona_ok,
                "ans_head": ans[:700],
                "error": None if r.status_code < 300 else data,
            }
            results.append(row)
            flag = "PASS" if sc["ok"] and r.status_code < 300 else "FAIL"
            print(
                f"{case['id']} {flag} {sc['pass']}/{sc['total']} {elapsed}s "
                f"{ans[:120].replace(chr(10), ' ')}",
                flush=True,
            )

    OUT.write_text(json.dumps({"doc_ids": doc_ids, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(1 for r in results if r["score"]["ok"] and r["status"] < 300)
    print(f"SCORE {passed}/{len(results)} wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

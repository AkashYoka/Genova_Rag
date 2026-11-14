"""Smoke non-medical PDFs through the same /chat/ask path as the Xolair battery."""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent / "_smoke_nonmedical_iterative.json"
BASE = "http://127.0.0.1:7860"

USER_DOCS = [
    {
        "name": "trading",
        "document_id": "3e8616b1-baaa-4648-89d6-a3669c11d35f",
        "questions": [
            "What tables or figures are present in this document, and what do they measure?",
            "Extract one concrete numeric value from a table, including units and page if available.",
        ],
    },
    {
        "name": "research",
        "document_id": "14d09766-cb93-4f4e-9232-27c75b42a971",
        "questions": [
            "Summarize a key quantitative finding from a table or results section with citations.",
            "If the document lacks enough evidence for a specific medical dosage recommendation, say so explicitly.",
        ],
    },
    {
        "name": "chemistry",
        "document_id": "59118577-e022-4b09-974f-764933e2a4cf",
        "questions": [
            "What quantitative values or properties appear in tables? Cite page and units.",
        ],
    },
    {
        "name": "algebra",
        "document_id": "6df2ad7b-4944-4e55-96c9-f5f00374a932",
        "questions": [
            "Is there a dosing or medical prescribing table in this document? If not, say insufficient evidence.",
        ],
    },
]


def main() -> None:
    results = []
    with httpx.Client(base_url=BASE, timeout=180.0) as client:
        tok = client.post(
            "/api/v1/auth/login",
            json={"email": "dev@genova.ai", "password": "localdev123"},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {tok}"}

        for doc in USER_DOCS:
            sess = client.post(
                "/api/v1/chat/sessions",
                headers=headers,
                json={
                    "title": f"Nonmed smoke: {doc['name']}",
                    "document_id": doc["document_id"],
                },
            )
            sid = (sess.json() or {}).get("id")
            for q in doc["questions"]:
                t0 = time.perf_counter()
                r = client.post(
                    "/api/v1/chat/ask",
                    headers=headers,
                    json={
                        "question": q,
                        "document_id": doc["document_id"],
                        "llm_provider": "bedrock",
                        "session_id": sid,
                    },
                )
                elapsed = round(time.perf_counter() - t0, 2)
                data = r.json() if r.status_code < 500 else {"error": r.text[:500]}
                ans = data.get("answer") or ""
                low = ans.lower()
                results.append(
                    {
                        "doc": doc["name"],
                        "document_id": doc["document_id"],
                        "question": q,
                        "status": r.status_code,
                        "elapsed_s": elapsed,
                        "session_id": sid,
                        "ans_head": ans[:600],
                        "mentions_insufficient": "insufficient" in low
                        or "not enough" in low
                        or "no evidence" in low
                        or "does not contain" in low,
                        "has_citation_or_page": "page" in low or "source" in low,
                        "error": None if r.status_code < 300 else data,
                    }
                )
                print(
                    f"{doc['name']}: {r.status_code} {elapsed}s :: "
                    f"{ans[:140].replace(chr(10), ' ')}",
                    flush=True,
                )

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in results if r["status"] < 300)
    print(f"wrote {OUT} ok={ok}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()

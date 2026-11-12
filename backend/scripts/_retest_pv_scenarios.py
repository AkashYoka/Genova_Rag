"""Clinical PV-style questions via the same /chat/ask path the frontend uses."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

DOC = "43ae024f-3c0d-40f5-aa49-83e9f3682641"
OUT = Path(__file__).with_name("_e2e_xolair_pv_frontend_scenarios.json")

QUESTIONS = [
    (
        "A 35-year-old patient with allergic asthma, body weight 78 kg and pretreatment "
        "IgE 350 IU/mL was intentionally prescribed Xolair 150 mg every 4 weeks. "
        "No adverse event was reported. Can you check this and identify any adverse event, "
        "special situation, or off-label usage?"
    ),
    (
        "A 42-year-old with allergic asthma, weight 80 kg, IgE 250 IU/mL was prescribed "
        "Xolair 75 mg every 4 weeks. No AE reported. Is this an adverse event, special "
        "situation, medication error, or off-label use?"
    ),
    (
        "A 30-year-old with allergic asthma weighs 65 kg with pretreatment IgE 350 IU/mL "
        "and was prescribed Xolair 225 mg every 2 weeks. No adverse event. Classify AE / "
        "special situation / off-label if any."
    ),
    (
        "A 55-year-old with COPD (not asthma), IgE 200 IU/mL, weight 70 kg was prescribed "
        "Xolair 300 mg every 4 weeks. No AE. Identify adverse event, special situation, "
        "or off-label usage."
    ),
    (
        "Patient 12+ years, allergic asthma, IgE 350 IU/mL, weight 80 kg: label dose is "
        "300 mg every 2 weeks, but they received 300 mg every 4 weeks intentionally. "
        "No AE. Classify."
    ),
    (
        "An 8-year-old with asthma, weight 30 kg, pretreatment IgE 300 IU/mL received the "
        "pediatric table dose. No AE. Any off-label or special situation?"
    ),
    (
        "A 40-year-old asthmatic, 50 kg, IgE 350 IU/mL received Xolair 600 mg every 2 weeks "
        "intentionally. No adverse event. Flag off-label / special situation / AE if present."
    ),
    (
        "What are the key anaphylaxis warnings for XOLAIR from the prescribing information?"
    ),
]


def flags(answer: str) -> dict:
    low = (answer or "").lower()
    return {
        "off_label": bool(re.search(r"off[- ]?label", low)),
        "special_situation": bool(
            re.search(
                r"special situation|medication error|underdos|overdos|dosing (error|discrepan)|not label[- ]concordant",
                low,
            )
        ),
        "adverse_event": bool(re.search(r"adverse event|\bno ae\b|no adverse", low)),
        "mentions_dose": bool(re.search(r"\b\d{2,4}\s*mg\b", low)),
        "bad_verified_banner": bool(
            re.search(r"authoritative verified table value:.*≥?\s*30[- ]?100", low, re.S)
        ),
        "label_concordant_claim": bool(re.search(r"label[- ]concordant", low)),
    }


def main() -> None:
    c = httpx.Client(base_url="http://127.0.0.1:7860", timeout=180.0)
    tok = c.post(
        "/api/v1/auth/login",
        json={"email": "dev@genova.ai", "password": "localdev123"},
    ).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        sess = c.post(
            "/api/v1/chat/sessions",
            headers=h,
            json={"title": f"pv scenario {i}", "document_id": DOC},
        )
        sid = (sess.json() or {}).get("id") if sess.status_code < 300 else None
        body = {"question": q, "document_id": DOC, "llm_provider": "bedrock"}
        if sid:
            body["session_id"] = sid
        t0 = time.perf_counter()
        r = c.post("/api/v1/chat/ask", headers=h, json=body)
        elapsed = time.perf_counter() - t0
        data = r.json() if r.status_code < 300 else {"error": r.text[:400]}
        ans = data.get("answer") or ""
        row = {
            "i": i,
            "question": q,
            "elapsed_s": round(elapsed, 2),
            "status": r.status_code,
            "flags": flags(ans),
            "ans_head": ans[:1100],
            "error": data.get("error"),
        }
        results.append(row)
        print(
            f"{i}/{len(QUESTIONS)} {elapsed:.1f}s "
            f"offlabel={row['flags']['off_label']} "
            f"special={row['flags']['special_situation']} "
            f"bad_banner={row['flags']['bad_verified_banner']}"
        )
        print(" ", ans[:220].replace("\n", " "))

    OUT.write_text(
        json.dumps(
            {"document_id": DOC, "n": len(results), "results": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()

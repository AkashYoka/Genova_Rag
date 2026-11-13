"""Additional Xolair PV classify questions via /chat/ask (frontend path)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

DOC = "43ae024f-3c0d-40f5-aa49-83e9f3682641"
OUT = Path(__file__).resolve().parent / "_xolair_pv_batch2.json"

CASES = [
    {
        "id": 1,
        "question": (
            "A 35-year-old patient with allergic asthma, body weight 78 kg and "
            "pretreatment IgE 350 IU/mL was intentionally prescribed Xolair 150 mg "
            "every 4 weeks. No adverse event was reported. Identify any adverse "
            "event, special situation, or off-label usage."
        ),
        "expect": {
            "label_dose": "300",
            "label_freq_2w": True,
            "ae_no": True,
            "off_label_yes": True,
            "ss_yes": True,
            "med_error_no": True,
        },
    },
    {
        "id": 2,
        "question": (
            "Adult allergic asthma, weight 80 kg, pretreatment IgE 250 IU/mL. "
            "Intentionally prescribed Xolair 75 mg every 4 weeks. No AE reported. "
            "Classify AE, off-label use, special situation, and medication error."
        ),
        "expect": {
            "label_dose": "225",
            "label_freq_4w": True,
            "ae_no": True,
            "off_label_yes": True,
            "ss_yes": True,
            "med_error_no": True,
        },
    },
    {
        "id": 3,
        "question": (
            "A 40-year-old with allergic asthma, 50 kg, IgE 350 IU/mL received "
            "600 mg every 2 weeks intentionally. No adverse event. Identify AE / "
            "special situation / off-label use / medication error."
        ),
        "expect": {
            "label_dose": "225",
            "ae_no": True,
            "off_label_yes": True,
            "ss_yes": True,
            "med_error_no": True,
        },
    },
    {
        "id": 4,
        "question": (
            "Adult asthma, IgE 350 IU/mL, weight 80 kg. Label supports 300 mg every "
            "2 weeks but they intentionally got 300 mg every 4 weeks. No AE. "
            "Classify AE, off-label, special situation, medication error."
        ),
        "expect": {
            "ae_no": True,
            "off_label_yes": True,
            "ss_yes": True,
            "med_error_no": True,
        },
    },
    {
        "id": 5,
        "question": (
            "COPD patient (not asthma), IgE 200 IU/mL, weight 70 kg, Xolair 300 mg "
            "every 4 weeks intentionally. No AE. Off-label / special situation / AE / "
            "medication error?"
        ),
        "expect": {
            "ae_no": True,
            "off_label_yes": True,
            "ss_yes": True,
            "med_error_no": True,
        },
    },
    {
        "id": 6,
        "question": (
            "8-year-old with asthma, 30 kg, IgE 300 IU/mL received the pediatric "
            "table dose of 150 mg every 4 weeks as labeled. No AE. Any off-label "
            "use or special situation?"
        ),
        "expect": {
            "label_dose": "150",
            "ae_no": True,
            "off_label_no": True,
            "concordant": True,
        },
    },
    {
        "id": 7,
        "question": (
            "An 8-year-old with asthma, 30 kg, IgE 300 IU/mL was intentionally given "
            "an adult Table 1 regimen of 300 mg every 2 weeks. No AE. Classify AE / "
            "off-label / special situation / medication error."
        ),
        "expect": {
            "ae_no": True,
            "off_label_yes": True,
            "ss_yes": True,
            "med_error_no": True,
        },
    },
    {
        "id": 8,
        "question": (
            "Patient with chronic spontaneous urticaria (CSU), weight 80 kg, IgE "
            "350 IU/mL was intentionally dosed using the asthma IgE/weight table. "
            "No AE. Is that off-label? Classify AE / special situation / medication error."
        ),
        "expect": {
            "ae_no": True,
            "off_label_yes": True,
            "ss_yes": True,
            "med_error_no": True,
        },
    },
    {
        "id": 9,
        "question": (
            "Asthma patient >=12 years, IgE 300 IU/mL, weight 70 kg was intentionally "
            "prescribed Xolair 225 mg every 4 weeks. No AE. Does this match the label? "
            "Classify AE / off-label / special situation / medication error."
        ),
        "expect": {
            "label_dose": "225",
            "label_freq_4w": True,
            "ae_no": True,
            "off_label_no": True,
            "concordant": True,
        },
    },
    {
        "id": 10,
        "question": (
            "A 28-year-old with allergic asthma, weight 55 kg, pretreatment IgE "
            "150 IU/mL was intentionally prescribed Xolair 150 mg every 4 weeks. "
            "No AE. Identify AE, special situation, or off-label usage."
        ),
        "expect": {
            # >100-200 × 30-60 kg → 300 mg q4w
            "label_dose": "300",
            "label_freq_4w": True,
            "ae_no": True,
            "off_label_yes": True,
            "ss_yes": True,
            "med_error_no": True,
        },
    },
    {
        "id": 11,
        "question": (
            "Adult allergic asthma, 90 kg, IgE 450 IU/mL intentionally prescribed "
            "Xolair 375 mg every 2 weeks. No AE. Classify AE / off-label / special "
            "situation / medication error."
        ),
        "expect": {
            # >400-500 × >70-90 → 375 q2w — should be concordant
            "label_dose": "375",
            "ae_no": True,
            "off_label_no": True,
            "concordant": True,
        },
    },
    {
        "id": 12,
        "question": (
            "A 45-year-old with allergic asthma, weight 78 kg, IgE 350 IU/mL developed "
            "anaphylaxis after an injection while on the labeled 300 mg every 2 weeks "
            "regimen. Classify AE / off-label / special situation / medication error."
        ),
        "expect": {
            "ae_yes": True,
            "off_label_no": True,
            "concordant": True,
        },
    },
]


def _low(ans: str) -> str:
    return (ans or "").lower()


def _has(ans: str, *pats: str) -> bool:
    low = _low(ans)
    return any(re.search(p, low) for p in pats)


def score(ans: str, expect: dict) -> dict:
    checks = {}
    pass_n = total = 0

    def ok(name: str, cond: bool) -> None:
        nonlocal pass_n, total
        total += 1
        checks[name] = bool(cond)
        if cond:
            pass_n += 1

    if "label_dose" in expect:
        ok(
            "label_dose",
            bool(re.search(rf"\b{re.escape(expect['label_dose'])}\s*mg\b", _low(ans))),
        )
    if expect.get("label_freq_2w"):
        ok("freq_2w", bool(re.search(r"every\s*2\s*weeks?", _low(ans))))
    if expect.get("label_freq_4w"):
        ok("freq_4w", bool(re.search(r"every\s*4\s*weeks?", _low(ans))))

    if expect.get("ae_no"):
        ok(
            "ae_no",
            _has(
                ans,
                r"adverse event.{0,40}(no|none|not reported)",
                r"(no|none|not reported).{0,40}adverse event",
                r"\bae\b.{0,20}(no|none|not)",
            )
            and not _has(ans, r"adverse event.{0,20}\byes\b"),
        )
    if expect.get("ae_yes"):
        ok(
            "ae_yes",
            _has(
                ans,
                r"adverse event.{0,30}\byes\b",
                r"\bae\b.{0,20}\byes\b",
                r"anaphylaxis",
            ),
        )

    if expect.get("off_label_yes"):
        ok(
            "off_label_yes",
            _has(
                ans,
                r"off[- ]?label.{0,40}(yes|identified|present)",
                r"(yes|identified).{0,40}off[- ]?label",
                r"off[- ]?label use",
            ),
        )
    if expect.get("off_label_no"):
        ok(
            "off_label_no",
            _has(
                ans,
                r"off[- ]?label.{0,40}(no|none|not)",
                r"not off[- ]?label",
                r"no off[- ]?label",
                r"within (the )?(approved|labeled)",
                r"matches the label|label[- ]concordant|concordant",
            )
            and not _has(ans, r"off[- ]?label.{0,20}\byes\b"),
        )

    if expect.get("ss_yes"):
        ok(
            "ss_yes",
            _has(
                ans,
                r"special situation.{0,40}(yes|identified|present)",
                r"(yes|identified).{0,40}special situation",
            ),
        )

    if expect.get("med_error_no"):
        ok(
            "med_error_no",
            _has(
                ans,
                r"medication error.{0,40}(no|none|not)",
                r"not (automatically )?a medication error",
                r"not.{0,20}medication error",
            )
            or not _has(ans, r"medication error.{0,20}\byes\b"),
        )

    if expect.get("concordant"):
        ok(
            "concordant",
            _has(
                ans,
                r"match(es)? the label|label[- ]concordant|as labeled|"
                r"within (the )?label|recommended regimen|concordant|"
                r"no off[- ]?label|not off[- ]?label",
            ),
        )

    return {"pass": pass_n, "total": total, "ok": pass_n == total and total > 0, "checks": checks}


def main() -> None:
    c = httpx.Client(base_url="http://127.0.0.1:7860", timeout=240.0)
    tok = c.post(
        "/api/v1/auth/login",
        json={"email": "dev@genova.ai", "password": "localdev123"},
    ).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    sid = c.post(
        "/api/v1/chat/sessions",
        headers=h,
        json={"title": "Xolair PV batch2 classify", "document_id": DOC},
    ).json()["id"]
    print(f"session={sid} n={len(CASES)}", flush=True)

    results = []
    for case in CASES:
        t0 = time.perf_counter()
        r = c.post(
            "/api/v1/chat/ask",
            headers=h,
            json={
                "question": case["question"],
                "document_id": DOC,
                "llm_provider": "bedrock",
                "session_id": sid,
            },
        )
        elapsed = time.perf_counter() - t0
        data = r.json() if r.status_code < 300 else {"error": r.text[:400]}
        ans = data.get("answer") or ""
        sc = score(ans, case["expect"])
        results.append(
            {
                "id": case["id"],
                "elapsed_s": round(elapsed, 2),
                "status": r.status_code,
                "score": sc,
                "question": case["question"],
                "ans_head": ans[:1600],
                "error": data.get("error"),
            }
        )
        mark = "PASS" if sc["ok"] else "FAIL"
        print(
            f"{case['id']:02d}/{len(CASES)} {mark} {sc['pass']}/{sc['total']} "
            f"{elapsed:.1f}s {sc['checks']}",
            flush=True,
        )
        print(" ", ans[:200].replace("\n", " "), flush=True)

    passed = sum(1 for x in results if x["score"]["ok"])
    report = {
        "document_id": DOC,
        "session_id": sid,
        "passed": passed,
        "total": len(results),
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSCORE {passed}/{len(results)} session={sid}", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

"""Run 12 Xolair PV/classify questions via the same /chat/ask path the frontend uses."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

DOC = "43ae024f-3c0d-40f5-aa49-83e9f3682641"
OUT = Path(__file__).resolve().parent / "_xolair_pv_frontend_battery.json"

# Focused AE / special-situation / off-label / regimen battery (12 qs).
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
            "label_freq": "2 week",
            "ae": False,
            "off_label_indication": False,
            "dosing_deviation": True,
        },
    },
    {
        "id": 2,
        "question": (
            "According to Table 1 for patients 12 years and older with asthma, what "
            "XOLAIR dose and frequency apply for pretreatment IgE 350 IU/mL and body "
            "weight 78 kg?"
        ),
        "expect": {"label_dose": "300", "label_freq": "2 week"},
    },
    {
        "id": 3,
        "question": (
            "Asthma patient >=12 years, IgE 300 IU/mL, weight 70 kg. What is the "
            "label dose and frequency from the dosing table?"
        ),
        "expect": {"label_dose": "225", "label_freq": "4 week"},
    },
    {
        "id": 4,
        "question": (
            "A 40-year-old with allergic asthma, 50 kg, IgE 350 IU/mL received "
            "600 mg every 2 weeks intentionally. No AE. Classify AE / special "
            "situation / off-label / medication error."
        ),
        "expect": {
            "ae": False,
            "off_label_indication": False,
            "dosing_deviation": True,
            "label_dose": "225",
        },
    },
    {
        "id": 5,
        "question": (
            "Adult asthma, 80 kg, IgE 250 IU/mL prescribed 75 mg every 4 weeks. "
            "No AE. Classify AE / special situation / off-label / dosing deviation."
        ),
        "expect": {
            "ae": False,
            "off_label_indication": False,
            "dosing_deviation": True,
            "label_dose": "225",
        },
    },
    {
        "id": 6,
        "question": (
            "Adult asthma, IgE 350 IU/mL, weight 80 kg: label is 300 mg every 2 weeks "
            "but they got 300 mg every 4 weeks intentionally. No AE. Classify."
        ),
        "expect": {
            "ae": False,
            "off_label_indication": False,
            "dosing_deviation": True,
        },
    },
    {
        "id": 7,
        "question": (
            "COPD patient (not asthma), IgE 200 IU/mL, weight 70 kg, Xolair 300 mg "
            "every 4 weeks. No AE. Off-label / special situation / AE?"
        ),
        "expect": {"ae": False, "off_label_indication": True},
    },
    {
        "id": 8,
        "question": (
            "8-year-old asthma, 30 kg, IgE 300 IU/mL received the pediatric table "
            "dose. No AE. Any off-label or special situation?"
        ),
        "expect": {
            "ae": False,
            "off_label_indication": False,
            "dosing_deviation": False,
        },
    },
    {
        "id": 9,
        "question": (
            "An 8-year-old with asthma, 30 kg, IgE 300 IU/mL was given an adult "
            "Table 1 regimen of 300 mg every 2 weeks. No AE. Is that appropriate?"
        ),
        "expect": {"ae": False, "dosing_deviation": True, "wrong_table": True},
    },
    {
        "id": 10,
        "question": (
            "Patient with chronic spontaneous urticaria (CSU), weight 80 kg, IgE "
            "350 IU/mL was dosed using the asthma IgE/weight table. Is that correct "
            "per the label? Classify AE / special situation / off-label / dosing."
        ),
        "expect": {"wrong_table": True, "dosing_deviation": True},
    },
    {
        "id": 11,
        "question": (
            "For asthma Table 1 patients >=12 years, IgE about 800 IU/mL and weight "
            "about 140 kg: does the table recommend a dose or show insufficient data?"
        ),
        "expect": {"no_labeled_dose": True},
    },
    {
        "id": 12,
        "question": (
            "A 55-year-old with allergic asthma, body weight 90 kg, pretreatment "
            "IgE 400 IU/mL was intentionally prescribed Xolair 150 mg every 4 weeks. "
            "No adverse event was reported. Identify any adverse event, special "
            "situation, or off-label usage."
        ),
        "expect": {
            "ae": False,
            "off_label_indication": False,
            "dosing_deviation": True,
        },
    },
]


def _low(ans: str) -> str:
    return (ans or "").lower()


def score(ans: str, expect: dict) -> dict:
    low = _low(ans)
    checks = {}
    pass_n = 0
    total = 0

    def ok(name: str, cond: bool) -> None:
        nonlocal pass_n, total
        total += 1
        checks[name] = bool(cond)
        if cond:
            pass_n += 1

    if "label_dose" in expect:
        dose = expect["label_dose"]
        ok(
            "label_dose",
            bool(re.search(rf"\b{re.escape(dose)}\s*mg\b", low)),
        )
    if "label_freq" in expect:
        if "2 week" in expect["label_freq"]:
            ok("label_freq", bool(re.search(r"every\s*2\s*weeks?", low)))
        elif "4 week" in expect["label_freq"]:
            ok("label_freq", bool(re.search(r"every\s*4\s*weeks?", low)))

    if "ae" in expect and expect["ae"] is False:
        invented_ae = bool(
            re.search(
                r"(adverse event|\bae\b).{0,40}(yes|present|identified|occurred)",
                low,
            )
        ) and not bool(
            re.search(
                r"(no|none|not\s+reported).{0,40}(adverse event|\bae\b)"
                r"|(adverse event|\bae\b).{0,40}(no|none|not reported|absent)",
                low,
            )
        )
        ok("ae_none", not invented_ae)

    if expect.get("off_label_indication") is True:
        ok(
            "off_label_yes",
            bool(re.search(r"off[- ]?label|not (an )?approved|not indicated", low)),
        )
    elif expect.get("off_label_indication") is False:
        # Should not claim off-label indication for approved asthma
        strong_off = bool(
            re.search(
                r"off[- ]?label (indication|use|usage).{0,20}(yes|present)"
                r"|indication.{0,30}off[- ]?label",
                low,
            )
        ) and not re.search(
            r"not off[- ]?label|off[- ]?label.{0,20}(no|none|not)",
            low,
        )
        ok("not_off_label_indication", not strong_off or re.search(
            r"off[- ]?label.{0,40}(no|not)|not (categorically )?off[- ]?label",
            low,
        ) is not None)

    if expect.get("dosing_deviation") is True:
        ok(
            "dosing_deviation",
            bool(
                re.search(
                    r"not (label[- ]?)?concordant|does not match|discrepan|"
                    r"inappropriate|medication error|under[- ]?dos|over[- ]?dos|"
                    r"deviation|incorrect|not appropriate|not the (recommended|labeled)",
                    low,
                )
            ),
        )
    elif expect.get("dosing_deviation") is False:
        ok(
            "no_dosing_issue",
            bool(
                re.search(
                    r"match(es)? the label|label[- ]concordant|appropriate|"
                    r"no off[- ]?label|no (special situation|ae)",
                    low,
                )
            )
            and not bool(
                re.search(r"medication error|not concordant|does not match", low)
            ),
        )

    if expect.get("wrong_table"):
        ok(
            "wrong_table",
            bool(
                re.search(
                    r"wrong|incorrect|not (the )?(correct|appropriate)|"
                    r"should not|pediatric|csu|not (use|using) the asthma|"
                    r"fixed dose|do not (use|dose)",
                    low,
                )
            ),
        )

    if expect.get("no_labeled_dose"):
        ok(
            "no_labeled_dose",
            bool(
                re.search(
                    r"insufficient|do not dose|no (labeled|recommended) dose|"
                    r"not recommend|blank|no dose",
                    low,
                )
            ),
        )

    return {
        "pass": pass_n,
        "total": total,
        "ok": pass_n == total and total > 0,
        "checks": checks,
    }


def main() -> None:
    c = httpx.Client(base_url="http://127.0.0.1:7860", timeout=240.0)
    tok = c.post(
        "/api/v1/auth/login",
        json={"email": "dev@genova.ai", "password": "localdev123"},
    ).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}

    sess = c.post(
        "/api/v1/chat/sessions",
        headers=h,
        json={
            "title": "Xolair PV battery (frontend path)",
            "document_id": DOC,
        },
    )
    sid = (sess.json() or {}).get("id")
    assert sid, sess.text
    print(f"session={sid} doc={DOC} n={len(CASES)}", flush=True)

    results = []
    for case in CASES:
        q = case["question"]
        t0 = time.perf_counter()
        r = c.post(
            "/api/v1/chat/ask",
            headers=h,
            json={
                "question": q,
                "document_id": DOC,
                "llm_provider": "bedrock",
                "session_id": sid,
            },
        )
        elapsed = time.perf_counter() - t0
        data = r.json() if r.status_code < 300 else {"error": r.text[:500]}
        ans = data.get("answer") or ""
        sc = score(ans, case["expect"])
        row = {
            "id": case["id"],
            "elapsed_s": round(elapsed, 2),
            "status": r.status_code,
            "score": sc,
            "question": q,
            "ans_head": ans[:1500],
            "error": data.get("error"),
        }
        results.append(row)
        mark = "PASS" if sc["ok"] else "FAIL"
        print(
            f"{case['id']:02d}/{len(CASES)} {mark} {sc['pass']}/{sc['total']} "
            f"{elapsed:.1f}s checks={sc['checks']}",
            flush=True,
        )
        print(" ", ans[:220].replace("\n", " "), flush=True)

    passed = sum(1 for r in results if r["score"]["ok"])
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
    print(f"frontend: http://localhost:3000/chat?session={sid}", flush=True)


if __name__ == "__main__":
    main()

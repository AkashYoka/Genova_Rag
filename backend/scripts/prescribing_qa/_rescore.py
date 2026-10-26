"""Manual clinical re-score of UI verification results against label facts."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
QA = json.loads((HERE / "ui_verification_qa.json").read_text(encoding="utf-8"))
RES = json.loads((HERE / "ui_verification_results.json").read_text(encoding="utf-8"))

UI_Q1 = """Key finding: The XOLAIR prescribing information recommends every-2-week dosing for Chronic Spontaneous Urticaria (CSU).

Label recommendation — Dose: 150 mg or 300 mg by subcutaneous injection; Frequency: Every 4 weeks; Key limits: Dosing is not dependent on serum IgE level or body weight.

Based on the information provided in the prescribing information, the prescription of XOLAIR 150 mg subcutaneously every 2 weeks does not match the label recommendation. The label recommends an every-4-week dosing schedule for CSU patients.

Proposed: XOLAIR 150 mg subcutaneously every 2 weeks
Label recommendation: XOLAIR 150 mg or 300 mg by subcutaneous injection every 4 weeks"""


def low(s: str) -> str:
    return (s or "").lower()


def has_any(text: str, phrases: list[str]) -> bool:
    t = low(text)
    return any(p.lower() in t for p in phrases)


# Manual judgments grounded in label + actual answers
MANUAL = {
    "UI-PV-01": {
        "actual": UI_Q1,
        "channel": "dashboard_ui",
        "content_pass": True,  # correctly concludes does not match / every 4 weeks
        "citation_pass": False,  # no [Source:] / page chips in rendered UI answer
        "rendering": "Proposed/Label structure present; Key finding incorrectly says every-2-week (contradicts body)",
        "pass": False,  # fail overall due to missing citations + contradictory key finding
        "notes": "Body correct; Key finding wrong; no visible citations in UI",
    },
    "UI-PV-02": {
        "content_pass": True,  # 6 years / does not cover 4yo
        "citation_pass": True,
        "pass": True,
        "notes": None,
    },
    "UI-PV-03": {
        "content_pass": True,  # IV does not match; SC only
        "citation_pass": True,
        "pass": True,
        "notes": None,
    },
    "UI-PV-04": {
        "content_pass": True,  # not indicated for status asthmaticus
        "citation_pass": True,
        "pass": True,
        "notes": None,
    },
    "UI-PV-05": {
        "content_pass": True,  # CRSwNP add-on matches
        "citation_pass": True,
        "pass": True,
        "notes": None,
    },
    "UI-PV-06": {
        # auto said miss 75/600 — verify from actual
        "content_pass": None,
        "citation_pass": True,
        "pass": None,
        "notes": "recheck dose range 75-600",
    },
    "UI-PV-07": {
        "content_pass": True,  # first dose and beyond 1 year
        "citation_pass": True,
        "pass": True,
        "notes": "Cited page 13; boxed warning also on page 1",
    },
    "UI-PV-08": {
        "content_pass": False,  # said one dose; label says at least 3
        "citation_pass": True,
        "pass": False,
        "notes": "WRONG: answered 'no more than one dose'; label requires at least 3 supervised doses",
    },
    "UI-PV-09": {
        "content_pass": True,  # CSU not IgE/weight based
        "citation_pass": True,
        "pass": True,
        "notes": None,
    },
    "UI-PV-10": {
        "content_pass": None,  # recheck — anaphylactic vs severe hypersensitivity
        "citation_pass": True,
        "pass": None,
        "notes": "recheck contraindication wording",
    },
    "UI-PV-11": {
        "content_pass": True,  # no formal DDI studies
        "citation_pass": True,
        "pass": True,
        "notes": "Cited page 30; section text is on page 23 of PDF extract — page may be off by numbering",
    },
    "UI-PV-12": {
        "content_pass": True,  # does not match; avoidance; not emergency
        "citation_pass": True,
        "pass": True,
        "notes": None,
    },
    "UI-PV-13": {
        "content_pass": False,  # said no monitoring; label says Monitor high-risk geohelminth
        "citation_pass": True,  # has source but wrong page/content
        "pass": False,
        "notes": "WRONG: label says Monitor patients at high risk of geohelminth infection",
    },
    "UI-PV-14": {
        "content_pass": None,  # does not match OK; need SC every 2 or 4 in label section
        "citation_pass": True,
        "pass": None,
        "notes": "recheck whether SC every 2/4 weeks stated",
    },
}

by_id = {r["id"]: r for r in RES["results"]}
qa_by = {q["id"]: q for q in QA["questions"]}

for qid, m in MANUAL.items():
    r = by_id[qid]
    actual = m.get("actual") or r.get("actual") or ""
    if qid == "UI-PV-06":
        ok = has_any(actual, ["75"]) and has_any(actual, ["600"]) and has_any(
            actual, ["every 2 or 4", "every 2 or every 4", "every 2–4", "2 or 4 weeks"]
        )
        # if missing exact 75-600, still pass if says based on IgE/weight every 2 or 4 SC?
        m["content_pass"] = ok
        m["pass"] = ok and True
        if not ok:
            m["notes"] = "Missing explicit 75–600 mg range (said body-weight based every 2 or 4 weeks only)"
            m["pass"] = False
    if qid == "UI-PV-10":
        ok = has_any(actual, ["contraindicated"]) and has_any(
            actual, ["hypersensitivity", "anaphylactic", "anaphylaxis", "severe"]
        )
        # reject if key finding says NOT contraindicated incorrectly as main answer
        contradictory = "is not contraindicat" in low(actual) and "is contraindicated" not in low(actual).split("not contraindicat")[0][-80:]
        # simpler: require positive contraindicated statement
        ok = "contraindicated in" in low(actual) or "is contraindicated" in low(actual)
        ok = ok and has_any(actual, ["hypersensitivity", "anaphylactic", "anaphylaxis", "component"])
        m["content_pass"] = ok
        m["pass"] = ok
        if "is not contraindicat" in low(actual):
            m["notes"] = "Contains conflicting 'not contraindicated' language; check carefully"
    if qid == "UI-PV-14":
        mismatch = has_any(actual, ["does not match", "does not"])
        route = has_any(actual, ["subcutaneous", "sc injection", "subcutaneously"])
        freq = has_any(actual, ["every 2 or 4", "every 2 or every 4", "2 or 4 weeks"])
        oral_wrong = has_any(actual, ["by mouth", "oral", "once daily"])
        m["content_pass"] = mismatch and (route or freq)
        m["pass"] = bool(m["content_pass"])
        if mismatch and not (route and freq):
            m["notes"] = "Correctly rejects oral daily but incomplete label dose/route/frequency detail"
            # still pass if mismatch asserted and some label detail
            m["pass"] = mismatch and (route or freq)
            m["content_pass"] = m["pass"]

    r["channel"] = m.get("channel") or r.get("channel") or "api_into_ui_session"
    if m.get("actual"):
        r["actual"] = m["actual"]
    r["content_pass"] = m["content_pass"]
    r["citation_pass"] = m["citation_pass"]
    r["pass"] = m["pass"]
    r["expected"] = qa_by[qid]["expected"]
    r["question"] = qa_by[qid]["question"]
    r["manual_notes"] = m.get("notes")
    if m.get("rendering"):
        r["rendering"] = m["rendering"]
    cite = r.get("citations") or {}
    # citation accuracy vs expected pages
    expected_pages = set(qa_by[qid].get("source_pages") or [])
    got = set(cite.get("inline_citation_pages") or []) | set(cite.get("source_panel_pages") or [])
    r["citation_page_overlap"] = sorted(expected_pages & got)
    r["citation_accuracy"] = (
        "none visible"
        if qid == "UI-PV-01"
        else (
            "sources returned; page overlap "
            + str(r["citation_page_overlap"] or "none with expected")
            + f"; source_count={cite.get('source_count')}"
        )
    )

passed = sum(1 for r in RES["results"] if r.get("pass") is True)
failed = sum(1 for r in RES["results"] if r.get("pass") is False)
RES["answered"] = f"{passed}/{len(RES['results'])}"
RES["summary"] = {
    "pass": passed,
    "fail": failed,
    "total": len(RES["results"]),
    "indexed_prescribing_ready": ["xolair_prescribing.pdf"],
    "indexed_prescribing_not_ready": [
        "lucentis_prescribing.pdf",
        "ocrevus_prescribing.pdf",
        "rozlytrek_prescribing.pdf",
        "avastin_prescribing.pdf",
    ],
    "session_id": RES["session_id"],
    "preserved_sessions": [
        "calculus-volume-1_-_WEB-1-300.pdf",
        "UPSC Prelims Complete Guide",
        "Chemistry - Learning Resources",
        "Math - Calculus Volume 1",
        "Invoice Related",
    ],
}

(HERE / "ui_verification_results.json").write_text(json.dumps(RES, indent=2), encoding="utf-8")

print(f"FINAL {passed}/{len(RES['results'])} PASS\n")
for r in RES["results"]:
    print(f"{r['id']} | {'PASS' if r.get('pass') else 'FAIL'} | content={r.get('content_pass')} cite={r.get('citation_pass')}")
    if r.get("manual_notes"):
        print(f"  note: {r['manual_notes']}")

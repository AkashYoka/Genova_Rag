"""Re-run the same single-session PV batteries for Xolair, Lucentis, Tecentriq Hybreza."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
OUT = HERE / "_e2e_three_pdfs_pv_battery.json"

# Same question sets as the previous single-session E2Es.
SOURCES = {
    "xolair": HERE / "_e2e_xolair_pv_extended_single_session.json",
    "lucentis": HERE / "_e2e_lucentis_pv_single_session.json",
    "tecentriq_hybreza": HERE / "_e2e_tecentriq_hybreza_pv_single_session.json",
}


def load_battery(name: str) -> dict:
    prev = json.loads(SOURCES[name].read_text(encoding="utf-8"))
    questions = [r["question"] for r in prev["results"]]
    return {
        "document_id": prev["document_id"],
        "title": f"{name} pv e2e after prompt improve",
        "questions": questions,
    }


def flags(ans: str) -> dict:
    low = (ans or "").lower()
    return {
        "insuff_prefix": low.lstrip().startswith("insufficient table evidence"),
        "not_concordant": bool(
            re.search(
                r"not label[- ]concordant|does not match|discrepan|inappropriate|"
                r"medication error|under[- ]?dos|over[- ]?dos|incorrect",
                low,
            )
        ),
        "matches": bool(
            re.search(
                r"label[- ]concordant|matches the label|matches the approved|recommended",
                low,
            )
        )
        and not bool(re.search(r"not label[- ]concordant|does not match", low)),
        "off_label": bool(re.search(r"off[- ]?label|not indicated|not an approved", low)),
        "bad_banner": bool(re.search(r"authoritative verified table value", low)),
    }


def run_battery(client: httpx.Client, headers: dict, name: str, cfg: dict) -> dict:
    doc = cfg["document_id"]
    sess = client.post(
        "/api/v1/chat/sessions",
        headers=headers,
        json={"title": cfg["title"], "document_id": doc},
    )
    sid = (sess.json() or {}).get("id")
    assert sid, sess.text
    print(f"\n=== {name} session={sid} doc={doc} n={len(cfg['questions'])}", flush=True)
    results = []
    for i, q in enumerate(cfg["questions"], 1):
        body = {
            "question": q,
            "document_id": doc,
            "llm_provider": "bedrock",
            "session_id": sid,
        }
        t0 = time.perf_counter()
        r = client.post("/api/v1/chat/ask", headers=headers, json=body)
        elapsed = time.perf_counter() - t0
        data = r.json() if r.status_code < 300 else {"error": r.text[:400]}
        ans = data.get("answer") or ""
        row = {
            "i": i,
            "elapsed_s": round(elapsed, 2),
            "status": r.status_code,
            "flags": flags(ans),
            "ans_head": ans[:1200],
            "question": q,
            "error": data.get("error"),
        }
        results.append(row)
        f = row["flags"]
        print(
            f"{i}/{len(cfg['questions'])} {elapsed:.1f}s "
            f"insuff={f['insuff_prefix']} match={f['matches']} "
            f"issue={f['not_concordant']} off={f['off_label']} banner={f['bad_banner']}",
            flush=True,
        )
        print(" ", ans[:180].replace("\n", " "), flush=True)
    return {"document_id": doc, "session_id": sid, "n": len(results), "results": results}


def main() -> None:
    c = httpx.Client(base_url="http://127.0.0.1:7860", timeout=180.0)
    tok = c.post(
        "/api/v1/auth/login",
        json={"email": "dev@genova.ai", "password": "localdev123"},
    ).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    report = {}
    for name in ("xolair", "lucentis", "tecentriq_hybreza"):
        cfg = load_battery(name)
        report[name] = run_battery(c, h, name, cfg)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

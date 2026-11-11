"""One-shot XOLAIR PV ask on a fresh session."""
from __future__ import annotations

import json
from pathlib import Path

import requests

BASE = "http://127.0.0.1:7860/api/v1"
DOC = "7cb23d1a-0c23-448c-993b-ac64b6ce97ed"
OUT = Path(__file__).resolve().parents[1] / "_xolair_pv_ask_once.json"
Q = (
    "Can you please check below content has any adverse event or special "
    "situations or any off-labels?\n"
    "A 35-year-old patient with allergic asthma, body weight 78 kg and "
    "pretreatment IgE 350 IU/mL was intentionally prescribed Xolair 150 mg "
    "every 4 weeks. No adverse event was reported."
)


def main() -> None:
    s = requests.Session()
    login = s.post(f"{BASE}/auth/dev-login", timeout=60)
    login.raise_for_status()
    token = login.json()["access_token"]
    csrf = s.cookies.get("csrf_token")
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if csrf:
        h["X-CSRF-Token"] = csrf
    sess = s.post(f"{BASE}/chat/sessions", headers=h, json={"title": "pv-once"}, timeout=60)
    sess.raise_for_status()
    sid = sess.json()["id"]
    resp = s.post(
        f"{BASE}/chat/ask",
        headers=h,
        json={
            "question": Q,
            "document_id": DOC,
            "session_id": sid,
            "llm_provider": "ollama",
        },
        timeout=600,
    )
    print("status", resp.status_code)
    data = resp.json()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(data.get("answer", ""))
    print("sources", len(data.get("sources") or []), "graph", bool(data.get("graph")))
    print("saved", OUT)


if __name__ == "__main__":
    main()

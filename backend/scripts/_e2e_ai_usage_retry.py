"""Retry the two failed e2e questions (use Ollama; optional Bedrock smoke)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

BASE = os.environ.get("GENOVA_API_BASE", "http://127.0.0.1:7860/api/v1")
DOCS = {
    "graph2": "27c3e634-7b00-47e1-bf72-b02e9de11b54",
    "graph1": "9809caa1-25a9-4d1c-ba6b-b70ed83e9945",
    "invoice_igst": "53b230c4-c480-4fbd-9e04-ab446b11c399",
}


def main() -> int:
    with httpx.Client(timeout=60.0) as client:
        login = client.post(f"{BASE}/auth/dev-login").json()
        headers = {"Authorization": f"Bearer {login['access_token']}"}
        session = client.post(
            f"{BASE}/chat/sessions",
            headers=headers,
            json={"title": "AI Usage E2E retry"},
        ).json()["id"]

        retries = [
            {
                "cat": "explanation",
                "document_id": DOCS["graph2"],
                "llm_provider": "ollama",
                "question": (
                    "Explain how Graph Retrieval-Augmented Generation (GraphRAG) differs "
                    "from traditional text-based RAG according to Graph2.pdf."
                ),
            },
            {
                "cat": "comparison",
                "document_ids": [DOCS["graph1"], DOCS["graph2"]],
                "llm_provider": "ollama",
                "question": (
                    "Compare Graph1.pdf and Graph2.pdf: how does each describe GraphRAG "
                    "relative to traditional RAG, and what problems such as hallucination "
                    "or relational knowledge gaps do they emphasize?"
                ),
            },
            {
                "cat": "bedrock_smoke",
                "document_id": DOCS["invoice_igst"],
                "llm_provider": "bedrock",
                "question": (
                    "What is the total amount on the CLOUDSCALE CONSULTING LLP invoice "
                    "to YENPU TECHNOLOGIES?"
                ),
            },
        ]

        outcomes = []
        for i, q in enumerate(retries, 1):
            payload = {
                "question": q["question"],
                "session_id": session,
                "top_k": 5,
                "llm_provider": q["llm_provider"],
            }
            if q.get("document_ids"):
                payload["document_ids"] = q["document_ids"]
            else:
                payload["document_id"] = q["document_id"]
            print(f"RETRY {i} provider={q['llm_provider']} cat={q['cat']}")
            try:
                r = client.post(
                    f"{BASE}/chat/ask", headers=headers, json=payload, timeout=300.0
                )
                print("status", r.status_code)
                if r.status_code >= 400:
                    print(r.text[:500])
                    outcomes.append({"ok": False, "provider": q["llm_provider"], "status": r.status_code, "body": r.text[:500]})
                    continue
                events = client.get(
                    f"{BASE}/usage/events", headers=headers, params={"limit": 3}
                ).json()["items"]
                latest = events[0] if events else None
                print("latest", json.dumps(latest, indent=2)[:600])
                outcomes.append({"ok": True, "provider": q["llm_provider"], "latest": latest})
            except Exception as exc:
                print("ERR", exc)
                outcomes.append({"ok": False, "provider": q["llm_provider"], "error": str(exc)})

        summary = client.get(f"{BASE}/usage/summary", headers=headers).json()
        print("SUMMARY", json.dumps({
            "request_count": summary["request_count"],
            "total_tokens": summary["total_tokens"],
            "estimated_cost_usd": summary["estimated_cost_usd"],
            "by_provider": summary["by_provider"],
        }, indent=2))
        out = Path(__file__).resolve().parents[1] / "data" / "e2e_ai_usage_retry.json"
        out.write_text(json.dumps(outcomes, indent=2, default=str), encoding="utf-8")
        print("wrote", out)
        return 0 if all(o.get("ok") or o.get("provider") == "bedrock" for o in outcomes[:2]) else 1


if __name__ == "__main__":
    sys.exit(main())

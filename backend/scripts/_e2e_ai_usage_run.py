"""End-to-end AI Usage verification with real RAG chat requests.

Uses authenticated localdev via /auth/dev-login and /chat/ask against indexed docs.
Does not mock responses or insert synthetic llm_usage rows.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

BASE = os.environ.get("GENOVA_API_BASE", "http://127.0.0.1:7860/api/v1")

# Indexed document IDs owned by localdev (from live DB /documents).
DOCS = {
    "note2": "74519cb8-16ed-4809-ac15-9f40eacf0cff",
    "backend_projects": "7190dc66-6eac-4709-9b1d-5ff620f2321e",
    "ai_roadmap": "b33b2964-821f-4429-b1e0-d1269b9ce302",
    "graph2": "27c3e634-7b00-47e1-bf72-b02e9de11b54",
    "graph1": "9809caa1-25a9-4d1c-ba6b-b70ed83e9945",
    "invoice_igst": "53b230c4-c480-4fbd-9e04-ab446b11c399",
    "calculus": "b479a191-6ceb-4bf1-a7d4-430adbc65a67",
    "xolair": "8c57d30c-9568-4437-b51f-b52d9ac495d1",
    "upsc": "79927b55-6a0c-4fd9-86c0-69dabc1e9a3a",
    "chemistry": "4f8679cf-6c64-4400-8661-bc57064dbdcf",
    "math_calc": "e4cc3e90-5659-4bb8-bd72-8c58f41e9482",
    "invoice_cgst": "1539a972-6afd-4424-aea6-92fb0fad375e",
}


def questions(bedrock_available: bool) -> list[dict[str, Any]]:
    """15 content-grounded questions across required categories."""
    qs: list[dict[str, Any]] = [
        # 5 factual
        {
            "cat": "factual",
            "document_id": DOCS["invoice_igst"],
            "question": (
                "What is the total invoice amount including taxes on the "
                "CLOUDSCALE CONSULTING LLP tax invoice issued to YENPU TECHNOLOGIES "
                "PRIVATE LIMITED?"
            ),
        },
        {
            "cat": "factual",
            "document_id": DOCS["calculus"],
            "question": (
                "Who are the authors of OpenStax Calculus Volume 1 according to "
                "the textbook front matter?"
            ),
        },
        {
            "cat": "factual",
            "document_id": DOCS["xolair"],
            "question": (
                "What serious risk highlighted in the prescribing information "
                "requires healthcare provider supervision when administering XOLAIR?"
            ),
        },
        {
            "cat": "factual",
            "document_id": DOCS["chemistry"],
            "question": (
                "List the contributing authors named for the OpenStax chemistry "
                "textbook on the authorship page."
            ),
        },
        {
            "cat": "factual",
            "document_id": DOCS["invoice_cgst"],
            "question": (
                "On the SRI VENKATESWARA ENTERPRISES invoice to YENPU TECHNOLOGIES, "
                "what is the grand total amount and when is payment due?"
            ),
        },
        # 3 explanation
        {
            "cat": "explanation",
            "document_id": DOCS["graph2"],
            "question": (
                "Explain how Graph Retrieval-Augmented Generation (GraphRAG) differs "
                "from traditional text-based RAG according to Graph2.pdf."
            ),
        },
        {
            "cat": "explanation",
            "document_id": DOCS["xolair"],
            "question": (
                "Explain why XOLAIR administration requires healthcare provider "
                "supervision and which acute conditions it is not indicated for."
            ),
        },
        {
            "cat": "explanation",
            "document_id": DOCS["note2"],
            "question": (
                "Explain the three methods Query Expansion, Subgraph Retrieval, and "
                "Relevant Answer Generation described in the notes document and how "
                "they relate to Graph1.pdf and Graph2.pdf."
            ),
        },
        # 3 multi-section / multi-chunk
        {
            "cat": "multi_section",
            "document_id": DOCS["graph1"],
            "question": (
                "Using Graph1.pdf, what limitations of traditional RAG for specialized "
                "domains does GraphRAG address, and how does graph-structured knowledge "
                "representation help preserve context and logical consistency?"
            ),
        },
        {
            "cat": "multi_section",
            "document_id": DOCS["backend_projects"],
            "question": (
                "From the backend projects PDF, describe the five projects and how "
                "the challenge progresses regarding concurrency control, reliability, "
                "and domain modeling without off-the-shelf solutions."
            ),
        },
        {
            "cat": "multi_section",
            "document_id": DOCS["ai_roadmap"],
            "question": (
                "According to the AI Engineer Roadmap 2026, what foundation skills "
                "(Python, data engineering, ML, generative AI/LLM) does it emphasize "
                "and how does it frame the shift from experimental RAG demos to "
                "enterprise production systems?"
            ),
        },
        # 2 comparison
        {
            "cat": "comparison",
            "document_ids": [DOCS["graph1"], DOCS["graph2"]],
            "question": (
                "Compare Graph1.pdf and Graph2.pdf: how does each describe GraphRAG "
                "relative to traditional RAG, and what problems such as hallucination "
                "or relational knowledge gaps do they emphasize?"
            ),
        },
        {
            "cat": "comparison",
            "document_ids": [DOCS["invoice_cgst"], DOCS["invoice_igst"]],
            "question": (
                "Compare the SRI VENKATESWARA ENTERPRISES goods invoice (CGST/SGST) "
                "with the CLOUDSCALE CONSULTING services invoice (IGST): who is the "
                "buyer, what is being purchased, and how do the tax structures and "
                "totals differ?"
            ),
        },
        # 2 multi-document
        {
            "cat": "multi_document",
            "document_ids": [DOCS["graph1"], DOCS["ai_roadmap"]],
            "question": (
                "Using both Graph1.pdf and the AI Engineer Roadmap 2026, how does "
                "RAG/GraphRAG appear both as a research/knowledge-graph paradigm and "
                "as a production engineering competency?"
            ),
        },
        {
            "cat": "multi_document",
            "document_ids": [DOCS["math_calc"], DOCS["calculus"]],
            "question": (
                "Using both math-calculus-text.pdf and calculus-volume-1, what common "
                "authorship and licensing details appear, and what early calculus "
                "topics (functions, graphs, derivatives) do they cover?"
            ),
        },
    ]

    # Prefer Ollama for local_zero verification; optionally mark first comparison
    # for Bedrock when a bearer token is configured.
    for q in qs:
        q["llm_provider"] = "ollama"
    if bedrock_available:
        qs[11]["llm_provider"] = "bedrock"
        qs[5]["llm_provider"] = "bedrock"
    return qs


def login(client: httpx.Client) -> tuple[str, dict]:
    r = client.post(f"{BASE}/auth/dev-login")
    r.raise_for_status()
    data = r.json()
    return data["access_token"], data["user"]


def create_session(client: httpx.Client, headers: dict) -> str:
    r = client.post(
        f"{BASE}/chat/sessions",
        headers=headers,
        json={"title": "AI Usage E2E"},
    )
    r.raise_for_status()
    return r.json()["id"]


def ask(
    client: httpx.Client,
    headers: dict,
    *,
    question: str,
    session_id: str,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    llm_provider: str = "ollama",
) -> dict:
    payload: dict[str, Any] = {
        "question": question,
        "session_id": session_id,
        "top_k": 5,
        "llm_provider": llm_provider,
    }
    if document_ids:
        payload["document_ids"] = document_ids
    elif document_id:
        payload["document_id"] = document_id
    r = client.post(f"{BASE}/chat/ask", headers=headers, json=payload, timeout=300.0)
    r.raise_for_status()
    return r.json()


def usage_snapshot(client: httpx.Client, headers: dict) -> dict:
    summary = client.get(f"{BASE}/usage/summary", headers=headers).json()
    events = client.get(
        f"{BASE}/usage/events",
        headers=headers,
        params={"limit": 50, "page": 1},
    ).json()
    daily = client.get(f"{BASE}/usage/daily", headers=headers).json()
    return {"summary": summary, "events": events, "daily": daily}


def main() -> int:
    bedrock_available = bool(
        (os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
    )
    # Also check .env without printing secrets
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists() and not bedrock_available:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("AWS_BEARER_TOKEN_BEDROCK=") and len(line.split("=", 1)[1].strip()) > 8:
                bedrock_available = True
                break

    qs = questions(bedrock_available)
    results: list[dict[str, Any]] = []

    with httpx.Client(timeout=60.0) as client:
        token, user = login(client)
        headers = {"Authorization": f"Bearer {token}"}
        print(f"AUTH user={user['username']} id={user['id']}")
        print(f"BEDROCK_AVAILABLE={bedrock_available}")

        before = usage_snapshot(client, headers)
        print(
            "BEFORE usage requests=",
            before["summary"]["request_count"],
            "tokens=",
            before["summary"]["total_tokens"],
        )

        session_id = create_session(client, headers)
        print(f"SESSION {session_id}")

        for i, q in enumerate(qs, 1):
            print(f"\n=== Q{i}/{len(qs)} [{q['cat']}] provider={q['llm_provider']} ===")
            print(q["question"][:120], "...")
            started = time.perf_counter()
            try:
                answer = ask(
                    client,
                    headers,
                    question=q["question"],
                    session_id=session_id,
                    document_id=q.get("document_id"),
                    document_ids=q.get("document_ids"),
                    llm_provider=q["llm_provider"],
                )
                elapsed = time.perf_counter() - started
                usage = answer.get("usage")  # ChatResponse may not include usage field
                # Pull latest event for this turn from ledger
                snap = usage_snapshot(client, headers)
                latest = (snap["events"].get("items") or [None])[0]
                results.append(
                    {
                        "index": i,
                        "cat": q["cat"],
                        "provider_requested": q["llm_provider"],
                        "elapsed_s": round(elapsed, 2),
                        "answer_len": len(answer.get("answer") or ""),
                        "sources": len(answer.get("sources") or []),
                        "response_usage": usage,
                        "latest_event": latest,
                        "ok": True,
                    }
                )
                if latest:
                    print(
                        f"OK answer_chars={len(answer.get('answer') or '')} "
                        f"sources={len(answer.get('sources') or [])} "
                        f"ledger provider={latest.get('provider')} "
                        f"model={latest.get('model_id')} "
                        f"in={latest.get('input_tokens')} out={latest.get('output_tokens')} "
                        f"total={latest.get('total_tokens')} "
                        f"cost={latest.get('estimated_cost_usd')} "
                        f"cost_type={latest.get('cost_type')} "
                        f"user_match={latest.get('id') is not None}"
                    )
                else:
                    print("OK but NO usage event in ledger yet")
            except Exception as exc:
                elapsed = time.perf_counter() - started
                print(f"FAIL after {elapsed:.1f}s: {exc}")
                results.append(
                    {
                        "index": i,
                        "cat": q["cat"],
                        "provider_requested": q["llm_provider"],
                        "elapsed_s": round(elapsed, 2),
                        "ok": False,
                        "error": str(exc)[:400],
                    }
                )

        after = usage_snapshot(client, headers)
        print("\n=== FINAL SUMMARY ===")
        print(json.dumps(after["summary"], indent=2)[:2000])
        print("events_total=", after["events"]["total"])
        print("daily_points_nonzero=", sum(1 for p in after["daily"]["points"] if p["request_count"]))

        out = Path(__file__).resolve().parents[1] / "data" / "e2e_ai_usage_report.json"
        out.write_text(
            json.dumps(
                {
                    "user": user,
                    "before": before,
                    "after": after,
                    "results": results,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {out}")

        # Verification gates
        ok_count = sum(1 for r in results if r.get("ok"))
        failures = [r for r in results if not r.get("ok")]
        events = after["events"]["items"]
        missing_fields = []
        for ev in events[:ok_count]:
            for field in (
                "created_at",
                "provider",
                "model_id",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_type",
            ):
                if ev.get(field) in (None, ""):
                    missing_fields.append((ev.get("id"), field))
            if ev.get("estimated_cost_usd") is None and ev.get("cost_type") not in {
                "unavailable"
            }:
                missing_fields.append((ev.get("id"), "estimated_cost_usd"))

        ollama_events = [e for e in events if e.get("provider") == "ollama"]
        bedrock_events = [e for e in events if e.get("provider") == "bedrock"]
        bad_ollama_cost = [
            e
            for e in ollama_events
            if e.get("estimated_cost_usd") not in (0, 0.0) or e.get("cost_type") != "local_zero"
        ]

        print(f"ok_questions={ok_count}/{len(qs)}")
        print(f"failures={len(failures)}")
        print(f"missing_fields={missing_fields[:20]}")
        print(f"ollama_events={len(ollama_events)} bad_ollama_cost={len(bad_ollama_cost)}")
        print(f"bedrock_events={len(bedrock_events)}")
        print(
            "delta_requests=",
            after["summary"]["request_count"] - before["summary"]["request_count"],
        )

        if ok_count < len(qs):
            return 2
        if after["summary"]["request_count"] < before["summary"]["request_count"] + ok_count:
            print("ERROR: usage request_count did not increase by number of successful asks")
            return 3
        if bad_ollama_cost:
            print("ERROR: Ollama cost not local_zero/$0")
            return 4
        if missing_fields:
            print("ERROR: required usage fields missing")
            return 5
        return 0


if __name__ == "__main__":
    sys.exit(main())

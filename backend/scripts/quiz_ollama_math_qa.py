"""
Ollama-only quiz QA against indexed math-related PDFs.

Mirrors quiz_ollama_qa.py but uses calculus / math documents, with
math-focused seed questions and equation checks. Never uses Bedrock.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

BASE = "http://127.0.0.1:7860/api/v1"
OUT_PATH = Path(__file__).resolve().parent / "quiz_ollama_math_qa_results.json"
PROVIDER = "ollama"

MATH_NAME_HINTS = (
    "calc",
    "math",
    "algebra",
    "equation",
    "geometry",
    "trig",
    "physics",
    "volume",
    "integral",
    "derivative",
)


def login(client: httpx.Client) -> str:
    r = client.post(f"{BASE}/auth/dev-login")
    r.raise_for_status()
    return r.json()["access_token"]


def list_ready_math_docs(client: httpx.Client, headers: dict) -> list[dict]:
    r = client.get(f"{BASE}/documents/", headers=headers)
    r.raise_for_status()
    payload = r.json()
    items = payload if isinstance(payload, list) else (
        payload.get("documents") or payload.get("items") or payload.get("data") or []
    )
    found = []
    for d in items:
        name = (d.get("original_name") or d.get("name") or "").lower()
        if d.get("status") != "ready":
            continue
        if any(h in name for h in MATH_NAME_HINTS):
            found.append(d)
    return found


def create_session(client: httpx.Client, headers: dict, title: str) -> str:
    r = client.post(f"{BASE}/chat/sessions", headers=headers, json={"title": title})
    r.raise_for_status()
    return r.json()["id"]


def ask(
    client: httpx.Client,
    headers: dict,
    *,
    question: str,
    session_id: str,
    document_id: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "session_id": session_id,
        "llm_provider": PROVIDER,
    }
    if document_id:
        payload["document_id"] = document_id
    t0 = time.time()
    r = client.post(f"{BASE}/chat/ask", headers=headers, json=payload)
    elapsed = round(time.time() - t0, 1)
    if r.status_code == 401:
        token = login(client)
        headers["Authorization"] = f"Bearer {token}"
        t0 = time.time()
        r = client.post(f"{BASE}/chat/ask", headers=headers, json=payload)
        elapsed = round(time.time() - t0, 1)
    if r.status_code >= 400:
        body: dict[str, Any] = {"error": r.status_code, "raw": r.text[:800]}
    else:
        body = r.json()
    body["_elapsed_s"] = elapsed
    body["_status"] = r.status_code
    return body


def has_quiz(body: dict) -> bool:
    quiz = body.get("quiz")
    return isinstance(quiz, dict) and bool(quiz.get("questions"))


def quiz_meta(body: dict) -> dict:
    quiz = body.get("quiz") or {}
    questions = quiz.get("questions") or []
    joined = " ".join(
        (q.get("question") or "")
        + " "
        + " ".join(o.get("text") or "" for o in (q.get("options") or []))
        + " "
        + (q.get("explanation") or "")
        for q in questions
    )
    return {
        "mode": quiz.get("mode"),
        "title": quiz.get("title"),
        "n_questions": len(questions),
        "has_explanations": all(bool(q.get("explanation")) for q in questions) if questions else False,
        "options_ok": all(len(q.get("options") or []) >= 2 for q in questions) if questions else False,
        "correct_ids_ok": all(
            q.get("correct_option_id") in {o.get("id") for o in (q.get("options") or [])}
            for q in questions
        )
        if questions
        else False,
        "mathish": any(
            tok in joined
            for tok in (
                "\\(",
                "\\[",
                "=",
                "integral",
                "derivative",
                "limit",
                "dx",
                "dy",
                "sin",
                "cos",
                "tan",
                "π",
                "pi",
                "fraction",
                "equation",
                "formula",
            )
        ),
    }


def find_assistant_with_quiz(client: httpx.Client, headers: dict, session_id: str) -> Optional[dict]:
    r = client.get(f"{BASE}/chat/history/session/{session_id}", headers=headers)
    r.raise_for_status()
    for msg in reversed(r.json().get("messages") or []):
        if msg.get("role") == "assistant" and msg.get("quiz"):
            return msg
    return None


def run_for_doc(client: httpx.Client, headers: dict, doc: dict) -> dict:
    doc_id = str(doc["id"])
    name = doc.get("original_name") or doc_id
    session_id = create_session(
        client,
        headers,
        f"Quiz Math Ollama QA — {name[:40]} {time.strftime('%H:%M')}",
    )
    results: list[dict] = []
    print(f"\n===== MATH DOC: {name} ({doc_id}) session={session_id} =====", flush=True)

    def record(case_id: str, ok: bool, detail: dict):
        row = {"id": case_id, "pass": ok, "document": name, **detail}
        results.append(row)
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {case_id}", flush=True)
        preview = detail.get("answer_preview") or ""
        if preview:
            print(f"       A: {preview[:220].replace(chr(10), ' ')}", flush=True)

    # Negative: ordinary conceptual question
    body = ask(
        client,
        headers,
        question="Explain the definition of the derivative from this document.",
        session_id=session_id,
        document_id=doc_id,
    )
    record(
        "neg_ordinary_math_question",
        body.get("_status") == 200 and not has_quiz(body) and bool(body.get("answer")),
        {
            "elapsed_s": body.get("_elapsed_s"),
            "quiz_present": has_quiz(body),
            "answer_preview": (body.get("answer") or body.get("raw") or "")[:400],
        },
    )

    # Negative: simple summary
    body = ask(
        client,
        headers,
        question="Please summarize the main calculus topics covered in this PDF.",
        session_id=session_id,
        document_id=doc_id,
    )
    record(
        "neg_simple_summary",
        body.get("_status") == 200 and not has_quiz(body) and bool(body.get("answer")),
        {
            "elapsed_s": body.get("_elapsed_s"),
            "quiz_present": has_quiz(body),
            "answer_preview": (body.get("answer") or body.get("raw") or "")[:400],
        },
    )

    # Math / equation question — must answer with math, not auto-quiz
    body = ask(
        client,
        headers,
        question=(
            "From this document, what is the derivative of x^2? "
            "Show the equation using LaTeX if possible."
        ),
        session_id=session_id,
        document_id=doc_id,
    )
    answer = body.get("answer") or ""
    record(
        "neg_math_equation_no_auto_quiz",
        body.get("_status") == 200 and not has_quiz(body) and bool(answer),
        {
            "elapsed_s": body.get("_elapsed_s"),
            "quiz_present": has_quiz(body),
            "has_equation_markup": ("\\(" in answer)
            or ("\\[" in answer)
            or ("=" in answer)
            or ("derivative" in answer.lower()),
            "answer_preview": answer[:400],
        },
    )

    # Seed more conversation for conversation-quiz mode
    body = ask(
        client,
        headers,
        question=(
            "According to this calculus PDF, state the power rule and use it to "
            "differentiate x^3. Show the equation."
        ),
        session_id=session_id,
        document_id=doc_id,
    )
    record(
        "seed_power_rule",
        body.get("_status") == 200 and not has_quiz(body),
        {
            "elapsed_s": body.get("_elapsed_s"),
            "quiz_present": has_quiz(body),
            "answer_preview": (body.get("answer") or "")[:400],
        },
    )

    # 1) Conversation quiz
    body = ask(
        client,
        headers,
        question="Quiz me on our conversation with 5 questions about derivatives and equations",
        session_id=session_id,
        document_id=doc_id,
    )
    meta = quiz_meta(body) if has_quiz(body) else {}
    record(
        "quiz_conversation",
        body.get("_status") == 200
        and has_quiz(body)
        and meta.get("mode") == "conversation"
        and meta.get("n_questions", 0) >= 1
        and meta.get("options_ok")
        and meta.get("correct_ids_ok"),
        {
            "elapsed_s": body.get("_elapsed_s"),
            **meta,
            "answer_preview": (body.get("answer") or body.get("raw") or "")[:400],
        },
    )

    # 2) Document quiz with count
    body = ask(
        client,
        headers,
        question="Quiz me on this PDF with 6 questions about limits, derivatives, and integrals",
        session_id=session_id,
        document_id=doc_id,
    )
    meta = quiz_meta(body) if has_quiz(body) else {}
    record(
        "quiz_document_count6",
        body.get("_status") == 200
        and has_quiz(body)
        and meta.get("mode") == "document"
        and 1 <= meta.get("n_questions", 0) <= 6
        and meta.get("options_ok")
        and meta.get("correct_ids_ok"),
        {
            "elapsed_s": body.get("_elapsed_s"),
            "requested": 6,
            **meta,
            "answer_preview": (body.get("answer") or body.get("raw") or "")[:400],
        },
    )

    # 3) Summary + quiz
    body = ask(
        client,
        headers,
        question="Summarize this document and then quiz me with 5 questions",
        session_id=session_id,
        document_id=doc_id,
    )
    meta = quiz_meta(body) if has_quiz(body) else {}
    answer = body.get("answer") or ""
    record(
        "quiz_summary_then_quiz",
        body.get("_status") == 200
        and len(answer) > 120
        and has_quiz(body)
        and meta.get("mode") == "summary"
        and meta.get("n_questions", 0) >= 1
        and meta.get("options_ok"),
        {
            "elapsed_s": body.get("_elapsed_s"),
            "answer_len": len(answer),
            **meta,
            "answer_preview": answer[:500],
        },
    )

    # State persistence
    msg = find_assistant_with_quiz(client, headers, session_id)
    if msg and msg.get("quiz") and (msg["quiz"].get("questions") or []):
        q0 = msg["quiz"]["questions"][0]
        wrong = next(
            (o["id"] for o in q0["options"] if o["id"] != q0["correct_option_id"]),
            q0["correct_option_id"],
        )
        right = q0["correct_option_id"]
        r1 = client.patch(
            f"{BASE}/chat/messages/{msg['id']}/quiz-state",
            headers=headers,
            json={"current_index": 0, "answers": {q0["id"]: wrong}, "completed": False},
        )
        r2 = client.patch(
            f"{BASE}/chat/messages/{msg['id']}/quiz-state",
            headers=headers,
            json={
                "current_index": min(1, max(0, len(msg["quiz"]["questions"]) - 1)),
                "answers": {q0["id"]: right},
                "completed": False,
            },
        )
        state2 = (r2.json().get("quiz") or {}).get("state") if r2.status_code == 200 else None
        record(
            "quiz_state_select_navigate",
            r1.status_code == 200 and r2.status_code == 200 and bool(state2),
            {
                "message_id": msg["id"],
                "right_saved": (state2 or {}).get("answers", {}).get(q0["id"]) == right,
                "current_index": (state2 or {}).get("current_index"),
            },
        )
    else:
        record("quiz_state_select_navigate", False, {"error": "no quiz message"})

    # Edge negatives
    for i, phrase in enumerate(
        [
            "What questions does the practice section include?",
            "Give me the formula for the power rule.",
            "Can you summarize only the chapter on limits?",
        ],
        1,
    ):
        body = ask(
            client,
            headers,
            question=phrase,
            session_id=session_id,
            document_id=doc_id,
        )
        record(
            f"neg_edge_{i}",
            body.get("_status") == 200 and not has_quiz(body),
            {
                "phrase": phrase,
                "elapsed_s": body.get("_elapsed_s"),
                "quiz_present": has_quiz(body),
                "answer_preview": (body.get("answer") or body.get("raw") or "")[:300],
            },
        )

    passed = sum(1 for r in results if r["pass"])
    return {
        "document_id": doc_id,
        "document_name": name,
        "session_id": session_id,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "failed_ids": [r["id"] for r in results if not r["pass"]],
        "results": results,
    }


def main() -> int:
    timeout = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        token = login(client)
        headers = {"Authorization": f"Bearer {token}"}
        docs = list_ready_math_docs(client, headers)
        print(f"Found {len(docs)} ready math-related document(s)", flush=True)
        for d in docs:
            print(
                f" - {d.get('id')} | {d.get('original_name')} | "
                f"pages={d.get('page_count')} chunks={d.get('chunk_count')}",
                flush=True,
            )
        if not docs:
            print("No ready math PDFs found.", flush=True)
            OUT_PATH.write_text(json.dumps({"error": "no_math_docs"}, indent=2), encoding="utf-8")
            return 2

        # Prefer calculus docs; dedupe by name and cap at 1 for runtime (full matrix is long on 3B).
        preferred = [
            d
            for d in docs
            if "calc" in (d.get("original_name") or "").lower()
            or "volume" in (d.get("original_name") or "").lower()
        ]
        seen_names: set[str] = set()
        selected = []
        for d in preferred or docs:
            name = (d.get("original_name") or "").lower()
            if name in seen_names:
                continue
            seen_names.add(name)
            selected.append(d)
            if len(selected) >= 1:
                break

        per_doc = []
        for doc in selected:
            per_doc.append(run_for_doc(client, headers, doc))

    summary = {
        "provider": PROVIDER,
        "documents_tested": [
            {"id": x["document_id"], "name": x["document_name"], "passed": x["passed"], "total": x["total"]}
            for x in per_doc
        ],
        "passed": sum(x["passed"] for x in per_doc),
        "total": sum(x["total"] for x in per_doc),
        "failed_ids": [f"{x['document_name']}:{i}" for x in per_doc for i in x["failed_ids"]],
        "per_document": per_doc,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n==== MATH QUIZ QA SUMMARY ====", flush=True)
    print(
        f"passed={summary['passed']}/{summary['total']} failed={summary['failed_ids']}",
        flush=True,
    )
    print(f"wrote {OUT_PATH}", flush=True)
    return 0 if not summary["failed_ids"] else 1


if __name__ == "__main__":
    sys.exit(main())

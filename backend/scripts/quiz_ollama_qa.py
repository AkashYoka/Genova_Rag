"""
Ollama-only QA for interactive quiz feature against the live Genova API.

Creates one chat session, seeds conversation history, then exercises:
  - negative cases (ordinary Q / summary must NOT return quiz)
  - conversation quiz
  - document quiz (with question count)
  - summary + quiz
  - math/equation grounding when present in history or document
  - quiz-state persistence (select answer / score / navigate)

Never uses Bedrock. Requires local backend + Ollama (qwen2.5:3b).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

BASE = "http://127.0.0.1:7860/api/v1"
DOC_ID = "8c57d30c-9568-4437-b51f-b52d9ac495d1"  # indexed PDF used by prior UI QA
OUT_PATH = Path(__file__).resolve().parent / "quiz_ollama_qa_results.json"
PROVIDER = "ollama"


def login(client: httpx.Client) -> str:
    r = client.post(f"{BASE}/auth/dev-login")
    r.raise_for_status()
    return r.json()["access_token"]


def create_session(client: httpx.Client, headers: dict, title: str) -> str:
    r = client.post(
        f"{BASE}/chat/sessions",
        headers=headers,
        json={"title": title},
    )
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
    body: dict[str, Any]
    if r.status_code >= 400:
        body = {"error": r.status_code, "raw": r.text[:800]}
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
            any(tok in ((q.get("question") or "") + " " + " ".join(o.get("text") or "" for o in (q.get("options") or [])))
                for tok in ("\\(", "\\[", "=", "mg", "dose", "equation", "formula", "×", "*"))
            for q in questions
        ),
    }


def find_assistant_with_quiz(client: httpx.Client, headers: dict, session_id: str) -> Optional[dict]:
    r = client.get(f"{BASE}/chat/history/session/{session_id}", headers=headers)
    r.raise_for_status()
    messages = r.json().get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("quiz"):
            return msg
    return None


def main() -> int:
    results: list[dict] = []
    timeout = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=30.0)

    def record(case_id: str, ok: bool, detail: dict):
        row = {"id": case_id, "pass": ok, **detail}
        results.append(row)
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {case_id}  {json.dumps({k: v for k, v in detail.items() if k not in {'answer_preview'}}, default=str)[:400]}", flush=True)
        if detail.get("answer_preview"):
            print(f"       A: {detail['answer_preview'][:220].replace(chr(10), ' ')}", flush=True)

    with httpx.Client(timeout=timeout) as client:
        token = login(client)
        headers = {"Authorization": f"Bearer {token}"}
        session_id = create_session(
            client, headers, f"Quiz Ollama QA {time.strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"session={session_id} doc={DOC_ID} provider={PROVIDER}", flush=True)

        # ── 0a. Ordinary factual question must NOT quiz ─────────────
        body = ask(
            client,
            headers,
            question="What is XOLAIR indicated for in chronic spontaneous urticaria?",
            session_id=session_id,
            document_id=DOC_ID,
        )
        record(
            "neg_ordinary_question",
            body.get("_status") == 200 and not has_quiz(body) and bool(body.get("answer")),
            {
                "elapsed_s": body.get("_elapsed_s"),
                "status": body.get("_status"),
                "quiz_present": has_quiz(body),
                "answer_preview": (body.get("answer") or body.get("raw") or "")[:400],
            },
        )

        # ── 0b. Simple summary must NOT quiz ────────────────────────
        body = ask(
            client,
            headers,
            question="Please summarize the key safety warnings in this document.",
            session_id=session_id,
            document_id=DOC_ID,
        )
        record(
            "neg_simple_summary",
            body.get("_status") == 200 and not has_quiz(body) and bool(body.get("answer")),
            {
                "elapsed_s": body.get("_elapsed_s"),
                "status": body.get("_status"),
                "quiz_present": has_quiz(body),
                "answer_preview": (body.get("answer") or body.get("raw") or "")[:400],
            },
        )

        # ── 0c. Math/equation style question must NOT auto-quiz ─────
        body = ask(
            client,
            headers,
            question=(
                "If a patient needs 300 mg and the vial strength is 150 mg/1.2 mL, "
                "how many vials are required? Show the equation clearly."
            ),
            session_id=session_id,
            document_id=DOC_ID,
        )
        answer = body.get("answer") or ""
        record(
            "neg_math_no_auto_quiz",
            body.get("_status") == 200 and not has_quiz(body) and bool(answer),
            {
                "elapsed_s": body.get("_elapsed_s"),
                "status": body.get("_status"),
                "quiz_present": has_quiz(body),
                "has_equation_markup": ("\\(" in answer) or ("\\[" in answer) or ("=" in answer),
                "answer_preview": answer[:400],
            },
        )

        # Seed a bit more conversation for conversation-quiz mode
        body = ask(
            client,
            headers,
            question="What dosing interval does the label recommend for CSU?",
            session_id=session_id,
            document_id=DOC_ID,
        )
        record(
            "seed_dosing_interval",
            body.get("_status") == 200 and not has_quiz(body),
            {
                "elapsed_s": body.get("_elapsed_s"),
                "quiz_present": has_quiz(body),
                "answer_preview": (body.get("answer") or "")[:400],
            },
        )

        # ── 1. Conversation-based quiz ──────────────────────────────
        body = ask(
            client,
            headers,
            question="Quiz me on our conversation with 5 questions",
            session_id=session_id,
            document_id=DOC_ID,
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
                "status": body.get("_status"),
                **meta,
                "answer_preview": (body.get("answer") or body.get("raw") or "")[:400],
            },
        )
        convo_quiz = body.get("quiz")

        # ── 2. Document-based quiz with custom count ────────────────
        body = ask(
            client,
            headers,
            question="Quiz me on this PDF with 6 questions about dosing and indications",
            session_id=session_id,
            document_id=DOC_ID,
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
                "status": body.get("_status"),
                "requested": 6,
                **meta,
                "answer_preview": (body.get("answer") or body.get("raw") or "")[:400],
            },
        )
        doc_quiz = body.get("quiz")

        # ── 3. Summary + quiz ───────────────────────────────────────
        body = ask(
            client,
            headers,
            question="Summarize this document and then quiz me with 5 questions",
            session_id=session_id,
            document_id=DOC_ID,
        )
        meta = quiz_meta(body) if has_quiz(body) else {}
        answer = body.get("answer") or ""
        # Summary should appear before quiz marker / be substantive
        summary_ok = len(answer) > 120 and has_quiz(body)
        record(
            "quiz_summary_then_quiz",
            body.get("_status") == 200
            and summary_ok
            and meta.get("mode") == "summary"
            and meta.get("n_questions", 0) >= 1
            and meta.get("options_ok"),
            {
                "elapsed_s": body.get("_elapsed_s"),
                "status": body.get("_status"),
                "answer_len": len(answer),
                **meta,
                "answer_preview": answer[:500],
            },
        )

        # ── 4. Persist option selection / scoring via quiz-state ────
        msg = find_assistant_with_quiz(client, headers, session_id)
        if msg and msg.get("quiz"):
            quiz = msg["quiz"]
            q0 = (quiz.get("questions") or [None])[0]
            if q0:
                wrong = next(
                    (o["id"] for o in q0["options"] if o["id"] != q0["correct_option_id"]),
                    q0["correct_option_id"],
                )
                right = q0["correct_option_id"]
                # select wrong first
                r = client.patch(
                    f"{BASE}/chat/messages/{msg['id']}/quiz-state",
                    headers=headers,
                    json={
                        "current_index": 0,
                        "answers": {q0["id"]: wrong},
                        "completed": False,
                    },
                )
                ok_wrong = r.status_code == 200
                state1 = (r.json().get("quiz") or {}).get("state") if ok_wrong else None

                # navigate + correct on same q for score check, then complete if single
                answers = {q0["id"]: right}
                completed = len(quiz.get("questions") or []) == 1
                r2 = client.patch(
                    f"{BASE}/chat/messages/{msg['id']}/quiz-state",
                    headers=headers,
                    json={
                        "current_index": min(1, max(0, len(quiz["questions"]) - 1)),
                        "answers": answers,
                        "completed": completed,
                    },
                )
                ok_right = r2.status_code == 200
                state2 = (r2.json().get("quiz") or {}).get("state") if ok_right else None
                record(
                    "quiz_state_select_navigate",
                    ok_wrong and ok_right and bool(state1) and bool(state2),
                    {
                        "message_id": msg["id"],
                        "wrong_saved": (state1 or {}).get("answers", {}).get(q0["id"]) == wrong,
                        "right_saved": (state2 or {}).get("answers", {}).get(q0["id"]) == right,
                        "current_index": (state2 or {}).get("current_index"),
                        "status_wrong": r.status_code,
                        "status_right": r2.status_code,
                    },
                )
            else:
                record("quiz_state_select_navigate", False, {"error": "no questions on saved quiz"})
        else:
            record("quiz_state_select_navigate", False, {"error": "no assistant quiz message in history"})

        # ── 5. Intent unit-ish checks via live routing edge phrases ─
        false_pos_phrases = [
            "What questions appear in the adverse reactions table?",
            "Give me the recommended dose for adolescents.",
            "Can you summarize the boxed warning only?",
        ]
        for i, phrase in enumerate(false_pos_phrases, 1):
            body = ask(
                client,
                headers,
                question=phrase,
                session_id=session_id,
                document_id=DOC_ID,
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
    failed = [r for r in results if not r["pass"]]
    summary = {
        "provider": PROVIDER,
        "document_id": DOC_ID,
        "total": len(results),
        "passed": passed,
        "failed": len(failed),
        "failed_ids": [r["id"] for r in failed],
        "results": results,
        "sample_quizzes": {
            "conversation_n": len((convo_quiz or {}).get("questions") or []) if "convo_quiz" in dir() else None,
            "document_n": len((doc_quiz or {}).get("questions") or []) if "doc_quiz" in dir() else None,
        },
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n==== SUMMARY ====", flush=True)
    print(f"passed={passed}/{len(results)} failed={summary['failed_ids']}", flush=True)
    print(f"wrote {OUT_PATH}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

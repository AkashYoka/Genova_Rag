"""Unit tests for interactive quiz intent parsing and payload normalization."""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.rag.utils.quiz import (
    DEFAULT_QUESTION_COUNT,
    _normalize_quiz_payload,
    clamp_question_count,
    generate_quiz_stream,
    is_quiz_request,
    parse_question_count,
    parse_quiz_intent,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("quiz me on this document", True),
        ("Give me 7 MCQs", True),
        ("summarize and quiz me", True),
        ("What is the recommended dose?", False),
        ("hello", False),
        ("What questions appear in the adverse reactions table?", False),
        ("Give me the recommended dose for adolescents.", False),
        ("Please summarize the key safety warnings in this document.", False),
        ("Can you summarize the boxed warning only?", False),
    ],
)
def test_is_quiz_request(text, expected):
    assert is_quiz_request(text) is expected


def test_parse_question_count_defaults_and_custom():
    assert parse_question_count("quiz me") == DEFAULT_QUESTION_COUNT
    assert parse_question_count("quiz me with 8 questions") == 8
    assert parse_question_count("give me 10 mcqs") == 10
    assert parse_question_count("Make a 3-question interactive quiz") == 3
    assert parse_question_count("Create an interactive quiz with 4 questions") == 4
    assert clamp_question_count(100) == 15
    assert clamp_question_count(0) == 1


def test_parse_quiz_intent_modes():
    convo = parse_quiz_intent("quiz me on our conversation", has_document=True)
    assert convo is not None
    assert convo.mode == "conversation"

    doc = parse_quiz_intent("quiz me on this PDF with 6 questions", has_document=True)
    assert doc is not None
    assert doc.mode == "document"
    assert doc.num_questions == 6

    summary = parse_quiz_intent("summarize this document and quiz me", has_document=True)
    assert summary is not None
    assert summary.mode == "summary"

    fallback_doc = parse_quiz_intent("quiz me", has_document=True)
    assert fallback_doc is not None
    assert fallback_doc.mode == "document"

    fallback_convo = parse_quiz_intent("quiz me", has_document=False)
    assert fallback_convo is not None
    assert fallback_convo.mode == "conversation"


def test_normalize_quiz_payload_filters_invalid():
    raw = {
        "title": "CSU basics",
        "questions": [
            {
                "question": "Recommended dosing interval?",
                "options": [
                    {"id": "a", "text": "Every 2 weeks"},
                    {"id": "b", "text": "Every 4 weeks"},
                    {"id": "c", "text": "Daily"},
                    {"id": "d", "text": "Yearly"},
                ],
                "correct_option_id": "b",
                "explanation": "Label states every 4 weeks.",
                "source": {"filename": "xolair.pdf", "page": 9},
            },
            {"question": "Missing options", "options": [{"id": "a", "text": "Only one"}]},
            {"question": "", "options": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]},
        ],
    }
    quiz = _normalize_quiz_payload(raw, mode="document", num_questions=5)
    assert quiz["mode"] == "document"
    assert quiz["title"] == "CSU basics"
    assert len(quiz["questions"]) == 1
    assert quiz["questions"][0]["correct_option_id"] == "b"
    assert quiz["state"]["current_index"] == 0
    assert quiz["state"]["answers"] == {}
    assert quiz["state"]["completed"] is False


def _fake_mcq(n: int, prefix: str = "Q") -> dict:
    return {
        "title": "Batch",
        "questions": [
            {
                "question": f"{prefix}{i} about limits?",
                "options": [
                    {"id": "a", "text": "A"},
                    {"id": "b", "text": "B"},
                    {"id": "c", "text": "C"},
                    {"id": "d", "text": "D"},
                ],
                "correct_option_id": "a",
                "explanation": "From the text.",
                "source": {"filename": "math.pdf", "page": i},
            }
            for i in range(1, n + 1)
        ],
    }


def test_generate_quiz_response_batches_to_requested_count(monkeypatch):
    """Asking for 6 must not stop at the old hard-coded retry of 3."""
    from app.rag.utils import quiz as quiz_mod

    calls: list[int] = []

    def fake_llm_json(*, system, user, llm_provider, max_tokens=3072):
        # Read requested count from the prompt template line.
        import re as _re

        m = _re.search(r"Requested question count:\s*(\d+)", user)
        n = int(m.group(1)) if m else 3
        calls.append(n)
        # Return exactly the asked batch size with unique stems.
        stem = f"batch{len(calls)}"
        return _fake_mcq(n, prefix=stem)

    monkeypatch.setattr(quiz_mod, "_llm_json", fake_llm_json)
    monkeypatch.setattr(
        quiz_mod,
        "_build_source_material",
        lambda *a, **k: (
            "Limits and derivatives appear throughout the chapter. " * 20,
            [{"filename": "math.pdf", "page": 1}],
            "",
        ),
    )

    result = quiz_mod.generate_quiz_response(
        "Quiz me on this PDF with 6 questions about limits, derivatives, and integrals",
        "user-1",
        document_id="doc-1",
        llm_provider="ollama",
    )
    quiz = result["quiz"]
    assert quiz is not None
    assert len(quiz["questions"]) == 6
    assert "6-question" in result["answer"]
    # First batch of 3, then a fill batch of 3.
    assert calls[0] == 3
    assert sum(calls) >= 6


def test_generate_quiz_stream_emits_quiz_event():
    fake_quiz = {
        "id": "quiz-1",
        "mode": "conversation",
        "title": "Chat check",
        "questions": [
            {
                "id": "q1",
                "question": "Q?",
                "options": [
                    {"id": "a", "text": "A"},
                    {"id": "b", "text": "B"},
                    {"id": "c", "text": "C"},
                    {"id": "d", "text": "D"},
                ],
                "correct_option_id": "a",
                "explanation": "Because the conversation said so.",
                "source": None,
            }
        ],
        "state": {"current_index": 0, "answers": {}, "completed": False},
    }

    with patch(
        "app.rag.utils.quiz.generate_quiz_response",
        return_value={
            "answer": "Here's a quiz.",
            "sources": [],
            "quiz": fake_quiz,
        },
    ):
        events = []
        for line in generate_quiz_stream("quiz me on our conversation", "user-1"):
            assert line.startswith("data: ")
            events.append(json.loads(line[6:].strip()))

    types = [e["type"] for e in events]
    assert types[0] == "sources"
    assert "token" in types
    assert "quiz" in types
    assert types[-1] == "done"
    quiz_event = next(e for e in events if e["type"] == "quiz")
    assert quiz_event["data"]["id"] == "quiz-1"


def test_agent_routes_quiz_before_rag(monkeypatch):
    from app.rag.llm import agent

    called = {"quiz": False, "direct": False}

    def fake_quiz_response(*_args, **_kwargs):
        called["quiz"] = True
        return {"answer": "quiz intro", "sources": [], "quiz": {"id": "x", "mode": "document", "title": "t", "questions": [], "state": {}}}

    def fake_direct(*_args, **_kwargs):
        called["direct"] = True
        return {"answer": "rag", "sources": []}

    monkeypatch.setattr("app.rag.utils.quiz.generate_quiz_response", fake_quiz_response)
    monkeypatch.setattr(agent, "_generate_answer_direct", fake_direct)

    result = agent.generate_answer(
        "Quiz me on this document with 5 questions",
        "user-1",
        document_id="doc-1",
        llm_provider="ollama",
    )
    assert called["quiz"] is True
    assert called["direct"] is False
    assert result["answer"] == "quiz intro"

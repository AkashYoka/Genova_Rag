"""Unit tests for interactive quiz intent parsing and payload normalization."""
import json
from unittest.mock import patch

import pytest

from app.rag.utils.quiz import (
    DEFAULT_QUESTION_COUNT,
    MAX_QUESTION_COUNT,
    QUIZ_FILL_BATCH_SIZE,
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
        ("Give me a 20-question assessment.", True),
        ("Prepare an assessment based on this document.", True),
        ("Create an assessment with 10 questions", True),
        ("Generate a quiz with 15 questions", True),
        ("practice quiz on this PDF", True),
        ("give me 10 practice questions", True),
        ("What is the recommended dose?", False),
        ("hello", False),
        ("What questions appear in the adverse reactions table?", False),
        ("Give me the recommended dose for adolescents.", False),
        ("Please summarize the key safety warnings in this document.", False),
        ("Can you summarize the boxed warning only?", False),
        # Ordinary prose mentioning "practice questions" must NOT trigger quiz mode.
        (
            "Does the collection include Modern History topics among practice questions?",
            False,
        ),
        ("How many practice questions are listed in chapter 3?", False),
        ("What is a risk assessment for anaphylaxis?", False),
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
    assert parse_question_count("Give me a 20-question assessment.") == 20
    assert parse_question_count("Generate a quiz with 15 questions") == 15
    assert clamp_question_count(100) == MAX_QUESTION_COUNT
    assert clamp_question_count(20) == 20
    assert clamp_question_count(0) == 1
    assert MAX_QUESTION_COUNT == 20


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


def test_parse_quiz_intent_assessment_and_counts():
    intent = parse_quiz_intent(
        "Give me a 20-question assessment.", has_document=True
    )
    assert intent is not None
    assert intent.num_questions == 20
    assert intent.mode == "document"

    intent15 = parse_quiz_intent(
        "Generate a quiz with 15 questions", has_document=True
    )
    assert intent15 is not None
    assert intent15.num_questions == 15

    # False positive: factual question mentioning practice questions.
    assert (
        parse_quiz_intent(
            "Does the collection include Modern History topics among practice questions?",
            has_document=True,
        )
        is None
    )


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


def _patch_material(monkeypatch, quiz_mod):
    monkeypatch.setattr(
        quiz_mod,
        "_build_source_material",
        lambda *a, **k: (
            "Limits and derivatives appear throughout the chapter. " * 20,
            [{"filename": "math.pdf", "page": 1}],
            "",
        ),
    )


@pytest.mark.parametrize("count", [5, 10, 15, 20])
def test_generate_quiz_response_exact_counts(monkeypatch, count):
    """Requests for 5/10/15/20 must return exactly that many interactive questions."""
    from app.rag.utils import quiz as quiz_mod

    calls: list[int] = []

    def fake_llm_json(*, system, user, llm_provider, max_tokens=3072):
        import re as _re

        m = _re.search(r"Requested question count:\s*(\d+)", user)
        n = int(m.group(1)) if m else 3
        calls.append(n)
        stem = f"batch{len(calls)}"
        return _fake_mcq(n, prefix=stem)

    monkeypatch.setattr(quiz_mod, "_llm_json", fake_llm_json)
    _patch_material(monkeypatch, quiz_mod)

    result = quiz_mod.generate_quiz_response(
        f"Generate a quiz with {count} questions from this PDF",
        "user-1",
        document_id="doc-1",
        llm_provider="ollama",
    )
    quiz = result["quiz"]
    assert quiz is not None, result.get("answer")
    assert len(quiz["questions"]) == count
    assert f"{count}-question" in result["answer"]
    # Small quizzes one-shot the full count; larger builds start at FILL batch size
    # (never the old hard-coded 3).
    if count <= QUIZ_FILL_BATCH_SIZE:
        assert calls[0] == count
    else:
        assert calls[0] == QUIZ_FILL_BATCH_SIZE
        assert sum(calls) >= count


def test_generate_quiz_response_fills_after_partial_batch(monkeypatch):
    """If an early batch returns fewer than asked, later batches fill to the target."""
    from app.rag.utils import quiz as quiz_mod

    calls: list[int] = []

    def fake_llm_json(*, system, user, llm_provider, max_tokens=3072):
        import re as _re

        m = _re.search(r"Requested question count:\s*(\d+)", user)
        n = int(m.group(1)) if m else 3
        calls.append(n)
        # First batch: pretend the model only emitted 3 of 5 (truncation).
        if len(calls) == 1:
            return _fake_mcq(3, prefix="first")
        stem = f"fill{len(calls)}"
        return _fake_mcq(n, prefix=stem)

    monkeypatch.setattr(quiz_mod, "_llm_json", fake_llm_json)
    _patch_material(monkeypatch, quiz_mod)

    result = quiz_mod.generate_quiz_response(
        "Generate a quiz with 10 questions",
        "user-1",
        document_id="doc-1",
        llm_provider="ollama",
    )
    quiz = result["quiz"]
    assert quiz is not None
    assert len(quiz["questions"]) == 10
    assert calls[0] == QUIZ_FILL_BATCH_SIZE
    assert sum(calls) >= 10


def test_generate_quiz_response_no_silent_short_quiz(monkeypatch):
    """Under-generation must return a clear error with quiz=None — not a 3-question payload."""
    from app.rag.utils import quiz as quiz_mod

    def fake_llm_json(*, system, user, llm_provider, max_tokens=3072):
        # Always return only 3 unique questions (duplicates on later calls).
        return _fake_mcq(3, prefix="stuck")

    monkeypatch.setattr(quiz_mod, "_llm_json", fake_llm_json)
    _patch_material(monkeypatch, quiz_mod)

    result = quiz_mod.generate_quiz_response(
        "Generate a quiz with 10 questions",
        "user-1",
        document_id="doc-1",
        llm_provider="ollama",
    )
    assert result["quiz"] is None
    assert "10" in result["answer"]
    assert "could only generate" in result["answer"].lower() or "couldn't generate" in result[
        "answer"
    ].lower()


def test_generate_quiz_response_batches_to_requested_count(monkeypatch):
    """Asking for 6 must reach 6 via one-shot (or fill), never stop at 3."""
    from app.rag.utils import quiz as quiz_mod

    calls: list[int] = []

    def fake_llm_json(*, system, user, llm_provider, max_tokens=3072):
        import re as _re

        m = _re.search(r"Requested question count:\s*(\d+)", user)
        n = int(m.group(1)) if m else 3
        calls.append(n)
        stem = f"batch{len(calls)}"
        return _fake_mcq(n, prefix=stem)

    monkeypatch.setattr(quiz_mod, "_llm_json", fake_llm_json)
    _patch_material(monkeypatch, quiz_mod)

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
    # 6 > FILL size → first batch asks for 5, then a remainder batch of 1.
    assert calls[0] == QUIZ_FILL_BATCH_SIZE
    assert sum(calls) >= 6


def test_assessment_intent_routes_to_quiz_generation(monkeypatch):
    from app.rag.utils import quiz as quiz_mod

    calls: list[int] = []

    def fake_llm_json(*, system, user, llm_provider, max_tokens=3072):
        import re as _re

        m = _re.search(r"Requested question count:\s*(\d+)", user)
        n = int(m.group(1)) if m else 5
        calls.append(n)
        return _fake_mcq(n, prefix=f"assess{len(calls)}")

    monkeypatch.setattr(quiz_mod, "_llm_json", fake_llm_json)
    _patch_material(monkeypatch, quiz_mod)

    result = quiz_mod.generate_quiz_response(
        "Give me a 20-question assessment.",
        "user-1",
        document_id="doc-1",
        llm_provider="ollama",
    )
    assert result["quiz"] is not None
    assert len(result["quiz"]["questions"]) == 20
    assert calls[0] == QUIZ_FILL_BATCH_SIZE


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
        return {
            "answer": "quiz intro",
            "sources": [],
            "quiz": {
                "id": "x",
                "mode": "document",
                "title": "t",
                "questions": [],
                "state": {},
            },
        }

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

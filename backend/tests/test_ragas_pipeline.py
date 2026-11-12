import json
from types import SimpleNamespace

from app.evaluation import ragas_pipeline
from app.evaluation.ragas_pipeline import (
    EvaluationQuestion,
    collect_records,
    load_questions,
    summarize_ragas_result,
)


def test_load_questions_requires_exact_limit(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    rows = [
        {"id": "q1", "question": "Question 1?", "reference": "Reference 1."},
        {"id": "q2", "question": "Question 2?", "reference": "Reference 2."},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    questions = load_questions(dataset, limit=2)

    assert [question.id for question in questions] == ["q1", "q2"]
    assert questions[0].question == "Question 1?"


def test_collect_records_builds_vector_samples(monkeypatch):
    questions = [
        EvaluationQuestion(id="q1", question="What is Alpha?", reference="Alpha is a product."),
    ]

    monkeypatch.setattr(
        ragas_pipeline,
        "retrieve_vector_contexts",
        lambda **_kwargs: ["Alpha vector context."],
    )

    records = collect_records(
        questions=questions,
        user_id="user-1",
        answer_generator=lambda question, contexts: f"{question} -> {len(contexts)} contexts",
    )

    assert records["vector"][0].mode == "vector"
    assert records["vector"][0].response.endswith("1 contexts")
    assert "graphrag" not in records


def test_summarize_ragas_result_averages_score_rows():
    result = SimpleNamespace(
        scores=[
            {"faithfulness": 1.0, "context_recall": 0.5},
            {"faithfulness": 0.5, "context_recall": 1.0},
        ]
    )

    assert summarize_ragas_result(result) == {
        "faithfulness": 0.75,
        "context_recall": 0.75,
    }


"""Integration tests for RAGAS metric execution (optional / slow).

Run separately from ordinary unit tests:
  pytest tests/test_ragas_integration.py -m ragas_integration
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.ragas_integration


def test_ragas_evaluate_with_mock_llm(monkeypatch):
    """Smoke-test EvaluationDataset + evaluate wiring with a stub judge LLM."""
    from app.rag.evaluation.ragas_compat import ensure_ragas_langchain_compat

    ensure_ragas_langchain_compat()

    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    from langchain_core.embeddings import FakeEmbeddings

    # Faithfulness issues multiple LLM calls; provide enough canned replies.
    replies = [
        '{"statements": ["The dose is 150 mg."]}',
        '{"verdicts": [{"statement": "The dose is 150 mg.", "verdict": 1, "reason": "supported"}]}',
    ] * 4
    llm = LangchainLLMWrapper(FakeListChatModel(responses=replies))
    embeddings = LangchainEmbeddingsWrapper(FakeEmbeddings(size=8))

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": "What is the dose?",
                "retrieved_contexts": ["Table row: dose 150 mg every 4 weeks."],
                "response": "The dose is 150 mg.",
                "reference": "150 mg",
            }
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness()],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    scores = result.to_pandas() if hasattr(result, "to_pandas") else None
    assert scores is not None
    assert "faithfulness" in scores.columns

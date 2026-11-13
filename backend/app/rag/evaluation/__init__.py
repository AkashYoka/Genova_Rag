"""RAGAS + table-specific evaluation against the live Genova RAG pipeline."""

from app.rag.evaluation.ragas_compat import ensure_ragas_langchain_compat

ensure_ragas_langchain_compat()

__all__ = [
    "ensure_ragas_langchain_compat",
    "evaluate_dataset",
    "run_live_pipeline_case",
]


def __getattr__(name: str):
    if name == "evaluate_dataset":
        from app.rag.evaluation.runner import evaluate_dataset

        return evaluate_dataset
    if name == "run_live_pipeline_case":
        from app.rag.evaluation.pipeline_adapter import run_live_pipeline_case

        return run_live_pipeline_case
    raise AttributeError(name)

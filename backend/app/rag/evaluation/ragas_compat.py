"""Compatibility shims so RAGAS can import on modern langchain-community.

RAGAS 0.2.x imports ``langchain_community.chat_models.vertexai.ChatVertexAI``,
which was removed from recent langchain-community builds. We only need the
symbol present at import time; evaluation uses the project's Bedrock/Ollama
LLM wrapper, not Vertex AI.
"""
from __future__ import annotations

import sys
import types
from typing import Any


def ensure_ragas_langchain_compat() -> None:
    """Install stubs for removed langchain_community Vertex AI chat exports."""
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:
        __import__(mod_name)
        return
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(mod_name)

    class ChatVertexAI:  # pragma: no cover - import-time stub only
        """Placeholder for RAGAS import compatibility."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "ChatVertexAI stub is not usable. Configure RAGAS with the "
                "project Bedrock/Ollama LLM wrapper instead."
            )

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = stub

    # Ensure parent package attribute exists when accessed dynamically.
    try:
        import langchain_community.chat_models as chat_models

        if not hasattr(chat_models, "vertexai"):
            chat_models.vertexai = stub  # type: ignore[attr-defined]
    except Exception:
        pass

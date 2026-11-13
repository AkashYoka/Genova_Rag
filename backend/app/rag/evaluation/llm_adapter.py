"""Project LLM / embedding adapters for RAGAS judge metrics."""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field


def _message_to_role_content(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = getattr(message, "type", "user") or "user"
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
    content = message.content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    return {"role": role, "content": str(content or "")}


class ProjectChatModel(BaseChatModel):
    """LangChain chat model that calls Genova ``get_llm_client().chat_completion``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_provider: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    model_id: str = Field(default="")

    @property
    def _llm_type(self) -> str:
        return "genova_project_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from app.rag.llm.agent import get_llm_client, llm_model_for_provider

        client = get_llm_client(llm_provider=self.llm_provider)
        model = self.model_id or llm_model_for_provider(self.llm_provider)
        payload = [_message_to_role_content(m) for m in messages]
        response = client.chat_completion(
            messages=payload,
            model=model,
            max_tokens=int(kwargs.get("max_tokens", self.max_tokens)),
            temperature=float(kwargs.get("temperature", self.temperature)),
            stream=False,
        )
        text = ""
        if getattr(response, "choices", None):
            text = (response.choices[0].message.content or "").strip()
        elif isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                text = str(msg.get("content") or "").strip()
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])


class ProjectEmbeddings(Embeddings):
    """LangChain embeddings backed by Genova ``embed_texts`` / ``embed_query``."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from app.rag.embeddings.embeddings import embed_texts

        return embed_texts(list(texts or []))

    def embed_query(self, text: str) -> List[float]:
        from app.rag.embeddings.embeddings import embed_query

        return embed_query(text or "")


def build_ragas_llm(*, llm_provider: Optional[str] = None):
    from app.rag.evaluation.ragas_compat import ensure_ragas_langchain_compat

    ensure_ragas_langchain_compat()
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ProjectChatModel(llm_provider=llm_provider, temperature=0.0)
    )


def build_ragas_embeddings():
    from app.rag.evaluation.ragas_compat import ensure_ragas_langchain_compat

    ensure_ragas_langchain_compat()
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(ProjectEmbeddings())

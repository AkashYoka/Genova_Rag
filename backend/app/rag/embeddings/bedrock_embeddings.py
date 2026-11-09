"""Amazon Bedrock Titan Text Embeddings V2.

Uses ``amazon.titan-embed-text-v2:0`` (default 1024-d). Auth follows the same
bearer-token-or-IAM path as ``BedrockNovaChatClient``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List

from app.rag.llm.bedrock_client import (
    BedrockAuthError,
    BedrockProviderError,
    create_bedrock_runtime_client,
    _aws_error_fields,
)

logger = logging.getLogger(__name__)

DEFAULT_TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
DEFAULT_TITAN_DIMENSION = 1024


class BedrockEmbeddings:
    """LangChain-compatible embedder over Bedrock Titan Text Embeddings V2."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        dimensions: int | None = None,
        client: Any | None = None,
    ):
        from app.config import get_settings

        settings = get_settings()
        self.model_id = (
            model_id
            or getattr(settings, "EMBEDDING_MODEL", None)
            or DEFAULT_TITAN_EMBED_V2
        )
        self.region = (
            region or getattr(settings, "BEDROCK_REGION", None) or "us-east-1"
        )
        self.dimensions = int(
            dimensions
            or getattr(settings, "EMBEDDING_DIMENSION", None)
            or DEFAULT_TITAN_DIMENSION
        )
        self._client = client

    def _client_or_create(self):
        if self._client is not None:
            return self._client
        self._client = create_bedrock_runtime_client(self.region)
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> List[float]:
        payload = {
            "inputText": text or " ",
            "dimensions": self.dimensions,
            "normalize": True,
        }
        client = self._client_or_create()
        try:
            response = client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            body = response.get("body")
            raw = body.read() if hasattr(body, "read") else body
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception as exc:
            raise self._map_error(exc) from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise BedrockProviderError(
                f"Bedrock Titan embed returned no embedding for model {self.model_id!r}.",
                code="EmptyEmbedding",
                kind="provider_error",
            )
        return [float(x) for x in embedding]

    def _map_error(self, exc: Exception) -> Exception:
        name = type(exc).__name__
        message = str(exc)
        lowered = message.lower()
        code, http_status, aws_message = _aws_error_fields(exc)
        code_l = (code or "").lower()
        aws_l = (aws_message or "").lower()
        if (
            name
            in {
                "ExpiredTokenException",
                "UnrecognizedClientException",
                "AccessDeniedException",
            }
            or code
            in {
                "ExpiredTokenException",
                "UnrecognizedClientException",
                "AccessDeniedException",
            }
            or any(
                token in lowered or token in aws_l or token in code_l
                for token in (
                    "expired",
                    "expiration",
                    "unrecognizedclient",
                    "invalidsecuritytoken",
                    "not authorized",
                    "unauthorized",
                    "accessdenied",
                    "forbidden",
                    "security token",
                )
            )
        ):
            return BedrockAuthError(
                "Bedrock embedding authentication failed: credentials are missing, "
                f"expired, or invalid for region {self.region} / model {self.model_id}."
                + (
                    f" (HTTP {http_status}, {code})"
                    if http_status or code
                    else ""
                )
            )
        return BedrockProviderError(
            f"Bedrock Titan embed failed ({code or name}).",
            code=code or name,
            http_status=http_status,
            kind="provider_error",
        )

"""Amazon Bedrock Titan Text Embeddings V2.

Uses ``amazon.titan-embed-text-v2:0`` (default 1024-d). Auth follows the same
bearer-token-or-IAM path as ``BedrockNovaChatClient``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional, Tuple

from app.rag.llm.bedrock_client import (
    BedrockAuthError,
    BedrockProviderError,
    create_bedrock_runtime_client,
    _aws_error_fields,
)

logger = logging.getLogger(__name__)

DEFAULT_TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
DEFAULT_TITAN_DIMENSION = 1024
DEFAULT_TITAN_MAX_INPUT_TOKENS = 8192
# Titan rejects on *tokens*, not characters. AWS cites ~4.7 chars/token for
# English prose; dense medical tables / schedules are far denser (~2–3).
# A previous 24k char cap still produced 8323-token requests in production.
# Stay well under 8192 tokens for dense payloads.
DEFAULT_TITAN_MAX_INPUT_CHARS = 16_000
_TOKEN_OVERFLOW_RE = re.compile(
    r"Max input tokens:\s*(\d+).*?(?:request input )?token count:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)


class BedrockEmbeddings:
    """LangChain-compatible embedder over Bedrock Titan Text Embeddings V2."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        dimensions: int | None = None,
        client: Any | None = None,
        max_input_chars: int | None = None,
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
        configured_chars = getattr(settings, "EMBEDDING_MAX_INPUT_CHARS", None)
        self.max_input_chars = int(
            max_input_chars
            or configured_chars
            or DEFAULT_TITAN_MAX_INPUT_CHARS
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

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return " "
        if len(text) <= max_chars:
            return text
        # Prefer cutting on a newline / space so we don't mid-slice a cell.
        cut = text[:max_chars]
        for sep in ("\n", " ", "\t"):
            idx = cut.rfind(sep)
            if idx > max_chars * 0.7:
                return cut[:idx].rstrip() or cut
        return cut

    @staticmethod
    def _parse_token_overflow(message: str) -> Tuple[Optional[int], Optional[int]]:
        match = _TOKEN_OVERFLOW_RE.search(message or "")
        if not match:
            return None, None
        try:
            return int(match.group(1)), int(match.group(2))
        except (TypeError, ValueError):
            return None, None

    def _invoke(self, raw: str) -> List[float]:
        payload = {
            "inputText": raw,
            "dimensions": self.dimensions,
            "normalize": True,
        }
        client = self._client_or_create()
        response = client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = response.get("body")
        payload_raw = body.read() if hasattr(body, "read") else body
        if isinstance(payload_raw, bytes):
            payload_raw = payload_raw.decode("utf-8")
        data = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise BedrockProviderError(
                f"Bedrock Titan embed returned no embedding for model {self.model_id!r}.",
                code="EmptyEmbedding",
                kind="provider_error",
            )
        return [float(x) for x in embedding]

    def _embed_one(self, text: str) -> List[float]:
        raw = text or " "
        max_chars = max(1, int(self.max_input_chars))
        if len(raw) > max_chars:
            logger.warning(
                "Truncating Titan embed input from %s to %s chars (token budget)",
                len(raw),
                max_chars,
            )
            raw = self._truncate(raw, max_chars)

        attempts = 0
        while True:
            attempts += 1
            try:
                return self._invoke(raw)
            except Exception as exc:
                mapped = self._map_error(exc)
                detail = str(mapped)
                max_tokens, used_tokens = self._parse_token_overflow(detail)
                is_overflow = (
                    "too many input tokens" in detail.lower()
                    or (max_tokens is not None and used_tokens is not None)
                )
                if not is_overflow or attempts >= 4 or len(raw) <= 512:
                    raise mapped from exc

                if max_tokens and used_tokens and used_tokens > 0:
                    # Scale by observed token density, with 10% headroom.
                    ratio = (max_tokens / used_tokens) * 0.90
                    next_chars = max(512, int(len(raw) * ratio))
                else:
                    next_chars = max(512, int(len(raw) * 0.75))

                if next_chars >= len(raw):
                    next_chars = max(512, len(raw) - 512)

                logger.warning(
                    "Titan embed token overflow (used=%s max=%s); retrying with %s chars (was %s)",
                    used_tokens,
                    max_tokens,
                    next_chars,
                    len(raw),
                )
                raw = self._truncate(raw, next_chars)

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
        detail = (aws_message or message or "").strip()
        suffix = f": {detail[:300]}" if detail else ""
        return BedrockProviderError(
            f"Bedrock Titan embed failed ({code or name}){suffix}",
            code=code or name,
            http_status=http_status,
            kind="provider_error",
        )

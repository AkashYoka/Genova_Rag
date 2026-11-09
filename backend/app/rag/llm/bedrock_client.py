"""Amazon Bedrock Nova generation via the Converse API.

Auth preference:
1. ``AWS_BEARER_TOKEN_BEDROCK`` when set (API key / bearer),
2. otherwise the normal AWS credential chain (IAM role, env keys, shared config).

Credentials are never logged or embedded in error messages.

Mirrors the genova-rag BedrockNovaGenerator approach, adapted to the
OllamaChatClient / InferenceClient chat_completion contract used by agent.py.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator, List, Optional

from app.pricing import estimate_cost_usd, lookup_pricing

logger = logging.getLogger(__name__)

# Prevent accidental DEBUG dumps of Authorization headers from AWS SDK.
for _noisy in ("botocore", "boto3", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

DEFAULT_NOVA_2_LITE_PROFILE = "us.amazon.nova-2-lite-v1:0"

BEDROCK_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"

# Back-compat aliases — rates live in app.pricing registry.
_nova_rates = lookup_pricing("bedrock", DEFAULT_NOVA_2_LITE_PROFILE) or {}
NOVA_2_LITE_INPUT_USD_PER_1M = float(_nova_rates.get("input_usd_per_1m") or 0.33)
NOVA_2_LITE_OUTPUT_USD_PER_1M = float(_nova_rates.get("output_usd_per_1m") or 2.75)
PRICING_NOTE = _nova_rates.get("pricing_note") or (
    f"Estimated cost using US on-demand list rates "
    f"${NOVA_2_LITE_INPUT_USD_PER_1M}/1M input + "
    f"${NOVA_2_LITE_OUTPUT_USD_PER_1M}/1M output for {DEFAULT_NOVA_2_LITE_PROFILE}. "
    "This is not an AWS bill."
)


class BedrockAuthError(RuntimeError):
    """Raised when Bedrock credentials are missing, expired, or rejected."""


class BedrockProviderError(RuntimeError):
    """Raised for non-auth Bedrock failures with a safe, actionable message."""


def _optional_bearer_token() -> str:
    """Return AWS_BEARER_TOKEN_BEDROCK when present; otherwise empty (use IAM)."""
    token = (os.environ.get(BEDROCK_TOKEN_ENV) or "").strip()
    if token:
        return token
    token = _token_from_dotenv()
    if token:
        os.environ[BEDROCK_TOKEN_ENV] = token
    return token


def create_bedrock_runtime_client(region: str, client: Any | None = None) -> Any:
    """Build a bedrock-runtime client using bearer token or IAM credentials."""
    if client is not None:
        return client
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for Bedrock. Install with: pip install boto3"
        ) from exc

    # Populate env so boto3 can pick up the bearer token when one exists.
    # When absent, boto3 uses the default credential chain (instance/task role).
    _optional_bearer_token()
    return boto3.client("bedrock-runtime", region_name=region)

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
        kind: str = "provider_error",
    ):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.kind = kind


def estimate_nova2_lite_cost_usd(
    input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Back-compat helper; prefer the shared pricing registry."""
    result = estimate_cost_usd(
        provider="bedrock",
        model_id=DEFAULT_NOVA_2_LITE_PROFILE,
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
    )
    return result.get("estimated_cost_usd")


def usage_from_bedrock(
    usage: dict | None, *, model_id: str | None = None
) -> dict[str, Any]:
    usage = usage or {}
    input_tokens = usage.get("inputTokens", usage.get("input_tokens"))
    output_tokens = usage.get("outputTokens", usage.get("output_tokens"))
    cache_read = usage.get("cacheReadInputTokens", usage.get("cache_read_tokens"))
    cache_write = usage.get("cacheWriteInputTokens", usage.get("cache_write_tokens"))
    total = usage.get("totalTokens", usage.get("total_tokens"))
    mid = model_id or DEFAULT_NOVA_2_LITE_PROFILE
    if total is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        extra = cache_read if isinstance(cache_read, int) else 0
        total = input_tokens + output_tokens + extra
    estimate = estimate_cost_usd(
        provider="bedrock",
        model_id=mid,
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        cache_read_tokens=cache_read if isinstance(cache_read, int) else None,
        cache_write_tokens=cache_write if isinstance(cache_write, int) else None,
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_tokens": total,
        "estimated_cost_usd": estimate.get("estimated_cost_usd"),
        "pricing_note": estimate.get("pricing_note") or PRICING_NOTE,
        "cost_type": estimate.get("cost_type") or "estimated",
        "model_id": mid,
        "provider": "bedrock",
    }


class BedrockNovaChatClient:
    """Nova 2 Lite through bedrock-runtime Converse (InferenceClient-compatible)."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        client: Any | None = None,
    ):
        from app.config import get_settings

        settings = get_settings()
        self.model_id = (
            model_id
            or getattr(settings, "BEDROCK_MODEL_ID", None)
            or DEFAULT_NOVA_2_LITE_PROFILE
        )
        self.region = (
            region or getattr(settings, "BEDROCK_REGION", None) or "us-east-1"
        )
        self.model = self.model_id
        self._client = client
        self.last_usage: dict[str, Any] = {}

    def _client_or_create(self):
        if self._client is not None:
            return self._client
        self._client = create_bedrock_runtime_client(self.region)
        return self._client

    def chat_completion(
        self,
        *,
        messages: List[dict],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stream: bool = False,
        **_kwargs: Any,
    ):
        if stream:
            return self._stream(messages, model=model, max_tokens=max_tokens, temperature=temperature)

        client = self._client_or_create()
        system, converse_messages = _split_messages(messages)
        inference = {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
        }
        kwargs: dict[str, Any] = {
            "modelId": model or self.model_id,
            "messages": converse_messages,
            "inferenceConfig": inference,
        }
        if system:
            kwargs["system"] = system
        try:
            response = client.converse(**kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        used_model = model or self.model_id
        self.last_usage = usage_from_bedrock(
            response.get("usage"), model_id=used_model
        )
        content = _text_from_converse(response)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def _stream(
        self,
        messages: List[dict],
        *,
        model: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Generator[Any, None, None]:
        """Yield answer chunks for the chat UI with Bedrock usage metadata.

        Uses non-stream ``converse`` (same as genova-rag) so ``usage.inputTokens`` /
        ``outputTokens`` are always available for estimated cost, then chunks the
        completed text for progressive rendering without a second billed call.
        """
        # Reuse the non-stream path for accurate usage accounting.
        response = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        content = ""
        if response and getattr(response, "choices", None):
            content = response.choices[0].message.content or ""
        # Pace faux-stream chunks so the UI can paint progressively instead of
        # receiving the whole answer in one event-loop burst.
        chunk_size = 24
        for i in range(0, len(content), chunk_size):
            piece = content[i : i + chunk_size]
            if piece:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))]
                )
                time.sleep(0.012)

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
                    "api key is valid",
                    "security token",
                )
            )
        ):
            return BedrockAuthError(
                "Bedrock authentication failed: credentials are missing, expired, or invalid "
                f"for region {self.region} / model {self.model_id}. "
                f"Use an IAM role with Bedrock access, or refresh {BEDROCK_TOKEN_ENV}."
                + (
                    f" (HTTP {http_status}, {code})"
                    if http_status or code
                    else ""
                )
            )
        if name == "ValidationException" or code == "ValidationException" or "validation" in lowered:
            return BedrockProviderError(
                f"Bedrock rejected model id {self.model_id!r} or the request shape in {self.region}. "
                "Use the US inference profile us.amazon.nova-2-lite-v1:0 (typically us-east-1).",
                code=code or "ValidationException",
                http_status=http_status,
                kind="validation",
            )
        if name in {"ThrottlingException", "TooManyRequestsException"} or code in {
            "ThrottlingException",
            "TooManyRequestsException",
        }:
            return BedrockProviderError(
                "Bedrock throttled the request. Wait briefly and retry.",
                code=code or name,
                http_status=http_status,
                kind="throttling",
            )
        return BedrockProviderError(
            f"Bedrock Converse failed ({code or name}).",
            code=code or name,
            http_status=http_status,
            kind="provider_error",
        )


def _split_messages(messages: List[dict]) -> tuple[list[dict], list[dict]]:
    system: list[dict] = []
    converse: list[dict] = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        if role == "system":
            system.append({"text": content})
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        converse.append({"role": role, "content": [{"text": content}]})
    if not converse:
        converse = [{"role": "user", "content": [{"text": ""}]}]
    return system, converse


def _text_from_converse(response: dict) -> str:
    content = ((response.get("output") or {}).get("message") or {}).get("content") or []
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _aws_error_fields(exc: Exception) -> tuple[str | None, int | None, str | None]:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return type(exc).__name__, None, str(exc)
    error = response.get("Error") or {}
    meta = response.get("ResponseMetadata") or {}
    code = error.get("Code")
    message = error.get("Message")
    status = meta.get("HTTPStatusCode")
    return (
        str(code) if code else type(exc).__name__,
        int(status) if isinstance(status, int) else None,
        str(message) if message else None,
    )


def _token_from_dotenv() -> str:
    """Read AWS_BEARER_TOKEN_BEDROCK from gitignored .env without printing it."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",  # backend/.env
        Path(__file__).resolve().parents[3] / ".env",  # repo root .env
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, _, value = raw.partition("=")
                if key.strip() != BEDROCK_TOKEN_ENV:
                    continue
                return value.strip().strip("'").strip('"')
        except OSError:
            continue
    return ""

"""Unit tests for Bedrock Titan embeddings + IAM credential fallback."""
from unittest.mock import MagicMock

import pytest

from app.rag.embeddings.bedrock_embeddings import (
    BedrockEmbeddings,
    DEFAULT_TITAN_MAX_INPUT_CHARS,
)
from app.rag.llm.bedrock_client import create_bedrock_runtime_client


def test_bedrock_embeddings_invoke_model():
    fake_vector = [0.1] * 8
    body = MagicMock()
    body.read.return_value = b'{"embedding": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}'
    client = MagicMock()
    client.invoke_model.return_value = {"body": body}

    emb = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region="us-east-1",
        dimensions=8,
        client=client,
    )
    docs = emb.embed_documents(["alpha"])
    query = emb.embed_query("beta")
    assert docs == [fake_vector]
    assert query == fake_vector
    assert client.invoke_model.call_count == 2
    kwargs = client.invoke_model.call_args_list[0].kwargs
    assert kwargs["modelId"] == "amazon.titan-embed-text-v2:0"


def test_create_bedrock_runtime_client_without_bearer(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    fake_boto = MagicMock()
    fake_client = object()
    fake_boto.client.return_value = fake_client

    import sys

    monkeypatch.setitem(sys.modules, "boto3", fake_boto)
    result = create_bedrock_runtime_client("us-east-1")
    assert result is fake_client
    fake_boto.client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")


def test_embed_truncates_oversized_char_budget():
    body = MagicMock()
    body.read.return_value = b'{"embedding": [0.2, 0.2]}'
    client = MagicMock()
    client.invoke_model.return_value = {"body": body}
    emb = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region="us-east-1",
        dimensions=2,
        client=client,
        max_input_chars=200,
    )
    huge = ("dose 150 mg weight band\n" * 40)
    emb.embed_documents([huge])
    body_json = client.invoke_model.call_args.kwargs["body"]
    import json

    payload = json.loads(body_json)
    assert len(payload["inputText"]) <= 200
    assert len(huge) > 200


def test_embed_retries_on_token_overflow():
    ok_body = MagicMock()
    ok_body.read.return_value = b'{"embedding": [0.3, 0.3]}'

    class Overflow(Exception):
        pass

    client = MagicMock()

    def _invoke(**_kwargs):
        if client.invoke_model.call_count == 1:
            raise Overflow(
                "ValidationException: 400 Bad Request: Too many input tokens. "
                "Max input tokens: 8192, request input token count: 8323"
            )
        return {"body": ok_body}

    client.invoke_model.side_effect = _invoke
    emb = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region="us-east-1",
        dimensions=2,
        client=client,
        max_input_chars=20_000,
    )
    text = "cell value 150 mg " * 800  # dense-ish medical table prose
    vector = emb.embed_query(text)
    assert vector == [0.3, 0.3]
    assert client.invoke_model.call_count == 2
    import json

    first = json.loads(client.invoke_model.call_args_list[0].kwargs["body"])
    second = json.loads(client.invoke_model.call_args_list[1].kwargs["body"])
    assert len(second["inputText"]) < len(first["inputText"])


def test_parse_token_overflow_message():
    max_t, used = BedrockEmbeddings._parse_token_overflow(
        "Too many input tokens. Max input tokens: 8192, request input token count: 8323"
    )
    assert max_t == 8192
    assert used == 8323
    assert DEFAULT_TITAN_MAX_INPUT_CHARS <= 20000

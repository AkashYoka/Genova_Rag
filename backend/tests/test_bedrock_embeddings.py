"""Unit tests for Bedrock Titan embeddings + IAM credential fallback."""
from unittest.mock import MagicMock

from app.rag.embeddings.bedrock_embeddings import BedrockEmbeddings
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

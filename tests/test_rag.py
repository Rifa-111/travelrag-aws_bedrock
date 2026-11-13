import io
import json
import numpy as np
import pytest
from src.rag import Retriever, TitanEmbedder, answer, chunk_documents


class FakeEmbedder:
    def embed(self, text):
        text = text.lower()
        return np.array([float("hotel" in text), float("meal" in text), float("rail" in text), float("flight" in text), float("receipt" in text)])


class FakeRuntime:
    def invoke_model(self, **kwargs):
        assert kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
        assert json.loads(kwargs["body"])["normalize"] is True
        return {"body": io.BytesIO(json.dumps({"embedding": [1.0, 0.0]}).encode())}


def test_titan_request():
    assert TitanEmbedder(client=FakeRuntime()).embed("hotel").tolist() == [1.0, 0.0]


def test_semantic_retriever_with_mock_embeddings():
    retriever = Retriever([{"source": "a", "text": "hotel policy"}, {"source": "b", "text": "meal policy"}], FakeEmbedder())
    assert retriever.search("hotel") [0]["source"] == "a"
    assert retriever.search("unrelated", threshold=0.35) == []


def test_abstention():
    retriever = Retriever([{"source": "a", "text": "hotel policy"}], FakeEmbedder())
    assert answer("unrelated", retriever)["mode"] == "abstain"


def test_validation():
    retriever = Retriever([{"source": "a", "text": "hotel policy"}], FakeEmbedder())
    with pytest.raises(ValueError):
        answer("", retriever)


def test_document_chunking():
    assert chunk_documents("data")

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from src.embeddings import EmbeddingService
from src.retrieval import RetrievalService, format_retrieval_results
from src.vector_store import RetrievedChunk, VectorStore


class StubEmbeddings:
    model_name = "test-model"
    model_revision = "test-revision"
    dimension = 2

    def __init__(self) -> None:
        self.last_question: str | None = None
        self.vector = np.array([1.0, 0.0], dtype=np.float32)

    def embed_query(self, question: str) -> np.ndarray:
        self.last_question = question
        return self.vector


class StubVectorStore:
    embedding_model = "test-model"
    embedding_revision = "test-revision"
    embedding_dimension = 2
    metadata_compatible = True

    def __init__(self, matches: list[RetrievedChunk]) -> None:
        self.matches = matches
        self.last_embedding: np.ndarray | None = None
        self.last_n_results: int | None = None

    def query(
        self,
        query_embedding: np.ndarray,
        *,
        n_results: int = 5,
    ) -> list[RetrievedChunk]:
        self.last_embedding = query_embedding
        self.last_n_results = n_results
        return self.matches[:n_results]


def make_match() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        text="--suite and --config are required for test runs.",
        metadata={
            "heading_path": "Regression Manager > Required arguments",
            "section_index": 0,
            "chunk_index": 0,
        },
        distance=0.1234,
    )


def make_service(
    embeddings: StubEmbeddings | None = None,
    store: StubVectorStore | None = None,
) -> tuple[RetrievalService, StubEmbeddings, StubVectorStore]:
    embeddings = embeddings or StubEmbeddings()
    store = store or StubVectorStore([make_match()])
    service = RetrievalService(
        cast(EmbeddingService, embeddings),
        cast(VectorStore, store),
    )
    return service, embeddings, store


def test_retrieve_embeds_question_and_forwards_top_k() -> None:
    expected = make_match()
    service, embeddings, store = make_service(store=StubVectorStore([expected]))

    results = service.retrieve("What arguments are required?", top_k=3)

    assert embeddings.last_question == "What arguments are required?"
    assert np.array_equal(store.last_embedding, embeddings.vector)
    assert store.last_n_results == 3
    assert results == [expected]


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("model_name", "another-model"),
        ("model_revision", "another-revision"),
        ("dimension", 3),
    ],
)
def test_rejects_embedding_configuration_mismatch(
    attribute: str,
    value: str | int,
) -> None:
    embeddings = StubEmbeddings()
    setattr(embeddings, attribute, value)

    with pytest.raises(ValueError, match="same model, revision, and dimension"):
        make_service(embeddings=embeddings)


def test_rejects_incompatible_collection() -> None:
    store = StubVectorStore([make_match()])
    store.metadata_compatible = False

    with pytest.raises(ValueError, match="incompatible embedding metadata"):
        make_service(store=store)


@pytest.mark.parametrize(
    ("question", "top_k", "message"),
    [("", 1, "question"), ("valid question", 0, "top_k")],
)
def test_rejects_invalid_retrieval_request(
    question: str,
    top_k: int,
    message: str,
) -> None:
    service, _, _ = make_service()

    with pytest.raises(ValueError, match=message):
        service.retrieve(question, top_k=top_k)


def test_formats_section_text_count_and_distance() -> None:
    formatted = format_retrieval_results("What is required?", [make_match()])

    assert "Retrieved chunks: 1" in formatted
    assert "Regression Manager > Required arguments" in formatted
    assert "Cosine distance: 0.1234" in formatted
    assert "--suite and --config" in formatted


def test_formats_empty_retrieval_explicitly() -> None:
    formatted = format_retrieval_results("Unsupported question", [])

    assert "Retrieved chunks: 0" in formatted
    assert "No chunks retrieved." in formatted

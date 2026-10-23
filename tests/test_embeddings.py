from __future__ import annotations

import numpy as np
import pytest

from src.chunking import DocumentChunk
from src.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    EmbeddingService,
)


class FakeTokenizer:
    def __init__(self, max_sequence_length: int) -> None:
        self.max_sequence_length = max_sequence_length

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens
        assert not truncation
        return {
            "input_ids": [
                [1] * (self.max_sequence_length + 1 if "over-limit" in text else 3)
                for text in texts
            ]
        }


class FakeModel:
    max_seq_length = 4

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer(self.max_seq_length)
        self.encode_calls = 0
        self.encode_options: dict[str, object] = {}

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **options: object) -> np.ndarray:
        self.encode_calls += 1
        self.encode_options = options
        vector = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        return np.tile(vector, (len(texts), 1))


@pytest.fixture
def fake_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeModel, dict[str, object]]:
    model = FakeModel()
    constructor_options: dict[str, object] = {}

    def create_model(model_name: str, **options: object) -> FakeModel:
        constructor_options["model_name"] = model_name
        constructor_options.update(options)
        return model

    monkeypatch.setattr("src.embeddings.SentenceTransformer", create_model)
    return model, constructor_options


def test_loads_pinned_model_locally_on_cpu(
    fake_model: tuple[FakeModel, dict[str, object]],
) -> None:
    model, options = fake_model

    service = EmbeddingService()

    assert service.model_name == DEFAULT_EMBEDDING_MODEL
    assert service.model_revision == DEFAULT_EMBEDDING_REVISION
    assert service.dimension == model.get_embedding_dimension()
    assert options == {
        "model_name": DEFAULT_EMBEDDING_MODEL,
        "revision": DEFAULT_EMBEDDING_REVISION,
        "device": "cpu",
        "local_files_only": True,
    }


def test_embeds_chunks_and_queries_with_normalized_vectors(
    fake_model: tuple[FakeModel, dict[str, object]],
) -> None:
    model, _ = fake_model
    service = EmbeddingService()
    chunks = [
        DocumentChunk("chunk-1", 0, 0, "Section", ("Section",), "First text."),
        DocumentChunk("chunk-2", 0, 1, "Section", ("Section",), "Second text."),
    ]

    chunk_vectors = service.embed_chunks(chunks)
    query_vector = service.embed_query("A query")

    assert chunk_vectors.shape == (2, 3)
    assert query_vector.shape == (3,)
    assert np.allclose(np.linalg.norm(chunk_vectors, axis=1), 1.0)
    assert np.isclose(np.linalg.norm(query_vector), 1.0)
    assert model.encode_options["batch_size"] == 32
    assert model.encode_options["normalize_embeddings"] is True
    assert model.encode_options["convert_to_numpy"] is True
    assert model.encode_options["show_progress_bar"] is False


def test_rejects_text_over_model_token_limit(
    fake_model: tuple[FakeModel, dict[str, object]],
) -> None:
    model, _ = fake_model
    service = EmbeddingService()

    with pytest.raises(ValueError, match="wordpiece limit"):
        service.embed_query("over-limit")

    assert model.encode_calls == 0


@pytest.mark.parametrize("texts", [[], ["  "]])
def test_rejects_empty_text_input(
    fake_model: tuple[FakeModel, dict[str, object]],
    texts: list[str],
) -> None:
    service = EmbeddingService()

    with pytest.raises(ValueError):
        service.embed_texts(texts)


def test_missing_local_model_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_model(model_name: str, **options: object) -> FakeModel:
        raise FileNotFoundError("model files not found")

    monkeypatch.setattr("src.embeddings.SentenceTransformer", missing_model)

    with pytest.raises(FileNotFoundError, match="not available in the local cache"):
        EmbeddingService()

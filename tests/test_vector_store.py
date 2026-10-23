from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.chunking import DocumentChunk
from src.vector_store import VectorStore


EMBEDDING_MODEL = "test-model"
EMBEDDING_REVISION = "test-revision"
EMBEDDING_DIMENSION = 2


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk-1",
            section_index=0,
            chunk_index=0,
            heading="Required arguments",
            heading_path=("Regression Manager", "Required arguments"),
            text="Suite and configuration are required.",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            section_index=1,
            chunk_index=0,
            heading="Exit codes",
            heading_path=("Regression Manager", "Exit codes"),
            text="Exit code two means invalid arguments.",
        ),
    ]


def make_embeddings() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def make_store(path: Path, revision: str = EMBEDDING_REVISION) -> VectorStore:
    return VectorStore(
        embedding_model=EMBEDDING_MODEL,
        embedding_revision=revision,
        embedding_dimension=EMBEDDING_DIMENSION,
        persist_directory=path,
        collection_name="test-docs",
    )


def test_persistent_upsert_query_and_metadata(tmp_path: Path) -> None:
    chunks = make_chunks()
    embeddings = make_embeddings()
    store = make_store(tmp_path / "chroma")

    assert store.metadata_compatible
    assert store.query(embeddings[0]) == []
    store.upsert_chunks(chunks, embeddings)
    store.upsert_chunks(chunks, embeddings)
    assert store.count() == 2

    reopened = make_store(tmp_path / "chroma")
    assert reopened.count() == 2
    results = reopened.query(embeddings[0], n_results=1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].metadata["heading_path"] == "Regression Manager > Required arguments"
    assert results[0].distance < 1e-5


def test_model_revision_mismatch_blocks_query_until_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    chunks = make_chunks()
    embeddings = make_embeddings()
    make_store(path).rebuild_index(chunks, embeddings)

    changed_model_store = make_store(path, revision="new-revision")
    assert not changed_model_store.metadata_compatible
    with pytest.raises(ValueError, match="different embedding metadata"):
        changed_model_store.query(embeddings[0])

    changed_model_store.rebuild_index(chunks[:1], embeddings[:1])

    assert changed_model_store.metadata_compatible
    assert changed_model_store.count() == 1


def test_invalid_rebuild_does_not_delete_existing_index(tmp_path: Path) -> None:
    chunks = make_chunks()
    store = make_store(tmp_path / "chroma")
    store.upsert_chunks(chunks, make_embeddings())

    with pytest.raises(ValueError, match="Expected embedding shape"):
        store.rebuild_index(chunks, make_embeddings()[:1])

    assert store.count() == 2


def test_rejects_wrong_query_dimension_and_nonpositive_result_count(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "chroma")

    with pytest.raises(ValueError, match="Expected query shape"):
        store.query(np.array([1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="n_results"):
        store.query(np.array([1.0, 0.0], dtype=np.float32), n_results=0)

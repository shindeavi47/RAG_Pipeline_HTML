from __future__ import annotations

from typing import Sequence

from .embeddings import EmbeddingService
from .vector_store import RetrievedChunk, VectorStore


class RetrievalService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        if not vector_store.metadata_compatible:
            raise ValueError(
                "The Chroma collection has incompatible embedding metadata; "
                "rebuild it before retrieval."
            )
        if (
            embeddings.model_name != vector_store.embedding_model
            or embeddings.model_revision != vector_store.embedding_revision
            or embeddings.dimension != vector_store.embedding_dimension
        ):
            raise ValueError(
                "Embedding service and Chroma collection must use the same "
                "model, revision, and dimension."
            )

        self._embeddings = embeddings
        self._vector_store = vector_store

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("question must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self._embeddings.embed_query(question)
        return self._vector_store.query(query_embedding, n_results=top_k)


def format_retrieval_results(
    question: str,
    matches: Sequence[RetrievedChunk],
) -> str:
    if not question.strip():
        raise ValueError("question must not be empty.")

    lines = [f"Question: {question}", f"Retrieved chunks: {len(matches)}"]
    if not matches:
        lines.append("No chunks retrieved.")
        return "\n".join(lines)

    for rank, match in enumerate(matches, start=1):
        heading_path = match.metadata["heading_path"]
        if not isinstance(heading_path, str):
            raise TypeError("Retrieved chunk heading_path metadata must be text.")
        lines.extend(
            [
                f"\n[{rank}] Section: {heading_path}",
                f"Cosine distance: {match.distance:.4f}",
                match.text,
            ]
        )
    return "\n".join(lines)

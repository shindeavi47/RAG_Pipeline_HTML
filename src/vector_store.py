from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
import numpy as np
from chromadb.config import Settings

from .chunking import DocumentChunk


DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parents[1] / "storage" / "chroma"
DEFAULT_COLLECTION_NAME = "regression_manager_docs"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, object]
    distance: float


class VectorStore:
    def __init__(
        self,
        *,
        embedding_model: str,
        embedding_revision: str,
        embedding_dimension: int,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if not embedding_model.strip():
            raise ValueError("embedding_model must not be empty.")
        if not embedding_revision.strip():
            raise ValueError("embedding_revision must not be empty.")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero.")
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.embedding_dimension = embedding_dimension
        self._expected_metadata: dict[str, str | int] = {
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "embedding_dimension": embedding_dimension,
        }
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=self._expected_metadata,
            embedding_function=None,
        )
        existing_metadata = self._collection.metadata or {}
        self.metadata_compatible = all(
            existing_metadata.get(key) == value
            for key, value in self._expected_metadata.items()
        )

    def count(self) -> int:
        self._ensure_metadata_compatible()
        return self._collection.count()

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        self._ensure_metadata_compatible()
        self._validate_batch(chunks, embeddings)
        self._write_batch(chunks, embeddings)

    def rebuild_index(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        self._validate_batch(chunks, embeddings)
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata=self._expected_metadata,
            embedding_function=None,
        )
        self.metadata_compatible = True
        self._write_batch(chunks, embeddings)

    def query(
        self,
        query_embedding: np.ndarray,
        *,
        n_results: int = 5,
    ) -> list[RetrievedChunk]:
        self._ensure_metadata_compatible()
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("query_embedding must be a NumPy array.")
        if query_embedding.shape != (self.embedding_dimension,):
            raise ValueError(
                f"Expected query shape {(self.embedding_dimension,)}, "
                f"received {query_embedding.shape}."
            )
        if not np.issubdtype(query_embedding.dtype, np.number) or not np.isfinite(
            query_embedding
        ).all():
            raise ValueError("query_embedding must contain only finite numbers.")

        collection_count = self._collection.count()
        if collection_count == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, collection_count),
            include=["documents", "metadatas", "distances"],
        )
        ids_by_query = result["ids"]
        if not ids_by_query or not ids_by_query[0]:
            return []

        documents_by_query = result["documents"]
        metadatas_by_query = result["metadatas"]
        distances_by_query = result["distances"]
        if (
            documents_by_query is None
            or metadatas_by_query is None
            or distances_by_query is None
            or documents_by_query[0] is None
            or metadatas_by_query[0] is None
            or distances_by_query[0] is None
        ):
            raise RuntimeError("Chroma returned an incomplete query result.")

        ids = ids_by_query[0]
        documents = documents_by_query[0]
        metadatas = metadatas_by_query[0]
        distances = distances_by_query[0]
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            raise RuntimeError("Chroma returned result fields with mismatched lengths.")

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            if metadata is None or distance is None:
                raise RuntimeError("Chroma returned a result without metadata or distance.")
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    distance=float(distance),
                )
            )
        return retrieved

    def _ensure_metadata_compatible(self) -> None:
        if not self.metadata_compatible:
            raise ValueError(
                "The existing Chroma collection uses different embedding metadata. "
                "Call rebuild_index with the complete current chunks and embeddings "
                "before querying or upserting."
            )

    def _validate_batch(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        if not chunks:
            raise ValueError("At least one document chunk is required.")
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a NumPy array.")
        if embeddings.shape != (len(chunks), self.embedding_dimension):
            raise ValueError(
                f"Expected embedding shape {(len(chunks), self.embedding_dimension)}, "
                f"received {embeddings.shape}."
            )
        if not np.issubdtype(embeddings.dtype, np.number) or not np.isfinite(
            embeddings
        ).all():
            raise ValueError("embeddings must contain only finite numbers.")

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if any(not chunk_id.strip() for chunk_id in chunk_ids):
            raise ValueError("Chunk IDs must not be empty.")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunk IDs in a batch must be unique.")
        if any(not chunk.text.strip() for chunk in chunks):
            raise ValueError("Chunk text must not be empty.")

    def _write_batch(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "section_heading": chunk.heading,
                    "heading_path": " > ".join(chunk.heading_path),
                    "section_index": chunk.section_index,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

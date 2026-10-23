from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import DocumentChunk


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class EmbeddingService:
    def __init__(self) -> None:
        try:
            self._model = SentenceTransformer(
                DEFAULT_EMBEDDING_MODEL,
                revision=DEFAULT_EMBEDDING_REVISION,
                device="cpu",
                local_files_only=True,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Embedding model '{DEFAULT_EMBEDDING_MODEL}' at revision "
                f"'{DEFAULT_EMBEDDING_REVISION}' is not available in the local cache. "
                "Download and verify it in a separate setup step before use."
            ) from error

        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                f"Embedding model '{DEFAULT_EMBEDDING_MODEL}' has no output dimension."
            )

        self.model_name = DEFAULT_EMBEDDING_MODEL
        self.model_revision = DEFAULT_EMBEDDING_REVISION
        self.dimension = dimension
        self.max_sequence_length = self._model.max_seq_length

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> np.ndarray:
        if not chunks:
            raise ValueError("At least one document chunk is required.")
        return self.embed_texts([chunk.text for chunk in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text is required.")

        inputs = list(texts)
        for index, text in enumerate(inputs):
            if not text.strip():
                raise ValueError(f"Text at position {index} is empty.")

        tokenized = self._model.tokenizer(
            inputs,
            add_special_tokens=True,
            truncation=False,
        )
        over_limit = [
            index
            for index, token_ids in enumerate(tokenized["input_ids"])
            if len(token_ids) > self.max_sequence_length
        ]
        if over_limit:
            positions = ", ".join(str(index) for index in over_limit)
            raise ValueError(
                f"Text at position(s) {positions} exceed the model's "
                f"{self.max_sequence_length}-wordpiece limit; reduce the chunk size."
            )

        embeddings = self._model.encode(
            inputs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("The embedding model did not return a NumPy array.")
        if embeddings.shape != (len(inputs), self.dimension):
            raise RuntimeError(
                f"Expected embedding shape {(len(inputs), self.dimension)}, "
                f"received {embeddings.shape}."
            )
        return embeddings

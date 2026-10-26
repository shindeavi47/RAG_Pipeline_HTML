from __future__ import annotations

from src import answer_generation


def test_persistent_index_is_built_only_when_empty(monkeypatch, capsys) -> None:
    state = {"count": 0, "document_loads": 0, "embedding_batches": 0, "rebuilds": 0}

    class StubEmbeddings:
        model_name = "test-model"
        model_revision = "test-revision"
        dimension = 2

        def embed_chunks(self, chunks):
            state["embedding_batches"] += 1
            return "chunk embeddings"

    class StubVectorStore:
        def __init__(self, **kwargs):
            assert kwargs["persist_directory"] == answer_generation.DEFAULT_PERSIST_DIRECTORY
            self._client = self

        def count(self):
            return state["count"]

        def rebuild_index(self, chunks, embeddings):
            state["rebuilds"] += 1
            state["count"] = 1

        def close(self):
            pass

    class StubAnswerGenerator:
        def __init__(self, retrieval):
            pass

        def answer(self, question):
            return answer_generation.GeneratedAnswer(
                status="answer",
                text=f"Answer to {question}",
                citations=(),
            )

    def load_document(path):
        state["document_loads"] += 1
        return ["document sections"]

    monkeypatch.setattr(answer_generation, "EmbeddingService", StubEmbeddings)
    monkeypatch.setattr(answer_generation, "VectorStore", StubVectorStore)
    monkeypatch.setattr(answer_generation, "load_document", load_document)
    monkeypatch.setattr(
        answer_generation, "chunk_sections", lambda sections: ["document chunks"]
    )
    monkeypatch.setattr(answer_generation, "RetrievalService", lambda *args: object())
    monkeypatch.setattr(answer_generation, "AnswerGenerator", StubAnswerGenerator)

    answer_generation._answer_with_persistent_index("First question")
    answer_generation._answer_with_persistent_index("Second question")

    assert state == {
        "count": 1,
        "document_loads": 1,
        "embedding_batches": 1,
        "rebuilds": 1,
    }
    assert "Answer to Second question" in capsys.readouterr().out

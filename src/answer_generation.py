from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.chunking import chunk_sections
    from src.embeddings import EmbeddingService
    from src.ingestion import load_document
    from src.retrieval import RetrievalService
    from src.vector_store import VectorStore
else:
    from .chunking import chunk_sections
    from .embeddings import EmbeddingService
    from .ingestion import load_document
    from .retrieval import RetrievalService
    from .vector_store import VectorStore


OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
AnswerStatus = Literal["answer", "clarification", "insufficient_evidence"]

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answer", "clarification", "insufficient_evidence"],
        },
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "answer", "citations"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """Answer questions about the fictional Regression Manager CLI using only supplied excerpts. Return exactly one JSON object with keys status, answer, citations. The status must be answer, clarification, or insufficient_evidence. citations is an array of exact supplied section heading paths.
Choose answer whenever an excerpt directly supports the requested fact. Do not ask for clarification merely because the user has not supplied a value that the question did not request. Choose clarification only when the request itself has materially different documented meanings and the document explicitly says to ask. Choose insufficient_evidence only when no excerpt supports the answer.
Example supported: Question: What does --threads require? Evidence: --threads requires --parallel and accepts integers 1 through 32. Output status answer; state those facts and cite the supplied heading.
Example ambiguous: Question: Rerun failed tests. Evidence: The document says to ask whether to retry failures from a new run or rerun a previous run. Output status clarification; ask that choice and cite the supplied heading.
Never invent flags, defaults, constraints, or facts. Do not claim a command was executed."""


@dataclass(frozen=True)
class GeneratedAnswer:
    status: AnswerStatus
    text: str
    citations: tuple[str, ...]


class AnswerGenerator:
    def __init__(
        self,
        retrieval: RetrievalService,
        *,
        timeout_seconds: float = 300.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        self._retrieval = retrieval
        self._timeout_seconds = timeout_seconds

    def answer(self, question: str, *, top_k: int = 3) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("question must not be empty.")

        matches = self._retrieval.retrieve(question, top_k=top_k)
        if not matches:
            return GeneratedAnswer(
                status="insufficient_evidence",
                text="I could not find documentation evidence to answer this question.",
                citations=(),
            )

        evidence: list[str] = []
        available_citations: list[str] = []
        for match in matches:
            heading_path = match.metadata.get("heading_path")
            if not isinstance(heading_path, str) or not heading_path.strip():
                raise TypeError("Retrieved chunk heading_path metadata must be text.")
            available_citations.append(heading_path)
            evidence.append(f"[{heading_path}]\n{match.text}")

        available_citations = list(dict.fromkeys(available_citations))
        evidence_text = "\n\n".join(evidence)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Documentation excerpts:\n{evidence_text}\n\n"
            f"Return citations only from this list: {json.dumps(available_citations)}"
        )
        request_body = json.dumps(
            {
                "model": OLLAMA_MODEL,
                "stream": False,
                "format": _ANSWER_SCHEMA,
                "options": {
                    "temperature": 0,
                    "num_predict": 192,
                    "num_ctx": 4096,
                },
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        request = Request(
            OLLAMA_CHAT_URL,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Ollama request failed with HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise ConnectionError(
                f"Could not reach the local Ollama API at {OLLAMA_CHAT_URL}: "
                f"{error.reason}"
            ) from error

        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned a response that was not a JSON object.")
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama response did not contain message content.")
        try:
            generated = json.loads(message["content"])
        except json.JSONDecodeError as error:
            raise RuntimeError("Ollama did not return valid JSON answer content.") from error
        if not isinstance(generated, dict):
            raise RuntimeError("Ollama answer content must be a JSON object.")

        status = generated.get("status")
        text = generated.get("answer")
        citations = generated.get("citations")
        if status not in ("answer", "clarification", "insufficient_evidence"):
            raise RuntimeError("Ollama returned an unsupported answer status.")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama returned an empty answer.")
        if not isinstance(citations, list) or any(
            not isinstance(citation, str) for citation in citations
        ):
            raise RuntimeError("Ollama returned invalid answer citations.")
        if any(citation not in available_citations for citation in citations):
            raise RuntimeError("Ollama cited a section that was not retrieved.")

        return GeneratedAnswer(
            status=status,
            text=text.strip(),
            citations=tuple(dict.fromkeys(citations)),
        )


def format_generated_answer(answer: GeneratedAnswer) -> str:
    lines = [answer.text]
    if answer.citations:
        lines.append("Sources: " + "; ".join(answer.citations))
    return "\n".join(lines)


def _answer_with_temporary_index(question: str, persist_directory: str) -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "documents"
        / "regression_manager_cli_documentation.html"
    )
    sections = load_document(source_path)
    chunks = chunk_sections(sections)
    embeddings = EmbeddingService()
    chunk_embeddings = embeddings.embed_chunks(chunks)
    vector_store = VectorStore(
        embedding_model=embeddings.model_name,
        embedding_revision=embeddings.model_revision,
        embedding_dimension=embeddings.dimension,
        persist_directory=persist_directory,
    )
    try:
        vector_store.rebuild_index(chunks, chunk_embeddings)

        # records = vector_store._collection.get(
        #     include=["documents", "metadatas", "embeddings"]
        # )
        # for i, chunk_id in enumerate(records["ids"]):
        #     print(f"ID: {chunk_id}")
        #     print(f"Metadata: {records['metadatas'][i]}")
        #     print(f"Text: {records['documents'][i]}")
        #     embedding = records["embeddings"][i]
        #     print(f"Embedding: {len(embedding)} dimensions; first 8 values: {embedding[:8]}")
        #     print()

        retrieval = RetrievalService(embeddings, vector_store)
        retrieval = RetrievalService(embeddings, vector_store)
        generator = AnswerGenerator(retrieval)
        answer = generator.answer(question)
        print(f"Status: {answer.status}")
        print(format_generated_answer(answer))
    finally:
        vector_store._client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer a question using the local Regression Manager docs and Ollama."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to answer; if omitted, the script prompts for one.",
    )
    parser.add_argument("--worker-db", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_db:
        if not args.question:
            parser.error("the internal worker requires a question.")
        _answer_with_temporary_index(args.question, args.worker_db)
        return

    question = args.question or input("Ask a question about Regression Manager: ").strip()
    if not question:
        parser.error("question must not be empty.")

    project_root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="test-generate-chroma-") as temporary_directory:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.answer_generation",
                "--worker-db",
                temporary_directory,
                question,
            ],
            cwd=project_root,
            check=False,
        )
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

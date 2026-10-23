from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from src.retrieval import RetrievalService
from src.retrieval_evaluation import (
    RetrievalEvaluationCase,
    evaluate_retrieval,
    evaluate_retrieval_file,
    format_evaluation_report,
    load_evaluation_cases,
)
from src.vector_store import RetrievedChunk


class StubRetrievalService:
    def __init__(self, matches_by_question: dict[str, list[RetrievedChunk]]) -> None:
        self.matches_by_question = matches_by_question
        self.requests: list[tuple[str, int]] = []

    def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        self.requests.append((question, top_k))
        return self.matches_by_question.get(question, [])[:top_k]


def make_match(heading_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=heading_path,
        text="Evidence text",
        metadata={"heading_path": heading_path},
        distance=0.1,
    )


def test_loads_approved_evaluation_dataset() -> None:
    path = Path(__file__).with_name("retrieval_eval_cases.json")

    top_k, cases = load_evaluation_cases(path)

    assert top_k == 3
    assert len(cases) == 9
    assert len({case.case_id for case in cases}) == 9


def test_evaluate_retrieval_calculates_section_and_case_metrics() -> None:
    cases = (
        RetrievalEvaluationCase("first", "Question one", ("Section A", "Section B")),
        RetrievalEvaluationCase("second", "Question two", ("Section C",)),
    )
    stub = StubRetrievalService(
        {
            "Question one": [make_match("Section A"), make_match("Other section")],
            "Question two": [make_match("Other section")],
        }
    )

    report = evaluate_retrieval(cast(RetrievalService, stub), cases, top_k=2)

    assert stub.requests == [("Question one", 2), ("Question two", 2)]
    assert report.matched_section_count == 1
    assert report.expected_section_count == 3
    assert report.section_recall == pytest.approx(1 / 3)
    assert report.complete_case_count == 0
    assert report.case_success_rate == 0
    assert report.case_results[0].retrieved_heading_paths == (
        "Section A",
        "Other section",
    )


def test_evaluate_retrieval_file_uses_dataset_top_k() -> None:
    path = Path(__file__).with_name("retrieval_eval_cases.json")
    _, cases = load_evaluation_cases(path)
    stub = StubRetrievalService({})

    report = evaluate_retrieval_file(cast(RetrievalService, stub), path)

    assert report.top_k == 3
    assert stub.requests == [(case.question, 3) for case in cases]
    assert report.matched_section_count == 0


def test_rejects_invalid_top_k_in_evaluation_file(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"top_k": 0, "cases": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        load_evaluation_cases(path)


def test_rejects_retrieved_chunk_without_heading_path() -> None:
    case = RetrievalEvaluationCase("missing-path", "Question", ("Section A",))
    match = RetrievedChunk(
        chunk_id="chunk-1",
        text="Evidence text",
        metadata={},
        distance=0.1,
    )
    stub = StubRetrievalService({"Question": [match]})

    with pytest.raises(TypeError, match="heading_path"):
        evaluate_retrieval(cast(RetrievalService, stub), [case], top_k=1)


def test_formats_evaluation_metrics() -> None:
    cases = (RetrievalEvaluationCase("case-1", "Question", ("Section A",)),)
    stub = StubRetrievalService({"Question": [make_match("Section A")]})
    report = evaluate_retrieval(cast(RetrievalService, stub), cases, top_k=1)

    formatted = format_evaluation_report(report)

    assert "Complete cases: 1/1 (100.0%)" in formatted
    assert "Section recall: 1/1 (100.0%)" in formatted
    assert "PASS case-1" in formatted

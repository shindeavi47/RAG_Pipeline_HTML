from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .retrieval import RetrievalService


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    case_id: str
    question: str
    expected_heading_paths: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    expected_heading_paths: tuple[str, ...]
    retrieved_heading_paths: tuple[str, ...]
    matched_heading_paths: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return len(self.matched_heading_paths) == len(self.expected_heading_paths)


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    top_k: int
    case_results: tuple[RetrievalCaseResult, ...]

    @property
    def matched_section_count(self) -> int:
        return sum(len(result.matched_heading_paths) for result in self.case_results)

    @property
    def expected_section_count(self) -> int:
        return sum(len(result.expected_heading_paths) for result in self.case_results)

    @property
    def section_recall(self) -> float:
        return self.matched_section_count / self.expected_section_count

    @property
    def complete_case_count(self) -> int:
        return sum(result.complete for result in self.case_results)

    @property
    def case_success_rate(self) -> float:
        return self.complete_case_count / len(self.case_results)


def load_evaluation_cases(
    path: str | Path,
) -> tuple[int, tuple[RetrievalEvaluationCase, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation file must contain a JSON object.")

    top_k = payload.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("Evaluation top_k must be a positive integer.")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Evaluation file must contain a non-empty cases list.")

    cases: list[RetrievalEvaluationCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Evaluation case at position {index} must be an object.")

        case_id = raw_case.get("id")
        question = raw_case.get("question")
        expected = raw_case.get("expected_heading_paths")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Evaluation case at position {index} has no valid id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Evaluation case '{case_id}' has no valid question.")
        if not isinstance(expected, list) or not expected:
            raise ValueError(
                f"Evaluation case '{case_id}' must expect at least one heading path."
            )
        if any(not isinstance(item, str) or not item.strip() for item in expected):
            raise ValueError(
                f"Evaluation case '{case_id}' contains an invalid heading path."
            )
        if len(set(expected)) != len(expected):
            raise ValueError(
                f"Evaluation case '{case_id}' contains duplicate heading paths."
            )

        seen_ids.add(case_id)
        cases.append(
            RetrievalEvaluationCase(
                case_id=case_id,
                question=question,
                expected_heading_paths=tuple(expected),
            )
        )

    return top_k, tuple(cases)


def evaluate_retrieval(
    service: RetrievalService,
    cases: Sequence[RetrievalEvaluationCase],
    *,
    top_k: int,
) -> RetrievalEvaluationReport:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not cases:
        raise ValueError("At least one evaluation case is required.")

    case_results: list[RetrievalCaseResult] = []
    for case in cases:
        matches = service.retrieve(case.question, top_k=top_k)
        retrieved_paths: list[str] = []
        for match in matches:
            heading_path = match.metadata.get("heading_path")
            if not isinstance(heading_path, str):
                raise TypeError(
                    f"Retrieved chunk for case '{case.case_id}' has no text heading_path."
                )
            retrieved_paths.append(heading_path)

        retrieved_path_set = set(retrieved_paths)
        matched_paths = tuple(
            path for path in case.expected_heading_paths if path in retrieved_path_set
        )
        case_results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                expected_heading_paths=case.expected_heading_paths,
                retrieved_heading_paths=tuple(retrieved_paths),
                matched_heading_paths=matched_paths,
            )
        )

    return RetrievalEvaluationReport(top_k, tuple(case_results))


def evaluate_retrieval_file(
    service: RetrievalService,
    path: str | Path,
) -> RetrievalEvaluationReport:
    top_k, cases = load_evaluation_cases(path)
    return evaluate_retrieval(service, cases, top_k=top_k)


def format_evaluation_report(report: RetrievalEvaluationReport) -> str:
    lines = [
        f"Top-k: {report.top_k}",
        f"Complete cases: {report.complete_case_count}/{len(report.case_results)} "
        f"({report.case_success_rate:.1%})",
        f"Section recall: {report.matched_section_count}/"
        f"{report.expected_section_count} ({report.section_recall:.1%})",
    ]
    for result in report.case_results:
        status = "PASS" if result.complete else "MISS"
        lines.append(f"{status} {result.case_id}")
    return "\n".join(lines)

from __future__ import annotations

import pytest

from src.chunking import chunk_sections
from src.ingestion import DocumentSection


def make_section(text: str) -> DocumentSection:
    return DocumentSection(
        heading="Parallel execution",
        heading_path=("Regression Manager", "Feature recipes", "Parallel execution"),
        text=text,
    )


def test_chunks_respect_size_and_keep_source_metadata() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 8

    chunks = chunk_sections([make_section(text)], chunk_size=80, chunk_overlap=12)

    assert len(chunks) > 1
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    assert all(chunk.heading == "Parallel execution" for chunk in chunks)
    assert all(
        chunk.heading_path
        == ("Regression Manager", "Feature recipes", "Parallel execution")
        for chunk in chunks
    )


def test_cli_rule_row_stays_together_when_it_fits() -> None:
    rule = "--threads | Integer 1-32; default 4 | Requires --parallel"
    section = make_section(f"{rule}\n--report | html, junit, all")

    chunks = chunk_sections([section], chunk_size=80, chunk_overlap=10)

    assert any(rule in chunk.text for chunk in chunks)


def test_chunk_ids_are_stable_and_unique() -> None:
    sections = [
        make_section("one two three four " * 12),
        make_section("another section"),
    ]

    first = chunk_sections(sections, chunk_size=40, chunk_overlap=5)
    second = chunk_sections(sections, chunk_size=40, chunk_overlap=5)

    first_ids = [chunk.chunk_id for chunk in first]
    assert first_ids == [chunk.chunk_id for chunk in second]
    assert len(first_ids) == len(set(first_ids))


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_rejects_invalid_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_sections(
            [make_section("some content")],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_rejects_empty_section_list() -> None:
    with pytest.raises(ValueError, match="At least one"):
        chunk_sections([])


def test_rejects_empty_section_text() -> None:
    with pytest.raises(ValueError, match="has no text"):
        chunk_sections([make_section("  ")])

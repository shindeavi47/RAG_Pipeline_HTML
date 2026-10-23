from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion import load_document


SOURCE_DOCUMENT = (
    Path(__file__).resolve().parents[1]
    / "documents"
    / "regression_manager_cli_documentation.html"
)


def test_loads_document_sections_tables_and_examples() -> None:
    sections = load_document(SOURCE_DOCUMENT)

    assert sections
    assert all(section.text.strip() for section in sections)
    assert any("Required arguments" in section.heading for section in sections)
    assert any("Exit codes" in section.heading for section in sections)
    assert any("--suite" in section.text for section in sections)
    assert any("java -jar" in section.text for section in sections)
    assert all(section.heading_path for section in sections)


def test_missing_document_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_document(tmp_path / "missing.html")


def test_empty_document_raises_value_error(tmp_path: Path) -> None:
    empty_document = tmp_path / "empty.html"
    empty_document.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_document(empty_document)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag


@dataclass(frozen=True)
class DocumentSection:
    heading: str
    heading_path: tuple[str, ...]
    text: str


_CONTENT_TAGS = ("p", "ul", "ol", "table", "pre", "blockquote")
_NESTED_BLOCK_TAGS = ("nav", "table", "ul", "ol", "pre", "blockquote")
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3}


def _format_block(element: Tag) -> str:
    if element.name in {"ul", "ol"}:
        items = [
            item.get_text(" ", strip=True)
            for item in element.find_all("li", recursive=False)
        ]
        return "\n".join(f"- {item}" for item in items if item)

    if element.name == "table":
        rows = []
        for row in element.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["th", "td"], recursive=False)
            ]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    if element.name == "pre":
        return element.get_text().strip("\r\n")

    return element.get_text(" ", strip=True)


def load_document(path: str | Path) -> list[DocumentSection]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Documentation file was not found: {source}")

    html = source.read_text(encoding="utf-8-sig")
    if not html.strip():
        raise ValueError(f"Documentation file is empty: {source}")

    soup = BeautifulSoup(html, "html.parser")
    root = soup.body or soup
    heading_stack: list[tuple[int, str]] = []
    current_blocks: list[str] = []
    sections: list[DocumentSection] = []

    def save_section() -> None:
        text = "\n\n".join(block for block in current_blocks if block)
        if text:
            path_parts = tuple(title for _, title in heading_stack)
            sections.append(
                DocumentSection(
                    heading=path_parts[-1] if path_parts else "",
                    heading_path=path_parts,
                    text=text,
                )
            )
        current_blocks.clear()

    for element in root.find_all(
        ["h1", "h2", "h3", *_CONTENT_TAGS]
    ):
        if element.find_parent("nav") is not None:
            continue

        if element.name in _HEADING_LEVELS:
            save_section()
            level = _HEADING_LEVELS[element.name]
            title = element.get_text(" ", strip=True)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if title:
                heading_stack.append((level, title))
            continue

        if element.find_parent(list(_NESTED_BLOCK_TAGS)) is not None:
            continue

        block = _format_block(element)
        if block:
            current_blocks.append(block)

    save_section()
    if not sections:
        raise ValueError(f"No extractable document content was found: {source}")
    return sections

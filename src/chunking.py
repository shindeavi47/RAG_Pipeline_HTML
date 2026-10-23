from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingestion import DocumentSection


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    section_index: int
    chunk_index: int
    heading: str
    heading_path: tuple[str, ...]
    text: str


def chunk_sections(
    sections: Sequence[DocumentSection],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
    if not sections:
        raise ValueError("At least one document section is required.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[DocumentChunk] = []

    for section_index, section in enumerate(sections):
        if not section.text.strip():
            raise ValueError(f"Document section {section_index} has no text.")

        split_documents = splitter.create_documents([section.text])
        for chunk_index, document in enumerate(split_documents):
            payload = json.dumps(
                [section_index, chunk_index, section.heading_path, document.page_content],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            chunk_id = hashlib.sha256(payload).hexdigest()
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    section_index=section_index,
                    chunk_index=chunk_index,
                    heading=section.heading,
                    heading_path=section.heading_path,
                    text=document.page_content,
                )
            )

    return chunks

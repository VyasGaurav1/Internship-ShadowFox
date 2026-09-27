"""Chunking strategy for long documents.

Choice of chunking method and sizes is explained in the README. The default is
a **word-window** chunker with overlap: it keeps a fixed token budget per chunk
(so embedding cost and retrieval granularity are predictable) and an overlap
band (so a fact split across a boundary is still retrievable from at least one
chunk).

Chunks carry their character offsets back into the source text, which lets the
UI highlight / cite the exact source region.
"""
from __future__ import annotations

from dataclasses import dataclass

from ingest import Document


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    index: int
    text: str
    start_word: int
    end_word: int
    start_char: int
    end_char: int

    @property
    def citation(self) -> str:
        return f"{self.doc_name}#chunk-{self.index}"


def chunk_words(
    text: str,
    chunk_size: int = 140,
    overlap: int = 30,
) -> list[tuple[int, int, int, int]]:
    """Return a list of ``(start_word, end_word, start_char, end_char)`` spans."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    # map word index -> char offset so we can slice the original text exactly
    words: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not text[i].isspace():
            i += 1
        words.append((start, i))

    if not words:
        return []

    spans = []
    step = chunk_size - overlap
    for start_idx in range(0, len(words), step):
        end_idx = min(start_idx + chunk_size, len(words))
        start_char = words[start_idx][0]
        end_char = words[end_idx - 1][1]
        spans.append((start_idx, end_idx, start_char, end_char))
        if end_idx == len(words):
            break
    return spans


def chunk_document(document: Document, chunk_size: int = 140, overlap: int = 30) -> list[Chunk]:
    spans = chunk_words(document.text, chunk_size=chunk_size, overlap=overlap)
    chunks: list[Chunk] = []
    for index, (sw, ew, sc, ec) in enumerate(spans):
        piece = document.text[sc:ec].strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                doc_id=document.doc_id,
                doc_name=document.name,
                index=index,
                text=piece,
                start_word=sw,
                end_word=ew,
                start_char=sc,
                end_char=ec,
            )
        )
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int = 140, overlap: int = 30) -> list[Chunk]:
    out: list[Chunk] = []
    for document in documents:
        out.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return out

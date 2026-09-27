"""Sentence-aware chunking with word-based budget and overlap.

Why sentence-aware: ending a chunk at a sentence boundary keeps each embedded
unit semantically self-contained, which improves retrieval precision. The word
budget keeps embedding cost predictable, and the overlap band preserves context
that would otherwise be cut at a boundary.

Each chunk records exact character offsets into the *preprocessed* document so
citations can point back to a real region.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|Mr|Dr|Fig|No|approx)\.", re.IGNORECASE)


@dataclass
class Chunk:
    doc_id: str
    filename: str
    index: int
    text: str
    start_char: int
    end_char: int
    start_word: int
    end_word: int

    @property
    def citation(self) -> str:
        return f"{self.filename}#chunk-{self.index}"


def _sentences_with_offsets(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets for sentence-like spans."""

    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\u0001"), text)
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    n = len(protected)
    while i < n:
        char = protected[i]
        if char in ".!?":
            # consume following closing quotes
            j = i + 1
            while j < n and protected[j] in "\"')]":
                j += 1
            if j >= n or protected[j].isspace() or protected[j] == "\n":
                end = j
                piece = text[start:end].strip()
                if piece:
                    leading = len(text[start:end]) - len(text[start:end].lstrip())
                    spans.append((start + leading, end))
                i = j
                while i < n and text[i].isspace():
                    i += 1
                start = i
                continue
        i += 1
    tail = text[start:].strip()
    if tail:
        leading = len(text[start:]) - len(text[start:].lstrip())
        spans.append((start + leading, len(text)))
    return spans


def chunk_text(text: str, chunk_size: int = 160, chunk_overlap: int = 40) -> list[tuple[int, int]]:
    """Return (start_char, end_char) spans for chunks of roughly ``chunk_size`` words."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(chunk_overlap, chunk_size - 1))

    sentences = _sentences_with_offsets(text)
    if not sentences:
        return []

    def word_count(span: tuple[int, int]) -> int:
        return len(text[span[0]:span[1]].split())

    chunks: list[tuple[int, int]] = []
    i = 0
    total = len(sentences)
    while i < total:
        start = sentences[i][0]
        words = 0
        j = i
        while j < total:
            words += word_count(sentences[j])
            j += 1
            if words >= chunk_size:
                break
        end = sentences[j - 1][1]
        chunks.append((start, end))
        if j >= total:
            break
        # walk back to create the overlap band
        back = j
        overlap_words = 0
        while back > i + 1 and overlap_words < overlap:
            back -= 1
            overlap_words += word_count(sentences[back])
        i = back if back > i else i + 1
    return chunks


def chunk_document(text: str, doc_id: str, filename: str, chunk_size: int = 160, chunk_overlap: int = 40) -> list[Chunk]:
    out: list[Chunk] = []
    for index, (start, end) in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
        piece = text[start:end].strip()
        if not piece:
            continue
        out.append(
            Chunk(
                doc_id=doc_id,
                filename=filename,
                index=index,
                text=piece,
                start_char=start,
                end_char=end,
                start_word=len(text[:start].split()),
                end_word=len(text[:end].split()),
            )
        )
    return out

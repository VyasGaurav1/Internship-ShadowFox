"""In-memory vector store with cosine similarity (+ optional FAISS backend).

The default backend is dependency-free pure Python, which is fine for the
document sizes this assistant targets (hundreds of chunks). If ``faiss`` is
installed the store can delegate to it - see :class:`FaissVectorStore`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from chunking import Chunk


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class VectorStore:
    """Simple list-backed store: exact search, no index build step."""

    name = "cosine-list"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def vectors(self) -> list[list[float]]:
        return self._vectors

    def search(self, query_vector: list[float], top_k: int = 4) -> list[Retrieved]:
        if not self._chunks:
            return []
        scored = [
            (cosine(query_vector, vec), chunk)
            for chunk, vec in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda pair: -pair[0])
        return [Retrieved(chunk=chunk, score=round(score, 4)) for score, chunk in scored[:top_k]]


class FaissVectorStore(VectorStore):  # pragma: no cover - requires faiss
    """FAISS-backed exact search (IndexFlatIP over L2-normalised vectors)."""

    name = "faiss-flatip"

    def __init__(self, dim: int) -> None:
        import faiss
        import numpy as np

        self._np = np
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        matrix = self._np.asarray(vectors, dtype="float32")
        self._index.add(matrix)
        self._chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query_vector: list[float], top_k: int = 4) -> list[Retrieved]:
        if not self._chunks:
            return []
        query = self._np.asarray([query_vector], dtype="float32")
        scores, indices = self._index.search(query, min(top_k, len(self._chunks)))
        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            out.append(Retrieved(chunk=self._chunks[int(idx)], score=round(float(score), 4)))
        return out


def build_store(dim: int):
    """Use FAISS when available, otherwise the dependency-free store."""

    try:
        return FaissVectorStore(dim)
    except Exception:
        return VectorStore()

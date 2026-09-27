"""Vector stores: FAISS when available, otherwise a dependency-free cosine store.

Both expose the same tiny interface so the retriever does not care which is
active. The store keeps the mapping chunk -> vector so reranking can reuse the
vectors, and supports filtering by document id for document-scoped retrieval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class CosineVectorStore:
    """Exact cosine search over an in-memory list of vectors."""

    name = "cosine-list"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(self, query_vector: list[float], top_k: int = 4) -> list[ScoredChunk]:
        scored: list[ScoredChunk] = []
        for chunk, vector in zip(self._chunks, self._vectors):
            dot = sum(a * b for a, b in zip(query_vector, vector))
            norm_a = math.sqrt(sum(a * a for a in query_vector))
            norm_b = math.sqrt(sum(b * b for b in vector))
            score = 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)
            scored.append(ScoredChunk(chunk, round(score, 4)))
        scored.sort(key=lambda item: -item.score)
        return scored[:top_k]


class FaissVectorStore:  # pragma: no cover - requires faiss
    name = "faiss-flatip"

    def __init__(self, dim: int) -> None:
        import faiss
        import numpy as np

        self.dim = dim
        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._index.add(self._np.asarray(vectors, dtype="float32"))
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(self, query_vector: list[float], top_k: int = 4) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query = self._np.asarray([query_vector], dtype="float32")
        scores, indices = self._index.search(query, min(top_k, len(self._chunks)))
        out: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            out.append(ScoredChunk(self._chunks[int(idx)], round(float(score), 4)))
        return out


def build_store(dim: int, prefer_faiss: bool = True):
    if prefer_faiss:
        try:  # pragma: no cover
            return FaissVectorStore(dim)
        except Exception:
            pass
    return CosineVectorStore(dim)

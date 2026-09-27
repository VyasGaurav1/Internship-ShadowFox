"""Hybrid reranking: vector relevance + lexical overlap, then MMR diversity.

Reranking improves the top-k quality before generation:

1. **Hybrid score** combines the embedding similarity with a lexical-overlap
   score (question content words found in the chunk). Vector search alone can
   surface loosely-related chunks; the lexical term keeps exact terminology
   matches at the top.
2. **MMR** (maximal marginal relevance) then removes redundancy, so the
   generator receives diverse passages instead of several paraphrases of the
   same sentence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .text_utils import content_words, jaccard
from .vector_store import ScoredChunk


@dataclass
class RerankedChunk:
    hit: ScoredChunk
    rerank_score: float
    lexical_score: float


def _normalise(values: list[float]) -> list[float]:
    top = max(values) if values else 0.0
    if top <= 0:
        return [0.0 for _ in values]
    return [v / top for v in values]


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    top_k: int = 4,
    mmr_lambda: float = 0.5,
) -> list[RerankedChunk]:
    if not candidates:
        return []

    q_terms = set(content_words(query))
    vec_scores = _normalise([c.score for c in candidates])

    lexical_raw: list[float] = []
    term_sets: list[set[str]] = []
    for candidate in candidates:
        terms = set(content_words(candidate.chunk.text))
        term_sets.append(terms)
        overlap = len(q_terms & terms)
        lexical_raw.append(overlap / math.sqrt(len(terms)) if terms else 0.0)
    lexical = _normalise(lexical_raw)

    # hybrid relevance
    relevance = [mmr_lambda * vec_scores[i] + (1 - mmr_lambda) * lexical[i] for i in range(len(candidates))]

    selected_indices: list[int] = []
    results: list[RerankedChunk] = []
    remaining = list(range(len(candidates)))
    while remaining and len(results) < top_k:
        best_index = remaining[0]
        best_value = -1e9
        for index in remaining:
            redundancy = max((jaccard(term_sets[index], term_sets[j]) for j in selected_indices), default=0.0)
            value = relevance[index] - (1 - mmr_lambda) * redundancy
            if value > best_value:
                best_value = value
                best_index = index
        selected_indices.append(best_index)
        results.append(
            RerankedChunk(
                hit=candidates[best_index],
                rerank_score=round(relevance[best_index], 4),
                lexical_score=round(lexical[best_index], 4),
            )
        )
        remaining.remove(best_index)

    return results

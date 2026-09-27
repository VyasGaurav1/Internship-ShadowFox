"""Shared mutable state passed between pipeline nodes."""
from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    # input
    question: str
    rewritten_query: str | None
    document_ids: list[str] | None
    top_k: int
    rewrite: bool
    rerank: bool

    # retrieval
    candidates: list[Any]
    contexts: list[Any]
    rerank_scores: dict[str, float]

    # grading / grounding
    grade_ok: bool
    lexical_overlap: int
    best_score: float
    grounded_ok: bool
    groundedness: float
    refused: bool

    # generation
    answer: str
    provider: str
    warnings: list[str]

    # observability
    steps: list[str]
    elapsed_ms: int
    orchestration: str

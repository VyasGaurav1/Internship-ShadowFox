"""Guardrails: context grading and groundedness verification.

These are the checks that keep the assistant honest:

* **Context grade** (pre-generation): refuse to answer when the retrieved
  context is too weak. Two independent conditions must hold - the best chunk
  must clear the similarity threshold *and* share terminology with the question.
* **Groundedness** (post-generation): the share of the answer's content words
  that also occur in the cited context. If it falls below the configured
  minimum, the answer is rejected and replaced with an explicit refusal.
"""
from __future__ import annotations

from .text_utils import content_words
from .vector_store import ScoredChunk

REFUSAL_MESSAGE = (
    "The uploaded documents do not contain enough information to answer this. "
    "Try rephrasing the question or uploading more relevant material."
)

NOT_GROUNDED_MESSAGE = (
    "I could not verify an answer against the uploaded documents, so I am not going to guess. "
    "Try asking a more specific question about the document content."
)


def grade_context(
    question: str,
    candidates: list[ScoredChunk],
    min_score: float = 0.05,
    min_lexical_overlap: int = 1,
) -> tuple[bool, int, float]:
    """Return ``(ok, lexical_overlap, best_score)``."""

    if not candidates:
        return False, 0, 0.0
    best = candidates[0]
    q_terms = set(content_words(question))
    c_terms = set(content_words(best.chunk.text))
    overlap = len(q_terms & c_terms)
    ok = best.score >= min_score and overlap >= min_lexical_overlap
    return ok, overlap, round(best.score, 4)


def groundedness(answer: str, context_texts: list[str]) -> float:
    answer_terms = set(content_words(answer))
    if not answer_terms:
        return 0.0
    context_terms: set[str] = set()
    for text in context_texts:
        context_terms.update(content_words(text))
    if not context_terms:
        return 0.0
    return round(len(answer_terms & context_terms) / len(answer_terms), 3)

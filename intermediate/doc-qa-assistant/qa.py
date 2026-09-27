"""Grounded question answering over an ingested document index.

The pipeline is strictly *retrieval first*:

    question -> embed -> vector search (top-k chunks) -> grounded generation

The generation step never sees the model's own knowledge as license to invent:
the system prompt restricts the answer to the retrieved context, requires chunk
citations, and if nothing relevant is retrieved the assistant refuses instead of
guessing (``refused=True``).

Two generation modes:
* ``llm``  - a real chat model conditioned on the retrieved chunks.
* ``extractive`` (offline) - deterministic sentence selection from the retrieved
  chunks. Because every answer sentence is copied from retrieved text, the
  answer is grounded by construction, which makes the anti-hallucination
  behaviour verifiable without any API key.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from chunking import Chunk, chunk_documents
from embeddings import build_embedder
from ingest import Document, IngestError
from llm_client import LLMError, call_llm
from text_utils import content_words, split_sentences
from validators import ValidationError, validate_question
from vector_store import Retrieved, build_store


@dataclass
class Source:
    citation: str
    doc_name: str
    chunk_index: int
    score: float
    text: str
    start_char: int
    end_char: int

    def as_dict(self) -> dict:
        return {
            "citation": self.citation,
            "doc_name": self.doc_name,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    groundedness: float = 0.0
    refused: bool = False
    provider: str = "extractive"
    warning: str | None = None
    elapsed_ms: int = 0


class DocumentIndex:
    """Holds chunks + vectors for a set of documents."""

    def __init__(self, chunks: list[Chunk], vectors: list[list[float]], store, embedder, settings) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.store = store
        self.embedder = embedder
        self.settings = settings

    @property
    def documents(self) -> list[str]:
        return sorted({c.doc_name for c in self.chunks})

    @property
    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "embedding_provider": getattr(self.embedder, "name", "unknown"),
            "vector_store": getattr(self.store, "name", "unknown"),
        }


def build_index(documents: list[Document], settings) -> DocumentIndex:
    """Ingest -> chunk -> embed -> index."""

    if not documents:
        raise IngestError("No documents were provided.", kind="empty")

    chunks = chunk_documents(documents, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise IngestError("The uploaded documents produced no usable text.", kind="empty")

    embedder = build_embedder(settings)
    vectors = embedder.embed([c.text for c in chunks])
    store = build_store(len(vectors[0]))
    store.add(chunks, vectors)
    return DocumentIndex(chunks=chunks, vectors=vectors, store=store, embedder=embedder, settings=settings)


# --------------------------------------------------------------------------- #
# grounded generation
# --------------------------------------------------------------------------- #
_GROUNDING_SYSTEM = (
    "You are a document-grounded question answering assistant. "
    "Answer ONLY using the numbered context passages provided by the user. "
    "If the passages do not contain the answer, reply exactly: "
    "'The uploaded documents do not contain enough information to answer this.' "
    "Cite the passages you used inline in the form [1], [2]. "
    "Be concise (2-5 sentences) and do not add outside knowledge."
)


def _extractive_answer(question: str, retrieved: list[Retrieved]) -> tuple[str, list[Source]]:
    """Deterministic grounded answer: top sentences copied from retrieved chunks."""

    q_terms = set(content_words(question))
    scored_sentences: list[tuple[float, int, str, Retrieved]] = []

    for rank, item in enumerate(retrieved):
        for sentence in split_sentences(item.chunk.text):
            s_terms = set(content_words(sentence))
            overlap = len(q_terms & s_terms)
            if overlap == 0:
                continue
            score = overlap / (1 + len(s_terms) ** 0.5) + item.score
            scored_sentences.append((score, rank, sentence, item))

    if not scored_sentences:
        best = retrieved[0]
        first = split_sentences(best.chunk.text)[:1]
        answer = first[0] if first else best.chunk.text[:280]
        scored_sentences = [(best.score, 0, answer, best)]

    # dedupe by sentence text, keep best score, limit to 3 sentences
    seen: set[str] = set()
    chosen: list[tuple[float, int, str, Retrieved]] = []
    for entry in sorted(scored_sentences, key=lambda e: -e[0]):
        if entry[2] in seen:
            continue
        seen.add(entry[2])
        chosen.append(entry)
        if len(chosen) >= 3:
            break

    answer = " ".join(text for _, _, text, _ in chosen)
    sources = _sources_for(chosen)
    return answer, sources


def _sources_for(chosen: list[tuple[float, int, str, Retrieved]]) -> list[Source]:
    out: list[Source] = []
    seen: set[str] = set()
    for _, _, _, item in chosen:
        citation = item.chunk.citation
        if citation in seen:
            continue
        seen.add(citation)
        out.append(
            Source(
                citation=citation,
                doc_name=item.chunk.doc_name,
                chunk_index=item.chunk.index,
                score=item.score,
                text=item.chunk.text,
                start_char=item.chunk.start_char,
                end_char=item.chunk.end_char,
            )
        )
    return out


def _groundedness(answer: str, sources: list[Source]) -> float:
    """Share of answer content words that also appear in the retrieved context."""

    answer_terms = set(content_words(answer))
    if not answer_terms:
        return 0.0
    context_terms = set()
    for source in sources:
        context_terms.update(content_words(source.text))
    if not context_terms:
        return 0.0
    return round(len(answer_terms & context_terms) / len(answer_terms), 3)


def ask(index: DocumentIndex, question: str) -> AnswerResult:
    """Answer a question strictly from the indexed documents."""

    question = validate_question(question)
    started = __import__("time").perf_counter()

    query_vector = index.embedder.embed([question])[0]
    retrieved = index.store.search(query_vector, top_k=index.settings.top_k)

    # Anti-hallucination guardrail: refuse unless the best chunk both clears the
    # similarity threshold AND shares at least one content word with the question.
    # The lexical check makes the refusal robust against hash-collision noise in
    # the offline embedder (and is configurable via MIN_LEXICAL_OVERLAP).
    lexical_overlap = 0
    if retrieved:
        q_terms = set(content_words(question))
        top_terms = set(content_words(retrieved[0].chunk.text))
        lexical_overlap = len(q_terms & top_terms)

    if (
        not retrieved
        or retrieved[0].score < index.settings.min_score
        or lexical_overlap < index.settings.min_lexical_overlap
    ):
        return AnswerResult(
            question=question,
            answer=(
                "The uploaded documents do not contain enough information to answer this. "
                "Try rephrasing the question or uploading more relevant material."
            ),
            sources=[],
            groundedness=0.0,
            refused=True,
            provider="guardrail",
            elapsed_ms=int((__import__("time").perf_counter() - started) * 1000),
        )

    warning: str | None = None
    if index.settings.offline_llm:
        answer, sources = _extractive_answer(question, retrieved)
        provider = "extractive (offline)"
    else:
        context = "\n\n".join(
            f"[{i + 1}] ({item.chunk.citation})\n{item.chunk.text}" for i, item in enumerate(retrieved)
        )
        user_prompt = f"Question: {question}\n\nContext passages:\n{context}"
        try:
            answer = call_llm(index.settings, _GROUNDING_SYSTEM, user_prompt)
            sources = _sources_for([(0.0, rank, "", item) for rank, item in enumerate(retrieved)])
            provider = f"{index.settings.llm_provider}:{index.settings.llm_model}"
        except LLMError as exc:
            answer, sources = _extractive_answer(question, retrieved)
            provider = "extractive (offline fallback)"
            warning = f"Live model unavailable ({exc.kind}); answered from retrieved text instead."

    return AnswerResult(
        question=question,
        answer=answer,
        sources=sources,
        groundedness=_groundedness(answer, sources),
        refused=False,
        provider=provider,
        warning=warning,
        elapsed_ms=int((__import__("time").perf_counter() - started) * 1000),
    )


__all__ = [
    "AnswerResult",
    "DocumentIndex",
    "Source",
    "ask",
    "build_index",
    "IngestError",
    "LLMError",
    "ValidationError",
]

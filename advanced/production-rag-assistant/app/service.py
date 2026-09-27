"""RAG service: the single place where the pipeline steps live.

Both the FastAPI layer and the Streamlit UI use this service. Each pipeline step
is an independent method so the orchestration layer (LangGraph, or the linear
fallback) can compose them as nodes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .pipeline.chunking import Chunk, chunk_document
from .pipeline.embeddings import build_embedder
from .pipeline.generator import generate
from .pipeline.guardrails import (
    NOT_GROUNDED_MESSAGE,
    REFUSAL_MESSAGE,
    grade_context,
    groundedness,
)
from .pipeline.ingest import IngestError, RawDocument, load_bytes, load_path
from .pipeline.preprocess import preprocess
from .pipeline.reranker import rerank
from .pipeline.retriever import Retriever
from .pipeline.text_utils import content_words
from .pipeline.vector_store import ScoredChunk, build_store
from .settings import Settings

_FILLER_PREFIXES = (
    "what is", "what are", "what does", "what do", "how do", "how does", "how is",
    "tell me about", "explain", "describe", "who is", "why is", "when is", "summarise", "summarize",
)
_SYNONYMS = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "pdf": "portable document format",
    "api": "application programming interface",
}


@dataclass
class IngestOutcome:
    documents: list[dict] = field(default_factory=list)
    total_chunks: int = 0
    elapsed_ms: int = 0


def rewrite_query(question: str) -> str:
    """Lightweight deterministic query refinement used before retrieval.

    Removes interrogative/filler framing and appends acronym expansions, so the
    retrieval query is denser and less sensitive to question phrasing.
    """

    text = question.strip().rstrip("?").strip()
    lowered = text.lower()
    for prefix in _FILLER_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    terms = content_words(text)
    extras = [_SYNONYMS[t] for t in terms if t in _SYNONYMS]
    return " ".join(part for part in [text] + extras if part).strip() or question


class RAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.embedder = build_embedder(self.settings)
        self.store = None
        self.chunks: list[Chunk] = []
        self.documents: dict[str, dict] = {}
        self.retriever: Retriever | None = None
        self._pipeline = None

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def ingest_bytes(self, payloads: list[tuple[str, bytes]]) -> IngestOutcome:
        started = time.perf_counter()
        if not payloads:
            raise IngestError("No documents were provided.", kind="empty", status=400)

        raws: list[RawDocument] = [
            load_bytes(name, data, max_file_mb=self.settings.max_file_mb, allowed=self.settings.allowed_extensions)
            for name, data in payloads
        ]
        return self._ingest_raws(raws, started)

    def ingest_paths(self, paths: list[str]) -> IngestOutcome:
        started = time.perf_counter()
        if not paths:
            raise IngestError("No documents were provided.", kind="empty", status=400)
        raws = [
            load_path(path, max_file_mb=self.settings.max_file_mb, allowed=self.settings.allowed_extensions)
            for path in paths
        ]
        return self._ingest_raws(raws, started)

    def _ingest_raws(self, raws: list[RawDocument], started: float) -> IngestOutcome:
        new_chunks: list[Chunk] = []
        documents: list[dict] = []

        for raw in raws:
            if raw.doc_id in self.documents:
                self.remove_document(raw.doc_id)
            cleaned, _stats = preprocess(raw.text)
            chunks = chunk_document(
                cleaned,
                raw.doc_id,
                raw.filename,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            new_chunks.extend(chunks)
            documents.append(
                {
                    "doc_id": raw.doc_id,
                    "filename": raw.filename,
                    "source_type": raw.source_type,
                    "chunks": len(chunks),
                    "chars": len(cleaned),
                    "words": len(cleaned.split()),
                }
            )

        if not new_chunks:
            raise IngestError("The provided documents produced no usable text.", kind="empty", status=422)

        vectors = self.embedder.embed([c.text for c in new_chunks])
        if self.store is None:
            self.store = build_store(len(vectors[0]))
        self.store.add(new_chunks, vectors)
        self.chunks.extend(new_chunks)
        for document in documents:
            self.documents[document["doc_id"]] = document
        self.retriever = Retriever(self.store, self.embedder)

        return IngestOutcome(
            documents=documents,
            total_chunks=len(self.chunks),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        self.documents.pop(doc_id, None)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self._rebuild_store()
        return True

    def clear(self) -> None:
        self.chunks = []
        self.documents = {}
        self.store = None
        self.retriever = None

    def _rebuild_store(self) -> None:
        if not self.chunks:
            self.store = None
            self.retriever = None
            return
        vectors = self.embedder.embed([c.text for c in self.chunks])
        self.store = build_store(len(vectors[0]))
        self.store.add(self.chunks, vectors)
        self.retriever = Retriever(self.store, self.embedder)

    # ------------------------------------------------------------------ #
    # pipeline nodes
    # ------------------------------------------------------------------ #
    def node_rewrite(self, state: dict) -> dict:
        question = state["question"]
        rewritten = rewrite_query(question) if state.get("rewrite", True) else question
        return {"rewritten_query": rewritten, "steps": state.get("steps", []) + ["rewrite_query"]}

    def node_retrieve(self, state: dict) -> dict:
        if self.retriever is None:
            return {"candidates": [], "steps": state.get("steps", []) + ["retrieve(empty-index)"]}
        candidates = self.retriever.retrieve(
            state.get("rewritten_query") or state["question"],
            candidate_k=self.settings.candidate_k,
            document_ids=state.get("document_ids"),
        )
        return {"candidates": candidates, "steps": state.get("steps", []) + ["retrieve"]}

    def node_rerank(self, state: dict) -> dict:
        candidates: list[ScoredChunk] = state.get("candidates", [])
        if not state.get("rerank", True):
            return {"contexts": candidates[: state.get("top_k", self.settings.top_k)], "steps": state.get("steps", []) + ["rerank(skipped)"]}
        ranked = rerank(
            state["question"],
            candidates,
            top_k=state.get("top_k", self.settings.top_k),
            mmr_lambda=self.settings.mmr_lambda,
        )
        contexts = [item.hit for item in ranked]
        return {
            "contexts": contexts,
            "rerank_scores": {item.hit.chunk.citation: item.rerank_score for item in ranked},
            "steps": state.get("steps", []) + ["rerank"],
        }

    def node_grade(self, state: dict) -> dict:
        candidates = state.get("contexts") or state.get("candidates", [])
        ok, overlap, best = grade_context(
            state["question"],
            candidates,
            min_score=self.settings.min_score,
            min_lexical_overlap=self.settings.min_lexical_overlap,
        )
        return {
            "grade_ok": ok,
            "lexical_overlap": overlap,
            "best_score": best,
            "steps": state.get("steps", []) + ["grade_context"],
        }

    def node_generate(self, state: dict) -> dict:
        answer, provider, warning = generate(state["question"], state.get("contexts", []), self.settings)
        updates = {
            "answer": answer,
            "provider": provider,
            "steps": state.get("steps", []) + ["generate"],
        }
        if warning:
            updates["warnings"] = state.get("warnings", []) + [warning]
        return updates

    def node_verify(self, state: dict) -> dict:
        contexts = state.get("contexts", [])
        score = groundedness(state.get("answer", ""), [c.chunk.text for c in contexts])
        return {
            "groundedness": score,
            "grounded_ok": score >= self.settings.min_groundedness,
            "steps": state.get("steps", []) + ["verify_groundedness"],
        }

    def node_refuse(self, state: dict) -> dict:
        message = REFUSAL_MESSAGE if not state.get("grade_ok", False) else NOT_GROUNDED_MESSAGE
        return {
            "answer": message,
            "refused": True,
            "contexts": [],
            "steps": state.get("steps", []) + ["refuse"],
        }

    def node_finish(self, state: dict) -> dict:
        return {"refused": False, "steps": state.get("steps", []) + ["finish"]}

    # ------------------------------------------------------------------ #
    # orchestration
    # ------------------------------------------------------------------ #
    def pipeline(self):
        if self._pipeline is None:
            from .graph import build_pipeline

            self._pipeline = build_pipeline(self, self.settings)
        return self._pipeline

    def ask(
        self,
        question: str,
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        rewrite: bool = True,
        rerank: bool = True,
    ) -> dict:
        started = time.perf_counter()
        state = {
            "question": question,
            "rewritten_query": None,
            "document_ids": document_ids,
            "top_k": top_k or self.settings.top_k,
            "rewrite": rewrite,
            "rerank": rerank,
            "candidates": [],
            "contexts": [],
            "rerank_scores": {},
            "grade_ok": False,
            "grounded_ok": False,
            "groundedness": 0.0,
            "refused": False,
            "answer": "",
            "provider": "",
            "warnings": [],
            "steps": [],
        }
        final = self.pipeline().run(state)
        final["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        final["orchestration"] = self.pipeline().mode
        return final

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedding_provider": getattr(self.embedder, "name", "unknown"),
            "vector_store": getattr(self.store, "name", "none"),
            "llm_provider": self.settings.llm_provider,
            "orchestration": self.pipeline().mode,
        }

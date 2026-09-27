"""Vector retrieval with optional document scoping."""
from __future__ import annotations

from .vector_store import ScoredChunk


class Retriever:
    def __init__(self, store, embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        candidate_k: int = 12,
        document_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        """Return candidate chunks for a query.

        When ``document_ids`` is given the search becomes *document-scoped*: only
        chunks from those documents are considered, so a question about one file
        cannot be answered from another.
        """

        query_vector = self.embedder.embed([query])[0]
        if document_ids:
            allowed = set(document_ids)
            # Filter client-side because the store may not support pre-filtering.
            all_hits = self.store.search(query_vector, top_k=max(candidate_k, len(self.store.chunks)))
            hits = [hit for hit in all_hits if hit.chunk.doc_id in allowed]
        else:
            hits = self.store.search(query_vector, top_k=candidate_k)
        return hits[:candidate_k]

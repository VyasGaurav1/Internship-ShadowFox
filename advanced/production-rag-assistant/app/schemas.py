"""Pydantic v2 request/response schemas.

All API input and output is typed. Request models enforce validation at the
edge (FastAPI returns HTTP 422 automatically), and response models make the
contract explicit for clients.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, description="Question to answer from the documents.")
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these document ids (document-scoped retrieval)."
    )
    top_k: int = Field(default=4, ge=1, le=20)
    rewrite: bool = Field(default=True, description="Apply query rewriting/refinement before retrieval.")
    rerank: bool = Field(default=True, description="Apply hybrid reranking after retrieval.")

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class IngestRequest(BaseModel):
    """Only used for the JSON/text ingestion convenience endpoint."""

    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)


# --------------------------------------------------------------------------- #
# responses
# --------------------------------------------------------------------------- #
class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    source_type: str
    chunks: int
    chars: int
    words: int


class IngestResponse(BaseModel):
    ingest_id: str
    documents: list[DocumentMeta]
    total_chunks: int
    embedding_provider: str
    vector_store: str
    elapsed_ms: int


class SourceModel(BaseModel):
    citation: str
    doc_id: str
    filename: str
    chunk_index: int
    score: float = Field(description="Vector similarity score.")
    rerank_score: float | None = Field(default=None, description="Hybrid rerank score when reranking is on.")
    start_char: int
    end_char: int
    text: str


class AskResponse(BaseModel):
    question: str
    rewritten_query: str | None = None
    answer: str
    sources: list[SourceModel]
    groundedness: float = Field(ge=0.0, le=1.0)
    refused: bool
    provider: str
    steps: list[str] = Field(default_factory=list, description="Observability: pipeline steps that executed.")
    elapsed_ms: int
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    embedding_provider: str
    vector_store: str
    llm_provider: str
    orchestration: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]
    total_chunks: int

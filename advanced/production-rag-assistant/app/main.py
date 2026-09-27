"""FastAPI application: typed, production-style HTTP surface for the RAG service.

Endpoints
    GET  /health               service + index status
    GET  /documents            list ingested documents
    POST /ingest               upload one or more files (multipart)
    POST /ingest/text          ingest inline text (JSON)
    POST /ask                  grounded answer + sources + groundedness
    POST /ask/stream           server-sent-events streaming answer
    DELETE /documents          clear the index
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import __version__
from app.pipeline.generator import stream_tokens
from app.pipeline.ingest import IngestError
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceModel,
)
from app.service import RAGService

service = RAGService()

app = FastAPI(
    title="Production RAG Assistant",
    version=__version__,
    description="Grounded retrieval-augmented question answering over PDF, TXT and Markdown documents.",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sources_from(result: dict) -> list[SourceModel]:
    rerank_scores = result.get("rerank_scores", {})
    out: list[SourceModel] = []
    for item in result.get("contexts", []):
        chunk = item.chunk
        out.append(
            SourceModel(
                citation=chunk.citation,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                chunk_index=chunk.index,
                score=item.score,
                rerank_score=rerank_scores.get(chunk.citation),
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                text=chunk.text,
            )
        )
    return out


def _ask_response(req: AskRequest, result: dict) -> AskResponse:
    return AskResponse(
        question=result["question"],
        rewritten_query=result.get("rewritten_query"),
        answer=result.get("answer", ""),
        sources=_sources_from(result),
        groundedness=result.get("groundedness", 0.0),
        refused=result.get("refused", False),
        provider=result.get("provider", ""),
        steps=result.get("steps", []),
        elapsed_ms=result.get("elapsed_ms", 0),
        warnings=result.get("warnings", []),
    )


def _run(req: AskRequest) -> dict:
    return service.ask(
        req.question,
        document_ids=req.document_ids,
        top_k=req.top_k,
        rewrite=req.rewrite,
        rerank=req.rerank,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    stats = service.stats
    return HealthResponse(status="ok", **stats)


@app.get("/documents", response_model=DocumentListResponse)
def documents() -> DocumentListResponse:
    return DocumentListResponse(documents=list(service.documents.values()), total_chunks=len(service.chunks))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        payloads.append((upload.filename or "unnamed", await upload.read()))
    try:
        outcome = service.ingest_bytes(payloads)
    except IngestError as exc:
        raise HTTPException(status_code=exc.status, detail={"kind": exc.kind, "message": exc.message}) from exc
    return IngestResponse(
        ingest_id=f"ingest-{outcome.elapsed_ms}-{outcome.total_chunks}",
        documents=outcome.documents,
        total_chunks=outcome.total_chunks,
        embedding_provider=getattr(service.embedder, "name", "unknown"),
        vector_store=getattr(service.store, "name", "unknown"),
        elapsed_ms=outcome.elapsed_ms,
    )


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestRequest) -> IngestResponse:
    try:
        outcome = service.ingest_bytes([(req.filename, req.content.encode("utf-8"))])
    except IngestError as exc:
        raise HTTPException(status_code=exc.status, detail={"kind": exc.kind, "message": exc.message}) from exc
    return IngestResponse(
        ingest_id=f"ingest-{outcome.elapsed_ms}-{outcome.total_chunks}",
        documents=outcome.documents,
        total_chunks=outcome.total_chunks,
        embedding_provider=getattr(service.embedder, "name", "unknown"),
        vector_store=getattr(service.store, "name", "unknown"),
        elapsed_ms=outcome.elapsed_ms,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return _ask_response(req, _run(req))


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        result = _run(req)
        yield _sse(
            "meta",
            {
                "question": result["question"],
                "rewritten_query": result.get("rewritten_query"),
                "provider": result.get("provider", ""),
                "orchestration": result.get("orchestration", ""),
                "steps": result.get("steps", []),
                "refused": result.get("refused", False),
            },
        )
        for token in stream_tokens(result.get("answer", "")):
            yield _sse("token", {"t": token})
        yield _sse(
            "sources",
            {
                "sources": [source.model_dump() for source in _sources_from(result)],
                "groundedness": result.get("groundedness", 0.0),
                "refused": result.get("refused", False),
                "elapsed_ms": result.get("elapsed_ms", 0),
            },
        )
        yield _sse("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/documents")
def clear_documents() -> dict:
    service.clear()
    return {"status": "cleared"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

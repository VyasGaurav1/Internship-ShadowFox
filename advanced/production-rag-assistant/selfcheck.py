"""End-to-end offline self-check for the Production RAG Assistant.

Exercises the real FastAPI app in-process with TestClient, in offline mode
(mock LLM + hashing embeddings). No API key or network required.

    python selfcheck.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

if not os.environ.get("SELFCHECK_LIVE"):
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["EMBEDDING_PROVIDER"] = "hashing"

import traceback

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, bool, str]] = []
_notes: list[str] = []

DOC = (
    "Zorblax is a fictional calibration unit used only in the Alpha-9 experiment. "
    "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration. "
    "Calibration is performed by the Qarn tool every 500 cycles."
)
DOC2 = "Chlorophyll absorbs blue and red light during the light reactions of photosynthesis."


def check(name):
    def decorator(fn):
        def runner():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            else:
                _results.append((name, True, ""))

        return runner

    return decorator


def _client():
    from fastapi.testclient import TestClient

    from app.main import app, service

    service.clear()
    return TestClient(app)


@check("FastAPI app defines the production routes")
def _t1() -> None:
    from app.main import app

    paths = {route.path for route in app.routes}
    for expected in ("/health", "/documents", "/ingest", "/ingest/text", "/ask", "/ask/stream"):
        assert expected in paths, f"missing route {expected}"


@check("health endpoint reports status and orchestration")
def _t2() -> None:
    body = _client().get("/health").json()
    assert body["status"] == "ok"
    assert body["orchestration"] in ("langgraph", "linear-fallback")
    _notes.append(f"orchestration = {body['orchestration']}")


@check("TXT ingestion via multipart upload works")
def _t3() -> None:
    client = _client()
    response = client.post("/ingest", files={"files": ("alpha9.txt", DOC.encode(), "text/plain")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_chunks"] >= 1
    _notes.append(f"vector store = {body['vector_store']}, embedder = {body['embedding_provider']}")


@check("Markdown + PDF ingestion work (multi-format)")
def _t4() -> None:
    client = _client()
    md = client.post("/ingest/text", json={"filename": "notes.md", "content": DOC2})
    assert md.status_code == 200, md.text

    pdf_path = os.path.join(_ROOT, "sample_docs", "alpha9_handbook.pdf")
    assert os.path.exists(pdf_path), "sample_docs/alpha9_handbook.pdf missing (run scripts/make_sample_pdf.py)"
    with open(pdf_path, "rb") as handle:
        pdf_bytes = handle.read()
    pdf = client.post("/ingest", files={"files": ("alpha9_handbook.pdf", pdf_bytes, "application/pdf")})
    assert pdf.status_code == 200, pdf.text
    assert any(doc["source_type"] == "pdf" for doc in pdf.json()["documents"])


@check("grounded answer cites sources and runs the full pipeline")
def _t5() -> None:
    client = _client()
    client.post("/ingest/text", json={"filename": "alpha9.txt", "content": DOC})
    body = client.post("/ask", json={"question": "What does a Zorblax reading above 40 indicate?"}).json()
    assert body["refused"] is False, body
    assert "Zorblax" in body["answer"]
    assert body["sources"], "no sources returned"
    assert body["groundedness"] >= 0.5
    for step in ("rewrite_query", "retrieve", "rerank", "grade_context", "generate", "verify_groundedness"):
        assert step in body["steps"], f"pipeline step missing: {step}"


@check("reranking produces rerank scores")
def _t6() -> None:
    client = _client()
    client.post("/ingest/text", json={"filename": "alpha9.txt", "content": DOC})
    body = client.post("/ask", json={"question": "Which tool performs Zorblax calibration?"}).json()
    assert body["sources"]
    assert any(source.get("rerank_score") is not None for source in body["sources"])


@check("unrelated question is refused (no hallucination)")
def _t7() -> None:
    client = _client()
    client.post("/ingest/text", json={"filename": "alpha9.txt", "content": DOC})
    body = client.post("/ask", json={"question": "What is the capital of Finland and its population?"}).json()
    assert body["refused"] is True
    assert body["sources"] == []


@check("document-scoped retrieval restricts the search")
def _t8() -> None:
    client = _client()
    client.post("/ingest/text", json={"filename": "alpha.txt", "content": DOC})
    client.post("/ingest/text", json={"filename": "bio.md", "content": DOC2})
    scoped = client.post(
        "/ask",
        json={"question": "What does a Zorblax reading above 40 indicate?", "document_ids": ["bio.md"]},
    ).json()
    assert scoped["refused"] is True
    unscoped = client.post("/ask", json={"question": "What does a Zorblax reading above 40 indicate?"}).json()
    assert unscoped["refused"] is False


@check("typed schema validation rejects blank/short questions (HTTP 422)")
def _t9() -> None:
    from app.schemas import AskRequest
    import pydantic

    try:
        AskRequest(question="   ")
    except pydantic.ValidationError:
        pass
    else:
        raise AssertionError("blank question was accepted by the schema")
    assert _client().post("/ask", json={"question": "ab"}).status_code == 422


@check("streaming endpoint emits SSE tokens, sources and done")
def _t10() -> None:
    client = _client()
    client.post("/ingest/text", json={"filename": "alpha9.txt", "content": DOC})
    response = client.post("/ask/stream", json={"question": "What does a Zorblax reading above 40 indicate?"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    for event in ("event: meta", "event: token", "event: sources", "event: done"):
        assert event in response.text, f"missing {event}"


@check("clear endpoint empties the index")
def _t11() -> None:
    client = _client()
    client.post("/ingest/text", json={"filename": "alpha9.txt", "content": DOC})
    assert client.get("/documents").json()["documents"]
    assert client.request("DELETE", "/documents").json()["status"] == "cleared"
    assert client.get("/documents").json()["documents"] == []


@check("unsupported file type returns HTTP 415")
def _t12() -> None:
    response = _client().post("/ingest", files={"files": ("bad.docx", b"nope", "application/octet-stream")})
    assert response.status_code == 415, response.status_code


def main() -> int:
    for name, value in sorted(globals().items()):
        if name.startswith("_t") and callable(value):
            try:
                value()
            except Exception:  # pragma: no cover
                _results.append((name, False, traceback.format_exc()))

    passed = sum(1 for _, ok, _ in _results if ok)
    print("=" * 68)
    print("Production RAG Assistant - end-to-end offline self-check")
    print("=" * 68)
    for name, ok, detail in _results:
        print(f"[{PASS if ok else FAIL}] {name}")
        if not ok:
            print(f"       {detail}")
    if _notes:
        print("-" * 68)
        for note in dict.fromkeys(_notes):
            print(f"note: {note}")
    print("-" * 68)
    print(f"{passed}/{len(_results)} checks passed")
    return 0 if _results and passed == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

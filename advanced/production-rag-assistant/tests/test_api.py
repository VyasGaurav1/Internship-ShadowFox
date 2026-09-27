import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["orchestration"] in ("langgraph", "linear-fallback")


def test_ingest_text_and_ask_grounded(client):
    ingest = client.post(
        "/ingest/text",
        json={
            "filename": "alpha9.txt",
            "content": (
                "Zorblax is a fictional calibration unit used only in the Alpha-9 experiment. "
                "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration. "
                "Calibration is performed by the Qarn tool every 500 cycles."
            ),
        },
    )
    assert ingest.status_code == 200, ingest.text
    assert ingest.json()["total_chunks"] >= 1

    answer = client.post("/ask", json={"question": "What does a Zorblax reading above 40 indicate?"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["refused"] is False
    assert "Zorblax" in body["answer"]
    assert body["sources"]
    assert body["groundedness"] >= 0.5
    assert "retrieve" in body["steps"]
    assert "rerank" in body["steps"]


def test_unrelated_question_is_refused(client):
    client.post("/ingest/text", json={"filename": "a.txt", "content": "Zorblax readings indicate recalibration needs."})
    body = client.post("/ask", json={"question": "Who won the 1998 FIFA world cup final?"}).json()
    assert body["refused"] is True
    assert body["sources"] == []


def test_empty_question_is_422(client):
    response = client.post("/ask", json={"question": "  "})
    assert response.status_code == 422


def test_short_question_is_422(client):
    response = client.post("/ask", json={"question": "ab"})
    assert response.status_code == 422


def test_document_scoped_retrieval(client):
    client.post("/ingest/text", json={"filename": "alpha.txt", "content": "Zorblax reading above 40 needs recalibration."})
    client.post("/ingest/text", json={"filename": "bio.md", "content": "Chlorophyll absorbs blue and red light in photosynthesis."})

    scoped = client.post(
        "/ask",
        json={"question": "What does a Zorblax reading above 40 indicate?", "document_ids": ["bio.md"]},
    ).json()
    assert scoped["refused"] is True, "document scoping failed to exclude the other document"

    unscoped = client.post("/ask", json={"question": "What does a Zorblax reading above 40 indicate?"}).json()
    assert unscoped["refused"] is False


def test_unsupported_file_type(client):
    response = client.post("/ingest", files={"files": ("bad.docx", b"nope", "application/octet-stream")})
    assert response.status_code == 415


def test_streaming_endpoint(client):
    client.post("/ingest/text", json={"filename": "a.txt", "content": "Zorblax reading above 40 needs recalibration."})
    response = client.post("/ask/stream", json={"question": "What does a Zorblax reading above 40 indicate?"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: token" in response.text
    assert "event: sources" in response.text
    assert "event: done" in response.text


def test_documents_and_clear(client):
    client.post("/ingest/text", json={"filename": "a.txt", "content": "Some content about Zorblax calibration units."})
    assert client.get("/documents").json()["documents"]
    assert client.request("DELETE", "/documents").json()["status"] == "cleared"
    assert client.get("/documents").json()["documents"] == []

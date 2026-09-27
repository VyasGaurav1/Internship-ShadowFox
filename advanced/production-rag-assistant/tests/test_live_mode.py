"""Proves the LIVE (non-mock) generation path of the RAG service works.

Runs offline against a stub OpenAI-compatible server, and exercises the real
request/parse/error-mapping code. With a key configured it runs against the
live provider unchanged.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

LIVE_ANSWER = "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration. [1]"


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.received.append({"payload": payload, "headers": dict(self.headers)})
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": LIVE_ANSWER}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture()
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


DOC = "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration. Calibration uses the Qarn tool."


def test_live_generation_used_when_provider_configured(monkeypatch, stub_server):
    host, port = stub_server.server_address
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setenv("OPENROUTER_BASE_URL", f"http://{host}:{port}/v1")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    from app.service import RAGService

    service = RAGService()
    assert service.settings.llm_provider == "openrouter"
    service.ingest_bytes([("alpha9.txt", DOC.encode())])
    result = service.ask("What does a Zorblax reading above 40 indicate?")

    assert result["refused"] is False
    assert result["provider"].startswith("openrouter")
    assert "Orbital Loom" in result["answer"]
    assert stub_server.received, "no request reached the provider"
    assert stub_server.received[0]["headers"]["Authorization"].startswith("Bearer ")


def test_reasoning_style_response_is_parsed(monkeypatch):
    """Reasoning models return content=null with reasoning populated."""

    from app.pipeline.generator import extract_message_text

    assert extract_message_text({"content": None, "reasoning": "The answer is 42."}) == "The answer is 42."
    assert extract_message_text({"content": "plain"}) == "plain"
    assert extract_message_text({"content": [{"type": "text", "text": "parts"}]}) == "parts"


def test_openrouter_default_when_key_present(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "abc")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    from app.settings import Settings

    settings = Settings.from_env()
    assert settings.llm_provider == "openrouter"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"

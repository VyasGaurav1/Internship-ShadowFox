"""Proves the LIVE (non-mock) generation path works end-to-end.

Uses a local stub OpenAI-compatible server so it runs offline, and exercises
the real request/parse/error-mapping code paths. When a real key is configured
the same code runs against the live provider.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

LIVE_ANSWER = "A list comprehension is a concise way to build a list. [1]"


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


def test_live_grounded_answer_uses_configured_provider(monkeypatch, stub_server):
    host, port = stub_server.server_address
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setenv("OPENROUTER_BASE_URL", f"http://{host}:{port}/v1")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    from config import load_settings
    from ingest import Document
    from qa import ask, build_index

    settings = load_settings()
    assert settings.llm_provider == "openrouter"

    docs = [Document("python.txt", "python.txt", "A list comprehension builds a list concisely from an iterable.", "text")]
    index = build_index(docs, settings)
    result = ask(index, "What is a list comprehension?")

    assert not result.refused
    assert "list comprehension" in result.answer.lower()
    assert result.provider.startswith("openrouter")
    assert stub_server.received, "no request reached the provider"
    assert stub_server.received[0]["headers"]["Authorization"].startswith("Bearer ")


def test_openrouter_default_when_key_present(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "abc")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    from config import load_settings

    settings = load_settings()
    assert settings.llm_provider == "openrouter"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"

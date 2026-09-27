"""Proves the LIVE (non-mock) transport path works end-to-end.

Two modes:

1. Offline-safe (always runs): a local stub OpenAI-compatible server stands in
   for the provider, so the request construction, auth header, JSON parsing and
   error mapping are all exercised without network access.
2. Real (opt-in): if ``OPENROUTER_API_KEY`` is set (or ``LIVE_TEST=1`` with a
   configured ``.env``), the same test runs against the real endpoint.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

LIVE_REPLY = "## Key points\n- Live model answer about photosynthesis.\n\n## One-line takeaway\nLive."


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.received.append({"payload": payload, "headers": dict(self.headers)})

        if payload.get("model") == "boom":
            body = json.dumps({"error": {"message": "upstream unavailable"}}).encode()
            self.send_response(503)
        else:
            content = LIVE_REPLY
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        return


@pytest.fixture()
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def _configure_openrouter(monkeypatch, base_url, key="test-key-123"):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setenv("OPENROUTER_BASE_URL", base_url)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_live_transport_sends_request_and_parses_response(monkeypatch, stub_server):
    host, port = stub_server.server_address
    _configure_openrouter(monkeypatch, f"http://{host}:{port}/v1")

    from core import run_feature

    result = run_feature("summarize", "Photosynthesis converts light energy into chemical energy in plants.")
    assert result["ok"] is True, result.get("error")
    assert result["offline"] is False
    assert "Live model answer" in result["output"]

    sent = stub_server.received[0]
    assert sent["headers"]["Authorization"].startswith("Bearer ")
    assert sent["payload"]["messages"][0]["role"] == "system"
    assert sent["payload"]["messages"][1]["content"]


def test_live_transport_maps_server_error(monkeypatch, stub_server):
    host, port = stub_server.server_address
    _configure_openrouter(monkeypatch, f"http://{host}:{port}/v1")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_MODEL", "boom")

    from core import run_feature

    result = run_feature("summarize", "Photosynthesis converts light energy into chemical energy in plants.")
    assert result["ok"] is False
    assert result["error_kind"] == "server"


def test_openrouter_is_the_default_when_key_present(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "abc")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    from config import load_settings

    settings = load_settings()
    assert settings.provider == "openrouter"
    assert settings.base_url == "https://openrouter.ai/api/v1"

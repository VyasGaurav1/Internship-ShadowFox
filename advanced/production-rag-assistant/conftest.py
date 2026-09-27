"""Shared test fixtures: project importability, offline mode, fresh API service."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if not os.environ.get("SELFCHECK_LIVE"):
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["EMBEDDING_PROVIDER"] = "hashing"

import pytest


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app, service

    service.clear()
    with TestClient(app) as test_client:
        yield test_client
    service.clear()

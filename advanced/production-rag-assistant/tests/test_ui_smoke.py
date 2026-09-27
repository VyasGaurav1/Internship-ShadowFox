"""Render smoke test for the Production RAG Assistant Streamlit UI (no browser).

AppTest cannot simulate a file upload, so the ingest/ask path is covered by
``tests/test_api.py``, ``tests/test_live_mode.py`` and ``selfcheck.py``.
"""
from __future__ import annotations

import os

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "streamlit_app.py"
)


def test_ui_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, [getattr(error, "value", error) for error in at.exception]

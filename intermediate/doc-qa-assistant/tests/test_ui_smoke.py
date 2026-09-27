"""Render + interaction smoke test for the DocuAsk Streamlit UI (no browser needed).

The reported crash happened while building the index from the sidebar sliders, so
this test also drives those widgets. Streamlit's AppTest cannot simulate a file
upload, so the upload->build path itself is covered by
``tests/test_settings_overrides.py`` and ``selfcheck.py``.
"""
from __future__ import annotations

import os

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, [getattr(error, "value", error) for error in at.exception]


def test_changing_sidebar_sliders_does_not_raise():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    sliders = at.sidebar.slider
    if len(sliders) >= 2:
        sliders[0].set_value(100)   # chunk size
        sliders[1].set_value(20)    # chunk overlap
        at.run()
    assert not at.exception, [getattr(error, "value", error) for error in at.exception]

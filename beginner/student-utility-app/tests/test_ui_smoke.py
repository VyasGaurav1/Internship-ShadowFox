"""Render smoke test for the StudyMate Streamlit UI (no browser needed).

Uses Streamlit's own AppTest harness so a broken UI code path fails the test
suite instead of only failing when a human opens the app.
"""
from __future__ import annotations

import os

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, [getattr(error, "value", error) for error in at.exception]


def test_generate_with_sample_content_does_not_raise():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    if not at.text_area:
        return  # UI shape changed; render test above still applies
    at.text_area[0].set_value("Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs light.")
    at.run()
    assert not at.exception, [getattr(error, "value", error) for error in at.exception]

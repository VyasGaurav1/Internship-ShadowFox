"""Ensure the project root is importable when running `pytest`, and pin the
deterministic offline provider so tests never depend on network or API keys."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("LLM_PROVIDER", "")
if not os.environ.get("SELFCHECK_LIVE"):
    os.environ["LLM_PROVIDER"] = "mock"

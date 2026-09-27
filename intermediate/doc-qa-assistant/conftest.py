"""Ensure project root importability + deterministic offline mode for tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if not os.environ.get("SELFCHECK_LIVE"):
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["EMBEDDING_PROVIDER"] = "hashing"

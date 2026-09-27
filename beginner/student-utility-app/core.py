"""Feature orchestration for the Student Utility App.

This is the single entry point used by both the Streamlit UI and the CLI:

    validate input -> build structured prompt -> call provider -> format result

Every failure mode is converted into a structured, user-safe result instead of
an exception bubbling into the UI.
"""
from __future__ import annotations

from typing import Any

from llm_client import LLMClient, LLMError
from prompts import FEATURES
from validators import ValidationError, validate_content, validate_feature


def run_feature(
    feature: str,
    content: str,
    options: dict | None = None,
    client: LLMClient | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Run one utility feature and return a structured result dict.

    Returns a dict with ``ok`` plus either the generation payload
    (``output``, ``provider``, ``model``, ``offline``, ``elapsed_ms``) or an
    error payload (``error``, ``error_kind``).
    """

    options = options or {}

    try:
        feature = validate_feature(feature)
    except ValidationError as exc:
        return {"ok": False, "feature": feature, "error": exc.message, "error_kind": "validation"}

    if min_chars is None or max_chars is None:
        from config import load_settings

        settings = load_settings()
        min_chars = min_chars if min_chars is not None else settings.min_input_chars
        max_chars = max_chars if max_chars is not None else settings.max_input_chars

    try:
        clean = validate_content(content, min_chars=min_chars, max_chars=max_chars)
    except ValidationError as exc:
        return {"ok": False, "feature": feature, "error": exc.message, "error_kind": "validation"}

    try:
        client = client or LLMClient()
        result = client.complete(feature, clean, options)
    except LLMError as exc:
        return {
            "ok": False,
            "feature": feature,
            "error": exc.message,
            "error_kind": exc.kind,
        }
    except Exception as exc:  # never let the UI crash on unexpected errors
        return {
            "ok": False,
            "feature": feature,
            "error": "Something went wrong while generating the answer. Please try again.",
            "error_kind": "unexpected",
        }

    return {
        "ok": True,
        "feature": feature,
        "output": result.text,
        "provider": result.provider,
        "model": result.model,
        "offline": result.offline,
        "elapsed_ms": result.elapsed_ms,
    }


__all__ = ["run_feature", "FEATURES", "LLMError", "ValidationError"]

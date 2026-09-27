"""Minimal OpenAI-compatible chat client used for grounded answer generation.

Supports OpenRouter (recommended), any OpenAI-compatible endpoint, and Zhipu
GLM. Only used when a real provider is configured; every failure becomes a
typed :class:`LLMError` and the QA layer falls back to deterministic extractive
grounding.
"""
from __future__ import annotations

import json
import os
import re
import time

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


class LLMError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "error", status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.message = message


def clean_model_text(text: str) -> str:
    """Strip surrounding markdown code fences from model output."""

    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_message_text(message: dict) -> str:
    """Extract text from an OpenAI-style ``message`` object.

    Handles string content, list-of-parts content, and reasoning models that
    return ``content: null`` with ``reasoning``/``reasoning_content`` populated.
    """

    content = message.get("content")
    if content is None:
        content = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise LLMError("The AI service returned an empty answer.", kind="empty")
    return clean_model_text(content)


def provider_headers(provider: str, api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if provider in ("openrouter", "omniroute"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "https://github.com/")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "AI Engineer Internship - DocuAsk")
    return headers


def _error_detail(response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])[:200]
            if isinstance(error, str):
                return error[:200]
    except Exception:
        pass
    return ""


def call_llm(settings, system: str, user: str) -> str:
    if requests is None:  # pragma: no cover
        raise LLMError("The 'requests' package is not installed.", kind="dependency")
    if not settings.llm_api_key and settings.llm_provider != "omniroute":
        raise LLMError(f"No API key for provider '{settings.llm_provider}'.", kind="auth")

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    headers = provider_headers(settings.llm_provider, settings.llm_api_key)

    last_error: Exception | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=settings.llm_timeout
            )
        except Exception as exc:
            last_error = LLMError("Could not reach the AI service.", kind="network")
            last_error.__cause__ = exc
        else:
            if response.status_code == 200:
                try:
                    message = response.json()["choices"][0]["message"]
                except Exception as exc:
                    raise LLMError("Unreadable response from the AI service.", kind="parse") from exc
                return extract_message_text(message)
            detail = _error_detail(response)
            if response.status_code in (401, 403):
                raise LLMError(
                    f"The AI service rejected the credentials. {detail}".strip(),
                    kind="auth",
                    status=response.status_code,
                )
            if response.status_code == 429:
                last_error = LLMError("The AI service is busy (rate limited).", kind="rate_limit", status=429)
            elif 500 <= response.status_code < 600:
                last_error = LLMError("The AI service had a temporary problem.", kind="server", status=response.status_code)
            else:
                raise LLMError(f"Unexpected AI response (HTTP {response.status_code}). {detail}".strip(), kind="http", status=response.status_code)

        if attempt < settings.llm_max_retries:
            time.sleep(0.6 * (attempt + 1))

    raise last_error or LLMError("The AI request failed.", kind="unknown")

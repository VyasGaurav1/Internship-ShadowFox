"""LLM provider abstraction for the Student Utility App.

The rest of the app talks to :class:`LLMClient` only. Providers supported
behind one interface:

* ``openrouter`` - OpenRouter (https://openrouter.ai/api/v1) or any
  OpenRouter-compatible gateway; the recommended default for live use.
* ``openai``     - any OpenAI-compatible ``/chat/completions`` endpoint
* ``zai``        - Zhipu GLM (OpenAI-compatible)
* ``mock``       - offline, deterministic, network-free (default when no key)

Real providers use only ``requests`` (no vendor SDK) so the dependency surface
stays tiny. All failures are surfaced as a typed :class:`LLMError` with a
friendly, user-safe message.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from config import Settings, load_settings
from offline_engine import offline_generate
from prompts import build_messages

try:  # optional dependency
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


class LLMError(RuntimeError):
    """A provider-agnostic error with a message that is safe to show users."""

    def __init__(self, message: str, *, kind: str = "error", status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.message = message


def clean_model_text(text: str) -> str:
    """Normalise model output: strip surrounding markdown code fences."""

    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_message_text(message: dict) -> str:
    """Extract text from an OpenAI-style ``message`` object.

    Handles three real-world shapes:
    * ``content`` as a plain string (normal chat models);
    * ``content`` as a list of parts (multimodal responses);
    * ``content`` null with ``reasoning``/``reasoning_content`` populated
      (reasoning models routed through aggregators such as OpenRouter).
    """

    content = message.get("content")
    if content is None:
        content = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise LLMError(
            "The AI service returned an empty answer. Please try again.", kind="empty"
        )
    return clean_model_text(content)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    offline: bool
    elapsed_ms: int


def provider_headers(provider: str, api_key: str) -> dict:
    """Build request headers.

    A local Omniroute router does not require a credential, so the Authorization
    header is only sent when a key is actually present.
    """

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if provider in ("openrouter", "omniroute"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "https://github.com/")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "AI Engineer Internship - Student Utility")
    return headers


class LLMClient:
    """Thin, well-behaved client around an OpenAI-compatible chat endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    # -- public API ---------------------------------------------------------
    def complete(self, feature: str, content: str, options: dict | None = None) -> LLMResult:
        """Run one generation for ``feature`` over ``content``."""

        start = time.perf_counter()
        messages = build_messages(feature, content, options)

        if self.settings.provider == "mock":
            text = offline_generate(feature, content, options)
            return LLMResult(
                text=text,
                provider="mock",
                model=self.settings.model,
                offline=True,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )

        text = self._call_openai_compatible(messages)
        return LLMResult(
            text=text,
            provider=self.settings.provider,
            model=self.settings.model,
            offline=False,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )

    # -- internals ----------------------------------------------------------
    def _call_openai_compatible(self, messages: dict) -> str:
        if requests is None:
            raise LLMError(
                "The 'requests' package is not installed, so live API calls are unavailable. "
                "Install the dependencies or set LLM_PROVIDER=mock to use offline mode.",
                kind="dependency",
            )
        if not self.settings.credential and self.settings.provider != "omniroute":
            raise LLMError(
                f"No API key found for provider '{self.settings.provider}'. "
                "Set the key environment variable or use offline mode.",
                kind="auth",
            )

        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        headers = provider_headers(self.settings.provider, self.settings.credential)

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = requests.post(
                    url, headers=headers, data=json.dumps(payload), timeout=self.settings.timeout
                )
            except Exception as exc:  # network level failure
                last_error = LLMError(
                    "Could not reach the AI service. Please check your connection and try again.",
                    kind="network",
                )
                last_error.__cause__ = exc
            else:
                if response.status_code == 200:
                    return self._parse_response(response)
                detail = self._error_detail(response)
                if response.status_code in (401, 403):
                    raise LLMError(
                        f"The AI service rejected the credentials. {detail}".strip(),
                        kind="auth",
                        status=response.status_code,
                    )
                if response.status_code == 429:
                    last_error = LLMError(
                        "The AI service is busy (rate limited). Please retry in a moment.",
                        kind="rate_limit",
                        status=429,
                    )
                elif 500 <= response.status_code < 600:
                    last_error = LLMError(
                        "The AI service had a temporary problem. Please try again.",
                        kind="server",
                        status=response.status_code,
                    )
                else:
                    raise LLMError(
                        f"The AI service returned an unexpected response (HTTP {response.status_code}). {detail}".strip(),
                        kind="http",
                        status=response.status_code,
                    )

            if attempt < self.settings.max_retries:
                time.sleep(0.6 * (attempt + 1))

        raise last_error or LLMError("The AI request failed for an unknown reason.", kind="unknown")

    @staticmethod
    def _error_detail(response) -> str:
        """Pull a short, safe error message out of the provider response body."""

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

    @staticmethod
    def _parse_response(response) -> str:
        try:
            message = response.json()["choices"][0]["message"]
        except Exception as exc:
            raise LLMError(
                "The AI service returned a response we could not read. Please try again.",
                kind="parse",
            ) from exc
        return extract_message_text(message)

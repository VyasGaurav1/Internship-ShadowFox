"""Grounded answer generation.

* ``llm`` path: a chat model conditioned strictly on the numbered context
  passages, required to cite them and to refuse when the passages are
  insufficient.
* ``extractive`` path (offline / fallback): the answer is assembled from the
  highest-overlap sentences of the retrieved chunks, so it is grounded by
  construction and needs no API key.
"""
from __future__ import annotations

import json
import os
import re
import time

from .text_utils import content_words, split_sentences
from .vector_store import ScoredChunk

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


class GenerationError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind
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
    return ``content: null`` with ``reasoning``/``reasoning_content`` populated
    (common with aggregators such as OpenRouter).
    """

    content = message.get("content")
    if content is None:
        content = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("The AI service returned an empty answer.", kind="empty")
    return clean_model_text(content)


def provider_headers(provider: str, api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if provider in ("openrouter", "omniroute"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "https://github.com/")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "AI Engineer Internship - Production RAG")
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


GROUNDING_SYSTEM = (
    "You are a document-grounded question answering assistant for a production RAG system. "
    "Answer ONLY from the numbered context passages supplied in the user message. "
    "Cite every claim with its passage number, e.g. [1]. "
    "If the passages are insufficient, reply exactly: "
    "'The uploaded documents do not contain enough information to answer this.' "
    "Do not use outside knowledge. Be concise (2-6 sentences)."
)


def _build_user_prompt(question: str, contexts: list[ScoredChunk]) -> str:
    blocks = [f"[{i + 1}] ({c.chunk.citation})\n{c.chunk.text}" for i, c in enumerate(contexts)]
    return f"Question: {question}\n\nContext passages:\n" + "\n\n".join(blocks)


def extractive_answer(question: str, contexts: list[ScoredChunk], max_sentences: int = 3) -> str:
    q_terms = set(content_words(question))
    scored: list[tuple[float, str]] = []
    for item in contexts:
        for sentence in split_sentences(item.chunk.text):
            s_terms = set(content_words(sentence))
            overlap = len(q_terms & s_terms)
            if overlap == 0:
                continue
            scored.append((overlap / (1 + len(s_terms) ** 0.5) + item.score, sentence))

    if not scored:
        first = split_sentences(contexts[0].chunk.text) if contexts else []
        return first[0] if first else (contexts[0].chunk.text[:280] if contexts else "")

    seen: set[str] = set()
    chosen: list[str] = []
    for _, sentence in sorted(scored, key=lambda pair: -pair[0]):
        if sentence in seen:
            continue
        seen.add(sentence)
        chosen.append(sentence)
        if len(chosen) >= max_sentences:
            break
    return " ".join(chosen)


def llm_answer(question: str, contexts: list[ScoredChunk], settings) -> str:
    if requests is None:  # pragma: no cover
        raise GenerationError("The 'requests' package is not installed.", kind="dependency")
    if not settings.llm_api_key and settings.llm_provider != "omniroute":
        raise GenerationError(f"No API key for provider '{settings.llm_provider}'.", kind="auth")

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": GROUNDING_SYSTEM},
            {"role": "user", "content": _build_user_prompt(question, contexts)},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    headers = provider_headers(settings.llm_provider, settings.llm_api_key)

    last: Exception | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=settings.llm_timeout)
        except Exception as exc:
            last = GenerationError("Could not reach the AI service.", kind="network")
            last.__cause__ = exc
        else:
            if response.status_code == 200:
                try:
                    message = response.json()["choices"][0]["message"]
                except Exception as exc:
                    raise GenerationError("Unreadable response from the AI service.", kind="parse") from exc
                return extract_message_text(message)
            detail = _error_detail(response)
            if response.status_code in (401, 403):
                raise GenerationError(f"The AI service rejected the credentials. {detail}".strip(), kind="auth")
            if response.status_code == 429:
                last = GenerationError("The AI service is busy (rate limited).", kind="rate_limit")
            elif 500 <= response.status_code < 600:
                last = GenerationError("The AI service had a temporary problem.", kind="server")
            else:
                raise GenerationError(f"Unexpected AI response (HTTP {response.status_code}). {detail}".strip(), kind="http")
        if attempt < settings.llm_max_retries:
            time.sleep(0.6 * (attempt + 1))
    raise last or GenerationError("The AI request failed.", kind="unknown")


def generate(question: str, contexts: list[ScoredChunk], settings) -> tuple[str, str, str | None]:
    """Return ``(answer, provider_label, warning)``."""

    if settings.offline_llm:
        return extractive_answer(question, contexts), "extractive (offline)", None
    try:
        return llm_answer(question, contexts, settings), f"{settings.llm_provider}:{settings.llm_model}", None
    except GenerationError as exc:
        return (
            extractive_answer(question, contexts),
            "extractive (offline fallback)",
            f"Live model unavailable ({exc.kind}); grounded extractive answer used instead.",
        )


def stream_tokens(text: str):
    """Yield ``text`` in word-sized pieces for streaming responses."""

    for token in re.findall(r"\S+\s*", text):
        yield token

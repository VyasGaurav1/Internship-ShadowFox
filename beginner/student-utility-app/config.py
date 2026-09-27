"""Configuration loading for the Student Utility App.

Everything is environment driven so the same code runs in four modes:

* ``mock``       - fully offline, deterministic, no key, no network (default fallback)
* ``openrouter`` - OpenRouter (recommended for live use) or a compatible gateway
* ``openai``     - any OpenAI-compatible ``/chat/completions`` endpoint
* ``zai``        - Zhipu GLM (OpenAI-compatible)

Resolution order for the provider:
1. explicit ``LLM_PROVIDER`` environment variable
2. ``openrouter`` when ``OPENROUTER_API_KEY`` is present
3. ``openai`` when ``OPENAI_API_KEY`` is present
4. ``zai`` when ``ZAI_API_KEY`` is present
5. ``mock``
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional dependency, safe to omit
    from dotenv import load_dotenv

    # Load the project-local .env by absolute path so the configuration works
    # no matter which directory the app is launched from, then fall back to the
    # standard cwd/upward search.
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


DEFAULT_BASE_URLS = {
    "omniroute": "http://localhost:20128/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "zai": "https://open.bigmodel.cn/api/paas/v4",
}

DEFAULT_MODELS = {
    "omniroute": "auto/best-fast",
    "openrouter": "openai/gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "zai": "glm-4-flash",
    "mock": "offline-deterministic-v1",
}

KNOWN_PROVIDERS = ("omniroute", "openrouter", "openai", "zai", "mock")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the app."""

    provider: str
    model: str
    credential: str | None
    base_url: str
    timeout: float
    max_retries: int
    temperature: float
    max_tokens: int
    min_input_chars: int
    max_input_chars: int

    @property
    def is_offline(self) -> bool:
        return self.provider == "mock"


def _resolve_provider() -> str:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("OMNIROUTE_API_KEY"):
        return "omniroute"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ZAI_API_KEY"):
        return "zai"
    return "mock"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _provider_credential(provider: str) -> tuple[str | None, str]:
    """Return ``(credential, base_url)`` for the selected provider."""

    if provider == "omniroute":
        return os.getenv("OMNIROUTE_API_KEY") or os.getenv("OPENROUTER_API_KEY"), os.getenv("OMNIROUTE_BASE_URL") or DEFAULT_BASE_URLS["omniroute"]
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URLS["openrouter"]
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URLS["openai"]
    if provider == "zai":
        return os.getenv("ZAI_API_KEY"), os.getenv("ZAI_BASE_URL") or DEFAULT_BASE_URLS["zai"]
    return None, ""


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""

    provider = _resolve_provider()
    if provider not in KNOWN_PROVIDERS:
        # Unknown provider -> fall back to the safe offline mode.
        provider = "mock"

    credential, base_url = _provider_credential(provider)
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    return Settings(
        provider=provider,
        model=model,
        credential=credential,
        base_url=base_url,
        timeout=_env_float("LLM_TIMEOUT", 45.0),
        max_retries=_env_int("LLM_MAX_RETRIES", 2),
        temperature=_env_float("LLM_TEMPERATURE", 0.3),
        max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
        min_input_chars=_env_int("MIN_INPUT_CHARS", 15),
        max_input_chars=_env_int("MAX_INPUT_CHARS", 12000),
    )

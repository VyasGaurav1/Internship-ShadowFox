"""Configuration for the Document Q&A Assistant (intermediate tier)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    # Project-local .env by absolute path (works from any launch directory), then the standard search.
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    load_dotenv()
except Exception:  # pragma: no cover
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
    "mock": "offline-extractive-v1",
}


@dataclass(frozen=True)
class Settings:
    # retrieval
    chunk_size: int          # words per chunk
    chunk_overlap: int       # overlapping words between chunks
    top_k: int               # chunks retrieved per query
    min_score: float         # below this, we refuse to answer (anti-hallucination)
    min_lexical_overlap: int # min shared content words between question and top chunk
    embedding_provider: str  # hashing | sentence-transformers | openai
    embedding_dim: int
    embedding_model: str
    # generation
    llm_provider: str        # openai | zai | mock
    llm_model: str
    llm_api_key: str | None
    llm_base_url: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout: float
    llm_max_retries: int
    # ingestion
    max_file_mb: float
    allowed_extensions: tuple

    @property
    def offline_llm(self) -> bool:
        return self.llm_provider == "mock"


def _env_float(name, default):
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_int(name, default):
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _resolve_llm_provider() -> str:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("omniroute", "openrouter", "openai", "zai", "mock"):
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


def resolve_embedding_provider(requested: str) -> str:
    """Fall back to the offline hashing embedder when the requested one is not installable."""

    requested = (requested or "hashing").strip().lower()
    if requested in ("sentence-transformers", "st", "sbert"):
        try:
            import sentence_transformers  # noqa: F401

            return "sentence-transformers"
        except Exception:
            return "hashing"
    if requested == "openai":
        return "openai" if os.getenv("OPENAI_API_KEY") else "hashing"
    return "hashing"


def load_settings() -> Settings:
    provider = _resolve_llm_provider()
    if provider == "omniroute":
        api_key = os.getenv("OMNIROUTE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OMNIROUTE_BASE_URL") or DEFAULT_BASE_URLS["omniroute"]
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URLS["openrouter"]
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URLS["openai"]
    elif provider == "zai":
        api_key = os.getenv("ZAI_API_KEY")
        base_url = os.getenv("ZAI_BASE_URL") or DEFAULT_BASE_URLS["zai"]
    else:
        api_key, base_url = None, ""

    emb_provider = resolve_embedding_provider(os.getenv("EMBEDDING_PROVIDER", "hashing"))

    return Settings(
        chunk_size=_env_int("CHUNK_SIZE", 140),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 30),
        top_k=_env_int("TOP_K", 4),
        min_score=_env_float("MIN_SCORE", 0.05),
        min_lexical_overlap=_env_int("MIN_LEXICAL_OVERLAP", 1),
        embedding_provider=emb_provider,
        embedding_dim=_env_int("EMBEDDING_DIM", 512),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider],
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 900),
        llm_timeout=_env_float("LLM_TIMEOUT", 30.0),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        max_file_mb=_env_float("MAX_FILE_MB", 10.0),
        allowed_extensions=(".txt", ".md", ".markdown", ".pdf"),
    )


def with_overrides(
    settings: Settings,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    top_k: int | None = None,
) -> Settings:
    """Return a copy of ``settings`` with sidebar/UI values applied.

    This is the *single* place where the UI overrides are translated into a
    :class:`Settings` copy. Keeping it here means a call site can never invent a
    field name (the original bug passed ``overlap=`` when the field is
    ``chunk_overlap``), and the behaviour is covered by tests.
    """

    from dataclasses import replace

    changes: dict = {}
    if chunk_size is not None and chunk_size != settings.chunk_size:
        changes["chunk_size"] = chunk_size
    if chunk_overlap is not None and chunk_overlap != settings.chunk_overlap:
        changes["chunk_overlap"] = chunk_overlap
    if top_k is not None and top_k != settings.top_k:
        changes["top_k"] = top_k
    return replace(settings, **changes) if changes else settings

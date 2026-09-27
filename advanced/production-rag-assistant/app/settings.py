"""Typed application settings, loaded from the environment.

Uses a Pydantic model so the whole configuration surface is validated and
documented in one place. No pydantic-settings dependency is required.
"""
from __future__ import annotations

import os

from pydantic import BaseModel, Field

try:  # optional dependency, safe to omit
    from dotenv import load_dotenv

    load_dotenv()
    # also look in the project root (handles running from a subdirectory)
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


class Settings(BaseModel):
    """Runtime configuration. All fields are overridable via environment variables."""

    # --- chunking ---
    chunk_size: int = Field(default=160, ge=40, le=1000)
    chunk_overlap: int = Field(default=40, ge=0, le=400)

    # --- retrieval ---
    top_k: int = Field(default=4, ge=1, le=20)
    candidate_k: int = Field(default=12, ge=1, le=100)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    min_lexical_overlap: int = Field(default=1, ge=0, le=20)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- groundedness ---
    min_groundedness: float = Field(default=0.55, ge=0.0, le=1.0)

    # --- embeddings ---
    embedding_provider: str = "hashing"      # hashing | sentence-transformers | openai
    embedding_dim: int = Field(default=512, ge=32, le=4096)
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- generation ---
    llm_provider: str = "mock"               # omniroute | openrouter | openai | zai | mock
    llm_model: str = "offline-extractive-v1"
    llm_api_key: str | None = None
    llm_base_url: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 900
    llm_timeout: float = 30.0
    llm_max_retries: int = 2

    # --- ingestion ---
    max_file_mb: float = 12.0
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".markdown", ".pdf")

    @property
    def offline_llm(self) -> bool:
        return self.llm_provider == "mock"

    @classmethod
    def from_env(cls) -> "Settings":
        explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        if explicit in ("omniroute", "openrouter", "openai", "zai", "mock"):
            provider = explicit
        elif os.getenv("OMNIROUTE_API_KEY"):
            provider = "omniroute"
        elif os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ZAI_API_KEY"):
            provider = "zai"
        else:
            provider = "mock"

        if provider == "omniroute":
            api_key = os.getenv("OMNIROUTE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OMNIROUTE_BASE_URL") or "http://localhost:20128/v1"
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        elif provider == "zai":
            api_key = os.getenv("ZAI_API_KEY")
            base_url = os.getenv("ZAI_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
        else:
            api_key, base_url = None, ""

        default_model = {
            "omniroute": "auto/best-fast",
            "openrouter": "openai/gpt-4o-mini",
            "openai": "gpt-4o-mini",
            "zai": "glm-4-flash",
            "mock": "offline-extractive-v1",
        }[provider]

        return cls(
            chunk_size=_env_int("CHUNK_SIZE", 160),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 40),
            top_k=_env_int("TOP_K", 4),
            candidate_k=_env_int("CANDIDATE_K", 12),
            min_score=_env_float("MIN_SCORE", 0.05),
            min_lexical_overlap=_env_int("MIN_LEXICAL_OVERLAP", 1),
            mmr_lambda=_env_float("MMR_LAMBDA", 0.5),
            min_groundedness=_env_float("MIN_GROUNDEDNESS", 0.55),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hashing").strip().lower(),
            embedding_dim=_env_int("EMBEDDING_DIM", 512),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL") or default_model,
            llm_api_key=api_key,
            llm_base_url=base_url,
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 900),
            llm_timeout=_env_float("LLM_TIMEOUT", 30.0),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
            max_file_mb=_env_float("MAX_FILE_MB", 12.0),
        )

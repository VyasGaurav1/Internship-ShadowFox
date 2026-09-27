"""Embedding providers: offline hashing (default), sentence-transformers, openai."""
from __future__ import annotations

import hashlib
import math
from collections import Counter

from .text_utils import content_words


def _stable_hash(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


def _l2(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return vector if norm == 0 else [v / norm for v in vector]


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = [0.0] * self.dim
            for token, count in Counter(content_words(text)).items():
                vector[_stable_hash(token, self.dim)] += 1.0 + math.log(count)
            out.append(_l2(vector))
        return out


class SentenceTransformerEmbedder:  # pragma: no cover - heavy optional dep
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]


class OpenAIEmbedder:  # pragma: no cover - network
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-3-small") -> None:
        import requests

        self._requests = requests
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=30,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


def build_embedder(settings) -> object:
    provider = (settings.embedding_provider or "hashing").lower()
    if provider in ("sentence-transformers", "st", "sbert"):  # pragma: no cover
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception:
            return HashingEmbedder(settings.embedding_dim)
    if provider == "openai" and settings.llm_api_key:  # pragma: no cover
        try:
            return OpenAIEmbedder(settings.llm_api_key, settings.llm_base_url)
        except Exception:
            return HashingEmbedder(settings.embedding_dim)
    return HashingEmbedder(settings.embedding_dim)

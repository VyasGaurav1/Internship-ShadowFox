"""Embedding providers.

Three providers are available behind one interface:

* ``hashing``              - offline, deterministic, dependency-free (default).
                             Feature hashing of content words with sub-linear
                             term weighting and L2 normalisation. Good enough for
                             keyword/lexical retrieval and fully reproducible.
* ``sentence-transformers`` - real semantic embeddings (optional dependency).
* ``openai``               - hosted embeddings (optional, needs a key).

The offline hashing embedder means the whole ingestion -> embedding -> retrieval
pipeline is testable with no network and no heavy install.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

from text_utils import content_words


def _stable_hash(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


def _l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class HashingEmbedder:
    """Deterministic offline embedder using the hashing trick."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts = Counter(content_words(text))
        for token, count in counts.items():
            weight = 1.0 + math.log(count)  # sub-linear term frequency
            index = _stable_hash(token, self.dim)
            vector[index] += weight
        return _l2_normalise(vector)


class _LazyEmbedder:
    """Base helper for optional providers that import lazily."""

    name = "lazy"

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - overridden
        raise NotImplementedError


class SentenceTransformerEmbedder(_LazyEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - needs heavy dep
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


class OpenAIEmbedder(_LazyEmbedder):
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-3-small") -> None:
        import requests  # noqa: PLC0415

        self._requests = requests
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        response = self._requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=30,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


def build_embedder(settings) -> object:
    """Construct the embedder named by the settings, with safe fallback."""

    if settings.embedding_provider == "sentence-transformers":  # pragma: no cover
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception:
            return HashingEmbedder(settings.embedding_dim)
    if settings.embedding_provider == "openai" and settings.llm_api_key:  # pragma: no cover
        return OpenAIEmbedder(settings.llm_api_key, settings.llm_base_url)
    return HashingEmbedder(settings.embedding_dim)

import math

from config import load_settings
from embeddings import HashingEmbedder
from vector_store import VectorStore, cosine


def test_hashing_embedder_is_deterministic():
    emb = HashingEmbedder(128)
    assert emb.embed(["hello world"]) == emb.embed(["hello world"])


def test_hashing_embedder_is_normalised():
    emb = HashingEmbedder(64)
    vec = emb.embed(["some words to embed"])[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0


def test_cosine_bounds():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([], []) == 0.0


def test_vector_store_ranks_by_similarity():
    from chunking import Chunk

    store = VectorStore()
    chunks = [
        Chunk("d", "d", 0, "apples and oranges", 0, 3, 0, 10),
        Chunk("d", "d", 1, "quantum physics", 3, 6, 11, 20),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    hits = store.search([1.0, 0.0], top_k=2)
    assert hits[0].chunk.text == "apples and oranges"
    assert hits[0].score > hits[1].score


def test_build_embedder_defaults_to_hashing():
    from embeddings import build_embedder

    settings = load_settings()
    embedder = build_embedder(settings)
    assert getattr(embedder, "name", "").startswith("hashing")

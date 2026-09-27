from app.pipeline.chunking import chunk_document, chunk_text
from app.pipeline.guardrails import grade_context, groundedness
from app.pipeline.preprocess import preprocess
from app.pipeline.reranker import rerank
from app.pipeline.vector_store import ScoredChunk, build_store
from app.pipeline.chunking import Chunk


def test_preprocess_removes_repeated_headers():
    text = "\n".join(["Report Header"] * 5 + ["Body line about Zorblax calibration."])
    cleaned, stats = preprocess(text)
    assert "Report Header" not in cleaned
    assert stats.removed_repeated_lines >= 4


def test_preprocess_joins_hyphen_breaks():
    cleaned, stats = preprocess("recali-\nbration is needed")
    assert "recalibration" in cleaned
    assert stats.joined_hyphen_breaks >= 1


def test_chunk_offsets_map_to_source():
    text = "Sentence one about alpha. Sentence two about beta. Sentence three about gamma. Sentence four."
    spans = chunk_text(text, chunk_size=8, chunk_overlap=2)
    assert spans
    for start, end in spans:
        assert text[start:end].strip()


def test_chunk_size_budget_respected():
    text = " ".join(f"word{i}." for i in range(400))
    spans = chunk_text(text, chunk_size=60, chunk_overlap=10)
    assert len(spans) >= 4
    for start, end in spans:
        # allow one sentence of overshoot
        assert len(text[start:end].split()) <= 90


def test_grade_context_flags_weak_match():
    chunk = Chunk("d", "d.txt", 0, "completely unrelated material about birds", 0, 6, 0, 40)
    ok, overlap, score = grade_context("What is the Zorblax threshold?", [ScoredChunk(chunk, 0.4)])
    assert ok is False
    assert overlap == 0


def test_groundedness_measures_overlap():
    context = ["Zorblax reading above 40 indicates recalibration"]
    assert groundedness("Zorblax reading above 40 indicates recalibration", context) == 1.0
    assert groundedness("totally unrelated sentence here", context) < 0.5


def test_rerank_returns_top_k():
    chunks = [
        ScoredChunk(Chunk("d", "d.txt", 0, "Zorblax calibration threshold reading", 0, 5, 0, 40), 0.9),
        ScoredChunk(Chunk("d", "d.txt", 1, "unrelated birds and trees content", 5, 10, 41, 75), 0.8),
        ScoredChunk(Chunk("d", "d.txt", 2, "Zorblax recalibration procedure details", 10, 15, 76, 115), 0.7),
    ]
    ranked = rerank("What is the Zorblax calibration threshold?", chunks, top_k=2, mmr_lambda=0.5)
    assert len(ranked) == 2
    assert all(rerank_item.rerank_score >= 0 for rerank_item in ranked)


def test_build_store_returns_a_store():
    store = build_store(64)
    chunk = Chunk("d", "d.txt", 0, "alpha beta", 0, 2, 0, 10)
    store.add([chunk], [[1.0] + [0.0] * 63])
    assert len(store) == 1
    assert store.search([1.0] + [0.0] * 63, top_k=1)[0].chunk.text == "alpha beta"


def test_langgraph_mode_active():
    from app.service import RAGService

    service = RAGService()
    assert service.pipeline().mode == "langgraph"


def test_rewrite_query_strips_filler():
    from app.service import rewrite_query

    assert "what is" not in rewrite_query("What is the Zorblax threshold?").lower()
    assert "Zorblax" in rewrite_query("What is the Zorblax threshold?")

"""Offline self-check for the Document Q&A Assistant.

No API key, no network, no heavy dependencies required.

    python selfcheck.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Deterministic offline run regardless of ambient API keys.
if not os.environ.get("SELFCHECK_LIVE"):
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["EMBEDDING_PROVIDER"] = "hashing"

import math
import tempfile
import traceback

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name):
    def decorator(fn):
        def runner():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            else:
                _results.append((name, True, ""))

        return runner

    return decorator


DOC_TRUTH = (
    "Zorblax is a fictional calibration unit used only in the Alpha-9 experiment. "
    "The Alpha-9 experiment measures thermal drift in the Orbital Loom device. "
    "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration. "
    "Calibration is performed by the Qarn tool every 500 cycles."
)

DOC_OTHER = (
    "Photosynthesis converts light energy into chemical energy in plants. "
    "Chlorophyll absorbs blue and red light. The Calvin cycle fixes carbon dioxide into glucose."
)


def _make_index():
    from config import load_settings
    from ingest import Document
    from qa import build_index

    docs = [
        Document(doc_id="alpha9.txt", name="alpha9.txt", text=DOC_TRUTH, source_type="text"),
        Document(doc_id="bio.md", name="bio.md", text=DOC_OTHER, source_type="markdown"),
    ]
    return build_index(docs, load_settings())


@check("chunking respects size and creates overlap")
def _t1() -> None:
    from chunking import chunk_words

    text = " ".join(f"word{i}" for i in range(500))
    spans = chunk_words(text, chunk_size=100, overlap=20)
    assert len(spans) >= 5, f"expected several chunks, got {len(spans)}"
    for sw, ew, _, _ in spans:
        assert ew - sw <= 100, "chunk larger than chunk_size"
    for (a_sw, a_ew, _, _), (b_sw, _, _, _) in zip(spans, spans[1:]):
        assert b_sw < a_ew, "no overlap between consecutive chunks"


@check("hashing embedder is deterministic and L2-normalised")
def _t2() -> None:
    from embeddings import HashingEmbedder

    emb = HashingEmbedder(256)
    v1 = emb.embed(["the quick brown fox"])[0]
    v2 = emb.embed(["the quick brown fox"])[0]
    v3 = emb.embed(["a completely different sentence"])[0]
    assert v1 == v2, "embedding is not deterministic"
    assert v1 != v3
    assert abs(math.sqrt(sum(x * x for x in v1)) - 1.0) < 1e-6, "vector is not L2 normalised"


@check("index builds with correct statistics")
def _t3() -> None:
    index = _make_index()
    stats = index.stats
    assert stats["documents"] == 2, stats
    assert stats["chunks"] >= 2, stats
    assert stats["embedding_provider"].startswith("hashing"), stats


@check("retrieval ranks the correct chunk first")
def _t4() -> None:
    from config import load_settings
    from embeddings import HashingEmbedder

    index = _make_index()
    settings = load_settings()
    emb = HashingEmbedder(settings.embedding_dim)
    q = emb.embed(["What does a Zorblax reading above 40 mean?"])[0]
    hits = index.store.search(q, top_k=3)
    assert hits, "no retrieval results"
    assert "Zorblax" in hits[0].chunk.text, f"wrong top chunk: {hits[0].chunk.text[:80]!r}"


@check("grounded answer uses document content and cites sources")
def _t5() -> None:
    from qa import ask

    index = _make_index()
    result = ask(index, "What does a Zorblax reading above 40 indicate?")
    assert not result.refused, "should have answered"
    assert "Zorblax" in result.answer, f"answer not grounded: {result.answer!r}"
    assert result.sources, "no sources returned"
    assert result.groundedness >= 0.9, f"low groundedness: {result.groundedness}"


@check("unrelated question is refused (no hallucination)")
def _t6() -> None:
    from qa import ask

    index = _make_index()
    result = ask(index, "What is the capital city of Finland and its population?")
    assert result.refused, f"should refuse, answered: {result.answer!r}"
    assert not result.sources


@check("empty question is rejected")
def _t7() -> None:
    from qa import ask
    from validators import ValidationError

    index = _make_index()
    try:
        ask(index, "   ")
    except ValidationError:
        return
    raise AssertionError("empty question was not rejected")


@check("unsupported file type is rejected")
def _t8() -> None:
    from ingest import IngestError, load_document

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "notes.docx")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("not really a docx")
        try:
            load_document(path)
        except IngestError as exc:
            assert exc.kind == "unsupported", exc.kind
            return
    raise AssertionError("unsupported file was accepted")


@check("text file ingestion works end to end")
def _t9() -> None:
    from ingest import load_document

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "alpha9.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(DOC_TRUTH)
        doc = load_document(path)
        assert doc.source_type == "text"
        assert "Zorblax" in doc.text


@check("PDF ingestion works (pypdf + sample_docs/sample.pdf)")
def _t10() -> None:
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs", "sample.pdf")
    if not os.path.exists(pdf_path):
        raise AssertionError("sample_docs/sample.pdf is missing")
    try:
        import pypdf  # noqa: F401
    except Exception:
        raise AssertionError("pypdf is not installed (install with `pip install pypdf`)")

    from ingest import load_document

    doc = load_document(pdf_path)
    assert doc.source_type == "pdf"
    assert "Zorblax" in doc.text, "PDF text did not extract as expected"


@check("chunk citations are stable identifiers")
def _t11() -> None:
    index = _make_index()
    citations = {c.citation for c in index.chunks}
    assert all("#chunk-" in c for c in citations), citations
    assert len(citations) == len(index.chunks), "duplicate citations"


def main() -> int:
    for name, value in sorted(globals().items()):
        if name.startswith("_t") and callable(value):
            try:
                value()
            except Exception:  # pragma: no cover
                _results.append((name, False, traceback.format_exc()))

    passed = sum(1 for _, ok, _ in _results if ok)
    print("=" * 64)
    print("Document Q&A Assistant - offline self-check")
    print("=" * 64)
    for name, ok, detail in _results:
        print(f"[{PASS if ok else FAIL}] {name}")
        if not ok:
            print(f"       {detail}")
    print("-" * 64)
    print(f"{passed}/{len(_results)} checks passed")
    return 0 if _results and passed == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

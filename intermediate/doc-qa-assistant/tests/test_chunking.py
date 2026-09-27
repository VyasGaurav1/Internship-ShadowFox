from chunking import chunk_document, chunk_words
from ingest import Document


def test_chunk_words_respects_size():
    text = " ".join(f"w{i}" for i in range(300))
    spans = chunk_words(text, chunk_size=80, overlap=10)
    assert spans
    for sw, ew, _, _ in spans:
        assert ew - sw <= 80


def test_chunk_words_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    spans = chunk_words(text, chunk_size=80, overlap=20)
    for (a_sw, a_ew, _, _), (b_sw, _, _, _) in zip(spans, spans[1:]):
        assert b_sw < a_ew


def test_chunk_words_empty_text():
    assert chunk_words("", chunk_size=50, overlap=5) == []


def test_chunk_document_offsets_slice_original_text():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    doc = Document(doc_id="d", name="d.txt", text=text, source_type="text")
    chunks = chunk_document(doc, chunk_size=3, overlap=1)
    assert chunks
    for chunk in chunks:
        assert text[chunk.start_char:chunk.end_char].strip() == chunk.text


def test_chunk_citation_format():
    doc = Document(doc_id="d", name="notes.md", text="one two three four five six", source_type="markdown")
    chunks = chunk_document(doc, chunk_size=3, overlap=0)
    assert chunks[0].citation == "notes.md#chunk-0"

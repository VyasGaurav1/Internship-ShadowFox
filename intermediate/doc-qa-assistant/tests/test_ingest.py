import pytest

from ingest import IngestError, load_bytes, load_document


def test_load_text_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Zorblax is a calibration unit.", encoding="utf-8")
    doc = load_document(str(path))
    assert doc.source_type == "text"
    assert "Zorblax" in doc.text


def test_load_markdown_file(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nContent here.", encoding="utf-8")
    doc = load_document(str(path))
    assert doc.source_type == "markdown"


def test_unsupported_extension(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_text("not a docx", encoding="utf-8")
    with pytest.raises(IngestError) as exc:
        load_document(str(path))
    assert exc.value.kind == "unsupported"


def test_missing_file():
    with pytest.raises(IngestError) as exc:
        load_document("does-not-exist.txt")
    assert exc.value.kind == "missing"


def test_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(IngestError) as exc:
        load_document(str(path))
    assert exc.value.kind == "empty"


def test_load_bytes_text():
    doc = load_bytes("a.txt", b"Zorblax reading is high.")
    assert doc.source_type == "text"
    assert "Zorblax" in doc.text


def test_load_bytes_unsupported():
    with pytest.raises(IngestError):
        load_bytes("a.xyz", b"data")

import pytest

from config import load_settings
from ingest import Document
from qa import ask, build_index
from validators import ValidationError, validate_question

TRUTH = (
    "Zorblax is a fictional calibration unit used only in the Alpha-9 experiment. "
    "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration. "
    "Calibration is performed by the Qarn tool every 500 cycles."
)
OTHER = "Photosynthesis converts light energy into chemical energy in plants."


@pytest.fixture()
def index():
    docs = [
        Document("a.txt", "a.txt", TRUTH, "text"),
        Document("b.md", "b.md", OTHER, "markdown"),
    ]
    return build_index(docs, load_settings())


def test_answer_is_grounded(index):
    result = ask(index, "What does a Zorblax reading above 40 indicate?")
    assert not result.refused
    assert "Zorblax" in result.answer
    assert result.groundedness >= 0.9
    assert result.sources


def test_unrelated_question_is_refused(index):
    result = ask(index, "Who won the 1998 world cup final?")
    assert result.refused
    assert result.sources == []
    assert result.groundedness == 0.0


def test_sources_carry_citations(index):
    result = ask(index, "What tool performs calibration?")
    assert result.sources
    assert all("#chunk-" in source.citation for source in result.sources)


def test_empty_question_rejected(index):
    with pytest.raises(ValidationError):
        ask(index, "")


def test_validate_question_bounds():
    with pytest.raises(ValidationError):
        validate_question("a")
    with pytest.raises(ValidationError):
        validate_question("x" * 2000)
    assert validate_question("  what is this?  ") == "what is this?"


def test_index_stats(index):
    stats = index.stats
    assert stats["documents"] == 2
    assert stats["chunks"] >= 2

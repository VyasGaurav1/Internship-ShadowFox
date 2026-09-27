import pytest

from core import run_feature
from prompts import FEATURES, REQUIRED_SECTIONS

SAMPLE = (
    "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
    "Chlorophyll is the green pigment that absorbs light. "
    "The Calvin cycle uses ATP and NADPH to fix carbon dioxide into glucose. "
    "The light reactions occur in the thylakoid membrane and produce ATP and NADPH."
)


@pytest.mark.parametrize("feature", FEATURES)
def test_feature_returns_required_sections(feature):
    result = run_feature(feature, SAMPLE, {"topic": "photosynthesis", "num_questions": 3})
    assert result["ok"], result.get("error")
    assert result["offline"] is True
    for section in REQUIRED_SECTIONS[feature]:
        assert section in result["output"]


def test_summarize_has_bullets():
    result = run_feature("summarize", SAMPLE)
    bullets = [line for line in result["output"].splitlines() if line.startswith("- ")]
    assert 3 <= len(bullets) <= 6


def test_quiz_has_numbered_questions_and_answers():
    result = run_feature("quiz", SAMPLE, {"num_questions": 4})
    assert "## Questions" in result["output"]
    assert "## Answer key" in result["output"]
    assert "1." in result["output"]


def test_improve_removes_hedge_words():
    weak = "I think photosynthesis is kind of important because it basically makes food for plants and stuff."
    result = run_feature("improve", weak, {"topic": "photosynthesis"})
    out = result["output"].lower()
    assert "i think" not in out
    assert "kind of" not in out
    assert "## What changed" in result["output"]


def test_explain_has_four_sections():
    result = run_feature("explain", SAMPLE, {"topic": "photosynthesis"})
    for section in ("## In one line", "## In simple terms", "## Deeper detail", "## Common mistake to avoid"):
        assert section in result["output"]

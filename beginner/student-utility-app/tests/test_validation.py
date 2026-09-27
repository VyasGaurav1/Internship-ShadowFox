import pytest

from validators import ValidationError, validate_content, validate_feature


def test_empty_input_rejected():
    with pytest.raises(ValidationError):
        validate_content("")


def test_whitespace_only_rejected():
    with pytest.raises(ValidationError):
        validate_content("     \n\t  ")


def test_none_rejected():
    with pytest.raises(ValidationError):
        validate_content(None)


def test_non_text_rejected():
    with pytest.raises(ValidationError):
        validate_content(12345)  # type: ignore[arg-type]


def test_too_short_rejected():
    with pytest.raises(ValidationError):
        validate_content("short")


def test_too_long_rejected():
    with pytest.raises(ValidationError):
        validate_content("x" * 20000)


def test_valid_input_is_stripped_and_returned():
    assert validate_content("   valid study content here   ") == "valid study content here"


def test_validate_feature_accepts_known():
    assert validate_feature("summarize") == "summarize"
    assert validate_feature("QUIZ") == "quiz"


def test_validate_feature_rejects_unknown():
    with pytest.raises(ValidationError):
        validate_feature("teleport")

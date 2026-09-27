import pytest

from prompts import FEATURES, REQUIRED_SECTIONS, SYSTEM_PROMPTS, build_messages


def test_build_messages_unknown_feature():
    with pytest.raises(ValueError):
        build_messages("teleport", "some content that is long enough")


@pytest.mark.parametrize("feature", FEATURES)
def test_system_prompt_encodes_output_contract(feature):
    for section in REQUIRED_SECTIONS[feature]:
        assert section in SYSTEM_PROMPTS[feature]


def test_build_messages_includes_content():
    messages = build_messages("summarize", "UniqueContentMarker12345")
    assert "UniqueContentMarker12345" in messages["user"]
    assert messages["system"]


def test_improve_prompt_mentions_topic():
    messages = build_messages("improve", "some answer", {"topic": "photosynthesis"})
    assert "photosynthesis" in messages["user"]

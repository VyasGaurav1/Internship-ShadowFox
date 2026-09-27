import pytest

from config import Settings
from core import run_feature
from llm_client import LLMClient, LLMError


class _FailingClient:
    def __init__(self, kind="rate_limit"):
        self.kind = kind

    def complete(self, feature, content, options=None):
        raise LLMError("The AI service is busy (rate limited). Please retry in a moment.", kind=self.kind)


class _BoomClient:
    def complete(self, feature, content, options=None):
        raise RuntimeError("boom")


GOOD = "This is a sufficiently long piece of study content for the test."


def test_llm_error_becomes_structured_result():
    result = run_feature("summarize", GOOD, client=_FailingClient())
    assert result["ok"] is False
    assert result["error_kind"] == "rate_limit"
    assert "retry" in result["error"].lower()


def test_unexpected_error_is_contained():
    result = run_feature("summarize", GOOD, client=_BoomClient())
    assert result["ok"] is False
    assert result["error_kind"] == "unexpected"


def test_validation_error_before_provider_call():
    result = run_feature("summarize", "")
    assert result["ok"] is False
    assert result["error_kind"] == "validation"


def test_unknown_feature_rejected():
    result = run_feature("nope", GOOD)
    assert result["ok"] is False
    assert result["error_kind"] == "validation"


def test_missing_api_key_raises_auth_error():
    settings = Settings(
        provider="openai",
        model="gpt-4o-mini",
        credential=None,
        base_url="https://api.openai.com/v1",
        timeout=5,
        max_retries=0,
        temperature=0.3,
        max_tokens=128,
        min_input_chars=15,
        max_input_chars=12000,
    )
    client = LLMClient(settings)
    with pytest.raises(LLMError) as exc:
        client.complete("summarize", GOOD)
    assert exc.value.kind == "auth"

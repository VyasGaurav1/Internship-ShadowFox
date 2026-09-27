"""Offline self-check for the Student Utility App.

Runs without any API key, network, or third-party package beyond the standard
library, and exits non-zero if any check fails.

    python selfcheck.py
"""
from __future__ import annotations

import os
import sys

# Some bundled Python builds (embeddable/zip based) do not put the script
# directory on sys.path, so make the project root importable explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# This self-check is the *offline* verification path: pin the deterministic
# mock provider so results are reproducible even on a machine that happens to
# have a real API key in its environment. (Set SELFCHECK_LIVE=1 to skip this.)
if not os.environ.get("SELFCHECK_LIVE"):
    os.environ["LLM_PROVIDER"] = "mock"

import traceback

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        def runner():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                detail = f"{type(exc).__name__}: {exc}"
                _results.append((name, False, detail))
            else:
                _results.append((name, True, ""))

        runner.__name__ = fn.__name__
        return runner

    return decorator


@check("validation rejects empty input")
def _t1() -> None:
    from validators import ValidationError, validate_content

    try:
        validate_content("   ")
    except ValidationError:
        return
    raise AssertionError("empty input was not rejected")


@check("validation rejects too-short input")
def _t2() -> None:
    from validators import ValidationError, validate_content

    try:
        validate_content("hi")
    except ValidationError:
        return
    raise AssertionError("too-short input was not rejected")


@check("validation accepts good input and strips it")
def _t3() -> None:
    from validators import validate_content

    assert validate_content("  This is a valid note with enough length.  ") == "This is a valid note with enough length."


@check("all four features return required sections (offline)")
def _t4() -> None:
    from core import run_feature
    from prompts import FEATURES, REQUIRED_SECTIONS

    sample = (
        "Photosynthesis is the process by which plants convert light into chemical energy. "
        "Chlorophyll absorbs light. The Calvin cycle fixes carbon dioxide into glucose. "
        "The light reactions occur in the thylakoid membrane and produce ATP."
    )
    for feature in FEATURES:
        result = run_feature(feature, sample, {"topic": "photosynthesis", "num_questions": 3})
        assert result["ok"], f"{feature}: {result.get('error')}"
        assert result["offline"] is True, f"{feature}: expected offline provider"
        for section in REQUIRED_SECTIONS[feature]:
            assert section in result["output"], f"{feature}: missing section {section!r}"


@check("quiz output has questions and an answer key")
def _t5() -> None:
    from core import run_feature

    sample = (
        "Mitosis is the process of cell division producing two identical daughter cells. "
        "The cell cycle includes interphase and the mitotic phase. Chromosomes are copied during interphase."
    )
    result = run_feature("quiz", sample, {"num_questions": 3})
    assert result["ok"]
    assert "## Questions" in result["output"] and "## Answer key" in result["output"]
    assert "1." in result["output"]


@check("failed API responses become friendly errors")
def _t6() -> None:
    from core import run_feature
    from llm_client import LLMError

    class FailingClient:
        def complete(self, feature, content, options=None):
            raise LLMError("The AI service is busy (rate limited). Please retry in a moment.", kind="rate_limit")

    result = run_feature("summarize", "This is a sufficiently long piece of study content.", client=FailingClient())
    assert result["ok"] is False
    assert result["error_kind"] == "rate_limit"
    assert "retry" in result["error"].lower()


@check("unexpected errors never crash the app")
def _t7() -> None:
    from core import run_feature

    class BoomClient:
        def complete(self, feature, content, options=None):
            raise RuntimeError("boom")

    result = run_feature("summarize", "This is a sufficiently long piece of study content.", client=BoomClient())
    assert result["ok"] is False and result["error_kind"] == "unexpected"


@check("unknown feature is rejected")
def _t8() -> None:
    from core import run_feature

    result = run_feature("teleport", "This is a sufficiently long piece of study content.")
    assert result["ok"] is False and result["error_kind"] == "validation"


@check("default provider is offline when no key is set")
def _t9() -> None:
    import os

    for var in ("LLM_PROVIDER", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ZAI_API_KEY"):
        os.environ.pop(var, None)
    from config import load_settings

    assert load_settings().provider == "mock"


@check("prompt templates define a system rule and output contract")
def _t10() -> None:
    from prompts import FEATURES, REQUIRED_SECTIONS, SYSTEM_PROMPTS, build_messages

    for feature in FEATURES:
        assert "StudyMate" in SYSTEM_PROMPTS[feature]
        messages = build_messages(feature, "Some study content long enough to pass.", {"topic": "x"})
        assert messages["system"] and messages["user"]
        for section in REQUIRED_SECTIONS[feature]:
            assert section in messages["system"]


def main() -> int:
    for name, value in sorted(globals().items()):
        if name.startswith("_t") and callable(value):
            try:
                value()
            except Exception:  # pragma: no cover
                _results.append((name, False, traceback.format_exc()))

    passed = sum(1 for _, ok, _ in _results if ok)
    print("=" * 64)
    print("Student Utility App - offline self-check")
    print("=" * 64)
    for name, ok, detail in _results:
        print(f"[{PASS if ok else FAIL}] {name}")
        if not ok:
            print(f"       {detail}")
    print("-" * 64)
    print(f"{passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) and _results else 1


if __name__ == "__main__":
    raise SystemExit(main())

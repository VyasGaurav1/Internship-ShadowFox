"""Streamlit UI for the Student Utility App.

Run with:

    streamlit run app.py

The UI is a thin layer over :func:`core.run_feature`: it collects input,
performs the same validation the core uses, shows errors clearly, and renders
the sectioned Markdown the model (or the offline engine) returns.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE_NOTES = """Photosynthesis is the process by which green plants convert light energy into chemical energy.
Chlorophyll is the green pigment that absorbs light, mainly in the blue and red parts of the spectrum.
The process happens in two stages: the light-dependent reactions and the Calvin cycle.
The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH.
The Calvin cycle occurs in the stroma and uses ATP and NADPH to fix carbon dioxide into glucose.
"""


def main() -> None:
    import streamlit as st

    from config import load_settings
    from core import run_feature
    from prompts import FEATURES, FEATURE_LABELS

    st.set_page_config(page_title="StudyMate - AI Student Utility", page_icon="📚", layout="centered")

    settings = load_settings()

    st.title("📚 StudyMate")
    st.caption("An AI-powered study utility: summarise notes, build quizzes, improve answers, explain concepts.")

    if settings.is_offline:
        st.info(
            "Running in **offline mode** (no API key detected). The app uses a built-in deterministic "
            "engine so every feature is fully usable. Set `LLM_PROVIDER` + an API key in `.env` for live AI output."
        )
    else:
        st.success(f"Connected provider: **{settings.provider}** · model `{settings.model}`")

    with st.sidebar:
        st.header("Settings")
        feature = st.selectbox(
            "Utility feature",
            options=list(FEATURES),
            format_func=lambda key: FEATURE_LABELS[key],
        )
        topic = ""
        num_questions = 5
        if feature in ("improve", "explain"):
            topic = st.text_input("Topic / question (optional)", placeholder="e.g. photosynthesis")
        if feature == "quiz":
            num_questions = st.slider("Number of questions", min_value=2, max_value=8, value=5)
        st.divider()
        if st.button("Load sample notes"):
            st.session_state["content"] = SAMPLE_NOTES
        st.caption(
            f"Input limits: {settings.min_input_chars}-{settings.max_input_chars} characters. "
            "Empty or invalid input is rejected before any request is made."
        )

    content = st.text_area(
        "Your content",
        key="content",
        height=240,
        placeholder="Paste your study notes, a question, or a draft answer here...",
    )

    char_count = len(content or "")
    st.caption(f"{char_count} characters")

    generate = st.button("Generate", type="primary", use_container_width=True)

    if generate:
        with st.spinner("Working on it..."):
            result = run_feature(
                feature,
                content,
                {"topic": topic, "num_questions": num_questions},
                min_chars=settings.min_input_chars,
                max_chars=settings.max_input_chars,
            )

        if not result["ok"]:
            st.error(result["error"])
        else:
            mode = "offline mode" if result["offline"] else f"{result['provider']} · {result['model']}"
            st.caption(f"Feature: {FEATURE_LABELS[result['feature']]} · Source: {mode} · {result['elapsed_ms']} ms")
            st.markdown(result["output"])
            st.download_button(
                "Download result (.md)",
                data=result["output"],
                file_name=f"{result['feature']}_output.md",
                mime="text/markdown",
            )

    with st.expander("How it works"):
        st.markdown(
            "1. **Validate** - empty, too-short, too-long or non-text input is rejected with a clear message.\n"
            "2. **Prompt** - each feature has a dedicated system instruction and a user template "
            "(see `prompts.py`) that fixes the output structure.\n"
            "3. **Generate** - the request goes to the configured provider, or to the offline engine if no key is set.\n"
            "4. **Handle errors** - missing keys, timeouts, rate limits and bad responses become friendly messages.\n"
            "5. **Display** - the sectioned Markdown result is rendered and can be downloaded."
        )


if __name__ == "__main__":
    main()

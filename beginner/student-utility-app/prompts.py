"""Prompt design for the Student Utility App.

Each utility feature has an explicit *system* instruction (role, constraints,
output contract) and a *user* template (the task + the learner's content).
Keeping the templates in one module makes the "prompt design thinking" that the
task asks for explicit, reviewable and testable.

Design principles applied here:

1. **Role priming** - the model is told it is a study assistant, not a chatbot.
2. **Explicit output contract** - every feature states the exact structure the
   model must return (headings, numbered lists), which makes the UI predictable.
3. **Grounding / no-invention guard** - the model is told to stay within the
   supplied notes and to say "not covered" instead of inventing facts.
4. **Tone + audience control** - student-friendly, concise, exam oriented.
"""
from __future__ import annotations

from typing import Dict

FEATURES = ("summarize", "quiz", "improve", "explain")

FEATURE_LABELS: Dict[str, str] = {
    "summarize": "Summarise notes",
    "quiz": "Generate a quiz",
    "improve": "Improve a weak answer",
    "explain": "Explain a concept",
}

_BASE_RULES = (
    "You are StudyMate, a focused study assistant for students. "
    "Work only from the material the student provides. "
    "Never invent facts, citations or examples that are not supported by the input. "
    "If something is not covered by the material, say 'not covered in the provided material'. "
    "Keep the tone clear, exam-oriented and free of filler."
)

SYSTEM_PROMPTS: Dict[str, str] = {
    "summarize": (
        _BASE_RULES
        + " Your task is EXTRACTIVE NOTE SUMMARISATION. Preserve the author's meaning."
        "\n\nReturn exactly this structure:\n"
        "## Key points\n"
        "- 3 to 6 concrete bullet points\n"
        "## One-line takeaway\n"
        "A single sentence a student could memorise."
    ),
    "quiz": (
        _BASE_RULES
        + " Your task is QUIZ GENERATION for active recall."
        "\n\nReturn exactly this structure:\n"
        "## Questions\n"
        "1. <question>  (mix multiple-choice and short answer; 4 to 6 questions)\n"
        "## Answer key\n"
        "1. <answer with a one-line justification>"
    ),
    "improve": (
        _BASE_RULES
        + " Your task is ANSWER IMPROVEMENT. Rewrite the student's weak answer into a"
        " stronger one without changing the intended meaning or adding unsupported facts."
        "\n\nReturn exactly this structure:\n"
        "## Improved answer\n"
        "The rewritten answer.\n"
        "## What changed\n"
        "- 2 to 5 bullets naming the concrete improvements (structure, precision, hedging, evidence)."
    ),
    "explain": (
        _BASE_RULES
        + " Your task is CONCEPT EXPLANATION at three depths."
        "\n\nReturn exactly this structure:\n"
        "## In one line\n"
        "## In simple terms\n"
        "## Deeper detail\n"
        "## Common mistake to avoid"
    ),
}

USER_TEMPLATES: Dict[str, str] = {
    "summarize": "Summarise the following study notes.\n\n---\n{content}\n---",
    "quiz": (
        "Create an active-recall quiz from the following notes."
        "\nNumber of questions: {num_questions}"
        "\n\n---\n{content}\n---"
    ),
    "improve": (
        "The student is working on this topic/question:\n{topic}\n\n"
        "Here is their current answer, which needs to be stronger:\n"
        "---\n{content}\n---"
    ),
    "explain": (
        "Explain exactly this concept for a student: {topic}\n"
        "Use the supporting material below only as background.\n\n"
        "---\n{content}\n---"
    ),
}

REQUIRED_SECTIONS: Dict[str, tuple] = {
    "summarize": ("## Key points", "## One-line takeaway"),
    "quiz": ("## Questions", "## Answer key"),
    "improve": ("## Improved answer", "## What changed"),
    "explain": ("## In one line", "## In simple terms", "## Deeper detail", "## Common mistake to avoid"),
}


def build_messages(feature: str, content: str, options: dict | None = None) -> dict:
    """Return ``{"system": ..., "user": ...}`` messages for a feature call."""

    if feature not in FEATURES:
        raise ValueError(f"Unknown feature: {feature!r}. Expected one of {FEATURES}.")

    options = options or {}
    topic = (options.get("topic") or "").strip() or "the provided material"
    num_questions = int(options.get("num_questions", 5) or 5)

    user = USER_TEMPLATES[feature].format(
        content=content.strip(),
        topic=topic,
        num_questions=num_questions,
    )
    return {"system": SYSTEM_PROMPTS[feature], "user": user}

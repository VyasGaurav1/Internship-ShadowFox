"""Offline deterministic engine for the Student Utility App.

When no API key is configured the app falls back to this module instead of
crashing. The algorithms are deliberately simple, transparent NLP heuristics
(no network, no external model) so the *whole* workflow - input, prompt
construction, generation, formatted output - can be demonstrated and tested
without any external service.

Every function returns output in the exact same sectioned Markdown format the
real LLM prompts demand, so the UI and the self-check treat both providers
identically. Offline output is clearly labelled downstream ("offline mode").

What each heuristic actually does:

* summarise   - frequency-scored extractive summarisation
* quiz        - definition extraction + cloze deletion with distractors
* improve     - deterministic rewriting rules (hedge removal, precision,
                structure, closing summary line)
* explain     - definition spotting + key-term scaffolding, grounded in the
                provided material only (nothing is invented)
"""
from __future__ import annotations

import re
from collections import Counter

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "it",
    "its", "as", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "about", "into", "over", "after", "before", "so", "such", "not", "no",
    "can", "could", "should", "would", "may", "might", "will", "shall", "do",
    "does", "did", "have", "has", "had", "we", "you", "they", "he", "she",
    "i", "me", "my", "our", "your", "their", "them", "us", "also", "very",
    "just", "there", "here", "which", "who", "what", "when", "how", "why",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")
_DEFINITION = re.compile(
    r"^(?P<subject>[A-Z][\w\s\-']{2,60}?)\s+(?P<verb>is|are|was|were|refers to|means|describes)\s+(?P<predicate>.+)$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# text utilities
# --------------------------------------------------------------------------- #
def _sentences(text: str) -> list[str]:
    parts = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        parts.extend(s.strip() for s in _SENTENCE_SPLIT.split(block) if s.strip())
    return parts or ([text.strip()] if text.strip() else [])


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in _STOPWORDS and len(w) > 2]


def _key_terms(text: str, limit: int = 8) -> list[str]:
    """Most frequent content words, longest-first tie-break."""

    counts = Counter(_content_words(text))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return [w for w, _ in ranked[:limit]]


def _title_case(word: str) -> str:
    return " ".join(p.capitalize() for p in word.split("-"))


# --------------------------------------------------------------------------- #
# feature: summarise
# --------------------------------------------------------------------------- #
def summarize(content: str, options: dict | None = None) -> str:
    sentences = _sentences(content)
    if not sentences:
        return "## Key points\n- The provided material is empty.\n\n## One-line takeaway\nNo content was supplied."

    counts = Counter(_content_words(content))
    scored = []
    for index, sentence in enumerate(sentences):
        tokens = _content_words(sentence)
        if not tokens:
            score = 0.0
        else:
            score = sum(counts[t] for t in tokens) / (len(tokens) ** 0.5)
        if index == 0:  # lead sentences usually carry the thesis
            score *= 1.15
        scored.append((index, sentence, score))

    wanted = max(3, min(6, len(sentences) // 3 or 3))
    top = sorted(sorted(scored, key=lambda s: -s[2])[:wanted], key=lambda s: s[0])

    bullets = "\n".join(f"- {s.strip()}" for _, s, _ in top)
    takeaway = max(scored, key=lambda s: s[2])[1].strip()
    if len(takeaway) > 220:
        takeaway = takeaway[:217].rstrip() + "..."

    return (
        "## Key points\n"
        f"{bullets}\n\n"
        "## One-line takeaway\n"
        f"{takeaway}"
    )


# --------------------------------------------------------------------------- #
# feature: quiz
# --------------------------------------------------------------------------- #
def generate_quiz(content: str, options: dict | None = None) -> str:
    options = options or {}
    try:
        num_questions = max(2, min(int(options.get("num_questions", 5) or 5), 8))
    except (TypeError, ValueError):
        num_questions = 5

    sentences = _sentences(content)
    terms = _key_terms(content, limit=12)
    questions: list[tuple[str, str]] = []

    # 1) definition-style questions, straight from the notes
    for sentence in sentences:
        if len(questions) >= num_questions:
            break
        if sentence is None:
            continue
        match = _DEFINITION.match(sentence.strip())
        if match and len(match.group("predicate")) > 15:
            predicate = match.group("predicate").strip()
            # skip passive constructions ("Chromosomes are copied ...") - they do
            # not make a clean "what is X?" question
            if re.match(r"^\w+ed\b", predicate, re.IGNORECASE):
                continue
            subject = match.group("subject").strip().rstrip(",")
            verb = match.group("verb").lower()
            question_word = "What are" if verb in ("are", "were") else "What is"
            questions.append((f"{question_word} {subject}?", predicate))

    # 2) cloze questions over key terms
    used: set[str] = set()
    for sentence in sentences:
        if len(questions) >= num_questions:
            break
        if sentence is None:
            continue
        for term in terms:
            if term in used or term not in sentence.lower():
                continue
            if len(questions) >= num_questions:
                break
            used.add(term)
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            blanked = pattern.sub("_____", sentence, count=1)
            distractors = [t for t in terms if t != term and t not in used][:3]
            if distractors:
                options_text = " / ".join(_title_case(d) for d in ([term] + distractors))
                questions.append(
                    (f"Fill in the blank: {blanked}\n   (choose from: {options_text})", _title_case(term))
                )
            else:
                questions.append((f"Fill in the blank: {blanked}", _title_case(term)))
            break

    if not questions:
        questions.append(("Which topic do these notes cover?", ", ".join(terms[:3]) or "the provided material"))

    q_block = "\n".join(f"{i}. {q}" for i, (q, _) in enumerate(questions, start=1))
    a_block = "\n".join(f"{i}. {a} — taken directly from your notes." for i, (_, a) in enumerate(questions, start=1))

    return f"## Questions\n{q_block}\n\n## Answer key\n{a_block}"


# --------------------------------------------------------------------------- #
# feature: improve a weak answer
# --------------------------------------------------------------------------- #
# (regex, replacement, label) - the label is reported in the change log so the
# student can see *which kind* of weakness was fixed without echoing the phrase.
_HEDGE_RULES = [
    (r"\bI think maybe\b", "", "hedging opener"),
    (r"\bI think\b", "", "hedging opener"),
    (r"\bI guess\b", "", "hedging opener"),
    (r"\bmaybe\b", "likely", "hedging adverb"),
    (r"\bkind of\b", "", "vague qualifier"),
    (r"\bsort of\b", "", "vague qualifier"),
    (r"\bbasically\b", "", "filler adverb"),
    (r"\bvery\b", "", "weak intensifier"),
    (r"\breally\b", "", "weak intensifier"),
    (r"\ba lot of\b", "many", "vague quantifier"),
    (r"\bstuff\b", "specific factors", "vague noun"),
    (r"\bthings\b", "specific factors", "vague noun"),
    (r"\betc\.?\b", "", "open-ended list"),
]


def improve_answer(content: str, options: dict | None = None) -> str:
    topic = (options or {}).get("topic")
    text = " ".join(line.strip() for line in content.splitlines() if line.strip())
    changes: list[str] = []

    improved = text
    applied: list[str] = []
    for pattern, replacement, label in _HEDGE_RULES:
        if re.search(pattern, improved, flags=re.IGNORECASE):
            improved = re.sub(pattern, replacement, improved, flags=re.IGNORECASE)
            improved = re.sub(r"\s+", " ", improved).strip(" ,")
            applied.append(label)

    if applied:
        changes.append(
            f"Removed {len(applied)} instance(s) of weak language "
            f"({', '.join(sorted(set(applied)))}) to make the claims direct."
        )
    else:
        changes.append("Tightened the wording without changing your meaning.")

    # Structural upgrade: direct opening, supporting detail, closing summary.
    sentences = _sentences(improved)
    if sentences:
        opening = sentences[0].strip().rstrip(".") + "."
        middle = [s.strip() for s in sentences[1:] if s.strip()]
        body = " ".join(middle) if middle else ""
        closing = f"In summary, {sentences[0].strip().rstrip('.').lower()}." if middle else ""
        restructured = " ".join(p for p in (opening, body, closing) if p)
        if restructured != improved:
            changes.append("Reorganised the answer into opening claim, supporting detail and a closing summary.")
        improved = restructured

    if topic:
        improved = f"Regarding {topic}: {improved}"
        changes.append("Tied the answer explicitly to the question/topic asked.")

    if len(improved.split()) < 25:
        changes.append("The answer is still short - add one concrete example or definition from your notes.")

    improved = re.sub(r"\s+", " ", improved).strip()
    changed_block = "\n".join(f"- {c}" for c in changes[:5])
    return f"## Improved answer\n{improved}\n\n## What changed\n{changed_block}"


# --------------------------------------------------------------------------- #
# feature: explain a concept
# --------------------------------------------------------------------------- #
def explain_concept(content: str, options: dict | None = None) -> str:
    topic = ((options or {}).get("topic") or "").strip()
    sentences = _sentences(content)
    terms = _key_terms(content, limit=6)

    focus = topic.lower() if topic else None
    definition = None
    for sentence in sentences:
        low = sentence.lower()
        if focus and focus in low and _DEFINITION.match(sentence.strip()):
            definition = sentence.strip()
            break
    if definition is None:
        for sentence in sentences:
            if _DEFINITION.match(sentence.strip()):
                definition = sentence.strip()
                break
    if definition is None and sentences:
        definition = sentences[0].strip()

    one_line = definition or (f"{topic or 'This topic'} is described in your notes; see the simple-terms section below.")

    simple_terms = " ".join(sentences[:2]).strip() if sentences else "No material was provided."
    detail_sentences = sentences[2:6] if len(sentences) > 2 else sentences[1:]
    detail = " ".join(s.strip() for s in detail_sentences).strip() or "Your notes do not go deeper on this yet."
    if terms:
        bolded = detail
        for term in terms[:4]:
            bolded = re.sub(rf"\b{re.escape(term)}\b", f"**{term}**", bolded, flags=re.IGNORECASE)
        detail = bolded

    if len(terms) >= 2:
        mistake = (
            f"Do not confuse **{terms[0]}** with **{terms[1]}** - they appear together in your notes "
            "but play different roles."
        )
    elif definition:
        mistake = "Students often paraphrase this definition too loosely; memorise it in its precise form."
    else:
        mistake = "Not enough material to identify a common mistake - add more notes on this topic."

    return (
        "## In one line\n"
        f"{one_line}\n\n"
        "## In simple terms\n"
        f"{simple_terms}\n\n"
        "## Deeper detail\n"
        f"{detail}\n\n"
        "## Common mistake to avoid\n"
        f"{mistake}"
    )


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
_ENGINES = {
    "summarize": summarize,
    "quiz": generate_quiz,
    "improve": improve_answer,
    "explain": explain_concept,
}


def offline_generate(feature: str, content: str, options: dict | None = None) -> str:
    """Deterministic offline generation for any supported feature."""

    engine = _ENGINES.get(feature)
    if engine is None:
        raise ValueError(f"Unknown feature: {feature!r}")
    return engine(content, options)

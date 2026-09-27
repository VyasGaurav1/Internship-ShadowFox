"""Small text utilities shared across the pipeline (no third-party deps)."""
from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "it",
    "its", "as", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "about", "into", "over", "after", "before", "so", "such", "not", "no",
    "can", "could", "should", "would", "may", "might", "will", "shall", "do",
    "does", "did", "have", "has", "had", "we", "you", "they", "he", "she",
    "i", "me", "my", "our", "your", "their", "them", "us", "also", "very",
    "just", "there", "here", "which", "who", "what", "when", "how", "why",
}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def content_words(text: str) -> list[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        out.extend(s.strip() for s in _SENT_RE.split(block) if s.strip())
    return out

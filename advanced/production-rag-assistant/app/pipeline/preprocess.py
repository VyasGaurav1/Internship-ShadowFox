"""Text preprocessing / normalisation before chunking.

Keeps the transformations deterministic and inspectable, and records what it did
so the API can report ingestion statistics.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass


@dataclass
class PreprocessStats:
    original_chars: int
    cleaned_chars: int
    removed_repeated_lines: int
    joined_hyphen_breaks: int
    collapsed_blocks: int

    def as_dict(self) -> dict:
        return {
            "original_chars": self.original_chars,
            "cleaned_chars": self.cleaned_chars,
            "removed_repeated_lines": self.removed_repeated_lines,
            "joined_hyphen_breaks": self.joined_hyphen_breaks,
            "collapsed_blocks": self.collapsed_blocks,
        }


def _remove_repeated_lines(lines: list[str]) -> tuple[list[str], int]:
    """Drop short lines that repeat many times (running headers/footers)."""

    counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, count in counts.items() if count >= 4 and len(line) <= 80}
    if not repeated:
        return lines, 0
    kept = [line for line in lines if line.strip() not in repeated]
    return kept, len(lines) - len(kept)


def preprocess(text: str) -> tuple[str, PreprocessStats]:
    original_chars = len(text)

    # 1) unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # 2) soft hyphen / trailing hyphen line breaks -> join words
    before = text
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    joined = len(re.findall(r"\w-\n\w", before))

    # 3) strip trailing whitespace, normalise newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    # 4) drop repeated header/footer lines
    lines, removed = _remove_repeated_lines(lines)
    text = "\n".join(lines)

    # 5) collapse 3+ blank lines into a paragraph break
    collapsed = len(re.findall(r"\n{3,}", text))
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6) collapse runs of spaces/tabs (keep newlines)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    return text, PreprocessStats(
        original_chars=original_chars,
        cleaned_chars=len(text),
        removed_repeated_lines=removed,
        joined_hyphen_breaks=joined,
        collapsed_blocks=collapsed,
    )

"""Input validation for the Document Q&A Assistant."""
from __future__ import annotations


class ValidationError(ValueError):
    def __init__(self, message: str, field: str = "question") -> None:
        super().__init__(message)
        self.field = field
        self.message = message


def validate_question(question: str, min_chars: int = 3, max_chars: int = 1000) -> str:
    if question is None:
        raise ValidationError("Please type a question.")
    if not isinstance(question, str):
        raise ValidationError("The question must be text.")
    text = question.strip()
    if not text:
        raise ValidationError("Your question is empty. Please type a question about the documents.")
    if len(text) < min_chars:
        raise ValidationError(f"Your question is too short ({len(text)} characters).")
    if len(text) > max_chars:
        raise ValidationError(f"Your question is too long ({len(text)} characters, limit {max_chars}).")
    return text

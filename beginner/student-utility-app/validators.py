"""Input validation for the Student Utility App.

Validation is intentionally strict and explicit so the UI can show a concrete,
actionable message instead of sending empty or nonsensical input to the model.
"""
from __future__ import annotations

GENERIC_MIN = 15
GENERIC_MAX = 12000


class ValidationError(ValueError):
    """Raised when user input is empty or otherwise invalid."""

    def __init__(self, message: str, field: str = "content") -> None:
        super().__init__(message)
        self.field = field
        self.message = message


def validate_content(
    content: str,
    *,
    min_chars: int = GENERIC_MIN,
    max_chars: int = GENERIC_MAX,
    field: str = "content",
    label: str = "input",
) -> str:
    """Validate and normalise free-text input. Returns the stripped text.

    Raises :class:`ValidationError` for empty, whitespace-only, too-short or
    too-long input, and for input that is not text.
    """

    if content is None:
        raise ValidationError(f"Please provide some {label}.", field=field)
    if not isinstance(content, str):
        raise ValidationError(f"The {label} must be text.", field=field)

    text = content.strip()
    if not text:
        raise ValidationError(f"Your {label} is empty. Please enter some {label}.", field=field)
    if len(text) < min_chars:
        raise ValidationError(
            f"Your {label} is too short ({len(text)} characters). "
            f"Please provide at least {min_chars} characters so the assistant has enough to work with.",
            field=field,
        )
    if len(text) > max_chars:
        raise ValidationError(
            f"Your {label} is too long ({len(text)} characters). "
            f"Please shorten it to under {max_chars} characters.",
            field=field,
        )
    return text


def validate_feature(feature: str) -> str:
    """Ensure the requested feature is supported."""

    from prompts import FEATURES

    if not isinstance(feature, str) or feature.strip().lower() not in FEATURES:
        raise ValidationError(
            f"Unknown feature {feature!r}. Choose one of: {', '.join(FEATURES)}.",
            field="feature",
        )
    return feature.strip().lower()

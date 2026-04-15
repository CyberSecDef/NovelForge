"""Input validation helpers for NovelForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import novelforge.config as config

CHARACTER_FIELD_LIMITS: dict[str, int] = {
    "name": 100,
    "age": 50,
    "role": 512,
    "background": 2000,
    "arc": 2000,
    "internal_flaw": 512,
    "personal_goal": 512,
}


def validate_character_fields(
    character: dict, limits: dict[str, int] | None = None,
) -> dict[str, int]:
    """Return a mapping of ``field → actual_length`` for every string field
    in *character* that exceeds its configured maximum.

    Fields not present in *limits* or not of type ``str`` are skipped.
    ``age`` is coerced to ``str`` before measuring so numeric ages (e.g.
    ``35``) are not flagged.
    """
    if limits is None:
        limits = CHARACTER_FIELD_LIMITS
    violations: dict[str, int] = {}
    for field_name, max_len in limits.items():
        value = character.get(field_name)
        if value is None:
            continue
        text = str(value)
        if len(text) > max_len:
            violations[field_name] = len(text)
    return violations


def truncate_character_field(
    value: str, max_len: int, suffix: str = "…",
) -> str:
    """Truncate *value* to *max_len* characters, preserving word boundaries
    where possible and appending *suffix* (which counts toward the cap).

    Used as a last-resort fallback when LLM repair cannot bring a field
    within its limit.
    """
    if len(value) <= max_len:
        return value
    budget = max_len - len(suffix)
    if budget <= 0:
        return value[:max_len]
    cut = value[:budget]
    space = cut.rfind(" ")
    if space > budget * 0.6:
        cut = cut[:space].rstrip(",;:- ")
    return cut + suffix


ALLOWED_GENRES = {
    "Adventure",
    "Contemporary Fiction",
    "Crime",
    "Dystopian",
    "Fantasy",
    "Gothic Fiction",
    "Historical Fiction",
    "Horror",
    "Literary Fiction",
    "Magical Realism",
    "Mystery",
    "Noir",
    "Paranormal",
    "Romance",
    "Satire Humor",
    "Science Fiction",
    "Speculative Fiction",
    "Thriller",
    "Urban Fantasy",
    "Western",
    "Young Adult",
}


@dataclass
class ValidationResult:
    """Structured result returned by :func:`validate_outline_input`.

    Attributes:
        ok: ``True`` when all fields passed validation.
        errors: Mapping of field name to a human-readable error message for
            every field that failed validation.  Empty when *ok* is ``True``.
        values: Normalised/coerced field values (stripped strings, parsed
            integers) for every field that passed validation.  Callers should
            use these instead of re-parsing the raw request data.
    """

    ok: bool
    errors: dict[str, str] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def first_error(self) -> str:
        """Return the first error message, or an empty string when there are none."""
        return next(iter(self.errors.values()), "")

    def __iter__(self) -> Iterator[object]:
        """Allow backward-compatible tuple unpacking: ``ok, err = validate_outline_input(data)``.

        Yields ``ok`` then :attr:`first_error` so existing call-sites that
        unpack the result as a two-tuple continue to work unchanged.
        """
        yield self.ok
        yield self.first_error


def validate_outline_input(data: dict) -> ValidationResult:
    """Validate the ``/generate_outline`` form data.

    All fields are checked independently so the caller receives every
    validation error in a single call rather than only the first one.

    Returns a :class:`ValidationResult` whose ``errors`` dict maps each
    failing field name to a user-facing message and whose ``values`` dict
    contains the normalised (stripped / coerced) value for every field that
    passed validation.
    """
    errors: dict[str, str] = {}
    values: dict[str, Any] = {}

    # --- premise ---
    premise = data.get("premise", "")
    if not isinstance(premise, str):
        errors["premise"] = "Story premise must be a string."
    else:
        premise = premise.strip()
        if not premise:
            errors["premise"] = "Story premise is required."
        elif len(premise) > 2000:
            errors["premise"] = "Story premise must be 2000 characters or fewer."
        else:
            values["premise"] = premise

    # --- genre ---
    genre = data.get("genre", "")
    if not isinstance(genre, str):
        errors["genre"] = "Genre must be a string."
    else:
        genre = genre.strip()
        if genre not in ALLOWED_GENRES:
            errors["genre"] = f"Invalid genre. Choose from: {', '.join(sorted(ALLOWED_GENRES))}."
        else:
            values["genre"] = genre

    # --- chapters ---
    try:
        chapters = int(data.get("chapters", 0))
        if chapters < 3:
            errors["chapters"] = "Number of chapters must be at least 3."
        elif chapters > config.MAX_CHAPTERS:
            errors["chapters"] = f"Number of chapters must be {config.MAX_CHAPTERS:,} or fewer."
        else:
            values["chapters"] = chapters
    except (ValueError, TypeError):
        errors["chapters"] = "Chapters must be a valid number."

    # --- word_count ---
    try:
        word_count = int(data.get("word_count", 0))
        if word_count < 1000:
            errors["word_count"] = "Word count must be at least 1000."
        elif word_count > config.MAX_WORD_COUNT:
            errors["word_count"] = f"Word count must be {config.MAX_WORD_COUNT:,} or fewer."
        else:
            values["word_count"] = word_count
    except (ValueError, TypeError):
        errors["word_count"] = "Word count must be a valid number."

    # --- special_events ---
    special_events = data.get("special_events", "")
    if not isinstance(special_events, str):
        errors["special_events"] = "Special events must be a string."
    else:
        special_events = special_events.strip()
        if len(special_events) > 5000:
            errors["special_events"] = "Special events must be 5,000 characters or fewer."
        else:
            values["special_events"] = special_events

    # --- special_instructions ---
    special_instructions = data.get("special_instructions", "")
    if not isinstance(special_instructions, str):
        errors["special_instructions"] = "Special instructions must be a string."
    else:
        special_instructions = special_instructions.strip()
        if len(special_instructions) > 5000:
            errors["special_instructions"] = "Special instructions must be 5,000 characters or fewer."
        else:
            values["special_instructions"] = special_instructions

    return ValidationResult(ok=not errors, errors=errors, values=values)

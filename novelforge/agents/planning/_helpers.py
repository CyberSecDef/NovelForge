"""Shared helpers used by all planning agent modules."""

import logging

from novelforge.llm.prompts import render_prompt  # noqa: F401 – re-exported for submodules

logger = logging.getLogger(__name__)


def _build_system_prompt(role: str) -> dict[str, str]:
    """Wrap a role description into an LLM system-message dict."""
    return {"role": "system", "content": role}


def choose_story_architecture_mode(total_chapters: int) -> str:
    """Choose a strict act model based on project size."""
    return "four-act" if total_chapters >= 16 else "three-act"


def _coerce_positive_int(value: object, default: int) -> int:
    """Safely coerce *value* to a positive int, returning *default* on failure."""
    try:
        coerced = int(value)  # type: ignore[call-overload]
        return coerced if coerced > 0 else default
    except (TypeError, ValueError):
        return default


_PLACEHOLDER_CHAPTER: dict[str, object] = {"number": 1, "title": "Chapter 1", "summary": ""}


def _safe_chapter_list(chapter_list: object) -> list[dict]:
    """Return a guaranteed non-empty list of dicts from *chapter_list*.

    Filters out any entry that is not a ``dict`` and, if the result would be
    empty, substitutes a single placeholder entry so that downstream code can
    always index ``[0]`` or iterate safely.
    """
    if not isinstance(chapter_list, list):
        return [dict(_PLACEHOLDER_CHAPTER)]
    filtered = [ch for ch in chapter_list if isinstance(ch, dict)]
    return filtered if filtered else [dict(_PLACEHOLDER_CHAPTER)]

"""Shared constants, pass-failure helpers, content-policy retry, and vocabulary scanning."""

import logging
import re
from collections.abc import Callable

from novelforge.llm.client import call_llm, ContentRejectionError

logger = logging.getLogger(__name__)

# Maximum number of content-sanitization retries per LLM call
_CONTENT_RETRY_LIMIT = 2

# ---------------------------------------------------------------------------
# Shared pass-failure helpers
# ---------------------------------------------------------------------------

#: Key injected into dict-returning runner helpers when the pass fails.
#: Value is a human-readable ``"ExcType: message"`` string.
PASS_FAILURE_KEY = "_pass_failed"

#: Value used when a pass is classified as optional (graceful degradation).
PASS_OPTIONAL = "optional"

#: Value used when a pass is classified as required (pipeline-critical).
PASS_REQUIRED = "required"


def _log_pass_failure(
    exc: Exception,
    *,
    pass_name: str,
    chapter_num: int | None = None,
    chapter_title: str | None = None,
    optional: bool = True,
) -> str:
    """Log a structured warning for a failed chapter-agent pass and return a
    failure summary string suitable for injecting into fallback return values.
    """
    pass_kind = PASS_OPTIONAL if optional else PASS_REQUIRED
    chapter_ctx = ""
    if chapter_num is not None:
        chapter_ctx = f" chapter={chapter_num}"
        if chapter_title:
            chapter_ctx += f" title={chapter_title!r}"
    logger.warning(
        "Chapter-agent pass FAILED — pass=%r%s kind=%s error=%s: %s",
        pass_name,
        chapter_ctx,
        pass_kind,
        type(exc).__name__,
        exc,
    )
    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Content-policy sanitization and retry
# ---------------------------------------------------------------------------

def _sanitize_for_content_policy(
    text: str, chapter_num: int, title: str, rejection_reason: str,
) -> str:
    """
    Ask the LLM to rewrite passages that triggered a content-policy filter.

    The sanitisation prompt is deliberately clinical and unlikely to trip
    the same filter.  If even the sanitisation call is rejected, a second,
    more aggressive rewrite is attempted.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an editorial assistant. A chapter of fiction was rejected by an "
                "automated content filter. Identify the specific passages that most likely "
                "triggered the rejection and rewrite ONLY those passages so they convey the "
                "same narrative events through implication, atmosphere, and restraint rather "
                "than explicit detail.\n\n"
                "Rules:\n"
                "- Preserve the chapter's narrative arc, characters, pacing, tone, and "
                "approximate word count.\n"
                "- Change as little as possible — only the passages that likely caused the "
                "rejection.\n"
                "- Return the COMPLETE chapter text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Chapter {chapter_num} of \"{title}\" was rejected by a content filter.\n\n"
                f"Filter response:\n{rejection_reason}\n\n"
                f"Rewrite only the problematic passages and return the complete chapter.\n\n"
                f"---\n\n{text}"
            ),
        },
    ]
    try:
        sanitized = call_llm(messages, action=f"Chapter {chapter_num}: content sanitization")
        logger.info(
            "Chapter %d: content sanitization complete (%d chars → %d chars)",
            chapter_num, len(text), len(sanitized),
        )
        return sanitized
    except ContentRejectionError:
        # The sanitisation request itself was rejected — try a more aggressive rewrite
        logger.warning(
            "Chapter %d: sanitization request also rejected, attempting aggressive rewrite",
            chapter_num,
        )
        messages[0]["content"] = (
            "You are an editorial assistant. Rewrite the following chapter so that all "
            "mature themes (violence, horror, psychological distress, etc.) are conveyed "
            "through implication and atmosphere rather than direct description. Replace "
            "any graphic or disturbing imagery with restrained literary alternatives. "
            "Keep all character names, plot points, and chapter structure intact. "
            "Return the COMPLETE rewritten chapter."
        )
        messages[1]["content"] = (
            f"Rewrite Chapter {chapter_num} of \"{title}\" with literary restraint "
            f"for all mature content. Return the complete chapter.\n\n---\n\n{text}"
        )
        return call_llm(messages, action=f"Chapter {chapter_num}: aggressive content rewrite")


def _call_with_content_retry(
    build_messages: Callable[[str], list[dict]],
    text: str,
    *,
    action: str,
    chapter_num: int,
    title: str,
    json_mode: bool = False,
) -> str:
    """
    Call the LLM with automatic content-rejection retry.

    On ``ContentRejectionError`` the chapter *text* (which is the most likely
    trigger) is sanitised and the prompt is rebuilt. Up to
    ``_CONTENT_RETRY_LIMIT`` sanitisation attempts are made before the error
    is re-raised.

    Returns the LLM response string.
    """
    current_text = text
    for attempt in range(_CONTENT_RETRY_LIMIT + 1):
        try:
            return call_llm(build_messages(current_text), action=action, json_mode=json_mode)
        except ContentRejectionError as exc:
            if attempt >= _CONTENT_RETRY_LIMIT:
                raise
            logger.warning(
                "Content rejection on '%s' (attempt %d/%d), sanitizing chapter text…",
                action, attempt + 1, _CONTENT_RETRY_LIMIT,
            )
            current_text = _sanitize_for_content_policy(
                current_text, chapter_num, title, str(exc),
            )
    # Unreachable, but keeps the type checker happy
    raise ContentRejectionError(f"Content retry limit exceeded for {action}")


# Content-guidance note injected into special_instructions on draft retries.
_DRAFT_CONTENT_NOTE = (
    "CONTENT NOTE: A previous draft attempt was rejected by a content "
    "filter. Handle all mature themes (violence, horror, psychological "
    "distress, body horror, etc.) through implication, atmosphere, "
    "tension, and literary restraint rather than graphic or explicit "
    "description. Show emotional and psychological impact. The story's "
    "dark tone must be preserved but conveyed through what is suggested "
    "and felt, not what is shown in detail."
)


def _draft_with_content_retry(
    build_prompt_fn: Callable[[str], list[dict]],
    *,
    action: str,
    special_instructions: str,
    chapter_num: int,
    max_attempts: int = 3,
) -> str:
    """
    Call the LLM to produce an initial chapter draft, with content-rejection retry.

    On ``ContentRejectionError`` a content-guidance note is appended to
    ``special_instructions`` and the prompt is rebuilt via ``build_prompt_fn``.
    Up to ``max_attempts`` are made before the error is re-raised.

    ``build_prompt_fn`` must accept a single ``instructions`` string and return
    the message list to pass to :func:`call_llm`.
    """
    content_note = ""
    for attempt in range(max_attempts):
        try:
            instructions = special_instructions
            if content_note:
                instructions = (
                    f"{special_instructions}\n\n{content_note}"
                    if special_instructions
                    else content_note
                )
            return call_llm(build_prompt_fn(instructions), action=action)
        except ContentRejectionError:
            if attempt >= max_attempts - 1:
                raise
            logger.warning(
                "Chapter %d draft rejected by content filter (attempt %d/%d), "
                "adding content guidance and retrying",
                chapter_num, attempt + 1, max_attempts - 1,
            )
            content_note = _DRAFT_CONTENT_NOTE
    # Unreachable, but keeps the type checker happy
    raise ContentRejectionError(f"Draft content retry limit exceeded for {action}")


# ---------------------------------------------------------------------------
# Vocabulary watchlists and scanning
# ---------------------------------------------------------------------------

# Words indicative of LLM-generated text that the anti-LLM agent should remove
# Hard-banned words: never use these — they are strong LLM fingerprints
_FORBIDDEN_WORDS = [
    # Original LLM tics
    "embark", "delve", "realm", "tapestry", "testament", "nuance",
    "beacon", "uncharted", "multifaceted", "leverage", "synergy",
    "pivotal", "groundbreaking", "commendable", "meticulous",
    # Discovered cross-novel overuse
    "provenance", "cadence", "choreography",
    # Bookkeeping / accounting metaphors used as emotional shorthand
    "ledger", "tally", "inventory", "audit", "balance sheet",
    "debit", "dividend",
]

# Soft-limited words: OK once or twice per novel, but the LLM wildly overuses them
_SOFT_LIMITED_WORDS = [
    "brittle", "tighten", "tightened", "tightening",
    "steady", "steadied", "steadiness", "deliberate", "measured",
    "calculus", "arithmetic",
]

# Overused patterns: phrases and constructions to avoid
_OVERUSED_PATTERNS = [
    "the economy of", "with the economy of",
    "someone who had", "the patience of someone",
    "the particular steadiness of", "the flat gaze of someone",
    "small mercy", "small victory", "small repair", "small rebellion",
    "small grief", "small comfort", "small kindness", "small cruelty",
    "sat behind her ribs", "sat behind his ribs",
    "lodged in her sternum", "lodged in his sternum",
    "lodged under her ribs", "lodged under his ribs",
    "tasted like metal", "tasted of metal", "taste of iron",
    "smelled of ozone", "jaw tightened", "jaw worked",
    "not a victory", "not heroic", "not comfort",
    "moral arithmetic", "moral calculus", "moral cost", "moral weight",
]


def _compile_word_pattern(words: list[str]) -> re.Pattern[str]:
    """Compile a regex that matches any of *words* at word boundaries."""
    alternation = "|".join(re.escape(w) for w in words)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


# Pre-compiled patterns for the vocabulary scanner (built once at import time)
_FORBIDDEN_RE = _compile_word_pattern(_FORBIDDEN_WORDS)
_SOFT_LIMITED_RE = _compile_word_pattern(_SOFT_LIMITED_WORDS)
_OVERUSED_PATTERN_RE = _compile_word_pattern(_OVERUSED_PATTERNS)


def _format_anti_repetition_rules() -> str:
    """Format the soft-limited words and overused patterns for prompt injection."""
    lines = []
    lines.append("SOFT-LIMITED WORDS (use at most 2-3 times in the ENTIRE novel):")
    lines.append(", ".join(_SOFT_LIMITED_WORDS))
    lines.append("")
    lines.append("OVERUSED PATTERNS (avoid entirely — find fresh alternatives):")
    for p in _OVERUSED_PATTERNS:
        lines.append(f'  - "{p}"')
    return "\n".join(lines)


# Maximum allowed occurrences per chapter for each word tier
_HARD_BAN_THRESHOLD = 0        # hard-banned: never
_SOFT_LIMIT_PER_CHAPTER = 1    # soft-limited: at most once per chapter
_PATTERN_THRESHOLD = 0          # overused patterns: never


def _count_word_matches(pattern: re.Pattern[str], text: str) -> dict[str, int]:
    """Return a ``{matched_word_lower: count}`` dict for all hits of *pattern* in *text*."""
    counts: dict[str, int] = {}
    for m in pattern.finditer(text):
        key = m.group().lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def scan_vocabulary_overuse(chapter_text: str) -> list[str]:
    """
    Scan a chapter for overused vocabulary from the watchlists.

    Returns a list of human-readable warnings for each violation found.
    Pure Python — no LLM call.  Uses pre-compiled word-boundary regexes
    so that ``"audit"`` does **not** match inside ``"auditor"`` or
    ``"ledger"`` inside ``"sledgehammer"``.
    """
    warnings: list[str] = []

    # Check hard-banned words
    for word, count in _count_word_matches(_FORBIDDEN_RE, chapter_text).items():
        if count > _HARD_BAN_THRESHOLD:
            warnings.append(
                f'BANNED WORD "{word}" appears {count}x — must be removed entirely'
            )

    # Check soft-limited words
    for word, count in _count_word_matches(_SOFT_LIMITED_RE, chapter_text).items():
        if count > _SOFT_LIMIT_PER_CHAPTER:
            warnings.append(
                f'OVERUSED WORD "{word}" appears {count}x in this chapter '
                f'(limit: {_SOFT_LIMIT_PER_CHAPTER}) — replace most occurrences '
                f'with varied alternatives'
            )

    # Check overused patterns
    for pattern, count in _count_word_matches(_OVERUSED_PATTERN_RE, chapter_text).items():
        if count > _PATTERN_THRESHOLD:
            warnings.append(
                f'OVERUSED PATTERN "{pattern}" appears {count}x — '
                f'rewrite with a fresh, specific alternative'
            )

    return warnings

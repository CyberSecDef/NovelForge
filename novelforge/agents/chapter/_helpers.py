"""Shared constants, pass-failure helpers, content-policy retry, vocabulary scanning, and length enforcement."""

import difflib
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
                chapter_num, attempt + 1, max_attempts,
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
    # Institutional / formal register words LLMs overuse
    "mandate", "decree", "edict",
    "apparatus", "machinations",
]

# Legal terms forbidden in all genres except crime-adjacent ones.
# When a crime-adjacent genre is detected these become soft-limited instead.
_LEGAL_TERMS = [
    "verdict", "indictment", "tribunal", "acquittal", "exonerate",
    "adjudicate", "clemency", "arbitrate", "testimony", "jurisprudence",
    "litigate", "prosecution", "prosecute",
]

# Genres where legal terminology is contextually appropriate (soft-limited, not banned).
_LEGAL_ADJACENT_GENRES = {"Crime", "Mystery", "Noir", "Thriller"}

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
_LEGAL_TERMS_RE = _compile_word_pattern(_LEGAL_TERMS)
_SOFT_LIMITED_RE = _compile_word_pattern(_SOFT_LIMITED_WORDS)
_OVERUSED_PATTERN_RE = _compile_word_pattern(_OVERUSED_PATTERNS)


def get_forbidden_words(genre: str = "") -> list[str]:
    """Return the full forbidden-word list, adding legal terms unless the genre is legal-adjacent."""
    if genre in _LEGAL_ADJACENT_GENRES:
        return list(_FORBIDDEN_WORDS)
    return list(_FORBIDDEN_WORDS) + list(_LEGAL_TERMS)


def get_soft_limited_words(genre: str = "") -> list[str]:
    """Return soft-limited words, including legal terms for legal-adjacent genres."""
    if genre in _LEGAL_ADJACENT_GENRES:
        return list(_SOFT_LIMITED_WORDS) + list(_LEGAL_TERMS)
    return list(_SOFT_LIMITED_WORDS)


def format_vocabulary_rules(genre: str = "") -> str:
    """Return a compact vocabulary-constraint block for injection into agent system prompts."""
    forbidden = get_forbidden_words(genre)
    soft = get_soft_limited_words(genre)
    return (
        "VOCABULARY CONSTRAINTS (strict — apply to every word you write):\n"
        f"NEVER use these words: {', '.join(forbidden)}.\n"
        f"Limit these to at most 1 occurrence per chapter: {', '.join(soft)}.\n"
        "Avoid: accounting/legal metaphors for emotions, "
        '"small [mercy/victory/repair]" constructions, emotions lodged in '
        "ribs/sternum/throat, metallic taste as distress, "
        '"jaw tightened," "the economy of someone who."'
    )


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


def scan_vocabulary_overuse(chapter_text: str, genre: str = "") -> list[str]:
    """
    Scan a chapter for overused vocabulary from the watchlists.

    Returns a list of human-readable warnings for each violation found.
    Pure Python — no LLM call.  Uses pre-compiled word-boundary regexes
    so that ``"audit"`` does **not** match inside ``"auditor"`` or
    ``"ledger"`` inside ``"sledgehammer"``.

    When *genre* is a legal-adjacent genre (Crime, Mystery, Noir, Thriller),
    legal terms are soft-limited instead of hard-banned.
    """
    warnings: list[str] = []
    is_legal_adjacent = genre in _LEGAL_ADJACENT_GENRES

    # Check hard-banned words
    for word, count in _count_word_matches(_FORBIDDEN_RE, chapter_text).items():
        if count > _HARD_BAN_THRESHOLD:
            warnings.append(
                f'BANNED WORD "{word}" appears {count}x — must be removed entirely'
            )

    # Check legal terms — hard-banned unless genre is legal-adjacent
    for word, count in _count_word_matches(_LEGAL_TERMS_RE, chapter_text).items():
        if is_legal_adjacent:
            if count > _SOFT_LIMIT_PER_CHAPTER:
                warnings.append(
                    f'OVERUSED LEGAL TERM "{word}" appears {count}x in this chapter '
                    f'(limit: {_SOFT_LIMIT_PER_CHAPTER}) — replace most occurrences '
                    f'with varied alternatives'
                )
        else:
            if count > _HARD_BAN_THRESHOLD:
                warnings.append(
                    f'BANNED LEGAL TERM "{word}" appears {count}x — must be removed '
                    f'entirely (not a legal-themed novel)'
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


# ---------------------------------------------------------------------------
# Named-character detection (for reconciliation against the canonical roster)
# ---------------------------------------------------------------------------

# Common capitalized English words that are NOT character names. Used to
# filter sentence-initial and conventional capitalization out of the
# named-character scanner. Roster-token matching is applied BEFORE this
# filter, so a character legitimately named "May" or "Crown" is still
# detected correctly — the stop list only catches spans that have no
# roster hit.
_NAMED_CHARACTER_STOP_WORDS: frozenset[str] = frozenset({
    # Pronouns / sentence-initial
    "i", "he", "she", "they", "it", "we", "you", "me", "him", "her", "them", "us",
    "his", "hers", "theirs", "its", "ours", "yours", "mine",
    "this", "that", "these", "those", "there", "here",
    "then", "when", "where", "why", "how", "what", "who", "whose", "which",
    # Conjunctions / modifiers
    "the", "a", "an", "and", "or", "but", "so", "yet", "as", "if", "while",
    "because", "since", "although", "though", "unless", "until",
    "not", "never", "always", "still", "only", "even", "also",
    "now", "before", "after", "later", "soon", "ago", "once", "twice",
    "yes", "no", "ok", "okay",
    # Days
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Months (excluding May — often a character name; roster check handles it)
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    # Honorifics / titles that commonly appear alone
    "mr", "mrs", "ms", "dr", "sir", "madam", "lord", "lady",
    "captain", "lieutenant", "sergeant", "major", "colonel", "general",
    "professor", "father", "mother", "sister", "brother", "uncle", "aunt",
    "detective", "inspector", "officer", "commander", "admiral", "chief",
    "doctor", "nurse", "reverend", "pastor",
    # Structural / narrative
    "chapter", "book", "part", "act", "scene", "volume", "prologue", "epilogue",
    # Exclamations / religious references
    "god", "christ", "jesus", "heaven", "hell", "lord",
    # Greetings / filler
    "hello", "goodbye", "thanks", "please",
    # Cardinal directions / generic place words
    "north", "south", "east", "west", "street", "road", "avenue", "place",
    "square", "city", "town", "village", "county", "state", "country",
})


# Candidate-name regex: one to three adjacent capitalized tokens.
# Matches "Sarah", "Sarah Miller", "John Fitzgerald Kennedy" but does not
# span apostrophes, hyphens, or punctuation — so "Sarah's" yields "Sarah".
_NAME_CANDIDATE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


def _roster_name_token_sets(roster: list[dict]) -> list[set[str]]:
    """Return a list of lowercase token sets — one set per roster character.

    Each character's ``name`` is split on whitespace; tokens shorter than
    two characters are discarded (they match too many false positives under
    fuzzy matching). Returning per-character sets (rather than a flat union)
    lets the scanner distinguish "Marcus Reid" from "Marcus Fellowes" —
    the shared "marcus" token alone is not enough to classify a prose span
    as a known roster character.
    """
    result: list[set[str]] = []
    for ch in roster or []:
        if not isinstance(ch, dict):
            continue
        name = str(ch.get("name", "")).strip()
        if not name:
            continue
        char_tokens = {
            tok.strip(".,;:'\"").lower()
            for tok in name.split()
            if len(tok.strip(".,;:'\"")) >= 2
        }
        if char_tokens:
            result.append(char_tokens)
    return result


def extract_named_characters(
    chapter_text: str,
    roster: list[dict],
    *,
    min_mentions: int = 2,
    fuzzy_cutoff: float = 0.85,
) -> dict:
    """Detect named characters in chapter prose and classify them against *roster*.

    Pure Python — no LLM call. Uses a capitalized-span regex, a stop-word
    filter, and :func:`difflib.get_close_matches` for variant detection.

    Parameters
    ----------
    chapter_text:  The chapter prose to scan.
    roster:        The canonical ``character_list`` (list of dicts with
                   a ``name`` key).
    min_mentions:  Minimum distinct mentions required before a capitalized
                   span is reported as an unknown character. Spans that
                   appear fewer times are treated as likely sentence-initial
                   false positives or throwaway walk-ons.
    fuzzy_cutoff:  :mod:`difflib` similarity threshold for variant matching.
                   Higher = stricter. 0.85 catches typos and short
                   diminutives without conflating distinct names.

    Returns
    -------
    dict with three keys:
        ``known``:    sorted list of capitalized spans that intersect the
                      roster's name tokens (for diagnostic logging).
        ``unknown``:  list of ``(prose_name, count)`` tuples for names with
                      no roster match and at least *min_mentions* occurrences,
                      ordered by descending count.
        ``variants``: list of ``(prose_name, roster_token, count)`` tuples
                      — likely misspellings or diminutives of roster names.
    """
    per_char_tokens = _roster_name_token_sets(roster)
    # Flat union is kept only for difflib variant matching below; the known/
    # unknown classification uses per-character sets to avoid cross-character
    # false positives like "Marcus Fellowes" matching "Marcus Reid".
    flat_tokens: set[str] = {t for s in per_char_tokens for t in s}

    raw_counts: dict[str, int] = {}
    for m in _NAME_CANDIDATE_RE.finditer(chapter_text):
        raw_counts[m.group()] = raw_counts.get(m.group(), 0) + 1

    known: set[str] = set()
    unknown_counts: dict[str, int] = {}
    for span, count in raw_counts.items():
        span_tokens_list = [t.lower() for t in span.split()]
        span_tokens = set(span_tokens_list)
        # Roster check first: a span is known only when it maps entirely to
        # a single roster character — either the span's tokens are a subset
        # of that character's tokens (e.g. "Marcus" → "Marcus Reid") or a
        # superset (e.g. "Marcus Reid the Third" → "Marcus Reid").
        is_known = any(
            span_tokens.issubset(char_set) or char_set.issubset(span_tokens)
            for char_set in per_char_tokens
        )
        if is_known:
            known.add(span)
            continue
        # Drop spans whose every token is a stop word (sentence-initial
        # noise, honorifics with no name attached, etc.).
        if all(t in _NAMED_CHARACTER_STOP_WORDS for t in span_tokens_list):
            continue
        if count < min_mentions:
            continue
        unknown_counts[span] = count

    variants: list[tuple[str, str, int]] = []
    unknowns: list[tuple[str, int]] = []
    roster_token_list = sorted(flat_tokens)
    for span, count in sorted(unknown_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        match_found: str | None = None
        if roster_token_list:
            for t in span.split():
                close = difflib.get_close_matches(
                    t.lower(), roster_token_list, n=1, cutoff=fuzzy_cutoff,
                )
                if close:
                    match_found = close[0]
                    break
        if match_found is not None:
            variants.append((span, match_found, count))
        else:
            unknowns.append((span, count))

    return {
        "known": sorted(known),
        "unknown": unknowns,
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# Chapter length enforcement
# ---------------------------------------------------------------------------

def check_chapter_length(
    text: str,
    target_words: int,
    min_pct: int = 85,
) -> tuple[int, int, bool]:
    """Check whether a chapter meets the minimum length threshold.

    Returns ``(actual_word_count, min_threshold, is_acceptable)`` where
    *min_threshold* is the minimum word count derived from *target_words*
    and *min_pct*.
    """
    actual = len(text.split())
    min_threshold = (target_words * min_pct) // 100
    return actual, min_threshold, actual >= min_threshold


def expand_chapter(
    text: str,
    *,
    target_words: int,
    min_words: int,
    chapter_num: int,
    title: str,
    max_attempts: int = 2,
) -> tuple[str, int]:
    """Expand an under-length chapter by calling the expansion agent.

    Tries up to *max_attempts* expansion calls.  Returns
    ``(expanded_text, final_word_count)``.  If the expansion agent fails or
    the chapter still doesn't meet the threshold, returns the best result
    achieved so far rather than raising.
    """
    from novelforge.agents.chapter.prompts import build_chapter_expansion_prompt

    current = text
    current_wc = len(current.split())
    for attempt in range(1, max_attempts + 1):
        if current_wc >= min_words:
            break
        logger.info(
            "Chapter %d: expansion attempt %d/%d (%d words, need %d)",
            chapter_num, attempt, max_attempts, current_wc, min_words,
        )
        try:
            expanded = call_llm(
                build_chapter_expansion_prompt(
                    chapter_text=current,
                    current_words=current_wc,
                    target_words=target_words,
                    min_words=min_words,
                ),
                action=f"Chapter {chapter_num}: expansion (attempt {attempt})",
            )
            new_wc = len(expanded.split())
            if new_wc > current_wc:
                current = expanded
                current_wc = new_wc
                logger.info(
                    "Chapter %d: expansion attempt %d produced %d words",
                    chapter_num, attempt, new_wc,
                )
            else:
                logger.warning(
                    "Chapter %d: expansion attempt %d did not increase length (%d → %d)",
                    chapter_num, attempt, current_wc, new_wc,
                )
                break
        except Exception as exc:
            logger.warning(
                "Chapter %d: expansion attempt %d failed: %s: %s",
                chapter_num, attempt, type(exc).__name__, exc,
            )
            break
    return current, current_wc

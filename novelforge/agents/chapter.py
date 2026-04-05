"""
Chapter-level agents: prompt builders, the 12-step agent pipeline,
pre/post-chapter passes, and post-manuscript audit builders.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from novelforge.llm.client import call_llm, parse_llm_json, ChapterTimeoutError, ContentRejectionError, PER_CHAPTER_TIMEOUT
from novelforge.llm.prompts import render_prompt
from novelforge.names import format_name_pool_for_prompt

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

    Parameters
    ----------
    exc:           The exception that caused the failure.
    pass_name:     Human-readable name of the agent pass (e.g. "continuity gatekeeper").
    chapter_num:   Chapter number being processed, when available.
    chapter_title: Chapter title being processed, when available.
    optional:      Whether the pass is optional (True) or required (False).
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


@dataclass
class ChapterContext:
    """Bundles the planning-agent context strings for a single chapter.

    Avoids passing 8+ individual string parameters through the chapter
    pipeline and its callers.
    """
    architecture: str = ""
    timeline: str = ""
    fate: str = ""
    arc: str = ""
    antagonist: str = ""
    technology: str = ""
    theme: str = ""
    pov: str = ""
    gatekeeper_brief: str = ""
    perspective_prompt: str = ""

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


def scan_vocabulary_overuse(chapter_text: str) -> list[str]:
    """
    Scan a chapter for overused vocabulary from the watchlists.

    Returns a list of human-readable warnings for each violation found.
    Pure Python — no LLM call. Fast enough to run after every chapter.
    """
    text_lower = chapter_text.lower()
    warnings: list[str] = []

    # Check hard-banned words
    for word in _FORBIDDEN_WORDS:
        count = text_lower.count(word.lower())
        if count > _HARD_BAN_THRESHOLD:
            warnings.append(
                f'BANNED WORD "{word}" appears {count}x — must be removed entirely'
            )

    # Check soft-limited words
    for word in _SOFT_LIMITED_WORDS:
        count = text_lower.count(word.lower())
        if count > _SOFT_LIMIT_PER_CHAPTER:
            warnings.append(
                f'OVERUSED WORD "{word}" appears {count}x in this chapter '
                f'(limit: {_SOFT_LIMIT_PER_CHAPTER}) — replace most occurrences '
                f'with varied alternatives'
            )

    # Check overused patterns
    for pattern in _OVERUSED_PATTERNS:
        count = text_lower.count(pattern.lower())
        if count > _PATTERN_THRESHOLD:
            warnings.append(
                f'OVERUSED PATTERN "{pattern}" appears {count}x — '
                f'rewrite with a fresh, specific alternative'
            )

    return warnings


def build_perspective_prompt(narrative_perspective: str) -> str:
    """Build a perspective directive string from the session's narrative_perspective value."""
    if not narrative_perspective or narrative_perspective == "third_person":
        return (
            "NARRATIVE PERSPECTIVE: Write this chapter in THIRD PERSON narration. "
            "Use \"he/she/they\" pronouns for all characters. The narrator is omniscient "
            "but should primarily follow the assigned POV character's experience. "
            "Do NOT use first person (\"I/me/my\") for any narration."
        )
    if narrative_perspective.startswith("first_person:"):
        pov_name = narrative_perspective[len("first_person:"):].strip()
        return (
            f"NARRATIVE PERSPECTIVE: Write this chapter in FIRST PERSON narration "
            f"from the perspective of {pov_name}. Use \"I/me/my\" pronouns for {pov_name}. "
            f"Everything must be filtered through {pov_name}'s direct experience — "
            f"the reader can only know what {pov_name} sees, hears, thinks, and feels. "
            f"{pov_name} cannot know other characters' unspoken thoughts or events "
            f"happening outside their presence. Maintain {pov_name}'s unique voice, "
            f"vocabulary, and worldview consistently throughout. "
            f"Do NOT switch to third person or any other character's perspective."
        )
    return ""


def _format_characters(character_list: list[dict]) -> str:
    if not character_list:
        return "No characters defined."
    lines = []
    for ch in character_list:
        lines.append(
            f"- {ch.get('name','?')} (age {ch.get('age','?')}): "
            f"{ch.get('role','?')}. Background: {ch.get('background','')}. "
            f"Arc: {ch.get('arc','')}."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outline / title / characters prompt builders
# ---------------------------------------------------------------------------

def build_title_prompt(premise: str, genre: str) -> list[dict[str, str]]:
    return render_prompt("title", premise=premise, genre=genre)


def build_outline_prompt(
    premise: str, genre: str, chapters: int, word_count: int,
    special_events: str, special_instructions: str,
) -> list[dict[str, str]]:
    return render_prompt(
        "outline", premise=premise, genre=genre, chapters=chapters,
        word_count=f"{word_count:,}", special_events=special_events or "",
        special_instructions=special_instructions or "",
    )


def collect_existing_character_names() -> str:
    """
    Scan existing session files to collect character names from prior novels.

    This is a filesystem helper intended to be called by the caller *before*
    invoking :func:`build_characters_prompt`.  Keeping it separate from the
    prompt builder ensures that prompt-builder functions remain pure
    (data-in / prompt-out) and are not coupled to on-disk state.
    """
    import novelforge.config as config
    from pathlib import Path
    names: set[str] = set()
    novels_dir = Path(config.NOVELS_DIR)
    if not novels_dir.exists():
        return ""
    for f in novels_dir.glob("*.json"):
        if f.name.endswith("_progress.json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for char in data.get("character_list", []):
                if isinstance(char, dict):
                    name = char.get("name", "").strip()
                    if name:
                        names.add(name)
        except Exception:
            continue
    return ", ".join(sorted(names)) if names else ""


def build_characters_prompt(
    premise: str, genre: str, outline_text: str, names_to_avoid: str = "",
) -> list[dict[str, str]]:
    """Build the character-generation prompt.

    Parameters
    ----------
    premise:        Novel premise text.
    genre:          Novel genre string.
    outline_text:   Chapter outline produced by the outline agent.
    names_to_avoid: Comma-separated character names from prior novels that
                    should not be reused.  Obtain this value by calling
                    :func:`collect_existing_character_names` in the caller
                    and passing the result here; do **not** rely on this
                    function to perform filesystem I/O itself.
    """
    name_pool = format_name_pool_for_prompt(genre)
    return render_prompt(
        "characters", premise=premise, genre=genre,
        outline_text=outline_text, names_to_avoid=names_to_avoid,
        name_pool=name_pool,
    )


# ---------------------------------------------------------------------------
# Chapter draft prompt builder
# ---------------------------------------------------------------------------

def build_chapter_draft_prompt(
    premise: str, genre: str, title: str, chapter_num: int, chapter_title: str, chapter_summary: str,
    characters_text: str, previous_summaries: str, target_words: int, special_instructions: str,
    chapter_architecture_context: str = "", chapter_timeline_context: str = "",
    chapter_fate_context: str = "", chapter_arc_context: str = "",
    chapter_antagonist_context: str = "", chapter_technology_context: str = "",
    chapter_theme_context: str = "", gatekeeper_brief: str = "", compression_guidance: str = "",
    chapter_rhythm_shape: str = "", chapter_rhythm_reason: str = "", chapter_pov_context: str = "",
    voice_prompt: str = "", perspective_prompt: str = "",
) -> list[dict[str, str]]:
    return render_prompt(
        "chapter_draft",
        title=title, genre=genre, premise=premise,
        chapter_num=chapter_num, chapter_title=chapter_title,
        chapter_summary=chapter_summary, characters_text=characters_text,
        previous_summaries=previous_summaries or "",
        target_words=f"{target_words:,}",
        special_instructions=special_instructions or "",
        chapter_architecture_context=chapter_architecture_context or "",
        chapter_timeline_context=chapter_timeline_context or "",
        chapter_fate_context=chapter_fate_context or "",
        chapter_arc_context=chapter_arc_context or "",
        chapter_antagonist_context=chapter_antagonist_context or "",
        chapter_technology_context=chapter_technology_context or "",
        chapter_theme_context=chapter_theme_context or "",
        chapter_pov_context=chapter_pov_context or "",
        gatekeeper_brief=gatekeeper_brief or "",
        compression_guidance=compression_guidance or "",
        chapter_rhythm_shape=chapter_rhythm_shape or "",
        chapter_rhythm_reason=chapter_rhythm_reason or "",
        forbidden_words=", ".join(_FORBIDDEN_WORDS),
        soft_limited_words=", ".join(_SOFT_LIMITED_WORDS),
        voice_prompt=voice_prompt or "",
        perspective_prompt=perspective_prompt or "",
    )


# ---------------------------------------------------------------------------
# Chapter refinement agent prompt builders
# ---------------------------------------------------------------------------

def build_prose_refinement_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    return render_prompt("prose_refinement_agent", title=title, chapter_num=chapter_num, chapter_text=chapter_text)


def build_scene_variety_compression_auditor_prompt(chapter_text: str, chapter_summary: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    return render_prompt("scene_variety_compression_auditor", title=title, chapter_num=chapter_num,
                         chapter_summary=chapter_summary, chapter_text=chapter_text)


def run_scene_variety_compression_auditor(chapter_text: str, chapter_summary: str, chapter_num: int, title: str,
                                           degraded_passes: list[dict] | None = None) -> str:
    try:
        return call_llm(
            build_scene_variety_compression_auditor_prompt(
                chapter_text=chapter_text, chapter_summary=chapter_summary,
                chapter_num=chapter_num, title=title,
            ),
            action=f"Chapter {chapter_num}: scene variety & compression audit",
        )
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="scene variety & compression auditor",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "scene variety & compression auditor",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return ""


def build_structure_agent_prompt(chapter_text: str, chapter_num: int, total_chapters: int, outline_summary: str,
                                  chapter_architecture_context: str = "") -> list[dict[str, str]]:
    from novelforge.chapter_position import ChapterPosition
    phase_hint = ChapterPosition(chapter_num, total_chapters).get_structure_phase_hint()
    return render_prompt(
        "structure_agent", chapter_num=chapter_num, total_chapters=total_chapters,
        phase_hint=phase_hint, outline_summary=outline_summary,
        chapter_architecture_context=chapter_architecture_context or "", chapter_text=chapter_text,
    )


def build_character_agent_prompt(chapter_text: str, characters_text: str, chapter_num: int, title: str,
                                  chapter_fate_context: str = "", chapter_arc_context: str = "",
                                  chapter_antagonist_context: str = "", chapter_pov_context: str = "",
                                  perspective_prompt: str = "") -> list[dict[str, str]]:
    return render_prompt(
        "character_agent", title=title, chapter_num=chapter_num,
        characters_text=characters_text,
        chapter_fate_context=chapter_fate_context or "",
        chapter_arc_context=chapter_arc_context or "",
        chapter_antagonist_context=chapter_antagonist_context or "",
        chapter_pov_context=chapter_pov_context or "",
        perspective_prompt=perspective_prompt or "",
        chapter_text=chapter_text,
    )


def build_context_analyzer_prompt(chapter_text: str, previous_summaries: str, chapter_num: int, title: str,
                                   chapter_timeline_context: str = "", chapter_technology_context: str = "",
                                   chapter_theme_context: str = "", gatekeeper_brief: str = "") -> list[dict[str, str]]:
    return render_prompt(
        "context_analyzer", title=title, chapter_num=chapter_num,
        previous_summaries=previous_summaries or "",
        chapter_timeline_context=chapter_timeline_context or "",
        chapter_technology_context=chapter_technology_context or "",
        chapter_theme_context=chapter_theme_context or "",
        gatekeeper_brief=gatekeeper_brief or "",
        chapter_text=chapter_text,
    )


def build_synthesizer_prompt(chapter_text: str, chapter_num: int, title: str, genre: str,
                              perspective_prompt: str = "") -> list[dict[str, str]]:
    return render_prompt("synthesizer", title=title, genre=genre, chapter_num=chapter_num,
                         perspective_prompt=perspective_prompt or "", chapter_text=chapter_text)


def build_quality_controller_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    return render_prompt("quality_controller", title=title, chapter_num=chapter_num, chapter_text=chapter_text)


def build_editing_agent_prompt(chapter_text: str, chapter_summary: str, chapter_num: int, title: str, scene_audit_directives: str = "") -> list[dict[str, str]]:
    return render_prompt(
        "editing_agent", title=title, chapter_num=chapter_num,
        chapter_summary=chapter_summary,
        scene_audit_directives=scene_audit_directives or "",
        chapter_text=chapter_text,
    )


def build_narrative_momentum_distinctiveness_prompt(
    chapter_text: str, previous_summaries: str, chapter_summary: str, chapter_num: int, title: str, total_chapters: int,
) -> list[dict[str, str]]:
    from novelforge.chapter_position import ChapterPosition
    escalation_target = ChapterPosition(chapter_num, total_chapters).get_escalation_target()
    return render_prompt(
        "narrative_momentum_distinctiveness", title=title, chapter_num=chapter_num,
        total_chapters=total_chapters, escalation_target=escalation_target,
        chapter_summary=chapter_summary, previous_summaries=previous_summaries or "",
        chapter_text=chapter_text,
    )


def build_operational_distinctiveness_prompt(chapter_text: str, previous_summaries: str, chapter_summary: str,
                                              chapter_num: int, title: str) -> list[dict[str, str]]:
    return render_prompt(
        "operational_distinctiveness", title=title, chapter_num=chapter_num,
        chapter_summary=chapter_summary, previous_summaries=previous_summaries or "",
        chapter_text=chapter_text,
    )


def build_polish_agent_prompt(chapter_text: str, chapter_num: int, title: str, genre: str) -> list[dict[str, str]]:
    return render_prompt("polish_agent", title=title, genre=genre, chapter_num=chapter_num, chapter_text=chapter_text)


def build_anti_llm_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    return render_prompt(
        "anti_llm_agent", title=title, chapter_num=chapter_num,
        chapter_text=chapter_text, forbidden_words=", ".join(_FORBIDDEN_WORDS),
        soft_limited_words=", ".join(_SOFT_LIMITED_WORDS),
    )


def build_vocabulary_fix_prompt(
    chapter_text: str, chapter_num: int, title: str, violations: list[str],
) -> list[dict[str, str]]:
    """Build a targeted fix-up prompt for vocabulary violations found by the scanner."""
    violation_block = "\n".join(f"- {v}" for v in violations)
    return [
        {
            "role": "system",
            "content": (
                "You are a prose editor. Your ONLY task is to fix specific vocabulary "
                "problems listed below. For each flagged word or pattern, replace it "
                "with a fresh, context-appropriate alternative. Do NOT change plot, "
                "characters, meaning, or sentence structure beyond what's needed for "
                "the replacement. Return the full revised chapter text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Novel: '{title}' — Chapter {chapter_num}\n\n"
                f"The following vocabulary problems were detected:\n{violation_block}\n\n"
                f"Fix ONLY these specific problems. Replace each flagged word or pattern "
                f"with a varied, natural alternative that fits the context.\n\n"
                f"***Return ONLY the complete revised chapter text with NO introduction, "
                f"NO explanation, NO markdown.***\n\n"
                f"{chapter_text}"
            ),
        },
    ]


def build_chapter_summary_prompt(chapter_text: str, chapter_num: int) -> list[dict[str, str]]:
    return render_prompt("chapter_summary", chapter_num=chapter_num, chapter_text=chapter_text)


# ---------------------------------------------------------------------------
# Post-chapter passes
# ---------------------------------------------------------------------------

def build_per_chapter_compression_check_prompt(chapter_num: int, chapter_summary: str, previous_summaries: str, title: str) -> list[dict[str, str]]:
    return render_prompt(
        "per_chapter_compression_check", title=title, chapter_num=chapter_num,
        chapter_summary=chapter_summary, previous_summaries=previous_summaries,
    )


def run_per_chapter_compression_check(chapter_num: int, chapter_summary: str, previous_summaries: str, title: str,
                                       degraded_passes: list[dict] | None = None) -> str:
    if chapter_num <= 1 or not previous_summaries.strip():
        return ""
    try:
        return call_llm(
            build_per_chapter_compression_check_prompt(
                chapter_num=chapter_num, chapter_summary=chapter_summary,
                previous_summaries=previous_summaries, title=title,
            ),
            action=f"Running Per-Chapter Compression Check for Chapter {chapter_num}"
        )
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="per-chapter compression check",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "per-chapter compression check",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return ""


def build_character_state_updater_prompt(chapter_text: str, chapter_summary: str, characters_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    return render_prompt(
        "character_state_updater", title=title, chapter_num=chapter_num,
        characters_text=characters_text, chapter_summary=chapter_summary,
        chapter_text=chapter_text,
    )


def run_character_state_updater(chapter_text: str, chapter_summary: str, characters_text: str, chapter_num: int, title: str,
                                 degraded_passes: list[dict] | None = None) -> str:
    try:
        return call_llm(
            build_character_state_updater_prompt(
                chapter_text=chapter_text, chapter_summary=chapter_summary,
                characters_text=characters_text, chapter_num=chapter_num, title=title,
            ),
            action=f"Running Character State Updater for Chapter {chapter_num}"
        )
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="character state updater",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "character state updater",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return ""


# ---------------------------------------------------------------------------
# Pre-chapter passes
# ---------------------------------------------------------------------------

def build_continuity_gatekeeper_prompt(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str,
    chapter_timeline_context: str = "", chapter_fate_context: str = "",
    chapter_arc_context: str = "", character_state_log: str = "",
) -> list[dict[str, str]]:
    return render_prompt(
        "continuity_gatekeeper",
        chapter_num=chapter_num, chapter_title=chapter_title,
        chapter_summary=chapter_summary,
        previous_summaries=previous_summaries or "",
        chapter_timeline_context=chapter_timeline_context or "",
        chapter_fate_context=chapter_fate_context or "",
        chapter_arc_context=chapter_arc_context or "",
        character_state_log=character_state_log or "",
    )


def run_continuity_gatekeeper(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str,
    chapter_timeline_context: str = "", chapter_fate_context: str = "",
    chapter_arc_context: str = "", character_state_log: str = "",
    degraded_passes: list[dict] | None = None,
) -> str:
    try:
        return call_llm(
            build_continuity_gatekeeper_prompt(
                chapter_num=chapter_num, chapter_title=chapter_title,
                chapter_summary=chapter_summary, previous_summaries=previous_summaries,
                chapter_timeline_context=chapter_timeline_context,
                chapter_fate_context=chapter_fate_context,
                chapter_arc_context=chapter_arc_context,
                character_state_log=character_state_log,
            ),
            action=f"Running Continuity Gatekeeper for Chapter {chapter_num}",
        )
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="continuity gatekeeper",
            chapter_num=chapter_num, chapter_title=chapter_title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "continuity gatekeeper",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return ""


def build_chapter_rhythm_classifier_prompt(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str, title: str,
    chapter_architecture_context: str = "",
) -> list[dict[str, str]]:
    return render_prompt(
        "chapter_rhythm_classifier", title=title, chapter_num=chapter_num,
        chapter_title=chapter_title, chapter_summary=chapter_summary,
        previous_summaries=previous_summaries or "",
        chapter_architecture_context=chapter_architecture_context or "",
    )


def run_chapter_rhythm_classifier(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str, title: str,
    chapter_architecture_context: str = "",
    degraded_passes: list[dict] | None = None,
) -> dict:
    try:
        raw = call_llm(
            build_chapter_rhythm_classifier_prompt(
                chapter_num=chapter_num, chapter_title=chapter_title,
                chapter_summary=chapter_summary, previous_summaries=previous_summaries,
                title=title, chapter_architecture_context=chapter_architecture_context,
            ),
            action=f"Classifying chapter rhythm for Chapter {chapter_num}",
            json_mode=True,
        )
        return parse_llm_json(raw)
    except Exception as exc:
        failure_summary = _log_pass_failure(
            exc, pass_name="chapter rhythm classifier",
            chapter_num=chapter_num, chapter_title=title, optional=True,
        )
        if degraded_passes is not None:
            degraded_passes.append({
                "pass_name": "chapter rhythm classifier",
                "chapter_num": chapter_num,
                "failure_summary": failure_summary,
            })
        return {PASS_FAILURE_KEY: failure_summary}


# ---------------------------------------------------------------------------
# Post-manuscript audit prompt builders
# ---------------------------------------------------------------------------

def build_chapter_revision_prompt(
    chapter_text: str, chapter_num: int, title: str, chapter_outline_summary: str, revision_instructions: str,
    chapter_architecture_context: str = "", chapter_timeline_context: str = "",
    chapter_fate_context: str = "", chapter_arc_context: str = "",
    chapter_antagonist_context: str = "", chapter_technology_context: str = "",
    chapter_theme_context: str = "", gatekeeper_brief: str = "",
    perspective_prompt: str = "",
) -> list[dict[str, str]]:
    return render_prompt(
        "chapter_revision", title=title, chapter_num=chapter_num,
        chapter_outline_summary=chapter_outline_summary,
        revision_instructions=revision_instructions,
        chapter_architecture_context=chapter_architecture_context or "",
        chapter_timeline_context=chapter_timeline_context or "",
        chapter_fate_context=chapter_fate_context or "",
        chapter_arc_context=chapter_arc_context or "",
        chapter_antagonist_context=chapter_antagonist_context or "",
        chapter_technology_context=chapter_technology_context or "",
        chapter_theme_context=chapter_theme_context or "",
        gatekeeper_brief=gatekeeper_brief or "",
        perspective_prompt=perspective_prompt or "",
        chapter_text=chapter_text,
    )


def build_consistency_pass_prompt(title: str, all_summaries: list[str], special_instructions: str) -> list[dict[str, str]]:
    summaries_text = "\n\n".join(
        f"Chapter {i+1}:\n{s}" for i, s in enumerate(all_summaries)
    )
    return render_prompt("consistency_pass", title=title, summaries_text=summaries_text,
                         special_instructions=special_instructions)


def build_global_continuity_auditor_prompt(title: str, all_summaries: list[str], character_state_log: list[str],
                                            master_timeline: dict, character_fate_registry: dict) -> list[dict[str, str]]:
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))

    state_log_text = (
        "\n\n".join(character_state_log)
        if character_state_log else "No character state log available."
    )

    timeline_lines: list[str] = []
    if isinstance(master_timeline, dict):
        for event in master_timeline.get("ledger", []):
            if isinstance(event, dict):
                timeline_lines.append(
                    f"  Ch {event.get('chapter', '?')}: {event.get('event', '')} "
                    f"[{event.get('event_type', 'other')}]"
                )
    timeline_text = "\n".join(timeline_lines) if timeline_lines else "No master timeline available."

    registry_lines: list[str] = []
    if isinstance(character_fate_registry, dict):
        for entry in character_fate_registry.get("registry", []):
            if isinstance(entry, dict):
                name = entry.get("character", "?")
                status = entry.get("current_status", "unknown")
                outcome = entry.get("definitive_outcome", "unknown")
                death_ch = entry.get("death_chapter")
                registry_lines.append(
                    f"  {name}: status={status}, outcome={outcome}"
                    + (f", death_chapter={death_ch}" if death_ch else "")
                )
    registry_text = "\n".join(registry_lines) if registry_lines else "No fate registry available."

    return render_prompt("global_continuity_auditor", title=title, summaries_text=summaries_text,
                         state_log_text=state_log_text, timeline_text=timeline_text,
                         registry_text=registry_text)


def build_narrative_compression_editor_prompt(title: str, all_summaries: list[str], continuity_audit: dict | None = None) -> list[dict[str, str]]:
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))

    audit_section = ""
    if continuity_audit and isinstance(continuity_audit, dict):
        contradictions = continuity_audit.get("contradictions", [])
        if contradictions:
            audit_lines = [
                f"  - Chapters {c.get('chapters', [])}: {c.get('description', '')}"
                for c in contradictions if isinstance(c, dict)
            ]
            if audit_lines:
                audit_section = (
                    "\n=== CONTINUITY AUDIT FLAGS (already identified) ===\n"
                    + "\n".join(audit_lines) + "\n"
                )

    return render_prompt("narrative_compression_editor", title=title, summaries_text=summaries_text,
                         audit_section=audit_section)


def build_character_resolution_validator_prompt(title: str, all_summaries: list[str], character_arc_plan: dict,
                                                 character_fate_registry: dict, character_state_log: list[str]) -> list[dict[str, str]]:
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))

    arc_lines: list[str] = []
    if isinstance(character_arc_plan, dict):
        for arc in character_arc_plan.get("arcs", []):
            if isinstance(arc, dict):
                arc_lines.append(
                    f"  {arc.get('character', '?')}: "
                    f"start='{arc.get('start_state', '')}' \u2192 "
                    f"midpoint='{arc.get('midpoint_transformation', '')}' \u2192 "
                    f"crisis='{arc.get('crisis_point', '')}' \u2192 "
                    f"final_choice='{arc.get('final_moral_choice', '')}'"
                )
    arc_text = "\n".join(arc_lines) if arc_lines else "No character arc plan available."

    registry_lines: list[str] = []
    if isinstance(character_fate_registry, dict):
        for entry in character_fate_registry.get("registry", []):
            if isinstance(entry, dict):
                name = entry.get("character", "?")
                outcome = entry.get("definitive_outcome", "unknown")
                locked = entry.get("outcome_locked", False)
                registry_lines.append(f"  {name}: required_outcome={outcome}, locked={locked}")
    registry_text = "\n".join(registry_lines) if registry_lines else "No fate registry available."

    state_log_text = "\n\n".join(character_state_log) if character_state_log else "No character state log available."

    return render_prompt("character_resolution_validator", title=title, arc_text=arc_text,
                         registry_text=registry_text, state_log_text=state_log_text,
                         summaries_text=summaries_text)


def build_thematic_payoff_analyzer_prompt(title: str, all_summaries: list[str], theme_reinforcement: dict, total_chapters: int) -> list[dict[str, str]]:
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))

    theme_lines: list[str] = []
    if isinstance(theme_reinforcement, dict):
        for theme in theme_reinforcement.get("themes", []):
            if not isinstance(theme, dict):
                continue
            name = theme.get("name", "?")
            desc = theme.get("description", "")
            appearances = theme.get("chapter_appearances", [])
            final_chapters = [ap.get("chapter") for ap in appearances if isinstance(ap, dict)]
            theme_lines.append(
                f"  Theme '{name}': {desc} | planned appearances: chapters {final_chapters}"
            )
        global_arcs = theme_reinforcement.get("global_thematic_arcs", [])
        if isinstance(global_arcs, list) and global_arcs:
            theme_lines.append("  Global thematic arcs: " + "; ".join(str(a) for a in global_arcs))
    theme_text = "\n".join(theme_lines) if theme_lines else "No theme reinforcement plan available."

    final_quarter_start = max(1, round(total_chapters * 0.75))

    return render_prompt("thematic_payoff_analyzer", title=title, total_chapters=total_chapters,
                         final_quarter_start=final_quarter_start, theme_text=theme_text,
                         summaries_text=summaries_text)


def build_climax_integrity_checker_prompt(title: str, all_summaries: list[str], character_arc_plan: dict, total_chapters: int) -> list[dict[str, str]]:
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))

    arc_lines: list[str] = []
    if isinstance(character_arc_plan, dict):
        for arc in character_arc_plan.get("arcs", []):
            if not isinstance(arc, dict):
                continue
            role = str(arc.get("role", "")).lower()
            if "protagonist" in role or "lead" in role or not role:
                arc_lines.append(
                    f"  {arc.get('character', '?')}: "
                    f"start='{arc.get('start_state', '')}' -> "
                    f"final_moral_choice='{arc.get('final_moral_choice', '')}'"
                )
    if not arc_lines and isinstance(character_arc_plan, dict):
        for arc in character_arc_plan.get("arcs", [])[:2]:
            if isinstance(arc, dict):
                arc_lines.append(
                    f"  {arc.get('character', '?')}: "
                    f"start='{arc.get('start_state', '')}' -> "
                    f"final_moral_choice='{arc.get('final_moral_choice', '')}'"
                )
    arc_text = "\n".join(arc_lines) if arc_lines else "No character arc plan available."

    climax_start = max(1, round(total_chapters * 0.85))

    return render_prompt("climax_integrity_checker", title=title, total_chapters=total_chapters,
                         climax_start=climax_start, arc_text=arc_text, summaries_text=summaries_text)


def build_loose_thread_resolver_prompt(title: str, all_summaries: list[str], character_state_log: list[str],
                                        continuity_audit: dict | None = None, resolution_report: dict | None = None) -> list[dict[str, str]]:
    summaries_block = "\n".join(
        f"Chapter {i + 1}: {s}" for i, s in enumerate(all_summaries)
    ) or "No chapter summaries available."

    state_block = "\n".join(character_state_log) if character_state_log else "No character state log available."

    audit_issues: list[str] = []
    if continuity_audit:
        for field in ("contradictions", "character_state_errors", "timeline_errors"):
            items = continuity_audit.get(field, [])
            if isinstance(items, list):
                audit_issues.extend(items)
    audit_block = "\n".join(f"- {x}" for x in audit_issues) if audit_issues else "No continuity issues flagged."

    unresolved_chars: list[str] = []
    if resolution_report:
        raw = resolution_report.get("unresolved_characters", [])
        if isinstance(raw, list):
            unresolved_chars = [str(x) for x in raw]
    unresolved_block = "\n".join(f"- {c}" for c in unresolved_chars) if unresolved_chars else "No unresolved characters flagged."

    return render_prompt("loose_thread_resolver", title=title, summaries_block=summaries_block,
                         state_block=state_block, audit_block=audit_block, unresolved_block=unresolved_block)


def build_reader_immersion_tester_prompt(title: str, all_summaries: list[str], character_arc_plan: dict | None = None, thematic_report: dict | None = None) -> list[dict[str, str]]:
    summaries_block = "\n".join(
        f"Chapter {i + 1}: {s}" for i, s in enumerate(all_summaries)
    ) or "No chapter summaries available."

    arc_lines: list[str] = []
    if isinstance(character_arc_plan, dict):
        for arc in character_arc_plan.get("arcs", []):
            name = arc.get("character", "Unknown")
            start = arc.get("start_state", "")
            end = arc.get("final_state", "")
            arc_lines.append(f"- {name}: {start} \u2192 {end}")
    arc_block = "\n".join(arc_lines) if arc_lines else "No character arc plan available."

    theme_lines: list[str] = []
    if isinstance(thematic_report, dict):
        for item in thematic_report.get("themes", []):
            if isinstance(item, dict):
                theme_lines.append(f"- {item.get('theme', item)}: payoff={item.get('payoff_present', '?')}")
            else:
                theme_lines.append(f"- {item}")
    theme_block = "\n".join(theme_lines) if theme_lines else "No thematic payoff data available."

    return render_prompt("reader_immersion_tester", title=title, summaries_block=summaries_block,
                         arc_block=arc_block, theme_block=theme_block)


def build_pacing_tension_heatmap_prompt(title: str, all_summaries: list[str], total_chapters: int) -> list[dict[str, str]]:
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))
    return render_prompt("pacing_tension_heatmap", title=title, total_chapters=total_chapters,
                         summaries_text=summaries_text)


def build_character_relationship_prompt(
    title: str, genre: str, character_list: list[dict], all_summaries: list[str],
) -> list[dict[str, str]]:
    characters_text = "\n".join(
        f"- {c.get('name', '?')}: role={c.get('role', '')}; background={c.get('background', '')}; arc={c.get('arc', '')}"
        for c in character_list
    )
    if not characters_text.strip():
        characters_text = "- No explicit characters provided."
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))
    return render_prompt(
        "character_relationship_mapper", title=title, genre=genre,
        characters_text=characters_text, summaries_text=summaries_text,
    )


def build_illustration_prompt_generator_prompt(title: str, genre: str, premise: str, character_list: list[dict], all_summaries: list[str]) -> list[dict[str, str]]:
    characters_text = "\n".join(
        f"- {c.get('name', '?')}: role={c.get('role', '')}; background={c.get('background', '')}"
        for c in character_list
    )
    if not characters_text.strip():
        characters_text = "- No explicit characters provided."
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))
    return render_prompt("illustration_prompt_generator", title=title, genre=genre,
                         premise=premise, characters_text=characters_text, summaries_text=summaries_text)


# ---------------------------------------------------------------------------
# Main chapter agent pipeline
# ---------------------------------------------------------------------------

def _run_all_chapter_agents(
    text: str,
    chapter_num: int,
    title: str,
    genre: str,
    total_chapters: int,
    chapter_outline_summary: str,
    characters_text: str,
    previous_summaries: str,
    ctx: ChapterContext | None = None,
    step_callback: Callable[[str], None] | None = None,
    deadline: float = 0,
    degraded_passes: list[dict] | None = None,
) -> tuple[str, str]:
    """
    Run all chapter refinement agents (post-draft) and return:
    (final_chapter_text, continuity_summary)

    Planning-agent context is passed via a ``ChapterContext`` dataclass.

    If *deadline* is non-zero (a ``time.monotonic()`` timestamp), each step
    checks the clock before calling the LLM and raises ``ChapterTimeoutError``
    if the deadline has passed.

    Optional pass failures are logged and, when *degraded_passes* is provided,
    appended to that list so the caller can surface them in progress metadata.
    """
    if ctx is None:
        ctx = ChapterContext()
    def _check_deadline() -> None:
        if deadline and time.monotonic() > deadline:
            raise ChapterTimeoutError(
                f"Chapter {chapter_num} exceeded the {PER_CHAPTER_TIMEOUT // 60}-minute time limit."
            )

    # Local shorthand: every agent call goes through the content-retry wrapper
    def _safe(build_msgs: Callable[[str], list[dict]], txt: str, *, action: str, json_mode: bool = False) -> str:
        return _call_with_content_retry(
            build_msgs, txt, action=action,
            chapter_num=chapter_num, title=title, json_mode=json_mode,
        )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: prose refinement (dialogue + scenes)")
    text = _safe(
        lambda t: build_prose_refinement_agent_prompt(t, chapter_num, title),
        text, action=f"Chapter {chapter_num}: prose refinement",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: scene variety & compression audit")
    scene_audit_directives = run_scene_variety_compression_auditor(
        chapter_text=text, chapter_summary=chapter_outline_summary,
        chapter_num=chapter_num, title=title,
        degraded_passes=degraded_passes,
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: verifying continuity")
    text = _safe(
        lambda t: build_context_analyzer_prompt(
            t, previous_summaries, chapter_num, title,
            ctx.timeline, ctx.technology, ctx.theme,
        ),
        text, action=f"Chapter {chapter_num}: verifying continuity",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: editing")
    text = _safe(
        lambda t: build_editing_agent_prompt(t, chapter_outline_summary, chapter_num, title, scene_audit_directives),
        text, action=f"Chapter {chapter_num}: editing",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: momentum & distinctiveness check")
    text = _safe(
        lambda t: build_narrative_momentum_distinctiveness_prompt(
            t, previous_summaries, chapter_outline_summary, chapter_num, title, total_chapters,
        ),
        text, action=f"Chapter {chapter_num}: momentum & distinctiveness",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: checking structure")
    text = _safe(
        lambda t: build_structure_agent_prompt(
            t, chapter_num, total_chapters, chapter_outline_summary, ctx.architecture,
        ),
        text, action=f"Chapter {chapter_num}: checking structure",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: verifying operational distinctiveness")
    text = _safe(
        lambda t: build_operational_distinctiveness_prompt(
            t, previous_summaries, chapter_outline_summary, chapter_num, title,
        ),
        text, action=f"Chapter {chapter_num}: verifying operational distinctiveness",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: deepening characters")
    text = _safe(
        lambda t: build_character_agent_prompt(
            t, characters_text, chapter_num, title,
            ctx.fate, ctx.arc, ctx.antagonist, ctx.pov, ctx.perspective_prompt,
        ),
        text, action=f"Chapter {chapter_num}: deepening characters",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: synthesizing")
    text = _safe(
        lambda t: build_synthesizer_prompt(t, chapter_num, title, genre, ctx.perspective_prompt),
        text, action=f"Chapter {chapter_num}: synthesizing",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: polishing")
    text = _safe(
        lambda t: build_polish_agent_prompt(t, chapter_num, title, genre),
        text, action=f"Chapter {chapter_num}: polishing",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: anti-LLM pass")
    text = _safe(
        lambda t: build_anti_llm_agent_prompt(t, chapter_num, title),
        text, action=f"Chapter {chapter_num}: anti-LLM pass",
    )

    # Vocabulary diversity scan — pure Python, no LLM call
    _check_deadline()
    violations = scan_vocabulary_overuse(text)
    if violations:
        if step_callback:
            step_callback(f"Chapter {chapter_num}: fixing {len(violations)} vocabulary issues")
        logger.info("Chapter %d: vocabulary scan found %d violations", chapter_num, len(violations))
        text = _safe(
            lambda t: build_vocabulary_fix_prompt(t, chapter_num, title, violations),
            text, action=f"Chapter {chapter_num}: vocabulary fix-up",
        )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: quality control")
    text = _safe(
        lambda t: build_quality_controller_prompt(t, chapter_num, title),
        text, action=f"Chapter {chapter_num}: quality control",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: summarising")
    summary = _safe(
        lambda t: build_chapter_summary_prompt(t, chapter_num),
        text, action=f"Chapter {chapter_num}: summarising",
    )

    return text, summary

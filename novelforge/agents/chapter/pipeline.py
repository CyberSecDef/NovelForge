"""Chapter agent pipeline and runner functions that wrap prompt builders with LLM calls."""

import logging
import time
from collections.abc import Callable

from novelforge.llm.client import call_llm, parse_llm_json, ChapterTimeoutError, PER_CHAPTER_TIMEOUT

from novelforge.agents.chapter._helpers import (
    PASS_FAILURE_KEY,
    _call_with_content_retry,
    _log_pass_failure,
    scan_vocabulary_overuse,
)
from novelforge.agents.chapter.context import ChapterContext
from novelforge.agents.chapter.prompts import (
    build_anti_llm_agent_prompt,
    build_chapter_rhythm_classifier_prompt,
    build_chapter_summary_prompt,
    build_character_agent_prompt,
    build_character_state_updater_prompt,
    build_context_analyzer_prompt,
    build_continuity_gatekeeper_prompt,
    build_copy_edit_prompt,
    build_editing_agent_prompt,
    build_human_oddities_prompt,
    build_metaphor_reduction_prompt,
    build_narrative_momentum_distinctiveness_prompt,
    build_operational_distinctiveness_prompt,
    build_per_chapter_compression_check_prompt,
    build_polish_agent_prompt,
    build_prose_refinement_agent_prompt,
    build_quality_controller_prompt,
    build_scene_variety_compression_auditor_prompt,
    build_structure_agent_prompt,
    build_synthesizer_prompt,
    build_vocabulary_fix_prompt,
    build_voice_dialogue_differentiation_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runner functions (prompt + LLM call + error handling)
# ---------------------------------------------------------------------------

def run_scene_variety_compression_auditor(chapter_text: str, chapter_summary: str, chapter_num: int, title: str,
                                           degraded_passes: list[dict] | None = None) -> str:
    """Run the scene variety audit, returning directives or empty string on failure."""
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


def run_per_chapter_compression_check(chapter_num: int, chapter_summary: str, previous_summaries: str, title: str,
                                       degraded_passes: list[dict] | None = None) -> str:
    """Run the post-chapter compression check, returning guidance or empty string on failure."""
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


def run_character_state_updater(chapter_text: str, chapter_summary: str, characters_text: str, chapter_num: int, title: str,
                                 degraded_passes: list[dict] | None = None) -> str:
    """Run the character state updater, returning state changes or empty string on failure."""
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


def run_continuity_gatekeeper(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str,
    chapter_timeline_context: str = "", chapter_fate_context: str = "",
    chapter_arc_context: str = "", character_state_log: str = "",
    degraded_passes: list[dict] | None = None,
) -> str:
    """Run the pre-chapter continuity gatekeeper, returning a brief or empty string on failure."""
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


def run_chapter_rhythm_classifier(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str, title: str,
    chapter_architecture_context: str = "",
    degraded_passes: list[dict] | None = None,
) -> dict:
    """Run the chapter rhythm classifier, returning a dict with shape recommendation or PASS_FAILURE_KEY on failure."""
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
        result = parse_llm_json(raw)
        if not isinstance(result, dict):
            raise ValueError("LLM returned a JSON array instead of an object")
        return result
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
        return {
            PASS_FAILURE_KEY: failure_summary,
            "recommended_shape_for_this_chapter": "",
            "recommendation_reason": "",
        }


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
        """Raise ChapterTimeoutError if the per-chapter deadline has passed."""
        if deadline and time.monotonic() > deadline:
            raise ChapterTimeoutError(
                f"Chapter {chapter_num} exceeded the {PER_CHAPTER_TIMEOUT // 60}-minute time limit."
            )

    # Local shorthand: every agent call goes through the content-retry wrapper
    def _safe(build_msgs: Callable[[str], list[dict]], txt: str, *, action: str, json_mode: bool = False) -> str:
        """Call the LLM via the content-retry wrapper."""
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
        step_callback(f"Chapter {chapter_num}: voice & dialogue differentiation")
    text = _safe(
        lambda t: build_voice_dialogue_differentiation_prompt(
            t, chapter_num, title, characters_text, ctx.perspective_prompt,
        ),
        text, action=f"Chapter {chapter_num}: voice & dialogue differentiation",
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
        step_callback(f"Chapter {chapter_num}: human oddities")
    text = _safe(
        lambda t: build_human_oddities_prompt(
            t, chapter_num, title, total_chapters, characters_text,
        ),
        text, action=f"Chapter {chapter_num}: human oddities",
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

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: metaphor reduction")
    text = _safe(
        lambda t: build_metaphor_reduction_prompt(t, chapter_num, title),
        text, action=f"Chapter {chapter_num}: metaphor reduction",
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
        step_callback(f"Chapter {chapter_num}: copy edit")
    text = _safe(
        lambda t: build_copy_edit_prompt(t, chapter_num, title),
        text, action=f"Chapter {chapter_num}: copy edit",
    )

    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: summarising")
    summary = _safe(
        lambda t: build_chapter_summary_prompt(t, chapter_num),
        text, action=f"Chapter {chapter_num}: summarising",
    )

    return text, summary

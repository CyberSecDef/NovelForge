"""Chapter generation endpoint, background worker, and progress routes."""

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, cast

import requests
from flask import Response, jsonify, request, session

from novelforge import limiter
import novelforge.config as config
from novelforge.progress import (
    progress_manager,
    set_correlation_token, clear_correlation_token,
)
from novelforge.llm.client import (
    call_llm, parse_llm_json, friendly_llm_error,
    reset_llm_usage, get_llm_usage,
    CircuitBreakerError, ChapterTimeoutError, ContentRejectionError,
    AllProvidersExhaustedError, PER_CHAPTER_TIMEOUT,
)
from novelforge.agents.planning import (
    normalise_story_architecture, normalise_master_timeline,
    normalise_character_fate_registry, normalise_character_arc_plan,
    normalise_antagonist_motivation_plan, normalise_technology_rules,
    normalise_theme_reinforcement, normalise_pov_focal_character_plan,
    get_chapter_architecture_context, get_chapter_timeline_context,
    get_chapter_fate_context, get_chapter_arc_context,
    get_chapter_antagonist_context, get_chapter_technology_context,
    get_chapter_theme_context, get_chapter_pov_context,
)
from novelforge.agents.chapter import (
    _format_characters,
    _draft_with_content_retry,
    build_chapter_draft_prompt,
    run_continuity_gatekeeper, run_chapter_rhythm_classifier,
    run_character_state_updater, run_per_chapter_compression_check,
    _run_all_chapter_agents, ChapterContext,
)
from novelforge.session.persistence import (
    get_session_id, save_session_state, persist_completed_chapters,
)
from novelforge.routes.generation._shared import (
    generation_bp, _PROGRESS_PERSIST_INTERVAL,
)
from novelforge.routes.generation.audits import run_post_manuscript_audits

logger = logging.getLogger(__name__)


@generation_bp.route("/generate_chapters", methods=["POST"])
@limiter.limit("1 per 10 minutes")
def generate_chapters() -> Response | tuple[Response, int]:
    """
    Phase 2: Start async chapter generation.
    Returns a progress token to poll via /progress/<token>.
    """
    for key in ("premise", "genre", "chapters", "word_count", "title", "chapter_list"):
        if key not in session:
            return jsonify({"error": "Session data missing. Please start over."}), 400

    token = str(uuid.uuid4())
    snapshot = {
        "session_id": get_session_id(),
        "premise": session["premise"],
        "genre": session["genre"],
        "chapters": session["chapters"],
        "word_count": session["word_count"],
        "special_instructions": session.get("special_instructions", ""),
        "title": session["title"],
        "chapter_list": session["chapter_list"],
        "character_list": session.get("character_list", []),
        "story_architecture": session.get("story_architecture", {}),
        "master_timeline": session.get("master_timeline", {}),
        "character_fate_registry": session.get("character_fate_registry", {}),
        "character_arc_plan": session.get("character_arc_plan", {}),
        "antagonist_motivation_plan": session.get("antagonist_motivation_plan", {}),
        "technology_rules": session.get("technology_rules", {}),
        "theme_reinforcement": session.get("theme_reinforcement", {}),
        "pov_focal_character_plan": session.get("pov_focal_character_plan", {}),
        "voice_seed": session.get("voice_seed", {}),
        "narrative_perspective": session.get("narrative_perspective", "third_person"),
    }

    # Atomic check-and-create: rejects if an existing token is still running
    duplicate_token = progress_manager.create_unless_running(
        existing_token=session.get("progress_token"),
        new_token=token,
        initial_state={
            "status": "running",
            "current": 0,
            "total": session["chapters"],
            "step": "Preparing\u2026",
            "chapters_done": [],
            "error": None,
            "_live": True,
            "snapshot": snapshot,
        },
    )
    if duplicate_token:
        return jsonify({
            "error": "A chapter generation is already in progress for this session. Please wait for it to complete.",
            "error_code": "generation_in_progress",
            "token": duplicate_token,
        }), 409

    thread = threading.Thread(
        target=_run_chapter_generation,
        args=(token, snapshot),
        daemon=True,
    )
    thread.start()

    session["progress_token"] = token
    save_session_state()

    return jsonify({"token": token})


def _resume_chapter_generation(
    token: str, snap: dict, chapters_done: list[dict],
    summaries: list[str], start_idx: int,
    character_state_log: list[str] | None = None,
) -> None:
    """Resume chapter generation from a specific chapter index after a crash."""
    _run_chapter_generation_internal(
        token, snap, chapters_done, summaries, start_idx,
        character_state_log=character_state_log,
    )


def _run_chapter_generation(token: str, snap: dict) -> None:
    """
    Background worker: generate all chapters sequentially using the full
    agent pipeline.
    """
    _run_chapter_generation_internal(token, snap, [], [], 0)


def _run_chapter_generation_internal(
    token: str,
    snap: dict,
    chapters_done: list[dict],
    summaries: list[str],
    start_idx: int,
    character_state_log: list[str] | None = None,
) -> None:
    """
    Internal function that performs the actual chapter generation.
    Can start from any chapter index to support resume functionality.
    """
    premise = snap["premise"]
    genre = snap["genre"]
    total_chapters = snap["chapters"]
    word_count = snap["word_count"]
    special_instructions = snap["special_instructions"]
    title = snap["title"]
    chapter_list = snap["chapter_list"]
    character_list = snap["character_list"]
    story_architecture = normalise_story_architecture(
        snap.get("story_architecture", {}), chapter_list, total_chapters,
    )
    master_timeline = normalise_master_timeline(
        snap.get("master_timeline", {}), chapter_list, character_list,
    )
    character_fate_registry = normalise_character_fate_registry(
        snap.get("character_fate_registry", {}), character_list, total_chapters,
    )
    character_arc_plan = normalise_character_arc_plan(
        snap.get("character_arc_plan", {}), character_list, chapter_list,
    )
    antagonist_motivation_plan = normalise_antagonist_motivation_plan(
        snap.get("antagonist_motivation_plan", {}), character_list, chapter_list,
    )
    technology_rules = normalise_technology_rules(
        snap.get("technology_rules", {}), chapter_list,
    )
    theme_reinforcement = normalise_theme_reinforcement(
        snap.get("theme_reinforcement", {}), chapter_list,
    )
    pov_focal_character_plan = normalise_pov_focal_character_plan(
        snap.get("pov_focal_character_plan", {}), character_list, chapter_list,
    )

    target_per_chapter = min(4500, max(2500, word_count // total_chapters))
    characters_text = _format_characters(character_list)
    character_state_log = list(character_state_log) if character_state_log else []
    compression_guidance: str = ""
    degraded_passes: list[dict] = []

    # Format voice seed for prompt injection
    from novelforge.voice import format_voice_prompt
    voice_seed = snap.get("voice_seed", {})
    voice_prompt = format_voice_prompt(voice_seed) if voice_seed else ""

    # Build narrative perspective prompt
    from novelforge.agents.chapter import build_perspective_prompt
    perspective_prompt = build_perspective_prompt(snap.get("narrative_perspective", "third_person"))

    # Reset thread-local LLM token usage for this generation run
    reset_llm_usage()
    set_correlation_token(token)

    _save_file = Path(config.NOVELS_DIR) / f"{token}_progress.json"
    _last_persist: list[float] = [0.0]  # mutable ref for closure

    def _persist_progress(*, force: bool = False) -> None:
        """Persist a progress snapshot to disk atomically."""
        now = time.monotonic()
        if not force and (now - _last_persist[0]) < _PROGRESS_PERSIST_INTERVAL:
            return
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(_save_file.parent),
                prefix=f".{_save_file.stem}_",
                suffix=".tmp",
            )
            try:
                os.chmod(tmp_path, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "token": token,
                            "snapshot": snap,
                            "chapters_done": chapters_done,
                            "summaries": summaries,
                            "character_state_log": character_state_log,
                            "progress": progress_manager.get(token),
                        },
                        fh,
                        indent=2,
                    )
                os.replace(tmp_path, str(_save_file))
                _last_persist[0] = time.monotonic()
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error(
                "Failed to persist progress snapshot for token %s: %s",
                token, e,
                exc_info=True,
            )

    def _set_step(step_label: str) -> None:
        """Update the progress step label and persist a throttled snapshot."""
        progress_manager.update(token, {"step": step_label})
        _persist_progress()  # throttled; force=True callers use _persist_progress directly

    try:
        for idx, ch in enumerate(chapter_list[start_idx:], start=start_idx):
            chapter_deadline = time.monotonic() + PER_CHAPTER_TIMEOUT
            chapter_start_time = time.monotonic()
            reset_llm_usage()
            chapter_num = ch.get("number", idx + 1)
            chapter_title = ch.get("title", f"Chapter {chapter_num}")
            chapter_outline_summary = ch.get("summary", "")
            chapter_architecture_context = get_chapter_architecture_context(story_architecture, chapter_num)
            chapter_timeline_context = get_chapter_timeline_context(master_timeline, chapter_num)
            chapter_fate_context = get_chapter_fate_context(character_fate_registry, chapter_num)
            chapter_arc_context = get_chapter_arc_context(character_arc_plan, chapter_num)
            chapter_antagonist_context = get_chapter_antagonist_context(antagonist_motivation_plan, chapter_num)
            chapter_technology_context = get_chapter_technology_context(technology_rules, chapter_num)
            chapter_theme_context = get_chapter_theme_context(theme_reinforcement, chapter_num)
            chapter_pov_context = get_chapter_pov_context(pov_focal_character_plan, chapter_num)

            previous_summaries = "\n\n".join(
                f"Chapter {i+1}: {s}" for i, s in enumerate(summaries)
            )

            _set_step(f"Chapter {chapter_num}: continuity gatekeeper")
            gatekeeper_brief = run_continuity_gatekeeper(
                chapter_num=chapter_num, chapter_title=chapter_title,
                chapter_summary=chapter_outline_summary,
                previous_summaries=previous_summaries,
                chapter_timeline_context=chapter_timeline_context,
                chapter_fate_context=chapter_fate_context,
                chapter_arc_context=chapter_arc_context,
                character_state_log="\n\n".join(character_state_log),
                degraded_passes=degraded_passes,
            )

            _set_step(f"Chapter {chapter_num}: classifying rhythm")
            rhythm_result = run_chapter_rhythm_classifier(
                chapter_num=chapter_num, chapter_title=chapter_title,
                chapter_summary=chapter_outline_summary,
                previous_summaries=previous_summaries,
                title=title, chapter_architecture_context=chapter_architecture_context,
                degraded_passes=degraded_passes,
            )
            chapter_rhythm_shape = str(rhythm_result.get("recommended_shape_for_this_chapter", "")).strip()
            chapter_rhythm_reason = str(rhythm_result.get("recommendation_reason", "")).strip()

            # 1. Draft (with content-rejection retry)
            _set_step(f"Chapter {chapter_num}: drafting")
            text = _draft_with_content_retry(
                lambda instr: build_chapter_draft_prompt(
                    premise, genre, title, chapter_num, chapter_title,
                    chapter_outline_summary, characters_text,
                    previous_summaries, target_per_chapter, instr,
                    chapter_architecture_context, chapter_timeline_context,
                    chapter_fate_context, chapter_arc_context,
                    chapter_antagonist_context, chapter_technology_context,
                    chapter_theme_context, gatekeeper_brief,
                    compression_guidance, chapter_rhythm_shape,
                    chapter_rhythm_reason, chapter_pov_context,
                    voice_prompt, perspective_prompt,
                ),
                action=f"Chapter {chapter_num}: drafting",
                special_instructions=special_instructions,
                chapter_num=chapter_num,
            )

            ch_ctx = ChapterContext(
                architecture=chapter_architecture_context,
                timeline=chapter_timeline_context,
                fate=chapter_fate_context,
                arc=chapter_arc_context,
                antagonist=chapter_antagonist_context,
                technology=chapter_technology_context,
                theme=chapter_theme_context,
                pov=chapter_pov_context,
                gatekeeper_brief=gatekeeper_brief,
                perspective_prompt=perspective_prompt,
            )
            text, summary = _run_all_chapter_agents(
                text=text, chapter_num=chapter_num, title=title,
                genre=genre, total_chapters=total_chapters,
                chapter_outline_summary=chapter_outline_summary,
                characters_text=characters_text,
                previous_summaries=previous_summaries,
                ctx=ch_ctx, step_callback=_set_step, deadline=chapter_deadline,
                degraded_passes=degraded_passes,
            )
            summaries.append(summary)

            _set_step(f"Chapter {chapter_num}: updating character states")
            state_update = run_character_state_updater(
                chapter_text=text, chapter_summary=summary,
                characters_text=characters_text, chapter_num=chapter_num, title=title,
                degraded_passes=degraded_passes,
            )
            if state_update.strip():
                character_state_log.append(f"--- After Chapter {chapter_num} ---\n{state_update}")

            _set_step(f"Chapter {chapter_num}: compression check")
            compression_guidance = run_per_chapter_compression_check(
                chapter_num=chapter_num, chapter_summary=summary,
                previous_summaries=previous_summaries, title=title,
                degraded_passes=degraded_passes,
            )

            chapter_elapsed = round(time.monotonic() - chapter_start_time, 1)
            chapter_usage = get_llm_usage()
            chapter_word_count = len(text.split())

            chapters_done.append({
                "number": chapter_num,
                "title": chapter_title,
                "content": text,
                "summary": summary,
                "word_count": chapter_word_count,
                "generation_time_seconds": chapter_elapsed,
                "llm_calls": chapter_usage["call_count"],
                "prompt_tokens": chapter_usage["prompt_tokens"],
                "completion_tokens": chapter_usage["completion_tokens"],
                "total_tokens": chapter_usage["total_tokens"],
            })

            progress_manager.update(token, {
                "current": idx + 1,
                "step": f"Chapter {chapter_num}: complete",
                "chapters_done": list(chapters_done),
                "character_state_log": list(character_state_log),
                "degraded_passes": list(degraded_passes),
            })
            _persist_progress(force=True)  # always persist on chapter completion

            session_id = snap.get("session_id")
            if session_id:
                persist_completed_chapters(session_id, chapters_done, token)

        # --- Post-manuscript audits ---
        audit_results = run_post_manuscript_audits(
            token=token,
            title=title,
            genre=genre,
            summaries=summaries,
            special_instructions=special_instructions,
            character_list=character_list,
            character_state_log=character_state_log,
            master_timeline=master_timeline,
            character_fate_registry=character_fate_registry,
            character_arc_plan=character_arc_plan,
            theme_reinforcement=theme_reinforcement,
            total_chapters=total_chapters,
        )

        _set_step("Complete")
        _persist_progress(force=True)  # always persist on successful completion

        session_id = snap.get("session_id")
        if session_id:
            persist_completed_chapters(session_id, chapters_done, token)

        progress_manager.update(token, {"status": "done"})

    except ContentRejectionError as exc:
        logger.error(
            "Content rejection during generation for token %s (after retries exhausted): %s",
            token, exc,
        )
        progress_manager.update(token, {
            "status": "error",
            "error": (
                "The AI service rejected chapter content due to content policy, even after "
                "automatic sanitisation retries. This can happen with horror, violence, or "
                "other mature themes. Your progress has been saved — you can try resuming "
                "or revising the chapter outline to use less explicit language."
            ),
            "error_code": "content_rejection",
        })
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            persist_completed_chapters(session_id, chapters_done, token)
        _set_step("Error: content policy rejection")
        _persist_progress(force=True)

    except ChapterTimeoutError as exc:
        logger.error("Chapter timeout during generation for token %s: %s", token, exc)
        progress_manager.update(token, {
            "status": "error",
            "error": str(exc),
            "error_code": "chapter_timeout",
        })
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            persist_completed_chapters(session_id, chapters_done, token)
        _set_step(f"Error: {exc}")
        _persist_progress(force=True)

    except CircuitBreakerError as exc:
        logger.error("Circuit breaker tripped during generation for token %s: %s", token, exc)
        progress_manager.update(token, {
            "status": "error",
            "error": (
                "LLM API is unavailable — 3 consecutive calls failed. "
                "Please check your API key, endpoint, and rate limits, then try again."
            ),
            "error_code": "circuit_breaker",
        })
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            persist_completed_chapters(session_id, chapters_done, token)
        _set_step("Error: LLM API circuit breaker tripped")
        _persist_progress(force=True)

    except AllProvidersExhaustedError as exc:
        logger.error("All LLM providers exhausted for token %s: %s", token, exc)
        progress_manager.update(token, {
            "status": "error",
            "error": (
                "All configured LLM providers failed. Check your API keys, "
                "endpoints, and rate limits, then try again. "
                "Your progress has been saved."
            ),
            "error_code": "all_providers_exhausted",
        })
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            persist_completed_chapters(session_id, chapters_done, token)
        _set_step("Error: all LLM providers exhausted")
        _persist_progress(force=True)

    except (RuntimeError, requests.exceptions.RequestException) as exc:
        logger.error("LLM communication error for token %s: %s", token, exc)
        progress_manager.update(token, {
            "status": "error",
            "error": friendly_llm_error(exc),
            "error_code": "llm_error",
        })
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            persist_completed_chapters(session_id, chapters_done, token)
        _set_step(f"Error: {str(exc)}")
        _persist_progress(force=True)

    except json.JSONDecodeError as exc:
        logger.error("LLM returned unparseable response for token %s: %s", token, exc)
        progress_manager.update(token, {
            "status": "error",
            "error": "The AI service returned a response that could not be parsed. "
                     "Your progress is saved — please try again.",
            "error_code": "json_parse_error",
        })
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            persist_completed_chapters(session_id, chapters_done, token)
        _set_step("Error: unparseable LLM response")
        _persist_progress(force=True)

    except (KeyError, ValueError) as exc:
        logger.error(
            "Internal error during chapter generation for token %s: %s",
            token, exc, exc_info=True,
        )
        progress_manager.update(token, {
            "status": "error",
            "error": "An internal error occurred during generation. "
                     "Your progress is saved — please try again.",
            "error_code": "internal_error",
        })
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            persist_completed_chapters(session_id, chapters_done, token)
        _set_step(f"Error: {str(exc)}")
        _persist_progress(force=True)

    finally:
        clear_correlation_token()


# All lightweight fields are scalar (str, int, bool, None) so a shallow extraction is safe.
_LIGHTWEIGHT_FIELDS = ("status", "current", "total", "step", "error", "error_code", "_live")


@generation_bp.route("/progress/<token>")
def progress(token: str) -> Response | tuple[Response, int]:
    """Lightweight poll endpoint – returns only status/step fields, not chapter content."""
    data = progress_manager.get(token)
    if data is None:
        return jsonify({"error": "Unknown token"}), 404
    safe_data = cast(Mapping[str, Any], data)
    light = {k: safe_data[k] for k in _LIGHTWEIGHT_FIELDS if k in safe_data}
    return jsonify(light)


@generation_bp.route("/progress/<token>/full")
def progress_full(token: str) -> Response | tuple[Response, int]:
    """Full progress endpoint – returns the complete payload including chapter content and reports."""
    data = progress_manager.get(token)
    if data is None:
        return jsonify({"error": "Unknown token"}), 404
    return jsonify(dict(data))

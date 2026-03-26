"""Chapter generation, revision, and progress routes."""

import json
import logging
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import Blueprint, Response, jsonify, request, session

from novelforge import limiter
import novelforge.config as config
from novelforge.progress import (
    _progress_store, _progress_lock,
    set_correlation_token, clear_correlation_token,
)
from novelforge.llm.client import (
    call_llm, parse_llm_json, _friendly_llm_error,
    _reset_llm_usage, _get_llm_usage, _llm_circuit_breaker,
    CircuitBreakerError, ChapterTimeoutError, PER_CHAPTER_TIMEOUT,
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
    build_chapter_draft_prompt, build_chapter_revision_prompt,
    build_consistency_pass_prompt, build_global_continuity_auditor_prompt,
    build_narrative_compression_editor_prompt,
    build_character_resolution_validator_prompt,
    build_thematic_payoff_analyzer_prompt,
    build_climax_integrity_checker_prompt,
    build_loose_thread_resolver_prompt,
    build_reader_immersion_tester_prompt,
    build_pacing_tension_heatmap_prompt,
    run_continuity_gatekeeper, run_chapter_rhythm_classifier,
    run_character_state_updater, run_per_chapter_compression_check,
    _run_all_chapter_agents, ChapterContext,
)
from novelforge.session.persistence import (
    get_session_id, save_session_state, _persist_completed_chapters,
)

logger = logging.getLogger(__name__)

generation_bp = Blueprint("generation", __name__)


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
    with _progress_lock:
        _progress_store[token] = {
            "status": "running",
            "current": 0,
            "total": session["chapters"],
            "step": "Preparing\u2026",
            "chapters_done": [],
            "error": None,
            "_live": True,
        }

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
    }

    thread = threading.Thread(
        target=_run_chapter_generation,
        args=(token, snapshot),
        daemon=True,
    )
    thread.start()

    session["progress_token"] = token
    save_session_state()

    return jsonify({"token": token})


def _resume_chapter_generation(token: str, snap: dict, chapters_done: list[dict], summaries: list[str], start_idx: int) -> None:
    """Resume chapter generation from a specific chapter index after a crash."""
    _run_chapter_generation_internal(token, snap, chapters_done, summaries, start_idx)


def _run_chapter_generation(token: str, snap: dict) -> None:
    """
    Background worker: generate all chapters sequentially using the full
    twelve-step agent pipeline.

    Per-chapter pipeline:
        0a. Continuity Gatekeeper – validate chapter constraints
        0b. Chapter Rhythm Classifier – recommend contrasting narrative rhythm
        1. Draft              – initial prose (Novelist)
        2. Prose Refinement   – dialogue naturalism + scene momentum (merged)
        3. Scene Variety Audit – detect intra-chapter repetition (directives only)
        4. Context Analyzer   – fix world-building continuity errors
        5. Editing Agent      – plot holes, pacing, character consistency + scene audit directives
        6. Momentum & Distinctiveness – cross-chapter redundancy + escalation (merged)
        7. Structure Agent    – confirm chapter fits story architecture
        8. Operational Distinctiveness – ensure each op differs in strategy and stakes
        9. Character Agent    – deepen arcs, consistency, and forward movement (absorbs thread tracking)
       10. Synthesizer        – unify voice and theme after multi-pass edits
       11. Polish Agent       – grammar, style, vivid language
       12. Anti-LLM Agent     – strip robotic patterns and forbidden words
       13. Quality Controller – engagement, tension, pacing check
       14. Summary            – 100-200 word continuity summary

    Post-chapter (after summary, before next chapter):
        A. Character State Updater – log definitive character states into running registry

    Post-manuscript (after all chapters):
        I.   Final consistency pass – arc, threads, thematic payoff (JSON)
        II.  Global Continuity Auditor – cross-chapter contradiction detection (JSON)
        III. Narrative Compression Editor – identify redundant sequences for consolidation (JSON)
        IV.  Character Resolution Validator – confirm every major character receives closure (JSON)
        V.   Thematic Payoff Analyzer – ensure all themes culminate in the final act (JSON)
        VI.  Climax Integrity Checker – verify protagonist makes definitive final moral decision (JSON)
        VII. Loose Thread Resolver – identify and close unresolved narrative questions (JSON)
        VIII.Reader Immersion Tester – evaluate pacing, tension, and engagement from reader POV (JSON)
        IX.  Pacing & Tension Heatmap – per-chapter metrics for tension, action, emotion, dialogue, description
    """
    _run_chapter_generation_internal(token, snap, [], [], 0)


def _run_chapter_generation_internal(
    token: str,
    snap: dict,
    chapters_done: list[dict],
    summaries: list[str],
    start_idx: int
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

    target_per_chapter = max(500, word_count // total_chapters)
    characters_text = _format_characters(character_list)
    character_state_log: list[str] = []
    compression_guidance: str = ""

    _llm_circuit_breaker.reset()
    _reset_llm_usage()
    set_correlation_token(token)

    def _set_step(step_label: str) -> None:
        with _progress_lock:
            _progress_store[token]["step"] = step_label
        try:
            save_file = Path(config.NOVELS_DIR) / f"{token}_progress.json"
            with _progress_lock:
                progress_data = dict(_progress_store[token])
            save_data = {
                "token": token,
                "snapshot": snap,
                "chapters_done": chapters_done,
                "summaries": summaries,
                "character_state_log": character_state_log,
                "progress": progress_data,
            }
            save_file.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to auto-save progress: {e}")

    try:
        for idx, ch in enumerate(chapter_list[start_idx:], start=start_idx):
            chapter_deadline = time.monotonic() + PER_CHAPTER_TIMEOUT
            chapter_start_time = time.monotonic()
            _reset_llm_usage()
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
            )

            _set_step(f"Chapter {chapter_num}: classifying rhythm")
            rhythm_result = run_chapter_rhythm_classifier(
                chapter_num=chapter_num, chapter_title=chapter_title,
                chapter_summary=chapter_outline_summary,
                previous_summaries=previous_summaries,
                title=title, chapter_architecture_context=chapter_architecture_context,
            )
            chapter_rhythm_shape = str(rhythm_result.get("recommended_shape_for_this_chapter", "")).strip()
            chapter_rhythm_reason = str(rhythm_result.get("recommendation_reason", "")).strip()

            # 1. Draft
            _set_step(f"Chapter {chapter_num}: drafting")
            text = call_llm(
                build_chapter_draft_prompt(
                    premise, genre, title, chapter_num, chapter_title,
                    chapter_outline_summary, characters_text,
                    previous_summaries, target_per_chapter, special_instructions,
                    chapter_architecture_context, chapter_timeline_context,
                    chapter_fate_context, chapter_arc_context,
                    chapter_antagonist_context, chapter_technology_context,
                    chapter_theme_context, gatekeeper_brief,
                    compression_guidance, chapter_rhythm_shape,
                    chapter_rhythm_reason, chapter_pov_context,
                ),
                action=f"Chapter {chapter_num}: drafting"
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
            )
            text, summary = _run_all_chapter_agents(
                text=text, chapter_num=chapter_num, title=title,
                genre=genre, total_chapters=total_chapters,
                chapter_outline_summary=chapter_outline_summary,
                characters_text=characters_text,
                previous_summaries=previous_summaries,
                ctx=ch_ctx, step_callback=_set_step, deadline=chapter_deadline,
            )
            summaries.append(summary)

            _set_step(f"Chapter {chapter_num}: updating character states")
            state_update = run_character_state_updater(
                chapter_text=text, chapter_summary=summary,
                characters_text=characters_text, chapter_num=chapter_num, title=title,
            )
            if state_update.strip():
                character_state_log.append(f"--- After Chapter {chapter_num} ---\n{state_update}")

            _set_step(f"Chapter {chapter_num}: compression check")
            compression_guidance = run_per_chapter_compression_check(
                chapter_num=chapter_num, chapter_summary=summary,
                previous_summaries=previous_summaries, title=title,
            )

            chapter_elapsed = round(time.monotonic() - chapter_start_time, 1)
            chapter_usage = _get_llm_usage()
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

            with _progress_lock:
                _progress_store[token]["current"] = idx + 1
                _progress_store[token]["step"] = f"Chapter {chapter_num}: complete"
                _progress_store[token]["chapters_done"] = list(chapters_done)

            session_id = snap.get("session_id")
            if session_id:
                _persist_completed_chapters(session_id, chapters_done)

        # --- Final consistency pass ---
        with _progress_lock:
            _progress_store[token]["step"] = "Final consistency pass"
        consistency_raw = call_llm(
            build_consistency_pass_prompt(title, summaries, special_instructions),
            action="Final consistency pass", json_mode=True,
        )
        try:
            consistency = parse_llm_json(consistency_raw)
        except json.JSONDecodeError:
            consistency = {"issues": [], "overall_assessment": ""}

        # --- Global Continuity Audit ---
        with _progress_lock:
            _progress_store[token]["step"] = "Global continuity audit"
        audit_raw = call_llm(
            build_global_continuity_auditor_prompt(
                title=title, all_summaries=summaries,
                character_state_log=character_state_log,
                master_timeline=master_timeline,
                character_fate_registry=character_fate_registry,
            ),
            action="Global continuity audit", json_mode=True,
        )
        try:
            global_audit = parse_llm_json(audit_raw)
        except json.JSONDecodeError:
            global_audit = {
                "contradictions": [], "character_state_errors": [],
                "timeline_errors": [], "location_errors": [],
                "overall_integrity": "unknown", "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["status"] = "done"
            _progress_store[token]["consistency"] = consistency
            _progress_store[token]["global_continuity_audit"] = global_audit

        # --- Narrative Compression Edit ---
        with _progress_lock:
            _progress_store[token]["step"] = "Narrative compression analysis"
        compression_raw = call_llm(
            build_narrative_compression_editor_prompt(
                title=title, all_summaries=summaries, continuity_audit=global_audit,
            ),
            action="Narrative compression analysis", json_mode=True,
        )
        try:
            compression_report = parse_llm_json(compression_raw)
        except json.JSONDecodeError:
            compression_report = {
                "redundant_sequences": [], "emotional_beat_repetitions": [],
                "compression_priority": "unknown", "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["narrative_compression_report"] = compression_report

        # --- Character Resolution Validation ---
        with _progress_lock:
            _progress_store[token]["step"] = "Character resolution validation"
        resolution_raw = call_llm(
            build_character_resolution_validator_prompt(
                title=title, all_summaries=summaries,
                character_arc_plan=character_arc_plan,
                character_fate_registry=character_fate_registry,
                character_state_log=character_state_log,
            ),
            action="Character resolution validation", json_mode=True,
        )
        try:
            resolution_report = parse_llm_json(resolution_raw)
        except json.JSONDecodeError:
            resolution_report = {
                "character_resolutions": [], "unresolved_characters": [],
                "resolution_integrity": "unknown", "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["character_resolution_report"] = resolution_report

        # --- Thematic Payoff Analysis ---
        with _progress_lock:
            _progress_store[token]["step"] = "Thematic payoff analysis"
        thematic_raw = call_llm(
            build_thematic_payoff_analyzer_prompt(
                title=title, all_summaries=summaries,
                theme_reinforcement=theme_reinforcement, total_chapters=total_chapters,
            ),
            action="Thematic payoff analysis", json_mode=True,
        )
        try:
            thematic_report = parse_llm_json(thematic_raw)
        except json.JSONDecodeError:
            thematic_report = {
                "theme_payoffs": [], "abandoned_themes": [],
                "weak_payoffs": [], "thematic_integrity": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["thematic_payoff_report"] = thematic_report

        # --- Climax Integrity Check ---
        with _progress_lock:
            _progress_store[token]["step"] = "Climax integrity check"
        climax_raw = call_llm(
            build_climax_integrity_checker_prompt(
                title=title, all_summaries=summaries,
                character_arc_plan=character_arc_plan, total_chapters=total_chapters,
            ),
            action="Climax integrity check", json_mode=True,
        )
        try:
            climax_report = parse_llm_json(climax_raw)
        except json.JSONDecodeError:
            climax_report = {
                "climax_decision_present": False, "decision_is_active": False,
                "moral_dimension_present": False, "arc_resolved": False,
                "protagonist_is_agent": False, "climax_chapter": None,
                "integrity_failures": [], "climax_integrity": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["climax_integrity_report"] = climax_report

        # --- Loose Thread Resolution ---
        with _progress_lock:
            _progress_store[token]["step"] = "Loose thread resolution"
        threads_raw = call_llm(
            build_loose_thread_resolver_prompt(
                title=title, all_summaries=summaries,
                character_state_log=character_state_log,
                continuity_audit=global_audit, resolution_report=resolution_report,
            ),
            action="Loose thread resolution", json_mode=True,
        )
        try:
            threads_report = parse_llm_json(threads_raw)
        except json.JSONDecodeError:
            threads_report = {
                "unresolved_threads": [], "dangling_setup_elements": [],
                "intentionally_open_threads": [], "thread_integrity": "unknown",
                "overall_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["loose_thread_report"] = threads_report

        # --- Reader Immersion Testing ---
        with _progress_lock:
            _progress_store[token]["step"] = "Reader immersion testing"
        immersion_raw = call_llm(
            build_reader_immersion_tester_prompt(
                title=title, all_summaries=summaries,
                character_arc_plan=character_arc_plan, thematic_report=thematic_report,
            ),
            action="Reader immersion testing", json_mode=True,
        )
        try:
            immersion_report = parse_llm_json(immersion_raw)
        except json.JSONDecodeError:
            immersion_report = {
                "pacing_assessment": "unknown", "tension_curve": "unknown",
                "stakes_clarity": "unknown", "engagement_score": 0,
                "weak_chapters": [], "immersion_breaks": [],
                "reader_experience_highlights": [], "overall_rating": "unknown",
                "recommendations": [],
            }

        with _progress_lock:
            _progress_store[token]["reader_immersion_report"] = immersion_report

        # --- Pacing & Tension Heatmap ---
        with _progress_lock:
            _progress_store[token]["step"] = "Pacing & tension heatmap"
        heatmap_raw = call_llm(
            build_pacing_tension_heatmap_prompt(
                title=title, all_summaries=summaries, total_chapters=total_chapters,
            ),
            action="Pacing & tension heatmap", json_mode=True,
        )
        try:
            pacing_heatmap = parse_llm_json(heatmap_raw)
        except json.JSONDecodeError:
            pacing_heatmap = {
                "chapter_metrics": [], "flat_sections": [],
                "overall_pacing_assessment": "",
            }

        with _progress_lock:
            _progress_store[token]["pacing_heatmap"] = pacing_heatmap

        _set_step("Complete")

        session_id = snap.get("session_id")
        if session_id:
            _persist_completed_chapters(session_id, chapters_done)

    except ChapterTimeoutError as exc:
        logger.error("Chapter timeout during generation for token %s: %s", token, exc)
        with _progress_lock:
            _progress_store[token]["status"] = "error"
            _progress_store[token]["error"] = str(exc)
            _progress_store[token]["error_code"] = "chapter_timeout"
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            _persist_completed_chapters(session_id, chapters_done)
        _set_step(f"Error: {exc}")

    except CircuitBreakerError as exc:
        logger.error("Circuit breaker tripped during generation for token %s: %s", token, exc)
        with _progress_lock:
            _progress_store[token]["status"] = "error"
            _progress_store[token]["error"] = (
                "LLM API is unavailable — 3 consecutive calls failed. "
                "Please check your API key, endpoint, and rate limits, then try again."
            )
            _progress_store[token]["error_code"] = "circuit_breaker"
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            _persist_completed_chapters(session_id, chapters_done)
        _set_step("Error: LLM API circuit breaker tripped")

    except (RuntimeError, requests.exceptions.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Chapter generation failed for token %s: %s", token, exc)
        with _progress_lock:
            _progress_store[token]["status"] = "error"
            _progress_store[token]["error"] = _friendly_llm_error(exc)
        session_id = snap.get("session_id")
        if session_id and chapters_done:
            _persist_completed_chapters(session_id, chapters_done)
        _set_step(f"Error: {str(exc)}")

    finally:
        clear_correlation_token()


@generation_bp.route("/revise_chapter", methods=["POST"])
@limiter.limit("5 per minute")
def revise_chapter() -> Response | tuple[Response, int]:
    """
    Apply custom editor instructions to one generated chapter, then re-run all
    chapter agents on that updated material.
    """
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")
    instructions = str(data.get("instructions", "")).strip()

    try:
        chapter_number = int(data.get("chapter_number", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Chapter number must be a valid number."}), 400

    if not token:
        return jsonify({"error": "Missing progress token."}), 400
    if chapter_number < 1:
        return jsonify({"error": "Chapter number must be at least 1."}), 400
    if not instructions:
        return jsonify({"error": "Revision instructions are required."}), 400

    with _progress_lock:
        progress_data = _progress_store.get(token)

    if not progress_data or progress_data.get("status") != "done":
        return jsonify({"error": "Novel generation not complete."}), 400

    chapters_done = list(progress_data.get("chapters_done", []))
    if not chapters_done:
        return jsonify({"error": "No generated chapters found."}), 400

    target_idx = next(
        (i for i, chapter in enumerate(chapters_done) if int(chapter.get("number", 0)) == chapter_number),
        None,
    )
    if target_idx is None:
        return jsonify({"error": "Selected chapter was not found."}), 404

    title = session.get("title", "Novel")
    genre = session.get("genre", "")
    total_chapters = int(session.get("chapters", len(chapters_done) or 1))
    chapter_list = session.get("chapter_list", [])
    character_list = session.get("character_list", [])
    special_instructions = session.get("special_instructions", "")
    story_architecture = normalise_story_architecture(session.get("story_architecture", {}), chapter_list, total_chapters)
    master_timeline = normalise_master_timeline(session.get("master_timeline", {}), chapter_list, character_list)
    character_fate_registry = normalise_character_fate_registry(session.get("character_fate_registry", {}), character_list, total_chapters)
    character_arc_plan = normalise_character_arc_plan(session.get("character_arc_plan", {}), character_list, chapter_list)
    antagonist_motivation_plan = normalise_antagonist_motivation_plan(session.get("antagonist_motivation_plan", {}), character_list, chapter_list)
    technology_rules = normalise_technology_rules(session.get("technology_rules", {}), chapter_list)
    theme_reinforcement = normalise_theme_reinforcement(session.get("theme_reinforcement", {}), chapter_list)
    pov_focal_character_plan = normalise_pov_focal_character_plan(session.get("pov_focal_character_plan", {}), character_list, chapter_list)

    chapter_outline_summary = ""
    for chapter_outline in chapter_list:
        try:
            if int(chapter_outline.get("number", 0)) == chapter_number:
                chapter_outline_summary = chapter_outline.get("summary", "")
                break
        except (TypeError, ValueError):
            continue

    target_chapter = chapters_done[target_idx]
    previous_summaries = "\n\n".join(
        f"Chapter {c.get('number', i+1)}: {c.get('summary', '')}"
        for i, c in enumerate(chapters_done[:target_idx])
    )
    characters_text = _format_characters(character_list)
    chapter_architecture_context = get_chapter_architecture_context(story_architecture, chapter_number)
    chapter_timeline_context = get_chapter_timeline_context(master_timeline, chapter_number)
    chapter_fate_context = get_chapter_fate_context(character_fate_registry, chapter_number)
    chapter_arc_context = get_chapter_arc_context(character_arc_plan, chapter_number)
    chapter_antagonist_context = get_chapter_antagonist_context(antagonist_motivation_plan, chapter_number)
    chapter_technology_context = get_chapter_technology_context(technology_rules, chapter_number)
    chapter_theme_context = get_chapter_theme_context(theme_reinforcement, chapter_number)
    chapter_pov_context = get_chapter_pov_context(pov_focal_character_plan, chapter_number)

    gatekeeper_brief = run_continuity_gatekeeper(
        chapter_num=chapter_number,
        chapter_title=target_chapter.get("title", f"Chapter {chapter_number}"),
        chapter_summary=chapter_outline_summary,
        previous_summaries=previous_summaries,
        chapter_timeline_context=chapter_timeline_context,
        chapter_fate_context=chapter_fate_context,
        chapter_arc_context=chapter_arc_context,
    )

    try:
        revised_text = call_llm(
            build_chapter_revision_prompt(
                chapter_text=target_chapter.get("content", ""),
                chapter_num=chapter_number, title=title,
                chapter_outline_summary=chapter_outline_summary,
                revision_instructions=instructions,
                chapter_architecture_context=chapter_architecture_context,
                chapter_timeline_context=chapter_timeline_context,
                chapter_fate_context=chapter_fate_context,
                chapter_arc_context=chapter_arc_context,
                chapter_antagonist_context=chapter_antagonist_context,
                chapter_technology_context=chapter_technology_context,
                chapter_theme_context=chapter_theme_context,
                gatekeeper_brief=gatekeeper_brief,
            ),
            action=f"Chapter {chapter_number}: applying revision instructions"
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
        )
        revised_text, revised_summary = _run_all_chapter_agents(
            text=revised_text, chapter_num=chapter_number,
            title=title, genre=genre, total_chapters=total_chapters,
            chapter_outline_summary=chapter_outline_summary,
            characters_text=characters_text, previous_summaries=previous_summaries,
            ctx=ch_ctx, step_callback=None,
        )

        chapters_done[target_idx]["content"] = revised_text
        chapters_done[target_idx]["summary"] = revised_summary

        all_summaries = [str(ch.get("summary", "")) for ch in chapters_done]
        consistency_raw = call_llm(
            build_consistency_pass_prompt(title, all_summaries, special_instructions),
            action="Final consistency pass after revision", json_mode=True,
        )
        try:
            consistency = parse_llm_json(consistency_raw)
        except json.JSONDecodeError:
            consistency = {"issues": [], "overall_assessment": ""}

        with _progress_lock:
            _progress_store[token]["status"] = "done"
            _progress_store[token]["step"] = f"Chapter {chapter_number}: revised"
            _progress_store[token]["chapters_done"] = chapters_done
            _progress_store[token]["consistency"] = consistency
            response_payload = dict(_progress_store[token])

        return jsonify(response_payload)

    except RuntimeError as exc:
        logger.error("Chapter revision failed for token %s: %s", token, exc)
        return jsonify({"error": _friendly_llm_error(exc)}), 502


@generation_bp.route("/progress/<token>")
def progress(token: str) -> Response | tuple[Response, int]:
    """Poll endpoint for chapter generation progress."""
    with _progress_lock:
        data = _progress_store.get(token)
    if data is None:
        return jsonify({"error": "Unknown token"}), 404
    return jsonify(data)

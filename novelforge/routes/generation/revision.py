"""Chapter revision endpoint."""

import json
import logging
import time

from flask import Response, jsonify, request

from novelforge import limiter
from novelforge.llm.client import (
    call_llm, parse_llm_json, friendly_llm_error,
    ChapterTimeoutError, ContentRejectionError, PER_CHAPTER_TIMEOUT,
)
from novelforge.progress import progress_manager
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
    _format_characters, build_chapter_revision_prompt,
    build_consistency_pass_prompt, build_perspective_prompt,
    run_continuity_gatekeeper, _run_all_chapter_agents, ChapterContext,
)
from novelforge.session.persistence import persist_completed_chapters
from novelforge.routes.generation._shared import (
    generation_bp, _DERIVED_REPORT_FIELDS,
)

logger = logging.getLogger(__name__)


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

    progress_data = progress_manager.get(token)

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

    snap = progress_data.get("snapshot", {})
    title = snap.get("title", "Novel")
    genre = snap.get("genre", "")
    total_chapters = int(snap.get("chapters", len(chapters_done) or 1))
    chapter_list = snap.get("chapter_list", [])
    character_list = snap.get("character_list", [])
    special_instructions = snap.get("special_instructions", "")
    story_architecture = normalise_story_architecture(snap.get("story_architecture", {}), chapter_list, total_chapters)
    master_timeline = normalise_master_timeline(snap.get("master_timeline", {}), chapter_list, character_list)
    character_fate_registry = normalise_character_fate_registry(snap.get("character_fate_registry", {}), character_list, total_chapters)
    character_arc_plan = normalise_character_arc_plan(snap.get("character_arc_plan", {}), character_list, chapter_list)
    antagonist_motivation_plan = normalise_antagonist_motivation_plan(snap.get("antagonist_motivation_plan", {}), character_list, chapter_list)
    technology_rules = normalise_technology_rules(snap.get("technology_rules", {}), chapter_list)
    theme_reinforcement = normalise_theme_reinforcement(snap.get("theme_reinforcement", {}), chapter_list)
    pov_focal_character_plan = normalise_pov_focal_character_plan(snap.get("pov_focal_character_plan", {}), character_list, chapter_list)

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

    perspective_prompt = build_perspective_prompt(snap.get("narrative_perspective", "third_person"))

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
                perspective_prompt=perspective_prompt,
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
            perspective_prompt=perspective_prompt,
        )
        revised_text, revised_summary = _run_all_chapter_agents(
            text=revised_text, chapter_num=chapter_number,
            title=title, genre=genre, total_chapters=total_chapters,
            chapter_outline_summary=chapter_outline_summary,
            characters_text=characters_text, previous_summaries=previous_summaries,
            ctx=ch_ctx, step_callback=None,
            deadline=time.monotonic() + PER_CHAPTER_TIMEOUT,
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

        progress_manager.update(token, {
            "status": "done",
            "step": f"Chapter {chapter_number}: revised",
            "chapters_done": chapters_done,
            "consistency": consistency,
        })

        # Invalidate derived reports that are now stale relative to the revised
        # chapter text.  Consumers should treat None values as "not yet computed"
        # and re-request generation if they need fresh analysis.
        progress_manager.update(token, {field: None for field in _DERIVED_REPORT_FIELDS})

        # Persist the revised state using the same durable persistence path as
        # the main generation pipeline so that revisions survive a restart.
        session_id = snap.get("session_id")
        if session_id:
            persist_completed_chapters(session_id, chapters_done, token)

        response_payload = progress_manager.get(token)

        return jsonify(response_payload)

    except ChapterTimeoutError as exc:
        logger.error("Chapter revision timed out for token %s: %s", token, exc)
        return jsonify({
            "error": f"Chapter revision exceeded the {PER_CHAPTER_TIMEOUT // 60}-minute "
                     "time limit. The chapter may be too complex for a single revision "
                     "pass. Try breaking your instructions into smaller changes."
        }), 504
    except ContentRejectionError as exc:
        logger.error("Content rejection during revision for token %s: %s", token, exc)
        return jsonify({
            "error": "The AI service rejected the chapter content due to content policy, "
                     "even after automatic sanitisation retries. Try revising the chapter "
                     "with instructions to tone down explicit content."
        }), 502
    except RuntimeError as exc:
        logger.error("Chapter revision failed for token %s: %s", token, exc)
        return jsonify({"error": friendly_llm_error(exc)}), 502

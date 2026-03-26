"""Outline generation and approval routes."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import markupsafe
from flask import Blueprint, Response, jsonify, request, session

from novelforge import limiter
from novelforge.validation import validate_outline_input
from novelforge.llm.client import call_llm, parse_llm_json, _friendly_llm_error
from novelforge.agents.chapter import (
    build_title_prompt, build_outline_prompt, build_characters_prompt,
)
from novelforge.agents.planning import (
    plan_story_architecture, plan_master_timeline, plan_character_fate_registry,
    plan_character_arc_plan, plan_antagonist_motivation_plan,
    plan_technology_rules, plan_theme_reinforcement, plan_pov_focal_character,
)
from novelforge.session.persistence import save_session_state

logger = logging.getLogger(__name__)

outline_bp = Blueprint("outline", __name__)


@outline_bp.route("/generate_outline", methods=["POST"])
@limiter.limit("5 per minute")
def generate_outline() -> Response | tuple[Response, int]:
    """
    Phase 1: Generate title, chapter outline, and main characters.

    Expects JSON body with: premise, genre, chapters, word_count,
    special_events, special_instructions.
    """
    data = request.get_json(silent=True) or {}

    ok, err = validate_outline_input(data)
    if not ok:
        return jsonify({"error": err}), 400

    premise = markupsafe.escape(data["premise"].strip())
    genre = data["genre"].strip()
    chapters = int(data["chapters"])
    word_count = int(data["word_count"])
    special_events = data.get("special_events", "").strip()
    special_instructions = data.get("special_instructions", "").strip()

    # Store sanitised inputs in session for later phases
    session["premise"] = str(premise)
    session["genre"] = genre
    session["chapters"] = chapters
    session["word_count"] = word_count
    session["special_events"] = special_events
    session["special_instructions"] = special_instructions

    try:
        # 1. Generate title
        title = call_llm(build_title_prompt(str(premise), genre), action="Generating Title").strip().strip('"')

        # 2. Generate outline
        outline_raw = call_llm(
            build_outline_prompt(
                str(premise), genre, chapters, word_count, special_events, special_instructions
            ),
            action="Generating Outline",
            json_mode=True,
        )
        try:
            outline_data = parse_llm_json(outline_raw)
            chapter_list = outline_data.get("chapters", [])
        except json.JSONDecodeError:
            chapter_list = [
                {"number": i + 1, "title": f"Chapter {i+1}", "summary": ""}
                for i in range(chapters)
            ]

        outline_text = "\n".join(
            f"Chapter {c['number']}: {c['title']} – {c['summary']}"
            for c in chapter_list
        )

        # 3. Generate characters (needed by Group 1 timeline agent)
        characters_raw = call_llm(
            build_characters_prompt(str(premise), genre, outline_text),
            action="Generating Characters",
            json_mode=True,
        )
        try:
            characters_data = parse_llm_json(characters_raw)
            character_list = characters_data.get("characters", [])
        except json.JSONDecodeError:
            character_list = []

        # Common kwargs shared by all planning agents
        common = dict(
            title=title, premise=str(premise), genre=genre,
            chapter_list=chapter_list, special_instructions=special_instructions,
        )

        # --- Group 1: independent agents (run in parallel) ---
        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_arch = pool.submit(plan_story_architecture, **common)
            fut_timeline = pool.submit(
                plan_master_timeline, character_list=character_list, **common,
            )
            fut_tech = pool.submit(plan_technology_rules, **common)
            fut_theme = pool.submit(plan_theme_reinforcement, **common)

            story_architecture = fut_arch.result()
            master_timeline = fut_timeline.result()
            technology_rules = fut_tech.result()
            theme_reinforcement = fut_theme.result()

        # --- Group 2: depend on Group 1 outputs (run in parallel) ---
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_fate = pool.submit(
                plan_character_fate_registry,
                character_list=character_list,
                master_timeline=master_timeline, **common,
            )
            fut_arc = pool.submit(
                plan_character_arc_plan,
                character_list=character_list, **common,
            )
            fut_antag = pool.submit(
                plan_antagonist_motivation_plan,
                character_list=character_list,
                master_timeline=master_timeline, **common,
            )

            character_fate_registry = fut_fate.result()
            character_arc_plan = fut_arc.result()
            antagonist_motivation_plan = fut_antag.result()

        # --- Group 3: depends on Group 2 (character_arc_plan) ---
        pov_focal_character_plan = plan_pov_focal_character(
            character_list=character_list,
            character_arc_plan=character_arc_plan, **common,
        )

        # Store outline data in session
        session["title"] = title
        session["chapter_list"] = chapter_list
        session["character_list"] = character_list
        session["story_architecture"] = story_architecture
        session["master_timeline"] = master_timeline
        session["character_fate_registry"] = character_fate_registry
        session["character_arc_plan"] = character_arc_plan
        session["antagonist_motivation_plan"] = antagonist_motivation_plan
        session["technology_rules"] = technology_rules
        session["theme_reinforcement"] = theme_reinforcement
        session["pov_focal_character_plan"] = pov_focal_character_plan

        return jsonify({
            "title": title,
            "chapters": chapter_list,
            "characters": character_list,
            "story_architecture": story_architecture,
            "master_timeline": master_timeline,
            "character_fate_registry": character_fate_registry,
            "character_arc_plan": character_arc_plan,
            "antagonist_motivation_plan": antagonist_motivation_plan,
            "technology_rules": technology_rules,
            "theme_reinforcement": theme_reinforcement,
            "pov_focal_character_plan": pov_focal_character_plan,
        })

    except RuntimeError as exc:
        logger.error("Outline generation failed: %s", exc)
        return jsonify({"error": _friendly_llm_error(exc)}), 502


@outline_bp.route("/approve_outline", methods=["POST"])
def approve_outline() -> Response | tuple[Response, int]:
    """
    Save user-edited outline and characters back to the session.
    Expects JSON with: title, chapters (list), characters (list).
    """
    data = request.get_json(silent=True) or {}

    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title is required."}), 400
    if len(title) > 200:
        return jsonify({"error": "Title must be 200 characters or fewer."}), 400

    chapter_list = data.get("chapters", [])
    if not isinstance(chapter_list, list) or len(chapter_list) == 0:
        return jsonify({"error": "Chapter list is required."}), 400

    # Validate chapter field lengths
    for i, ch in enumerate(chapter_list, 1):
        ch_title = ch.get("title", "") if isinstance(ch, dict) else ""
        ch_summary = ch.get("summary", "") if isinstance(ch, dict) else ""
        if isinstance(ch_title, str) and len(ch_title) > 200:
            return jsonify({"error": f"Chapter {i} title must be 200 characters or fewer."}), 400
        if isinstance(ch_summary, str) and len(ch_summary) > 2000:
            return jsonify({"error": f"Chapter {i} summary must be 2,000 characters or fewer."}), 400

    character_list = data.get("characters", [])

    # Validate character field lengths
    _char_limits = {"name": 100, "age": 50, "role": 200, "background": 2000, "arc": 2000}
    for i, char in enumerate(character_list, 1):
        if not isinstance(char, dict):
            continue
        for field, limit in _char_limits.items():
            value = char.get(field, "")
            if isinstance(value, str) and len(value) > limit:
                label = char.get("name", f"Character {i}")
                return jsonify({"error": f"{label}: {field} must be {limit:,} characters or fewer."}), 400

    # Sanitise string fields to prevent XSS leaking into stored session data
    def sanitise_str(v: object) -> object:
        return str(markupsafe.escape(v)) if isinstance(v, str) else v

    session["title"] = sanitise_str(title)

    # Detect character renames
    old_characters = session.get("character_list", [])
    new_characters = [
        {k: sanitise_str(v) for k, v in ch.items()} for ch in character_list
    ]
    rename_map: dict[str, str] = {}
    for old_ch, new_ch in zip(old_characters, new_characters):
        old_name = (old_ch.get("name") or "").strip()
        new_name = (new_ch.get("name") or "").strip()
        if old_name and new_name and old_name != new_name:
            rename_map[old_name] = new_name

    sanitised_chapters = [
        {k: sanitise_str(v) for k, v in ch.items()} for ch in chapter_list
    ]

    if rename_map:
        logger.info("Character renames detected: %s", rename_map)
        for ch in sanitised_chapters:
            for old_name, new_name in rename_map.items():
                for field in ("title", "summary"):
                    if isinstance(ch.get(field), str):
                        ch[field] = ch[field].replace(old_name, new_name)
        for session_field in ("premise", "special_events", "special_instructions"):
            text = session.get(session_field, "")
            if isinstance(text, str) and text:
                for old_name, new_name in rename_map.items():
                    text = text.replace(old_name, new_name)
                session[session_field] = text

    session["chapter_list"] = sanitised_chapters
    session["character_list"] = new_characters

    # Common kwargs shared by all planning agents
    common = dict(
        title=session["title"], premise=session.get("premise", ""),
        genre=session.get("genre", ""), chapter_list=session["chapter_list"],
        special_instructions=session.get("special_instructions", ""),
    )

    # --- Group 1: independent agents (run in parallel) ---
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_arch = pool.submit(plan_story_architecture, **common)
        fut_timeline = pool.submit(
            plan_master_timeline,
            character_list=session["character_list"], **common,
        )
        fut_tech = pool.submit(plan_technology_rules, **common)
        fut_theme = pool.submit(plan_theme_reinforcement, **common)

        session["story_architecture"] = fut_arch.result()
        session["master_timeline"] = fut_timeline.result()
        session["technology_rules"] = fut_tech.result()
        session["theme_reinforcement"] = fut_theme.result()

    # --- Group 2: depend on Group 1 outputs (run in parallel) ---
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_fate = pool.submit(
            plan_character_fate_registry,
            character_list=session["character_list"],
            master_timeline=session["master_timeline"], **common,
        )
        fut_arc = pool.submit(
            plan_character_arc_plan,
            character_list=session["character_list"], **common,
        )
        fut_antag = pool.submit(
            plan_antagonist_motivation_plan,
            character_list=session["character_list"],
            master_timeline=session["master_timeline"], **common,
        )

        session["character_fate_registry"] = fut_fate.result()
        session["character_arc_plan"] = fut_arc.result()
        session["antagonist_motivation_plan"] = fut_antag.result()

    # --- Group 3: depends on Group 2 (character_arc_plan) ---
    session["pov_focal_character_plan"] = plan_pov_focal_character(
        character_list=session["character_list"],
        character_arc_plan=session["character_arc_plan"], **common,
    )

    # Auto-save session state after outline approval
    save_session_state()

    return jsonify({
        "status": "approved",
        "story_architecture": session["story_architecture"],
        "master_timeline": session["master_timeline"],
        "character_fate_registry": session["character_fate_registry"],
        "character_arc_plan": session["character_arc_plan"],
        "antagonist_motivation_plan": session["antagonist_motivation_plan"],
        "technology_rules": session["technology_rules"],
        "theme_reinforcement": session["theme_reinforcement"],
        "pov_focal_character_plan": session["pov_focal_character_plan"],
    })

"""Outline generation and approval routes."""

import json
import logging
import re

from flask import Blueprint, Response, jsonify, request, session

from novelforge import limiter
from novelforge.validation import validate_outline_input
from novelforge.llm.client import call_llm, parse_llm_json, friendly_llm_error
from novelforge.voice import select_voice_seed, format_voice_prompt
from novelforge.agents.chapter import (
    build_title_prompt, build_outline_prompt, build_characters_prompt,
    collect_existing_character_names,
)
from novelforge.services.planning import run_full_planning, run_selective_planning
from novelforge.session.persistence import save_session_state

logger = logging.getLogger(__name__)

outline_bp = Blueprint("outline", __name__)


def _apply_rename(text: str, old_name: str, new_name: str) -> str:
    """Replace *old_name* with *new_name* using whole-word boundary matching.

    Uses ``\b`` anchors so that a short name such as ``"Al"`` is never
    substituted inside ``"Alan"`` or any other word that merely contains the
    name as a substring.  Possessive forms (e.g. ``"Alice's"``) are handled
    correctly: the apostrophe-s suffix is captured and re-emitted after the
    new name.
    """
    pattern = r"\b" + re.escape(old_name) + r"('s)?\b"
    return re.sub(pattern, lambda m: new_name + (m.group(1) or ""), text)


@outline_bp.route("/generate_outline", methods=["POST"])
@limiter.limit("5 per minute")
def generate_outline() -> Response | tuple[Response, int]:
    """
    Phase 1: Generate title, chapter outline, and main characters.

    Expects JSON body with: premise, genre, chapters, word_count,
    special_events, special_instructions.
    """
    data = request.get_json(silent=True) or {}

    result = validate_outline_input(data)
    if not result.ok:
        return jsonify({"error": result.first_error, "errors": result.errors}), 400

    premise = result.values["premise"]
    genre = result.values["genre"]
    chapters = result.values["chapters"]
    word_count = result.values["word_count"]
    special_events = result.values.get("special_events", "")
    special_instructions = result.values.get("special_instructions", "")

    # Store sanitised inputs in session for later phases
    session["premise"] = premise
    session["genre"] = genre
    session["chapters"] = chapters
    session["word_count"] = word_count
    session["special_events"] = special_events
    session["special_instructions"] = special_instructions

    # Select a voice seed for this novel — stored in session for all chapters
    voice_seed = select_voice_seed(genre=genre, premise=str(premise))
    session["voice_seed"] = voice_seed
    logger.info("Selected voice seed: %s", voice_seed["name"])

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
            chapter_list = outline_data.get("chapters", []) if isinstance(outline_data, dict) else []
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
            build_characters_prompt(str(premise), genre, outline_text,
                                    names_to_avoid=collect_existing_character_names()),
            action="Generating Characters",
            json_mode=True,
        )
        try:
            characters_data = parse_llm_json(characters_raw)
            character_list = characters_data.get("characters", []) if isinstance(characters_data, dict) else []
        except json.JSONDecodeError:
            character_list = []

        # Run all planning agents via the shared orchestration service
        planning = run_full_planning(
            title=title,
            premise=str(premise),
            genre=genre,
            chapter_list=chapter_list,
            character_list=character_list,
            special_instructions=special_instructions,
        )

        # Store outline data in session
        session["title"] = title
        session["chapter_list"] = chapter_list
        session["character_list"] = character_list
        session.update(planning)

        # Auto-save session state after outline generation for crash recovery
        save_session_state()

        return jsonify({
            "title": title,
            "chapters": chapter_list,
            "characters": character_list,
            **planning,
        })

    except RuntimeError as exc:
        logger.error("Outline generation failed: %s", exc)
        return jsonify({"error": friendly_llm_error(exc)}), 502


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

    # String fields are stored as plain text; XSS escaping is handled at
    # render time by Jinja2 auto-escape and jQuery .text().
    def sanitise_str(v: object) -> object:
        """Return the value unchanged (sanitisation handled by Jinja2 auto-escape)."""
        return v

    # -------------------------------------------------------------------------
    # Build a working copy of all state that will change.
    # The live session is NOT mutated until every planning-agent recomputation
    # succeeds, so a mid-flight failure leaves the session intact.
    # -------------------------------------------------------------------------
    working: dict = {}

    # Detect character renames using the current (pre-mutation) session values
    old_characters = session.get("character_list", [])
    new_characters = [
        {k: sanitise_str(v) for k, v in ch.items()} for ch in character_list
    ]
    rename_map: dict[str, str] = {}
    for old_ch, new_ch in zip(old_characters, new_characters):
        old_name = str(old_ch.get("name") or "").strip()
        new_name = str(new_ch.get("name") or "").strip()
        if old_name and new_name and old_name != new_name:
            rename_map[old_name] = new_name

    sanitised_chapters = [
        {k: sanitise_str(v) for k, v in ch.items()} for ch in chapter_list
    ]

    # Force sequential chapter numbers 1..N regardless of what the client sent
    for idx, ch in enumerate(sanitised_chapters, 1):
        ch["number"] = idx

    if rename_map:
        logger.info("Character renames detected: %s", rename_map)
        for ch in sanitised_chapters:
            for old_name, new_name in rename_map.items():
                for field in ("title", "summary"):
                    val = ch.get(field)
                    if isinstance(val, str):
                        ch[field] = _apply_rename(val, old_name, new_name)
        for session_field in ("premise", "special_events", "special_instructions"):
            text = session.get(session_field, "")
            if isinstance(text, str) and text:
                for old_name, new_name in rename_map.items():
                    text = _apply_rename(text, old_name, new_name)
                working[session_field] = text
        # Update narrative perspective if the POV character was renamed
        cur_persp = session.get("narrative_perspective", "third_person")
        if cur_persp.startswith("first_person:"):
            cur_pov_name = cur_persp[len("first_person:"):].strip()
            if not cur_pov_name:
                # Malformed — reset to third person
                working["narrative_perspective"] = "third_person"
            elif cur_pov_name in rename_map:
                working["narrative_perspective"] = f"first_person:{rename_map[cur_pov_name]}"

    # Validate narrative perspective and store in working copy
    narrative_perspective = str(data.get("narrative_perspective", "third_person")).strip()
    if narrative_perspective != "third_person" and not narrative_perspective.startswith("first_person:"):
        narrative_perspective = "third_person"
    if narrative_perspective.startswith("first_person:"):
        pov_char_name = narrative_perspective[len("first_person:"):].strip()
        if not pov_char_name:
            narrative_perspective = "third_person"
        elif len(pov_char_name) > 100:
            return jsonify({"error": "Perspective character name too long."}), 400
        else:
            # Verify the character exists in the list
            char_names = [str(c.get("name", "")).strip() for c in new_characters]
            if pov_char_name not in char_names:
                narrative_perspective = "third_person"

    working["title"] = sanitise_str(title)
    working["chapter_list"] = sanitised_chapters
    working["character_list"] = new_characters
    working["narrative_perspective"] = narrative_perspective

    # Helper: read a key from the working copy, falling back to the live session
    def _from_working(key: str, default: object = "") -> object:
        """Read a key from the working copy, falling back to the live session."""
        return working.get(key, session.get(key, default))

    # Resolve planning inputs from the working copy (post-rename values)
    _title = str(_from_working("title"))
    _premise = str(_from_working("premise"))
    _genre = str(_from_working("genre", ""))
    _chapters = list(_from_working("chapter_list", []))  # type: ignore[call-overload]
    _characters = list(_from_working("character_list", []))  # type: ignore[call-overload]
    _instructions = str(_from_working("special_instructions"))

    # Collect existing session planning outputs for use by skipped agents
    _prev_results = {
        key: session.get(key, {})
        for key in (
            "story_architecture", "master_timeline", "technology_rules",
            "theme_reinforcement", "character_fate_registry",
            "character_arc_plan", "antagonist_motivation_plan",
            "pov_focal_character_plan",
        )
    }

    try:
        planning, new_hashes = run_selective_planning(
            title=_title,
            premise=_premise,
            genre=_genre,
            chapter_list=_chapters,
            character_list=_characters,
            special_instructions=_instructions,
            narrative_perspective=narrative_perspective,
            prev_hashes=session.get("_agent_input_hashes", {}),
            prev_results=_prev_results,
        )
    except RuntimeError as exc:
        logger.error("Outline re-planning failed: %s", exc)
        return jsonify({"error": friendly_llm_error(exc)}), 502

    # -------------------------------------------------------------------------
    # ALL recomputations succeeded.  Commit the full working copy to the
    # session in one block so partial state is never observable.
    # -------------------------------------------------------------------------
    working.update(planning)
    for key, value in working.items():
        session[key] = value
    session["_agent_input_hashes"] = new_hashes

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

"""Outline generation and approval routes."""

import hashlib
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor

from flask import Blueprint, Response, jsonify, request, session

from novelforge import limiter
from novelforge.validation import validate_outline_input
from novelforge.llm.client import call_llm, parse_llm_json, friendly_llm_error
from novelforge.voice import select_voice_seed, format_voice_prompt
from novelforge.agents.chapter import (
    build_title_prompt, build_outline_prompt, build_characters_prompt,
    collect_existing_character_names,
)
from novelforge.agents.planning import (
    plan_story_architecture, plan_master_timeline, plan_character_fate_registry,
    plan_character_arc_plan, plan_antagonist_motivation_plan,
    plan_technology_rules, plan_theme_reinforcement, plan_pov_focal_character,
)
from novelforge.session.persistence import save_session_state

logger = logging.getLogger(__name__)

outline_bp = Blueprint("outline", __name__)


def _input_hash(*args: object) -> str:
    """Compute a stable hash of serialised inputs for cache comparison."""
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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

        # Auto-save session state after outline generation for crash recovery
        save_session_state()

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
                        ch[field] = val.replace(old_name, new_name)
        for session_field in ("premise", "special_events", "special_instructions"):
            text = session.get(session_field, "")
            if isinstance(text, str) and text:
                for old_name, new_name in rename_map.items():
                    text = text.replace(old_name, new_name)
                working[session_field] = text
        # Update narrative perspective if the POV character was renamed
        cur_persp = session.get("narrative_perspective", "third_person")
        if cur_persp.startswith("first_person:"):
            cur_pov_name = cur_persp[len("first_person:"):].strip()
            if cur_pov_name in rename_map:
                working["narrative_perspective"] = f"first_person:{rename_map[cur_pov_name]}"

    # Validate narrative perspective and store in working copy
    narrative_perspective = str(data.get("narrative_perspective", "third_person")).strip()
    if narrative_perspective != "third_person" and not narrative_perspective.startswith("first_person:"):
        narrative_perspective = "third_person"
    if narrative_perspective.startswith("first_person:"):
        pov_char_name = narrative_perspective[len("first_person:"):].strip()
        if len(pov_char_name) > 100:
            return jsonify({"error": "Perspective character name too long."}), 400
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
        return working.get(key, session.get(key, default))

    # Common kwargs shared by all planning agents (use working-copy values so
    # rename propagation is reflected even before session is committed)
    _title = str(_from_working("title"))
    _premise = str(_from_working("premise"))
    _genre = str(_from_working("genre", ""))
    _chapters = list(_from_working("chapter_list", []))  # type: ignore[call-overload]
    _characters = list(_from_working("character_list", []))  # type: ignore[call-overload]
    _instructions = str(_from_working("special_instructions"))
    common = dict(
        title=_title, premise=_premise, genre=_genre,
        chapter_list=_chapters, special_instructions=_instructions,
    )

    # Compute per-agent input hashes and compare with cached values
    prev_hashes = session.get("_agent_input_hashes", {})

    # Base hash: inputs shared by chapter-only agents
    base_inputs = (_title, _premise, _genre, _chapters, _instructions)
    # Extended hash: adds character list
    char_inputs = (*base_inputs, _characters)

    new_hashes = {
        "story_architecture":       _input_hash(*base_inputs),
        "technology_rules":         _input_hash(*base_inputs),
        "theme_reinforcement":      _input_hash(*base_inputs),
        "master_timeline":          _input_hash(*char_inputs),
        "character_arc_plan":       _input_hash(*char_inputs),
    }
    # These depend on Group 1 outputs — hash computed after Group 1 runs

    def _needs_regen(agent_key: str) -> bool:
        return bool(new_hashes.get(agent_key) != prev_hashes.get(agent_key))

    try:
        # --- Group 1: independent agents (run in parallel, skip if unchanged) ---
        # Results are collected into g1_results; the session is not touched yet.
        g1_results: dict[str, dict] = {}
        g1_agents: dict[str, Future[dict]] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            if _needs_regen("story_architecture"):
                g1_agents["story_architecture"] = pool.submit(plan_story_architecture, **common)
            if _needs_regen("master_timeline"):
                g1_agents["master_timeline"] = pool.submit(
                    plan_master_timeline, character_list=_characters, **common,
                )
            if _needs_regen("technology_rules"):
                g1_agents["technology_rules"] = pool.submit(plan_technology_rules, **common)
            if _needs_regen("theme_reinforcement"):
                g1_agents["theme_reinforcement"] = pool.submit(plan_theme_reinforcement, **common)

            for key, fut in g1_agents.items():
                g1_results[key] = fut.result()  # may raise; session still untouched

        skipped_g1 = 4 - len(g1_agents)
        if skipped_g1:
            logger.info("Skipped %d unchanged Group 1 planning agents", skipped_g1)

        # Merge Group 1 results into the working copy; unchanged agents fall back
        # to their existing session values.
        for key in ("story_architecture", "master_timeline", "technology_rules", "theme_reinforcement"):
            working[key] = g1_results.get(key, session.get(key, {}))

        # Now compute Group 2 hashes (depend on master_timeline working value)
        _master_timeline = working["master_timeline"]
        new_hashes["character_fate_registry"] = _input_hash(*char_inputs, _master_timeline)
        new_hashes["antagonist_motivation_plan"] = _input_hash(*char_inputs, _master_timeline)

        # --- Group 2: depend on Group 1 outputs (run in parallel, skip if unchanged) ---
        g2_results: dict[str, dict] = {}
        g2_agents: dict[str, Future[dict]] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            if _needs_regen("character_fate_registry"):
                g2_agents["character_fate_registry"] = pool.submit(
                    plan_character_fate_registry,
                    character_list=_characters,
                    master_timeline=_master_timeline, **common,
                )
            if _needs_regen("character_arc_plan"):
                g2_agents["character_arc_plan"] = pool.submit(
                    plan_character_arc_plan,
                    character_list=_characters, **common,
                )
            if _needs_regen("antagonist_motivation_plan"):
                g2_agents["antagonist_motivation_plan"] = pool.submit(
                    plan_antagonist_motivation_plan,
                    character_list=_characters,
                    master_timeline=_master_timeline, **common,
                )

            for key, fut in g2_agents.items():
                g2_results[key] = fut.result()  # may raise; session still untouched

        skipped_g2 = 3 - len(g2_agents)
        if skipped_g2:
            logger.info("Skipped %d unchanged Group 2 planning agents", skipped_g2)

        # Merge Group 2 results into the working copy
        for key in ("character_fate_registry", "character_arc_plan", "antagonist_motivation_plan"):
            working[key] = g2_results.get(key, session.get(key, {}))

        # --- Group 3: depends on Group 2 (character_arc_plan) + narrative perspective ---
        _character_arc_plan = working["character_arc_plan"]
        pov_hash = _input_hash(*char_inputs, _character_arc_plan, narrative_perspective)
        new_hashes["pov_focal_character_plan"] = pov_hash
        if pov_hash != prev_hashes.get("pov_focal_character_plan"):
            working["pov_focal_character_plan"] = plan_pov_focal_character(
                character_list=_characters,
                character_arc_plan=_character_arc_plan,
                narrative_perspective=narrative_perspective, **common,
            )  # may raise; session still untouched
        else:
            logger.info("Skipped unchanged POV & Focal Character Planner")
            working["pov_focal_character_plan"] = session.get("pov_focal_character_plan", {})

    except RuntimeError as exc:
        logger.error("Outline re-planning failed: %s", exc)
        return jsonify({"error": friendly_llm_error(exc)}), 502

    # -------------------------------------------------------------------------
    # ALL recomputations succeeded.  Commit the full working copy to the
    # session in one block so partial state is never observable.
    # -------------------------------------------------------------------------
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

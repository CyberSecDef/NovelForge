"""Prompt builder functions for chapter drafting, refinement, and post-manuscript audits.

Every function in this module is a pure transformer: data in, prompt messages
out.  No LLM calls, no filesystem I/O, no side effects.
"""

from novelforge.llm.prompts import render_prompt
from novelforge.names import format_name_pool_for_prompt

from novelforge.agents.chapter._helpers import (
    get_forbidden_words,
    get_soft_limited_words,
)


# ---------------------------------------------------------------------------
# Outline / title / characters prompt builders
# ---------------------------------------------------------------------------

def build_title_prompt(premise: str, genre: str) -> list[dict[str, str]]:
    """Build the title generation prompt from premise and genre."""
    return render_prompt("title", premise=premise, genre=genre)


def build_outline_prompt(
    premise: str, genre: str, chapters: int, word_count: int,
    special_events: str, special_instructions: str,
) -> list[dict[str, str]]:
    """Build the chapter outline prompt from premise, genre, and word count."""
    return render_prompt(
        "outline", premise=premise, genre=genre, chapters=chapters,
        word_count=f"{word_count:,}", special_events=special_events or "",
        special_instructions=special_instructions or "",
    )


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
    procedural_exemplars: str = "", chapter_openings_log: str = "",
    consequence_log: str = "", total_chapters: int = 0,
) -> list[dict[str, str]]:
    """Build the initial chapter draft prompt with all planning context."""
    return render_prompt(
        "chapter_draft",
        title=title, genre=genre, premise=premise,
        chapter_num=chapter_num, chapter_title=chapter_title,
        total_chapters=total_chapters,
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
        procedural_exemplars=procedural_exemplars or "",
        chapter_openings_log=chapter_openings_log or "",
        consequence_log=consequence_log or "",
        forbidden_words=", ".join(get_forbidden_words(genre)),
        soft_limited_words=", ".join(get_soft_limited_words(genre)),
        voice_prompt=voice_prompt or "",
        perspective_prompt=perspective_prompt or "",
    )


def build_chapter_expansion_prompt(
    chapter_text: str,
    current_words: int,
    target_words: int,
    min_words: int,
) -> list[dict[str, str]]:
    """Build the expansion prompt for under-length chapters."""
    return render_prompt(
        "chapter_expansion",
        chapter_text=chapter_text,
        current_words=f"{current_words:,}",
        target_words=f"{target_words:,}",
        min_words=f"{min_words:,}",
    )


# ---------------------------------------------------------------------------
# Chapter refinement agent prompt builders
# ---------------------------------------------------------------------------

def build_prose_refinement_agent_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    """Build the prose refinement prompt for dialogue and scene momentum."""
    return render_prompt("prose_refinement_agent", title=title, chapter_num=chapter_num, chapter_text=chapter_text)


def build_voice_dialogue_differentiation_prompt(
    chapter_text: str, chapter_num: int, title: str,
    characters_text: str, perspective_prompt: str = "",
) -> list[dict[str, str]]:
    """Build the voice and dialogue differentiation prompt for character-specific speech."""
    return render_prompt(
        "voice_dialogue_differentiation",
        title=title, chapter_num=chapter_num,
        characters_text=characters_text,
        perspective_prompt=perspective_prompt or "",
        chapter_text=chapter_text,
    )


def build_scene_variety_compression_auditor_prompt(
    chapter_text: str, chapter_summary: str, chapter_num: int, title: str,
    previous_summaries: str = "",
) -> list[dict[str, str]]:
    """Build the scene variety and compression audit prompt."""
    return render_prompt("scene_variety_compression_auditor", title=title, chapter_num=chapter_num,
                         chapter_summary=chapter_summary, previous_summaries=previous_summaries or "",
                         chapter_text=chapter_text)


def build_structure_agent_prompt(chapter_text: str, chapter_num: int, total_chapters: int, outline_summary: str,
                                  chapter_architecture_context: str = "",
                                  chapter_rhythm_shape: str = "") -> list[dict[str, str]]:
    """Build the structure validation prompt with phase hints from ChapterPosition."""
    from novelforge.chapter_position import ChapterPosition
    phase_hint = ChapterPosition(chapter_num, total_chapters).get_structure_phase_hint()
    return render_prompt(
        "structure_agent", chapter_num=chapter_num, total_chapters=total_chapters,
        phase_hint=phase_hint, outline_summary=outline_summary,
        chapter_architecture_context=chapter_architecture_context or "",
        chapter_rhythm_shape=chapter_rhythm_shape or "",
        chapter_text=chapter_text,
    )


def build_character_agent_prompt(chapter_text: str, characters_text: str, chapter_num: int, title: str,
                                  chapter_fate_context: str = "", chapter_arc_context: str = "",
                                  chapter_antagonist_context: str = "", chapter_pov_context: str = "",
                                  perspective_prompt: str = "") -> list[dict[str, str]]:
    """Build the character deepening prompt with fate, arc, and POV context."""
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
    """Build the continuity and context analysis prompt."""
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
    """Build the voice and theme synthesis prompt."""
    return render_prompt("synthesizer", title=title, genre=genre, chapter_num=chapter_num,
                         perspective_prompt=perspective_prompt or "", chapter_text=chapter_text)


def build_quality_controller_prompt(chapter_text: str, chapter_num: int, title: str,
                                    genre: str = "", total_chapters: int = 0) -> list[dict[str, str]]:
    """Build the engagement, pacing, and tension quality check prompt."""
    return render_prompt("quality_controller", title=title, chapter_num=chapter_num,
                         chapter_text=chapter_text, genre=genre, total_chapters=total_chapters)


def build_editing_agent_prompt(chapter_text: str, chapter_summary: str, chapter_num: int, title: str, scene_audit_directives: str = "") -> list[dict[str, str]]:
    """Build the developmental editing prompt with scene audit directives."""
    return render_prompt(
        "editing_agent", title=title, chapter_num=chapter_num,
        chapter_summary=chapter_summary,
        scene_audit_directives=scene_audit_directives or "",
        chapter_text=chapter_text,
    )


def build_narrative_momentum_distinctiveness_prompt(
    chapter_text: str, previous_summaries: str, chapter_summary: str, chapter_num: int, title: str, total_chapters: int,
    chapter_rhythm_shape: str = "",
) -> list[dict[str, str]]:
    """Build the cross-chapter redundancy and escalation prompt."""
    from novelforge.chapter_position import ChapterPosition
    escalation_target = ChapterPosition(chapter_num, total_chapters).get_escalation_target()
    return render_prompt(
        "narrative_momentum_distinctiveness", title=title, chapter_num=chapter_num,
        total_chapters=total_chapters, escalation_target=escalation_target,
        chapter_summary=chapter_summary, previous_summaries=previous_summaries or "",
        chapter_rhythm_shape=chapter_rhythm_shape or "",
        chapter_text=chapter_text,
    )


def build_human_oddities_prompt(
    chapter_text: str, chapter_num: int, title: str,
    total_chapters: int, characters_text: str,
) -> list[dict[str, str]]:
    """Build the human oddities injection prompt for non-plot-serving moments."""
    return render_prompt(
        "human_oddities",
        title=title, chapter_num=chapter_num,
        total_chapters=total_chapters,
        characters_text=characters_text,
        chapter_text=chapter_text,
    )


def build_operational_distinctiveness_prompt(chapter_text: str, previous_summaries: str, chapter_summary: str,
                                              chapter_num: int, title: str,
                                              chapter_rhythm_shape: str = "") -> list[dict[str, str]]:
    """Build the operational distinctiveness prompt for strategy variation."""
    return render_prompt(
        "operational_distinctiveness", title=title, chapter_num=chapter_num,
        chapter_summary=chapter_summary, previous_summaries=previous_summaries or "",
        chapter_rhythm_shape=chapter_rhythm_shape or "",
        chapter_text=chapter_text,
    )


def build_polish_agent_prompt(chapter_text: str, chapter_num: int, title: str, genre: str) -> list[dict[str, str]]:
    """Build the grammar, style, and vivid language polish prompt."""
    return render_prompt("polish_agent", title=title, genre=genre, chapter_num=chapter_num, chapter_text=chapter_text)


def build_anti_llm_agent_prompt(chapter_text: str, chapter_num: int, title: str,
                                genre: str = "") -> list[dict[str, str]]:
    """Build the anti-LLM pattern removal prompt with forbidden word lists."""
    return render_prompt(
        "anti_llm_agent", title=title, chapter_num=chapter_num,
        chapter_text=chapter_text, forbidden_words=", ".join(get_forbidden_words(genre)),
        soft_limited_words=", ".join(get_soft_limited_words(genre)),
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


def build_metaphor_reduction_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    """Build the metaphor density reduction prompt."""
    return render_prompt("metaphor_reduction", title=title, chapter_num=chapter_num, chapter_text=chapter_text)


def build_copy_edit_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    """Build the copy-edit prompt for prose repetitions and dash cleanup."""
    return render_prompt("copy_edit", title=title, chapter_num=chapter_num, chapter_text=chapter_text)


def build_chapter_summary_prompt(chapter_text: str, chapter_num: int) -> list[dict[str, str]]:
    """Build the 100-200 word continuity summary prompt."""
    return render_prompt("chapter_summary", chapter_num=chapter_num, chapter_text=chapter_text)


# ---------------------------------------------------------------------------
# Pre-chapter pass prompt builders
# ---------------------------------------------------------------------------

def build_continuity_gatekeeper_prompt(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str,
    chapter_timeline_context: str = "", chapter_fate_context: str = "",
    chapter_arc_context: str = "", character_state_log: str = "",
) -> list[dict[str, str]]:
    """Build the pre-chapter continuity validation prompt."""
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


def build_rhythm_compliance_verifier_prompt(
    chapter_text: str, chapter_num: int, title: str,
    chapter_rhythm_shape: str, chapter_rhythm_reason: str,
) -> list[dict[str, str]]:
    """Build the rhythm compliance verification prompt."""
    return render_prompt(
        "rhythm_compliance_verifier", title=title, chapter_num=chapter_num,
        chapter_rhythm_shape=chapter_rhythm_shape,
        chapter_rhythm_reason=chapter_rhythm_reason,
        chapter_text=chapter_text,
    )


def build_chapter_rhythm_classifier_prompt(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str, title: str,
    chapter_architecture_context: str = "",
    rhythm_log: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Build the chapter rhythm classification prompt."""
    formatted_rhythm_log = ""
    if rhythm_log:
        lines = []
        for entry in rhythm_log:
            lines.append(f"Chapter {entry['chapter']}: rhythm = {entry['recommended']}")
        formatted_rhythm_log = "\n".join(lines)
    return render_prompt(
        "chapter_rhythm_classifier", title=title, chapter_num=chapter_num,
        chapter_title=chapter_title, chapter_summary=chapter_summary,
        previous_summaries=previous_summaries or "",
        chapter_architecture_context=chapter_architecture_context or "",
        rhythm_log=formatted_rhythm_log,
    )


# ---------------------------------------------------------------------------
# Post-chapter pass prompt builders
# ---------------------------------------------------------------------------

def build_per_chapter_compression_check_prompt(chapter_num: int, chapter_summary: str, previous_summaries: str, title: str) -> list[dict[str, str]]:
    """Build the post-chapter compression check prompt."""
    return render_prompt(
        "per_chapter_compression_check", title=title, chapter_num=chapter_num,
        chapter_summary=chapter_summary, previous_summaries=previous_summaries,
    )


def build_character_state_updater_prompt(chapter_text: str, chapter_summary: str, characters_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    """Build the post-chapter character state tracking prompt."""
    return render_prompt(
        "character_state_updater", title=title, chapter_num=chapter_num,
        characters_text=characters_text, chapter_summary=chapter_summary,
        chapter_text=chapter_text,
    )


def build_chapter_pattern_extractor_prompt(chapter_text: str, chapter_num: int, title: str) -> list[dict[str, str]]:
    """Build the post-chapter pattern extraction prompt for procedural exemplar and opening style tracking."""
    return render_prompt(
        "chapter_pattern_extractor", title=title, chapter_num=chapter_num,
        chapter_text=chapter_text,
    )


# ---------------------------------------------------------------------------
# Chapter revision prompt builder
# ---------------------------------------------------------------------------

def build_chapter_revision_prompt(
    chapter_text: str, chapter_num: int, title: str, chapter_outline_summary: str, revision_instructions: str,
    chapter_architecture_context: str = "", chapter_timeline_context: str = "",
    chapter_fate_context: str = "", chapter_arc_context: str = "",
    chapter_antagonist_context: str = "", chapter_technology_context: str = "",
    chapter_theme_context: str = "", gatekeeper_brief: str = "",
    perspective_prompt: str = "",
) -> list[dict[str, str]]:
    """Build the chapter revision prompt with user instructions and planning context."""
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


# ---------------------------------------------------------------------------
# Post-manuscript audit prompt builders
# ---------------------------------------------------------------------------

def build_consistency_pass_prompt(title: str, all_summaries: list[str], special_instructions: str,
                                  genre: str = "") -> list[dict[str, str]]:
    """Build the post-manuscript consistency audit prompt."""
    summaries_text = "\n\n".join(
        f"Chapter {i+1}:\n{s}" for i, s in enumerate(all_summaries)
    )
    return render_prompt("consistency_pass", title=title, summaries_text=summaries_text, genre=genre,
                         special_instructions=special_instructions)


def build_global_continuity_auditor_prompt(title: str, all_summaries: list[str], character_state_log: list[str],
                                            master_timeline: dict, character_fate_registry: dict) -> list[dict[str, str]]:
    """Build the global continuity audit prompt with timeline and fate registry."""
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
    """Build the narrative compression analysis prompt."""
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
    """Build the character resolution validation prompt with arc and fate data."""
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
    """Build the thematic payoff analysis prompt."""
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
    """Build the climax integrity check prompt with protagonist arc data."""
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
    """Build the loose thread resolution prompt."""
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


def build_reader_immersion_tester_prompt(title: str, all_summaries: list[str], character_arc_plan: dict | None = None, thematic_report: dict | None = None, genre: str = "") -> list[dict[str, str]]:
    """Build the reader immersion and engagement testing prompt."""
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
                         arc_block=arc_block, theme_block=theme_block, genre=genre)


def build_pacing_tension_heatmap_prompt(title: str, all_summaries: list[str], total_chapters: int) -> list[dict[str, str]]:
    """Build the per-chapter pacing and tension heatmap prompt."""
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))
    return render_prompt("pacing_tension_heatmap", title=title, total_chapters=total_chapters,
                         summaries_text=summaries_text)


def build_character_relationship_prompt(
    title: str, genre: str, character_list: list[dict], all_summaries: list[str],
) -> list[dict[str, str]]:
    """Build the character relationship mapping prompt."""
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
    """Build the illustration prompt generation prompt for cover and scene art."""
    characters_text = "\n".join(
        f"- {c.get('name', '?')}: role={c.get('role', '')}; background={c.get('background', '')}"
        for c in character_list
    )
    if not characters_text.strip():
        characters_text = "- No explicit characters provided."
    summaries_text = "\n\n".join(f"Chapter {i + 1}:\n{s}" for i, s in enumerate(all_summaries))
    return render_prompt("illustration_prompt_generator", title=title, genre=genre,
                         premise=premise, characters_text=characters_text, summaries_text=summaries_text)

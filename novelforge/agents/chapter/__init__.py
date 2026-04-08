"""Chapter-level agents: prompt builders, the agent pipeline, and audit builders.

All public symbols are re-exported here so that existing imports of the form
``from novelforge.agents.chapter import build_chapter_draft_prompt`` continue
to work unchanged after the module was split into submodules.
"""

# -- LLM imports (kept at package level so test fixtures can patch them) -----

from novelforge.llm.client import call_llm, parse_llm_json  # noqa: F401

# -- Helpers & constants -----------------------------------------------------

from novelforge.agents.chapter._helpers import (  # noqa: F401
    PASS_FAILURE_KEY,
    PASS_OPTIONAL,
    PASS_REQUIRED,
    _CONTENT_RETRY_LIMIT,
    _DRAFT_CONTENT_NOTE,
    _FORBIDDEN_WORDS,
    _HARD_BAN_THRESHOLD,
    _OVERUSED_PATTERNS,
    _PATTERN_THRESHOLD,
    _SOFT_LIMIT_PER_CHAPTER,
    _SOFT_LIMITED_WORDS,
    _call_with_content_retry,
    _draft_with_content_retry,
    _format_anti_repetition_rules,
    _log_pass_failure,
    _sanitize_for_content_policy,
    scan_vocabulary_overuse,
)

# -- Context & utilities -----------------------------------------------------

from novelforge.agents.chapter.context import (  # noqa: F401
    ChapterContext,
    _format_characters,
    build_perspective_prompt,
    collect_existing_character_names,
)

# -- Prompt builders ---------------------------------------------------------

from novelforge.agents.chapter.prompts import (  # noqa: F401
    build_anti_llm_agent_prompt,
    build_chapter_draft_prompt,
    build_chapter_revision_prompt,
    build_chapter_rhythm_classifier_prompt,
    build_chapter_summary_prompt,
    build_character_agent_prompt,
    build_character_relationship_prompt,
    build_character_resolution_validator_prompt,
    build_character_state_updater_prompt,
    build_characters_prompt,
    build_climax_integrity_checker_prompt,
    build_consistency_pass_prompt,
    build_context_analyzer_prompt,
    build_continuity_gatekeeper_prompt,
    build_copy_edit_prompt,
    build_editing_agent_prompt,
    build_global_continuity_auditor_prompt,
    build_human_oddities_prompt,
    build_illustration_prompt_generator_prompt,
    build_loose_thread_resolver_prompt,
    build_metaphor_reduction_prompt,
    build_narrative_compression_editor_prompt,
    build_narrative_momentum_distinctiveness_prompt,
    build_operational_distinctiveness_prompt,
    build_outline_prompt,
    build_pacing_tension_heatmap_prompt,
    build_polish_agent_prompt,
    build_prose_refinement_agent_prompt,
    build_quality_controller_prompt,
    build_reader_immersion_tester_prompt,
    build_scene_variety_compression_auditor_prompt,
    build_structure_agent_prompt,
    build_synthesizer_prompt,
    build_thematic_payoff_analyzer_prompt,
    build_voice_dialogue_differentiation_prompt,
    build_title_prompt,
    build_vocabulary_fix_prompt,
    build_per_chapter_compression_check_prompt,
)

# -- Pipeline & runners ------------------------------------------------------

from novelforge.agents.chapter.pipeline import (  # noqa: F401
    _run_all_chapter_agents,
    run_chapter_rhythm_classifier,
    run_character_state_updater,
    run_continuity_gatekeeper,
    run_per_chapter_compression_check,
    run_scene_variety_compression_auditor,
)

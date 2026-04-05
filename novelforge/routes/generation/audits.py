"""Post-manuscript audit functions.

Each audit follows the same pattern: build prompt, call LLM with json_mode,
parse response, fall back to a default dict on parse failure, update progress.

Fallback structures are defined as module-level constants so that consumers
(e.g. ``routes/export.py``) can rely on every key being present even when the
LLM call fails or returns unparseable output.
"""

import json
import logging

from novelforge.llm.client import call_llm, parse_llm_json
from novelforge.progress import progress_manager

from novelforge.agents.chapter import (
    build_consistency_pass_prompt,
    build_global_continuity_auditor_prompt,
    build_narrative_compression_editor_prompt,
    build_character_resolution_validator_prompt,
    build_thematic_payoff_analyzer_prompt,
    build_climax_integrity_checker_prompt,
    build_loose_thread_resolver_prompt,
    build_reader_immersion_tester_prompt,
    build_pacing_tension_heatmap_prompt,
    build_character_relationship_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback structures — one per audit, used when LLM parsing fails.
# Keys must match what routes/export.py reads via .get().
# ---------------------------------------------------------------------------

CONSISTENCY_FALLBACK: dict = {
    "issues": [],
    "overall_assessment": "",
}

GLOBAL_CONTINUITY_AUDIT_FALLBACK: dict = {
    "contradictions": [],
    "character_state_errors": [],
    "timeline_errors": [],
    "location_errors": [],
    "overall_integrity": "unknown",
    "overall_assessment": "",
}

NARRATIVE_COMPRESSION_FALLBACK: dict = {
    "redundant_sequences": [],
    "emotional_beat_repetitions": [],
    "compression_priority": "unknown",
    "overall_assessment": "",
}

CHARACTER_RESOLUTION_FALLBACK: dict = {
    "character_resolutions": [],
    "unresolved_characters": [],
    "resolution_integrity": "unknown",
    "overall_assessment": "",
}

THEMATIC_PAYOFF_FALLBACK: dict = {
    "theme_payoffs": [],
    "abandoned_themes": [],
    "weak_payoffs": [],
    "thematic_integrity": "unknown",
    "overall_assessment": "",
}

CLIMAX_INTEGRITY_FALLBACK: dict = {
    "climax_decision_present": False,
    "decision_is_active": False,
    "moral_dimension_present": False,
    "arc_resolved": False,
    "protagonist_is_agent": False,
    "climax_chapter": None,
    "integrity_failures": [],
    "climax_integrity": "unknown",
    "overall_assessment": "",
}

LOOSE_THREAD_FALLBACK: dict = {
    "unresolved_threads": [],
    "dangling_setup_elements": [],
    "intentionally_open_threads": [],
    "thread_integrity": "unknown",
    "overall_assessment": "",
}

READER_IMMERSION_FALLBACK: dict = {
    "pacing_assessment": "unknown",
    "tension_curve": "unknown",
    "stakes_clarity": "unknown",
    "engagement_score": 0,
    "weak_chapters": [],
    "immersion_breaks": [],
    "reader_experience_highlights": [],
    "overall_rating": "unknown",
    "recommendations": [],
}

PACING_HEATMAP_FALLBACK: dict = {
    "chapter_metrics": [],
    "flat_sections": [],
    "overall_pacing_assessment": "",
}

CHARACTER_RELATIONSHIP_FALLBACK: dict = {
    "characters": [],
    "relationships": [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_audit_json(raw: str, fallback: dict) -> dict:
    """Parse LLM JSON output, returning *fallback* on failure.

    Ensures the result is always a dict (rejects JSON arrays) and merges
    any missing keys from *fallback* so the caller can rely on every
    expected key being present.
    """
    try:
        parsed = parse_llm_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from LLM, got an array")
        # Merge fallback keys so consumers never hit a missing key
        merged = dict(fallback)
        merged.update(parsed)
        return merged
    except (json.JSONDecodeError, ValueError):
        return dict(fallback)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_post_manuscript_audits(
    *,
    token: str,
    title: str,
    genre: str,
    summaries: list[str],
    special_instructions: str,
    character_list: list[dict],
    character_state_log: list[str],
    master_timeline: dict,
    character_fate_registry: dict,
    character_arc_plan: dict,
    theme_reinforcement: dict,
    total_chapters: int,
) -> dict:
    """Run all 10 post-manuscript audits and update progress state.

    Returns a dict with consistency and all audit report fields.
    """
    # --- Final consistency pass ---
    progress_manager.update(token, {"step": "Final consistency pass"})
    consistency_raw = call_llm(
        build_consistency_pass_prompt(title, summaries, special_instructions),
        action="Final consistency pass", json_mode=True,
    )
    consistency = _parse_audit_json(consistency_raw, CONSISTENCY_FALLBACK)

    # --- Global Continuity Audit ---
    progress_manager.update(token, {"step": "Global continuity audit"})
    audit_raw = call_llm(
        build_global_continuity_auditor_prompt(
            title=title, all_summaries=summaries,
            character_state_log=character_state_log,
            master_timeline=master_timeline,
            character_fate_registry=character_fate_registry,
        ),
        action="Global continuity audit", json_mode=True,
    )
    global_audit = _parse_audit_json(audit_raw, GLOBAL_CONTINUITY_AUDIT_FALLBACK)

    progress_manager.update(token, {
        "consistency": consistency,
        "global_continuity_audit": global_audit,
    })

    # --- Narrative Compression Edit ---
    progress_manager.update(token, {"step": "Narrative compression analysis"})
    compression_raw = call_llm(
        build_narrative_compression_editor_prompt(
            title=title, all_summaries=summaries, continuity_audit=global_audit,
        ),
        action="Narrative compression analysis", json_mode=True,
    )
    compression_report = _parse_audit_json(compression_raw, NARRATIVE_COMPRESSION_FALLBACK)

    progress_manager.update(token, {"narrative_compression_report": compression_report})

    # --- Character Resolution Validation ---
    progress_manager.update(token, {"step": "Character resolution validation"})
    resolution_raw = call_llm(
        build_character_resolution_validator_prompt(
            title=title, all_summaries=summaries,
            character_arc_plan=character_arc_plan,
            character_fate_registry=character_fate_registry,
            character_state_log=character_state_log,
        ),
        action="Character resolution validation", json_mode=True,
    )
    resolution_report = _parse_audit_json(resolution_raw, CHARACTER_RESOLUTION_FALLBACK)

    progress_manager.update(token, {"character_resolution_report": resolution_report})

    # --- Thematic Payoff Analysis ---
    progress_manager.update(token, {"step": "Thematic payoff analysis"})
    thematic_raw = call_llm(
        build_thematic_payoff_analyzer_prompt(
            title=title, all_summaries=summaries,
            theme_reinforcement=theme_reinforcement, total_chapters=total_chapters,
        ),
        action="Thematic payoff analysis", json_mode=True,
    )
    thematic_report = _parse_audit_json(thematic_raw, THEMATIC_PAYOFF_FALLBACK)

    progress_manager.update(token, {"thematic_payoff_report": thematic_report})

    # --- Climax Integrity Check ---
    progress_manager.update(token, {"step": "Climax integrity check"})
    climax_raw = call_llm(
        build_climax_integrity_checker_prompt(
            title=title, all_summaries=summaries,
            character_arc_plan=character_arc_plan, total_chapters=total_chapters,
        ),
        action="Climax integrity check", json_mode=True,
    )
    climax_report = _parse_audit_json(climax_raw, CLIMAX_INTEGRITY_FALLBACK)

    progress_manager.update(token, {"climax_integrity_report": climax_report})

    # --- Loose Thread Resolution ---
    progress_manager.update(token, {"step": "Loose thread resolution"})
    threads_raw = call_llm(
        build_loose_thread_resolver_prompt(
            title=title, all_summaries=summaries,
            character_state_log=character_state_log,
            continuity_audit=global_audit, resolution_report=resolution_report,
        ),
        action="Loose thread resolution", json_mode=True,
    )
    threads_report = _parse_audit_json(threads_raw, LOOSE_THREAD_FALLBACK)

    progress_manager.update(token, {"loose_thread_report": threads_report})

    # --- Reader Immersion Testing ---
    progress_manager.update(token, {"step": "Reader immersion testing"})
    immersion_raw = call_llm(
        build_reader_immersion_tester_prompt(
            title=title, all_summaries=summaries,
            character_arc_plan=character_arc_plan, thematic_report=thematic_report,
        ),
        action="Reader immersion testing", json_mode=True,
    )
    immersion_report = _parse_audit_json(immersion_raw, READER_IMMERSION_FALLBACK)

    progress_manager.update(token, {"reader_immersion_report": immersion_report})

    # --- Pacing & Tension Heatmap ---
    progress_manager.update(token, {"step": "Pacing & tension heatmap"})
    heatmap_raw = call_llm(
        build_pacing_tension_heatmap_prompt(
            title=title, all_summaries=summaries, total_chapters=total_chapters,
        ),
        action="Pacing & tension heatmap", json_mode=True,
    )
    pacing_heatmap = _parse_audit_json(heatmap_raw, PACING_HEATMAP_FALLBACK)

    progress_manager.update(token, {"pacing_heatmap": pacing_heatmap})

    # --- Character Relationship Map ---
    progress_manager.update(token, {"step": "Mapping character relationships"})
    relationship_raw = call_llm(
        build_character_relationship_prompt(
            title=title, genre=genre,
            character_list=character_list, all_summaries=summaries,
        ),
        action="Character relationship mapping", json_mode=True,
    )
    relationship_map = _parse_audit_json(relationship_raw, CHARACTER_RELATIONSHIP_FALLBACK)

    progress_manager.update(token, {"character_relationship_map": relationship_map})

    return {"consistency": consistency}

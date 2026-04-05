"""Planning orchestration service for the outline pipeline.

Both ``generate_outline`` (full run) and ``approve_outline`` (selective
regeneration) delegate here so that group ordering, parallel execution,
dependency handling, and hash-based caching are defined in one place.
"""

import hashlib
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor

from novelforge.agents.planning import (
    plan_antagonist_motivation_plan,
    plan_character_arc_plan,
    plan_character_fate_registry,
    plan_master_timeline,
    plan_pov_focal_character,
    plan_story_architecture,
    plan_technology_rules,
    plan_theme_reinforcement,
)

logger = logging.getLogger(__name__)


def _input_hash(*args: object) -> str:
    """Compute a stable SHA-256 prefix of serialised inputs for cache comparison."""
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_full_planning(
    *,
    title: str,
    premise: str,
    genre: str,
    chapter_list: list,
    character_list: list,
    special_instructions: str,
) -> dict:
    """Run all eight planning agents unconditionally in dependency order.

    Intended for the initial outline generation phase where no prior results
    exist and every agent must execute.

    Returns a dict keyed by agent name with each agent's output dict.
    """
    common = dict(
        title=title,
        premise=premise,
        genre=genre,
        chapter_list=chapter_list,
        special_instructions=special_instructions,
    )

    # --- Group 1: independent agents (run in parallel) ---
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_arch: Future[dict] = pool.submit(plan_story_architecture, **common)
        fut_timeline: Future[dict] = pool.submit(
            plan_master_timeline, character_list=character_list, **common
        )
        fut_tech: Future[dict] = pool.submit(plan_technology_rules, **common)
        fut_theme: Future[dict] = pool.submit(plan_theme_reinforcement, **common)

        story_architecture = fut_arch.result()
        master_timeline = fut_timeline.result()
        technology_rules = fut_tech.result()
        theme_reinforcement = fut_theme.result()

    # --- Group 2: depend on Group 1 outputs (run in parallel) ---
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_fate: Future[dict] = pool.submit(
            plan_character_fate_registry,
            character_list=character_list,
            master_timeline=master_timeline,
            **common,
        )
        fut_arc: Future[dict] = pool.submit(
            plan_character_arc_plan,
            character_list=character_list,
            **common,
        )
        fut_antag: Future[dict] = pool.submit(
            plan_antagonist_motivation_plan,
            character_list=character_list,
            master_timeline=master_timeline,
            **common,
        )

        character_fate_registry = fut_fate.result()
        character_arc_plan = fut_arc.result()
        antagonist_motivation_plan = fut_antag.result()

    # --- Group 3: depends on Group 2 (character_arc_plan) ---
    # narrative_perspective is not yet chosen at the full-planning stage; the
    # agent defaults to "third_person" and the perspective is finalised during
    # the approve_outline phase (run_selective_planning).
    pov_focal_character_plan = plan_pov_focal_character(
        character_list=character_list,
        character_arc_plan=character_arc_plan,
        narrative_perspective="third_person",
        **common,
    )

    return {
        "story_architecture": story_architecture,
        "master_timeline": master_timeline,
        "character_fate_registry": character_fate_registry,
        "character_arc_plan": character_arc_plan,
        "antagonist_motivation_plan": antagonist_motivation_plan,
        "technology_rules": technology_rules,
        "theme_reinforcement": theme_reinforcement,
        "pov_focal_character_plan": pov_focal_character_plan,
    }


def run_selective_planning(
    *,
    title: str,
    premise: str,
    genre: str,
    chapter_list: list,
    character_list: list,
    special_instructions: str,
    narrative_perspective: str,
    prev_hashes: dict,
    prev_results: dict,
) -> tuple[dict, dict]:
    """Run planning agents selectively, skipping those whose inputs are unchanged.

    Agents are grouped by dependency order:

    - Group 1 (parallel): ``story_architecture``, ``master_timeline``,
      ``technology_rules``, ``theme_reinforcement``
    - Group 2 (parallel, depends on Group 1): ``character_fate_registry``,
      ``character_arc_plan``, ``antagonist_motivation_plan``
    - Group 3 (sequential, depends on Group 2): ``pov_focal_character_plan``

    Each agent is skipped when its input hash matches the stored hash in
    ``prev_hashes``; the corresponding value from ``prev_results`` is used
    instead.  Callers must not mutate the live session until this function
    returns successfully so that a mid-flight failure leaves session state
    intact.

    Args:
        title: Novel title (post-rename).
        premise: Story premise (post-rename).
        genre: Story genre.
        chapter_list: Chapter list (post-rename).
        character_list: Character list (post-rename).
        special_instructions: Any special instructions.
        narrative_perspective: Narrative perspective string
            (``"third_person"`` or ``"first_person:<name>"``).
        prev_hashes: Previously stored agent-input hashes from the session.
        prev_results: Previously stored agent outputs from the session, used
            for agents that are skipped.

    Returns:
        A ``(results, new_hashes)`` pair where *results* maps each agent name
        to its (possibly cached) output dict and *new_hashes* maps each agent
        name to the input hash computed during this run.

    Raises:
        Any exception raised by a planning agent propagates to the caller.
    """
    common = dict(
        title=title,
        premise=premise,
        genre=genre,
        chapter_list=chapter_list,
        special_instructions=special_instructions,
    )

    base_inputs = (title, premise, genre, chapter_list, special_instructions)
    char_inputs = (*base_inputs, character_list)

    # Group 1 and Group 2 base hashes (Group 2 hashes that depend on
    # master_timeline are computed after Group 1 completes).
    new_hashes: dict[str, str] = {
        "story_architecture": _input_hash(*base_inputs),
        "technology_rules": _input_hash(*base_inputs),
        "theme_reinforcement": _input_hash(*base_inputs),
        "master_timeline": _input_hash(*char_inputs),
        "character_arc_plan": _input_hash(*char_inputs),
    }

    def _needs_regen(agent_key: str) -> bool:
        """Return True if the agent's input hash differs from the stored hash."""
        return bool(new_hashes.get(agent_key) != prev_hashes.get(agent_key))

    results: dict[str, dict] = {}

    # --- Group 1: independent agents (run in parallel, skip if unchanged) ---
    g1_agents: dict[str, Future[dict]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        if _needs_regen("story_architecture"):
            g1_agents["story_architecture"] = pool.submit(
                plan_story_architecture, **common
            )
        if _needs_regen("master_timeline"):
            g1_agents["master_timeline"] = pool.submit(
                plan_master_timeline, character_list=character_list, **common
            )
        if _needs_regen("technology_rules"):
            g1_agents["technology_rules"] = pool.submit(
                plan_technology_rules, **common
            )
        if _needs_regen("theme_reinforcement"):
            g1_agents["theme_reinforcement"] = pool.submit(
                plan_theme_reinforcement, **common
            )

        g1_results: dict[str, dict] = {}
        for key, fut in g1_agents.items():
            g1_results[key] = fut.result()  # may raise; session still untouched

    skipped_g1 = 4 - len(g1_agents)
    if skipped_g1:
        logger.info("Skipped %d unchanged Group 1 planning agents", skipped_g1)

    for key in ("story_architecture", "master_timeline", "technology_rules", "theme_reinforcement"):
        results[key] = g1_results.get(key, prev_results.get(key, {}))

    # Compute Group 2 hashes that depend on the resolved master_timeline.
    master_timeline = results["master_timeline"]
    new_hashes["character_fate_registry"] = _input_hash(*char_inputs, master_timeline)
    new_hashes["antagonist_motivation_plan"] = _input_hash(*char_inputs, master_timeline)

    # --- Group 2: depend on Group 1 outputs (run in parallel, skip if unchanged) ---
    g2_agents: dict[str, Future[dict]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        if _needs_regen("character_fate_registry"):
            g2_agents["character_fate_registry"] = pool.submit(
                plan_character_fate_registry,
                character_list=character_list,
                master_timeline=master_timeline,
                **common,
            )
        if _needs_regen("character_arc_plan"):
            g2_agents["character_arc_plan"] = pool.submit(
                plan_character_arc_plan,
                character_list=character_list,
                **common,
            )
        if _needs_regen("antagonist_motivation_plan"):
            g2_agents["antagonist_motivation_plan"] = pool.submit(
                plan_antagonist_motivation_plan,
                character_list=character_list,
                master_timeline=master_timeline,
                **common,
            )

        g2_results: dict[str, dict] = {}
        for key, fut in g2_agents.items():
            g2_results[key] = fut.result()  # may raise; session still untouched

    skipped_g2 = 3 - len(g2_agents)
    if skipped_g2:
        logger.info("Skipped %d unchanged Group 2 planning agents", skipped_g2)

    for key in ("character_fate_registry", "character_arc_plan", "antagonist_motivation_plan"):
        results[key] = g2_results.get(key, prev_results.get(key, {}))

    # --- Group 3: depends on Group 2 (character_arc_plan) + narrative_perspective ---
    character_arc_plan = results["character_arc_plan"]
    pov_hash = _input_hash(*char_inputs, character_arc_plan, narrative_perspective)
    new_hashes["pov_focal_character_plan"] = pov_hash

    if pov_hash != prev_hashes.get("pov_focal_character_plan"):
        results["pov_focal_character_plan"] = plan_pov_focal_character(
            character_list=character_list,
            character_arc_plan=character_arc_plan,
            narrative_perspective=narrative_perspective,
            **common,
        )  # may raise; session still untouched
    else:
        logger.info("Skipped unchanged POV & Focal Character Planner")
        results["pov_focal_character_plan"] = prev_results.get("pov_focal_character_plan", {})

    return results, new_hashes

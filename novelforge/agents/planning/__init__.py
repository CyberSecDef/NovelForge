"""Planning agents for NovelForge.

All public symbols are re-exported here so that existing imports of the form
``from novelforge.agents.planning import plan_story_architecture`` continue to
work unchanged after the module was split into per-agent files.
"""

# -- LLM imports (kept at package level so test fixtures can patch them) -----

from novelforge.llm.client import call_llm, parse_llm_json  # noqa: F401

# -- Shared helpers ----------------------------------------------------------

from novelforge.agents.planning._helpers import (  # noqa: F401
    _PLACEHOLDER_CHAPTER,
    _coerce_positive_int,
    _safe_chapter_list,
    choose_story_architecture_mode,
)

# -- Agent classes -----------------------------------------------------------

from novelforge.agents.planning.story_architecture import (  # noqa: F401
    StoryArchitectureAgent,
    get_chapter_architecture_context,
    normalise_story_architecture,
    plan_story_architecture,
)
from novelforge.agents.planning.master_timeline import (  # noqa: F401
    MasterTimelineAgent,
    get_chapter_timeline_context,
    normalise_master_timeline,
    plan_master_timeline,
)
from novelforge.agents.planning.character_fate import (  # noqa: F401
    CharacterFateRegistryAgent,
    get_chapter_fate_context,
    normalise_character_fate_registry,
    plan_character_fate_registry,
)
from novelforge.agents.planning.character_arc import (  # noqa: F401
    CharacterArcPlanAgent,
    get_chapter_arc_context,
    normalise_character_arc_plan,
    plan_character_arc_plan,
)
from novelforge.agents.planning.antagonist_motivation import (  # noqa: F401
    AntagonistMotivationAgent,
    get_chapter_antagonist_context,
    normalise_antagonist_motivation_plan,
    plan_antagonist_motivation_plan,
)
from novelforge.agents.planning.technology_rules import (  # noqa: F401
    TechnologyRulesAgent,
    get_chapter_technology_context,
    normalise_technology_rules,
    plan_technology_rules,
)
from novelforge.agents.planning.theme_reinforcement import (  # noqa: F401
    ThemeReinforcementAgent,
    get_chapter_theme_context,
    normalise_theme_reinforcement,
    plan_theme_reinforcement,
)
from novelforge.agents.planning.pov_focal import (  # noqa: F401
    PovFocalCharacterAgent,
    get_chapter_pov_context,
    normalise_pov_focal_character_plan,
    plan_pov_focal_character,
)

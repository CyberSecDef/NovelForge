"""
Boundary tests for planning-agent normalizers and fallback helpers.

Covers:
- Empty chapter lists
- Invalid chapter numbers
- Non-dict / malformed chapter list entries
- Malformed planner JSON (non-dict, None, invalid types)
- Partial fallback outputs
- _safe_chapter_list helper directly
"""

import pytest

from novelforge.agents.planning import (
    StoryArchitectureAgent,
    MasterTimelineAgent,
    CharacterFateRegistryAgent,
    CharacterArcPlanAgent,
    AntagonistMotivationAgent,
    TechnologyRulesAgent,
    ThemeReinforcementAgent,
    PovFocalCharacterAgent,
    _safe_chapter_list,
    _PLACEHOLDER_CHAPTER,
)


# ---------------------------------------------------------------------------
# _safe_chapter_list helper
# ---------------------------------------------------------------------------

class TestSafeChapterList:
    """Unit tests for the _safe_chapter_list helper."""

    def test_empty_list_returns_placeholder(self):
        result = _safe_chapter_list([])
        assert len(result) == 1
        assert result[0]["number"] == _PLACEHOLDER_CHAPTER["number"]

    def test_none_returns_placeholder(self):
        result = _safe_chapter_list(None)  # type: ignore[arg-type]
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_non_list_returns_placeholder(self):
        result = _safe_chapter_list("bad input")  # type: ignore[arg-type]
        assert len(result) == 1

    def test_list_with_only_non_dicts_returns_placeholder(self):
        result = _safe_chapter_list([None, "string", 42])
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_filters_out_non_dicts_keeps_dicts(self):
        chapters = [None, {"number": 3, "title": "Three"}, "bad", {"number": 5}]
        result = _safe_chapter_list(chapters)
        assert len(result) == 2
        assert result[0]["number"] == 3
        assert result[1]["number"] == 5

    def test_valid_list_returned_unchanged(self):
        chapters = [{"number": 1}, {"number": 2}]
        result = _safe_chapter_list(chapters)
        assert result == chapters

    def test_returns_copy_of_placeholder_not_same_object(self):
        r1 = _safe_chapter_list([])
        r2 = _safe_chapter_list([])
        assert r1 is not r2
        assert r1[0] is not r2[0]


# ---------------------------------------------------------------------------
# StoryArchitectureAgent
# ---------------------------------------------------------------------------

class TestStoryArchitectureNormalizerBoundary:
    agent = StoryArchitectureAgent()

    def test_empty_chapter_list_no_index_error(self):
        result = self.agent.normalise({}, chapter_list=[], total_chapters=1)
        assert isinstance(result, dict)
        assert "chapter_plan" in result
        assert len(result["chapter_plan"]) >= 1

    def test_invalid_chapter_entries_no_attribute_error(self):
        chapter_list = [None, "string", 42]
        result = self.agent.normalise({}, chapter_list=chapter_list, total_chapters=1)
        assert isinstance(result, dict)
        assert "chapter_plan" in result
        assert len(result["chapter_plan"]) >= 1

    def test_mixed_chapter_list_processes_dicts_only(self):
        chapter_list = [None, {"number": 2, "title": "Good", "summary": "s"}, "bad"]
        result = self.agent.normalise({}, chapter_list=chapter_list, total_chapters=2)
        assert isinstance(result, dict)
        numbers = [ch["number"] for ch in result["chapter_plan"]]
        assert 2 in numbers

    def test_malformed_planner_json_returns_fallback(self):
        result = self.agent.normalise(None, chapter_list=[{"number": 1}], total_chapters=1)  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert "chapter_plan" in result

    def test_planner_json_empty_dict_returns_deterministic_result(self):
        result = self.agent.normalise({}, chapter_list=[{"number": 1}], total_chapters=1)
        assert isinstance(result, dict)
        assert len(result["chapter_plan"]) == 1
        assert result["chapter_plan"][0]["number"] == 1

    def test_invalid_chapter_number_uses_index_default(self):
        chapter_list = [{"number": -99, "title": "Bad num"}, {"number": 0}]
        result = self.agent.normalise({}, chapter_list=chapter_list, total_chapters=2)
        assert isinstance(result, dict)
        # _coerce_positive_int(-99, idx) returns idx; no IndexError should occur
        assert "chapter_plan" in result

    def test_partial_planner_output_fills_from_fallback(self):
        chapter_list = [{"number": 1}, {"number": 2}]
        # planner only supplied chapter 2
        planner_data = {
            "chapter_plan": [{"number": 2, "act": "Act I", "phase": "Escalation",
                               "purpose": "X", "escalation": "Y", "operation_limit": 1,
                               "required_turn": "None", "carry_forward": "Z"}]
        }
        result = self.agent.normalise(planner_data, chapter_list=chapter_list, total_chapters=2)
        numbers = {ch["number"] for ch in result["chapter_plan"]}
        assert 1 in numbers
        assert 2 in numbers

    def test_fallback_build_with_empty_chapter_list(self):
        result = StoryArchitectureAgent._build_fallback_impl([], 1)
        assert "chapter_plan" in result
        assert len(result["chapter_plan"]) >= 1

    def test_fallback_build_with_non_dict_items(self):
        result = StoryArchitectureAgent._build_fallback_impl([None, "bad"], 1)
        assert "chapter_plan" in result
        assert len(result["chapter_plan"]) >= 1


# ---------------------------------------------------------------------------
# MasterTimelineAgent
# ---------------------------------------------------------------------------

class TestMasterTimelineNormalizerBoundary:
    agent = MasterTimelineAgent()

    def test_empty_chapter_list_fallback_no_error(self):
        result = self.agent.normalise({}, chapter_list=[], character_list=[])
        assert isinstance(result, dict)
        assert "ledger" in result

    def test_non_dict_chapter_entries_fallback(self):
        result = self.agent.normalise({}, chapter_list=[None, "bad"], character_list=[])
        assert isinstance(result, dict)

    def test_malformed_data_returns_fallback(self):
        result = self.agent.normalise("bad_input", chapter_list=[{"number": 1}], character_list=[])  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert "ledger" in result

    def test_fallback_impl_empty_chapter_list(self):
        result = MasterTimelineAgent._build_fallback_impl([], [])
        assert "ledger" in result
        assert len(result["ledger"]) >= 1

    def test_fallback_impl_non_dict_chapter_list(self):
        result = MasterTimelineAgent._build_fallback_impl([None, "bad", 42], [])
        assert "ledger" in result
        assert len(result["ledger"]) >= 1


# ---------------------------------------------------------------------------
# CharacterArcPlanAgent
# ---------------------------------------------------------------------------

class TestCharacterArcNormalizerBoundary:
    agent = CharacterArcPlanAgent()

    def test_empty_chapter_list_no_error(self):
        result = self.agent.normalise({}, character_list=[], chapter_list=[])
        assert isinstance(result, dict)

    def test_non_dict_chapter_entries_no_error(self):
        result = self.agent.normalise({}, character_list=[], chapter_list=[None, "str"])
        assert isinstance(result, dict)

    def test_malformed_planner_output_uses_fallback(self):
        result = self.agent.normalise(42, character_list=[], chapter_list=[{"number": 1}])  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert "arcs" in result

    def test_fallback_impl_empty_chapter_list(self):
        result = CharacterArcPlanAgent._build_fallback_impl([], [])
        assert "chapter_constraints" in result
        assert len(result["chapter_constraints"]) >= 1

    def test_fallback_impl_non_dict_chapter_list(self):
        result = CharacterArcPlanAgent._build_fallback_impl([], [None, "bad"])
        assert "chapter_constraints" in result
        assert len(result["chapter_constraints"]) >= 1


# ---------------------------------------------------------------------------
# AntagonistMotivationAgent
# ---------------------------------------------------------------------------

class TestAntagonistNormalizerBoundary:
    agent = AntagonistMotivationAgent()

    def test_empty_chapter_list_no_error(self):
        result = self.agent.normalise({}, character_list=[], chapter_list=[])
        assert isinstance(result, dict)

    def test_non_dict_chapter_entries_no_error(self):
        result = self.agent.normalise({}, character_list=[], chapter_list=[None, "x"])
        assert isinstance(result, dict)

    def test_malformed_data_uses_fallback(self):
        result = self.agent.normalise(None, character_list=[], chapter_list=[{"number": 1}])  # type: ignore[arg-type]
        assert isinstance(result, dict)

    def test_fallback_impl_empty_chapter_list(self):
        result = AntagonistMotivationAgent._build_fallback_impl([], [])
        assert "chapter_constraints" in result
        assert len(result["chapter_constraints"]) >= 1

    def test_fallback_impl_non_dict_chapter_list(self):
        result = AntagonistMotivationAgent._build_fallback_impl([], [None, "bad"])
        assert "chapter_constraints" in result
        assert len(result["chapter_constraints"]) >= 1


# ---------------------------------------------------------------------------
# TechnologyRulesAgent
# ---------------------------------------------------------------------------

class TestTechnologyRulesNormalizerBoundary:
    agent = TechnologyRulesAgent()

    def test_empty_chapter_list_no_error(self):
        result = self.agent.normalise({}, chapter_list=[])
        assert isinstance(result, dict)

    def test_non_dict_chapter_entries_no_error(self):
        result = self.agent.normalise({}, chapter_list=[None, "bad"])
        assert isinstance(result, dict)

    def test_malformed_data_uses_fallback(self):
        result = self.agent.normalise(None, chapter_list=[{"number": 1}])  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert "systems" in result

    def test_fallback_impl_empty_chapter_list(self):
        result = TechnologyRulesAgent._build_fallback_impl([])
        assert "chapter_constraints" in result
        assert len(result["chapter_constraints"]) >= 1

    def test_fallback_impl_non_dict_chapter_list(self):
        result = TechnologyRulesAgent._build_fallback_impl([None, "bad"])
        assert "chapter_constraints" in result
        assert len(result["chapter_constraints"]) >= 1


# ---------------------------------------------------------------------------
# ThemeReinforcementAgent
# ---------------------------------------------------------------------------

class TestThemeReinforcementNormalizerBoundary:
    agent = ThemeReinforcementAgent()

    def test_empty_chapter_list_no_error(self):
        result = self.agent.normalise({}, chapter_list=[])
        assert isinstance(result, dict)

    def test_non_dict_chapter_entries_no_error(self):
        result = self.agent.normalise({}, chapter_list=[None, "x"])
        assert isinstance(result, dict)

    def test_malformed_data_uses_fallback(self):
        result = self.agent.normalise(None, chapter_list=[{"number": 1}])  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert "themes" in result

    def test_fallback_impl_empty_chapter_list(self):
        result = ThemeReinforcementAgent._build_fallback_impl([])
        assert "themes" in result
        # chapter_appearances should have placeholder entries
        assert len(result["themes"][0]["chapter_appearances"]) >= 1

    def test_fallback_impl_non_dict_chapter_list(self):
        result = ThemeReinforcementAgent._build_fallback_impl([None, "bad"])
        assert "themes" in result
        assert len(result["themes"][0]["chapter_appearances"]) >= 1


# ---------------------------------------------------------------------------
# PovFocalCharacterAgent
# ---------------------------------------------------------------------------

class TestPovFocalNormalizerBoundary:
    agent = PovFocalCharacterAgent()

    def test_empty_chapter_list_no_error(self):
        result = self.agent.normalise({}, character_list=[], chapter_list=[])
        assert isinstance(result, dict)

    def test_non_dict_chapter_entries_no_error(self):
        result = self.agent.normalise({}, character_list=[], chapter_list=[None, "bad"])
        assert isinstance(result, dict)

    def test_malformed_data_uses_fallback(self):
        result = self.agent.normalise(None, character_list=[], chapter_list=[{"number": 1}])  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert "chapter_pov_plan" in result

    def test_non_dict_character_entries_no_error(self):
        result = self.agent.normalise({}, character_list=[None, "bad"], chapter_list=[{"number": 1}])
        assert isinstance(result, dict)

    def test_fallback_impl_empty_chapter_list(self):
        result = PovFocalCharacterAgent._build_fallback_impl([], [])
        assert "chapter_pov_plan" in result
        assert len(result["chapter_pov_plan"]) >= 1

    def test_fallback_impl_non_dict_chapter_list(self):
        result = PovFocalCharacterAgent._build_fallback_impl([], [None, "bad"])
        assert "chapter_pov_plan" in result
        assert len(result["chapter_pov_plan"]) >= 1


# ---------------------------------------------------------------------------
# CharacterFateRegistryAgent  (uses total_chapters int, not chapter_list iter)
# ---------------------------------------------------------------------------

class TestCharacterFateNormalizerBoundary:
    agent = CharacterFateRegistryAgent()

    def test_empty_chapter_list_no_error(self):
        result = self.agent.normalise({}, character_list=[], chapter_list=[], total_chapters=1)
        assert isinstance(result, dict)

    def test_malformed_data_uses_fallback(self):
        result = self.agent.normalise(None, character_list=[], chapter_list=[{"number": 1}], total_chapters=1)  # type: ignore[arg-type]
        assert isinstance(result, dict)
        assert "registry" in result

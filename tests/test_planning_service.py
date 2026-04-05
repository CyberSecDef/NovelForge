"""Unit tests for novelforge.services.planning orchestration service.

These tests exercise the shared planning orchestration layer directly,
independently of the route handlers, to ensure that:
  - run_full_planning() calls all eight agents in the correct dependency order.
  - run_selective_planning() skips agents whose input hashes are unchanged.
  - run_selective_planning() regenerates agents whose inputs changed.
  - Group dependency ordering is preserved (Group 2 receives Group 1 outputs).
  - Agent failures propagate from both orchestration functions.
  - _input_hash() is stable and deterministic.
"""

from __future__ import annotations

import pytest

from novelforge.services.planning import (
    _input_hash,
    run_full_planning,
    run_selective_planning,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_CHAPTER_LIST = [
    {"number": 1, "title": "The Beginning", "summary": "It begins."},
    {"number": 2, "title": "The Middle", "summary": "Conflict rises."},
    {"number": 3, "title": "The End", "summary": "Resolution."},
]

_CHARACTER_LIST = [
    {
        "name": "Alice",
        "age": "28",
        "role": "Protagonist",
        "background": "A brave explorer.",
        "arc": "Learns to trust others.",
    }
]

_BASE_KWARGS = dict(
    title="Test Novel",
    premise="A hero discovers a hidden world",
    genre="Fantasy",
    chapter_list=_CHAPTER_LIST,
    character_list=_CHARACTER_LIST,
    special_instructions="",
)


def _agent_result(name: str) -> dict:
    """Return a minimal stub dict for a named agent."""
    return {"_agent": name, "_planning_source": "llm"}


def _patch_all_agents(mocker) -> dict:
    """Patch all eight planning agents and return the mock objects."""
    mocks = {}
    for agent in (
        "plan_story_architecture",
        "plan_master_timeline",
        "plan_technology_rules",
        "plan_theme_reinforcement",
        "plan_character_fate_registry",
        "plan_character_arc_plan",
        "plan_antagonist_motivation_plan",
        "plan_pov_focal_character",
    ):
        mocks[agent] = mocker.patch(
            f"novelforge.services.planning.{agent}",
            return_value=_agent_result(agent),
        )
    return mocks


# ---------------------------------------------------------------------------
# _input_hash
# ---------------------------------------------------------------------------


class TestInputHash:
    def test_deterministic(self):
        assert _input_hash("a", "b", 1) == _input_hash("a", "b", 1)

    def test_different_inputs_differ(self):
        assert _input_hash("x") != _input_hash("y")

    def test_returns_16_char_hex(self):
        h = _input_hash("anything")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_order_matters(self):
        assert _input_hash("a", "b") != _input_hash("b", "a")

    def test_dict_sort_keys(self):
        """Dictionaries with same content but different insertion order hash identically."""
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert _input_hash(d1) == _input_hash(d2)


# ---------------------------------------------------------------------------
# run_full_planning
# ---------------------------------------------------------------------------


class TestRunFullPlanning:
    def test_returns_all_eight_agent_keys(self, mocker):
        _patch_all_agents(mocker)
        result = run_full_planning(**_BASE_KWARGS)
        expected_keys = {
            "story_architecture",
            "master_timeline",
            "technology_rules",
            "theme_reinforcement",
            "character_fate_registry",
            "character_arc_plan",
            "antagonist_motivation_plan",
            "pov_focal_character_plan",
        }
        assert set(result.keys()) == expected_keys

    def test_all_agents_called_once(self, mocker):
        mocks = _patch_all_agents(mocker)
        run_full_planning(**_BASE_KWARGS)
        for name, m in mocks.items():
            assert m.call_count == 1, f"{name} was not called exactly once"

    def test_group2_receives_master_timeline_from_group1(self, mocker):
        """character_fate_registry and antagonist_motivation_plan must receive
        the master_timeline output produced by plan_master_timeline."""
        mocks = _patch_all_agents(mocker)
        expected_timeline = {"ledger": ["event1"]}
        mocks["plan_master_timeline"].return_value = expected_timeline

        run_full_planning(**_BASE_KWARGS)

        fate_call_kwargs = mocks["plan_character_fate_registry"].call_args.kwargs
        antag_call_kwargs = mocks["plan_antagonist_motivation_plan"].call_args.kwargs
        assert fate_call_kwargs["master_timeline"] == expected_timeline
        assert antag_call_kwargs["master_timeline"] == expected_timeline

    def test_group3_receives_character_arc_plan_from_group2(self, mocker):
        """plan_pov_focal_character must receive the character_arc_plan produced
        by plan_character_arc_plan."""
        mocks = _patch_all_agents(mocker)
        expected_arcs = {"arcs": ["protagonist growth"]}
        mocks["plan_character_arc_plan"].return_value = expected_arcs

        run_full_planning(**_BASE_KWARGS)

        pov_call_kwargs = mocks["plan_pov_focal_character"].call_args.kwargs
        assert pov_call_kwargs["character_arc_plan"] == expected_arcs

    def test_propagates_agent_exception(self, mocker):
        """Exceptions from any planning agent must propagate to the caller."""
        _patch_all_agents(mocker)
        mocker.patch(
            "novelforge.services.planning.plan_story_architecture",
            side_effect=RuntimeError("LLM down"),
        )
        with pytest.raises(RuntimeError, match="LLM down"):
            run_full_planning(**_BASE_KWARGS)

    def test_group2_exception_propagates(self, mocker):
        _patch_all_agents(mocker)
        mocker.patch(
            "novelforge.services.planning.plan_character_arc_plan",
            side_effect=RuntimeError("timeout"),
        )
        with pytest.raises(RuntimeError, match="timeout"):
            run_full_planning(**_BASE_KWARGS)

    def test_group3_exception_propagates(self, mocker):
        _patch_all_agents(mocker)
        mocker.patch(
            "novelforge.services.planning.plan_pov_focal_character",
            side_effect=RuntimeError("network error"),
        )
        with pytest.raises(RuntimeError, match="network error"):
            run_full_planning(**_BASE_KWARGS)


# ---------------------------------------------------------------------------
# run_selective_planning
# ---------------------------------------------------------------------------


class TestRunSelectivePlanningAllChanged:
    """When prev_hashes is empty every agent should run."""

    def test_all_agents_run_when_no_prev_hashes(self, mocker):
        mocks = _patch_all_agents(mocker)
        results, new_hashes = run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes={},
            prev_results={},
        )
        for name, m in mocks.items():
            assert m.call_count == 1, f"{name} should have been called"

    def test_returns_all_eight_keys(self, mocker):
        _patch_all_agents(mocker)
        results, new_hashes = run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes={},
            prev_results={},
        )
        expected_keys = {
            "story_architecture",
            "master_timeline",
            "technology_rules",
            "theme_reinforcement",
            "character_fate_registry",
            "character_arc_plan",
            "antagonist_motivation_plan",
            "pov_focal_character_plan",
        }
        assert set(results.keys()) == expected_keys

    def test_returns_new_hashes_for_all_agents(self, mocker):
        _patch_all_agents(mocker)
        _, new_hashes = run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes={},
            prev_results={},
        )
        expected_hash_keys = {
            "story_architecture",
            "master_timeline",
            "technology_rules",
            "theme_reinforcement",
            "character_fate_registry",
            "character_arc_plan",
            "antagonist_motivation_plan",
            "pov_focal_character_plan",
        }
        assert set(new_hashes.keys()) == expected_hash_keys
        for key, h in new_hashes.items():
            assert len(h) == 16, f"Hash for {key} must be 16 chars"


class TestRunSelectivePlanningAllUnchanged:
    """When prev_hashes matches all inputs every agent should be skipped."""

    def _run_once(self, mocker) -> tuple[dict, dict]:
        """Run once and return (results, hashes) for use as prev values."""
        _patch_all_agents(mocker)
        results, hashes = run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes={},
            prev_results={},
        )
        return results, hashes

    def test_no_agents_called_when_all_hashes_match(self, mocker):
        prev_results, first_hashes = self._run_once(mocker)
        # Second run with the exact same inputs and previous results/hashes
        mocks = _patch_all_agents(mocker)
        run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes=first_hashes,
            prev_results=prev_results,
        )
        for name, m in mocks.items():
            assert m.call_count == 0, f"{name} should have been skipped"

    def test_cached_results_returned_when_skipped(self, mocker):
        prev_results, first_hashes = self._run_once(mocker)
        _patch_all_agents(mocker)
        results, _ = run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes=first_hashes,
            prev_results=prev_results,
        )
        for key, cached in prev_results.items():
            assert results[key] == cached, f"{key} should return cached value"


class TestRunSelectivePlanningPartialChange:
    """Only the agents whose inputs changed (or that depend on changed outputs)
    should be re-run."""

    def _base_hashes(self, mocker) -> dict:
        _patch_all_agents(mocker)
        _, hashes = run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes={},
            prev_results={},
        )
        return hashes

    def test_only_affected_agents_rerun_when_title_changes(self, mocker):
        """Changing only the title should cause all base-input agents to rerun
        while character-only agents are skipped if characters are unchanged."""
        prev_hashes = self._base_hashes(mocker)
        prev_results = {
            k: {"cached": True}
            for k in (
                "story_architecture", "master_timeline", "technology_rules",
                "theme_reinforcement", "character_fate_registry",
                "character_arc_plan", "antagonist_motivation_plan",
                "pov_focal_character_plan",
            )
        }

        mocks = _patch_all_agents(mocker)
        run_selective_planning(
            **{**_BASE_KWARGS, "title": "A Brand New Title"},
            narrative_perspective="third_person",
            prev_hashes=prev_hashes,
            prev_results=prev_results,
        )
        # story_architecture, technology_rules, theme_reinforcement use base_inputs
        assert mocks["plan_story_architecture"].call_count == 1
        assert mocks["plan_technology_rules"].call_count == 1
        assert mocks["plan_theme_reinforcement"].call_count == 1
        # master_timeline also uses base_inputs (plus character_list)
        assert mocks["plan_master_timeline"].call_count == 1

    def test_only_pov_reruns_when_narrative_perspective_changes(self, mocker):
        """Changing only the narrative perspective should only rerun Group 3."""
        # First compute hashes and results with the default perspective
        _patch_all_agents(mocker)
        prev_results, prev_hashes = run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes={},
            prev_results={},
        )

        mocks = _patch_all_agents(mocker)
        run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="first_person:Alice",
            prev_hashes=prev_hashes,
            prev_results=prev_results,
        )
        # Only Group 3 should rerun
        assert mocks["plan_pov_focal_character"].call_count == 1
        # Groups 1 and 2 should be skipped
        assert mocks["plan_story_architecture"].call_count == 0
        assert mocks["plan_master_timeline"].call_count == 0
        assert mocks["plan_character_arc_plan"].call_count == 0
        assert mocks["plan_character_fate_registry"].call_count == 0

    def test_group3_receives_cached_arc_when_group2_skipped(self, mocker):
        """When character_arc_plan is cached, the cached value must be forwarded
        to plan_pov_focal_character (not a stale session value)."""
        # First full run to get real results and hashes
        _patch_all_agents(mocker)
        prev_results, prev_hashes = run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes={},
            prev_results={},
        )
        # Remove POV hash so Group 3 must rerun while Groups 1+2 are cached
        prev_hashes.pop("pov_focal_character_plan")

        mocks = _patch_all_agents(mocker)
        run_selective_planning(
            **_BASE_KWARGS,
            narrative_perspective="third_person",
            prev_hashes=prev_hashes,
            prev_results=prev_results,
        )
        pov_kwargs = mocks["plan_pov_focal_character"].call_args.kwargs
        assert pov_kwargs["character_arc_plan"] == prev_results["character_arc_plan"]


class TestRunSelectivePlanningExceptionPropagation:
    def test_group1_exception_propagates(self, mocker):
        _patch_all_agents(mocker)
        mocker.patch(
            "novelforge.services.planning.plan_story_architecture",
            side_effect=RuntimeError("group1 down"),
        )
        with pytest.raises(RuntimeError, match="group1 down"):
            run_selective_planning(
                **_BASE_KWARGS,
                narrative_perspective="third_person",
                prev_hashes={},
                prev_results={},
            )

    def test_group2_exception_propagates(self, mocker):
        _patch_all_agents(mocker)
        mocker.patch(
            "novelforge.services.planning.plan_character_arc_plan",
            side_effect=RuntimeError("group2 down"),
        )
        with pytest.raises(RuntimeError, match="group2 down"):
            run_selective_planning(
                **_BASE_KWARGS,
                narrative_perspective="third_person",
                prev_hashes={},
                prev_results={},
            )

    def test_group3_exception_propagates(self, mocker):
        _patch_all_agents(mocker)
        mocker.patch(
            "novelforge.services.planning.plan_pov_focal_character",
            side_effect=RuntimeError("group3 down"),
        )
        with pytest.raises(RuntimeError, match="group3 down"):
            run_selective_planning(
                **_BASE_KWARGS,
                narrative_perspective="third_person",
                prev_hashes={},
                prev_results={},
            )

    def test_exception_does_not_corrupt_prev_results(self, mocker):
        """When a planning agent raises, prev_results must remain unchanged."""
        _patch_all_agents(mocker)
        prev_results = {"story_architecture": {"old": "arch"}}
        snapshot = dict(prev_results)

        mocker.patch(
            "novelforge.services.planning.plan_master_timeline",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(RuntimeError):
            run_selective_planning(
                **_BASE_KWARGS,
                narrative_perspective="third_person",
                prev_hashes={},
                prev_results=prev_results,
            )
        assert prev_results == snapshot

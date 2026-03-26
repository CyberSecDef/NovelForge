"""Shared pytest fixtures for NovelForge tests."""

import json
import pytest


@pytest.fixture
def app(tmp_path):
    """Create a fresh Flask app instance for testing with isolated directories."""
    import novelforge.config as config

    # Redirect session/novel files to temp directory so tests don't pollute
    original_novels_dir = config.NOVELS_DIR
    test_novels_dir = str(tmp_path / "novels")
    (tmp_path / "novels").mkdir()
    config.NOVELS_DIR = test_novels_dir

    from novelforge import create_app, limiter
    flask_app = create_app(testing=True)
    flask_app.config["SECRET_KEY"] = "test-secret"
    flask_app.config["WTF_CSRF_ENABLED"] = False
    flask_app.config["RATELIMIT_ENABLED"] = False
    limiter.enabled = False

    yield flask_app

    # Restore original path
    config.NOVELS_DIR = original_novels_dir


@pytest.fixture
def client(app):
    """Flask test client."""
    with app.test_client() as c:
        yield c


def _canned_llm_response(messages, *, action="", json_mode=False):
    """
    Smart canned LLM response that inspects the prompt to return
    contextually appropriate responses for each agent/route.
    """
    system = messages[0]["content"] if messages else ""
    user = messages[-1]["content"] if messages else ""
    combined = (system + " " + user).lower()
    act = (action or "").lower()

    # Title generation
    if "title" in act and "generat" in act:
        return '"The Test Novel"'

    # Outline generation
    if "outline" in act or "chapter-by-chapter" in combined:
        return json.dumps({
            "chapters": [
                {"number": 1, "title": "The Beginning", "summary": "The story begins."},
                {"number": 2, "title": "The Middle", "summary": "The conflict rises."},
                {"number": 3, "title": "The End", "summary": "The resolution."},
            ]
        })

    # Character generation
    if "character" in act and "generat" in act:
        return json.dumps({
            "characters": [
                {
                    "name": "Alice",
                    "age": "28",
                    "role": "Protagonist",
                    "background": "A brave explorer.",
                    "arc": "Learns to trust others.",
                },
            ]
        })

    # Planning agents (all return JSON)
    if "planning" in act or "story architecture" in act:
        return json.dumps({
            "architecture_type": "three-act",
            "acts": [],
            "global_turns": {
                "inciting_incident": {"chapter": 1, "detail": "Test"},
                "midpoint_reversal": {"chapter": 2, "detail": "Test"},
                "climax": {"chapter": 3, "detail": "Test"},
                "resolution": {"chapter": 3, "detail": "Test"},
            },
            "chapter_plan": [],
        })

    if "timeline" in act:
        return json.dumps({
            "ledger": [],
            "character_states": [],
            "chapter_constraints": [],
            "continuity_risks": [],
        })

    if "fate" in act or "character fate" in act:
        return json.dumps({
            "characters": [],
            "chapter_status_overrides": [],
            "conflict_pairs": [],
            "dead_lock": [],
        })

    if "character arc" in act or "arc plan" in act:
        return json.dumps({"arcs": [], "consistency_rules": []})

    if "antagonist" in act:
        return json.dumps({
            "antagonists": [],
            "escalation_plan": [],
            "pressure_points": [],
        })

    if "technology" in act:
        return json.dumps({"systems": [], "chapter_constraints": []})

    if "theme" in act:
        return json.dumps({"themes": []})

    if "pov" in act or "focal" in act:
        return json.dumps({"chapter_pov_plan": [], "default_pov": "Third limited"})

    # Chapter drafting
    if "drafting" in act:
        return "It was a dark and stormy night. The hero set forth on a journey."

    # Chapter summary
    if "summar" in act:
        return "The hero began their journey through the dark forest."

    # Consistency pass
    if "consistency" in act:
        return json.dumps({
            "issues": [],
            "overall_assessment": "Consistent throughout.",
        })

    # Post-manuscript audit agents
    if "continuity audit" in act:
        return json.dumps({
            "contradictions": [],
            "character_state_errors": [],
            "timeline_errors": [],
            "location_errors": [],
            "overall_integrity": "high",
            "overall_assessment": "No issues found.",
        })

    if "compression" in act:
        return json.dumps({
            "redundant_sequences": [],
            "emotional_beat_repetitions": [],
            "compression_priority": "low",
            "overall_assessment": "Well-paced.",
        })

    if "resolution" in act:
        return json.dumps({
            "character_resolutions": [],
            "unresolved_characters": [],
            "resolution_integrity": "strong",
            "overall_assessment": "All arcs resolved.",
        })

    if "thematic" in act:
        return json.dumps({
            "theme_payoffs": [],
            "abandoned_themes": [],
            "weak_payoffs": [],
            "thematic_integrity": "strong",
            "overall_assessment": "Themes land.",
        })

    if "climax" in act:
        return json.dumps({
            "climax_decision_present": True,
            "decision_is_active": True,
            "moral_dimension_present": True,
            "arc_resolved": True,
            "protagonist_is_agent": True,
            "climax_chapter": 3,
            "integrity_failures": [],
            "climax_integrity": "strong",
            "overall_assessment": "Effective climax.",
        })

    if "loose thread" in act:
        return json.dumps({
            "unresolved_threads": [],
            "dangling_setup_elements": [],
            "intentionally_open_threads": [],
            "thread_integrity": "strong",
            "overall_assessment": "All threads resolved.",
        })

    if "immersion" in act:
        return json.dumps({
            "pacing_assessment": "good",
            "tension_curve": "rising",
            "stakes_clarity": "clear",
            "engagement_score": 8,
            "weak_chapters": [],
            "immersion_breaks": [],
            "reader_experience_highlights": [],
            "overall_rating": "strong",
            "recommendations": [],
        })

    if "heatmap" in act or "pacing" in act:
        return json.dumps({
            "chapter_metrics": [],
            "flat_sections": [],
            "overall_pacing_assessment": "Good pacing.",
        })

    if "illustration" in act:
        return json.dumps({
            "illustrations": [
                {
                    "type": "cover",
                    "chapter": None,
                    "scene_description": "A dramatic cover scene",
                    "art_prompt": "A hero standing at the edge of a cliff at sunset",
                },
            ]
        })

    if "revision" in act:
        return "The revised chapter content with improvements."

    # Generic fallback — return the input text slightly modified
    return "Processed output from the LLM agent."


@pytest.fixture
def mock_llm(mocker):
    """
    Mock call_llm across all modules that import it.

    Returns the mock object so tests can inspect calls or override
    return_value/side_effect per-test.
    """
    mock = mocker.patch(
        "novelforge.llm.client.call_llm",
        side_effect=_canned_llm_response,
    )
    # Also patch the imported references in all consuming modules
    for module in (
        "novelforge.agents.base",
        "novelforge.agents.chapter",
        "novelforge.agents.planning",
        "novelforge.routes.outline",
        "novelforge.routes.generation",
        "novelforge.routes.export",
    ):
        mocker.patch(f"{module}.call_llm", side_effect=_canned_llm_response)

    # Reset the circuit breaker so mocked calls aren't blocked by prior failures
    from novelforge.llm.client import _llm_circuit_breaker
    _llm_circuit_breaker.reset()

    return mock

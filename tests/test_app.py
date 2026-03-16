"""
Tests for NovelForge – validate_outline_input boundary conditions, Flask routes,
and specialized agent prompt builders.
"""

import json
import pytest


@pytest.fixture
def client():
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret"
    with flask_app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# validate_outline_input
# ---------------------------------------------------------------------------

class TestValidateOutlineInput:
    def _call(self, data):
        from app import validate_outline_input
        return validate_outline_input(data)

    def test_valid_input(self):
        ok, err = self._call(
            {"premise": "A hero's journey", "genre": "Fantasy", "chapters": 10, "word_count": 80000}
        )
        assert ok
        assert err == ""

    def test_empty_premise(self):
        ok, err = self._call(
            {"premise": "", "genre": "Fantasy", "chapters": 5, "word_count": 50000}
        )
        assert not ok
        assert "premise" in err.lower()

    def test_premise_exactly_2000(self):
        ok, err = self._call(
            {"premise": "x" * 2000, "genre": "Fantasy", "chapters": 5, "word_count": 50000}
        )
        assert ok, f"Exactly 2000 chars should be valid, got: {err}"

    def test_premise_2001_chars(self):
        ok, err = self._call(
            {"premise": "x" * 2001, "genre": "Fantasy", "chapters": 5, "word_count": 50000}
        )
        assert not ok
        assert "2000" in err

    def test_invalid_genre(self):
        ok, err = self._call(
            {"premise": "A story", "genre": "SciFi_bad", "chapters": 5, "word_count": 50000}
        )
        assert not ok
        assert "genre" in err.lower() or "Genre" in err

    def test_all_valid_genres(self):
        from app import ALLOWED_GENRES
        for genre in ALLOWED_GENRES:
            ok, err = self._call(
                {"premise": "A story", "genre": genre, "chapters": 5, "word_count": 50000}
            )
            assert ok, f"Genre '{genre}' should be valid, got: {err}"

    def test_chapters_below_minimum(self):
        ok, err = self._call(
            {"premise": "A story", "genre": "Mystery", "chapters": 2, "word_count": 50000}
        )
        assert not ok
        assert "chapter" in err.lower() or "3" in err

    def test_chapters_exactly_3(self):
        ok, err = self._call(
            {"premise": "A story", "genre": "Mystery", "chapters": 3, "word_count": 50000}
        )
        assert ok

    def test_chapters_not_a_number(self):
        ok, err = self._call(
            {"premise": "A story", "genre": "Mystery", "chapters": "abc", "word_count": 50000}
        )
        assert not ok

    def test_word_count_below_minimum(self):
        ok, err = self._call(
            {"premise": "A story", "genre": "Mystery", "chapters": 5, "word_count": 999}
        )
        assert not ok

    def test_word_count_exactly_1000(self):
        ok, err = self._call(
            {"premise": "A story", "genre": "Mystery", "chapters": 5, "word_count": 1000}
        )
        assert ok

    def test_word_count_not_a_number(self):
        ok, err = self._call(
            {"premise": "A story", "genre": "Mystery", "chapters": 5, "word_count": "bad"}
        )
        assert not ok

    def test_missing_fields_returns_error(self):
        ok, err = self._call({})
        assert not ok


# ---------------------------------------------------------------------------
# Flask route tests
# ---------------------------------------------------------------------------

class TestRoutes:
    def test_index_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"NovelForge" in r.data

    def test_generate_outline_empty_body(self, client):
        r = client.post(
            "/generate_outline",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_generate_outline_invalid_genre(self, client):
        r = client.post(
            "/generate_outline",
            data=json.dumps(
                {"premise": "A story", "genre": "Alien", "chapters": 5, "word_count": 50000}
            ),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_approve_outline_empty_title(self, client):
        r = client.post(
            "/approve_outline",
            data=json.dumps({"title": "", "chapters": [], "characters": []}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_approve_outline_empty_chapters(self, client):
        r = client.post(
            "/approve_outline",
            data=json.dumps({"title": "My Novel", "chapters": [], "characters": []}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_approve_outline_valid(self, client):
        r = client.post(
            "/approve_outline",
            data=json.dumps(
                {
                    "title": "My Novel",
                    "chapters": [{"number": 1, "title": "Ch1", "summary": "Intro"}],
                    "characters": [],
                }
            ),
            content_type="application/json",
        )
        assert r.status_code == 200
        assert r.get_json()["status"] == "approved"

    def test_generate_chapters_no_session(self, client):
        r = client.post(
            "/generate_chapters",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_progress_unknown_token(self, client):
        r = client.get("/progress/does-not-exist")
        assert r.status_code == 404

    def test_download_nonexistent(self, client):
        r = client.get("/download/nonexistent.md")
        assert r.status_code == 404

    def test_export_no_token(self, client):
        r = client.post(
            "/export",
            data=json.dumps({"token": "fake-token"}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_export_editors_notes_no_token(self, client):
        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": "fake-token"}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_export_editors_notes_success_filename(self, client):
        from app import _progress_lock, _progress_store

        token = "test-token-editors-notes"
        with _progress_lock:
            _progress_store[token] = {
                "status": "done",
                "consistency": {
                    "overall_assessment": "Strong arc with minor pacing issues.",
                    "issues": ["Chapter 3 timeline inconsistency"],
                },
            }

        with client.session_transaction() as sess:
            sess["title"] = "My Great Novel"

        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )

        assert r.status_code == 200
        payload = r.get_json()
        assert "download_url" in payload
        assert payload["download_url"].endswith("My_Great_Novel-Editors_Notes.md")

    def test_revise_chapter_requires_instructions(self, client):
        from app import _progress_lock, _progress_store

        token = "test-token-revise-empty"
        with _progress_lock:
            _progress_store[token] = {
                "status": "done",
                "chapters_done": [{"number": 1, "title": "Ch1", "content": "Text", "summary": "S"}],
                "consistency": {"issues": [], "overall_assessment": ""},
            }

        r = client.post(
            "/revise_chapter",
            data=json.dumps({"token": token, "chapter_number": 1, "instructions": ""}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_revise_chapter_success_reruns_agents(self, client, monkeypatch):
        from app import _progress_lock, _progress_store

        token = "test-token-revise-success"
        with _progress_lock:
            _progress_store[token] = {
                "status": "done",
                "current": 2,
                "total": 2,
                "step": "Complete",
                "chapters_done": [
                    {"number": 1, "title": "Ch1", "content": "Old chapter 1", "summary": "Old summary 1"},
                    {"number": 2, "title": "Ch2", "content": "Old chapter 2", "summary": "Old summary 2"},
                ],
                "consistency": {"issues": [], "overall_assessment": ""},
            }

        with client.session_transaction() as sess:
            sess["title"] = "My Novel"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 2
            sess["chapter_list"] = [
                {"number": 1, "title": "Ch1", "summary": "Outline 1"},
                {"number": 2, "title": "Ch2", "summary": "Outline 2"},
            ]
            sess["character_list"] = []
            sess["special_instructions"] = ""

        def fake_call_llm(messages, json_mode=False):
            user_content = messages[-1]["content"]
            if "Write a 100-200 word summary" in user_content:
                return "Updated summary"
            if "Required structure" in user_content and '"issues"' in user_content:
                return '{"issues": ["Issue after revision"], "overall_assessment": "Revised consistency"}'
            return "Updated chapter content"

        monkeypatch.setattr("app.call_llm", fake_call_llm)

        r = client.post(
            "/revise_chapter",
            data=json.dumps(
                {
                    "token": token,
                    "chapter_number": 1,
                    "instructions": "Increase tension in final scene.",
                }
            ),
            content_type="application/json",
        )

        assert r.status_code == 200
        payload = r.get_json()
        assert payload["status"] == "done"
        assert payload["step"] == "Chapter 1: revised"
        assert payload["chapters_done"][0]["content"] == "Updated chapter content"
        assert payload["chapters_done"][0]["summary"] == "Updated summary"
        assert payload["consistency"]["overall_assessment"] == "Revised consistency"

    def test_download_path_traversal(self, client):
        """Directory traversal attempt should 404 (file won't exist)."""
        r = client.get("/download/../../etc/passwd")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Specialized agent prompt builders
# ---------------------------------------------------------------------------

class TestPromptBuilders:
    """
    Validate that each specialized agent prompt builder returns a well-formed
    two-message list with non-empty system and user content.
    """

    def _check_messages(self, messages):
        assert isinstance(messages, list), "Expected a list of messages"
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        system_msg, user_msg = messages
        assert system_msg["role"] == "system"
        assert user_msg["role"] == "user"
        assert system_msg["content"].strip(), "System prompt must not be empty"
        assert user_msg["content"].strip(), "User prompt must not be empty"

    def test_dialog_agent_prompt(self):
        from app import build_dialog_agent_prompt
        self._check_messages(build_dialog_agent_prompt("Some chapter text with dialogue."))

    def test_dialog_agent_prompt_contains_text(self):
        from app import build_dialog_agent_prompt
        text = "He said hello. She replied."
        msgs = build_dialog_agent_prompt(text)
        assert text in msgs[1]["content"]

    def test_scene_agent_prompt(self):
        from app import build_scene_agent_prompt
        self._check_messages(build_scene_agent_prompt("A scene in the forest."))

    def test_scene_agent_prompt_contains_pattern(self):
        from app import build_scene_agent_prompt
        msgs = build_scene_agent_prompt("text")
        combined = msgs[0]["content"] + msgs[1]["content"]
        assert "Goal" in combined and "Obstacle" in combined

    def test_structure_agent_prompt(self):
        from app import build_structure_agent_prompt
        self._check_messages(
            build_structure_agent_prompt("Chapter text.", 5, 20, "Chapter 5 outline")
        )

    def test_structure_agent_beginning(self):
        from app import build_structure_agent_prompt
        msgs = build_structure_agent_prompt("text", 1, 20, "intro")
        assert "Beginning" in msgs[1]["content"]

    def test_structure_agent_middle(self):
        from app import build_structure_agent_prompt
        msgs = build_structure_agent_prompt("text", 10, 20, "midpoint")
        assert "Middle" in msgs[1]["content"]

    def test_structure_agent_end(self):
        from app import build_structure_agent_prompt
        msgs = build_structure_agent_prompt("text", 18, 20, "climax")
        assert "End" in msgs[1]["content"]

    def test_character_agent_prompt(self):
        from app import build_character_agent_prompt
        self._check_messages(
            build_character_agent_prompt("Chapter text.", "Alice: protagonist, brave.")
        )

    def test_character_agent_prompt_contains_characters(self):
        from app import build_character_agent_prompt
        chars = "Alice: protagonist"
        msgs = build_character_agent_prompt("text", chars)
        assert chars in msgs[1]["content"]

    def test_context_analyzer_prompt_with_previous(self):
        from app import build_context_analyzer_prompt
        msgs = build_context_analyzer_prompt("current chapter", "Chapter 1: something happened")
        self._check_messages(msgs)
        assert "Chapter 1" in msgs[1]["content"]

    def test_context_analyzer_prompt_first_chapter(self):
        from app import build_context_analyzer_prompt
        msgs = build_context_analyzer_prompt("current chapter", "")
        self._check_messages(msgs)
        assert "first chapter" in msgs[1]["content"]

    def test_synthesizer_prompt(self):
        from app import build_synthesizer_prompt
        self._check_messages(
            build_synthesizer_prompt("Chapter text.", 3, "My Novel", "Fantasy")
        )

    def test_synthesizer_prompt_contains_title(self):
        from app import build_synthesizer_prompt
        msgs = build_synthesizer_prompt("text", 1, "Dragon Fire", "Fantasy")
        assert "Dragon Fire" in msgs[1]["content"]

    def test_quality_controller_prompt(self):
        from app import build_quality_controller_prompt
        self._check_messages(build_quality_controller_prompt("Chapter text."))

    def test_anti_llm_agent_prompt(self):
        from app import build_anti_llm_agent_prompt
        self._check_messages(build_anti_llm_agent_prompt("Some text with embark delve."))

    def test_anti_llm_agent_prompt_contains_forbidden_words(self):
        from app import build_anti_llm_agent_prompt, _FORBIDDEN_WORDS
        msgs = build_anti_llm_agent_prompt("text")
        # At least one forbidden word should appear in the system prompt
        forbidden_in_prompt = any(w in msgs[0]["content"] for w in _FORBIDDEN_WORDS)
        assert forbidden_in_prompt, "Anti-LLM system prompt should list forbidden words"

    def test_polish_agent_prompt(self):
        from app import build_polish_agent_prompt
        self._check_messages(build_polish_agent_prompt("Draft chapter text."))

    def test_editing_agent_prompt(self):
        from app import build_editing_agent_prompt
        self._check_messages(
            build_editing_agent_prompt("Draft text.", "Chapter should end with a fight.")
        )

    def test_chapter_summary_prompt(self):
        from app import build_chapter_summary_prompt
        self._check_messages(build_chapter_summary_prompt("Full chapter text here."))

    def test_progress_token_includes_step(self, client):
        """After /generate_chapters, the initial progress record should include 'step'."""
        with client.session_transaction() as sess:
            sess["premise"] = "A hero's journey"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 3
            sess["word_count"] = 10000
            sess["special_instructions"] = ""
            sess["title"] = "Test Novel"
            sess["chapter_list"] = [
                {"number": 1, "title": "Ch1", "summary": "Intro"},
                {"number": 2, "title": "Ch2", "summary": "Rising"},
                {"number": 3, "title": "Ch3", "summary": "End"},
            ]
            sess["character_list"] = []

        r = client.post(
            "/generate_chapters",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 200
        token = r.get_json()["token"]

        r2 = client.get(f"/progress/{token}")
        assert r2.status_code == 200
        data = r2.get_json()
        assert "step" in data, "Progress response should include 'step' field"


class TestMasterTimelineBuilder:
    def test_normalise_master_timeline_uses_fallback_shape(self):
        from app import normalise_master_timeline

        chapter_list = [
            {"number": 1, "title": "Ch1", "summary": "Setup"},
            {"number": 2, "title": "Ch2", "summary": "Complication"},
        ]
        character_list = [{"name": "Alice", "role": "Protagonist"}]

        timeline = normalise_master_timeline({}, chapter_list, character_list)

        assert isinstance(timeline, dict)
        assert "ledger" in timeline and len(timeline["ledger"]) >= 2
        assert "character_states" in timeline
        assert "chapter_constraints" in timeline

    def test_get_chapter_timeline_context_contains_constraints(self):
        from app import get_chapter_timeline_context

        master_timeline = {
            "ledger": [
                {"index": 1, "chapter": 1, "event_type": "capture", "event": "Jon is captured"},
                {"index": 2, "chapter": 2, "event_type": "rescue", "event": "Jon is rescued"},
            ],
            "character_states": [
                {"character": "Jon", "status": "recovered", "location": "Harbor", "notes": "Post-rescue"}
            ],
            "chapter_constraints": [
                {
                    "chapter": 2,
                    "must_include": ["Confirm rescue injuries"],
                    "must_avoid": ["Treat Jon as deceased"],
                }
            ],
            "continuity_risks": ["Timeline jump between Ch1 and Ch2"],
        }

        context_text = get_chapter_timeline_context(master_timeline, 2)
        assert "Master Timeline Builder" in context_text
        assert "Planned chapter event" in context_text
        assert "Must include" in context_text
        assert "Must avoid" in context_text


class TestCharacterFateRegistry:
    def test_normalise_character_fate_registry_uses_fallback(self):
        from app import normalise_character_fate_registry

        character_list = [{"name": "Mara", "role": "Protagonist"}]
        timeline = normalise_character_fate_registry({}, character_list, 3)

        assert isinstance(timeline, dict)
        assert "registry" in timeline
        assert "chapter_constraints" in timeline
        assert len(timeline["chapter_constraints"]) == 3

    def test_get_chapter_fate_context_contains_constraints(self):
        from app import get_chapter_fate_context

        fate_registry = {
            "registry": [
                {
                    "character": "Mara",
                    "current_status": "captured",
                    "capture_state": "captured",
                    "injuries": ["Broken wrist"],
                    "narrative_status": "active",
                    "definitive_outcome": "survival",
                    "outcome_locked": False,
                    "definitive_outcome_required": True,
                    "state_constraints": ["Cannot appear free before rescue."],
                    "recovery_conditions": ["Requires medical treatment after rescue."],
                    "pivotal_chapters": [2],
                }
            ],
            "chapter_constraints": [
                {
                    "chapter": 2,
                    "must_track": ["Mara remains in custody through chapter midpoint."],
                    "must_not_contradict": ["Do not depict Mara as free without rescue event."],
                }
            ],
            "conflict_checks": ["Captured status conflicts with free movement scene."],
        }

        context_text = get_chapter_fate_context(fate_registry, 2)
        assert "Character Fate Registry" in context_text
        assert "Mara" in context_text
        assert "Must track" in context_text
        assert "Must not contradict" in context_text


class TestCharacterArcPlanner:
    def test_normalise_character_arc_plan_uses_fallback(self):
        from app import normalise_character_arc_plan

        character_list = [{"name": "Elena", "role": "Protagonist", "arc": "Learns trust"}]
        chapter_list = [
            {"number": 1, "title": "Start", "summary": "Beginning"},
            {"number": 2, "title": "Middle", "summary": "Complication"},
            {"number": 3, "title": "End", "summary": "Resolution"},
        ]

        arc_plan = normalise_character_arc_plan({}, character_list, chapter_list)
        assert isinstance(arc_plan, dict)
        assert "arcs" in arc_plan
        assert "chapter_constraints" in arc_plan
        assert len(arc_plan["chapter_constraints"]) == 3

    def test_get_chapter_arc_context_contains_beats(self):
        from app import get_chapter_arc_context

        arc_plan = {
            "arcs": [
                {
                    "character": "Elena",
                    "role": "primary",
                    "start_state": "Distrustful and isolated",
                    "midpoint_transformation": "Learns to rely on allies",
                    "crisis_point": "Must risk herself for others",
                    "final_moral_choice": "Chooses solidarity over control",
                    "arc_theme": "Trust",
                    "chapter_beats": [
                        {"chapter": 2, "phase": "midpoint", "beat": "Accepts help during escape."}
                    ],
                    "consistency_rules": ["No regression to total isolation after midpoint."],
                }
            ],
            "chapter_constraints": [
                {
                    "chapter": 2,
                    "must_advance": ["Elena shifts from isolation to cooperation."],
                    "must_not_undo": ["Do not erase midpoint trust gain."],
                }
            ],
            "global_arc_risks": ["Midpoint growth not reflected in dialogue."],
        }

        context_text = get_chapter_arc_context(arc_plan, 2)
        assert "Character Arc Planner" in context_text
        assert "Elena" in context_text
        assert "Beat (midpoint)" in context_text
        assert "Must advance" in context_text


class TestAntagonistMotivationArchitect:
    def test_normalise_antagonist_motivation_plan_uses_fallback(self):
        from app import normalise_antagonist_motivation_plan

        character_list = [
            {"name": "Drex", "role": "Antagonist", "arc": "Escalates control"},
            {"name": "Mara", "role": "Protagonist", "arc": "Learns trust"},
        ]
        chapter_list = [
            {"number": 1, "title": "Setup", "summary": "Threat appears"},
            {"number": 2, "title": "Pressure", "summary": "Escalation"},
            {"number": 3, "title": "Climax", "summary": "Confrontation"},
        ]

        plan = normalise_antagonist_motivation_plan({}, character_list, chapter_list)

        assert isinstance(plan, dict)
        assert "antagonists" in plan
        assert "chapter_constraints" in plan
        assert len(plan["chapter_constraints"]) == 3

    def test_get_chapter_antagonist_context_contains_escalation(self):
        from app import get_chapter_antagonist_context

        plan = {
            "antagonists": [
                {
                    "character": "Drex",
                    "motivation_core": "Control the city",
                    "external_goal": "Neutralize resistance",
                    "internal_need": "Avoid humiliation",
                    "fear_trigger": "Losing authority",
                    "moral_line": "Avoid mass civilian casualties",
                    "pressure_points": ["Reputation", "Supply chain"],
                    "escalation_plan": [
                        {
                            "chapter": 2,
                            "action": "Orders targeted raids",
                            "tactic": "Coercion",
                            "motivation_link": "Reassert dominance after setback",
                        }
                    ],
                    "consistency_rules": ["Escalate only when pressure increases."],
                }
            ],
            "chapter_constraints": [
                {
                    "chapter": 2,
                    "must_show": ["Motivated retaliation by Drex."],
                    "must_not_break": ["Do not make Drex benevolent without cause."],
                }
            ],
            "global_risks": ["Escalation appears disconnected from chapter 1 events."],
        }

        context_text = get_chapter_antagonist_context(plan, 2)
        assert "Antagonist Motivation Architect" in context_text
        assert "Drex" in context_text
        assert "Escalation" in context_text
        assert "Must show" in context_text


class TestTechnologyRulesDesigner:
    def test_normalise_technology_rules_uses_fallback(self):
        from app import normalise_technology_rules

        chapter_list = [
            {"number": 1, "title": "Setup", "summary": "System introduced"},
            {"number": 2, "title": "Complication", "summary": "Failure under stress"},
        ]

        rules = normalise_technology_rules({}, chapter_list)

        assert isinstance(rules, dict)
        assert "systems" in rules
        assert "chapter_constraints" in rules
        assert len(rules["chapter_constraints"]) == 2

    def test_get_chapter_technology_context_contains_constraints(self):
        from app import get_chapter_technology_context

        rules = {
            "systems": [
                {
                    "name": "Sentinel Grid",
                    "purpose": "Monitors zone ingress",
                    "latency_ms": 2400,
                    "detection_methods": ["Pattern scan"],
                    "detection_blind_spots": ["Storm interference"],
                    "resource_constraints": ["Finite processing budget"],
                    "operational_limits": ["Cannot process full city in real time"],
                    "failure_modes": ["Queue overflow"],
                    "countermeasures": ["Manual escalation"],
                    "forbidden_capabilities": ["Instant omniscience"],
                }
            ],
            "global_constraints": ["All detections incur latency."],
            "chapter_constraints": [
                {
                    "chapter": 2,
                    "must_respect": ["Account for detection lag before response."],
                    "must_not_allow": ["No instant citywide trace."],
                }
            ],
            "continuity_risks": ["Response time too fast for established latency."],
        }

        context_text = get_chapter_technology_context(rules, 2)
        assert "Technology Rules Designer" in context_text
        assert "Sentinel Grid" in context_text
        assert "Must respect" in context_text
        assert "Must not allow" in context_text


class TestThemeReinforcementPlanner:
    def test_normalise_theme_reinforcement_uses_fallback(self):
        from app import normalise_theme_reinforcement

        chapter_list = [
            {"number": 1, "title": "Opening", "summary": "Setup."},
            {"number": 2, "title": "Rising", "summary": "Conflict deepens."},
        ]

        # Empty dict should produce a valid fallback structure
        result = normalise_theme_reinforcement({}, chapter_list)
        assert isinstance(result, dict)
        assert "themes" in result
        assert "global_thematic_arcs" in result
        assert "chapter_constraints" in result
        assert len(result["chapter_constraints"]) == 2

    def test_get_chapter_theme_context_contains_guidance(self):
        from app import get_chapter_theme_context

        theme_data = {
            "themes": [
                {
                    "name": "Memory as Control",
                    "description": "The state uses curated memory to shape loyalty.",
                    "motifs": ["mirrors", "archives"],
                    "pillar_moments": ["Chapter 3 revelation", "Final reckoning"],
                    "chapter_appearances": [
                        {"chapter": 3, "role": "peak", "guidance": "Show memory erasure directly affecting the protagonist."}
                    ],
                }
            ],
            "global_thematic_arcs": ["Erasure of individual history enables collective obedience."],
            "chapter_constraints": [
                {
                    "chapter": 3,
                    "themes_present": ["Memory as Control"],
                    "thematic_guidance": "The protagonist confronts a doctored record of their past.",
                }
            ],
            "continuity_risks": ["Theme dropped after chapter 5 with no resolution."],
        }

        context_text = get_chapter_theme_context(theme_data, 3)
        assert "Theme Reinforcement Planner" in context_text
        assert "Memory as Control" in context_text
        assert "Themes active this chapter" in context_text
        assert "Thematic guidance" in context_text


class TestContinuityGatekeeper:
    def test_build_continuity_gatekeeper_prompt_mentions_core_sources(self):
        from app import build_continuity_gatekeeper_prompt

        messages = build_continuity_gatekeeper_prompt(
            chapter_num=4,
            chapter_title="The Locked Room",
            chapter_summary="The protagonist tries to recover a witness before transport.",
            previous_summaries="Chapter 1: Arrest.\n\nChapter 2: Interrogation.\n\nChapter 3: Escape attempt.",
            chapter_timeline_context="Chapter 3: Witness transferred to Annex-4.",
            chapter_fate_context="- Mara: alive, injured.\n- Iven: captured.",
            chapter_arc_context="- Mara must show growing willingness to risk others.",
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "Continuity Gatekeeper brief" in content
        assert "Master Timeline Builder" in content
        assert "Character Fate Registry" in content
        assert "FORBIDDEN SCENARIOS" in content

    def test_run_continuity_gatekeeper_returns_empty_on_failure(self, monkeypatch):
        import app

        def _boom(*args, **kwargs):
            raise RuntimeError("llm failed")

        monkeypatch.setattr(app, "call_llm", _boom)

        result = app.run_continuity_gatekeeper(
            chapter_num=2,
            chapter_title="Checkpoint",
            chapter_summary="A risky crossing.",
            previous_summaries="Chapter 1: Setup.",
            chapter_timeline_context="",
            chapter_fate_context="",
            chapter_arc_context="",
        )

        assert result == ""


class TestNarrativeRedundancyDetector:
    def test_build_prompt_mentions_redundancy_targets(self):
        from app import build_narrative_redundancy_detector_prompt

        messages = build_narrative_redundancy_detector_prompt(
            chapter_text="Mara plans another extraction from the same prison convoy.",
            previous_summaries="Chapter 2: Mara already raided a convoy to free one witness.",
            chapter_summary="Mara attempts a second rescue under heavier surveillance.",
            chapter_num=4,
            title="Glass Province",
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "repeated operations" in content
        assert "sacrifice" in content.lower()
        assert "Previous chapter summaries" in content
        assert "Return ONLY the complete revised chapter text" in content

    def test_build_prompt_handles_first_chapter(self):
        from app import build_narrative_redundancy_detector_prompt

        messages = build_narrative_redundancy_detector_prompt(
            chapter_text="Opening chapter text.",
            previous_summaries="",
            chapter_summary="Introduce the protagonist and central threat.",
            chapter_num=1,
            title="Opening Signal",
        )

        assert "no prior chapters to compare" in messages[1]["content"].lower()


class TestCharacterThreadTracker:
    def test_build_prompt_mentions_forward_movement(self):
        from app import build_character_thread_tracker_prompt

        messages = build_character_thread_tracker_prompt(
            chapter_text="Mara and Iven walk to the checkpoint. Kael watches.",
            characters_text=(
                "- Mara: protagonist, determined.\n"
                "- Iven: ally, fearful.\n"
                "- Kael: antagonist, calculating."
            ),
            chapter_num=5,
            title="Glass Province",
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "forward narrative movement" in content
        assert "static" in content
        assert "Glass Province" in content

    def test_build_prompt_includes_arc_context(self):
        from app import build_character_thread_tracker_prompt

        messages = build_character_thread_tracker_prompt(
            chapter_text="Chapter text here.",
            characters_text="- Mara: lead.",
            chapter_num=3,
            title="Test Novel",
            chapter_arc_context="Mara: midpoint crisis – must choose loyalty over survival.",
        )

        content = messages[1]["content"]
        assert "Character Arc Planner" in content
        assert "midpoint crisis" in content


class TestOperationalDistinctivenessAgent:
    def test_build_prompt_checks_operational_repetition(self):
        from app import build_operational_distinctiveness_prompt

        messages = build_operational_distinctiveness_prompt(
            chapter_text="They infiltrate the server room using the same vent route as before.",
            previous_summaries="Chapter 1: team infiltrates facility via ventilation system.",
            chapter_summary="The team must extract the drive from floor seven.",
            chapter_num=4,
            title="Iron Meridian",
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "operational repetition" in content
        assert "same strategic approach" in content
        assert "Iron Meridian" in content
        assert "Return ONLY the complete revised chapter text" in content

    def test_build_prompt_handles_first_chapter(self):
        from app import build_operational_distinctiveness_prompt

        messages = build_operational_distinctiveness_prompt(
            chapter_text="Opening operation text.",
            previous_summaries="",
            chapter_summary="Introduce the heist and the team.",
            chapter_num=1,
            title="First Strike",
        )

        assert "no prior chapters to compare" in messages[1]["content"].lower()


class TestStoryMomentumTracker:
    def test_build_prompt_checks_escalation(self):
        from app import build_story_momentum_tracker_prompt

        messages = build_story_momentum_tracker_prompt(
            chapter_text="Raya escapes again through the back channel unharmed.",
            previous_summaries=(
                "Chapter 1: Raya loses her cover identity.\n"
                "Chapter 2: Raya escapes a checkpoint, losing her partner."
            ),
            chapter_num=5,
            title="Pale Frequency",
            total_chapters=20,
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "escalation target" in content.lower()
        assert "Pale Frequency" in content
        assert "stakes" in content
        assert "Return ONLY the complete revised chapter text" in content

    def test_build_prompt_handles_first_chapter(self):
        from app import build_story_momentum_tracker_prompt

        messages = build_story_momentum_tracker_prompt(
            chapter_text="The story begins.",
            previous_summaries="",
            chapter_num=1,
            title="Pale Frequency",
            total_chapters=20,
        )

        assert "chapter 1" in messages[1]["content"].lower()


class TestCharacterStateUpdater:
    def test_build_prompt_records_character_states(self):
        from app import build_character_state_updater_prompt

        messages = build_character_state_updater_prompt(
            chapter_text="Mara is shot and captured by Kael's forces. Iven escapes through the east gate.",
            chapter_summary="Mara is captured; Iven escapes.",
            characters_text=(
                "- Mara: protagonist, determined.\n"
                "- Iven: ally, resourceful.\n"
                "- Kael: antagonist, ruthless."
            ),
            chapter_num=7,
            title="Iron Meridian",
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "Iron Meridian" in content
        assert "Chapter 7" in content
        assert "captured" in content.lower() or "injuries" in content.lower()
        assert "Return ONLY the character state log" in content

    def test_build_prompt_includes_character_roster(self):
        from app import build_character_state_updater_prompt

        messages = build_character_state_updater_prompt(
            chapter_text="Brief chapter text.",
            chapter_summary="A tense standoff.",
            characters_text="- Raya: lead operative.\n- Solen: handler.",
            chapter_num=3,
            title="Pale Frequency",
        )

        content = messages[1]["content"]
        assert "Raya" in content
        assert "Solen" in content

    def test_build_continuity_gatekeeper_prompt_includes_state_log(self):
        from app import build_continuity_gatekeeper_prompt

        messages = build_continuity_gatekeeper_prompt(
            chapter_num=8,
            chapter_title="The Crossing",
            chapter_summary="Mara must be extracted.",
            previous_summaries="Chapter 7: Mara captured.",
            character_state_log="--- After Chapter 7 ---\nMARA: captured – taken by Kael's unit.",
        )

        content = messages[1]["content"]
        assert "Character State Updater" in content
        assert "MARA" in content


class TestGlobalContinuityAuditor:
    def test_build_prompt_includes_all_sources(self):
        from app import build_global_continuity_auditor_prompt

        messages = build_global_continuity_auditor_prompt(
            title="Iron Meridian",
            all_summaries=[
                "Chapter 1: Mara loses her cover identity.",
                "Chapter 2: Kael's unit captures Mara at the border.",
                "Chapter 3: Mara appears free and operational without explanation.",
            ],
            character_state_log=[
                "--- After Chapter 2 ---\nMARA: captured – taken by Kael's forces at the border crossing.",
            ],
            master_timeline={
                "ledger": [
                    {"chapter": 1, "event": "Mara's cover is blown", "event_type": "revelation"},
                    {"chapter": 2, "event": "Mara captured", "event_type": "capture"},
                ]
            },
            character_fate_registry={
                "registry": [
                    {"character": "Mara", "current_status": "captured", "definitive_outcome": "survival"},
                ]
            },
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "Iron Meridian" in content
        assert "CHAPTER SUMMARIES" in content
        assert "CHARACTER STATE LOG" in content
        assert "MASTER TIMELINE LEDGER" in content
        assert "CHARACTER FATE REGISTRY" in content
        assert "contradictions" in content
        assert "overall_integrity" in content

    def test_build_prompt_handles_empty_supporting_data(self):
        from app import build_global_continuity_auditor_prompt

        messages = build_global_continuity_auditor_prompt(
            title="First Draft",
            all_summaries=["Chapter 1: The story begins."],
            character_state_log=[],
            master_timeline={},
            character_fate_registry={},
        )

        content = messages[1]["content"]
        assert "No character state log available." in content
        assert "No master timeline available." in content
        assert "No fate registry available." in content


class TestNarrativeCompressionEditor:
    def test_build_prompt_identifies_redundancy_targets(self):
        from app import build_narrative_compression_editor_prompt

        messages = build_narrative_compression_editor_prompt(
            title="Iron Meridian",
            all_summaries=[
                "Chapter 1: Team infiltrates the facility via vents and extracts a data drive.",
                "Chapter 2: Mara is captured during a checkpoint ambush.",
                "Chapter 3: Team infiltrates a second facility via vents to recover a second drive.",
                "Chapter 4: Mara escapes captivity and rejoins the team.",
            ],
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "Iron Meridian" in content
        assert "CHAPTER SUMMARIES" in content
        assert "redundant_sequences" in content
        assert "compression_priority" in content
        assert "Return ONLY a valid JSON object" in content

    def test_build_prompt_includes_audit_flags_when_provided(self):
        from app import build_narrative_compression_editor_prompt

        audit = {
            "contradictions": [
                {
                    "chapters": [1, 3],
                    "description": "Same infiltration method used twice without change.",
                    "suggested_resolution": "Merge or differentiate the two operations.",
                }
            ],
            "overall_integrity": "medium",
            "overall_assessment": "",
        }

        messages = build_narrative_compression_editor_prompt(
            title="Iron Meridian",
            all_summaries=["Chapter 1: First op.", "Chapter 3: Second op."],
            continuity_audit=audit,
        )

        content = messages[1]["content"]
        assert "CONTINUITY AUDIT FLAGS" in content
        assert "Same infiltration method" in content

    def test_build_prompt_handles_no_audit(self):
        from app import build_narrative_compression_editor_prompt

        messages = build_narrative_compression_editor_prompt(
            title="First Draft",
            all_summaries=["Chapter 1: Introduction."],
        )

        content = messages[1]["content"]
        assert "CONTINUITY AUDIT FLAGS" not in content


class TestCharacterResolutionValidator:
    def test_build_prompt_includes_all_sources(self):
        from app import build_character_resolution_validator_prompt

        messages = build_character_resolution_validator_prompt(
            title="Iron Meridian",
            all_summaries=[
                "Chapter 1: Mara loses her cover.",
                "Chapter 8: Mara makes her final choice and goes into exile.",
            ],
            character_arc_plan={
                "arcs": [
                    {
                        "character": "Mara",
                        "start_state": "idealistic operative",
                        "midpoint_transformation": "doubts the cause",
                        "crisis_point": "must betray colleague or mission",
                        "final_moral_choice": "chooses exile over complicity",
                    }
                ]
            },
            character_fate_registry={
                "registry": [
                    {
                        "character": "Mara",
                        "current_status": "alive",
                        "definitive_outcome": "exile",
                        "outcome_locked": True,
                    }
                ]
            },
            character_state_log=[
                "--- After Chapter 8 ---\nMARA: exile – departed the faction voluntarily."
            ],
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "Iron Meridian" in content
        assert "CHARACTER ARC PLAN" in content
        assert "CHARACTER FATE REGISTRY" in content
        assert "CHARACTER STATE LOG" in content
        assert "CHAPTER SUMMARIES" in content
        assert "character_resolutions" in content
        assert "resolution_integrity" in content

    def test_build_prompt_handles_empty_supporting_data(self):
        from app import build_character_resolution_validator_prompt

        messages = build_character_resolution_validator_prompt(
            title="Empty Draft",
            all_summaries=["Chapter 1: The story begins."],
            character_arc_plan={},
            character_fate_registry={},
            character_state_log=[],
        )

        content = messages[1]["content"]
        assert "No character arc plan available." in content
        assert "No fate registry available." in content
        assert "No character state log available." in content


class TestThematicPayoffAnalyzer:
    def test_build_prompt_includes_theme_plan_and_summaries(self):
        from app import build_thematic_payoff_analyzer_prompt

        messages = build_thematic_payoff_analyzer_prompt(
            title="Iron Meridian",
            all_summaries=[
                "Chapter 1: Mara questions whether loyalty to the faction is worth the cost.",
                "Chapter 8: Mara's final act embodies the cost of moral compromise.",
            ],
            theme_reinforcement={
                "themes": [
                    {
                        "name": "moral compromise",
                        "description": "Every act of survival requires betraying something.",
                        "chapter_appearances": [
                            {"chapter": 1, "role": "introduction"},
                            {"chapter": 8, "role": "culmination"},
                        ],
                    }
                ],
                "global_thematic_arcs": ["loyalty vs survival"],
            },
            total_chapters=8,
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "Iron Meridian" in content
        assert "THEME REINFORCEMENT PLAN" in content
        assert "CHAPTER SUMMARIES" in content
        assert "moral compromise" in content
        assert "theme_payoffs" in content
        assert "thematic_integrity" in content
        assert "Final quarter begins at Chapter" in content

    def test_build_prompt_handles_empty_theme_plan(self):
        from app import build_thematic_payoff_analyzer_prompt

        messages = build_thematic_payoff_analyzer_prompt(
            title="Empty Draft",
            all_summaries=["Chapter 1: The story begins."],
            theme_reinforcement={},
            total_chapters=10,
        )

        content = messages[1]["content"]
        assert "No theme reinforcement plan available." in content


class TestClimaxIntegrityChecker:
    def test_build_prompt_checks_protagonist_decision(self):
        from app import build_climax_integrity_checker_prompt

        messages = build_climax_integrity_checker_prompt(
            title="Iron Meridian",
            all_summaries=[
                "Chapter 1: Mara joins the faction believing in the cause.",
                "Chapter 7: Mara confronts Kael and chooses exile over complicity.",
                "Chapter 8: Mara destroys the registry and disappears.",
            ],
            character_arc_plan={
                "arcs": [
                    {
                        "character": "Mara",
                        "role": "protagonist",
                        "start_state": "idealistic operative",
                        "final_moral_choice": "chooses exile over complicity with atrocity",
                    }
                ]
            },
            total_chapters=8,
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        content = messages[1]["content"]
        assert "Iron Meridian" in content
        assert "PROTAGONIST ARC PLAN" in content
        assert "CHAPTER SUMMARIES" in content
        assert "Climax region" in content
        assert "climax_decision_present" in content
        assert "climax_integrity" in content

    def test_build_prompt_handles_empty_arc_plan(self):
        from app import build_climax_integrity_checker_prompt

        messages = build_climax_integrity_checker_prompt(
            title="Empty Draft",
            all_summaries=["Chapter 1: The story begins."],
            character_arc_plan={},
            total_chapters=10,
        )

        content = messages[1]["content"]
        assert "No character arc plan available." in content


class TestLooseThreadResolver(unittest.TestCase):
    """Tests for build_loose_thread_resolver_prompt."""

    def test_build_prompt_full_content(self):
        from app import build_loose_thread_resolver_prompt

        messages = build_loose_thread_resolver_prompt(
            title="Iron Meridian",
            all_summaries=[
                "Chapter 1: The stolen locket is introduced.",
                "Chapter 2: Kael discovers the conspiracy.",
                "Chapter 3: The locket vanishes from the story.",
            ],
            character_state_log=[
                "Kael: alive, disillusioned",
                "Serra: fate unknown",
            ],
            continuity_audit={
                "contradictions": ["Locket described as gold in ch1, silver in ch3"],
                "character_state_errors": [],
                "timeline_errors": ["Day count discrepancy in ch2"],
            },
            resolution_report={
                "unresolved_characters": ["Serra"],
            },
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        content = messages[1]["content"]
        assert "Iron Meridian" in content
        assert "CHAPTER SUMMARIES" in content
        assert "stolen locket" in content
        assert "CHARACTER STATE LOG" in content
        assert "Kael: alive" in content
        assert "CONTINUITY AUDIT FLAGS" in content
        assert "gold in ch1" in content
        assert "Day count discrepancy" in content
        assert "UNRESOLVED CHARACTERS" in content
        assert "Serra" in content
        assert "unresolved_threads" in content
        assert "thread_integrity" in content
        assert "intentionally_open_threads" in content
        assert "dangling_setup_elements" in content
        assert "overall_assessment" in content

    def test_build_prompt_handles_empty_supporting_data(self):
        from app import build_loose_thread_resolver_prompt

        messages = build_loose_thread_resolver_prompt(
            title="Empty Draft",
            all_summaries=[],
            character_state_log=[],
            continuity_audit=None,
            resolution_report=None,
        )

        content = messages[1]["content"]
        assert "No chapter summaries available." in content
        assert "No character state log available." in content
        assert "No continuity issues flagged." in content
        assert "No unresolved characters flagged." in content


class TestReaderImmersionTester(unittest.TestCase):
    """Tests for build_reader_immersion_tester_prompt."""

    def test_build_prompt_full_content(self):
        from app import build_reader_immersion_tester_prompt

        messages = build_reader_immersion_tester_prompt(
            title="Iron Meridian",
            all_summaries=[
                "Chapter 1: Kael discovers the conspiracy.",
                "Chapter 2: The first confrontation.",
                "Chapter 3: Betrayal at the tower.",
            ],
            character_arc_plan={
                "arcs": [
                    {
                        "character": "Kael",
                        "start_state": "naive idealist",
                        "final_state": "disillusioned realist",
                    }
                ]
            },
            thematic_report={
                "themes": [
                    {"theme": "memory", "payoff_present": True},
                    {"theme": "bureaucracy", "payoff_present": False},
                ]
            },
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        content = messages[1]["content"]
        assert "Iron Meridian" in content
        assert "CHAPTER SUMMARIES" in content
        assert "Kael discovers the conspiracy" in content
        assert "CHARACTER ARC OVERVIEW" in content
        assert "naive idealist" in content
        assert "disillusioned realist" in content
        assert "THEMATIC PAYOFF STATUS" in content
        assert "memory" in content
        assert "bureaucracy" in content
        assert "pacing_assessment" in content
        assert "tension_curve" in content
        assert "engagement_score" in content
        assert "weak_chapters" in content
        assert "immersion_breaks" in content
        assert "overall_rating" in content
        assert "recommendations" in content

    def test_build_prompt_handles_empty_supporting_data(self):
        from app import build_reader_immersion_tester_prompt

        messages = build_reader_immersion_tester_prompt(
            title="Empty Draft",
            all_summaries=[],
            character_arc_plan=None,
            thematic_report=None,
        )

        content = messages[1]["content"]
        assert "No chapter summaries available." in content
        assert "No character arc plan available." in content
        assert "No thematic payoff data available." in content

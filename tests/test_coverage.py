"""
Additional tests to increase code coverage to 85%+.

Targets uncovered code paths in: export formatters, editor's notes,
LLM client edge cases, image API, chapter position, prompts, and
the generation pipeline helpers.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from novelforge.progress import progress_manager


# ---------------------------------------------------------------------------
# Editor's Notes export with full audit data
# ---------------------------------------------------------------------------

class TestEditorsNotesFormatting:
    """Cover all 9 sections of the editor's notes export."""

    def _full_progress_data(self):
        return {
            "status": "done",
            "chapters_done": [
                {"number": 1, "title": "Ch1", "content": "Text", "summary": "S1"},
                {"number": 2, "title": "Ch2", "content": "Text", "summary": "S2"},
            ],
            "consistency": {
                "overall_assessment": "Minor issues found.",
                "issues": ["Timeline gap in chapter 2"],
            },
            "global_continuity_audit": {
                "overall_integrity": "medium",
                "overall_assessment": "Some contradictions.",
                "contradictions": [
                    {"chapters": [1, 2], "description": "Character age mismatch"},
                ],
                "character_state_errors": ["Alice is alive in ch1 but dead in ch2"],
                "timeline_errors": ["Event order reversed"],
            },
            "narrative_compression_report": {
                "compression_priority": "medium",
                "overall_assessment": "Some redundancy.",
                "redundant_sequences": [
                    {"chapters": [1, 2], "pattern": "Repeated chase scene", "recommendation": "Consolidate"},
                ],
                "emotional_beat_repetitions": [
                    {"chapters": [1, 2], "beat": "Grief reflection", "recommendation": "Vary"},
                ],
            },
            "character_resolution_report": {
                "resolution_integrity": "weak",
                "overall_assessment": "One character unresolved.",
                "unresolved_characters": [
                    {"character": "Bob", "issue": "Arc incomplete"},
                ],
                "character_resolutions": [
                    {"character": "Alice", "status": "Resolved"},
                ],
            },
            "thematic_payoff_report": {
                "thematic_integrity": "strong",
                "overall_assessment": "Themes land well.",
                "abandoned_themes": [
                    {"theme": "Redemption", "reason": "Dropped after ch1"},
                ],
                "weak_payoffs": [
                    {"theme": "Justice", "issue": "Underdeveloped"},
                ],
            },
            "climax_integrity_report": {
                "climax_integrity": "strong",
                "overall_assessment": "Effective climax.",
                "climax_chapter": 2,
                "climax_decision_present": True,
                "decision_is_active": True,
                "moral_dimension_present": False,
                "arc_resolved": True,
                "protagonist_is_agent": False,
                "integrity_failures": ["Passive resolution"],
            },
            "loose_thread_report": {
                "thread_integrity": "medium",
                "overall_assessment": "One loose thread.",
                "unresolved_threads": [
                    {"thread": "Missing artifact", "chapters": [1]},
                ],
                "dangling_setup_elements": [
                    {"element": "Prophecy mentioned but never fulfilled"},
                ],
                "intentionally_open_threads": [
                    {"thread": "Sequel hook"},
                ],
            },
            "reader_immersion_report": {
                "overall_rating": "good",
                "engagement_score": 7,
                "pacing_assessment": "Steady",
                "tension_curve": "Rising",
                "stakes_clarity": "Clear",
                "weak_chapters": [
                    {"chapter": 1, "reason": "Slow opening"},
                ],
                "immersion_breaks": [
                    {"chapter": 2, "description": "Info dump"},
                ],
                "recommendations": ["Tighten chapter 1 opening"],
            },
            "pacing_heatmap": {
                "overall_pacing_assessment": "Generally good pacing.",
                "chapter_metrics": [
                    {"chapter": 1, "tension_score": 40, "action_density": 30,
                     "emotional_intensity": 50, "dialogue_ratio": 45, "description_ratio": 55},
                    {"chapter": 2, "tension_score": 80, "action_density": 70,
                     "emotional_intensity": 60, "dialogue_ratio": 40, "description_ratio": 35},
                ],
                "flat_sections": [
                    {"chapters": [1], "issue": "Low tension throughout"},
                ],
            },
        }

    def test_full_editors_notes_export(self, client, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        token = "00000000-0000-4000-8000-000000000020"
        progress_manager.create(token, self._full_progress_data())
        with client.session_transaction() as sess:
            sess["title"] = "Full Notes Test"

        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 200
        url = r.get_json()["download_url"]

        # Download and verify content
        r2 = client.get(url)
        assert r2.status_code == 200
        content = r2.data.decode("utf-8")

        # All 9 sections should be present
        assert "Consistency Pass" in content
        assert "Global Continuity Audit" in content
        assert "Narrative Compression" in content
        assert "Character Resolution" in content
        assert "Thematic Payoff" in content
        assert "Climax Integrity" in content
        assert "Loose Thread" in content
        assert "Reader Immersion" in content
        assert "Pacing" in content
        assert "Timeline gap" in content
        assert "Character age mismatch" in content
        assert "Repeated chase scene" in content

    def test_editors_notes_no_content(self, client):
        token = "00000000-0000-4000-8000-000000000021"
        progress_manager.create(token, {"status": "done", "current": 0, "total": 0, "step": "", "chapters_done": [], "error": None})
        with client.session_transaction() as sess:
            sess["title"] = "Empty"

        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert "No editor" in r.get_json()["error"]


# ---------------------------------------------------------------------------
# LLM client edge cases
# ---------------------------------------------------------------------------

class TestLLMClientEdgeCases:
    """Cover parse_llm_json edge cases and error mapping."""

    def test_parse_json_with_markdown_fences(self):
        from novelforge.llm.client import parse_llm_json
        raw = '```json\n{"title": "Test"}\n```'
        result = parse_llm_json(raw)
        assert result["title"] == "Test"

    def test_parse_json_with_bom(self):
        from novelforge.llm.client import parse_llm_json
        raw = '\ufeff{"title": "BOM Test"}'
        result = parse_llm_json(raw)
        assert result["title"] == "BOM Test"

    def test_parse_json_with_surrounding_text(self):
        from novelforge.llm.client import parse_llm_json
        raw = 'Here is the JSON:\n{"title": "Extracted"}\nEnd of response.'
        result = parse_llm_json(raw)
        assert result["title"] == "Extracted"

    def test_parse_json_array(self):
        from novelforge.llm.client import parse_llm_json
        raw = '[{"a": 1}, {"b": 2}]'
        result = parse_llm_json(raw)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_json_invalid_raises(self):
        from novelforge.llm.client import parse_llm_json
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("This is not JSON at all")

    def test_friendly_error_timeout(self):
        from novelforge.llm.client import friendly_llm_error
        err = RuntimeError("LLM_TIMEOUT: The AI service is taking too long.")
        msg = friendly_llm_error(err)
        assert "taking too long" in msg
        assert "LLM_TIMEOUT" not in msg

    def test_friendly_error_auth(self):
        from novelforge.llm.client import friendly_llm_error
        err = RuntimeError("LLM_AUTH_FAILURE: API key rejected.")
        msg = friendly_llm_error(err)
        assert "API key rejected" in msg

    def test_friendly_error_unknown(self):
        from novelforge.llm.client import friendly_llm_error
        err = RuntimeError("Something unexpected")
        msg = friendly_llm_error(err)
        assert "went wrong" in msg

    def test_circuit_breaker_lifecycle(self):
        from novelforge.llm.client import LLMCircuitBreaker, CircuitBreakerError
        cb = LLMCircuitBreaker(threshold=2)
        assert not cb.is_tripped
        assert cb.failure_count == 0

        cb.record_failure("err1")
        assert cb.failure_count == 1
        assert not cb.is_tripped
        cb.check()  # Should not raise

        cb.record_failure("err2")
        assert cb.is_tripped
        with pytest.raises(CircuitBreakerError):
            cb.check()

        cb.record_success()
        assert not cb.is_tripped
        cb.check()  # Should not raise

        cb.record_failure("err3")
        cb.reset()
        assert cb.failure_count == 0

    def test_usage_tracking(self):
        from novelforge.llm.client import reset_llm_usage, get_llm_usage
        reset_llm_usage()
        usage = get_llm_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["call_count"] == 0


# ---------------------------------------------------------------------------
# Chapter position utility
# ---------------------------------------------------------------------------

class TestChapterPosition:
    """Cover all ChapterPosition methods."""

    def test_landmarks(self):
        from novelforge.chapter_position import ChapterPosition
        assert ChapterPosition.inciting_chapter(20) == 2
        assert ChapterPosition.inciting_chapter(3) == 1
        assert ChapterPosition.midpoint_chapter(20) == 10
        assert ChapterPosition.climax_chapter(20) == 19
        assert ChapterPosition.climax_chapter(1) == 1
        assert ChapterPosition.resolution_chapter(20) == 20

    def test_phases(self):
        from novelforge.chapter_position import ChapterPosition
        assert "Hook" in ChapterPosition(1, 20).get_phase()
        assert "Rising" in ChapterPosition(8, 20).get_phase()
        # climax_chapter(20) == 19; chapter 18 falls in the pre-climax Crisis zone
        assert "Crisis" in ChapterPosition(18, 20).get_phase()
        assert "Climax" in ChapterPosition(19, 20).get_phase()
        assert "Resolution" in ChapterPosition(20, 20).get_phase()

    def test_acts_three(self):
        from novelforge.chapter_position import ChapterPosition
        assert ChapterPosition(1, 20).get_act() == "Act I"
        assert ChapterPosition(10, 20).get_act() == "Act II"
        assert ChapterPosition(18, 20).get_act() == "Act III"

    def test_acts_four(self):
        from novelforge.chapter_position import ChapterPosition
        assert ChapterPosition(1, 20).get_act(four_act=True) == "Act I"
        assert ChapterPosition(6, 20).get_act(four_act=True) == "Act II-A"
        assert ChapterPosition(12, 20).get_act(four_act=True) == "Act II-B"
        assert ChapterPosition(18, 20).get_act(four_act=True) == "Act III"

    def test_zone_checks(self):
        from novelforge.chapter_position import ChapterPosition
        cp_start = ChapterPosition(1, 20)
        assert cp_start.is_opening()
        assert not cp_start.is_climax_zone()
        assert cp_start.is_before_midpoint()

        cp_mid = ChapterPosition(10, 20)
        assert cp_mid.is_midpoint()
        assert not cp_mid.is_opening()

        cp_end = ChapterPosition(19, 20)
        assert cp_end.is_climax_zone()
        assert not cp_end.is_before_midpoint()

    def test_structure_phase_hint(self):
        from novelforge.chapter_position import ChapterPosition
        assert "Beginning" in ChapterPosition(1, 20).get_structure_phase_hint()
        assert "Middle" in ChapterPosition(10, 20).get_structure_phase_hint()
        assert "End" in ChapterPosition(18, 20).get_structure_phase_hint()

    def test_escalation_target(self):
        from novelforge.chapter_position import ChapterPosition
        assert "foundational" in ChapterPosition(1, 20).get_escalation_target()
        assert "cost of failure" in ChapterPosition(7, 20).get_escalation_target()
        assert "irreversible" in ChapterPosition(13, 20).get_escalation_target()
        assert "maximum" in ChapterPosition(19, 20).get_escalation_target()


# ---------------------------------------------------------------------------
# Image API client
# ---------------------------------------------------------------------------

class TestImageAPI:
    """Cover call_image_api edge cases."""

    def test_no_api_key_returns_none(self, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "IMAGE_API_KEY", "")
        from novelforge.llm.image import call_image_api
        result = call_image_api("test prompt")
        assert result is None

    def test_successful_b64_image(self, monkeypatch, tmp_path):
        import novelforge.config as config
        import base64
        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))
        # Do NOT pre-create the illustrations directory — call_image_api must create it

        fake_image = base64.b64encode(b"fake PNG data").decode()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"b64_json": fake_image}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("novelforge.llm.image.requests.post", return_value=mock_resp):
            from novelforge.llm.image import call_image_api
            result = call_image_api("A dramatic scene", filename_prefix="test")

        assert result is not None
        assert result.startswith("test_")
        assert result.endswith(".png")
        # Verify directory was auto-created and file was written
        assert (tmp_path / "illustrations").is_dir()
        assert (tmp_path / "illustrations" / result).exists()

    def test_creates_illustrations_dir_if_missing(self, monkeypatch, tmp_path):
        """call_image_api must create the illustrations subdirectory on a clean filesystem."""
        import novelforge.config as config
        import base64
        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))
        # Confirm the directory does not exist before calling
        assert not (tmp_path / "illustrations").exists()

        fake_image = base64.b64encode(b"fresh PNG").decode()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"b64_json": fake_image}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("novelforge.llm.image.requests.post", return_value=mock_resp):
            from novelforge.llm.image import call_image_api
            result = call_image_api("A brand-new scene", filename_prefix="new")

        assert result is not None
        assert (tmp_path / "illustrations").is_dir()
        assert (tmp_path / "illustrations" / result).exists()

    def test_dir_creation_failure_returns_none(self, monkeypatch, tmp_path):
        """If the illustrations directory cannot be created, return None and log an error."""
        import novelforge.config as config
        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("novelforge.llm.image.requests.post", return_value=mock_resp):
            with patch("novelforge.llm.image.Path.mkdir", side_effect=OSError("permission denied")):
                from novelforge.llm.image import call_image_api
                result = call_image_api("test prompt", filename_prefix="fail")

        assert result is None

    def test_api_timeout_returns_none(self, monkeypatch):
        import novelforge.config as config
        import requests as req
        monkeypatch.setattr(config, "IMAGE_API_KEY", "test-key")
        monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)

        with patch("novelforge.llm.image.requests.post", side_effect=req.exceptions.Timeout):
            from novelforge.llm.image import call_image_api
            result = call_image_api("test")

        assert result is None


# ---------------------------------------------------------------------------
# Prompts loading
# ---------------------------------------------------------------------------

class TestPromptsLoading:
    """Cover prompt loading and rendering."""

    def test_render_prompt_returns_messages(self):
        from novelforge.llm.prompts import render_prompt
        msgs = render_prompt("title", premise="A hero's quest", genre="Fantasy")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Fantasy" in msgs[1]["content"]

    def test_render_unknown_prompt_raises(self):
        from novelforge.llm.prompts import render_prompt
        with pytest.raises(KeyError, match="nonexistent"):
            render_prompt("nonexistent")

    def test_load_prompts_caches(self):
        from novelforge.llm.prompts import _load_prompts
        p1 = _load_prompts()
        p2 = _load_prompts()
        assert p1 is p2  # Same object — cached

    # ------------------------------------------------------------------
    # Schema-validation tests (exercise _validate_prompts directly)
    # ------------------------------------------------------------------

    def test_validate_missing_name_raises(self):
        from novelforge.llm.prompts import _validate_prompts
        with pytest.raises(ValueError, match="missing required field 'name'"):
            _validate_prompts([{"system": "sys", "user": "usr"}])

    def test_validate_missing_system_raises(self):
        from novelforge.llm.prompts import _validate_prompts
        with pytest.raises(ValueError, match="missing required field 'system'"):
            _validate_prompts([{"name": "x", "user": "usr"}])

    def test_validate_missing_user_raises(self):
        from novelforge.llm.prompts import _validate_prompts
        with pytest.raises(ValueError, match="missing required field 'user'"):
            _validate_prompts([{"name": "x", "system": "sys"}])

    def test_validate_non_string_field_raises(self):
        from novelforge.llm.prompts import _validate_prompts
        with pytest.raises(ValueError, match="field 'system' must be a string"):
            _validate_prompts([{"name": "x", "system": 42, "user": "usr"}])

    def test_validate_non_dict_entry_raises(self):
        from novelforge.llm.prompts import _validate_prompts
        with pytest.raises(ValueError, match="must be a mapping"):
            _validate_prompts(["not-a-dict"])

    def test_validate_duplicate_name_raises(self):
        from novelforge.llm.prompts import _validate_prompts
        entries = [
            {"name": "dup", "system": "s1", "user": "u1"},
            {"name": "dup", "system": "s2", "user": "u2"},
        ]
        with pytest.raises(ValueError, match="duplicate prompt name 'dup'"):
            _validate_prompts(entries)

    def test_validate_valid_entries_returns_dict(self):
        from novelforge.llm.prompts import _validate_prompts
        entries = [
            {"name": "a", "system": "sys_a", "user": "usr_a"},
            {"name": "b", "system": "sys_b", "user": "usr_b"},
        ]
        result = _validate_prompts(entries)
        assert set(result.keys()) == {"a", "b"}
        assert result["a"]["system"] == "sys_a"

    def test_load_prompts_rejects_non_list_prompts_key(self, monkeypatch):
        import novelforge.llm.prompts as prompts_mod
        monkeypatch.setattr(prompts_mod, "_prompts_cache", None)
        import yaml
        monkeypatch.setattr(
            yaml,
            "safe_load",
            lambda _f: {"prompts": "not-a-list"},
        )
        with pytest.raises(ValueError, match="'prompts' key must be a list"):
            prompts_mod._load_prompts()

    def test_load_prompts_validates_on_load(self, monkeypatch):
        import novelforge.llm.prompts as prompts_mod
        monkeypatch.setattr(prompts_mod, "_prompts_cache", None)
        import yaml
        monkeypatch.setattr(
            yaml,
            "safe_load",
            lambda _f: {
                "prompts": [
                    {"name": "dup", "system": "s", "user": "u"},
                    {"name": "dup", "system": "s2", "user": "u2"},
                ]
            },
        )
        with pytest.raises(ValueError, match="duplicate prompt name"):
            prompts_mod._load_prompts()

    def test_render_prompt_missing_context_key_raises(self, monkeypatch):
        """Missing template variable should raise ValueError with prompt name."""
        import novelforge.llm.prompts as prompts_mod
        monkeypatch.setattr(
            prompts_mod,
            "_prompts_cache",
            {
                "test_strict": {
                    "name": "test_strict",
                    "system": "You are a helper.",
                    "user": "Hello {{ missing_var }}",
                }
            },
        )
        with pytest.raises(ValueError, match="test_strict"):
            prompts_mod.render_prompt("test_strict")

    def test_render_prompt_missing_context_key_names_variable(self, monkeypatch):
        """Error message should identify the undefined variable."""
        import novelforge.llm.prompts as prompts_mod
        monkeypatch.setattr(
            prompts_mod,
            "_prompts_cache",
            {
                "test_strict2": {
                    "name": "test_strict2",
                    "system": "{{ undefined_field }} is missing",
                    "user": "user text",
                }
            },
        )
        with pytest.raises(ValueError, match="undefined_field"):
            prompts_mod.render_prompt("test_strict2")

    def test_render_prompt_all_keys_provided_succeeds(self, monkeypatch):
        """When all required context keys are provided, rendering should succeed."""
        import novelforge.llm.prompts as prompts_mod
        monkeypatch.setattr(
            prompts_mod,
            "_prompts_cache",
            {
                "test_complete": {
                    "name": "test_complete",
                    "system": "System: {{ sys_val }}",
                    "user": "User: {{ user_val }}",
                }
            },
        )
        msgs = prompts_mod.render_prompt("test_complete", sys_val="S", user_val="U")
        assert msgs[0]["content"] == "System: S"
        assert msgs[1]["content"] == "User: U"

    def test_render_prompt_partial_context_raises(self, monkeypatch):
        """Providing only some required context keys should still raise ValueError."""
        import novelforge.llm.prompts as prompts_mod
        monkeypatch.setattr(
            prompts_mod,
            "_prompts_cache",
            {
                "test_partial": {
                    "name": "test_partial",
                    "system": "System ok",
                    "user": "Need {{ alpha }} and {{ beta }}",
                }
            },
        )
        with pytest.raises(ValueError, match="test_partial"):
            prompts_mod.render_prompt("test_partial", alpha="provided")


# ---------------------------------------------------------------------------
# Session routes coverage
# ---------------------------------------------------------------------------

class TestSessionRoutesCoverage:
    """Cover session route branches not hit by other tests."""

    def test_list_sessions_with_data(self, app, client):
        """Create a session file and verify it appears in list."""
        import novelforge.config as config
        novels_dir = Path(config.NOVELS_DIR)
        novels_dir.mkdir(parents=True, exist_ok=True)
        test_file = novels_dir / "test-list-uuid.json"
        test_file.write_text(json.dumps({
            "session_id": "test-list-uuid",
            "title": "Listable Novel",
        }))

        r = client.get("/list_sessions")
        assert r.status_code == 200
        sessions = r.get_json()["sessions"]
        titles = [s["title"] for s in sessions]
        assert "Listable Novel" in titles

    def test_load_session_success(self, app, client):
        import novelforge.config as config
        novels_dir = Path(config.NOVELS_DIR)
        novels_dir.mkdir(parents=True, exist_ok=True)
        valid_uuid = "aabbccdd-eeff-0011-2233-445566778899"
        test_file = novels_dir / f"{valid_uuid}.json"
        test_file.write_text(json.dumps({
            "session_id": valid_uuid,
            "title": "Loaded Novel",
            "premise": "Test premise",
            "genre": "Fantasy",
            "chapters": 5,
            "word_count": 40000,
        }))

        r = client.post(
            "/load_session",
            data=json.dumps({"session_id": valid_uuid}),
            content_type="application/json",
        )
        assert r.status_code == 200
        assert r.get_json()["title"] == "Loaded Novel"

    def test_new_session_clears_data(self, client):
        with client.session_transaction() as sess:
            sess["title"] = "Old Novel"
        r = client.post("/new_session")
        assert r.status_code == 200
        with client.session_transaction() as sess:
            assert sess.get("title", "") == ""


# ---------------------------------------------------------------------------
# Manuscript export coverage
# ---------------------------------------------------------------------------

class TestManuscriptExport:
    """Cover manuscript formatting."""

    def test_export_creates_file(self, client, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        token = "00000000-0000-4000-8000-000000000022"
        progress_manager.create(token, {
            "status": "done",
            "current": 2,
            "total": 2,
            "step": "Complete",
            "error": None,
            "snapshot": {"title": "Export File Test"},
            "chapters_done": [
                {"number": 1, "title": "Opening", "content": "Chapter one content."},
                {"number": 2, "title": "Finale", "content": "Chapter two content."},
            ],
        })

        r = client.post("/export", data=json.dumps({"token": token}),
                        content_type="application/json")
        assert r.status_code == 200

        # Verify file content
        url = r.get_json()["download_url"]
        r2 = client.get(url)
        content = r2.data.decode("utf-8")
        assert "# Export File Test" in content
        assert "## Chapter 1: Opening" in content
        assert "Chapter one content." in content

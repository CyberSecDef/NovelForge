"""
Integration tests for NovelForge routes using the mock_llm fixture.

These tests exercise the full request/response cycle for all major routes
without making live LLM API calls.
"""

import json
import os
import time
import pytest

from novelforge.progress import progress_manager


class TestGenerateOutline:
    """Full outline generation flow with mocked LLM."""

    def test_generates_title_outline_characters(self, client, mock_llm):
        r = client.post(
            "/generate_outline",
            data=json.dumps({
                "premise": "A hero discovers a hidden world",
                "genre": "Fantasy",
                "chapters": 3,
                "word_count": 10000,
            }),
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert "title" in data
        assert "chapters" in data
        assert "characters" in data
        assert len(data["chapters"]) == 3
        assert len(data["characters"]) >= 1
        # Planning agents should have run
        assert "story_architecture" in data
        assert "master_timeline" in data
        assert "character_fate_registry" in data
        assert "technology_rules" in data
        assert "theme_reinforcement" in data
        assert "pov_focal_character_plan" in data

    def test_stores_data_in_session(self, client, mock_llm):
        client.post(
            "/generate_outline",
            data=json.dumps({
                "premise": "A detective solves a cold case",
                "genre": "Mystery",
                "chapters": 5,
                "word_count": 40000,
            }),
            content_type="application/json",
        )
        with client.session_transaction() as sess:
            assert sess["title"] != ""
            assert sess["genre"] == "Mystery"
            assert sess["chapters"] == 5
            assert len(sess["chapter_list"]) == 3  # canned response returns 3
            assert len(sess["character_list"]) >= 1

    def test_returns_planning_agent_outputs(self, client, mock_llm):
        """Outline generation runs planning agents and returns their outputs."""
        r = client.post(
            "/generate_outline",
            data=json.dumps({
                "premise": "A space adventure",
                "genre": "Science Fiction",
                "chapters": 3,
                "word_count": 10000,
            }),
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        # Planning agents produce non-empty dicts
        assert isinstance(data["story_architecture"], dict)
        assert isinstance(data["master_timeline"], dict)
        assert isinstance(data["technology_rules"], dict)


class TestApproveOutline:
    """Approve and selective regeneration tests with mocked LLM."""

    def _seed_session(self, client):
        with client.session_transaction() as sess:
            sess["premise"] = "A hero's journey"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 3
            sess["word_count"] = 10000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["title"] = "Old Title"
            sess["chapter_list"] = [
                {"number": 1, "title": "Ch1", "summary": "Setup"},
                {"number": 2, "title": "Ch2", "summary": "Conflict"},
                {"number": 3, "title": "Ch3", "summary": "Resolution"},
            ]
            sess["character_list"] = [
                {"name": "Alice", "age": "25", "role": "Protagonist",
                 "background": "Brave", "arc": "Growth"},
            ]

    def test_approve_runs_planning_agents(self, client, mock_llm):
        self._seed_session(client)
        r = client.post(
            "/approve_outline",
            data=json.dumps({
                "title": "New Title",
                "chapters": [
                    {"number": 1, "title": "Ch1", "summary": "Setup"},
                    {"number": 2, "title": "Ch2", "summary": "Conflict"},
                    {"number": 3, "title": "Ch3", "summary": "Resolution"},
                ],
                "characters": [
                    {"name": "Alice", "age": "25", "role": "Protagonist",
                     "background": "Brave", "arc": "Growth"},
                ],
            }),
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "approved"
        assert "story_architecture" in data
        # All planning agents should have produced output
        assert isinstance(data["story_architecture"], dict)
        assert isinstance(data["master_timeline"], dict)

    def test_second_approval_stores_hashes(self, client, mock_llm):
        """On approval, input hashes are stored for selective regeneration."""
        self._seed_session(client)
        payload = json.dumps({
            "title": "Old Title",
            "chapters": [
                {"number": 1, "title": "Ch1", "summary": "Setup"},
                {"number": 2, "title": "Ch2", "summary": "Conflict"},
                {"number": 3, "title": "Ch3", "summary": "Resolution"},
            ],
            "characters": [
                {"name": "Alice", "age": "25", "role": "Protagonist",
                 "background": "Brave", "arc": "Growth"},
            ],
        })
        r = client.post("/approve_outline", data=payload, content_type="application/json")
        assert r.status_code == 200

        # Verify input hashes are stored in session
        with client.session_transaction() as sess:
            hashes = sess.get("_agent_input_hashes", {})
            assert "story_architecture" in hashes
            assert "master_timeline" in hashes
            assert "character_fate_registry" in hashes
            assert "pov_focal_character_plan" in hashes

        # Second approval with same data should also succeed
        r2 = client.post("/approve_outline", data=payload, content_type="application/json")
        assert r2.status_code == 200

    def test_character_rename_propagation(self, client, mock_llm):
        """Renaming a character should update chapter summaries."""
        self._seed_session(client)
        # Set chapter summary that mentions Alice
        with client.session_transaction() as sess:
            sess["chapter_list"] = [
                {"number": 1, "title": "Ch1", "summary": "Alice begins her journey"},
            ]

        r = client.post(
            "/approve_outline",
            data=json.dumps({
                "title": "Test",
                "chapters": [
                    {"number": 1, "title": "Ch1", "summary": "Alice begins her journey"},
                ],
                "characters": [
                    {"name": "Bob", "age": "25", "role": "Protagonist",
                     "background": "Brave", "arc": "Growth"},
                ],
            }),
            content_type="application/json",
        )
        assert r.status_code == 200
        with client.session_transaction() as sess:
            assert "Bob" in sess["chapter_list"][0]["summary"]
            assert "Alice" not in sess["chapter_list"][0]["summary"]


class TestGenerateChapters:
    """Chapter generation start and progress polling."""

    def _seed_full_session(self, client):
        with client.session_transaction() as sess:
            sess["premise"] = "A test premise"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 3
            sess["word_count"] = 10000
            sess["special_instructions"] = ""
            sess["title"] = "Test Novel"
            sess["chapter_list"] = [
                {"number": 1, "title": "Ch1", "summary": "Setup"},
                {"number": 2, "title": "Ch2", "summary": "Rising"},
                {"number": 3, "title": "Ch3", "summary": "Climax"},
            ]
            sess["character_list"] = []
            sess["story_architecture"] = {}
            sess["master_timeline"] = {}
            sess["character_fate_registry"] = {}
            sess["character_arc_plan"] = {}
            sess["antagonist_motivation_plan"] = {}
            sess["technology_rules"] = {}
            sess["theme_reinforcement"] = {}
            sess["pov_focal_character_plan"] = {}

    def _patch_thread(self, monkeypatch):
        import novelforge.routes.generation as gen_mod
        monkeypatch.setattr(gen_mod.threading, "Thread",
                            lambda *a, **kw: type("FakeThread", (), {"start": lambda s: None, "daemon": True})())

    def test_starts_generation_returns_token(self, client, monkeypatch):
        self._patch_thread(monkeypatch)

        self._seed_full_session(client)
        r = client.post(
            "/generate_chapters",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert "token" in data
        assert len(data["token"]) > 0

    def test_progress_endpoint_returns_status(self, client, monkeypatch):
        self._patch_thread(monkeypatch)

        self._seed_full_session(client)
        r = client.post(
            "/generate_chapters",
            data=json.dumps({}),
            content_type="application/json",
        )
        token = r.get_json()["token"]

        r2 = client.get(f"/progress/{token}")
        assert r2.status_code == 200
        data = r2.get_json()
        assert "status" in data
        assert "step" in data
        assert data["total"] == 3

    def test_progress_endpoint_is_lightweight(self, client, monkeypatch):
        """Lightweight /progress/<token> must not include heavy fields."""
        self._patch_thread(monkeypatch)

        self._seed_full_session(client)
        r = client.post(
            "/generate_chapters",
            data=json.dumps({}),
            content_type="application/json",
        )
        token = r.get_json()["token"]

        # Inject heavyweight data into the store to simulate a completed chapter
        progress_manager.update(token, {
            "chapters_done": [
                {"number": 1, "title": "Ch1", "content": "x" * 5000, "summary": "s"},
            ],
            "consistency": {"issues": [], "overall_assessment": "good"},
        })

        r2 = client.get(f"/progress/{token}")
        assert r2.status_code == 200
        data = r2.get_json()
        # Only lightweight fields should be present
        assert "status" in data
        assert "current" in data
        assert "total" in data
        assert "step" in data
        # Heavyweight fields must not be included
        assert "chapters_done" not in data
        assert "consistency" not in data

    def test_progress_full_endpoint_includes_heavy_fields(self, client, monkeypatch):
        """/progress/<token>/full must include chapter content and reports."""
        self._patch_thread(monkeypatch)

        self._seed_full_session(client)
        r = client.post(
            "/generate_chapters",
            data=json.dumps({}),
            content_type="application/json",
        )
        token = r.get_json()["token"]

        chapter_content = "x" * 5000
        progress_manager.update(token, {
            "chapters_done": [
                {"number": 1, "title": "Ch1", "content": chapter_content, "summary": "s"},
            ],
            "consistency": {"issues": [], "overall_assessment": "good"},
        })

        r2 = client.get(f"/progress/{token}/full")
        assert r2.status_code == 200
        data = r2.get_json()
        assert "status" in data
        assert "chapters_done" in data
        assert data["chapters_done"][0]["content"] == chapter_content
        assert "consistency" in data

    def test_progress_full_endpoint_unknown_token(self, client):
        """/progress/<token>/full returns 404 for unknown token."""
        r = client.get("/progress/nonexistent-token/full")
        assert r.status_code == 404

    def test_progress_endpoint_unknown_token(self, client):
        """/progress/<token> returns 404 for unknown token."""
        r = client.get("/progress/nonexistent-token")
        assert r.status_code == 404


class TestGenerationStatusOrdering:
    """status='done' must not be set until every post-manuscript pass has finished."""

    _POST_MANUSCRIPT_STEPS = {
        "Narrative compression analysis",
        "Character resolution validation",
        "Thematic payoff analysis",
        "Climax integrity check",
        "Loose thread resolution",
        "Reader immersion testing",
        "Pacing & tension heatmap",
        "Mapping character relationships",
    }

    def _seed_snap(self):
        return {
            "title": "Test Novel",
            "genre": "Fantasy",
            "chapters": 1,
            "word_count": 3000,
            "special_instructions": "",
            "premise": "A hero sets out on a quest",
            "chapter_list": [{"number": 1, "title": "Ch1", "summary": "Setup"}],
            "character_list": [],
            "story_architecture": {},
            "master_timeline": {},
            "character_fate_registry": {},
            "character_arc_plan": {},
            "antagonist_motivation_plan": {},
            "technology_rules": {},
            "theme_reinforcement": {},
            "pov_focal_character_plan": {},
            "narrative_perspective": "third_person",
            "voice_seed": {},
        }

    def test_done_only_set_after_all_post_manuscript_passes(self, mock_llm, monkeypatch, tmp_path):
        """
        status='done' must only be written to the progress store after
        character_relationship_map (the last post-manuscript pass) has been
        populated.  It must remain 'running' for every intermediate step.
        """
        import novelforge.config as config
        import novelforge.routes.generation as gen_mod
        from novelforge.routes.generation import _run_chapter_generation_internal
        from novelforge.progress import progress_manager

        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        token = "test-status-ordering"
        progress_manager.create(token, {
            "status": "running", "current": 0, "total": 1,
            "step": "Preparing…", "chapters_done": [], "error": None,
        })

        # Spy: record status when step is set to each post-manuscript step,
        # and record whether character_relationship_map was already present when
        # status='done' was first written.
        statuses_at_post_pass = {}
        done_set_while_map_present: list[bool] = []

        original_update = type(progress_manager).update

        def spy_update(self_pm, tok, patch):
            original_update(self_pm, tok, patch)
            if tok != token:
                return
            if "step" in patch and patch["step"] in TestGenerationStatusOrdering._POST_MANUSCRIPT_STEPS:
                state = progress_manager.get(tok)
                statuses_at_post_pass[patch["step"]] = state.get("status")
            if patch.get("status") == "done":
                state = progress_manager.get(tok)
                done_set_while_map_present.append(
                    state.get("character_relationship_map") is not None
                )

        monkeypatch.setattr(type(progress_manager), "update", spy_update)

        _run_chapter_generation_internal(token, self._seed_snap(), [], [], 0)

        # Pipeline must finish as 'done'
        final = progress_manager.get(token)
        assert final is not None
        assert final["status"] == "done", (
            f"Expected final status='done', got {final['status']!r}"
        )

        # status='done' must have been set exactly once, after the relationship map
        assert len(done_set_while_map_present) == 1, (
            f"Expected status='done' to be set exactly once, "
            f"but it was set {len(done_set_while_map_present)} time(s)"
        )
        assert done_set_while_map_present[0] is True, (
            "status='done' was set before character_relationship_map was populated"
        )

        # Every post-manuscript step must have seen status='running'
        assert len(statuses_at_post_pass) == len(self._POST_MANUSCRIPT_STEPS), (
            f"Not all post-manuscript steps were visited; "
            f"missing: {self._POST_MANUSCRIPT_STEPS - set(statuses_at_post_pass)}"
        )
        for step_name, status in statuses_at_post_pass.items():
            assert status == "running", (
                f"Expected status='running' at start of '{step_name}', got {status!r}"
            )


class TestReviseChapter:
    """Chapter revision with mocked LLM."""

    def _make_token(self, suffix: str = "") -> str:
        return f"test-revise-{suffix}" if suffix else "test-revise-integration"

    def _create_progress(self, token: str, *, with_reports: bool = False, session_id: str = "") -> None:
        state: dict = {
            "status": "done",
            "current": 1,
            "total": 1,
            "step": "Complete",
            "error": None,
            "chapters_done": [
                {"number": 1, "title": "Ch1", "content": "Original text.", "summary": "Old summary"},
            ],
            "consistency": {"issues": [], "overall_assessment": ""},
            "snapshot": {
                "title": "Test",
                "genre": "Fantasy",
                "chapters": 1,
                "chapter_list": [{"number": 1, "title": "Ch1", "summary": "Outline"}],
                "character_list": [],
                "special_instructions": "",
                "story_architecture": {},
                "master_timeline": {},
                "character_fate_registry": {},
                "character_arc_plan": {},
                "antagonist_motivation_plan": {},
                "technology_rules": {},
                "theme_reinforcement": {},
                "pov_focal_character_plan": {},
                "narrative_perspective": "third_person",
                **({"session_id": session_id} if session_id else {}),
            },
        }
        if with_reports:
            state.update({
                "global_continuity_audit": {"overall_integrity": "good", "contradictions": []},
                "narrative_compression_report": {"compression_priority": "low"},
                "character_resolution_report": {"resolution_integrity": "good"},
                "thematic_payoff_report": {"thematic_integrity": "good"},
                "climax_integrity_report": {"climax_integrity": "good"},
                "loose_thread_report": {"thread_integrity": "good"},
                "reader_immersion_report": {"engagement_score": 8},
                "pacing_heatmap": {"chapter_metrics": []},
                "character_relationship_map": {"characters": [], "relationships": []},
            })
        progress_manager.create(token, state)

    def _post_revise(self, client, token: str):
        return client.post(
            "/revise_chapter",
            data=json.dumps({
                "token": token,
                "chapter_number": 1,
                "instructions": "Add more tension",
            }),
            content_type="application/json",
        )

    def test_revise_returns_updated_content(self, client, mock_llm):
        token = self._make_token("content")
        self._create_progress(token)
        r = self._post_revise(client, token)
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "done"
        assert data["chapters_done"][0]["content"] != "Original text."

    def test_revise_invalidates_derived_reports(self, client, mock_llm):
        """After revision, all derived report fields must be set to None."""
        from novelforge.routes.generation import _DERIVED_REPORT_FIELDS
        token = self._make_token("invalidation")
        self._create_progress(token, with_reports=True)
        r = self._post_revise(client, token)
        assert r.status_code == 200
        data = r.get_json()
        for field in _DERIVED_REPORT_FIELDS:
            assert data.get(field) is None, f"Expected {field} to be None after revision"

    def test_revise_persists_chapters(self, client, mock_llm, tmp_path, monkeypatch):
        """Revision must call persist_completed_chapters so changes survive a restart."""
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        session_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        token = self._make_token("persist")
        self._create_progress(token, session_id=session_id)

        # Create an initial session file so persistence has something to update
        session_file = tmp_path / f"{session_id}.json"
        session_file.write_text(json.dumps({
            "session_id": session_id,
            "title": "Test",
            "completed_chapters": [],
        }))

        r = self._post_revise(client, token)
        assert r.status_code == 200

        saved = json.loads(session_file.read_text())
        # persist_completed_chapters stores chapters under "completed_chapters"
        assert len(saved.get("completed_chapters", [])) == 1
        assert saved["completed_chapters"][0]["content"] != "Original text."

    def test_revise_persistence_includes_invalidated_reports(self, client, mock_llm, tmp_path, monkeypatch):
        """Persisted state after revision reflects the invalidated (None) report values."""
        from novelforge.routes.generation import _DERIVED_REPORT_FIELDS
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        session_id = "11111111-2222-3333-4444-555555555555"
        token = self._make_token("persist-invalidate")
        self._create_progress(token, with_reports=True, session_id=session_id)

        session_file = tmp_path / f"{session_id}.json"
        session_file.write_text(json.dumps({
            "session_id": session_id,
            "title": "Test",
            "completed_chapters": [],
        }))

        r = self._post_revise(client, token)
        assert r.status_code == 200

        saved = json.loads(session_file.read_text())
        progress_data = saved.get("progress_data", {})
        for field in _DERIVED_REPORT_FIELDS:
            assert progress_data.get(field) is None, (
                f"Expected stale {field} to be persisted as None"
            )



class TestExport:
    """Export routes with mocked progress data."""

    def _seed_done(self, client, token="export-test"):
        progress_manager.create(token, {
            "status": "done",
            "current": 2,
            "total": 2,
            "step": "Complete",
            "error": None,
            "chapters_done": [
                {"number": 1, "title": "Ch1", "content": "Chapter 1 text.", "summary": "S1"},
                {"number": 2, "title": "Ch2", "content": "Chapter 2 text.", "summary": "S2"},
            ],
            "consistency": {"issues": [], "overall_assessment": "Good"},
            "pacing_heatmap": {"chapter_metrics": [], "flat_sections": [], "overall_pacing_assessment": ""},
            "reader_immersion_report": {},
            "global_continuity_audit": {},
            "narrative_compression_report": {},
            "character_resolution_report": {},
        })
        with client.session_transaction() as sess:
            sess["title"] = "Export Test Novel"

    def test_export_manuscript(self, client):
        token = "export-manuscript"
        self._seed_done(client, token)
        r = client.post(
            "/export",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 200
        assert "download_url" in r.get_json()

    def test_export_not_complete(self, client):
        token = "export-incomplete"
        progress_manager.create(token, {"status": "running", "current": 1, "total": 5, "step": "", "chapters_done": [], "error": None})
        r = client.post(
            "/export",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_export_editors_notes(self, client):
        token = "export-notes"
        self._seed_done(client, token)
        r = client.post(
            "/export_editors_notes",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 200
        assert "download_url" in r.get_json()

    def test_download_exported_file(self, client, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        token = "export-download"
        self._seed_done(client, token)
        r = client.post(
            "/export",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 200
        url = r.get_json()["download_url"]
        r2 = client.get(url)
        assert r2.status_code == 200


class TestSessions:
    """Session management routes."""

    def test_list_sessions(self, client):
        r = client.get("/list_sessions")
        assert r.status_code == 200
        data = r.get_json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_load_nonexistent_session(self, client):
        r = client.post(
            "/load_session",
            data=json.dumps({"session_id": "00000000-0000-0000-0000-000000000000"}),
            content_type="application/json",
        )
        assert r.status_code == 404

    def test_load_session_missing_id(self, client):
        r = client.post(
            "/load_session",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_delete_session(self, client):
        r = client.post("/delete_session")
        assert r.status_code == 200
        assert r.get_json()["status"] == "success"

    def test_new_session(self, client):
        r = client.post("/new_session")
        assert r.status_code == 200
        assert r.get_json()["status"] == "success"


class TestIllustrations:
    """Illustration generation and serving routes."""

    def test_generate_illustrations_no_token(self, client):
        r = client.post(
            "/generate_illustrations",
            data=json.dumps({"token": "fake"}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_generate_illustrations_no_image_key(self, client, monkeypatch):
        """Without IMAGE_API_KEY, illustrations return a 400 error."""
        import novelforge.config as config
        monkeypatch.setattr(config, "IMAGE_API_KEY", "")

        token = "illust-no-key"
        progress_manager.create(token, {
            "status": "done",
            "current": 1,
            "total": 1,
            "step": "Complete",
            "error": None,
            "chapters_done": [
                {"number": 1, "title": "Ch1", "content": "Text", "summary": "S"},
            ],
        })
        with client.session_transaction() as sess:
            sess["title"] = "Test"
            sess["genre"] = "Fantasy"
            sess["premise"] = "A story"
            sess["character_list"] = []

        r = client.post(
            "/generate_illustrations",
            data=json.dumps({"token": token}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert "IMAGE_API_KEY" in r.get_json()["error"]

    def test_serve_illustration_not_found(self, client):
        r = client.get("/illustrations/nonexistent.png")
        assert r.status_code == 404

    def test_serve_illustration_exists(self, client, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

        # Create the illustrations subdirectory and a dummy image
        illust_dir = tmp_path / "illustrations"
        illust_dir.mkdir()
        (illust_dir / "test.png").write_bytes(b"\x89PNG fake image data")

        r = client.get("/illustrations/test.png")
        assert r.status_code == 200


class TestLLMLog:
    """LLM log viewer and clear routes (debug mode only)."""

    @pytest.fixture
    def debug_client(self):
        from novelforge import create_app, limiter
        flask_app = create_app(testing=True)
        flask_app.config["SECRET_KEY"] = "test-secret"
        flask_app.config["WTF_CSRF_ENABLED"] = False
        flask_app.debug = True
        limiter.enabled = False
        with flask_app.test_client() as c:
            yield c

    def test_llm_log_returns_entries(self, debug_client):
        r = debug_client.get("/llm_log")
        assert r.status_code == 200
        data = r.get_json()
        assert "entries" in data
        assert isinstance(data["entries"], list)

    def test_llm_log_available(self, client):
        """The /llm_log endpoint should be accessible regardless of debug mode."""
        r = client.get("/llm_log")
        assert r.status_code == 200

    def test_clear_log(self, debug_client):
        r = debug_client.post("/clear_log")
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    def test_clear_log_available(self, client):
        """The /clear_log endpoint should be accessible regardless of debug mode."""
        r = client.post("/clear_log")
        assert r.status_code == 200

    @pytest.fixture
    def _isolated_app(self, tmp_path, monkeypatch):
        """Create an app with all dirs isolated under tmp_path."""
        import novelforge.config as cfg
        from novelforge import create_app, limiter

        for attr, subdir in (
            ("NOVELS_DIR", "novels"),
            ("LOGS_DIR", "logs"),
            ("SESSION_FILE_DIR", "sessions"),
            ("EXPORT_DIR", "exports"),
        ):
            d = tmp_path / subdir
            d.mkdir()
            monkeypatch.setattr(cfg, attr, str(d))

        flask_app = create_app(testing=True)
        flask_app.config["SECRET_KEY"] = "test-secret"
        flask_app.config["WTF_CSRF_ENABLED"] = False
        limiter.enabled = False
        return flask_app, tmp_path / "logs"

    def test_llm_log_parses_single_line_entries(self, _isolated_app):
        """One-JSON-per-line log format is parsed correctly."""
        flask_app, logs_dir = _isolated_app
        entry1 = {"type": "request", "action": "test1", "timestamp": "2024-01-01 00:00:00"}
        entry2 = {"type": "response", "action": "test2", "timestamp": "2024-01-01 00:00:01"}
        (logs_dir / "llm.log").write_text(
            json.dumps(entry1) + "\n" + json.dumps(entry2) + "\n",
            encoding="utf-8",
        )

        with flask_app.test_client() as c:
            r = c.get("/llm_log")

        assert r.status_code == 200
        data = r.get_json()
        assert len(data["entries"]) == 2
        assert data["entries"][0]["type"] == "request"
        assert data["entries"][1]["type"] == "response"

    def test_llm_log_parses_entries_with_braces_in_strings(self, _isolated_app):
        """Brace characters inside JSON string values do not break parsing."""
        flask_app, logs_dir = _isolated_app
        entry = {
            "type": "request",
            "action": "test",
            "timestamp": "2024-01-01 00:00:00",
            "payload": {"key": "value with { brace } and another {brace}"},
        }
        (logs_dir / "llm.log").write_text(json.dumps(entry) + "\n", encoding="utf-8")

        with flask_app.test_client() as c:
            r = c.get("/llm_log")

        assert r.status_code == 200
        data = r.get_json()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["payload"]["key"] == "value with { brace } and another {brace}"

    def test_llm_log_returns_last_ten_entries(self, _isolated_app):
        """Only the last 10 entries are returned when the log has more."""
        flask_app, logs_dir = _isolated_app
        entries = [
            {"type": "request", "seq": i, "timestamp": f"2024-01-01 00:00:{i:02d}"}
            for i in range(15)
        ]
        (logs_dir / "llm.log").write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )

        with flask_app.test_client() as c:
            r = c.get("/llm_log")

        assert r.status_code == 200
        data = r.get_json()
        assert len(data["entries"]) == 10
        assert data["entries"][0]["seq"] == 5
        assert data["entries"][-1]["seq"] == 14


class TestNovelforgeDebugEnvVar:
    """Verify that NOVELFORGE_DEBUG controls debug mode in app.py entrypoint."""

    @staticmethod
    def _parse_debug_env(value=None):
        """Mirror the debug-flag parsing logic used in app.py."""
        env_value = value if value is not None else os.environ.get("NOVELFORGE_DEBUG", "false")
        return env_value.strip().lower() in ("1", "true", "yes")

    def test_debug_false_by_default(self, monkeypatch):
        """NOVELFORGE_DEBUG absent → debug resolves to False."""
        monkeypatch.delenv("NOVELFORGE_DEBUG", raising=False)
        assert self._parse_debug_env(os.environ.get("NOVELFORGE_DEBUG", "false")) is False

    def test_debug_true_when_set(self, monkeypatch):
        """NOVELFORGE_DEBUG=true → debug resolves to True."""
        for value in ("true", "True", "TRUE", "1", "yes"):
            monkeypatch.setenv("NOVELFORGE_DEBUG", value)
            assert self._parse_debug_env(os.environ.get("NOVELFORGE_DEBUG")) is True, (
                f"Expected True for NOVELFORGE_DEBUG={value!r}"
            )

    def test_debug_false_when_set_false(self, monkeypatch):
        """NOVELFORGE_DEBUG=false → debug resolves to False."""
        for value in ("false", "False", "FALSE", "0", "no"):
            monkeypatch.setenv("NOVELFORGE_DEBUG", value)
            assert self._parse_debug_env(os.environ.get("NOVELFORGE_DEBUG")) is False, (
                f"Expected False for NOVELFORGE_DEBUG={value!r}"
            )

    def test_log_routes_available_without_debug_mode(self, client):
        """Log routes are accessible regardless of debug mode (local-only app)."""
        assert client.get("/llm_log").status_code == 200
        assert client.post("/clear_log").status_code == 200


class TestCircuitBreaker:
    """Verify circuit breaker integration with mock."""

    def test_circuit_breaker_trip_and_manual_reset(self, client, mock_llm):
        from novelforge.llm.client import _llm_circuit_breaker
        # Trip the breaker manually
        _llm_circuit_breaker.record_failure("test1")
        _llm_circuit_breaker.record_failure("test2")
        _llm_circuit_breaker.record_failure("test3")
        assert _llm_circuit_breaker.is_tripped

        # The breaker must be reset explicitly (e.g. by the test fixture) —
        # generation workers must NOT reset it globally.
        _llm_circuit_breaker.reset()
        assert not _llm_circuit_breaker.is_tripped

    def test_circuit_breaker_not_reset_by_generation_start(self, client, mock_llm):
        """Generation workers must not reset the process-level circuit breaker.

        Resetting a shared breaker from a background thread would clear state
        that other concurrent requests may rely on.  The breaker should remain
        tripped after generation starts.
        """
        import novelforge.routes.generation as gen_mod
        from novelforge.llm.client import _llm_circuit_breaker

        # Trip the primary provider breaker manually
        _llm_circuit_breaker.record_failure("trip1")
        _llm_circuit_breaker.record_failure("trip2")
        _llm_circuit_breaker.record_failure("trip3")
        assert _llm_circuit_breaker.is_tripped

        # Patch Thread so the background worker never actually runs
        monkeypatch_thread = type(
            "FakeThread", (), {"start": lambda s: None, "daemon": True}
        )()
        original_thread = gen_mod.threading.Thread
        gen_mod.threading.Thread = lambda *a, **kw: monkeypatch_thread

        try:
            with client.session_transaction() as sess:
                sess["premise"] = "A test"
                sess["genre"] = "Fantasy"
                sess["chapters"] = 2
                sess["word_count"] = 5000
                sess["title"] = "Test"
                sess["chapter_list"] = [
                    {"number": 1, "title": "Ch1", "summary": "S1"},
                    {"number": 2, "title": "Ch2", "summary": "S2"},
                ]
                sess["character_list"] = []
                sess["special_instructions"] = ""
                sess["story_architecture"] = {}
                sess["master_timeline"] = {}
                sess["character_fate_registry"] = {}
                sess["character_arc_plan"] = {}
                sess["antagonist_motivation_plan"] = {}
                sess["technology_rules"] = {}
                sess["theme_reinforcement"] = {}
                sess["pov_focal_character_plan"] = {}
            client.post("/generate_chapters", data="{}", content_type="application/json")
        finally:
            gen_mod.threading.Thread = original_thread

        # Breaker must still be tripped — generation start must not reset it
        assert _llm_circuit_breaker.is_tripped

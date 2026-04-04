"""
Session persistence tests for NovelForge.

Tests the full save → crash-simulate → load → restore cycle,
including partial generation state, persist_completed_chapters
from background threads, and clear/delete operations.
"""

import json
import stat
import pytest
from pathlib import Path

import novelforge.config as config
from novelforge.progress import progress_manager


def _make_chapters(n: int) -> list[dict]:
    """Create n dummy completed chapter dicts."""
    return [
        {
            "number": i + 1,
            "title": f"Chapter {i + 1}",
            "content": f"Content for chapter {i + 1}.",
            "summary": f"Summary of chapter {i + 1}.",
            "word_count": 3000 + i * 100,
        }
        for i in range(n)
    ]


class TestSaveAndLoadCycle:
    """Test save_session_state → load_session_state round-trip."""

    def test_save_and_load_basic(self, app):
        from novelforge.session.persistence import (
            save_session_state, load_session_state, get_session_id,
        )

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["premise"] = "A hero's journey"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 10
            sess["word_count"] = 80000
            sess["special_events"] = "Dragon attack"
            sess["special_instructions"] = "Dark tone"
            sess["title"] = "The Dragon's Call"
            sess["chapter_list"] = [
                {"number": 1, "title": "Ch1", "summary": "Setup"},
                {"number": 2, "title": "Ch2", "summary": "Rising"},
            ]
            sess["character_list"] = [
                {"name": "Alice", "age": "25", "role": "Protagonist",
                 "background": "Brave", "arc": "Growth"},
            ]
            sess["completed_chapters"] = _make_chapters(2)
            sess["illustrations"] = [{"type": "cover", "image_url": "/illustrations/test.png"}]

            save_session_state()
            session_id = get_session_id()

            # Verify the file was written
            session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
            assert session_file.exists()

            # Load it back
            loaded = load_session_state()
            assert loaded is not None
            assert loaded["title"] == "The Dragon's Call"
            assert loaded["genre"] == "Fantasy"
            assert loaded["chapters"] == 10
            assert loaded["word_count"] == 80000
            assert loaded["special_events"] == "Dragon attack"
            assert len(loaded["chapter_list"]) == 2
            assert len(loaded["character_list"]) == 1
            assert len(loaded["completed_chapters"]) == 2
            assert loaded["completed_chapters"][0]["content"] == "Content for chapter 1."
            assert len(loaded["illustrations"]) == 1

            # Cleanup
            session_file.unlink(missing_ok=True)

    def test_save_includes_progress_data(self, app):
        from novelforge.session.persistence import save_session_state, get_session_id

        token = "test-save-progress"
        progress_manager.create(token, {
            "status": "running",
            "current": 3,
            "total": 10,
            "step": "Chapter 3: polishing",
            "chapters_done": _make_chapters(3),
            "error": None,
        })

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "Progress Test"
            sess["progress_token"] = token
            sess["premise"] = ""
            sess["genre"] = ""
            sess["chapters"] = 10
            sess["word_count"] = 50000

            save_session_state()
            session_id = get_session_id()

            session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
            state = json.loads(session_file.read_text())

            assert "progress_data" in state
            assert state["progress_data"]["status"] == "running"
            assert state["progress_data"]["current"] == 3
            assert len(state["completed_chapters"]) == 3

            # Cleanup
            session_file.unlink(missing_ok=True)
            progress_manager.delete(token)


class TestCrashRecovery:
    """Simulate crash → restart → restore cycle."""

    def test_full_crash_recovery_cycle(self, app):
        """Save state mid-generation, clear everything, restore from file."""
        from novelforge.session.persistence import (
            save_session_state, restore_session_from_state, get_session_id,
        )

        token = "crash-test-token"
        chapters_done = _make_chapters(3)
        progress_manager.create(token, {
            "status": "running",
            "current": 3,
            "total": 10,
            "step": "Chapter 4: drafting",
            "chapters_done": chapters_done,
            "error": None,
        })

        # Phase 1: Save state (simulates normal operation)
        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "Crash Test Novel"
            sess["premise"] = "A test of persistence"
            sess["genre"] = "Sci-Fi"
            sess["chapters"] = 10
            sess["word_count"] = 90000
            sess["special_instructions"] = "Test instructions"
            sess["special_events"] = ""
            sess["progress_token"] = token
            sess["chapter_list"] = [{"number": i+1, "title": f"Ch{i+1}", "summary": f"S{i+1}"} for i in range(10)]
            sess["character_list"] = [{"name": "Bob", "age": "30", "role": "Hero", "background": "B", "arc": "A"}]
            sess["story_architecture"] = {"architecture_type": "three-act"}
            sess["completed_chapters"] = chapters_done

            save_session_state()
            session_id = get_session_id()

        # Phase 2: Simulate crash — clear everything
        progress_manager.clear()

        # Phase 3: Restore from file (simulates restart)
        session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
        assert session_file.exists()
        state = json.loads(session_file.read_text())

        with app.test_request_context():
            import flask
            restore_session_from_state(state)
            sess = flask.session

            # Verify all data restored
            assert sess["title"] == "Crash Test Novel"
            assert sess["premise"] == "A test of persistence"
            assert sess["genre"] == "Sci-Fi"
            assert sess["chapters"] == 10
            assert sess["word_count"] == 90000
            assert sess["progress_token"] == token
            assert len(sess["chapter_list"]) == 10
            assert len(sess["character_list"]) == 1
            assert sess["character_list"][0]["name"] == "Bob"
            assert len(sess["completed_chapters"]) == 3
            assert sess["story_architecture"]["architecture_type"] == "three-act"

            # Progress store should be restored too
            restored = progress_manager.get(token)
            assert restored is not None
            assert restored["current"] == 3
            assert len(restored["chapters_done"]) == 3

        # Cleanup
        session_file.unlink(missing_ok=True)
        progress_manager.delete(token)

    def test_partial_generation_3_of_10(self, app):
        """Simulate crash after 3 of 10 chapters; verify partial state is restorable."""
        from novelforge.session.persistence import save_session_state, get_session_id

        token = "partial-3-of-10"
        progress_manager.create(token, {
            "status": "running",
            "current": 3,
            "total": 10,
            "step": "Chapter 4: drafting",
            "chapters_done": _make_chapters(3),
            "error": None,
            "_live": True,
        })

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "Partial Novel"
            sess["premise"] = "Partially generated"
            sess["genre"] = "Mystery"
            sess["chapters"] = 10
            sess["word_count"] = 70000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["progress_token"] = token
            sess["chapter_list"] = [{"number": i+1, "title": f"Ch{i+1}", "summary": ""} for i in range(10)]
            sess["character_list"] = []

            save_session_state()
            session_id = get_session_id()

        session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
        state = json.loads(session_file.read_text())

        # Verify partial state on disk
        assert state["progress_data"]["status"] == "running"
        assert state["progress_data"]["current"] == 3
        assert state["progress_data"]["total"] == 10
        assert len(state["completed_chapters"]) == 3
        assert state["completed_chapters"][2]["number"] == 3

        # Cleanup
        session_file.unlink(missing_ok=True)
        progress_manager.delete(token)


class TestPersistCompletedChapters:
    """Test persist_completed_chapters called from background threads."""

    def test_incremental_persist(self, app):
        """Simulate background thread persisting chapters one at a time."""
        from novelforge.session.persistence import (
            save_session_state, persist_completed_chapters, get_session_id,
        )

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "Incremental Test"
            sess["premise"] = "Test"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 5
            sess["word_count"] = 5000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["chapter_list"] = []
            sess["character_list"] = []

            save_session_state()
            session_id = get_session_id()

        session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"

        # Simulate background thread persisting chapters incrementally
        for i in range(1, 6):
            chapters_so_far = _make_chapters(i)
            persist_completed_chapters(session_id, chapters_so_far)

            # Verify file updated
            state = json.loads(session_file.read_text())
            assert len(state["completed_chapters"]) == i
            assert state["completed_chapters"][-1]["number"] == i

        # Final state should have all 5 chapters
        state = json.loads(session_file.read_text())
        assert len(state["completed_chapters"]) == 5

        # Cleanup
        session_file.unlink(missing_ok=True)

    def test_persist_to_nonexistent_file(self):
        """Persisting to a missing session file should not crash."""
        from novelforge.session.persistence import persist_completed_chapters
        # Valid UUID format but no corresponding file; should silently return without error
        persist_completed_chapters("00000000-0000-0000-0000-000000000000", _make_chapters(3))


class TestClearSession:
    """Test clear_session_state removes the file."""

    def test_clear_removes_file(self, app):
        from novelforge.session.persistence import (
            save_session_state, clear_session_state, get_session_id,
        )

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "To Be Cleared"
            sess["premise"] = "T"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 3
            sess["word_count"] = 5000

            save_session_state()
            session_id = get_session_id()

            session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
            assert session_file.exists()

            clear_session_state()
            assert not session_file.exists()

    def test_clear_nonexistent_does_not_crash(self, app):
        from novelforge.session.persistence import clear_session_state

        with app.test_request_context():
            import flask
            flask.session["session_id"] = "nonexistent-clear-test"
            # Should not raise
            clear_session_state()


class TestFilePermissions:
    """Test that saved session files have mode 0o600."""

    def test_session_file_mode_is_0o600(self, app):
        """Novel persistence files must be owner-read/write only (0o600)."""
        from novelforge.session.persistence import save_session_state, get_session_id

        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "Permission Test Novel"
            sess["premise"] = "Testing file permissions"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 3
            sess["word_count"] = 5000

            save_session_state()
            session_id = get_session_id()

            session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
            assert session_file.exists()

            file_mode = stat.S_IMODE(session_file.stat().st_mode)
            assert file_mode == 0o600, (
                f"Expected session file mode 0o600, got 0o{file_mode:o}"
            )

            # Cleanup
            session_file.unlink(missing_ok=True)


class TestLoadSessionViaRoute:
    """Test loading a session via the /load_session route and verifying restoration."""

    def test_load_session_restores_to_page(self, app, client):
        """Save a session file, then load it via the route and verify index shows data."""
        from novelforge.session.persistence import save_session_state, get_session_id

        # Create a session with data
        with app.test_request_context():
            import flask
            sess = flask.session
            sess["title"] = "Loadable Novel"
            sess["premise"] = "A loadable story"
            sess["genre"] = "Thriller"
            sess["chapters"] = 5
            sess["word_count"] = 40000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["chapter_list"] = [{"number": 1, "title": "Ch1", "summary": "S1"}]
            sess["character_list"] = [{"name": "Eve", "age": "35", "role": "Agent",
                                       "background": "Spy", "arc": "Redemption"}]
            sess["completed_chapters"] = _make_chapters(2)

            save_session_state()
            session_id = get_session_id()

        # Load via route
        r = client.post(
            "/load_session",
            data=json.dumps({"session_id": session_id}),
            content_type="application/json",
        )
        assert r.status_code == 200
        assert r.get_json()["title"] == "Loadable Novel"

        # Verify session was restored
        with client.session_transaction() as sess:
            assert sess["title"] == "Loadable Novel"
            assert sess["genre"] == "Thriller"
            assert len(sess["completed_chapters"]) == 2

        # Cleanup
        (Path(config.NOVELS_DIR) / f"{session_id}.json").unlink(missing_ok=True)


class TestGenerateOutlinePersistence:
    """Verify that /generate_outline persists session state immediately."""

    def test_session_file_created_after_generate_outline(self, client, mock_llm):
        """A session file must exist on disk after a successful /generate_outline."""
        from novelforge.session.persistence import get_session_id

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

        with client.session_transaction() as sess:
            session_id = sess.get("session_id")

        assert session_id is not None
        session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
        assert session_file.exists(), "Session file must be created after /generate_outline"

        state = json.loads(session_file.read_text())
        assert state["title"] != ""
        assert len(state["chapter_list"]) == 3
        assert len(state["character_list"]) >= 1
        assert state["genre"] == "Fantasy"

        # Cleanup
        session_file.unlink(missing_ok=True)

    def test_generate_outline_state_is_recoverable(self, client, mock_llm):
        """State persisted after /generate_outline can be restored without /approve_outline."""
        from novelforge.session.persistence import restore_session_from_state

        r = client.post(
            "/generate_outline",
            data=json.dumps({
                "premise": "A detective solves an ancient mystery",
                "genre": "Mystery",
                "chapters": 3,
                "word_count": 20000,
            }),
            content_type="application/json",
        )
        assert r.status_code == 200

        with client.session_transaction() as sess:
            session_id = sess.get("session_id")

        session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
        assert session_file.exists()

        state = json.loads(session_file.read_text())

        # Verify all expected planning artifacts are persisted
        assert state["story_architecture"] != {}
        assert state["master_timeline"] != {}
        assert state["character_arc_plan"] != {}
        assert state["technology_rules"] != {}
        assert state["theme_reinforcement"] != {}
        assert state["pov_focal_character_plan"] != {}
        assert isinstance(state["voice_seed"], dict)

        # Simulate restart: restore session from the persisted file
        with client.application.test_request_context():
            import flask
            restore_session_from_state(state)
            sess = flask.session
            assert sess["title"] == state["title"]
            assert sess["genre"] == "Mystery"
            assert len(sess["chapter_list"]) == len(state["chapter_list"])
            assert len(sess["character_list"]) == len(state["character_list"])

        # Cleanup
        session_file.unlink(missing_ok=True)

    def test_generate_outline_persists_voice_seed(self, client, mock_llm):
        """The voice seed selected during outline generation is persisted."""
        r = client.post(
            "/generate_outline",
            data=json.dumps({
                "premise": "A romance across time",
                "genre": "Romance",
                "chapters": 3,
                "word_count": 15000,
            }),
            content_type="application/json",
        )
        assert r.status_code == 200

        with client.session_transaction() as sess:
            session_id = sess.get("session_id")

        session_file = Path(config.NOVELS_DIR) / f"{session_id}.json"
        state = json.loads(session_file.read_text())
        assert isinstance(state["voice_seed"], dict)
        assert state["voice_seed"] != {}

        # Cleanup
        session_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# rebuild_stale_progress tests
# ---------------------------------------------------------------------------

class TestRebuildStaleProgress:
    """Tests for the rebuild_stale_progress helper in persistence."""

    def test_rebuilds_from_completed_chapters(self):
        """rebuild_stale_progress returns a 'done' snapshot and updates the store."""
        from novelforge.session.persistence import rebuild_stale_progress

        token = "rebuild-test-basic"
        chapters = [{"number": i + 1, "title": f"Ch{i + 1}"} for i in range(5)]
        try:
            result = rebuild_stale_progress(chapters, 5, token)

            assert result["status"] == "done"
            assert result["current"] == 5
            assert result["total"] == 5
            assert result["step"] == "Complete"
            assert result["chapters_done"] == chapters
            assert result["error"] is None

            stored = progress_manager.get(token)
            assert stored == result
        finally:
            progress_manager.delete(token)

    def test_uses_total_chapters_param(self):
        """total_chapters parameter is reflected in the rebuilt snapshot."""
        from novelforge.session.persistence import rebuild_stale_progress

        token = "rebuild-test-total"
        chapters = [{"number": 1}]
        try:
            result = rebuild_stale_progress(chapters, 10, token)
            assert result["total"] == 10
        finally:
            progress_manager.delete(token)

    def test_fallback_total_when_zero(self):
        """When total_chapters is 0, total falls back to len(completed_chapters)."""
        from novelforge.session.persistence import rebuild_stale_progress

        token = "rebuild-test-fallback"
        chapters = [{"number": i + 1} for i in range(4)]
        try:
            result = rebuild_stale_progress(chapters, 0, token)
            assert result["total"] == 4
        finally:
            progress_manager.delete(token)


# ---------------------------------------------------------------------------
# restore_session_from_state: stale-progress rebuild tests
# ---------------------------------------------------------------------------

class TestRestoreSessionRebuildsBrokenProgress:
    """restore_session_from_state must rebuild stale progress, not leave it as-is."""

    def test_restore_rebuilds_stale_running_progress(self, app):
        """Stale 'running' progress (no _live flag) is rebuilt to 'done' on restore."""
        from novelforge.session.persistence import restore_session_from_state

        token = "restore-stale-running"
        chapters = _make_chapters(5)
        state = {
            "title": "Stale Running Novel",
            "premise": "Test",
            "genre": "Fantasy",
            "chapters": 5,
            "word_count": 5000,
            "progress_token": token,
            "completed_chapters": chapters,
            "progress_data": {
                "status": "running",
                "current": 3,
                "total": 5,
                "step": "Chapter 3: drafting",
                "chapters_done": chapters[:3],
                # No "_live" flag → stale snapshot
            },
        }

        with app.test_request_context():
            progress_manager.delete(token)
            restore_session_from_state(state)
            rebuilt = progress_manager.get(token)

        try:
            assert rebuilt is not None
            assert rebuilt["status"] == "done"
            assert rebuilt["current"] == 5
            assert rebuilt["step"] == "Complete"
            assert len(rebuilt["chapters_done"]) == 5
        finally:
            progress_manager.delete(token)

    def test_restore_rebuilds_when_no_progress_data(self, app):
        """Missing progress_data with completed chapters triggers a rebuild."""
        from novelforge.session.persistence import restore_session_from_state

        token = "restore-no-pd"
        chapters = _make_chapters(3)
        state = {
            "title": "Missing PD Novel",
            "premise": "Test",
            "genre": "Mystery",
            "chapters": 3,
            "word_count": 3000,
            "progress_token": token,
            "completed_chapters": chapters,
            # No "progress_data" key
        }

        with app.test_request_context():
            progress_manager.delete(token)
            assert progress_manager.get(token) is None, "token must be absent before restore"
            restore_session_from_state(state)
            rebuilt = progress_manager.get(token)

        try:
            assert rebuilt is not None
            assert rebuilt["status"] == "done"
            assert len(rebuilt["chapters_done"]) == 3
        finally:
            progress_manager.delete(token)

    def test_restore_preserves_valid_progress_data(self, app):
        """Valid (done) progress_data is stored as-is without a rebuild."""
        from novelforge.session.persistence import restore_session_from_state

        token = "restore-valid-pd"
        chapters = _make_chapters(5)
        done_pd = {
            "status": "done",
            "current": 5,
            "total": 5,
            "step": "Complete",
            "chapters_done": chapters,
            "error": None,
            "extra_key": "preserved",
        }
        state = {
            "title": "Done Novel",
            "premise": "Test",
            "genre": "Fantasy",
            "chapters": 5,
            "word_count": 5000,
            "progress_token": token,
            "completed_chapters": chapters,
            "progress_data": done_pd,
        }

        with app.test_request_context():
            progress_manager.delete(token)
            restore_session_from_state(state)
            stored = progress_manager.get(token)

        try:
            assert stored is not None
            assert stored.get("extra_key") == "preserved"
            assert stored["status"] == "done"
        finally:
            progress_manager.delete(token)

    def test_restore_preserves_live_running_progress(self, app):
        """Running progress with _live=True is stored as-is (active generation)."""
        from novelforge.session.persistence import restore_session_from_state

        token = "restore-live-running"
        chapters = _make_chapters(3)
        live_pd = {
            "status": "running",
            "current": 3,
            "total": 10,
            "step": "Chapter 3: drafting",
            "chapters_done": chapters,
            "error": None,
            "_live": True,
        }
        state = {
            "title": "Live Running Novel",
            "premise": "Test",
            "genre": "Fantasy",
            "chapters": 10,
            "word_count": 10000,
            "progress_token": token,
            "completed_chapters": chapters,
            "progress_data": live_pd,
        }

        with app.test_request_context():
            progress_manager.delete(token)
            restore_session_from_state(state)
            stored = progress_manager.get(token)

        try:
            assert stored is not None
            assert stored["status"] == "running"
            assert stored.get("_live") is True
        finally:
            progress_manager.delete(token)


_VALID_UUID = "12345678-1234-1234-1234-123456789abc"


class TestListSessionSummaries:
    """Tests for list_session_summaries() in the persistence layer."""

    def _write_session(self, sessions_dir: Path, session_id: str, state: dict) -> Path:
        """Write a session JSON file directly to disk."""
        p = sessions_dir / f"{session_id}.json"
        p.write_text(json.dumps(state), encoding="utf-8")
        return p

    def test_returns_sessions_with_title(self, app):
        from novelforge.session.persistence import list_session_summaries

        sid = _VALID_UUID
        sessions_dir = Path(config.NOVELS_DIR)
        self._write_session(sessions_dir, sid, {
            "session_id": sid,
            "title": "My Novel",
            "premise": "A test premise",
            "genre": "Fantasy",
            "chapters": 5,
            "word_count": 5000,
        })

        with app.test_request_context():
            result = list_session_summaries()

        assert any(s["session_id"] == sid and s["title"] == "My Novel" for s in result)

    def test_skips_untitled_sessions(self, app):
        from novelforge.session.persistence import list_session_summaries

        sid = _VALID_UUID
        sessions_dir = Path(config.NOVELS_DIR)
        self._write_session(sessions_dir, sid, {
            "session_id": sid,
            "title": "",
            "premise": "A test",
            "genre": "Fantasy",
        })

        with app.test_request_context():
            result = list_session_summaries()

        assert not any(s["session_id"] == sid for s in result)

    def test_skips_corrupt_json_file(self, app):
        from novelforge.session.persistence import list_session_summaries

        sessions_dir = Path(config.NOVELS_DIR)
        corrupt = sessions_dir / f"{_VALID_UUID}.json"
        corrupt.write_text("{ this is not valid JSON !!!", encoding="utf-8")

        with app.test_request_context():
            result = list_session_summaries()

        # Corrupt file must not appear and must not raise
        assert not any(s["session_id"] == _VALID_UUID for s in result)

    def test_skips_progress_json_files(self, app):
        from novelforge.session.persistence import list_session_summaries

        sessions_dir = Path(config.NOVELS_DIR)
        (sessions_dir / f"{_VALID_UUID}_progress.json").write_text(
            json.dumps({"title": "Should Be Ignored"}), encoding="utf-8"
        )

        with app.test_request_context():
            result = list_session_summaries()

        assert not any(s.get("session_id", "").endswith("_progress") for s in result)

    def test_partial_session_included_when_title_present(self, app):
        """Legacy/partial sessions missing optional fields are included if they have a title."""
        from novelforge.session.persistence import list_session_summaries

        sid = _VALID_UUID
        sessions_dir = Path(config.NOVELS_DIR)
        # Only title and session_id; all other fields missing
        self._write_session(sessions_dir, sid, {
            "session_id": sid,
            "title": "Partial Novel",
        })

        with app.test_request_context():
            result = list_session_summaries()

        assert any(s["session_id"] == sid and s["title"] == "Partial Novel" for s in result)

    def test_result_sorted_by_title(self, app):
        from novelforge.session.persistence import list_session_summaries

        sessions_dir = Path(config.NOVELS_DIR)
        ids = [
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "cccccccc-cccc-cccc-cccc-cccccccccccc",
        ]
        titles = ["Zebra Story", "Alpha Story", "Middle Story"]
        for sid, title in zip(ids, titles):
            self._write_session(sessions_dir, sid, {
                "session_id": sid,
                "title": title,
                "premise": "x",
                "genre": "Fantasy",
            })

        with app.test_request_context():
            result = list_session_summaries()

        returned_titles = [s["title"] for s in result]
        assert returned_titles == sorted(returned_titles, key=str.lower)

    def test_valid_and_corrupt_files_coexist(self, app):
        """One corrupt file must not prevent valid sessions from being listed."""
        from novelforge.session.persistence import list_session_summaries

        sessions_dir = Path(config.NOVELS_DIR)
        good_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        bad_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

        self._write_session(sessions_dir, good_id, {
            "session_id": good_id,
            "title": "Good Novel",
            "premise": "fine",
            "genre": "Fantasy",
        })
        (sessions_dir / f"{bad_id}.json").write_text("not json at all", encoding="utf-8")

        with app.test_request_context():
            result = list_session_summaries()

        titles = [s["title"] for s in result]
        assert "Good Novel" in titles


class TestLoadSessionById:
    """Tests for load_session_by_id() in the persistence layer."""

    def _write_session(self, sessions_dir: Path, session_id: str, state: dict) -> Path:
        p = sessions_dir / f"{session_id}.json"
        p.write_text(json.dumps(state), encoding="utf-8")
        return p

    def test_returns_validated_state_for_valid_session(self, app):
        from novelforge.session.persistence import load_session_by_id

        sid = _VALID_UUID
        sessions_dir = Path(config.NOVELS_DIR)
        self._write_session(sessions_dir, sid, {
            "session_id": sid,
            "title": "Loadable",
            "premise": "Test",
            "genre": "Sci-Fi",
            "chapters": 3,
            "word_count": 3000,
        })

        with app.test_request_context():
            state = load_session_by_id(sid)

        assert state is not None
        assert state["title"] == "Loadable"
        assert state["genre"] == "Sci-Fi"
        # validate_session_state fills in missing fields with defaults
        assert "chapter_list" in state
        assert "character_list" in state

    def test_returns_none_for_missing_file(self, app):
        from novelforge.session.persistence import load_session_by_id

        with app.test_request_context():
            result = load_session_by_id(_VALID_UUID)

        assert result is None

    def test_raises_value_error_for_invalid_id(self, app):
        from novelforge.session.persistence import load_session_by_id

        with app.test_request_context():
            with pytest.raises(ValueError):
                load_session_by_id("not-a-uuid")

    def test_raises_on_corrupt_file(self, app):
        from novelforge.session.persistence import load_session_by_id
        import json as _json

        sid = _VALID_UUID
        (Path(config.NOVELS_DIR) / f"{sid}.json").write_text(
            "{ corrupt JSON !!!", encoding="utf-8"
        )

        with app.test_request_context():
            with pytest.raises(_json.JSONDecodeError):
                load_session_by_id(sid)

    def test_partial_session_gets_defaults(self, app):
        """A session file with only a title returns a fully normalised state dict."""
        from novelforge.session.persistence import load_session_by_id

        sid = _VALID_UUID
        sessions_dir = Path(config.NOVELS_DIR)
        self._write_session(sessions_dir, sid, {
            "session_id": sid,
            "title": "Partial",
        })

        with app.test_request_context():
            state = load_session_by_id(sid)

        assert state is not None
        assert state["title"] == "Partial"
        assert state["chapters"] == 0          # default
        assert state["chapter_list"] == []     # default
        assert state["voice_seed"] == {}       # default

    def test_legacy_numeric_string_coerced(self, app):
        """Legacy files that store chapters as a string are coerced to int."""
        from novelforge.session.persistence import load_session_by_id

        sid = _VALID_UUID
        sessions_dir = Path(config.NOVELS_DIR)
        self._write_session(sessions_dir, sid, {
            "session_id": sid,
            "title": "Legacy Novel",
            "chapters": "7",   # stored as string in old format
            "word_count": "50000",
        })

        with app.test_request_context():
            state = load_session_by_id(sid)

        assert state is not None
        assert state["chapters"] == 7
        assert state["word_count"] == 50000


class TestRouteUsesPersistenceLayer:
    """Verify that route handlers delegate to persistence helpers, not raw JSON."""

    def test_list_sessions_route_uses_list_session_summaries(self, app, mocker):
        """list_sessions() route must call list_session_summaries(), not json.loads."""
        from novelforge.routes import sessions as sessions_module

        mock_list = mocker.patch.object(
            sessions_module,
            "list_session_summaries",
            return_value=[{"session_id": "abc", "title": "Mocked"}],
        )

        with app.test_client() as c:
            r = c.get("/list_sessions")

        assert r.status_code == 200
        mock_list.assert_called_once()
        data = r.get_json()
        assert data["sessions"] == [{"session_id": "abc", "title": "Mocked"}]

    def test_load_session_route_uses_load_session_by_id(self, app, mocker):
        """load_session() route must call load_session_by_id(), not json.loads."""
        from novelforge.routes import sessions as sessions_module

        fake_state = {
            "session_id": _VALID_UUID,
            "title": "Mocked Session",
            "premise": "",
            "genre": "Fantasy",
            "chapters": 0,
            "word_count": 0,
            "special_events": "",
            "special_instructions": "",
            "chapter_list": [],
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
            "progress_token": "",
            "completed_chapters": [],
            "illustrations": [],
            "voice_seed": {},
        }

        mock_load = mocker.patch.object(
            sessions_module,
            "load_session_by_id",
            return_value=fake_state,
        )
        mocker.patch.object(sessions_module, "restore_session_from_state")

        with app.test_client() as c:
            r = c.post(
                "/load_session",
                data=json.dumps({"session_id": _VALID_UUID}),
                content_type="application/json",
            )

        assert r.status_code == 200
        mock_load.assert_called_once_with(_VALID_UUID)

    def test_load_session_returns_404_when_load_returns_none(self, app, mocker):
        """load_session() must return 404 when load_session_by_id() returns None."""
        from novelforge.routes import sessions as sessions_module

        mocker.patch.object(sessions_module, "load_session_by_id", return_value=None)

        with app.test_client() as c:
            r = c.post(
                "/load_session",
                data=json.dumps({"session_id": _VALID_UUID}),
                content_type="application/json",
            )

        assert r.status_code == 404

    def test_load_session_returns_500_on_corrupt_file(self, app, mocker):
        """load_session() must return 500 when load_session_by_id() raises."""
        import json as _json
        from novelforge.routes import sessions as sessions_module

        mocker.patch.object(
            sessions_module,
            "load_session_by_id",
            side_effect=_json.JSONDecodeError("bad", "doc", 0),
        )

        with app.test_client() as c:
            r = c.post(
                "/load_session",
                data=json.dumps({"session_id": _VALID_UUID}),
                content_type="application/json",
            )

        assert r.status_code == 500
        assert "error" in r.get_json()


class TestResolveSessionPath:
    """Test the resolve_session_path validation helper."""

    def test_valid_uuid_returns_correct_filename(self):
        from novelforge.session.persistence import resolve_session_path

        path = resolve_session_path("12345678-1234-1234-1234-123456789abc")
        assert path.name == "12345678-1234-1234-1234-123456789abc.json"

    def test_valid_uuid_uppercase_accepted(self):
        from novelforge.session.persistence import resolve_session_path

        path = resolve_session_path("12345678-1234-1234-1234-123456789ABC")
        assert path.name == "12345678-1234-1234-1234-123456789ABC.json"

    @pytest.mark.parametrize("bad_id", [
        "",                                          # empty string
        "   ",                                       # whitespace only
        "nonexistent-uuid",                          # plausible but not a UUID
        "foo",                                       # no dashes at all
        "../etc/passwd",                             # path traversal (single)
        "../../etc/passwd",                          # path traversal (nested)
        "12345678-1234-1234-1234-123456789abc.json", # with .json suffix
        "12345678/1234/1234/1234/123456789abc",      # forward slashes
        "12345678\\1234-1234-1234-123456789abc",     # backslash
        "12345678-1234-1234-1234-12345678901",       # wrong length (too short in last group)
        "12345678-1234-1234-1234-12345678901z",      # non-hex character
        " 12345678-1234-1234-1234-123456789abc",     # leading space
        "12345678-1234-1234-1234-123456789abc ",     # trailing space
        "\x0012345678-1234-1234-1234-123456789abc",  # null byte prefix
    ])
    def test_invalid_ids_raise_value_error(self, bad_id: str) -> None:
        from novelforge.session.persistence import resolve_session_path

        with pytest.raises(ValueError):
            resolve_session_path(bad_id)


class TestLoadSessionRouteValidation:
    """Test that /load_session rejects invalid session IDs with HTTP 400."""

    @pytest.mark.parametrize("bad_id", [
        "../etc/passwd",
        "../../secret",
        "foo",
        "nonexistent-uuid",
        "12345678-1234-1234-1234-123456789abc.json",
        "12345678/1234/1234/1234/123456789abc",
        " ",
        "\x00bad",
    ])
    def test_invalid_session_id_returns_400(self, client, bad_id: str) -> None:
        r = client.post(
            "/load_session",
            data=json.dumps({"session_id": bad_id}),
            content_type="application/json",
        )
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_missing_session_id_returns_400(self, client) -> None:
        r = client.post(
            "/load_session",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_valid_uuid_not_on_disk_returns_404(self, client) -> None:
        r = client.post(
            "/load_session",
            data=json.dumps({"session_id": "12345678-1234-1234-1234-123456789abc"}),
            content_type="application/json",
        )
        assert r.status_code == 404

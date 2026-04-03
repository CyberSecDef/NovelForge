"""
Session persistence tests for NovelForge.

Tests the full save → crash-simulate → load → restore cycle,
including partial generation state, _persist_completed_chapters
from background threads, and clear/delete operations.
"""

import json
import stat
import pytest
from pathlib import Path

import novelforge.config as config
from novelforge.progress import _progress_store, _progress_lock


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
        with _progress_lock:
            _progress_store[token] = {
                "status": "running",
                "current": 3,
                "total": 10,
                "step": "Chapter 3: polishing",
                "chapters_done": _make_chapters(3),
            }

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
            with _progress_lock:
                _progress_store.pop(token, None)


class TestCrashRecovery:
    """Simulate crash → restart → restore cycle."""

    def test_full_crash_recovery_cycle(self, app):
        """Save state mid-generation, clear everything, restore from file."""
        from novelforge.session.persistence import (
            save_session_state, restore_session_from_state, get_session_id,
        )

        token = "crash-test-token"
        chapters_done = _make_chapters(3)
        with _progress_lock:
            _progress_store[token] = {
                "status": "running",
                "current": 3,
                "total": 10,
                "step": "Chapter 4: drafting",
                "chapters_done": chapters_done,
            }

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
        with _progress_lock:
            _progress_store.clear()

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
            with _progress_lock:
                assert token in _progress_store
                assert _progress_store[token]["current"] == 3
                assert len(_progress_store[token]["chapters_done"]) == 3

        # Cleanup
        session_file.unlink(missing_ok=True)
        with _progress_lock:
            _progress_store.pop(token, None)

    def test_partial_generation_3_of_10(self, app):
        """Simulate crash after 3 of 10 chapters; verify partial state is restorable."""
        from novelforge.session.persistence import save_session_state, get_session_id

        token = "partial-3-of-10"
        with _progress_lock:
            _progress_store[token] = {
                "status": "running",
                "current": 3,
                "total": 10,
                "step": "Chapter 4: drafting",
                "chapters_done": _make_chapters(3),
                "error": None,
                "_live": True,
            }

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
        with _progress_lock:
            _progress_store.pop(token, None)


class TestPersistCompletedChapters:
    """Test _persist_completed_chapters called from background threads."""

    def test_incremental_persist(self, app):
        """Simulate background thread persisting chapters one at a time."""
        from novelforge.session.persistence import (
            save_session_state, _persist_completed_chapters, get_session_id,
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
            _persist_completed_chapters(session_id, chapters_so_far)

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
        from novelforge.session.persistence import _persist_completed_chapters
        # Should silently return without error
        _persist_completed_chapters("nonexistent-uuid", _make_chapters(3))


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

"""
Tests for progress-store cleanup during session lifecycle events.

Covers:
- delete_session() removes the active session's progress entry
- new_session() removes the active session's progress entry
- restore_session_from_state() removes the old session's stale progress entry
  when loading a different session (token mismatch)
- restore_session_from_state() leaves the entry intact when the token is the
  same in both old and new sessions (crash-recovery reload)
"""

import json
import uuid
from pathlib import Path

import pytest

import novelforge.config as config
from novelforge.progress import progress_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_progress(**overrides) -> dict:
    base = {
        "status": "running",
        "current": 2,
        "total": 10,
        "step": "Chapter 2: complete",
        "chapters_done": [{"number": 1}, {"number": 2}],
        "error": None,
    }
    base.update(overrides)
    return base


def _make_session_file(novels_dir: str, session_id: str, token: str) -> Path:
    """Write a minimal session JSON file and return its path."""
    state = {
        "session_id": session_id,
        "premise": "Old premise",
        "genre": "Fantasy",
        "chapters": 10,
        "word_count": 80000,
        "special_events": "",
        "special_instructions": "",
        "title": "Old Session Novel",
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
        "progress_token": token,
        "completed_chapters": [],
        "illustrations": [],
        "voice_seed": {},
    }
    path = Path(novels_dir) / f"{session_id}.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fixture: clean progress store per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_store():
    """Wipe the progress store before and after every test."""
    progress_manager.clear()
    yield
    progress_manager.clear()


# ---------------------------------------------------------------------------
# delete_session() — progress cleanup
# ---------------------------------------------------------------------------

class TestDeleteSessionCleansProgress:
    """delete_session() must remove the active session's progress entry."""

    def test_delete_session_removes_progress_entry(self, client):
        """Progress entry for the current token is gone after DELETE."""
        token = "delete-test-token"
        progress_manager.create(token, _base_progress())

        with client.session_transaction() as sess:
            sess["progress_token"] = token

        r = client.post("/delete_session")
        assert r.status_code == 200
        assert r.get_json()["status"] == "success"

        assert progress_manager.get(token) is None, (
            "Progress entry should be removed on session delete"
        )

    def test_delete_session_no_token_does_not_crash(self, client):
        """delete_session() is safe when no progress_token is set."""
        # Ensure the session has no progress_token
        with client.session_transaction() as sess:
            sess.pop("progress_token", None)

        r = client.post("/delete_session")
        assert r.status_code == 200

    def test_delete_session_unknown_token_is_noop(self, client):
        """delete_session() with a token that has no in-memory entry is a no-op."""
        with client.session_transaction() as sess:
            sess["progress_token"] = "ghost-token-that-does-not-exist"

        r = client.post("/delete_session")
        assert r.status_code == 200

    def test_delete_session_clears_flask_session(self, client):
        """Flask session is cleared even when progress cleanup runs."""
        token = "delete-flask-clear-token"
        progress_manager.create(token, _base_progress())

        with client.session_transaction() as sess:
            sess["progress_token"] = token
            sess["title"] = "My Novel"

        client.post("/delete_session")

        with client.session_transaction() as sess:
            assert "title" not in sess
            assert "progress_token" not in sess

    def test_delete_session_only_removes_own_token(self, client):
        """Unrelated progress entries survive a session delete."""
        own_token = "delete-own-token"
        other_token = "delete-other-token"
        progress_manager.create(own_token, _base_progress())
        progress_manager.create(other_token, _base_progress())

        with client.session_transaction() as sess:
            sess["progress_token"] = own_token

        client.post("/delete_session")

        assert progress_manager.get(own_token) is None
        assert progress_manager.get(other_token) is not None


# ---------------------------------------------------------------------------
# new_session() — progress cleanup
# ---------------------------------------------------------------------------

class TestNewSessionCleansProgress:
    """new_session() must remove the active session's progress entry."""

    def test_new_session_removes_progress_entry(self, client):
        """Progress entry for the current token is gone after new_session."""
        token = "new-session-token"
        progress_manager.create(token, _base_progress())

        with client.session_transaction() as sess:
            sess["progress_token"] = token

        r = client.post("/new_session")
        assert r.status_code == 200
        assert r.get_json()["status"] == "success"

        assert progress_manager.get(token) is None, (
            "Progress entry should be removed when a new session is started"
        )

    def test_new_session_no_token_does_not_crash(self, client):
        """new_session() is safe when no progress_token is set."""
        with client.session_transaction() as sess:
            sess.pop("progress_token", None)

        r = client.post("/new_session")
        assert r.status_code == 200

    def test_new_session_clears_flask_session(self, client):
        """Flask session is cleared even when progress cleanup runs."""
        token = "new-session-flask-clear"
        progress_manager.create(token, _base_progress())

        with client.session_transaction() as sess:
            sess["progress_token"] = token
            sess["title"] = "Old Novel"

        client.post("/new_session")

        with client.session_transaction() as sess:
            assert "title" not in sess
            assert "progress_token" not in sess

    def test_new_session_done_progress_also_removed(self, client):
        """A completed (done) progress entry is removed on new_session."""
        token = "new-session-done-token"
        progress_manager.create(token, _base_progress(status="done"))

        with client.session_transaction() as sess:
            sess["progress_token"] = token

        client.post("/new_session")

        assert progress_manager.get(token) is None

    def test_new_session_only_removes_own_token(self, client):
        """Unrelated progress entries survive a new_session call."""
        own_token = "new-session-own"
        other_token = "new-session-other"
        progress_manager.create(own_token, _base_progress())
        progress_manager.create(other_token, _base_progress())

        with client.session_transaction() as sess:
            sess["progress_token"] = own_token

        client.post("/new_session")

        assert progress_manager.get(own_token) is None
        assert progress_manager.get(other_token) is not None


# ---------------------------------------------------------------------------
# restore_session_from_state() — stale token cleanup on session switch
# ---------------------------------------------------------------------------

class TestRestoreSessionCleansOldToken:
    """Loading a different session via restore_session_from_state should evict
    the old session's stale progress entry."""

    def test_load_over_existing_removes_old_token(self, app):
        """When an old token is active and a new session (different token) is
        loaded, the old token's progress entry must be deleted."""
        from novelforge.session.persistence import restore_session_from_state

        old_token = "old-session-token"
        new_token = "new-session-token"

        progress_manager.create(old_token, _base_progress())

        new_state = {
            "session_id": str(uuid.uuid4()),
            "premise": "New premise",
            "genre": "Mystery",
            "chapters": 5,
            "word_count": 50000,
            "special_events": "",
            "special_instructions": "",
            "title": "New Novel",
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
            "progress_token": new_token,
            "completed_chapters": [],
            "illustrations": [],
            "voice_seed": {},
        }

        with app.test_request_context():
            import flask
            flask.session["progress_token"] = old_token

            restore_session_from_state(new_state)

        assert progress_manager.get(old_token) is None, (
            "Old session's progress entry must be removed when a different session is loaded"
        )

    def test_load_same_token_preserves_progress(self, app):
        """Crash-recovery reload (same token) must not delete the progress entry."""
        from novelforge.session.persistence import restore_session_from_state

        token = "same-token-recovery"
        progress_manager.create(token, _base_progress())

        state = {
            "session_id": str(uuid.uuid4()),
            "premise": "Same session",
            "genre": "Fantasy",
            "chapters": 10,
            "word_count": 80000,
            "special_events": "",
            "special_instructions": "",
            "title": "Same Session Novel",
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
            "progress_token": token,
            "completed_chapters": [],
            "illustrations": [],
            "voice_seed": {},
        }

        with app.test_request_context():
            import flask
            flask.session["progress_token"] = token

            restore_session_from_state(state)

        # Entry must still exist (restored, not deleted) since it's the same token
        assert progress_manager.get(token) is not None, (
            "Same-token crash-recovery must not remove the progress entry"
        )

    def test_load_session_with_no_old_token_does_not_crash(self, app):
        """restore_session_from_state() is safe when the old session has no token."""
        from novelforge.session.persistence import restore_session_from_state

        new_token = "load-no-old-token"
        state = {
            "session_id": str(uuid.uuid4()),
            "premise": "Fresh load",
            "genre": "Sci-Fi",
            "chapters": 3,
            "word_count": 30000,
            "special_events": "",
            "special_instructions": "",
            "title": "Fresh Novel",
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
            "progress_token": new_token,
            "completed_chapters": [],
            "illustrations": [],
            "voice_seed": {},
        }

        with app.test_request_context():
            # No progress_token set in session — must not crash
            restore_session_from_state(state)

    def test_load_session_via_route_cleans_old_token(self, app, client):
        """Using the /load_session route removes any prior session's progress."""
        old_token = "route-load-old-token"
        progress_manager.create(old_token, _base_progress())

        # Seed the new session file
        new_session_id = str(uuid.uuid4())
        new_token = "route-load-new-token"
        _make_session_file(config.NOVELS_DIR, new_session_id, new_token)

        # Set the old token in the client session before loading
        with client.session_transaction() as sess:
            sess["progress_token"] = old_token

        r = client.post(
            "/load_session",
            data=json.dumps({"session_id": new_session_id}),
            content_type="application/json",
        )
        assert r.status_code == 200

        assert progress_manager.get(old_token) is None, (
            "Old session's progress entry must be removed by /load_session"
        )

        # Cleanup
        (Path(config.NOVELS_DIR) / f"{new_session_id}.json").unlink(missing_ok=True)

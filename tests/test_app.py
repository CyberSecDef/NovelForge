"""
Tests for NovelForge – validate_outline_input boundary conditions and Flask routes.
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

    def test_premise_exactly_1200(self):
        ok, err = self._call(
            {"premise": "x" * 1200, "genre": "Fantasy", "chapters": 5, "word_count": 50000}
        )
        assert ok, f"Exactly 1200 chars should be valid, got: {err}"

    def test_premise_1201_chars(self):
        ok, err = self._call(
            {"premise": "x" * 1201, "genre": "Fantasy", "chapters": 5, "word_count": 50000}
        )
        assert not ok
        assert "1200" in err

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

    def test_download_path_traversal(self, client):
        """Directory traversal attempt should 404 (file won't exist)."""
        r = client.get("/download/../../etc/passwd")
        assert r.status_code == 404

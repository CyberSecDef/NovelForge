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

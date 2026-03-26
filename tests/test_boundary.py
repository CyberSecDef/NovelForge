"""
Boundary condition tests for NovelForge.

Tests edge cases: Unicode/emoji in premises, special characters in titles,
upper-bound validation limits, empty character lists, contenteditable field
limits, and session schema coercion.
"""

import json
import pytest

from novelforge.validation import validate_outline_input, ALLOWED_GENRES


# ---------------------------------------------------------------------------
# Input validation boundaries
# ---------------------------------------------------------------------------

class TestValidationBoundaries:
    """Edge cases for validate_outline_input."""

    def test_word_count_exactly_1000(self):
        ok, err = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 3, "word_count": 1000}
        )
        assert ok, f"Exactly 1000 should pass: {err}"

    def test_word_count_999(self):
        ok, _ = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 3, "word_count": 999}
        )
        assert not ok

    def test_chapters_exactly_3(self):
        ok, err = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 3, "word_count": 5000}
        )
        assert ok, f"Exactly 3 should pass: {err}"

    def test_chapters_exactly_2(self):
        ok, _ = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 2, "word_count": 5000}
        )
        assert not ok

    def test_chapters_at_max(self):
        import novelforge.config as config
        ok, err = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy",
             "chapters": config.MAX_CHAPTERS, "word_count": 5000}
        )
        assert ok, f"Exactly {config.MAX_CHAPTERS} should pass: {err}"

    def test_chapters_over_max(self):
        import novelforge.config as config
        ok, err = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy",
             "chapters": config.MAX_CHAPTERS + 1, "word_count": 5000}
        )
        assert not ok
        assert str(config.MAX_CHAPTERS) in err

    def test_word_count_at_max(self):
        import novelforge.config as config
        ok, err = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy",
             "chapters": 5, "word_count": config.MAX_WORD_COUNT}
        )
        assert ok, f"Exactly {config.MAX_WORD_COUNT} should pass: {err}"

    def test_word_count_over_max(self):
        import novelforge.config as config
        ok, err = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy",
             "chapters": 5, "word_count": config.MAX_WORD_COUNT + 1}
        )
        assert not ok

    def test_premise_exactly_2000_chars(self):
        ok, _ = validate_outline_input(
            {"premise": "a" * 2000, "genre": "Fantasy", "chapters": 3, "word_count": 5000}
        )
        assert ok

    def test_premise_2001_chars(self):
        ok, _ = validate_outline_input(
            {"premise": "a" * 2001, "genre": "Fantasy", "chapters": 3, "word_count": 5000}
        )
        assert not ok

    def test_special_events_at_limit(self):
        ok, _ = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 3,
             "word_count": 5000, "special_events": "x" * 5000}
        )
        assert ok

    def test_special_events_over_limit(self):
        ok, err = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 3,
             "word_count": 5000, "special_events": "x" * 5001}
        )
        assert not ok
        assert "5,000" in err

    def test_special_instructions_over_limit(self):
        ok, err = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 3,
             "word_count": 5000, "special_instructions": "x" * 5001}
        )
        assert not ok

    def test_chapters_as_float(self):
        ok, _ = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 5.7, "word_count": 5000}
        )
        # int(5.7) = 5, which is valid
        assert ok

    def test_chapters_zero(self):
        ok, _ = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 0, "word_count": 5000}
        )
        assert not ok

    def test_chapters_negative(self):
        ok, _ = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": -1, "word_count": 5000}
        )
        assert not ok

    def test_word_count_zero(self):
        ok, _ = validate_outline_input(
            {"premise": "A story", "genre": "Fantasy", "chapters": 5, "word_count": 0}
        )
        assert not ok


# ---------------------------------------------------------------------------
# Unicode and special characters
# ---------------------------------------------------------------------------

class TestUnicodeHandling:
    """Verify Unicode, emoji, and special characters don't break the pipeline."""

    def test_unicode_premise_validates(self):
        ok, err = validate_outline_input(
            {"premise": "Ein Held entdeckt eine verborgene Welt der Magie und Gefahr",
             "genre": "Fantasy", "chapters": 5, "word_count": 50000}
        )
        assert ok, f"Unicode premise should be valid: {err}"

    def test_emoji_premise_validates(self):
        ok, err = validate_outline_input(
            {"premise": "A hero discovers a hidden world of magic and danger ✨🐉🗡️",
             "genre": "Fantasy", "chapters": 5, "word_count": 50000}
        )
        assert ok, f"Emoji premise should be valid: {err}"

    def test_cjk_premise_validates(self):
        ok, err = validate_outline_input(
            {"premise": "英雄が魔法と危険に満ちた隠された世界を発見する物語",
             "genre": "Fantasy", "chapters": 5, "word_count": 50000}
        )
        assert ok, f"CJK premise should be valid: {err}"

    def test_special_chars_in_chapter_title(self, client, mock_llm):
        """Chapter titles with special characters should not break approve_outline."""
        with client.session_transaction() as sess:
            sess["premise"] = "A story"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 1
            sess["word_count"] = 5000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["title"] = "Test"
            sess["chapter_list"] = [{"number": 1, "title": "Ch1", "summary": "S"}]
            sess["character_list"] = []

        r = client.post(
            "/approve_outline",
            data=json.dumps({
                "title": "The Hero's Journey — Part I: «Réveil»",
                "chapters": [
                    {"number": 1, "title": "L'éveil du héros — début", "summary": "Le début de l'aventure"},
                ],
                "characters": [],
            }),
            content_type="application/json",
        )
        assert r.status_code == 200

    def test_emoji_in_chapter_title(self, client, mock_llm):
        with client.session_transaction() as sess:
            sess["premise"] = "A story"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 1
            sess["word_count"] = 5000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["title"] = "Test"
            sess["chapter_list"] = [{"number": 1, "title": "Ch1", "summary": "S"}]
            sess["character_list"] = []

        r = client.post(
            "/approve_outline",
            data=json.dumps({
                "title": "Dragon Quest 🐉",
                "chapters": [
                    {"number": 1, "title": "The Awakening 🌅", "summary": "Dawn breaks ☀️"},
                ],
                "characters": [],
            }),
            content_type="application/json",
        )
        assert r.status_code == 200

    def test_unicode_character_names(self, client, mock_llm):
        with client.session_transaction() as sess:
            sess["premise"] = "A story"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 1
            sess["word_count"] = 5000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["title"] = "Test"
            sess["chapter_list"] = [{"number": 1, "title": "Ch1", "summary": "S"}]
            sess["character_list"] = []

        r = client.post(
            "/approve_outline",
            data=json.dumps({
                "title": "Test",
                "chapters": [{"number": 1, "title": "Ch1", "summary": "S"}],
                "characters": [
                    {"name": "Ñoño García", "age": "30", "role": "Protagonist",
                     "background": "A músico from México", "arc": "Finds inner strength"},
                    {"name": "田中太郎", "age": "45", "role": "Mentor",
                     "background": "Ancient warrior", "arc": "Passes on wisdom"},
                ],
            }),
            content_type="application/json",
        )
        assert r.status_code == 200
        with client.session_transaction() as sess:
            names = [c["name"] for c in sess["character_list"]]
            assert any("Ñ" in n or "ñ" in n.lower() for n in names)


# ---------------------------------------------------------------------------
# Empty and minimal data
# ---------------------------------------------------------------------------

class TestEmptyAndMinimalData:
    """Verify graceful handling of empty or minimal inputs."""

    def test_empty_character_list_approves(self, client, mock_llm):
        with client.session_transaction() as sess:
            sess["premise"] = "A story"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 1
            sess["word_count"] = 5000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["title"] = "Test"
            sess["chapter_list"] = [{"number": 1, "title": "Ch1", "summary": "S"}]
            sess["character_list"] = []

        r = client.post(
            "/approve_outline",
            data=json.dumps({
                "title": "No Characters Novel",
                "chapters": [{"number": 1, "title": "Alone", "summary": "A solo journey"}],
                "characters": [],
            }),
            content_type="application/json",
        )
        assert r.status_code == 200

    def test_minimum_valid_outline(self, client, mock_llm):
        """The absolute minimum: 3 chapters, 1000 words, shortest premise."""
        r = client.post(
            "/generate_outline",
            data=json.dumps({
                "premise": "A",
                "genre": "Horror",
                "chapters": 3,
                "word_count": 1000,
            }),
            content_type="application/json",
        )
        assert r.status_code == 200

    def test_single_chapter_outline(self, client, mock_llm):
        """Approving a single-chapter outline should work."""
        with client.session_transaction() as sess:
            sess["premise"] = "A story"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 1
            sess["word_count"] = 5000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["title"] = "Test"
            sess["chapter_list"] = [{"number": 1, "title": "Ch1", "summary": "S"}]
            sess["character_list"] = []

        r = client.post(
            "/approve_outline",
            data=json.dumps({
                "title": "One Chapter",
                "chapters": [{"number": 1, "title": "The Story", "summary": "Everything happens"}],
                "characters": [],
            }),
            content_type="application/json",
        )
        assert r.status_code == 200

    def test_empty_chapter_summary(self, client, mock_llm):
        with client.session_transaction() as sess:
            sess["premise"] = "A story"
            sess["genre"] = "Fantasy"
            sess["chapters"] = 1
            sess["word_count"] = 5000
            sess["special_instructions"] = ""
            sess["special_events"] = ""
            sess["title"] = "Test"
            sess["chapter_list"] = [{"number": 1, "title": "Ch1", "summary": "S"}]
            sess["character_list"] = []

        r = client.post(
            "/approve_outline",
            data=json.dumps({
                "title": "Test",
                "chapters": [{"number": 1, "title": "Ch1", "summary": ""}],
                "characters": [],
            }),
            content_type="application/json",
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Session schema validation boundaries
# ---------------------------------------------------------------------------

class TestSessionSchemaEdgeCases:
    """Test session state validation with unusual/corrupt data."""

    def test_coerces_string_chapters_to_int(self):
        from novelforge.session.persistence import validate_session_state
        state = {"chapters": "15"}
        result = validate_session_state(state)
        assert result["chapters"] == 15
        assert isinstance(result["chapters"], int)

    def test_coerces_int_title_to_str(self):
        from novelforge.session.persistence import validate_session_state
        state = {"title": 42}
        result = validate_session_state(state)
        assert result["title"] == "42"
        assert isinstance(result["title"], str)

    def test_replaces_invalid_chapter_list_entries(self):
        from novelforge.session.persistence import validate_session_state
        state = {"chapter_list": ["not a dict", 123, None, {"number": 1, "title": "Good"}]}
        result = validate_session_state(state)
        assert all(isinstance(ch, dict) for ch in result["chapter_list"])
        assert result["chapter_list"][-1]["title"] == "Good"

    def test_replaces_invalid_character_list_entries(self):
        from novelforge.session.persistence import validate_session_state
        state = {"character_list": ["bad", {"name": "Alice"}]}
        result = validate_session_state(state)
        assert all(isinstance(ch, dict) for ch in result["character_list"])
        assert result["character_list"][-1]["name"] == "Alice"

    def test_removes_invalid_completed_chapters(self):
        from novelforge.session.persistence import validate_session_state
        state = {"completed_chapters": ["bad", None, {"number": 1}, 42]}
        result = validate_session_state(state)
        assert len(result["completed_chapters"]) == 1
        assert result["completed_chapters"][0]["number"] == 1

    def test_fills_missing_keys_with_defaults(self):
        from novelforge.session.persistence import validate_session_state
        state = {}
        result = validate_session_state(state)
        assert result["title"] == ""
        assert result["chapters"] == 0
        assert result["chapter_list"] == []
        assert result["character_list"] == []
        assert result["completed_chapters"] == []
        assert result["illustrations"] == []

    def test_preserves_extra_keys(self):
        from novelforge.session.persistence import validate_session_state
        state = {"title": "Test", "progress_data": {"status": "done"}, "custom_field": 42}
        result = validate_session_state(state)
        assert result["progress_data"] == {"status": "done"}
        assert result["custom_field"] == 42

    def test_replaces_wrong_type_list_with_default(self):
        from novelforge.session.persistence import validate_session_state
        state = {"chapter_list": "not a list"}
        result = validate_session_state(state)
        assert result["chapter_list"] == []

    def test_replaces_wrong_type_dict_with_default(self):
        from novelforge.session.persistence import validate_session_state
        state = {"story_architecture": "not a dict"}
        result = validate_session_state(state)
        assert result["story_architecture"] == {}

"""
Tests for chapter prompt builders in novelforge.agents.chapter.

Verifies that:
- build_characters_prompt is a pure function (data-in / prompt-out).
- No filesystem I/O occurs inside build_characters_prompt.
- The names_to_avoid parameter is correctly forwarded to the rendered prompt.
- collect_existing_character_names correctly scans a directory for prior names.
"""

import json
import pytest

from novelforge.agents.chapter import (
    build_characters_prompt,
    collect_existing_character_names,
)


# ---------------------------------------------------------------------------
# build_characters_prompt — pure-function contract
# ---------------------------------------------------------------------------

class TestBuildCharactersPromptPure:
    """build_characters_prompt must be deterministic and free of filesystem I/O."""

    def _call(self, names_to_avoid: str = "") -> list[dict[str, str]]:
        return build_characters_prompt(
            premise="A lone detective uncovers a city-wide conspiracy.",
            genre="Crime",
            outline_text="Chapter 1: Clue found. Chapter 2: Trail goes cold.",
            names_to_avoid=names_to_avoid,
        )

    def test_returns_list_of_message_dicts(self):
        result = self._call()
        assert isinstance(result, list)
        assert len(result) >= 1
        for msg in result:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg

    def test_deterministic_output_for_same_inputs(self):
        """Calling twice with identical inputs must produce identical output."""
        r1 = self._call(names_to_avoid="Vera, Hugo")
        r2 = self._call(names_to_avoid="Vera, Hugo")
        assert r1 == r2

    def test_names_to_avoid_appears_in_prompt(self):
        """The injected names_to_avoid string must surface somewhere in the prompt."""
        avoid = "Vera, Hugo, Marlowe"
        result = self._call(names_to_avoid=avoid)
        combined = " ".join(msg["content"] for msg in result)
        assert avoid in combined

    def test_empty_names_to_avoid_produces_valid_prompt(self):
        """An empty names_to_avoid should not break the prompt."""
        result = self._call(names_to_avoid="")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_no_filesystem_access(self, monkeypatch, tmp_path):
        """
        Even if NOVELS_DIR is set to a non-existent path, build_characters_prompt
        must not raise or attempt to read from disk.
        """
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path / "nonexistent"))
        # Must not raise FileNotFoundError or similar
        result = self._call()
        assert isinstance(result, list)

    def test_output_changes_when_names_to_avoid_differs(self):
        """Different names_to_avoid values should produce different prompt content."""
        result_empty = self._call(names_to_avoid="")
        result_names = self._call(names_to_avoid="SomeUniqueName42")
        # At minimum, one message's content should differ
        assert result_empty != result_names


# ---------------------------------------------------------------------------
# collect_existing_character_names — filesystem helper
# ---------------------------------------------------------------------------

class TestCollectExistingCharacterNames:
    """collect_existing_character_names scans NOVELS_DIR for prior character names."""

    def test_returns_empty_string_when_dir_missing(self, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path / "nonexistent"))
        result = collect_existing_character_names()
        assert result == ""

    def test_returns_empty_string_for_empty_dir(self, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))
        result = collect_existing_character_names()
        assert result == ""

    def test_collects_names_from_json_files(self, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        novel_data = {
            "character_list": [
                {"name": "Zelda", "role": "Protagonist"},
                {"name": "Orson", "role": "Antagonist"},
            ]
        }
        (tmp_path / "novel_1.json").write_text(
            json.dumps(novel_data), encoding="utf-8"
        )

        result = collect_existing_character_names()
        assert "Zelda" in result
        assert "Orson" in result

    def test_ignores_progress_json_files(self, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        progress_data = {
            "character_list": [{"name": "ShouldBeIgnored", "role": "Extra"}]
        }
        (tmp_path / "novel_1_progress.json").write_text(
            json.dumps(progress_data), encoding="utf-8"
        )

        result = collect_existing_character_names()
        assert "ShouldBeIgnored" not in result

    def test_skips_malformed_json_files(self, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        (tmp_path / "bad.json").write_text("not valid json", encoding="utf-8")
        (tmp_path / "good.json").write_text(
            json.dumps({"character_list": [{"name": "Valid"}]}),
            encoding="utf-8",
        )

        result = collect_existing_character_names()
        assert "Valid" in result

    def test_deduplicates_names_across_files(self, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        for i in range(3):
            (tmp_path / f"novel_{i}.json").write_text(
                json.dumps({"character_list": [{"name": "Recurring"}]}),
                encoding="utf-8",
            )

        result = collect_existing_character_names()
        assert result.count("Recurring") == 1

    def test_returns_sorted_comma_separated_string(self, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        (tmp_path / "novel.json").write_text(
            json.dumps({"character_list": [
                {"name": "Zara"},
                {"name": "Abel"},
                {"name": "Mira"},
            ]}),
            encoding="utf-8",
        )

        result = collect_existing_character_names()
        names = [n.strip() for n in result.split(",")]
        assert names == sorted(names)

    def test_ignores_entries_without_name_key(self, tmp_path, monkeypatch):
        import novelforge.config as config
        monkeypatch.setattr(config, "NOVELS_DIR", str(tmp_path))

        (tmp_path / "novel.json").write_text(
            json.dumps({"character_list": [{"role": "Extra"}]}),
            encoding="utf-8",
        )

        result = collect_existing_character_names()
        assert result == ""

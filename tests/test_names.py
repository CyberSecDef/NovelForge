"""Tests for the novelforge.names module (genre-aware name pools)."""

import pytest

from novelforge.names import (
    _GENRE_GROUP,
    _NAMES,
    _group_for_genre,
    format_name_pool_for_prompt,
    get_name_pool,
)
from novelforge.validation import ALLOWED_GENRES


# ---------------------------------------------------------------------------
# _group_for_genre
# ---------------------------------------------------------------------------

class TestGroupForGenre:
    def test_all_allowed_genres_map_to_known_group(self):
        known_groups = set(_NAMES.keys())
        for genre in ALLOWED_GENRES:
            group = _group_for_genre(genre)
            assert group in known_groups, (
                f"Genre {genre!r} mapped to unknown group {group!r}"
            )

    def test_unknown_genre_falls_back_to_contemporary(self):
        assert _group_for_genre("NonExistentGenre") == "contemporary"

    def test_explicit_genre_mappings(self):
        assert _group_for_genre("Science Fiction") == "scifi"
        assert _group_for_genre("Dystopian") == "scifi"
        assert _group_for_genre("Speculative Fiction") == "scifi"
        assert _group_for_genre("Fantasy") == "fantasy"
        assert _group_for_genre("Urban Fantasy") == "fantasy"
        assert _group_for_genre("Paranormal") == "fantasy"
        assert _group_for_genre("Magical Realism") == "fantasy"
        assert _group_for_genre("Historical Fiction") == "historical"
        assert _group_for_genre("Western") == "historical"
        assert _group_for_genre("Gothic Fiction") == "gothic"
        assert _group_for_genre("Contemporary Fiction") == "contemporary"
        assert _group_for_genre("Literary Fiction") == "contemporary"
        assert _group_for_genre("Romance") == "contemporary"
        assert _group_for_genre("Young Adult") == "contemporary"
        assert _group_for_genre("Satire Humor") == "contemporary"
        assert _group_for_genre("Crime") == "thriller"
        assert _group_for_genre("Mystery") == "thriller"
        assert _group_for_genre("Thriller") == "thriller"
        assert _group_for_genre("Noir") == "thriller"
        assert _group_for_genre("Horror") == "thriller"
        assert _group_for_genre("Adventure") == "thriller"


# ---------------------------------------------------------------------------
# Name pool data integrity
# ---------------------------------------------------------------------------

class TestNamePoolData:
    def test_all_groups_have_required_keys(self):
        required = {"male_first", "female_first", "last"}
        for group, pool in _NAMES.items():
            assert required == set(pool.keys()), (
                f"Group {group!r} missing keys: {required - set(pool.keys())}"
            )

    def test_each_pool_has_at_least_ten_names(self):
        for group, pool in _NAMES.items():
            for key, names in pool.items():
                assert len(names) >= 10, (
                    f"Group {group!r}[{key!r}] has fewer than 10 names: {len(names)}"
                )

    def test_no_duplicates_within_a_pool_key(self):
        for group, pool in _NAMES.items():
            for key, names in pool.items():
                assert len(names) == len(set(names)), (
                    f"Group {group!r}[{key!r}] has duplicate names"
                )

    def test_no_banned_names_in_pools(self):
        banned = {
            "james", "john", "sarah", "elena", "marcus",
            "alexander", "elizabeth", "catherine", "william", "michael",
            "david", "thomas", "robert", "alice", "emma",
            "kai", "liam", "aria", "luna", "maya",
            "zara", "ethan", "aiden", "elara", "kira",
            "mira", "sera", "nyx", "raven",
        }
        for group, pool in _NAMES.items():
            for key, names in pool.items():
                for name in names:
                    assert name.lower() not in banned, (
                        f"Banned name {name!r} found in group {group!r}[{key!r}]"
                    )

    def test_all_names_are_non_empty_strings(self):
        for group, pool in _NAMES.items():
            for key, names in pool.items():
                for name in names:
                    assert isinstance(name, str) and name.strip(), (
                        f"Empty/non-string name in group {group!r}[{key!r}]"
                    )


# ---------------------------------------------------------------------------
# get_name_pool
# ---------------------------------------------------------------------------

class TestGetNamePool:
    def test_returns_dict_with_correct_keys(self):
        pool = get_name_pool("Fantasy")
        assert set(pool.keys()) == {"male_first", "female_first", "last"}

    def test_returns_copy_not_original(self):
        pool1 = get_name_pool("Fantasy")
        pool1["male_first"].append("__test__")
        pool2 = get_name_pool("Fantasy")
        assert "__test__" not in pool2["male_first"]

    def test_all_allowed_genres_return_valid_pool(self):
        for genre in ALLOWED_GENRES:
            pool = get_name_pool(genre)
            assert isinstance(pool, dict)
            assert pool["male_first"]
            assert pool["female_first"]
            assert pool["last"]

    def test_unknown_genre_returns_contemporary_pool(self):
        pool_unknown = get_name_pool("UnknownGenre")
        pool_contemporary = get_name_pool("Contemporary Fiction")
        assert pool_unknown == pool_contemporary


# ---------------------------------------------------------------------------
# format_name_pool_for_prompt
# ---------------------------------------------------------------------------

class TestFormatNamePoolForPrompt:
    def test_returns_string(self):
        result = format_name_pool_for_prompt("Fantasy")
        assert isinstance(result, str)

    def test_contains_pool_names(self):
        result = format_name_pool_for_prompt("Fantasy")
        pool = get_name_pool("Fantasy")
        # At least the first name from each category should appear
        assert pool["male_first"][0] in result
        assert pool["female_first"][0] in result
        assert pool["last"][0] in result

    def test_scifi_prompt_describes_futuristic_style(self):
        result = format_name_pool_for_prompt("Science Fiction")
        assert "futuristic" in result.lower() or "alien" in result.lower()

    def test_fantasy_prompt_describes_old_english_style(self):
        result = format_name_pool_for_prompt("Fantasy")
        assert "old english" in result.lower() or "norse" in result.lower() or "anglo-saxon" in result.lower()

    def test_all_allowed_genres_produce_non_empty_output(self):
        for genre in ALLOWED_GENRES:
            result = format_name_pool_for_prompt(genre)
            assert result.strip(), f"Empty prompt for genre {genre!r}"

    def test_output_mentions_male_female_last(self):
        result = format_name_pool_for_prompt("Crime")
        assert "male first names" in result.lower()
        assert "female first names" in result.lower()
        assert "last names" in result.lower()

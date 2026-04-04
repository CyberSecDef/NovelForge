"""
Tests for novelforge/voice.py — voice seed selection and formatting.
"""

import random

import pytest

from novelforge.voice import (
    _PREMISE_KEYWORD_BOOST,
    _PREMISE_KEYWORDS,
    _VOICE_SEEDS,
    format_voice_prompt,
    select_voice_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_counts(results: list[dict]) -> dict[str, int]:
    """Return a frequency map of voice names from a list of seed dicts."""
    counts: dict[str, int] = {}
    for seed in results:
        counts[seed["name"]] = counts.get(seed["name"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# select_voice_seed — basic contract
# ---------------------------------------------------------------------------

class TestSelectVoiceSeedBasic:
    """select_voice_seed always returns a valid voice seed dict."""

    def test_returns_dict_with_required_keys(self):
        seed = select_voice_seed()
        assert isinstance(seed, dict)
        assert {"name", "prose_style", "emotional_register", "sensory_preference"} <= seed.keys()

    def test_returned_seed_is_from_known_pool(self):
        known_names = {s["name"] for s in _VOICE_SEEDS}
        for _ in range(50):
            seed = select_voice_seed()
            assert seed["name"] in known_names

    def test_no_args_does_not_raise(self):
        select_voice_seed()

    def test_empty_strings_do_not_raise(self):
        select_voice_seed(genre="", premise="")

    def test_unknown_genre_does_not_raise(self):
        seed = select_voice_seed(genre="NonExistentGenre", premise="A hero's journey")
        assert seed["name"] in {s["name"] for s in _VOICE_SEEDS}


# ---------------------------------------------------------------------------
# select_voice_seed — genre bias
# ---------------------------------------------------------------------------

class TestSelectVoiceSeedGenreBias:
    """Genre weighting makes preferred voices significantly more likely."""

    def test_horror_favors_gothic_atmospheric(self):
        random.seed(42)
        results = [select_voice_seed(genre="Horror") for _ in range(200)]
        counts = _name_counts(results)
        # gothic_atmospheric, visceral_kinetic, sharp_angular are all preferred
        preferred_total = (
            counts.get("gothic_atmospheric", 0)
            + counts.get("visceral_kinetic", 0)
            + counts.get("sharp_angular", 0)
        )
        assert preferred_total > 100, (
            f"Horror preferred voices should dominate; got {preferred_total}/200"
        )

    def test_romance_favors_lyrical_flowing(self):
        random.seed(0)
        results = [select_voice_seed(genre="Romance") for _ in range(200)]
        counts = _name_counts(results)
        preferred_total = (
            counts.get("lyrical_flowing", 0)
            + counts.get("conversational_intimate", 0)
            + counts.get("dense_literary", 0)
        )
        assert preferred_total > 100


# ---------------------------------------------------------------------------
# select_voice_seed — premise bias
# ---------------------------------------------------------------------------

class TestSelectVoiceSeedPremiseBias:
    """Premise keyword signals genuinely shift the distribution."""

    def test_battle_premise_boosts_visceral_kinetic(self):
        """A premise full of combat keywords should favor visceral_kinetic."""
        random.seed(7)
        premise = "A brutal battle and desperate chase for survival in combat"
        results = [select_voice_seed(genre="", premise=premise) for _ in range(300)]
        counts = _name_counts(results)
        # visceral_kinetic has multiple keyword matches → should appear more
        # than its equal-weight baseline of ~300/8 ≈ 37
        assert counts.get("visceral_kinetic", 0) > 50, (
            f"Expected visceral_kinetic to be boosted; counts={counts}"
        )

    def test_ghost_premise_boosts_gothic_atmospheric(self):
        """A premise with haunting/eerie words should favour gothic_atmospheric."""
        random.seed(13)
        premise = "A haunted mansion where shadows hide something sinister and eerie"
        results = [select_voice_seed(genre="", premise=premise) for _ in range(300)]
        counts = _name_counts(results)
        assert counts.get("gothic_atmospheric", 0) > 50, (
            f"Expected gothic_atmospheric to be boosted; counts={counts}"
        )

    def test_family_premise_boosts_conversational_intimate(self):
        random.seed(21)
        premise = "A family saga about friendship, community, and personal relationships"
        results = [select_voice_seed(genre="", premise=premise) for _ in range(300)]
        counts = _name_counts(results)
        assert counts.get("conversational_intimate", 0) > 50, (
            f"Expected conversational_intimate to be boosted; counts={counts}"
        )

    def test_premise_with_no_keywords_does_not_change_distribution_from_baseline(self):
        """A premise with zero matching keywords should behave like no premise."""
        random.seed(99)
        results_no_premise = [select_voice_seed(genre="Fantasy") for _ in range(500)]
        random.seed(99)
        results_irrelevant = [
            select_voice_seed(genre="Fantasy", premise="xyz qqq zzz")
            for _ in range(500)
        ]
        # Both runs use the same seed so should produce identical results
        assert results_no_premise == results_irrelevant

    def test_premise_bias_is_additive_with_genre_bias(self):
        """When genre and premise agree, that voice should be especially likely."""
        random.seed(55)
        # Gothic Fiction already weights gothic_atmospheric; add premise keywords too
        premise = "A haunted, cursed estate with sinister shadows and eerie dread"
        results = [
            select_voice_seed(genre="Gothic Fiction", premise=premise) for _ in range(300)
        ]
        counts = _name_counts(results)
        # gothic_atmospheric: 3 (genre) + ≤3 (premise) copies → very dominant
        assert counts.get("gothic_atmospheric", 0) > 100, (
            f"gothic_atmospheric should dominate; counts={counts}"
        )

    def test_premise_keyword_boost_cap_is_respected(self):
        """Pool size never grows beyond expected maximum."""
        # Maximum pool size = sum of (3 or 1 base + up to _PREMISE_KEYWORD_BOOST) for each seed
        max_per_seed = 3 + _PREMISE_KEYWORD_BOOST
        max_pool_size = len(_VOICE_SEEDS) * max_per_seed
        # Build a premise with ALL keywords to trigger maximum boosting
        all_keywords = [kw for kws in _PREMISE_KEYWORDS.values() for kw in kws]
        premise = " ".join(all_keywords)
        # Exercise select_voice_seed many times to implicitly rely on no crash
        for _ in range(50):
            seed = select_voice_seed(genre="Horror", premise=premise)
            assert seed["name"] in {s["name"] for s in _VOICE_SEEDS}

    def test_premise_case_insensitive(self):
        """Keywords in the premise should be matched regardless of case."""
        random.seed(3)
        results_lower = [
            select_voice_seed(genre="", premise="haunted ghost shadow") for _ in range(200)
        ]
        random.seed(3)
        results_upper = [
            select_voice_seed(genre="", premise="HAUNTED GHOST SHADOW") for _ in range(200)
        ]
        assert results_lower == results_upper

    def test_premise_punctuation_stripped(self):
        """Punctuation around keywords should not prevent matching."""
        random.seed(77)
        results_clean = [
            select_voice_seed(genre="", premise="haunted ghost shadow") for _ in range(200)
        ]
        random.seed(77)
        results_punct = [
            select_voice_seed(genre="", premise="haunted! ghost, shadow.") for _ in range(200)
        ]
        assert results_clean == results_punct


# ---------------------------------------------------------------------------
# format_voice_prompt
# ---------------------------------------------------------------------------

class TestFormatVoicePrompt:
    """format_voice_prompt produces a correctly structured string."""

    def test_contains_all_sections(self):
        seed = _VOICE_SEEDS[0]
        prompt = format_voice_prompt(seed)
        assert "VOICE & STYLE GUIDE" in prompt
        assert "Prose style:" in prompt
        assert "Emotional register:" in prompt
        assert "Sensory preference:" in prompt

    def test_embeds_seed_content(self):
        seed = _VOICE_SEEDS[0]
        prompt = format_voice_prompt(seed)
        assert seed["prose_style"] in prompt
        assert seed["emotional_register"] in prompt
        assert seed["sensory_preference"] in prompt

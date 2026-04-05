"""
Tests for the vocabulary overuse scanner.

Covers:
- Word-boundary matching: forbidden words must NOT match as substrings of
  longer words (e.g., "audit" must not flag "auditor").
- Exact matches: standalone forbidden words ARE detected.
- Case insensitivity: "Delve" and "DELVE" both match "delve".
- Soft-limit thresholds: one occurrence is allowed; two+ triggers a warning.
- Overused pattern detection: multi-word phrases matched at word boundaries.
- Clean text: text with no violations returns an empty list.
- Mixed violations: text with multiple tiers produces warnings from each tier.
"""

import pytest

from novelforge.agents.chapter._helpers import (
    scan_vocabulary_overuse,
    _FORBIDDEN_WORDS,
    _SOFT_LIMITED_WORDS,
    _OVERUSED_PATTERNS,
    _HARD_BAN_THRESHOLD,
    _SOFT_LIMIT_PER_CHAPTER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_warning_for(warnings: list[str], word: str) -> bool:
    """Check if any warning message references the given word (case-insensitive)."""
    word_lower = word.lower()
    return any(word_lower in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Word-boundary matching — false positive prevention
# ---------------------------------------------------------------------------

class TestForbiddenWordBoundaries:
    """Forbidden words must only match as whole words, not substrings."""

    @pytest.mark.parametrize("text,should_not_match", [
        ("The auditor reviewed the case.", "audit"),
        ("She was auditing the records.", "audit"),
        ("The auditorium was packed.", "audit"),
        ("He used a sledgehammer on the wall.", "ledger"),
        ("The metallurgy professor spoke.", "tally"),
        ("She was totally unprepared.", "tally"),
        ("The items were inventoried and catalogued.", "inventory"),
        ("She embarked on a journey.", "embark"),
        ("The uncharted territory was vast.", "uncharted"),
        # "uncharted" IS itself in the forbidden list, so it WILL match.
        # Let's test a word that contains a forbidden word as a true substring:
        ("The beachhead was secured.", "beacon"),
        ("His dividends were reinvested.", "dividend"),
    ])
    def test_no_false_positive_on_longer_word(self, text, should_not_match):
        # Skip cases where the longer word actually IS in the forbidden list
        if should_not_match == "uncharted":
            pytest.skip("'uncharted' is itself a forbidden word")
        warnings = scan_vocabulary_overuse(text)
        assert not _has_warning_for(warnings, should_not_match), (
            f'"{should_not_match}" should not match in: {text!r}\nWarnings: {warnings}'
        )

    def test_embarked_does_not_match_embark(self):
        """'embarked' is a conjugation — word boundary sits after 'embark' + 'ed'."""
        # Actually: "embarked" — \bembark\b would NOT match because 'e' follows 'k'.
        # The regex requires a non-word char (or string end) after "embark".
        warnings = scan_vocabulary_overuse("She embarked on a journey.")
        assert not _has_warning_for(warnings, "embark")

    def test_auditor_does_not_match_audit(self):
        warnings = scan_vocabulary_overuse("The auditor found discrepancies.")
        assert not _has_warning_for(warnings, "audit")

    def test_sledgehammer_does_not_match_ledger(self):
        warnings = scan_vocabulary_overuse("He swung the sledgehammer hard.")
        assert not _has_warning_for(warnings, "ledger")

    def test_metallurgy_does_not_match_tally(self):
        warnings = scan_vocabulary_overuse("She studied metallurgy at university.")
        assert not _has_warning_for(warnings, "tally")

    def test_totally_does_not_match_tally(self):
        warnings = scan_vocabulary_overuse("He was totally confused.")
        assert not _has_warning_for(warnings, "tally")


# ---------------------------------------------------------------------------
# Exact match detection — true positives
# ---------------------------------------------------------------------------

class TestForbiddenWordDetection:
    """Standalone forbidden words must be detected."""

    def test_exact_forbidden_word_detected(self):
        warnings = scan_vocabulary_overuse("She wanted to delve into the mystery.")
        assert _has_warning_for(warnings, "delve")

    def test_forbidden_word_at_start_of_sentence(self):
        warnings = scan_vocabulary_overuse("Realm of the lost was her destination.")
        assert _has_warning_for(warnings, "realm")

    def test_forbidden_word_at_end_of_sentence(self):
        warnings = scan_vocabulary_overuse("It was a true testament.")
        assert _has_warning_for(warnings, "testament")

    def test_forbidden_word_with_punctuation(self):
        warnings = scan_vocabulary_overuse('He said, "delve deeper."')
        assert _has_warning_for(warnings, "delve")

    def test_multi_word_forbidden_entry(self):
        """'balance sheet' is a multi-word forbidden entry."""
        warnings = scan_vocabulary_overuse("The balance sheet showed a deficit.")
        assert _has_warning_for(warnings, "balance sheet")

    def test_count_is_accurate(self):
        text = "The realm was vast. Another realm appeared. A third realm emerged."
        warnings = scan_vocabulary_overuse(text)
        matching = [w for w in warnings if "realm" in w.lower()]
        assert len(matching) == 1
        assert "3x" in matching[0]


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

class TestCaseInsensitivity:

    def test_uppercase_forbidden_word(self):
        warnings = scan_vocabulary_overuse("DELVE into the unknown.")
        assert _has_warning_for(warnings, "delve")

    def test_mixed_case_forbidden_word(self):
        warnings = scan_vocabulary_overuse("He wanted to Delve deeper.")
        assert _has_warning_for(warnings, "delve")

    def test_uppercase_soft_limited_word(self):
        # Need 2+ occurrences to trigger soft limit
        warnings = scan_vocabulary_overuse("BRITTLE ice. BRITTLE bones.")
        assert _has_warning_for(warnings, "brittle")


# ---------------------------------------------------------------------------
# Soft-limited words — threshold behavior
# ---------------------------------------------------------------------------

class TestSoftLimitThreshold:
    """Soft-limited words are allowed once per chapter but flagged at 2+."""

    def test_one_occurrence_not_flagged(self):
        warnings = scan_vocabulary_overuse("The brittle ice cracked.")
        assert not _has_warning_for(warnings, "brittle")

    def test_two_occurrences_flagged(self):
        warnings = scan_vocabulary_overuse("The brittle ice cracked. Her brittle smile faded.")
        assert _has_warning_for(warnings, "brittle")

    def test_three_occurrences_shows_correct_count(self):
        text = "Brittle ice. Brittle bones. Brittle resolve."
        warnings = scan_vocabulary_overuse(text)
        matching = [w for w in warnings if "brittle" in w.lower()]
        assert len(matching) == 1
        assert "3x" in matching[0]

    def test_soft_limit_boundary_at_threshold(self):
        """Exactly _SOFT_LIMIT_PER_CHAPTER occurrences should NOT trigger."""
        assert _SOFT_LIMIT_PER_CHAPTER == 1  # Ensure our assumption holds
        warnings = scan_vocabulary_overuse("A steady hand guided the ship.")
        assert not _has_warning_for(warnings, "steady")

    def test_steadied_does_not_match_steady(self):
        """Conjugated forms should not match the base word."""
        warnings = scan_vocabulary_overuse("She steadied herself. He steadied his aim.")
        # "steadied" should NOT match "steady" due to word boundaries
        assert not _has_warning_for(warnings, "steady")
        # But "steadied" IS in the soft-limited list, so it should match itself
        assert _has_warning_for(warnings, "steadied")


# ---------------------------------------------------------------------------
# Overused patterns — multi-word phrase detection
# ---------------------------------------------------------------------------

class TestOverusedPatterns:

    def test_pattern_detected(self):
        warnings = scan_vocabulary_overuse("It was a small mercy that she survived.")
        assert _has_warning_for(warnings, "small mercy")

    def test_pattern_not_detected_as_substring(self):
        """Patterns should match at word boundaries."""
        # "small mercy" should not match inside "smallest mercy" — "small" != "smallest"
        warnings = scan_vocabulary_overuse("It was the smallest mercy.")
        assert not _has_warning_for(warnings, "small mercy")

    def test_pattern_with_surrounding_punctuation(self):
        warnings = scan_vocabulary_overuse('"Small mercy," she whispered.')
        assert _has_warning_for(warnings, "small mercy")

    def test_jaw_tightened_detected(self):
        warnings = scan_vocabulary_overuse("His jaw tightened as he stared.")
        assert _has_warning_for(warnings, "jaw tightened")

    def test_moral_calculus_detected(self):
        warnings = scan_vocabulary_overuse("The moral calculus was inescapable.")
        assert _has_warning_for(warnings, "moral calculus")


# ---------------------------------------------------------------------------
# Clean text — no false positives
# ---------------------------------------------------------------------------

class TestCleanText:

    def test_no_warnings_on_clean_text(self):
        text = (
            "Sarah walked through the rain-slicked streets, her coat pulled tight "
            "against the wind. The city hummed around her — taxis, voices, the "
            "distant clatter of a train crossing the bridge overhead. She had "
            "twenty minutes before the meeting and no idea what she would say."
        )
        warnings = scan_vocabulary_overuse(text)
        assert warnings == []

    def test_empty_string_returns_no_warnings(self):
        assert scan_vocabulary_overuse("") == []


# ---------------------------------------------------------------------------
# Mixed violations
# ---------------------------------------------------------------------------

class TestMixedViolations:

    def test_multiple_tiers_in_one_text(self):
        text = (
            "She wanted to delve into the tapestry of the realm. "       # 3 forbidden
            "Brittle silence. Brittle resolve. "                          # 2 soft-limited
            "It sat behind her ribs like a stone."                        # 1 overused pattern
        )
        warnings = scan_vocabulary_overuse(text)
        assert _has_warning_for(warnings, "delve")
        assert _has_warning_for(warnings, "tapestry")
        assert _has_warning_for(warnings, "realm")
        assert _has_warning_for(warnings, "brittle")
        assert _has_warning_for(warnings, "sat behind her ribs")
        assert len(warnings) >= 5

    def test_warning_messages_contain_counts(self):
        text = "Delve here. Delve there. Delve everywhere."
        warnings = scan_vocabulary_overuse(text)
        matching = [w for w in warnings if "delve" in w.lower()]
        assert len(matching) == 1
        assert "3x" in matching[0]
        assert "BANNED WORD" in matching[0]

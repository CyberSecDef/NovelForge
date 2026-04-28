"""Tests for the character reconciliation pipeline.

Covers the three layers that close the character consistency gap:

1. ``extract_named_characters`` — pure-Python scanner that classifies
   proper-noun spans in prose against the canonical roster (known,
   unknown, variant).
2. ``_apply_name_substitution`` — regex-based prose rewriter used by
   RENAME_TO and DEMOTE_TO_EXTRA resolutions.
3. ``run_character_reconciliation`` — LLM-backed runner that applies
   decisions (register, rename, demote) to roster and prose.
"""

import json

import pytest

from novelforge.agents.chapter._helpers import extract_named_characters
from novelforge.agents.chapter.pipeline import (
    _apply_name_substitution,
    run_character_reconciliation,
)


# ---------------------------------------------------------------------------
# extract_named_characters — pure Python scanner
# ---------------------------------------------------------------------------

class TestScannerRosterMatching:
    """A roster character mentioned in prose should be classified as known."""

    def test_roster_first_name_detected(self):
        roster = [{"name": "Sarah Miller"}]
        text = "Sarah walked into the room. Sarah smiled."
        result = extract_named_characters(text, roster)
        assert "Sarah" in result["known"]
        assert result["unknown"] == []
        assert result["variants"] == []

    def test_roster_last_name_detected(self):
        roster = [{"name": "Aldric Holt"}]
        text = "Holt raised his hand. Holt looked tired."
        result = extract_named_characters(text, roster)
        assert "Holt" in result["known"]
        assert result["unknown"] == []

    def test_roster_full_name_detected(self):
        roster = [{"name": "John Smith"}]
        text = "John Smith entered. John Smith smiled."
        result = extract_named_characters(text, roster)
        # The two-token span is matched as known via token overlap
        assert any(s in result["known"] for s in ("John Smith", "John"))

    def test_possessive_handled_by_regex_boundary(self):
        """`\\b` anchoring means `Sarah's` yields `Sarah`, matching the roster."""
        roster = [{"name": "Sarah Miller"}]
        text = "Sarah's key was gone. Sarah's coat remained."
        result = extract_named_characters(text, roster)
        assert "Sarah" in result["known"]


class TestScannerUnknownDetection:
    """Unknown capitalized names above the mention threshold must surface."""

    def test_unknown_name_meets_threshold(self):
        roster = [{"name": "Sarah Miller"}]
        text = (
            "Thomas lit a cigarette. Sarah watched. "
            "Thomas crossed the room. Thomas never looked back."
        )
        result = extract_named_characters(text, roster)
        names = [n for n, _ in result["unknown"]]
        assert "Thomas" in names

    def test_single_mention_not_flagged(self):
        """A walk-on with a single mention is almost always a false positive."""
        roster = [{"name": "Sarah Miller"}]
        text = "Sarah talked to a stranger named Orion. The encounter ended."
        result = extract_named_characters(text, roster)
        # "Orion" appears only once → below the default min_mentions=2 threshold.
        names = [n for n, _ in result["unknown"]]
        assert "Orion" not in names

    def test_sentence_initial_stop_words_filtered(self):
        """Sentence-initial common words must not be reported as unknowns."""
        roster = [{"name": "Sarah Miller"}]
        text = (
            "Sarah walked home. Then she paused. "
            "Then she turned. Then she ran. "
            "Then the rain started. Then the lights flickered."
        )
        result = extract_named_characters(text, roster)
        names = [n for n, _ in result["unknown"]]
        # "Then" appears many times but it's a stop word.
        assert "Then" not in names

    def test_sentence_initial_prepositions_filtered(self):
        """Prepositions like 'In' or 'On' must never be flagged as names."""
        roster = [{"name": "Sarah Miller"}]
        text = (
            "Sarah listened. In the dark, the wind moaned. "
            "On the table, the lamp flickered. "
            "In the kitchen, water ran. "
            "On the porch, footsteps faded."
        )
        result = extract_named_characters(text, roster)
        names = [n for n, _ in result["unknown"]]
        assert "In" not in names
        assert "On" not in names

    def test_sentence_initial_modals_and_verbs_filtered(self):
        """Modals like 'Will' and verbs like 'Let' must not be flagged."""
        roster = [{"name": "Sarah Miller"}]
        text = (
            "Sarah waited. Will the storm pass? "
            "Let the bells ring. Will you answer? "
            "Let them in. Will it hold?"
        )
        result = extract_named_characters(text, roster)
        names = [n for n, _ in result["unknown"]]
        assert "Will" not in names
        assert "Let" not in names

    def test_leading_stopword_strip_resolves_to_roster(self):
        """A two-token span like 'If Meridian' must collapse onto roster 'Meridian'."""
        roster = [{"name": "Meridian"}]
        text = (
            "Meridian crossed the floor. If Meridian sees the door, she will run. "
            "Meridian paused."
        )
        result = extract_named_characters(text, roster)
        unknown_names = [n for n, _ in result["unknown"]]
        # "If Meridian" should not surface as an unknown — Meridian is on the roster.
        assert "If Meridian" not in unknown_names


class TestScannerVariantDetection:
    """Fuzzy-close spellings should be reported as variants, not unknowns."""

    def test_diminutive_matches_roster(self):
        roster = [{"name": "Sarah Miller"}]
        text = "Sara grinned. Sara pushed open the door."
        result = extract_named_characters(text, roster)
        variant_names = [n for n, _, _ in result["variants"]]
        assert "Sara" in variant_names

    def test_typo_matches_roster(self):
        roster = [{"name": "Aldric Holt"}]
        text = "Aldrick entered. Aldrick began to explain."
        result = extract_named_characters(text, roster)
        variant_names = [n for n, _, _ in result["variants"]]
        assert "Aldrick" in variant_names

    def test_different_name_is_unknown_not_variant(self):
        """Distinct names (e.g., Sarah vs. Jonathan) must not fuzzy-match."""
        roster = [{"name": "Sarah Miller"}]
        text = "Jonathan walked home. Jonathan opened the door."
        result = extract_named_characters(text, roster)
        unknown_names = [n for n, _ in result["unknown"]]
        variant_names = [n for n, _, _ in result["variants"]]
        assert "Jonathan" in unknown_names
        assert "Jonathan" not in variant_names


class TestScannerEdgeCases:
    """Boundary conditions: empty roster, empty text, min_mentions override."""

    def test_empty_roster_treats_all_capitalized_as_unknown(self):
        text = "Thomas walked. Thomas paused. Thomas left."
        result = extract_named_characters(text, roster=[])
        names = [n for n, _ in result["unknown"]]
        assert "Thomas" in names

    def test_empty_text_returns_empty_result(self):
        roster = [{"name": "Sarah Miller"}]
        result = extract_named_characters("", roster)
        assert result == {"known": [], "unknown": [], "variants": []}

    def test_min_mentions_override(self):
        """Lowering min_mentions should expose single-mention names."""
        roster = [{"name": "Sarah Miller"}]
        text = "Sarah saw Orion in the corner."
        result = extract_named_characters(text, roster, min_mentions=1)
        names = [n for n, _ in result["unknown"]]
        assert "Orion" in names


# ---------------------------------------------------------------------------
# _apply_name_substitution — regex rewrite with sentence-boundary capitalization
# ---------------------------------------------------------------------------

class TestApplyNameSubstitution:
    """The rewrite helper is used by both RENAME_TO and DEMOTE_TO_EXTRA."""

    def test_mid_sentence_uses_lowercase(self):
        out = _apply_name_substitution(
            "The crowd watched Thomas cross the square.",
            "Thomas", "the guard",
        )
        assert "the guard" in out
        assert "Thomas" not in out

    def test_sentence_start_capitalizes(self):
        out = _apply_name_substitution(
            "Thomas crossed the square.",
            "Thomas", "the guard",
        )
        # Must be capitalized at sentence start.
        assert out.startswith("The guard")

    def test_possessive_preserved(self):
        out = _apply_name_substitution(
            "Thomas's coat was wet.",
            "Thomas", "the guard",
        )
        assert "the guard's coat" in out.lower()

    def test_whole_word_only(self):
        """A short name must never replace a substring inside another word."""
        out = _apply_name_substitution(
            "Al was the Alabama native. Al left. Al returned.",
            "Al", "the stranger",
        )
        # Alabama must survive unchanged
        assert "Alabama" in out
        # Al itself is replaced (all three occurrences are sentence-initial → capitalized)
        assert "The stranger" in out
        assert " Al " not in f" {out} "

    def test_empty_replacement_no_change(self):
        text = "Thomas waited."
        assert _apply_name_substitution(text, "Thomas", "") == text


# ---------------------------------------------------------------------------
# run_character_reconciliation — LLM-backed end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_reconciliation_llm(mocker):
    """Patch call_llm everywhere the pipeline reaches so the runner can drive decisions from a controlled JSON payload."""

    def _install(payload: dict):
        def _side_effect(messages, *, action="", json_mode=False):
            act = (action or "").lower()
            if "character reconciliation" in act:
                return json.dumps(payload)
            # Field-repair call for REGISTER — return short string value.
            if "reconciliation: repairing" in act:
                return "short value"
            return "ok"

        for module in (
            "novelforge.llm.client",
            "novelforge.agents.base",
            "novelforge.agents.chapter",
            "novelforge.agents.chapter._helpers",
            "novelforge.agents.chapter.pipeline",
        ):
            mocker.patch(f"{module}.call_llm", side_effect=_side_effect)
        from novelforge.llm.client import reset_circuit_breakers
        reset_circuit_breakers()

    return _install


class TestRunReconciliation:
    """The runner must apply REGISTER, RENAME_TO, and DEMOTE_TO_EXTRA decisions."""

    def test_no_detections_returns_inputs_unchanged(self, patch_reconciliation_llm):
        patch_reconciliation_llm({"decisions": []})
        roster = [{"name": "Sarah Miller"}]
        text = "Sarah walked home. Sarah locked the door."
        out_text, out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        assert out_text == text
        assert out_roster == roster
        assert log["decisions"] == []

    def test_register_appends_new_character(self, patch_reconciliation_llm):
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "Thomas",
                "resolution": "REGISTER",
                "roster_name": "Thomas Black",
                "profile": {
                    "name": "Thomas Black",
                    "age": 42,
                    "role": "recurring ally",
                    "background": "former engineer",
                    "arc": "learns to trust the team",
                    "internal_flaw": "guarded",
                    "personal_goal": "rebuild his reputation",
                },
            }]
        })
        roster = [{"name": "Sarah Miller"}]
        text = (
            "Thomas lit the lamp. Thomas waited for Sarah. "
            "Thomas finally spoke. Sarah answered."
        )
        out_text, out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        names = [c.get("name") for c in out_roster]
        assert "Thomas Black" in names
        # Prose is not modified for REGISTER
        assert out_text == text
        assert log["decisions"][0]["resolution"] == "REGISTER"

    def test_rename_rewrites_prose(self, patch_reconciliation_llm):
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "Sara",
                "resolution": "RENAME_TO",
                "roster_name": "Sarah Miller",
            }]
        })
        roster = [{"name": "Sarah Miller"}]
        text = "Sara opened the door. Sara locked it again."
        out_text, out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        assert "Sara " not in out_text and "Sara." not in out_text
        assert "Sarah Miller" in out_text
        assert out_roster == roster  # No registration
        assert log["decisions"][0]["resolution"] == "RENAME_TO"

    def test_demote_replaces_prose_with_role(self, patch_reconciliation_llm):
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "Jenkins",
                "resolution": "DEMOTE_TO_EXTRA",
                "replacement": "the clerk",
            }]
        })
        roster = [{"name": "Sarah Miller"}]
        text = "Sarah nodded at Jenkins. Jenkins stamped the form."
        out_text, out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        assert "Jenkins" not in out_text
        assert "the clerk" in out_text
        # Sentence-initial instance is capitalized
        assert "The clerk" in out_text
        assert out_roster == roster
        assert log["decisions"][0]["resolution"] == "DEMOTE_TO_EXTRA"

    def test_register_does_not_duplicate_existing_name(self, patch_reconciliation_llm):
        """Attempting to REGISTER an already-rostered name is a silent no-op."""
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "Sarah",
                "resolution": "REGISTER",
                "roster_name": "Sarah Miller",
                "profile": {"name": "Sarah Miller", "role": "duplicate attempt"},
            }]
        })
        roster = [{"name": "Sarah Miller", "role": "protagonist"}]
        text = "Sarah walked. Sarah paused. Sarah turned."
        _, out_roster, _ = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        # No duplication — only one Sarah Miller in the roster.
        names = [c.get("name") for c in out_roster]
        assert names.count("Sarah Miller") == 1

    def test_demote_with_grammatical_descriptor_is_rejected(self, patch_reconciliation_llm):
        """A 'replacement' like \"the preposition 'in'\" must not corrupt prose."""
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "In",
                "resolution": "DEMOTE_TO_EXTRA",
                "replacement": "the preposition 'in'",
            }]
        })
        roster = [{"name": "Sarah Miller"}]
        text = "Sarah waited. In the dark, she moved. In the corner she paused."
        out_text, out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        # Prose must not be rewritten with the grammatical descriptor.
        assert "preposition" not in out_text
        assert "In the dark" in out_text
        assert log["decisions"] == []

    def test_demote_with_stopword_prose_name_is_rejected(self, patch_reconciliation_llm):
        """A DEMOTE for a prose_name that's purely a stop word must be rejected."""
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "Control",
                "resolution": "DEMOTE_TO_EXTRA",
                "replacement": "the concept 'control'",
            }]
        })
        roster = [{"name": "Sarah Miller"}]
        text = "Control the story. Sarah listened. Control was paramount."
        out_text, _out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        assert "concept" not in out_text
        assert "Control the story" in out_text
        assert log["decisions"] == []

    @pytest.mark.parametrize("descriptor,replacement", [
        ("modifier", "the modifier 'low'"),
        ("direction", "the direction 'back'"),
        ("descriptor", "the descriptor 'immediate'"),
        ("quantifier", "the quantifier 'each'"),
        ("numeral", "a numeral"),
        ("possessive", "a possessive pronoun"),
        ("conditional", "the conditional phrase 'if Meridian'"),
    ])
    def test_demote_with_extra_grammatical_descriptors_is_rejected(
        self, patch_reconciliation_llm, descriptor, replacement,
    ):
        """All grammatical descriptors observed in the wild must be rejected."""
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "Foo",
                "resolution": "DEMOTE_TO_EXTRA",
                "replacement": replacement,
            }]
        })
        roster = [{"name": "Sarah"}]
        text = "Foo arrived. Sarah looked up. Foo waited."
        out_text, _out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        assert descriptor not in out_text.lower(), f"replacement '{replacement}' leaked into prose"
        assert log["decisions"] == []

    def test_demote_sentence_initial_only_word_is_rejected(self, patch_reconciliation_llm):
        """A single-token prose_name that only appears at sentence starts must not be demoted."""
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "Photograph",
                "resolution": "DEMOTE_TO_EXTRA",
                "replacement": "a photograph",
            }]
        })
        roster = [{"name": "Sarah Miller"}]
        text = "Sarah waited. Photograph the artifact. Photograph the seal too."
        out_text, _out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        # Prose must not be rewritten — both occurrences are sentence-initial.
        assert "Photograph the artifact" in out_text
        assert "Photograph the seal" in out_text
        assert log["decisions"] == []

    def test_demote_with_mid_sentence_occurrence_still_applies(self, patch_reconciliation_llm):
        """A walk-on with even one mid-sentence mention is a real character — demotion applies."""
        patch_reconciliation_llm({
            "decisions": [{
                "prose_name": "Jenkins",
                "resolution": "DEMOTE_TO_EXTRA",
                "replacement": "the clerk",
            }]
        })
        roster = [{"name": "Sarah Miller"}]
        # Jenkins appears mid-sentence, so the positional heuristic must NOT block.
        text = "Sarah nodded at Jenkins. Jenkins stamped the form."
        out_text, _out_roster, log = run_character_reconciliation(
            chapter_text=text, character_list=roster,
            chapter_num=1, title="Test", genre="Thriller",
        )
        assert "Jenkins" not in out_text
        assert "the clerk" in out_text
        assert log["decisions"][0]["resolution"] == "DEMOTE_TO_EXTRA"

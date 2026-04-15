"""Tests for the chapter length enforcement feature.

Covers:
- check_chapter_length() pure validation logic
- expand_chapter() retry and failure handling
- build_chapter_expansion_prompt() prompt builder
- Config constants (CHAPTER_MIN_LENGTH_PCT, MAX_EXPANSION_ATTEMPTS)
- Integration with _run_all_chapter_agents() via target_words parameter
"""

import pytest

from novelforge.agents.chapter._helpers import check_chapter_length, expand_chapter
from novelforge.agents.chapter.prompts import build_chapter_expansion_prompt


# ---------------------------------------------------------------------------
# check_chapter_length — pure validation
# ---------------------------------------------------------------------------

class TestCheckChapterLength:
    """Validate the pure-Python length-checking function."""

    def test_meets_target_exactly(self):
        text = " ".join(["word"] * 4000)
        actual, min_threshold, ok = check_chapter_length(text, target_words=4000, min_pct=85)
        assert actual == 4000
        assert min_threshold == 3400
        assert ok is True

    def test_above_target(self):
        text = " ".join(["word"] * 5000)
        actual, _, ok = check_chapter_length(text, target_words=4000, min_pct=85)
        assert actual == 5000
        assert ok is True

    def test_below_threshold(self):
        text = " ".join(["word"] * 2000)
        actual, min_threshold, ok = check_chapter_length(text, target_words=4000, min_pct=85)
        assert actual == 2000
        assert min_threshold == 3400
        assert ok is False

    def test_exactly_at_threshold(self):
        text = " ".join(["word"] * 3400)
        actual, min_threshold, ok = check_chapter_length(text, target_words=4000, min_pct=85)
        assert actual == 3400
        assert min_threshold == 3400
        assert ok is True

    def test_one_below_threshold(self):
        text = " ".join(["word"] * 3399)
        actual, _, ok = check_chapter_length(text, target_words=4000, min_pct=85)
        assert actual == 3399
        assert ok is False

    def test_custom_min_pct(self):
        text = " ".join(["word"] * 3500)
        _, min_threshold, ok = check_chapter_length(text, target_words=4000, min_pct=90)
        assert min_threshold == 3600
        assert ok is False

    def test_empty_text(self):
        actual, _, ok = check_chapter_length("", target_words=4000, min_pct=85)
        # "".split() returns [], len == 0
        assert actual == 0
        assert ok is False


# ---------------------------------------------------------------------------
# build_chapter_expansion_prompt — prompt builder
# ---------------------------------------------------------------------------

class TestBuildChapterExpansionPrompt:
    """Validate the expansion prompt builder returns well-formed messages."""

    def test_returns_message_list(self):
        msgs = build_chapter_expansion_prompt(
            chapter_text="Short chapter text.",
            current_words=3,
            target_words=4000,
            min_words=3400,
        )
        assert isinstance(msgs, list)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_includes_word_counts_in_user_prompt(self):
        msgs = build_chapter_expansion_prompt(
            chapter_text="Hello world",
            current_words=2,
            target_words=4000,
            min_words=3400,
        )
        user_content = msgs[1]["content"]
        assert "3,400" in user_content
        assert "4,000" in user_content

    def test_includes_chapter_text(self):
        text = "Once upon a time there was a brave knight."
        msgs = build_chapter_expansion_prompt(
            chapter_text=text,
            current_words=9,
            target_words=4000,
            min_words=3400,
        )
        assert text in msgs[1]["content"]


# ---------------------------------------------------------------------------
# expand_chapter — expansion agent retry logic
# ---------------------------------------------------------------------------

class TestExpandChapter:
    """Test the expansion retry logic with mocked LLM calls."""

    def test_successful_expansion(self, monkeypatch):
        """When LLM returns longer text, expansion succeeds."""
        expanded_text = " ".join(["word"] * 4000)

        monkeypatch.setattr(
            "novelforge.agents.chapter._helpers.call_llm",
            lambda msgs, action="": expanded_text,
        )
        result_text, result_wc = expand_chapter(
            "short text",
            target_words=4000,
            min_words=3400,
            chapter_num=1,
            title="Test",
            max_attempts=2,
        )
        assert result_wc == 4000
        assert result_text == expanded_text

    def test_expansion_stops_when_threshold_met(self, monkeypatch):
        """If first attempt meets threshold, second attempt is not called."""
        call_count = [0]

        def mock_llm(msgs, action=""):
            call_count[0] += 1
            return " ".join(["word"] * 4000)

        monkeypatch.setattr(
            "novelforge.agents.chapter._helpers.call_llm",
            mock_llm,
        )
        expand_chapter(
            "short text",
            target_words=4000,
            min_words=3400,
            chapter_num=1,
            title="Test",
            max_attempts=3,
        )
        assert call_count[0] == 1

    def test_expansion_retries_on_no_increase(self, monkeypatch):
        """If expansion doesn't increase length, it stops early and keeps original."""
        call_count = [0]

        def mock_llm(msgs, action=""):
            call_count[0] += 1
            return "still short"  # same 2 words as input

        monkeypatch.setattr(
            "novelforge.agents.chapter._helpers.call_llm",
            mock_llm,
        )
        original = "short text"
        result_text, result_wc = expand_chapter(
            original,
            target_words=4000,
            min_words=3400,
            chapter_num=1,
            title="Test",
            max_attempts=3,
        )
        # Should try once then stop since word count didn't increase
        assert call_count[0] == 1
        # Original text is preserved since expansion didn't improve things
        assert result_text == original

    def test_expansion_handles_llm_failure(self, monkeypatch):
        """If LLM call fails, returns original text."""
        def mock_llm(msgs, action=""):
            raise RuntimeError("API down")

        monkeypatch.setattr(
            "novelforge.agents.chapter._helpers.call_llm",
            mock_llm,
        )
        original = "short text with few words"
        result_text, result_wc = expand_chapter(
            original,
            target_words=4000,
            min_words=3400,
            chapter_num=1,
            title="Test",
            max_attempts=2,
        )
        assert result_text == original
        assert result_wc == len(original.split())

    def test_zero_max_attempts_returns_original(self, monkeypatch):
        """If max_attempts=0, no expansion is attempted."""
        call_count = [0]

        def mock_llm(msgs, action=""):
            call_count[0] += 1
            return " ".join(["word"] * 4000)

        monkeypatch.setattr(
            "novelforge.agents.chapter._helpers.call_llm",
            mock_llm,
        )
        result_text, result_wc = expand_chapter(
            "short",
            target_words=4000,
            min_words=3400,
            chapter_num=1,
            title="Test",
            max_attempts=0,
        )
        assert call_count[0] == 0
        assert result_text == "short"

    def test_gradual_expansion_across_attempts(self, monkeypatch):
        """Multiple expansion attempts can progressively increase length."""
        attempt = [0]

        def mock_llm(msgs, action=""):
            attempt[0] += 1
            if attempt[0] == 1:
                return " ".join(["word"] * 2500)  # still under threshold
            return " ".join(["word"] * 4000)  # now over threshold

        monkeypatch.setattr(
            "novelforge.agents.chapter._helpers.call_llm",
            mock_llm,
        )
        result_text, result_wc = expand_chapter(
            "short",
            target_words=4000,
            min_words=3400,
            chapter_num=1,
            title="Test",
            max_attempts=3,
        )
        assert attempt[0] == 2
        assert result_wc == 4000


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

class TestLengthEnforcementConfig:
    """Verify config constants exist and have sensible defaults."""

    def test_chapter_min_length_pct_default(self, monkeypatch):
        import importlib
        import novelforge.config as cfg
        monkeypatch.delenv("CHAPTER_MIN_LENGTH_PCT", raising=False)
        importlib.reload(cfg)
        try:
            assert cfg.CHAPTER_MIN_LENGTH_PCT == 85
        finally:
            importlib.reload(cfg)

    def test_max_expansion_attempts_default(self, monkeypatch):
        import importlib
        import novelforge.config as cfg
        monkeypatch.delenv("MAX_EXPANSION_ATTEMPTS", raising=False)
        importlib.reload(cfg)
        try:
            assert cfg.MAX_EXPANSION_ATTEMPTS == 2
        finally:
            importlib.reload(cfg)

    def test_chapter_min_length_pct_in_all(self):
        import novelforge.config as cfg
        assert "CHAPTER_MIN_LENGTH_PCT" in cfg.__all__

    def test_max_expansion_attempts_in_all(self):
        import novelforge.config as cfg
        assert "MAX_EXPANSION_ATTEMPTS" in cfg.__all__

    def test_config_shim_exports_new_constants(self):
        import config as shim
        assert hasattr(shim, "CHAPTER_MIN_LENGTH_PCT")
        assert hasattr(shim, "MAX_EXPANSION_ATTEMPTS")


# ---------------------------------------------------------------------------
# ProgressState schema
# ---------------------------------------------------------------------------

class TestProgressStateLengthField:
    """Verify the length_enforcement field is present in ProgressState."""

    def test_length_enforcement_in_typed_dict(self):
        from novelforge.progress import ProgressState
        annotations = ProgressState.__annotations__
        assert "length_enforcement" in annotations


# ---------------------------------------------------------------------------
# Expansion prompt in prompts.yml
# ---------------------------------------------------------------------------

class TestExpansionPromptExists:
    """Verify the expansion prompt template loads correctly."""

    def test_chapter_expansion_prompt_loads(self):
        from novelforge.llm.prompts import _load_prompts
        prompts = _load_prompts()
        assert "chapter_expansion" in prompts

    def test_chapter_expansion_prompt_has_required_fields(self):
        from novelforge.llm.prompts import _load_prompts
        prompts = _load_prompts()
        entry = prompts["chapter_expansion"]
        assert "system" in entry
        assert "user" in entry
        assert "name" in entry

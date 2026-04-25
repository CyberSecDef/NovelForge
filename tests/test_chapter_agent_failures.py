"""
Tests for chapter-agent pass failure observability.

Verifies that:
- Optional-pass failures are logged with structured context (pass name,
  chapter number/title, optional/required kind, exception type and message).
- Optional-pass failures return typed fallbacks (empty string or empty dict
  carrying PASS_FAILURE_KEY) rather than silently disappearing.
- The degraded_passes collector is populated when a failure occurs.
- Dict-returning helpers embed PASS_FAILURE_KEY in their fallback result.
- Str-returning helpers still return "" on failure (backward-compatible).
- Required-pass failures (those that propagate) are not swallowed.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from novelforge.agents.chapter import (
    PASS_FAILURE_KEY,
    run_chapter_rhythm_classifier,
    run_character_state_updater,
    run_continuity_gatekeeper,
    run_per_chapter_compression_check,
    run_scene_variety_compression_auditor,
    _run_all_chapter_agents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHAPTER_KWARGS_STR = dict(
    chapter_num=3,
    chapter_title="The Crossing",
    chapter_summary="Heroes cross the bridge.",
    previous_summaries="Chapter 1: Setup.\nChapter 2: Conflict.",
)


def _raise_runtime(msg: str):
    """Return a stub callable that raises RuntimeError(msg)."""
    def _stub(*args, **kwargs):
        raise RuntimeError(msg)
    return _stub


def _patch_chapter_call_llm(monkeypatch, stub):
    """Patch call_llm in all chapter submodules."""
    import novelforge.agents.chapter as chap
    import novelforge.agents.chapter._helpers as chap_helpers
    import novelforge.agents.chapter.pipeline as chap_pipeline
    monkeypatch.setattr(chap, "call_llm", stub)
    monkeypatch.setattr(chap_helpers, "call_llm", stub)
    monkeypatch.setattr(chap_pipeline, "call_llm", stub)


# ---------------------------------------------------------------------------
# Optional-pass failures — structured logging
# ---------------------------------------------------------------------------


class TestOptionalPassStructuredLogging:
    """Each optional runner must emit a structured WARNING when it fails."""

    def _assert_warning_logged(
        self,
        caplog: pytest.LogCaptureFixture,
        pass_name: str,
        chapter_num: int,
        exc_type: str,
        exc_msg: str,
    ) -> None:
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, f"No WARNING emitted for pass {pass_name!r}"
        text = warnings[-1].getMessage()
        assert pass_name in text, f"Pass name missing from log: {text!r}"
        assert str(chapter_num) in text, f"Chapter num missing from log: {text!r}"
        assert exc_type in text, f"Exception type missing from log: {text!r}"
        assert exc_msg in text, f"Exception message missing from log: {text!r}"
        assert "optional" in text, f"'optional' kind missing from log: {text!r}"

    def test_continuity_gatekeeper_logs_on_failure(self, monkeypatch, caplog):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("lm boom"))
        with caplog.at_level(logging.WARNING, logger="novelforge.agents.chapter"):
            run_continuity_gatekeeper(**_CHAPTER_KWARGS_STR)
        self._assert_warning_logged(
            caplog, "continuity gatekeeper", 3, "RuntimeError", "lm boom"
        )

    def test_scene_variety_auditor_logs_on_failure(self, monkeypatch, caplog):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("audit boom"))
        with caplog.at_level(logging.WARNING, logger="novelforge.agents.chapter"):
            run_scene_variety_compression_auditor(
                chapter_text="text", chapter_summary="summary",
                chapter_num=3, title="My Novel",
            )
        self._assert_warning_logged(
            caplog, "scene variety & compression auditor", 3, "RuntimeError", "audit boom"
        )

    def test_character_state_updater_logs_on_failure(self, monkeypatch, caplog):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("state boom"))
        with caplog.at_level(logging.WARNING, logger="novelforge.agents.chapter"):
            run_character_state_updater(
                chapter_text="text", chapter_summary="summary",
                characters_text="chars", chapter_num=3, title="My Novel",
            )
        self._assert_warning_logged(
            caplog, "character state updater", 3, "RuntimeError", "state boom"
        )

    def test_compression_check_logs_on_failure(self, monkeypatch, caplog):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("comp boom"))
        with caplog.at_level(logging.WARNING, logger="novelforge.agents.chapter"):
            run_per_chapter_compression_check(
                chapter_num=3, chapter_summary="summary",
                previous_summaries="Chapter 1: x.", title="My Novel",
            )
        self._assert_warning_logged(
            caplog, "per-chapter compression check", 3, "RuntimeError", "comp boom"
        )

    def test_rhythm_classifier_logs_on_failure(self, monkeypatch, caplog):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("rhythm boom"))
        with caplog.at_level(logging.WARNING, logger="novelforge.agents.chapter"):
            run_chapter_rhythm_classifier(**_CHAPTER_KWARGS_STR, title="My Novel")
        self._assert_warning_logged(
            caplog, "chapter rhythm classifier", 3, "RuntimeError", "rhythm boom"
        )


# ---------------------------------------------------------------------------
# Optional-pass failures — typed fallback values
# ---------------------------------------------------------------------------


class TestOptionalPassFallbackValues:
    """Optional runner helpers must return appropriate typed fallbacks on failure."""

    def test_continuity_gatekeeper_returns_empty_string(self, monkeypatch):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("x"))
        result = run_continuity_gatekeeper(**_CHAPTER_KWARGS_STR)
        assert result == ""

    def test_scene_variety_auditor_returns_empty_string(self, monkeypatch):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("x"))
        result = run_scene_variety_compression_auditor(
            chapter_text="t", chapter_summary="s", chapter_num=3, title="Novel",
        )
        assert result == ""

    def test_character_state_updater_returns_empty_tuple(self, monkeypatch):
        """The state updater now returns ``(state_log, ledger_updates)`` —
        on failure both elements are empty."""
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("x"))
        result = run_character_state_updater(
            chapter_text="t", chapter_summary="s",
            characters_text="c", chapter_num=3, title="Novel",
        )
        assert result == ("", [])

    def test_compression_check_returns_empty_string(self, monkeypatch):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("x"))
        result = run_per_chapter_compression_check(
            chapter_num=3, chapter_summary="s",
            previous_summaries="Chapter 1: x.", title="Novel",
        )
        assert result == ""

    def test_rhythm_classifier_returns_dict_with_pass_failure_key(self, monkeypatch):
        """Dict-returning helpers must embed PASS_FAILURE_KEY in their fallback."""
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("boom"))
        result = run_chapter_rhythm_classifier(**_CHAPTER_KWARGS_STR, title="Novel")
        assert isinstance(result, dict)
        assert PASS_FAILURE_KEY in result
        assert "RuntimeError" in result[PASS_FAILURE_KEY]
        assert "boom" in result[PASS_FAILURE_KEY]

    def test_rhythm_classifier_fallback_is_otherwise_empty(self, monkeypatch):
        """The fallback dict must carry PASS_FAILURE_KEY and safe-default business keys."""
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("x"))
        result = run_chapter_rhythm_classifier(**_CHAPTER_KWARGS_STR, title="Novel")
        assert PASS_FAILURE_KEY in result
        # Minimal safe defaults must be present so callers can .get() without KeyError
        assert "recommended_shape_for_this_chapter" in result
        assert "recommendation_reason" in result
        # No spurious keys beyond the three above
        assert set(result.keys()) == {
            PASS_FAILURE_KEY,
            "recommended_shape_for_this_chapter",
            "recommendation_reason",
        }


# ---------------------------------------------------------------------------
# Optional-pass failures — degraded_passes collector
# ---------------------------------------------------------------------------


class TestDegradedPassesCollector:
    """When degraded_passes is provided, failures must be appended to it."""

    def test_continuity_gatekeeper_populates_degraded_passes(self, monkeypatch):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("fail"))
        degraded: list[dict] = []
        run_continuity_gatekeeper(**_CHAPTER_KWARGS_STR, degraded_passes=degraded)
        assert len(degraded) == 1
        entry = degraded[0]
        assert entry["pass_name"] == "continuity gatekeeper"
        assert entry["chapter_num"] == 3
        assert "RuntimeError" in entry["failure_summary"]

    def test_rhythm_classifier_populates_degraded_passes(self, monkeypatch):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("fail"))
        degraded: list[dict] = []
        run_chapter_rhythm_classifier(**_CHAPTER_KWARGS_STR, title="Novel", degraded_passes=degraded)
        assert len(degraded) == 1
        entry = degraded[0]
        assert entry["pass_name"] == "chapter rhythm classifier"
        assert entry["chapter_num"] == 3

    def test_character_state_updater_populates_degraded_passes(self, monkeypatch):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("fail"))
        degraded: list[dict] = []
        run_character_state_updater(
            chapter_text="t", chapter_summary="s",
            characters_text="c", chapter_num=3, title="Novel",
            degraded_passes=degraded,
        )
        assert len(degraded) == 1
        assert degraded[0]["pass_name"] == "character state updater"

    def test_compression_check_populates_degraded_passes(self, monkeypatch):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("fail"))
        degraded: list[dict] = []
        run_per_chapter_compression_check(
            chapter_num=3, chapter_summary="s",
            previous_summaries="Chapter 1: x.", title="Novel",
            degraded_passes=degraded,
        )
        assert len(degraded) == 1
        assert degraded[0]["pass_name"] == "per-chapter compression check"

    def test_scene_variety_auditor_populates_degraded_passes(self, monkeypatch):
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("fail"))
        degraded: list[dict] = []
        run_scene_variety_compression_auditor(
            chapter_text="t", chapter_summary="s", chapter_num=3, title="Novel",
            degraded_passes=degraded,
        )
        assert len(degraded) == 1
        assert degraded[0]["pass_name"] == "scene variety & compression auditor"

    def test_no_degraded_passes_when_none_provided(self, monkeypatch):
        """Passing degraded_passes=None must not raise; it simply skips collection."""
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("x"))
        result = run_continuity_gatekeeper(**_CHAPTER_KWARGS_STR, degraded_passes=None)
        assert result == ""

    def test_degraded_passes_not_populated_on_success(self, monkeypatch):
        """A successful pass must NOT append anything to degraded_passes."""
        _patch_chapter_call_llm(monkeypatch, lambda *a, **kw: "gatekeeper result")
        degraded: list[dict] = []
        result = run_continuity_gatekeeper(**_CHAPTER_KWARGS_STR, degraded_passes=degraded)
        assert result == "gatekeeper result"
        assert degraded == []

    def test_multiple_failures_accumulate_in_collector(self, monkeypatch):
        """Each failing pass appends its own record; the list grows cumulatively."""
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("x"))
        degraded: list[dict] = []
        run_continuity_gatekeeper(**_CHAPTER_KWARGS_STR, degraded_passes=degraded)
        run_character_state_updater(
            chapter_text="t", chapter_summary="s",
            characters_text="c", chapter_num=3, title="Novel",
            degraded_passes=degraded,
        )
        assert len(degraded) == 2
        pass_names = {e["pass_name"] for e in degraded}
        assert "continuity gatekeeper" in pass_names
        assert "character state updater" in pass_names


# ---------------------------------------------------------------------------
# Required-pass failure — propagation (pipeline-critical steps)
# ---------------------------------------------------------------------------


class TestRequiredPassPropagation:
    """
    The main chapter-agent pipeline (_run_all_chapter_agents) uses required passes
    that must propagate rather than silently degrade.  An LLM failure on a
    required step (e.g., prose refinement) must raise, not return empty text.
    """

    def test_pipeline_propagates_llm_error_on_required_step(self, monkeypatch):
        """An LLM failure inside _run_all_chapter_agents must propagate to the caller."""
        _patch_chapter_call_llm(monkeypatch, _raise_runtime("hard fail"))
        with pytest.raises(RuntimeError, match="hard fail"):
            _run_all_chapter_agents(
                text="Chapter text.",
                chapter_num=1,
                title="Test Novel",
                genre="Thriller",
                total_chapters=3,
                chapter_outline_summary="A begins.",
                characters_text="- Alice: protagonist",
                previous_summaries="",
            )

    def test_pipeline_timeout_propagates(self, monkeypatch):
        """A ChapterTimeoutError must propagate out of _run_all_chapter_agents."""
        from novelforge.llm.client import ChapterTimeoutError

        def _raise_timeout(*args, **kwargs):
            raise ChapterTimeoutError("timed out")

        _patch_chapter_call_llm(monkeypatch, _raise_timeout)
        with pytest.raises(ChapterTimeoutError):
            _run_all_chapter_agents(
                text="Chapter text.",
                chapter_num=2,
                title="Test Novel",
                genre="Drama",
                total_chapters=5,
                chapter_outline_summary="B escalates.",
                characters_text="- Bob: antagonist",
                previous_summaries="",
            )

    def test_pipeline_passes_degraded_passes_for_optional_auditor(self, monkeypatch):
        """
        The optional scene-variety auditor inside the pipeline must propagate its
        failure into degraded_passes without halting the pipeline.
        We stub all required LLM calls to succeed but make the auditor call fail
        by having call_llm raise on the audit-specific action string.
        """
        def _selective_llm(messages, *, action="", **kwargs):
            # Fail the optional auditor; let all required passes return dummy text
            if "scene variety" in action.lower():
                raise RuntimeError("auditor boom")
            return "dummy chapter output"

        _patch_chapter_call_llm(monkeypatch, _selective_llm)

        degraded: list[dict] = []
        text_out, summary_out = _run_all_chapter_agents(
            text="Chapter text.",
            chapter_num=1,
            title="Test Novel",
            genre="Sci-Fi",
            total_chapters=3,
            chapter_outline_summary="X happens.",
            characters_text="- Alice: lead",
            previous_summaries="",
            degraded_passes=degraded,
        )
        assert text_out  # pipeline produced output
        assert summary_out is not None
        # The optional auditor failure must be recorded
        auditor_failures = [e for e in degraded if e["pass_name"] == "scene variety & compression auditor"]
        assert len(auditor_failures) == 1
        assert "auditor boom" in auditor_failures[0]["failure_summary"]


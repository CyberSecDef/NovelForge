"""
Tests for the four new chapter-generation pipeline prompt builders and their
pipeline integration under the shared ``mock_llm`` fixture.

Covers:
- build_voice_dialogue_differentiation_prompt
- build_human_oddities_prompt
- build_metaphor_reduction_prompt
- build_copy_edit_prompt
- Pipeline integration: all four steps execute and produce non-empty output
  when wired through ``_run_all_chapter_agents`` with ``mock_llm``.
"""

from __future__ import annotations

import pytest

from novelforge.agents.chapter.prompts import (
    build_copy_edit_prompt,
    build_human_oddities_prompt,
    build_metaphor_reduction_prompt,
    build_voice_dialogue_differentiation_prompt,
)
from novelforge.agents.chapter import _run_all_chapter_agents

# Import the canned response helper so integration tests can delegate to it
from tests.conftest import _canned_llm_response


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_CHAPTER_TEXT = (
    "The detective walked through the rain-soaked streets, "
    "her coat pulled tight against the wind."
)
_CHAPTER_NUM = 2
_TITLE = "Shadows of the City"
_CHARACTERS = "Alice — protagonist, seasoned detective\nBob — informant, nervous disposition"
_TOTAL_CHAPTERS = 5


# ---------------------------------------------------------------------------
# build_voice_dialogue_differentiation_prompt
# ---------------------------------------------------------------------------


class TestBuildVoiceDialogueDifferentiationPrompt:
    """Unit tests for the voice & dialogue differentiation prompt builder."""

    def _call(
        self,
        chapter_text: str = _CHAPTER_TEXT,
        chapter_num: int = _CHAPTER_NUM,
        title: str = _TITLE,
        characters_text: str = _CHARACTERS,
        perspective_prompt: str = "",
    ) -> list[dict[str, str]]:
        return build_voice_dialogue_differentiation_prompt(
            chapter_text=chapter_text,
            chapter_num=chapter_num,
            title=title,
            characters_text=characters_text,
            perspective_prompt=perspective_prompt,
        )

    def test_returns_list_of_message_dicts(self):
        result = self._call()
        assert isinstance(result, list)
        assert len(result) >= 1
        for msg in result:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg

    def test_chapter_text_appears_in_prompt(self):
        result = self._call()
        combined = " ".join(m["content"] for m in result)
        assert _CHAPTER_TEXT in combined

    def test_characters_text_appears_in_prompt(self):
        result = self._call()
        combined = " ".join(m["content"] for m in result)
        assert _CHARACTERS in combined

    def test_perspective_prompt_appears_when_provided(self):
        perspective = "Third-person limited, close to Alice."
        result = self._call(perspective_prompt=perspective)
        combined = " ".join(m["content"] for m in result)
        assert perspective in combined

    def test_empty_perspective_prompt_does_not_raise(self):
        result = self._call(perspective_prompt="")
        assert isinstance(result, list)

    def test_deterministic_for_same_inputs(self):
        r1 = self._call()
        r2 = self._call()
        assert r1 == r2


# ---------------------------------------------------------------------------
# build_human_oddities_prompt
# ---------------------------------------------------------------------------


class TestBuildHumanOdditiesPrompt:
    """Unit tests for the human oddities prompt builder."""

    def _call(
        self,
        chapter_text: str = _CHAPTER_TEXT,
        chapter_num: int = _CHAPTER_NUM,
        title: str = _TITLE,
        total_chapters: int = _TOTAL_CHAPTERS,
        characters_text: str = _CHARACTERS,
    ) -> list[dict[str, str]]:
        return build_human_oddities_prompt(
            chapter_text=chapter_text,
            chapter_num=chapter_num,
            title=title,
            total_chapters=total_chapters,
            characters_text=characters_text,
        )

    def test_returns_list_of_message_dicts(self):
        result = self._call()
        assert isinstance(result, list)
        assert len(result) >= 1
        for msg in result:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg

    def test_chapter_text_appears_in_prompt(self):
        result = self._call()
        combined = " ".join(m["content"] for m in result)
        assert _CHAPTER_TEXT in combined

    def test_characters_text_appears_in_prompt(self):
        result = self._call()
        combined = " ".join(m["content"] for m in result)
        assert _CHARACTERS in combined

    def test_deterministic_for_same_inputs(self):
        r1 = self._call()
        r2 = self._call()
        assert r1 == r2

    def test_different_total_chapters_changes_output(self):
        r1 = self._call(total_chapters=3)
        r2 = self._call(total_chapters=20)
        # At least one message must differ (chapter position context changes)
        assert r1 != r2


# ---------------------------------------------------------------------------
# build_metaphor_reduction_prompt
# ---------------------------------------------------------------------------


class TestBuildMetaphorReductionPrompt:
    """Unit tests for the metaphor reduction prompt builder."""

    def _call(
        self,
        chapter_text: str = _CHAPTER_TEXT,
        chapter_num: int = _CHAPTER_NUM,
        title: str = _TITLE,
    ) -> list[dict[str, str]]:
        return build_metaphor_reduction_prompt(
            chapter_text=chapter_text,
            chapter_num=chapter_num,
            title=title,
        )

    def test_returns_list_of_message_dicts(self):
        result = self._call()
        assert isinstance(result, list)
        assert len(result) >= 1
        for msg in result:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg

    def test_chapter_text_appears_in_prompt(self):
        result = self._call()
        combined = " ".join(m["content"] for m in result)
        assert _CHAPTER_TEXT in combined

    def test_deterministic_for_same_inputs(self):
        r1 = self._call()
        r2 = self._call()
        assert r1 == r2

    def test_different_titles_produce_different_prompts(self):
        r1 = self._call(title="Title One")
        r2 = self._call(title="Title Two")
        assert r1 != r2


# ---------------------------------------------------------------------------
# build_copy_edit_prompt
# ---------------------------------------------------------------------------


class TestBuildCopyEditPrompt:
    """Unit tests for the copy-edit prompt builder."""

    def _call(
        self,
        chapter_text: str = _CHAPTER_TEXT,
        chapter_num: int = _CHAPTER_NUM,
        title: str = _TITLE,
    ) -> list[dict[str, str]]:
        return build_copy_edit_prompt(
            chapter_text=chapter_text,
            chapter_num=chapter_num,
            title=title,
        )

    def test_returns_list_of_message_dicts(self):
        result = self._call()
        assert isinstance(result, list)
        assert len(result) >= 1
        for msg in result:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg

    def test_chapter_text_appears_in_prompt(self):
        result = self._call()
        combined = " ".join(m["content"] for m in result)
        assert _CHAPTER_TEXT in combined

    def test_deterministic_for_same_inputs(self):
        r1 = self._call()
        r2 = self._call()
        assert r1 == r2

    def test_different_chapter_numbers_produce_different_prompts(self):
        r1 = self._call(chapter_num=1)
        r2 = self._call(chapter_num=5)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Pipeline integration: all four steps execute under mock_llm
# ---------------------------------------------------------------------------


class TestNewStepsPipelineIntegration:
    """
    Verify that all four new pipeline steps are wired into _run_all_chapter_agents
    and execute without error under the shared mock_llm fixture.

    Strategy: additionally patch ``novelforge.agents.chapter._helpers.call_llm``
    with a recording wrapper so we can inspect which ``action`` strings were
    passed, then assert each new step is present.
    """

    _PIPELINE_KWARGS = dict(
        text=_CHAPTER_TEXT,
        chapter_num=_CHAPTER_NUM,
        title=_TITLE,
        genre="Mystery",
        total_chapters=_TOTAL_CHAPTERS,
        chapter_outline_summary="A detective investigates a cold case.",
        characters_text=_CHARACTERS,
        previous_summaries="Chapter 1: Setup.",
    )

    def _run_and_collect_actions(self, mock_llm, mocker) -> list[str]:
        """Run the full pipeline and return the list of recorded action strings."""
        actions: list[str] = []

        def _recording(messages, *, action: str = "", json_mode: bool = False) -> str:
            actions.append(action)
            return _canned_llm_response(messages, action=action, json_mode=json_mode)

        # Layer a recording wrapper on top of the mock_llm patch for the module
        # that actually executes the calls inside _run_all_chapter_agents.
        mocker.patch(
            "novelforge.agents.chapter._helpers.call_llm",
            side_effect=_recording,
        )
        _run_all_chapter_agents(**self._PIPELINE_KWARGS)
        return actions

    @staticmethod
    def _action_present(actions: list[str], *substrings: str) -> bool:
        """Return True if any recorded action contains all of the given substrings."""
        return any(all(s in a for s in substrings) for a in actions)

    def test_voice_dialogue_step_executes(self, mock_llm, mocker):
        actions = self._run_and_collect_actions(mock_llm, mocker)
        assert self._action_present(actions, "voice", "dialogue"), (
            f"Expected 'voice & dialogue differentiation' action in LLM calls; "
            f"got: {actions}"
        )

    def test_human_oddities_step_executes(self, mock_llm, mocker):
        actions = self._run_and_collect_actions(mock_llm, mocker)
        assert self._action_present(actions, "human oddities"), (
            f"Expected 'human oddities' action in LLM calls; got: {actions}"
        )

    def test_metaphor_reduction_step_executes(self, mock_llm, mocker):
        actions = self._run_and_collect_actions(mock_llm, mocker)
        assert self._action_present(actions, "metaphor"), (
            f"Expected 'metaphor reduction' action in LLM calls; got: {actions}"
        )

    def test_copy_edit_step_executes(self, mock_llm, mocker):
        actions = self._run_and_collect_actions(mock_llm, mocker)
        assert self._action_present(actions, "copy edit"), (
            f"Expected 'copy edit' action in LLM calls; got: {actions}"
        )

    def test_all_four_steps_execute_in_single_run(self, mock_llm, mocker):
        """All four new steps must appear in a single pipeline execution."""
        actions = self._run_and_collect_actions(mock_llm, mocker)
        assert self._action_present(actions, "voice", "dialogue"), "voice & dialogue step missing"
        assert self._action_present(actions, "human oddities"), "human oddities step missing"
        assert self._action_present(actions, "metaphor"), "metaphor reduction step missing"
        assert self._action_present(actions, "copy edit"), "copy edit step missing"

    def test_pipeline_returns_non_empty_text_and_summary(self, mock_llm, mocker):
        """Pipeline must return a non-empty (text, summary) tuple."""
        mocker.patch(
            "novelforge.agents.chapter._helpers.call_llm",
            side_effect=_canned_llm_response,
        )
        text, summary = _run_all_chapter_agents(**self._PIPELINE_KWARGS)
        assert isinstance(text, str) and text.strip()
        assert isinstance(summary, str) and summary.strip()

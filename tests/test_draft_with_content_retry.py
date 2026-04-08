"""
Tests for the _draft_with_content_retry helper.

Verifies that:
- A successful first attempt returns the LLM response immediately.
- On ContentRejectionError the helper retries with a content-guidance note
  appended to special_instructions.
- The content note is combined with special_instructions correctly when
  special_instructions is non-empty vs. empty.
- After max_attempts the ContentRejectionError is re-raised.
- The chapter_num is logged in the warning message on retry.
"""

from __future__ import annotations

import logging
from unittest.mock import call, MagicMock

import pytest

import novelforge.agents.chapter._helpers as chap_helpers
from novelforge.agents.chapter import _draft_with_content_retry, _DRAFT_CONTENT_NOTE
from novelforge.llm.client import ContentRejectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rejection(*args, **kwargs):
    raise ContentRejectionError("content blocked")


def _make_prompt_fn(recorded: list[str]):
    """Return a prompt builder that records the instructions it receives."""
    def _build(instructions: str) -> list[dict]:
        recorded.append(instructions)
        return [{"role": "user", "content": instructions}]
    return _build


# ---------------------------------------------------------------------------
# Success on first attempt
# ---------------------------------------------------------------------------


class TestDraftWithContentRetrySuccess:
    def test_returns_llm_response_on_first_attempt(self, monkeypatch):
        monkeypatch.setattr(chap_helpers, "call_llm", lambda msgs, *, action: "draft text")
        recorded: list[str] = []
        result = _draft_with_content_retry(
            _make_prompt_fn(recorded),
            action="Chapter 1: drafting",
            special_instructions="Write dark themes.",
            chapter_num=1,
        )
        assert result == "draft text"
        assert len(recorded) == 1
        assert recorded[0] == "Write dark themes."

    def test_prompt_fn_receives_empty_instructions(self, monkeypatch):
        monkeypatch.setattr(chap_helpers, "call_llm", lambda msgs, *, action: "ok")
        recorded: list[str] = []
        _draft_with_content_retry(
            _make_prompt_fn(recorded),
            action="Ch 2: drafting",
            special_instructions="",
            chapter_num=2,
        )
        assert recorded[0] == ""


# ---------------------------------------------------------------------------
# Retry behaviour on ContentRejectionError
# ---------------------------------------------------------------------------


class TestDraftWithContentRetryOnRejection:
    def test_retries_once_on_rejection_then_succeeds(self, monkeypatch):
        """First call raises ContentRejectionError; second call succeeds."""
        call_count = 0

        def _selective_llm(msgs, *, action):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContentRejectionError("blocked")
            return "retry draft"

        monkeypatch.setattr(chap_helpers, "call_llm", _selective_llm)
        recorded: list[str] = []
        result = _draft_with_content_retry(
            _make_prompt_fn(recorded),
            action="Ch 3: drafting",
            special_instructions="My instructions.",
            chapter_num=3,
        )
        assert result == "retry draft"
        assert call_count == 2
        # First call: original instructions; second call: with content note appended
        assert recorded[0] == "My instructions."
        assert recorded[1] == f"My instructions.\n\n{_DRAFT_CONTENT_NOTE}"

    def test_content_note_used_alone_when_no_special_instructions(self, monkeypatch):
        """When special_instructions is empty, the retry uses only the content note."""
        call_count = 0

        def _selective_llm(msgs, *, action):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContentRejectionError("blocked")
            return "ok"

        monkeypatch.setattr(chap_helpers, "call_llm", _selective_llm)
        recorded: list[str] = []
        _draft_with_content_retry(
            _make_prompt_fn(recorded),
            action="Ch 4: drafting",
            special_instructions="",
            chapter_num=4,
        )
        assert recorded[1] == _DRAFT_CONTENT_NOTE

    def test_raises_after_max_attempts_exhausted(self, monkeypatch):
        """ContentRejectionError is re-raised once max_attempts is reached."""
        monkeypatch.setattr(chap_helpers, "call_llm", _make_rejection)
        with pytest.raises(ContentRejectionError):
            _draft_with_content_retry(
                _make_prompt_fn([]),
                action="Ch 5: drafting",
                special_instructions="instr",
                chapter_num=5,
                max_attempts=3,
            )

    def test_exact_attempt_count_matches_max_attempts(self, monkeypatch):
        """call_llm is called exactly max_attempts times before re-raising."""
        call_count = 0

        def _always_reject(msgs, *, action):
            nonlocal call_count
            call_count += 1
            raise ContentRejectionError("always blocked")

        monkeypatch.setattr(chap_helpers, "call_llm", _always_reject)
        with pytest.raises(ContentRejectionError):
            _draft_with_content_retry(
                _make_prompt_fn([]),
                action="Ch 6: drafting",
                special_instructions="",
                chapter_num=6,
                max_attempts=2,
            )
        assert call_count == 2

    def test_warning_logged_on_retry(self, monkeypatch, caplog):
        """A WARNING must be logged on each retry, including the chapter number."""
        call_count = 0

        def _fail_once(msgs, *, action):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContentRejectionError("blocked")
            return "ok"

        monkeypatch.setattr(chap_helpers, "call_llm", _fail_once)
        with caplog.at_level(logging.WARNING, logger="novelforge.agents.chapter._helpers"):
            _draft_with_content_retry(
                _make_prompt_fn([]),
                action="Ch 7: drafting",
                special_instructions="",
                chapter_num=7,
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "Expected at least one WARNING to be emitted on retry"
        text = warnings[-1].getMessage()
        assert "7" in text, f"Chapter number missing from warning: {text!r}"

    def test_custom_max_attempts_respected(self, monkeypatch):
        """max_attempts=1 means a single attempt with no retries."""
        monkeypatch.setattr(chap_helpers, "call_llm", _make_rejection)
        with pytest.raises(ContentRejectionError):
            _draft_with_content_retry(
                _make_prompt_fn([]),
                action="Ch 8: drafting",
                special_instructions="",
                chapter_num=8,
                max_attempts=1,
            )

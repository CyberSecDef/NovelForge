"""
Tests for BaseAgent.plan() exception-handling contract.

Operational failures (LLM errors, JSON parse failures, payload validation
errors, content policy rejections) must degrade gracefully to fallback output.

Programmer-error exceptions (AttributeError, TypeError, KeyError, …) must
propagate so they surface as bugs rather than silent fallbacks.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from novelforge.agents.base import BaseAgent
from novelforge.llm.client import ContentRejectionError


# ---------------------------------------------------------------------------
# Minimal concrete BaseAgent for testing
# ---------------------------------------------------------------------------


class _SimpleAgent(BaseAgent):
    """Minimal concrete implementation of BaseAgent for unit testing."""

    name = "TestAgent"
    prompt_action = "Testing"

    def build_prompt(self, **ctx) -> list[dict]:
        return [{"role": "user", "content": "test prompt"}]

    def normalise(self, data: dict, **ctx) -> dict:
        return data

    def build_fallback(self, **ctx) -> dict:
        return {"fallback": True}

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        return ""


# ---------------------------------------------------------------------------
# Operational failures → fallback
# ---------------------------------------------------------------------------


class TestOperationalFailuresFallback:
    """Operational failures must degrade gracefully to build_fallback()."""

    def test_runtime_error_falls_back(self) -> None:
        """A RuntimeError from call_llm (e.g. circuit-breaker) triggers fallback."""
        agent = _SimpleAgent()
        with patch("novelforge.agents.base.call_llm", side_effect=RuntimeError("LLM unreachable")):
            result = agent.plan()
        assert result == {"fallback": True}

    def test_json_decode_error_falls_back(self) -> None:
        """An unparseable LLM response triggers fallback."""
        agent = _SimpleAgent()
        with patch("novelforge.agents.base.call_llm", return_value="not-json"):
            result = agent.plan()
        assert result == {"fallback": True}

    def test_value_error_in_normalise_falls_back(self) -> None:
        """A ValueError raised by normalise() (bad LLM payload) triggers fallback."""
        agent = _SimpleAgent()

        def _bad_normalise(data: dict, **ctx) -> dict:
            raise ValueError("payload missing required key")

        agent.normalise = _bad_normalise  # type: ignore[method-assign]
        with patch("novelforge.agents.base.call_llm", return_value='{"ok": true}'):
            result = agent.plan()
        assert result == {"fallback": True}

    def test_content_rejection_error_falls_back(self) -> None:
        """A ContentRejectionError (content-policy block) triggers fallback."""
        agent = _SimpleAgent()
        with patch(
            "novelforge.agents.base.call_llm",
            side_effect=ContentRejectionError("blocked by policy", status_code=400),
        ):
            result = agent.plan()
        assert result == {"fallback": True}

    def test_json_decode_error_from_parse_falls_back(self) -> None:
        """A json.JSONDecodeError raised directly triggers fallback."""
        agent = _SimpleAgent()
        with patch(
            "novelforge.agents.base.call_llm",
            side_effect=json.JSONDecodeError("expecting value", "", 0),
        ):
            result = agent.plan()
        assert result == {"fallback": True}


# ---------------------------------------------------------------------------
# Programmer errors → re-raised
# ---------------------------------------------------------------------------


class TestProgrammerErrorsPropagated:
    """Unexpected coding errors must not be silently converted to fallback output."""

    def test_attribute_error_in_build_prompt_propagates(self) -> None:
        """An AttributeError in build_prompt() must not be swallowed."""
        agent = _SimpleAgent()

        def _broken_build_prompt(**ctx) -> list[dict]:
            raise AttributeError("missing attribute")

        agent.build_prompt = _broken_build_prompt  # type: ignore[method-assign]
        with pytest.raises(AttributeError, match="missing attribute"):
            agent.plan()

    def test_type_error_in_build_prompt_propagates(self) -> None:
        """A TypeError in build_prompt() must not be swallowed."""
        agent = _SimpleAgent()

        def _broken_build_prompt(**ctx) -> list[dict]:
            raise TypeError("wrong type")

        agent.build_prompt = _broken_build_prompt  # type: ignore[method-assign]
        with pytest.raises(TypeError, match="wrong type"):
            agent.plan()

    def test_key_error_in_normalise_propagates(self) -> None:
        """A KeyError in normalise() must not be swallowed."""
        agent = _SimpleAgent()

        def _broken_normalise(data: dict, **ctx) -> dict:
            raise KeyError("missing_key")

        agent.normalise = _broken_normalise  # type: ignore[method-assign]
        with patch("novelforge.agents.base.call_llm", return_value='{"ok": true}'):
            with pytest.raises(KeyError, match="missing_key"):
                agent.plan()

    def test_attribute_error_in_normalise_propagates(self) -> None:
        """An AttributeError in normalise() must not be swallowed."""
        agent = _SimpleAgent()

        def _broken_normalise(data: dict, **ctx) -> dict:
            raise AttributeError("programmer mistake")

        agent.normalise = _broken_normalise  # type: ignore[method-assign]
        with patch("novelforge.agents.base.call_llm", return_value='{"ok": true}'):
            with pytest.raises(AttributeError, match="programmer mistake"):
                agent.plan()

    def test_type_error_in_normalise_propagates(self) -> None:
        """A TypeError in normalise() must not be swallowed."""
        agent = _SimpleAgent()

        def _broken_normalise(data: dict, **ctx) -> dict:
            raise TypeError("type mismatch in code")

        agent.normalise = _broken_normalise  # type: ignore[method-assign]
        with patch("novelforge.agents.base.call_llm", return_value='{"ok": true}'):
            with pytest.raises(TypeError, match="type mismatch in code"):
                agent.plan()

"""
Base class for NovelForge planning agents.

All planning agents share a common orchestration pattern:
    build_prompt → call_llm → parse_json → normalise → fallback on error

Subclasses provide the unique logic for each step; BaseAgent handles the
shared try/except/log/fallback flow in ``plan()``.

Planning results always carry a ``_planning_source`` key that indicates the
origin of the returned artifact:

* ``"llm"``      – the LLM call succeeded and the result was normalised.
* ``"fallback"`` – an operational failure occurred; the deterministic
                   ``build_fallback()`` result was used instead.

A ``_failure_summary`` key is also injected on the fallback path so that
callers and tests can inspect the failure without scraping logs.
"""

import json
import logging
from abc import ABC, abstractmethod

from novelforge.llm.client import ContentRejectionError, call_llm, parse_llm_json

logger = logging.getLogger(__name__)

#: Metadata key injected into every dict returned by :meth:`BaseAgent.plan`.
PLANNING_SOURCE_KEY = "_planning_source"

#: Metadata key injected on the fallback path with a brief error description.
PLANNING_FAILURE_KEY = "_failure_summary"

# Exceptions that represent expected operational failures for which the agent
# should degrade gracefully to ``build_fallback()``.
#
# * RuntimeError  – covers CircuitBreakerError, ChapterTimeoutError, and general
#                   LLM call failures.
# * json.JSONDecodeError – the LLM returned text that cannot be parsed as JSON.
# * ValueError    – ``normalise()`` rejected the LLM payload due to bad values.
# * ContentRejectionError – the LLM API refused the request on content-policy
#                           grounds.
#
# Exceptions that are NOT listed here (AttributeError, TypeError, KeyError,
# NameError, IndexError, …) indicate programmer defects and are intentionally
# allowed to propagate so they surface as bugs rather than silent fallbacks.
_OPERATIONAL_ERRORS = (RuntimeError, json.JSONDecodeError, ValueError, ContentRejectionError)


class BaseAgent(ABC):
    """Abstract base class for planning agents."""

    # Subclasses set these
    name: str = "BaseAgent"
    prompt_action: str = "Planning"

    @abstractmethod
    def build_prompt(self, **ctx) -> list[dict]:
        """Build the LLM prompt messages for this agent."""
        ...

    @abstractmethod
    def normalise(self, data: dict, **ctx) -> dict:
        """Clean and validate the parsed LLM response."""
        ...

    @abstractmethod
    def build_fallback(self, **ctx) -> dict:
        """Return a deterministic fallback when the LLM call fails."""
        ...

    @abstractmethod
    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        """Format the plan output as prompt text for a specific chapter."""
        ...

    def plan(self, **ctx) -> dict:
        """
        Shared orchestration: build prompt, call LLM, parse, normalise.

        Falls back to ``build_fallback()`` on expected operational failures
        (LLM errors, JSON parse failures, payload validation errors, content
        policy rejections).  Unexpected programmer errors such as
        ``AttributeError``, ``TypeError``, and ``KeyError`` are intentionally
        re-raised so they surface as bugs rather than silent fallbacks.

        Every returned dict is annotated with a ``_planning_source`` key so
        that callers can tell whether the result came from a live LLM call
        (``"llm"``) or a deterministic fallback (``"fallback"``).  On the
        fallback path a ``_failure_summary`` key is also added with a brief
        description of the originating error.
        """
        try:
            messages = self.build_prompt(**ctx)
            raw = call_llm(messages, action=self.prompt_action, json_mode=True)
            parsed = parse_llm_json(raw)
            result = self.normalise(parsed, **ctx)
            result[PLANNING_SOURCE_KEY] = "llm"
            return result
        except _OPERATIONAL_ERRORS as exc:
            self._log_failure(exc, **ctx)
            result = self.build_fallback(**ctx)
            result[PLANNING_SOURCE_KEY] = "fallback"
            result[PLANNING_FAILURE_KEY] = f"{type(exc).__name__}: {exc}"
            return result

    def _log_failure(self, exc: Exception, **ctx) -> None:
        """Log a structured error entry when this agent fails."""
        title = str(ctx.get("title", ""))
        premise = str(ctx.get("premise", ""))
        genre = str(ctx.get("genre", ""))
        chapter_list = ctx.get("chapter_list", [])
        character_list = ctx.get("character_list", [])
        logger.warning(
            "Planning agent FAILED — agent=%s | genre=%s | chapters=%d | "
            "characters=%d | title=%s | premise=%s | error=%s: %s",
            self.name,
            genre,
            len(chapter_list) if isinstance(chapter_list, list) else 0,
            len(character_list) if isinstance(character_list, list) else 0,
            title[:60],
            (premise[:80] + "…") if len(premise) > 80 else premise,
            type(exc).__name__,
            exc,
        )

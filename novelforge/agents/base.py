"""
Base class for NovelForge planning agents.

All planning agents share a common orchestration pattern:
    build_prompt → call_llm → parse_json → normalise → fallback on error

Subclasses provide the unique logic for each step; BaseAgent handles the
shared try/except/log/fallback flow in ``plan()``.
"""

import json
import logging
from abc import ABC, abstractmethod

from novelforge.llm.client import call_llm, parse_llm_json

logger = logging.getLogger(__name__)


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

        Falls back to ``build_fallback()`` on any error and logs the failure
        with structured context.
        """
        try:
            messages = self.build_prompt(**ctx)
            raw = call_llm(messages, action=self.prompt_action, json_mode=True)
            parsed = parse_llm_json(raw)
            return self.normalise(parsed, **ctx)
        except (RuntimeError, json.JSONDecodeError, TypeError, ValueError, Exception) as exc:
            self._log_failure(exc, **ctx)
            return self.build_fallback(**ctx)

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

"""Prompt loading and rendering from prompts.yml."""

import logging
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError

logger = logging.getLogger(__name__)


_prompts_cache: dict | None = None


_REQUIRED_FIELDS = ("name", "system", "user")


def _validate_prompts(entries: list) -> dict:
    """Validate prompt entries and return a name-keyed dict.

    Raises:
        ValueError: If any entry is missing a required field, a field is not a
            string, or a duplicate prompt name is found.
    """
    cache: dict = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"prompts.yml: entry at index {index} must be a mapping, "
                f"got {type(entry).__name__!r}"
            )
        for field in _REQUIRED_FIELDS:
            if field not in entry:
                raise ValueError(
                    f"prompts.yml: entry at index {index} is missing required "
                    f"field {field!r}"
                )
            if not isinstance(entry[field], str):
                raise ValueError(
                    f"prompts.yml: entry at index {index} field {field!r} must "
                    f"be a string, got {type(entry[field]).__name__!r}"
                )
        name = entry["name"]
        if name in cache:
            raise ValueError(
                f"prompts.yml: duplicate prompt name {name!r} found at index {index}"
            )
        cache[name] = entry
    return cache


def _load_prompts() -> dict:
    """Load and cache prompts from prompts.yml, keyed by prompt name."""
    global _prompts_cache
    if _prompts_cache is None:
        filepath = Path(__file__).resolve().parent.parent.parent / "prompts.yml"
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        entries = data.get("prompts", [])
        if not isinstance(entries, list):
            raise ValueError(
                "prompts.yml: 'prompts' key must be a list, "
                f"got {type(entries).__name__!r}"
            )
        _prompts_cache = _validate_prompts(entries)
    return _prompts_cache


def render_prompt(name: str, **context) -> list[dict]:
    """Render a named prompt from prompts.yml using Jinja2 and return a message list.

    Raises:
        KeyError: If the prompt name is not found in prompts.yml.
        ValueError: If the template references a context key that was not supplied.
    """
    prompts = _load_prompts()
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found in prompts.yml")
    prompt = prompts[name]
    env = Environment(undefined=StrictUndefined, autoescape=False)
    try:
        system_text = env.from_string(prompt["system"]).render(**context)
        user_text = env.from_string(prompt["user"]).render(**context)
    except UndefinedError as exc:
        raise ValueError(
            f"Prompt '{name}' rendering failed: {exc}"
        ) from exc
    return [
        {"role": "system", "content": system_text.strip()},
        {"role": "user", "content": user_text.strip()},
    ]

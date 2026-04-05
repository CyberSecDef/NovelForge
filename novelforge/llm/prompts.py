"""Prompt loading and rendering from prompts.yml."""

import logging
import threading
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError

logger = logging.getLogger(__name__)

_jinja_env = Environment(undefined=StrictUndefined, autoescape=False)


_prompts_cache: dict | None = None
_prompts_lock = threading.Lock()


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
    """Load and cache prompts from prompts.yml, keyed by prompt name.

    Uses double-checked locking so that concurrent threads never read
    the YAML file twice.
    """
    global _prompts_cache
    if _prompts_cache is not None:
        return _prompts_cache  # fast path — no lock overhead
    with _prompts_lock:
        if _prompts_cache is not None:
            return _prompts_cache  # another thread populated it while we waited
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
    env = _jinja_env
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

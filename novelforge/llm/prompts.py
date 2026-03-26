"""Prompt loading and rendering from prompts.yml."""

import logging
from pathlib import Path

import yaml
from jinja2 import Template

logger = logging.getLogger(__name__)


_prompts_cache: dict | None = None


def _load_prompts() -> dict:
    """Load and cache prompts from prompts.yml, keyed by prompt name."""
    global _prompts_cache
    if _prompts_cache is None:
        filepath = Path(__file__).resolve().parent.parent.parent / "prompts.yml"
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _prompts_cache = {p["name"]: p for p in data.get("prompts", [])}
    return _prompts_cache


def render_prompt(name: str, **context) -> list[dict]:
    """Render a named prompt from prompts.yml using Jinja2 and return a message list."""
    prompts = _load_prompts()
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found in prompts.yml")
    prompt = prompts[name]
    system_text = Template(prompt["system"]).render(**context)
    user_text = Template(prompt["user"]).render(**context)
    return [
        {"role": "system", "content": system_text.strip()},
        {"role": "user", "content": user_text.strip()},
    ]

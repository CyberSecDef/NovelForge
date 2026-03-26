"""Prompt loading and rendering from prompts.yml."""

import logging
import os
from pathlib import Path

import yaml
from jinja2 import Template

logger = logging.getLogger(__name__)


def load_prompt_by_name(prompt_name: str, filename: str = 'prompts.yaml') -> dict | None:
    """
    Loads a specific prompt by name from a YAML file.

    Args:
        prompt_name (str): The name (key) of the prompt to retrieve.
        filename (str): The path to the YAML file.

    Returns:
        dict or None: The prompt dictionary if found, otherwise None.
    """
    # Ensure the file path is correct
    filepath = os.path.join(os.getcwd(), filename)
    if not os.path.exists(filepath):
        logger.error("Prompt file not found: %s at %s", filename, filepath)
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Use safe_load to avoid potential security issues from untrusted sources
            prompts_data = yaml.safe_load(file)

            if prompts_data and prompt_name in prompts_data:
                return prompts_data[prompt_name]
            else:
                logger.warning("Prompt '%s' not found in %s", prompt_name, filename)
                return None

    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file: %s", e)
        return None


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

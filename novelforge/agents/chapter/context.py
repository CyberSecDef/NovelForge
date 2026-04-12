"""ChapterContext dataclass and chapter-level utility functions."""

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChapterContext:
    """Bundles the planning-agent context strings for a single chapter.

    Avoids passing 8+ individual string parameters through the chapter
    pipeline and its callers.
    """
    architecture: str = ""
    timeline: str = ""
    fate: str = ""
    arc: str = ""
    antagonist: str = ""
    technology: str = ""
    theme: str = ""
    pov: str = ""
    gatekeeper_brief: str = ""
    perspective_prompt: str = ""


def build_perspective_prompt(narrative_perspective: str) -> str:
    """Build a perspective directive string from the session's narrative_perspective value."""
    if not narrative_perspective or narrative_perspective == "third_person":
        return (
            "NARRATIVE PERSPECTIVE: Write this chapter in THIRD PERSON narration. "
            "Use \"he/she/they\" pronouns for all characters. The narrator is omniscient "
            "but should primarily follow the assigned POV character's experience. "
            "Do NOT use first person (\"I/me/my\") for any narration."
        )
    if narrative_perspective.startswith("first_person:"):
        pov_name = narrative_perspective[len("first_person:"):].strip()
        if not pov_name:
            # Malformed perspective value — fall back to third person
            return build_perspective_prompt("third_person")
        return (
            f"NARRATIVE PERSPECTIVE: Write this chapter in FIRST PERSON narration "
            f"from the perspective of {pov_name}. Use \"I/me/my\" pronouns for {pov_name}. "
            f"Everything must be filtered through {pov_name}'s direct experience — "
            f"the reader can only know what {pov_name} sees, hears, thinks, and feels. "
            f"{pov_name} cannot know other characters' unspoken thoughts or events "
            f"happening outside their presence. Maintain {pov_name}'s unique voice, "
            f"vocabulary, and worldview consistently throughout. "
            f"Do NOT switch to third person or any other character's perspective."
        )
    return ""


def _format_characters(character_list: list[dict]) -> str:
    """Format a character list as a human-readable string for prompt injection."""
    if not character_list:
        return "No characters defined."
    lines = []
    for ch in character_list:
        base = (
            f"- {ch.get('name','?')} (age {ch.get('age','?')}): "
            f"{ch.get('role','?')}. Background: {ch.get('background','')}. "
            f"Arc: {ch.get('arc','')}."
        )
        extras = []
        flaw = ch.get("internal_flaw", "")
        if flaw:
            extras.append(f"  Internal flaw: {flaw}")
        goal = ch.get("personal_goal", "")
        if goal:
            extras.append(f"  Personal goal: {goal}")
        lines.append("\n".join([base] + extras) if extras else base)
    return "\n".join(lines)


def collect_existing_character_names() -> str:
    """
    Scan existing session files to collect character names from prior novels.

    This is a filesystem helper intended to be called by the caller *before*
    invoking :func:`build_characters_prompt`.  Keeping it separate from the
    prompt builder ensures that prompt-builder functions remain pure
    (data-in / prompt-out) and are not coupled to on-disk state.
    """
    import novelforge.config as config
    from pathlib import Path
    names: set[str] = set()
    novels_dir = Path(config.NOVELS_DIR)
    if not novels_dir.exists():
        return ""
    for f in novels_dir.glob("*.json"):
        if f.name.endswith("_progress.json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for char in data.get("character_list", []):
                if isinstance(char, dict):
                    name = char.get("name", "").strip()
                    if name:
                        names.add(name)
        except Exception:
            continue
    return ", ".join(sorted(names)) if names else ""

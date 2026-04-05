"""Character Fate Registry planning agent."""

from novelforge.agents.base import BaseAgent
from novelforge.agents.planning._helpers import (
    _coerce_positive_int,
    _safe_chapter_list,
    render_prompt,
)


class CharacterFateRegistryAgent(BaseAgent):
    name = "Character Fate Registry"
    prompt_action = "Planning Character Fate Registry"

    def build_prompt(self, **ctx) -> list[dict]:
        title = ctx["title"]
        premise = ctx["premise"]
        genre = ctx["genre"]
        character_list = ctx.get("character_list", [])
        chapter_list = ctx["chapter_list"]
        master_timeline = ctx.get("master_timeline")
        special_instructions = ctx.get("special_instructions", "")

        characters_text = "\n".join(
            f"- {c.get('name', '?')}: role={c.get('role', '')}; arc={c.get('arc', '')}; "
            f"background={c.get('background', '')}"
            for c in character_list
        )
        if not characters_text.strip():
            characters_text = "- No explicit character list provided. Infer only from outline."

        chapter_text = "\n".join(
            f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
            for i, ch in enumerate(chapter_list)
        )

        timeline_lines = []
        if isinstance(master_timeline, dict):
            for event in master_timeline.get("ledger", [])[:60]:
                if isinstance(event, dict):
                    timeline_lines.append(
                        f"- Ch {event.get('chapter')}: {event.get('event', '')} [{event.get('event_type', 'other')}]"
                    )
        timeline_text = "\n".join(timeline_lines)

        return render_prompt(
            "character_fate_registry",
            title=title,
            premise=premise,
            genre=genre,
            characters_text=characters_text,
            chapter_text=chapter_text,
            timeline_text=timeline_text,
            special_instructions=special_instructions or "",
        )

    def build_fallback(self, **ctx) -> dict:
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        total_chapters = ctx.get("total_chapters", max(1, len(chapter_list)))
        return self._build_fallback_impl(character_list, total_chapters)

    @staticmethod
    def _build_fallback_impl(character_list: list[dict], total_chapters: int) -> dict:
        """Deterministic fallback registry if planning fails."""
        total_chapters = max(1, total_chapters)
        safe_characters = character_list or []

        registry = []
        for ch in safe_characters:
            name = str(ch.get("name", "")).strip()
            if not name:
                continue
            registry.append(
                {
                    "character": name,
                    "current_status": "alive",
                    "capture_state": "free",
                    "injuries": [],
                    "narrative_status": "active",
                    "definitive_outcome_required": False,
                    "definitive_outcome": "unknown",
                    "outcome_locked": False,
                    "single_death_rule": True,
                    "death_chapter": None,
                    "recovery_conditions": [],
                    "state_constraints": ["Do not contradict established status transitions."],
                    "pivotal_chapters": [1, total_chapters],
                }
            )

        chapter_constraints = []
        for chapter in range(1, total_chapters + 1):
            chapter_constraints.append(
                {
                    "chapter": chapter,
                    "must_track": ["Character status continuity from prior chapters."],
                    "must_not_contradict": ["No dead character appears active without explicit recovery mechanism."],
                }
            )

        return {
            "registry": registry,
            "chapter_constraints": chapter_constraints,
            "conflict_checks": [],
        }

    def normalise(self, data: dict, **ctx) -> dict:
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        total_chapters = ctx.get("total_chapters", max(1, len(chapter_list)))
        fallback = self._build_fallback_impl(character_list, total_chapters)
        if not isinstance(data, dict):
            return fallback

        allowed_statuses = {"alive", "captured", "injured", "deceased", "missing", "recovered", "unknown"}
        allowed_capture = {"free", "captured", "escaped", "unknown"}
        allowed_narrative = {"active", "inactive", "resolved", "deceased"}
        allowed_outcome = {"unknown", "survival", "death", "redemption", "exile", "betrayal"}

        raw_registry = data.get("registry", [])
        if not isinstance(raw_registry, list):
            raw_registry = []

        normalised_registry = []
        seen_names = set()
        for item in raw_registry:
            if not isinstance(item, dict):
                continue
            name = str(item.get("character", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            current_status = str(item.get("current_status", "alive")).strip().lower()
            if current_status not in allowed_statuses:
                current_status = "unknown"

            capture_state = str(item.get("capture_state", "unknown")).strip().lower()
            if capture_state not in allowed_capture:
                capture_state = "unknown"

            narrative_status = str(item.get("narrative_status", "active")).strip().lower()
            if narrative_status not in allowed_narrative:
                narrative_status = "active"

            definitive_outcome = str(item.get("definitive_outcome", "unknown")).strip().lower()
            if definitive_outcome not in allowed_outcome:
                definitive_outcome = "unknown"

            death_chapter_raw = item.get("death_chapter")
            death_chapter = None
            if death_chapter_raw not in (None, ""):
                death_chapter = _coerce_positive_int(death_chapter_raw, 0) or None
                if death_chapter and death_chapter > total_chapters:
                    death_chapter = total_chapters

            entry = {
                "character": name,
                "current_status": "deceased" if definitive_outcome == "death" else current_status,
                "capture_state": capture_state,
                "injuries": [str(x) for x in item.get("injuries", []) if str(x).strip()],
                "narrative_status": "deceased" if definitive_outcome == "death" else narrative_status,
                "definitive_outcome_required": bool(item.get("definitive_outcome_required", False)),
                "definitive_outcome": definitive_outcome,
                "outcome_locked": bool(item.get("outcome_locked", False)),
                "single_death_rule": bool(item.get("single_death_rule", True)),
                "death_chapter": death_chapter,
                "recovery_conditions": [str(x) for x in item.get("recovery_conditions", []) if str(x).strip()],
                "state_constraints": [str(x) for x in item.get("state_constraints", []) if str(x).strip()],
                "pivotal_chapters": [
                    _coerce_positive_int(ch, 1)
                    for ch in item.get("pivotal_chapters", [])
                    if _coerce_positive_int(ch, 0) > 0
                ],
            }
            if entry["definitive_outcome"] == "death" and entry["death_chapter"] is None:
                entry["death_chapter"] = max(1, total_chapters - 1)
            normalised_registry.append(entry)

        chapter_constraints = data.get("chapter_constraints", [])
        if not isinstance(chapter_constraints, list):
            chapter_constraints = []
        normalised_chapter_constraints = []
        for idx, item in enumerate(chapter_constraints, start=1):
            if not isinstance(item, dict):
                continue
            chapter = _coerce_positive_int(item.get("chapter"), idx)
            if chapter > total_chapters:
                chapter = total_chapters
            normalised_chapter_constraints.append(
                {
                    "chapter": chapter,
                    "must_track": [str(x) for x in item.get("must_track", []) if str(x).strip()],
                    "must_not_contradict": [str(x) for x in item.get("must_not_contradict", []) if str(x).strip()],
                }
            )

        conflict_checks = data.get("conflict_checks", [])
        if not isinstance(conflict_checks, list):
            conflict_checks = []

        merged = {
            "registry": normalised_registry or fallback["registry"],
            "chapter_constraints": normalised_chapter_constraints or fallback["chapter_constraints"],
            "conflict_checks": [str(x) for x in conflict_checks if str(x).strip()],
        }
        return merged

    def plan(self, **ctx) -> dict:
        chapter_list = ctx.get("chapter_list", [])
        total_chapters = max(1, len(chapter_list))
        ctx["total_chapters"] = total_chapters
        return super().plan(**ctx)

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        if not isinstance(plan, dict):
            return ""

        registry = plan.get("registry", [])
        if not isinstance(registry, list):
            registry = []

        lines = ["Character Fate Registry output for this chapter:"]
        for entry in registry:
            if not isinstance(entry, dict):
                continue
            pivotal_chapters = entry.get("pivotal_chapters", [])
            chapter_in_scope = False
            if isinstance(pivotal_chapters, list):
                chapter_in_scope = any(_coerce_positive_int(ch, 0) == chapter_num for ch in pivotal_chapters)

            should_include = chapter_in_scope or entry.get("outcome_locked") or entry.get("definitive_outcome_required")
            if should_include:
                lines.append(
                    f"- {entry.get('character', '?')}: status={entry.get('current_status', 'unknown')}, "
                    f"capture={entry.get('capture_state', 'unknown')}, narrative={entry.get('narrative_status', 'active')}, "
                    f"outcome={entry.get('definitive_outcome', 'unknown')}, locked={entry.get('outcome_locked', False)}"
                )
                if entry.get("state_constraints"):
                    lines.append(
                        "  - State constraints: " + "; ".join(str(x) for x in entry.get("state_constraints", [])[:4])
                    )
                if entry.get("injuries"):
                    lines.append("  - Injuries: " + "; ".join(str(x) for x in entry.get("injuries", [])[:4]))
                if entry.get("recovery_conditions"):
                    lines.append(
                        "  - Recovery conditions: " + "; ".join(str(x) for x in entry.get("recovery_conditions", [])[:3])
                    )

        chapter_constraints = plan.get("chapter_constraints", [])
        if isinstance(chapter_constraints, list):
            chapter_constraint = next(
                (
                    c for c in chapter_constraints
                    if isinstance(c, dict) and _coerce_positive_int(c.get("chapter"), 0) == chapter_num
                ),
                None,
            )
            if chapter_constraint:
                must_track = chapter_constraint.get("must_track", [])
                must_not = chapter_constraint.get("must_not_contradict", [])
                if must_track:
                    lines.append("- Must track: " + "; ".join(str(x) for x in must_track[:6]))
                if must_not:
                    lines.append("- Must not contradict: " + "; ".join(str(x) for x in must_not[:6]))

        checks = plan.get("conflict_checks", [])
        if isinstance(checks, list) and checks:
            lines.append("- Conflict checks: " + "; ".join(str(x) for x in checks[:5]))

        return "\n".join(lines)


# -- Singleton & wrapper functions -------------------------------------------

_character_fate_registry_agent = CharacterFateRegistryAgent()


def plan_character_fate_registry(**kwargs: object) -> dict:
    return _character_fate_registry_agent.plan(**kwargs)


def normalise_character_fate_registry(registry_data: dict, character_list: list[dict], total_chapters: int) -> dict:
    return _character_fate_registry_agent.normalise(registry_data, character_list=character_list, total_chapters=total_chapters)


def get_chapter_fate_context(character_fate_registry: dict, chapter_num: int) -> str:
    return _character_fate_registry_agent.get_chapter_context(character_fate_registry, chapter_num)

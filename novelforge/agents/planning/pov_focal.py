"""POV & Focal Character planning agent."""

from novelforge.agents.base import BaseAgent
from novelforge.agents.planning._helpers import (
    _coerce_positive_int,
    _safe_chapter_list,
    render_prompt,
)


class PovFocalCharacterAgent(BaseAgent):
    name = "POV & Focal Character Planner"
    prompt_action = "Planning POV & Focal Characters"

    def build_prompt(self, **ctx) -> list[dict]:
        title = ctx["title"]
        premise = ctx["premise"]
        genre = ctx["genre"]
        character_list = ctx.get("character_list", [])
        chapter_list = ctx["chapter_list"]
        character_arc_plan = ctx.get("character_arc_plan")
        special_instructions = ctx.get("special_instructions", "")
        narrative_perspective = ctx.get("narrative_perspective", "third_person")

        characters_text = "\n".join(
            f"- {c.get('name', '?')}: role={c.get('role', '')}; arc={c.get('arc', '')}; background={c.get('background', '')}"
            for c in character_list
        )
        if not characters_text.strip():
            characters_text = "- No explicit characters provided. Infer conservatively from outline."

        chapters_text = "\n".join(
            f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
            for i, ch in enumerate(chapter_list)
        )

        arc_lines: list[str] = []
        if isinstance(character_arc_plan, dict):
            for arc in character_arc_plan.get("arcs", []):
                if isinstance(arc, dict):
                    arc_lines.append(
                        f"- {arc.get('character', '?')}: start='{arc.get('start_state', '')}' \u2192 "
                        f"midpoint='{arc.get('midpoint_transformation', '')}' \u2192 "
                        f"crisis='{arc.get('crisis_point', '')}' \u2192 "
                        f"final_choice='{arc.get('final_moral_choice', '')}'"
                    )
        arc_text = "\n".join(arc_lines)

        # Build perspective override for POV planning
        perspective_override = ""
        if isinstance(narrative_perspective, str) and narrative_perspective.startswith("first_person:"):
            pov_name = narrative_perspective[len("first_person:"):].strip()
            perspective_override = (
                f"IMPORTANT: The author has chosen FIRST PERSON narration from {pov_name}'s perspective. "
                f"{pov_name} MUST be the primary POV character for EVERY chapter. Secondary observers "
                f"and focal internal characters can vary, but primary_pov must always be \"{pov_name}\"."
            )

        return render_prompt(
            "pov_focal_character_planner",
            title=title, premise=premise, genre=genre,
            characters_text=characters_text, chapters_text=chapters_text,
            arc_text=arc_text, special_instructions=special_instructions or "",
            perspective_override=perspective_override,
        )

    def build_fallback(self, **ctx) -> dict:
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        return self._build_fallback_impl(character_list, chapter_list)

    @staticmethod
    def _build_fallback_impl(character_list: list[dict], chapter_list: list[dict]) -> dict:
        safe_characters = [c for c in (character_list or []) if isinstance(c, dict) and str(c.get("name", "")).strip()]
        if not safe_characters:
            safe_characters = [{"name": "Protagonist", "role": "protagonist"}]

        protagonist_names = []
        for c in safe_characters:
            role = str(c.get("role", "")).lower()
            if any(tag in role for tag in ("protagonist", "lead", "main", "hero")):
                protagonist_names.append(str(c.get("name", "")).strip())
        if not protagonist_names:
            protagonist_names = [str(safe_characters[0].get("name", "Protagonist")).strip()]

        safe_chapter_list = _safe_chapter_list(chapter_list)
        total_chapters = max(1, len(safe_chapter_list))
        chapter_pov_plan = []
        for idx, ch in enumerate(safe_chapter_list):
            chapter_num = _coerce_positive_int(ch.get("number", idx + 1), idx + 1)
            pov_char = safe_characters[idx % len(safe_characters)]
            pov_name = str(pov_char.get("name", "")).strip()
            if chapter_num > total_chapters - 2:
                pov_name = protagonist_names[0]
            chapter_pov_plan.append({
                "chapter": chapter_num,
                "primary_pov": pov_name,
                "secondary_observers": [],
                "focal_internal_character": pov_name,
                "pov_justification": "Fallback round-robin assignment.",
            })

        return {
            "chapter_pov_plan": chapter_pov_plan,
            "rotation_rules": [
                "No same primary POV for more than 2 consecutive chapters.",
                "Focal internal character should rotate to prevent emotional monotony.",
            ],
            "rotation_violations": [],
        }

    def normalise(self, data: dict, **ctx) -> dict:
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        fallback = self._build_fallback_impl(character_list, chapter_list)
        if not isinstance(data, dict):
            return fallback

        raw_plan = data.get("chapter_pov_plan", [])
        if not isinstance(raw_plan, list) or not raw_plan:
            raw_plan = []

        total_chapters = max(1, len(chapter_list or []))
        normalised_plan = []
        seen_chapters = set()
        for item in raw_plan:
            if not isinstance(item, dict):
                continue
            chapter_num = _coerce_positive_int(item.get("chapter"), 0)
            if chapter_num < 1 or chapter_num > total_chapters or chapter_num in seen_chapters:
                continue
            seen_chapters.add(chapter_num)
            primary_pov = str(item.get("primary_pov", "")).strip()
            focal = str(item.get("focal_internal_character", "")).strip()
            secondary = item.get("secondary_observers", [])
            if not isinstance(secondary, list):
                secondary = []
            secondary = [str(s).strip() for s in secondary if str(s).strip()]
            normalised_plan.append({
                "chapter": chapter_num,
                "primary_pov": primary_pov or "Unknown",
                "secondary_observers": secondary,
                "focal_internal_character": focal or primary_pov or "Unknown",
                "pov_justification": str(item.get("pov_justification", "")).strip(),
            })

        fallback_map = {item["chapter"]: item for item in fallback["chapter_pov_plan"]}
        normalised_map = {item["chapter"]: item for item in normalised_plan}
        for ch_num in range(1, total_chapters + 1):
            if ch_num not in normalised_map and ch_num in fallback_map:
                normalised_plan.append(fallback_map[ch_num])
        normalised_plan.sort(key=lambda x: int(x.get("chapter", 0)))  # type: ignore[call-overload]

        rotation_rules = data.get("rotation_rules", [])
        if not isinstance(rotation_rules, list):
            rotation_rules = fallback["rotation_rules"]
        rotation_violations = data.get("rotation_violations", [])
        if not isinstance(rotation_violations, list):
            rotation_violations = []
        normalised_violations = []
        for v in rotation_violations:
            if not isinstance(v, dict):
                continue
            normalised_violations.append({
                "chapter_range": v.get("chapter_range", []),
                "character": str(v.get("character", "")).strip(),
                "reason": str(v.get("reason", "")).strip(),
            })

        return {
            "chapter_pov_plan": normalised_plan or fallback["chapter_pov_plan"],
            "rotation_rules": [str(r) for r in rotation_rules if str(r).strip()] or fallback["rotation_rules"],
            "rotation_violations": normalised_violations,
        }

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        if not isinstance(plan, dict):
            return ""
        chapter_pov_plan = plan.get("chapter_pov_plan", [])
        if not isinstance(chapter_pov_plan, list):
            return ""
        entry = next(
            (item for item in chapter_pov_plan
             if isinstance(item, dict) and _coerce_positive_int(item.get("chapter"), 0) == chapter_num),
            None,
        )
        if not entry:
            return ""

        lines = ["POV & Focal Character Planner output for this chapter:"]
        lines.append(f"- Primary POV character: {entry.get('primary_pov', 'Unknown')}")
        secondary = entry.get("secondary_observers", [])
        if isinstance(secondary, list) and secondary:
            lines.append(f"- Secondary observers: {', '.join(str(s) for s in secondary)}")
        else:
            lines.append("- Secondary observers: none")
        focal = entry.get("focal_internal_character", "")
        if focal:
            lines.append(f"- Focal internal character (emotions drive scene weight): {focal}")
        justification = entry.get("pov_justification", "")
        if justification:
            lines.append(f"- Justification: {justification}")
        lines.append("- Write this chapter primarily through the primary POV character's perspective.")
        lines.append("- The focal internal character's emotional state and internal conflict should carry the most weight.")
        return "\n".join(lines)


# -- Singleton & wrapper functions -------------------------------------------

_pov_focal_character_agent = PovFocalCharacterAgent()


def plan_pov_focal_character(**kwargs: object) -> dict:
    return _pov_focal_character_agent.plan(**kwargs)


def normalise_pov_focal_character_plan(plan_data: dict, character_list: list[dict], chapter_list: list[dict]) -> dict:
    return _pov_focal_character_agent.normalise(plan_data, character_list=character_list, chapter_list=chapter_list)


def get_chapter_pov_context(pov_plan: dict, chapter_num: int) -> str:
    return _pov_focal_character_agent.get_chapter_context(pov_plan, chapter_num)

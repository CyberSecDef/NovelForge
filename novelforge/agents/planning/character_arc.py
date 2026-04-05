"""Character Arc planning agent."""

from novelforge.agents.base import BaseAgent
from novelforge.agents.planning._helpers import (
    _coerce_positive_int,
    _safe_chapter_list,
    render_prompt,
)


class CharacterArcPlanAgent(BaseAgent):
    name = "Character Arc Planner"
    prompt_action = "Planning Character Arcs"

    def build_prompt(self, **ctx) -> list[dict]:
        """Build the character arc planning prompt from character and chapter context."""
        title = ctx["title"]
        premise = ctx["premise"]
        genre = ctx["genre"]
        character_list = ctx.get("character_list", [])
        chapter_list = ctx["chapter_list"]
        special_instructions = ctx.get("special_instructions", "")

        characters_text = "\n".join(
            f"- {c.get('name', '?')}: role={c.get('role', '')}; arc={c.get('arc', '')}; background={c.get('background', '')}"
            for c in character_list
        )
        if not characters_text.strip():
            characters_text = "- No explicit characters provided. Infer primary characters conservatively."

        chapters_text = "\n".join(
            f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
            for i, ch in enumerate(chapter_list)
        )

        return render_prompt(
            "character_arc_planner",
            title=title,
            premise=premise,
            genre=genre,
            characters_text=characters_text,
            chapters_text=chapters_text,
            special_instructions=special_instructions or "",
        )

    def build_fallback(self, **ctx) -> dict:
        """Build a deterministic fallback arc plan from character and chapter lists."""
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        return self._build_fallback_impl(character_list, chapter_list)

    @staticmethod
    def _build_fallback_impl(character_list: list[dict], chapter_list: list[dict]) -> dict:
        """Deterministic fallback character arc plan."""
        from novelforge.chapter_position import ChapterPosition
        safe_chapter_list = _safe_chapter_list(chapter_list)
        total_chapters = max(1, len(safe_chapter_list))
        midpoint_chapter = ChapterPosition.midpoint_chapter(total_chapters)
        crisis_chapter = ChapterPosition.climax_chapter(total_chapters)
        final_chapter = ChapterPosition.resolution_chapter(total_chapters)

        arcs = []
        for idx, character in enumerate(character_list or []):
            name = str(character.get("name", "")).strip()
            if not name:
                continue
            role = str(character.get("role", "")).strip().lower()
            role_class = "primary" if idx < 3 or "protagonist" in role or "antagonist" in role else "secondary"
            if role_class != "primary":
                continue

            arcs.append(
                {
                    "character": name,
                    "role": role_class,
                    "start_state": "Begins with a constrained worldview and unresolved internal tension.",
                    "midpoint_transformation": "Confronts disconfirming evidence that shifts priorities.",
                    "crisis_point": "Must choose between self-protection and core values.",
                    "final_moral_choice": "Makes a definitive ethical choice that resolves the arc.",
                    "arc_theme": str(character.get("arc", "Identity under pressure") or "Identity under pressure"),
                    "chapter_beats": [
                        {"chapter": 1, "phase": "start", "beat": "Establishes baseline motivations and flaws."},
                        {
                            "chapter": midpoint_chapter,
                            "phase": "midpoint",
                            "beat": "Midpoint shift challenges assumptions and role.",
                        },
                        {
                            "chapter": crisis_chapter,
                            "phase": "crisis",
                            "beat": "Faces hardest internal/external decision.",
                        },
                        {
                            "chapter": final_chapter,
                            "phase": "final",
                            "beat": "Commits to final moral choice and consequence.",
                        },
                    ],
                    "consistency_rules": [
                        "Arc must move forward each appearance.",
                        "No regression to start state after midpoint without explicit cause.",
                    ],
                }
            )

        chapter_constraints = []
        for idx, chapter in enumerate(safe_chapter_list, start=1):
            chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
            chapter_constraints.append(
                {
                    "chapter": chapter_num,
                    "must_advance": ["At least one active arc beat or consequence must progress."],
                    "must_not_undo": ["Do not reset established character growth without explicit trigger."],
                }
            )

        return {
            "arcs": arcs,
            "chapter_constraints": chapter_constraints,
            "global_arc_risks": [],
        }

    def normalise(self, data: dict, **ctx) -> dict:
        """Validate and merge LLM arc plan with deterministic fallback."""
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        fallback = self._build_fallback_impl(character_list, chapter_list)
        if not isinstance(data, dict):
            return fallback

        raw_arcs = data.get("arcs", [])
        if not isinstance(raw_arcs, list):
            raw_arcs = []

        normalised_arcs = []
        seen_names = set()
        for item in raw_arcs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("character", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            chapter_beats = item.get("chapter_beats", [])
            if not isinstance(chapter_beats, list):
                chapter_beats = []
            consistency_rules = item.get("consistency_rules", [])
            if not isinstance(consistency_rules, list):
                consistency_rules = []

            normalised_beats = []
            for beat in chapter_beats:
                if not isinstance(beat, dict):
                    continue
                normalised_beats.append(
                    {
                        "chapter": _coerce_positive_int(beat.get("chapter"), 1),
                        "phase": str(beat.get("phase", "start")),
                        "beat": str(beat.get("beat", "")).strip(),
                    }
                )

            normalised_arcs.append(
                {
                    "character": name,
                    "role": str(item.get("role", "primary") or "primary"),
                    "start_state": str(item.get("start_state", "")).strip(),
                    "midpoint_transformation": str(item.get("midpoint_transformation", "")).strip(),
                    "crisis_point": str(item.get("crisis_point", "")).strip(),
                    "final_moral_choice": str(item.get("final_moral_choice", "")).strip(),
                    "arc_theme": str(item.get("arc_theme", "")).strip(),
                    "chapter_beats": normalised_beats,
                    "consistency_rules": [str(x) for x in consistency_rules if str(x).strip()],
                }
            )

        raw_constraints = data.get("chapter_constraints", [])
        if not isinstance(raw_constraints, list):
            raw_constraints = []
        normalised_constraints = []
        for idx, item in enumerate(raw_constraints, start=1):
            if not isinstance(item, dict):
                continue
            normalised_constraints.append(
                {
                    "chapter": _coerce_positive_int(item.get("chapter"), idx),
                    "must_advance": [str(x) for x in item.get("must_advance", []) if str(x).strip()],
                    "must_not_undo": [str(x) for x in item.get("must_not_undo", []) if str(x).strip()],
                }
            )

        global_arc_risks = data.get("global_arc_risks", [])
        if not isinstance(global_arc_risks, list):
            global_arc_risks = []

        return {
            "arcs": normalised_arcs or fallback["arcs"],
            "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
            "global_arc_risks": [str(x) for x in global_arc_risks if str(x).strip()],
        }

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        """Format arc beats and constraints as a prompt snippet for a chapter."""
        if not isinstance(plan, dict):
            return ""

        arcs = plan.get("arcs", [])
        if not isinstance(arcs, list):
            arcs = []
        constraints = plan.get("chapter_constraints", [])
        if not isinstance(constraints, list):
            constraints = []

        lines = ["Character Arc Planner output for this chapter:"]
        for arc in arcs:
            if not isinstance(arc, dict):
                continue
            beats = arc.get("chapter_beats", [])
            if not isinstance(beats, list):
                beats = []
            matching_beats = [
                beat for beat in beats
                if isinstance(beat, dict) and _coerce_positive_int(beat.get("chapter"), 0) == chapter_num
            ]
            if matching_beats:
                lines.append(
                    f"- {arc.get('character', '?')}: start={arc.get('start_state', '')}; midpoint={arc.get('midpoint_transformation', '')}; "
                    f"crisis={arc.get('crisis_point', '')}; final_choice={arc.get('final_moral_choice', '')}"
                )
                for beat in matching_beats[:3]:
                    lines.append(f"  - Beat ({beat.get('phase', 'arc')}): {beat.get('beat', '')}")
                rules = arc.get("consistency_rules", [])
                if isinstance(rules, list) and rules:
                    lines.append("  - Arc rules: " + "; ".join(str(x) for x in rules[:4]))

        chapter_constraint = next(
            (
                item for item in constraints
                if isinstance(item, dict) and _coerce_positive_int(item.get("chapter"), 0) == chapter_num
            ),
            None,
        )
        if chapter_constraint:
            must_advance = chapter_constraint.get("must_advance", [])
            must_not_undo = chapter_constraint.get("must_not_undo", [])
            if must_advance:
                lines.append("- Must advance: " + "; ".join(str(x) for x in must_advance))
            if must_not_undo:
                lines.append("- Must not undo: " + "; ".join(str(x) for x in must_not_undo))

        risks = plan.get("global_arc_risks", [])
        if isinstance(risks, list) and risks:
            lines.append("- Global arc risks: " + "; ".join(str(x) for x in risks[:5]))

        return "\n".join(lines)


# -- Singleton & wrapper functions -------------------------------------------

_character_arc_plan_agent = CharacterArcPlanAgent()


def plan_character_arc_plan(**kwargs: object) -> dict:
    """Delegate to the singleton CharacterArcPlanAgent instance."""
    return _character_arc_plan_agent.plan(**kwargs)


def normalise_character_arc_plan(arc_data: dict, character_list: list[dict], chapter_list: list[dict]) -> dict:
    """Delegate normalisation to the singleton CharacterArcPlanAgent."""
    return _character_arc_plan_agent.normalise(arc_data, character_list=character_list, chapter_list=chapter_list)


def get_chapter_arc_context(character_arc_plan: dict, chapter_num: int) -> str:
    """Delegate context formatting to the singleton CharacterArcPlanAgent."""
    return _character_arc_plan_agent.get_chapter_context(character_arc_plan, chapter_num)

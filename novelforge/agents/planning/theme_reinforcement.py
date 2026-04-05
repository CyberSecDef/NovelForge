"""Theme Reinforcement planning agent."""

from novelforge.agents.base import BaseAgent
from novelforge.agents.planning._helpers import (
    _safe_chapter_list,
    render_prompt,
)


class ThemeReinforcementAgent(BaseAgent):
    name = "Theme Reinforcement Planner"
    prompt_action = "Planning Theme Reinforcement"

    def build_prompt(self, **ctx) -> list[dict]:
        title = ctx["title"]
        premise = ctx["premise"]
        genre = ctx["genre"]
        chapter_list = ctx["chapter_list"]
        special_instructions = ctx.get("special_instructions", "")

        chapters_text = "\n".join(
            f"Chapter {c.get('number', i+1)}: {c.get('title', '')} – {c.get('summary', '')}"
            for i, c in enumerate(chapter_list)
        )
        return render_prompt("theme_reinforcement", title=title, premise=premise, genre=genre,
                             chapters_text=chapters_text, special_instructions=special_instructions or "")

    def build_fallback(self, **ctx) -> dict:
        chapter_list = ctx.get("chapter_list", [])
        return self._build_fallback_impl(chapter_list)

    @staticmethod
    def _build_fallback_impl(chapter_list: list[dict]) -> dict:
        safe_chapter_list = _safe_chapter_list(chapter_list)
        fallback_themes = [
            {
                "name": "Identity Under Pressure",
                "description": "How characters maintain or lose their sense of self under systemic pressure.",
                "motifs": ["mirrors", "names", "documents"],
                "pillar_moments": ["Inciting incident", "Midpoint crisis", "Final moral choice"],
                "chapter_appearances": [
                    {"chapter": c.get("number", i + 1), "role": "background", "guidance": "Show character making a small compromise."}
                    for i, c in enumerate(safe_chapter_list)
                ],
            },
            {
                "name": "Moral Compromise",
                "description": "The cost of choosing safety over principle.",
                "motifs": ["closed doors", "silence", "small betrayals"],
                "pillar_moments": ["First compromise", "Point of no return", "Reckoning"],
                "chapter_appearances": [
                    {"chapter": c.get("number", i + 1), "role": "background", "guidance": "Show institutional pressure shaping a decision."}
                    for i, c in enumerate(safe_chapter_list)
                ],
            },
        ]
        chapter_constraints = [
            {
                "chapter": c.get("number", i + 1),
                "themes_present": ["Identity Under Pressure"],
                "thematic_guidance": "Reinforce the protagonist's internal conflict quietly.",
            }
            for i, c in enumerate(safe_chapter_list)
        ]
        return {
            "themes": fallback_themes,
            "global_thematic_arcs": [
                "Individual identity erodes under systemic control.",
                "Moral compromise accumulates until a breaking point forces reckoning.",
            ],
            "chapter_constraints": chapter_constraints,
            "continuity_risks": [
                "Theme abandoned mid-story without resolution.",
                "Motifs introduced but never paid off.",
            ],
        }

    def normalise(self, data: dict, **ctx) -> dict:
        chapter_list = ctx.get("chapter_list", [])
        fallback = self._build_fallback_impl(chapter_list)
        if not isinstance(data, dict):
            return fallback

        themes = data.get("themes", [])
        if not isinstance(themes, list) or len(themes) == 0:
            themes = fallback["themes"]
        else:
            valid = []
            for t in themes:
                if not isinstance(t, dict):
                    continue
                valid.append({
                    "name": str(t.get("name", "Theme")),
                    "description": str(t.get("description", "")),
                    "motifs": t.get("motifs", []) if isinstance(t.get("motifs"), list) else [],
                    "pillar_moments": t.get("pillar_moments", []) if isinstance(t.get("pillar_moments"), list) else [],
                    "chapter_appearances": t.get("chapter_appearances", []) if isinstance(t.get("chapter_appearances"), list) else [],
                })
            themes = valid if valid else fallback["themes"]

        global_arcs = data.get("global_thematic_arcs", [])
        if not isinstance(global_arcs, list):
            global_arcs = fallback["global_thematic_arcs"]

        chapter_constraints = data.get("chapter_constraints", [])
        if not isinstance(chapter_constraints, list) or len(chapter_constraints) == 0:
            chapter_constraints = fallback["chapter_constraints"]
        else:
            valid_cc = []
            for cc in chapter_constraints:
                if not isinstance(cc, dict):
                    continue
                try:
                    ch_num = int(cc.get("chapter", 0))
                except (TypeError, ValueError):
                    ch_num = 0
                valid_cc.append({
                    "chapter": ch_num,
                    "themes_present": cc.get("themes_present", []) if isinstance(cc.get("themes_present"), list) else [],
                    "thematic_guidance": str(cc.get("thematic_guidance", "")),
                })
            chapter_constraints = valid_cc if valid_cc else fallback["chapter_constraints"]

        continuity_risks = data.get("continuity_risks", [])
        if not isinstance(continuity_risks, list):
            continuity_risks = fallback["continuity_risks"]

        return {
            "themes": themes,
            "global_thematic_arcs": global_arcs,
            "chapter_constraints": chapter_constraints,
            "continuity_risks": continuity_risks,
        }

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        if not isinstance(plan, dict):
            return ""

        lines = ["Theme Reinforcement Planner – Chapter guidance:"]

        themes = plan.get("themes", [])
        for theme in themes[:4]:
            if not isinstance(theme, dict):
                continue
            name = theme.get("name", "")
            desc = theme.get("description", "")
            if name:
                lines.append(f"- Theme '{name}': {desc}")
            appearances = theme.get("chapter_appearances", [])
            for ap in appearances:
                if not isinstance(ap, dict):
                    continue
                try:
                    if int(ap.get("chapter", -1)) == chapter_num:
                        role = ap.get("role", "")
                        guidance = ap.get("guidance", "")
                        lines.append(f"  \u25b8 Role in this chapter: {role}. {guidance}")
                        break
                except (TypeError, ValueError):
                    continue

        global_arcs = plan.get("global_thematic_arcs", [])
        if isinstance(global_arcs, list) and global_arcs:
            lines.append("- Global thematic arcs: " + "; ".join(str(a) for a in global_arcs[:3]))

        chapter_constraints = plan.get("chapter_constraints", [])
        for cc in chapter_constraints:
            if not isinstance(cc, dict):
                continue
            try:
                if int(cc.get("chapter", -1)) == chapter_num:
                    themes_present = cc.get("themes_present", [])
                    if isinstance(themes_present, list) and themes_present:
                        lines.append("- Themes active this chapter: " + ", ".join(str(t) for t in themes_present))
                    thematic_guidance = cc.get("thematic_guidance", "")
                    if thematic_guidance:
                        lines.append(f"- Thematic guidance: {thematic_guidance}")
                    break
            except (TypeError, ValueError):
                continue

        risks = plan.get("continuity_risks", [])
        if isinstance(risks, list) and risks:
            lines.append("- Thematic continuity risks: " + "; ".join(str(r) for r in risks[:3]))

        return "\n".join(lines)


# -- Singleton & wrapper functions -------------------------------------------

_theme_reinforcement_agent = ThemeReinforcementAgent()


def plan_theme_reinforcement(**kwargs: object) -> dict:
    return _theme_reinforcement_agent.plan(**kwargs)


def normalise_theme_reinforcement(theme_data: dict, chapter_list: list[dict]) -> dict:
    return _theme_reinforcement_agent.normalise(theme_data, chapter_list=chapter_list)


def get_chapter_theme_context(theme_reinforcement: dict, chapter_num: int) -> str:
    return _theme_reinforcement_agent.get_chapter_context(theme_reinforcement, chapter_num)

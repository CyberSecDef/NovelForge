"""Antagonist Motivation planning agent."""

from novelforge.agents.base import BaseAgent
from novelforge.agents.planning._helpers import (
    _coerce_positive_int,
    _safe_chapter_list,
    render_prompt,
)


class AntagonistMotivationAgent(BaseAgent):
    name = "Antagonist Motivation Architect"
    prompt_action = "Planning Antagonist Motivation"

    def build_prompt(self, **ctx) -> list[dict]:
        title = ctx["title"]
        premise = ctx["premise"]
        genre = ctx["genre"]
        character_list = ctx.get("character_list", [])
        chapter_list = ctx["chapter_list"]
        master_timeline = ctx.get("master_timeline")
        special_instructions = ctx.get("special_instructions", "")

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

        timeline_lines = []
        if isinstance(master_timeline, dict):
            for event in master_timeline.get("ledger", [])[:60]:
                if isinstance(event, dict):
                    timeline_lines.append(
                        f"- Ch {event.get('chapter')}: {event.get('event', '')} [{event.get('event_type', 'other')}]"
                    )
        timeline_text = "\n".join(timeline_lines)

        return render_prompt(
            "antagonist_motivation",
            title=title,
            premise=premise,
            genre=genre,
            characters_text=characters_text,
            chapters_text=chapters_text,
            timeline_text=timeline_text,
            special_instructions=special_instructions or "",
        )

    def build_fallback(self, **ctx) -> dict:
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        return self._build_fallback_impl(character_list, chapter_list)

    @staticmethod
    def _build_fallback_impl(character_list: list[dict], chapter_list: list[dict]) -> dict:
        safe_chapter_list = _safe_chapter_list(chapter_list)
        total_chapters = max(1, len(safe_chapter_list))

        antagonists = []
        for character in character_list or []:
            name = str(character.get("name", "")).strip()
            if not name:
                continue
            role = str(character.get("role", "")).strip().lower()
            is_antagonist = any(tag in role for tag in ("antagonist", "villain", "rival", "enemy", "opposition"))
            if not is_antagonist:
                continue
            antagonists.append(
                {
                    "character": name,
                    "motivation_core": "Preserve control in response to a perceived existential threat.",
                    "external_goal": "Block protagonist progress toward decisive objective.",
                    "internal_need": "Avoid vulnerability and loss of authority.",
                    "fear_trigger": "Loss of control or public exposure of weakness.",
                    "moral_line": "Will escalate harm strategically but avoids indiscriminate destruction.",
                    "pressure_points": ["Public legitimacy", "Trusted lieutenant", "Resource access"],
                    "escalation_plan": [
                        {
                            "chapter": 1,
                            "action": "Signals opposition through indirect interference.",
                            "tactic": "Plausible deniability",
                            "motivation_link": "Tests threat level while preserving cover.",
                        },
                        {
                            "chapter": max(1, min(total_chapters, round((total_chapters + 1) / 2))),
                            "action": "Commits to direct pressure after setbacks.",
                            "tactic": "Targeted retaliation",
                            "motivation_link": "Escalates to restore control.",
                        },
                        {
                            "chapter": total_chapters,
                            "action": "Makes final high-risk move aligned with core fear.",
                            "tactic": "All-in confrontation",
                            "motivation_link": "Chooses decisive action over gradual containment.",
                        },
                    ],
                    "consistency_rules": [
                        "Escalation must track rising pressure; no random reversals.",
                        "Tactics should follow established risk tolerance and moral line.",
                    ],
                }
            )

        chapter_constraints = []
        for idx, chapter in enumerate(safe_chapter_list, start=1):
            chapter_num = _coerce_positive_int(chapter.get("number"), idx)
            chapter_constraints.append(
                {
                    "chapter": chapter_num,
                    "must_show": ["Antagonist pressure should have clear motivation and objective."],
                    "must_not_break": ["Do not use antagonist tactics that contradict prior moral line or incentives."],
                }
            )

        return {
            "antagonists": antagonists,
            "chapter_constraints": chapter_constraints,
            "global_risks": [],
        }

    def normalise(self, data: dict, **ctx) -> dict:
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        fallback = self._build_fallback_impl(character_list, chapter_list)
        if not isinstance(data, dict):
            return fallback

        total_chapters = max(1, len(chapter_list))
        raw_antagonists = data.get("antagonists", [])
        if not isinstance(raw_antagonists, list):
            raw_antagonists = []

        normalised_antagonists = []
        seen_names = set()
        for item in raw_antagonists:
            if not isinstance(item, dict):
                continue
            name = str(item.get("character", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            escalation_plan = item.get("escalation_plan", [])
            if not isinstance(escalation_plan, list):
                escalation_plan = []
            normalised_escalation = []
            for step in escalation_plan:
                if not isinstance(step, dict):
                    continue
                chapter = _coerce_positive_int(step.get("chapter"), 1)
                chapter = min(chapter, total_chapters)
                normalised_escalation.append(
                    {
                        "chapter": chapter,
                        "action": str(step.get("action", "")).strip(),
                        "tactic": str(step.get("tactic", "")).strip(),
                        "motivation_link": str(step.get("motivation_link", "")).strip(),
                    }
                )

            normalised_antagonists.append(
                {
                    "character": name,
                    "motivation_core": str(item.get("motivation_core", "")).strip(),
                    "external_goal": str(item.get("external_goal", "")).strip(),
                    "internal_need": str(item.get("internal_need", "")).strip(),
                    "fear_trigger": str(item.get("fear_trigger", "")).strip(),
                    "moral_line": str(item.get("moral_line", "")).strip(),
                    "pressure_points": [str(x) for x in item.get("pressure_points", []) if str(x).strip()],
                    "escalation_plan": normalised_escalation,
                    "consistency_rules": [str(x) for x in item.get("consistency_rules", []) if str(x).strip()],
                }
            )

        raw_constraints = data.get("chapter_constraints", [])
        if not isinstance(raw_constraints, list):
            raw_constraints = []
        normalised_constraints = []
        for idx, item in enumerate(raw_constraints, start=1):
            if not isinstance(item, dict):
                continue
            chapter = _coerce_positive_int(item.get("chapter"), idx)
            chapter = min(chapter, total_chapters)
            normalised_constraints.append(
                {
                    "chapter": chapter,
                    "must_show": [str(x) for x in item.get("must_show", []) if str(x).strip()],
                    "must_not_break": [str(x) for x in item.get("must_not_break", []) if str(x).strip()],
                }
            )

        global_risks = data.get("global_risks", [])
        if not isinstance(global_risks, list):
            global_risks = []

        return {
            "antagonists": normalised_antagonists or fallback["antagonists"],
            "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
            "global_risks": [str(x) for x in global_risks if str(x).strip()],
        }

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        if not isinstance(plan, dict):
            return ""

        antagonists = plan.get("antagonists", [])
        if not isinstance(antagonists, list):
            antagonists = []
        constraints = plan.get("chapter_constraints", [])
        if not isinstance(constraints, list):
            constraints = []

        lines = ["Antagonist Motivation Architect output for this chapter:"]
        for antagonist in antagonists:
            if not isinstance(antagonist, dict):
                continue
            escalation_plan = antagonist.get("escalation_plan", [])
            if not isinstance(escalation_plan, list):
                escalation_plan = []
            matching_steps = [
                step for step in escalation_plan
                if isinstance(step, dict) and _coerce_positive_int(step.get("chapter"), 0) == chapter_num
            ]
            if matching_steps:
                lines.append(
                    f"- {antagonist.get('character', '?')}: core={antagonist.get('motivation_core', '')}; "
                    f"goal={antagonist.get('external_goal', '')}; fear={antagonist.get('fear_trigger', '')}; "
                    f"moral_line={antagonist.get('moral_line', '')}"
                )
                pressure_points = antagonist.get("pressure_points", [])
                if isinstance(pressure_points, list) and pressure_points:
                    lines.append("  - Pressure points: " + "; ".join(str(x) for x in pressure_points[:4]))
                for step in matching_steps[:3]:
                    lines.append(
                        f"  - Escalation: action={step.get('action', '')}; tactic={step.get('tactic', '')}; "
                        f"motivation_link={step.get('motivation_link', '')}"
                    )
                rules = antagonist.get("consistency_rules", [])
                if isinstance(rules, list) and rules:
                    lines.append("  - Consistency rules: " + "; ".join(str(x) for x in rules[:4]))

        chapter_constraint = next(
            (
                item for item in constraints
                if isinstance(item, dict) and _coerce_positive_int(item.get("chapter"), 0) == chapter_num
            ),
            None,
        )
        if chapter_constraint:
            must_show = chapter_constraint.get("must_show", [])
            must_not_break = chapter_constraint.get("must_not_break", [])
            if must_show:
                lines.append("- Must show: " + "; ".join(str(x) for x in must_show[:6]))
            if must_not_break:
                lines.append("- Must not break: " + "; ".join(str(x) for x in must_not_break[:6]))

        risks = plan.get("global_risks", [])
        if isinstance(risks, list) and risks:
            lines.append("- Global motivation risks: " + "; ".join(str(x) for x in risks[:5]))

        return "\n".join(lines)


# -- Singleton & wrapper functions -------------------------------------------

_antagonist_motivation_agent = AntagonistMotivationAgent()


def plan_antagonist_motivation_plan(**kwargs: object) -> dict:
    return _antagonist_motivation_agent.plan(**kwargs)


def normalise_antagonist_motivation_plan(plan_data: dict, character_list: list[dict], chapter_list: list[dict]) -> dict:
    return _antagonist_motivation_agent.normalise(plan_data, character_list=character_list, chapter_list=chapter_list)


def get_chapter_antagonist_context(antagonist_motivation_plan: dict, chapter_num: int) -> str:
    return _antagonist_motivation_agent.get_chapter_context(antagonist_motivation_plan, chapter_num)

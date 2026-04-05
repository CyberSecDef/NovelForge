"""Story Architecture planning agent."""

from novelforge.agents.base import BaseAgent
from novelforge.agents.planning._helpers import (
    _coerce_positive_int,
    _safe_chapter_list,
    choose_story_architecture_mode,
    render_prompt,
)


class StoryArchitectureAgent(BaseAgent):
    name = "Story Architecture Planner"
    prompt_action = "Planning Story Architecture"

    def build_prompt(self, **ctx) -> list[dict]:
        """Build the story architecture planning prompt from outline and genre context."""
        title = ctx["title"]
        premise = ctx["premise"]
        genre = ctx["genre"]
        chapter_list = ctx["chapter_list"]
        special_instructions = ctx.get("special_instructions", "")

        total_chapters = max(1, len(chapter_list))
        architecture_mode = choose_story_architecture_mode(total_chapters)
        outline_text = "\n".join(
            f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
            for i, ch in enumerate(chapter_list)
        )
        return render_prompt(
            "story_architecture_planning",
            title=title,
            premise=premise,
            genre=genre,
            total_chapters=total_chapters,
            architecture_mode=architecture_mode,
            outline_text=outline_text,
            special_instructions=special_instructions or "",
        )

    def build_fallback(self, **ctx) -> dict:
        """Build a deterministic fallback using chapter list and total chapters."""
        chapter_list = ctx.get("chapter_list", [])
        total_chapters = ctx.get("total_chapters", max(1, len(chapter_list)))
        return self._build_fallback_impl(chapter_list, total_chapters)

    @staticmethod
    def _build_fallback_impl(chapter_list: list[dict], total_chapters: int) -> dict:
        """Create a deterministic fallback architecture if the planner fails."""
        total_chapters = max(1, total_chapters)
        architecture_type = choose_story_architecture_mode(total_chapters)

        if architecture_type == "four-act":
            split_one = max(1, round(total_chapters * 0.25))
            split_two = max(split_one + 1, round(total_chapters * 0.50)) if total_chapters > 2 else total_chapters
            split_three = max(split_two + 1, round(total_chapters * 0.75)) if total_chapters > 3 else total_chapters
            split_two = min(split_two, total_chapters)
            split_three = min(split_three, total_chapters)
            acts = [
                {
                    "act": 1,
                    "label": "Act I",
                    "chapter_start": 1,
                    "chapter_end": split_one,
                    "purpose": "Setup, hook, and inciting pressure.",
                    "escalation": "Introduce the core conflict with tightly limited operations.",
                },
                {
                    "act": 2,
                    "label": "Act II-A",
                    "chapter_start": min(split_one + 1, total_chapters),
                    "chapter_end": split_two,
                    "purpose": "First wave of pursuit, resistance, and complications.",
                    "escalation": "Increase pressure while keeping each chapter focused on one main turn.",
                },
                {
                    "act": 3,
                    "label": "Act II-B",
                    "chapter_start": min(split_two + 1, total_chapters),
                    "chapter_end": split_three,
                    "purpose": "After the midpoint reversal, consequences intensify and options narrow.",
                    "escalation": "Escalate consequences and compress the room for recovery.",
                },
                {
                    "act": 4,
                    "label": "Act III",
                    "chapter_start": min(split_three + 1, total_chapters),
                    "chapter_end": total_chapters,
                    "purpose": "Crisis, climax, and resolution.",
                    "escalation": "Deliver the decisive confrontation, then resolve remaining tension.",
                },
            ]
        else:
            split_one = max(1, round(total_chapters * 0.25))
            split_two = max(split_one + 1, round(total_chapters * 0.75)) if total_chapters > 2 else total_chapters
            split_two = min(split_two, total_chapters)
            acts = [
                {
                    "act": 1,
                    "label": "Act I",
                    "chapter_start": 1,
                    "chapter_end": split_one,
                    "purpose": "Setup, hook, and inciting disruption.",
                    "escalation": "Constrain the chapter actions and establish the conflict cleanly.",
                },
                {
                    "act": 2,
                    "label": "Act II",
                    "chapter_start": min(split_one + 1, total_chapters),
                    "chapter_end": split_two,
                    "purpose": "Escalation, midpoint reversal, and mounting complications.",
                    "escalation": "Increase pressure in measured steps, with each chapter creating a sharper problem.",
                },
                {
                    "act": 3,
                    "label": "Act III",
                    "chapter_start": min(split_two + 1, total_chapters),
                    "chapter_end": total_chapters,
                    "purpose": "Crisis, climax, and resolution.",
                    "escalation": "Convert accumulated pressure into decisive action and payoff.",
                },
            ]

        from novelforge.chapter_position import ChapterPosition

        _inciting = ChapterPosition.inciting_chapter(total_chapters)
        _midpoint = ChapterPosition.midpoint_chapter(total_chapters)
        _climax = ChapterPosition.climax_chapter(total_chapters)
        _resolution = ChapterPosition.resolution_chapter(total_chapters)

        def _act_for_chapter(chapter_num: int) -> dict[str, object]:
            """Return the act dict whose chapter range contains *chapter_num*."""
            for act in acts:
                if int(act["chapter_start"]) <= chapter_num <= int(act["chapter_end"]):  # type: ignore[call-overload]
                    return act
            return acts[-1]

        chapter_plan = []
        for idx, chapter in enumerate(_safe_chapter_list(chapter_list), start=1):
            chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
            act = _act_for_chapter(chapter_num)

            if chapter_num == _inciting:
                phase = "Inciting Incident"
                required_turn = "Inciting incident"
                operation_limit = 1
                escalation = "Disrupt the status quo with one irreversible turn."
            elif chapter_num == _midpoint:
                phase = "Midpoint Reversal"
                required_turn = "Midpoint reversal"
                operation_limit = 2
                escalation = "Deliver a reversal that changes the protagonist's understanding or options."
            elif chapter_num == _climax:
                phase = "Climax Build"
                required_turn = "Climax setup"
                operation_limit = 2
                escalation = "Narrow choices and force commitment to the final confrontation."
            elif chapter_num == _resolution:
                phase = "Resolution"
                required_turn = "Resolution"
                operation_limit = 1
                escalation = "Resolve consequences and land the emotional aftermath."
            elif chapter_num < _midpoint:
                phase = "Escalation"
                required_turn = "None"
                operation_limit = 1
                escalation = "Advance the conflict through one clear pressure increase."
            else:
                phase = "Complication"
                required_turn = "None"
                operation_limit = 2 if chapter_num >= max(2, total_chapters - 2) else 1
                escalation = "Tighten consequences and hand a harder problem into the next chapter."

            chapter_plan.append(
                {
                    "number": chapter_num,
                    "title": chapter.get("title", f"Chapter {chapter_num}"),
                    "act": act["label"],
                    "phase": phase,
                    "purpose": chapter.get("summary", "") or act["purpose"],
                    "escalation": escalation,
                    "operation_limit": operation_limit,
                    "required_turn": required_turn,
                    "carry_forward": "End by handing the protagonist a sharper next problem.",
                }
            )

        return {
            "architecture_type": architecture_type,
            "acts": acts,
            "global_turns": {
                "inciting_incident": {
                    "chapter": _inciting,
                    "detail": "The core conflict becomes unavoidable.",
                },
                "midpoint_reversal": {
                    "chapter": _midpoint,
                    "detail": "A major revelation or reversal changes the trajectory.",
                },
                "climax": {
                    "chapter": _climax,
                    "detail": "The decisive confrontation reaches its peak.",
                },
                "resolution": {
                    "chapter": _resolution,
                    "detail": "Aftermath and payoff settle the story.",
                },
            },
            "chapter_plan": sorted(chapter_plan, key=lambda item: item["number"]),
        }

    def normalise(self, data: dict, **ctx) -> dict:
        """Validate and merge LLM architecture output with deterministic fallback."""
        chapter_list = ctx["chapter_list"]
        total_chapters = ctx.get("total_chapters", max(1, len(chapter_list)))
        fallback = self._build_fallback_impl(chapter_list, total_chapters)
        if not isinstance(data, dict):
            return fallback

        architecture_type = data.get("architecture_type")
        if architecture_type not in {"three-act", "four-act"}:
            architecture_type = fallback["architecture_type"]

        acts = data.get("acts")
        if not isinstance(acts, list) or not acts:
            acts = fallback["acts"]

        global_turns = data.get("global_turns")
        if not isinstance(global_turns, dict):
            global_turns = fallback["global_turns"]

        raw_chapter_plan = data.get("chapter_plan")
        if not isinstance(raw_chapter_plan, list):
            raw_chapter_plan = data.get("chapters", [])
        raw_map = {
            _coerce_positive_int(item.get("number"), idx + 1): item
            for idx, item in enumerate(raw_chapter_plan)
            if isinstance(item, dict)
        }
        fallback_map = {item["number"]: item for item in fallback["chapter_plan"]}
        fallback_first = fallback["chapter_plan"][0]

        merged_plan = []
        safe_chapter_list = _safe_chapter_list(chapter_list)
        for idx, chapter in enumerate(safe_chapter_list, start=1):
            chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
            fallback_item = fallback_map.get(chapter_num, fallback_first)
            planner_item = raw_map.get(chapter_num, {})

            merged_plan.append(
                {
                    "number": chapter_num,
                    "title": chapter.get("title", fallback_item.get("title", f"Chapter {chapter_num}")),
                    "act": str(planner_item.get("act") or fallback_item["act"]),
                    "phase": str(planner_item.get("phase") or fallback_item["phase"]),
                    "purpose": str(
                        planner_item.get("purpose")
                        or planner_item.get("summary")
                        or chapter.get("summary", "")
                        or fallback_item["purpose"]
                    ),
                    "escalation": str(
                        planner_item.get("escalation")
                        or planner_item.get("escalation_target")
                        or fallback_item["escalation"]
                    ),
                    "operation_limit": _coerce_positive_int(
                        planner_item.get("operation_limit"), fallback_item["operation_limit"]
                    ),
                    "required_turn": str(
                        planner_item.get("required_turn")
                        or planner_item.get("turn")
                        or fallback_item["required_turn"]
                    ),
                    "carry_forward": str(
                        planner_item.get("carry_forward")
                        or planner_item.get("handoff")
                        or fallback_item["carry_forward"]
                    ),
                }
            )

        return {
            "architecture_type": architecture_type,
            "acts": acts,
            "global_turns": global_turns,
            "chapter_plan": sorted(merged_plan, key=lambda item: item["number"]),
        }

    def plan(self, **ctx) -> dict:
        """Inject total_chapters into context, then delegate to BaseAgent.plan()."""
        chapter_list = ctx.get("chapter_list", [])
        total_chapters = max(1, len(chapter_list))
        ctx["total_chapters"] = total_chapters
        return super().plan(**ctx)

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        """Format the architecture plan as a prompt snippet for a specific chapter."""
        if not isinstance(plan, dict):
            return ""

        chapter_entry = next(
            (
                item for item in plan.get("chapter_plan", [])
                if _coerce_positive_int(item.get("number"), -1) == chapter_num
            ),
            None,
        )
        if not chapter_entry:
            return ""

        lines = [
            "Story Architecture Planner output for this chapter:",
            f"- Architecture model: {plan.get('architecture_type', 'three-act')}",
            f"- Act: {chapter_entry.get('act', '')}",
            f"- Phase: {chapter_entry.get('phase', '')}",
            f"- Chapter purpose: {chapter_entry.get('purpose', '')}",
            f"- Escalation target: {chapter_entry.get('escalation', '')}",
            f"- Major operations limit: {chapter_entry.get('operation_limit', 1)}",
            f"- Required turning point: {chapter_entry.get('required_turn', 'None')}",
            f"- Carry forward: {chapter_entry.get('carry_forward', '')}",
        ]

        global_turns = plan.get("global_turns", {})
        for turn_key, label in (
            ("inciting_incident", "Inciting incident"),
            ("midpoint_reversal", "Midpoint reversal"),
            ("climax", "Climax"),
            ("resolution", "Resolution"),
        ):
            turn = global_turns.get(turn_key)
            if isinstance(turn, dict):
                turn_chapter = _coerce_positive_int(turn.get("chapter"), 0)
                turn_detail = str(turn.get("detail", "")).strip()
                if turn_chapter:
                    suffix = f" – {turn_detail}" if turn_detail else ""
                    lines.append(f"- {label}: Chapter {turn_chapter}{suffix}")

        return "\n".join(lines)


# -- Singleton & wrapper functions -------------------------------------------

_story_architecture_agent = StoryArchitectureAgent()


def plan_story_architecture(**kwargs: object) -> dict:
    """Delegate to the singleton StoryArchitectureAgent instance."""
    return _story_architecture_agent.plan(**kwargs)


def normalise_story_architecture(architecture_data: dict, chapter_list: list[dict], total_chapters: int) -> dict:
    """Delegate normalisation to the singleton StoryArchitectureAgent."""
    return _story_architecture_agent.normalise(architecture_data, chapter_list=chapter_list, total_chapters=total_chapters)


def get_chapter_architecture_context(story_architecture: dict, chapter_num: int) -> str:
    """Delegate context formatting to the singleton StoryArchitectureAgent."""
    return _story_architecture_agent.get_chapter_context(story_architecture, chapter_num)

"""
Planning agents for NovelForge: story architecture, timeline, character fate,
character arcs, antagonist motivation, technology rules, theme reinforcement,
and POV/focal character planning.
"""

import json
import logging

from novelforge.agents.base import BaseAgent
from novelforge.llm.client import call_llm, parse_llm_json
from novelforge.llm.prompts import render_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(role: str) -> dict[str, str]:
    return {"role": "system", "content": role}


def choose_story_architecture_mode(total_chapters: int) -> str:
    """Choose a strict act model based on project size."""
    return "four-act" if total_chapters >= 16 else "three-act"


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        coerced = int(value)
        return coerced if coerced > 0 else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Story Architecture Agent
# ---------------------------------------------------------------------------

class StoryArchitectureAgent(BaseAgent):
    name = "Story Architecture Planner"
    prompt_action = "Planning Story Architecture"

    def build_prompt(self, **ctx) -> list[dict]:
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

        def _act_for_chapter(chapter_num: int) -> dict:
            for act in acts:
                if act["chapter_start"] <= chapter_num <= act["chapter_end"]:
                    return act
            return acts[-1]

        chapter_plan = []
        for idx, chapter in enumerate(chapter_list or [{"number": 1, "title": "Chapter 1", "summary": ""}], start=1):
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

        merged_plan = []
        safe_chapter_list = chapter_list or [{"number": 1, "title": "Chapter 1", "summary": ""}]
        for idx, chapter in enumerate(safe_chapter_list, start=1):
            chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
            fallback_item = fallback_map.get(chapter_num, fallback["chapter_plan"][0])
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
        chapter_list = ctx.get("chapter_list", [])
        total_chapters = max(1, len(chapter_list))
        ctx["total_chapters"] = total_chapters
        return super().plan(**ctx)

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
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


# ---------------------------------------------------------------------------
# Master Timeline Agent
# ---------------------------------------------------------------------------

class MasterTimelineAgent(BaseAgent):
    name = "Master Timeline Builder"
    prompt_action = "Planning Master Timeline"

    def build_prompt(self, **ctx) -> list[dict]:
        title = ctx["title"]
        premise = ctx["premise"]
        genre = ctx["genre"]
        chapter_list = ctx["chapter_list"]
        character_list = ctx.get("character_list", [])
        special_instructions = ctx.get("special_instructions", "")

        chapter_text = "\n".join(
            f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
            for i, ch in enumerate(chapter_list)
        )
        characters_text = "\n".join(
            f"- {c.get('name', '?')}: role={c.get('role', '')}; arc={c.get('arc', '')}; background={c.get('background', '')}"
            for c in character_list
        )
        if not characters_text.strip():
            characters_text = "- No explicit characters provided. Infer conservatively from outline."

        return render_prompt(
            "master_timeline_building",
            title=title,
            premise=premise,
            genre=genre,
            chapter_text=chapter_text,
            characters_text=characters_text,
            special_instructions=special_instructions or "",
        )

    def build_fallback(self, **ctx) -> dict:
        chapter_list = ctx.get("chapter_list", [])
        character_list = ctx.get("character_list", [])
        return self._build_fallback_impl(chapter_list, character_list)

    @staticmethod
    def _build_fallback_impl(chapter_list: list[dict], character_list: list[dict]) -> dict:
        """Deterministic fallback timeline when planner output is unavailable."""
        safe_chapters = chapter_list or [{"number": 1, "title": "Chapter 1", "summary": ""}]
        safe_characters = character_list or []

        ledger = []
        chapter_constraints = []
        for idx, chapter in enumerate(safe_chapters, start=1):
            chapter_num = _coerce_positive_int(chapter.get("number", idx), idx)
            summary = str(chapter.get("summary", "")).strip()
            ledger.append(
                {
                    "index": idx,
                    "chapter": chapter_num,
                    "event_type": "operation",
                    "event": summary or f"Primary operation for Chapter {chapter_num}",
                    "actors": [],
                    "targets": [],
                    "state_changes": [],
                    "continuity_note": "Advance one major operation and preserve prior state continuity.",
                }
            )
            chapter_constraints.append(
                {
                    "chapter": chapter_num,
                    "must_include": ["At least one continuity-consistent consequence from prior events."],
                    "must_avoid": ["Contradicting established character status without explicit transition."],
                }
            )

        character_states = []
        for character in safe_characters:
            name = str(character.get("name", "")).strip()
            if not name:
                continue
            character_states.append(
                {
                    "character": name,
                    "status": "active",
                    "location": "unknown",
                    "last_event_index": 0,
                    "notes": "Baseline state before chapter drafting.",
                }
            )

        return {
            "ledger": ledger,
            "character_states": character_states,
            "chapter_constraints": chapter_constraints,
            "continuity_risks": [],
        }

    def normalise(self, data: dict, **ctx) -> dict:
        chapter_list = ctx.get("chapter_list", [])
        character_list = ctx.get("character_list", [])
        fallback = self._build_fallback_impl(chapter_list, character_list)
        if not isinstance(data, dict):
            return fallback

        ledger = data.get("ledger", [])
        if not isinstance(ledger, list):
            ledger = []

        normalised_ledger = []
        for idx, event in enumerate(ledger, start=1):
            if not isinstance(event, dict):
                continue
            chapter_num = _coerce_positive_int(event.get("chapter"), 1)
            state_changes = event.get("state_changes", [])
            if not isinstance(state_changes, list):
                state_changes = []
            normalised_ledger.append(
                {
                    "index": _coerce_positive_int(event.get("index"), idx),
                    "chapter": chapter_num,
                    "event_type": str(event.get("event_type", "other")),
                    "event": str(event.get("event", "")).strip(),
                    "actors": [str(a) for a in event.get("actors", []) if str(a).strip()],
                    "targets": [str(t) for t in event.get("targets", []) if str(t).strip()],
                    "state_changes": [sc for sc in state_changes if isinstance(sc, dict)],
                    "continuity_note": str(event.get("continuity_note", "")).strip(),
                }
            )

        character_states = data.get("character_states", [])
        if not isinstance(character_states, list):
            character_states = []
        normalised_character_states = []
        for state in character_states:
            if not isinstance(state, dict):
                continue
            name = str(state.get("character", "")).strip()
            if not name:
                continue
            normalised_character_states.append(
                {
                    "character": name,
                    "status": str(state.get("status", "active")),
                    "location": str(state.get("location", "unknown")),
                    "last_event_index": _coerce_positive_int(state.get("last_event_index"), 0),
                    "notes": str(state.get("notes", "")).strip(),
                }
            )

        chapter_constraints = data.get("chapter_constraints", [])
        if not isinstance(chapter_constraints, list):
            chapter_constraints = []
        normalised_constraints = []
        for idx, constraint in enumerate(chapter_constraints, start=1):
            if not isinstance(constraint, dict):
                continue
            normalised_constraints.append(
                {
                    "chapter": _coerce_positive_int(constraint.get("chapter"), idx),
                    "must_include": [str(x) for x in constraint.get("must_include", []) if str(x).strip()],
                    "must_avoid": [str(x) for x in constraint.get("must_avoid", []) if str(x).strip()],
                }
            )

        continuity_risks = data.get("continuity_risks", [])
        if not isinstance(continuity_risks, list):
            continuity_risks = []

        merged = {
            "ledger": normalised_ledger or fallback["ledger"],
            "character_states": normalised_character_states or fallback["character_states"],
            "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
            "continuity_risks": [str(r) for r in continuity_risks if str(r).strip()],
        }
        return merged

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        if not isinstance(plan, dict):
            return ""

        ledger = plan.get("ledger", [])
        if not isinstance(ledger, list):
            ledger = []
        chapter_events = [
            event for event in ledger
            if isinstance(event, dict) and _coerce_positive_int(event.get("chapter"), 0) == chapter_num
        ]
        recent_events = [
            event for event in ledger
            if isinstance(event, dict) and _coerce_positive_int(event.get("chapter"), 0) < chapter_num
        ]
        recent_events = recent_events[-3:]

        constraints = plan.get("chapter_constraints", [])
        if not isinstance(constraints, list):
            constraints = []
        chapter_constraint = next(
            (
                c for c in constraints
                if isinstance(c, dict) and _coerce_positive_int(c.get("chapter"), 0) == chapter_num
            ),
            None,
        )

        states = plan.get("character_states", [])
        if not isinstance(states, list):
            states = []

        lines = ["Master Timeline Builder output for this chapter:"]
        for event in recent_events:
            lines.append(
                f"- Prior event (Ch {event.get('chapter')}): {event.get('event', '')} "
                f"[{event.get('event_type', 'other')}]"
            )
        for event in chapter_events:
            lines.append(
                f"- Planned chapter event: {event.get('event', '')} "
                f"[{event.get('event_type', 'other')}]"
            )

        if chapter_constraint:
            must_include = chapter_constraint.get("must_include", [])
            must_avoid = chapter_constraint.get("must_avoid", [])
            if must_include:
                lines.append("- Must include: " + "; ".join(str(x) for x in must_include))
            if must_avoid:
                lines.append("- Must avoid: " + "; ".join(str(x) for x in must_avoid))

        if states:
            lines.append("- Character state ledger (current):")
            for state in states[:12]:
                lines.append(
                    f"  - {state.get('character', '?')}: status={state.get('status', 'active')}, "
                    f"location={state.get('location', 'unknown')}, notes={state.get('notes', '')}"
                )

        risks = plan.get("continuity_risks", [])
        if isinstance(risks, list) and risks:
            lines.append("- Continuity risks: " + "; ".join(str(r) for r in risks[:5]))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Character Fate Registry Agent
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Character Arc Planner Agent
# ---------------------------------------------------------------------------

class CharacterArcPlanAgent(BaseAgent):
    name = "Character Arc Planner"
    prompt_action = "Planning Character Arcs"

    def build_prompt(self, **ctx) -> list[dict]:
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
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        return self._build_fallback_impl(character_list, chapter_list)

    @staticmethod
    def _build_fallback_impl(character_list: list[dict], chapter_list: list[dict]) -> dict:
        """Deterministic fallback character arc plan."""
        from novelforge.chapter_position import ChapterPosition
        total_chapters = max(1, len(chapter_list))
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
        for idx, chapter in enumerate(chapter_list or [{"number": 1}], start=1):
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


# ---------------------------------------------------------------------------
# Antagonist Motivation Agent
# ---------------------------------------------------------------------------

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
        total_chapters = max(1, len(chapter_list))

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
        for idx, chapter in enumerate(chapter_list or [{"number": 1}], start=1):
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


# ---------------------------------------------------------------------------
# Technology Rules Agent
# ---------------------------------------------------------------------------

class TechnologyRulesAgent(BaseAgent):
    name = "Technology Rules Designer"
    prompt_action = "Planning Technology Rules"

    def build_prompt(self, **ctx) -> list[dict]:
        title = ctx["title"]
        premise = ctx["premise"]
        genre = ctx["genre"]
        chapter_list = ctx["chapter_list"]
        special_instructions = ctx.get("special_instructions", "")

        chapters_text = "\n".join(
            f"Chapter {ch.get('number', i + 1)}: {ch.get('title', f'Chapter {i + 1}')} – {ch.get('summary', '')}"
            for i, ch in enumerate(chapter_list)
        )
        return render_prompt("technology_rules", title=title, premise=premise, genre=genre,
                             chapters_text=chapters_text, special_instructions=special_instructions or "")

    def build_fallback(self, **ctx) -> dict:
        chapter_list = ctx.get("chapter_list", [])
        return self._build_fallback_impl(chapter_list)

    @staticmethod
    def _build_fallback_impl(chapter_list: list[dict]) -> dict:
        total_chapters = max(1, len(chapter_list))
        systems = [
            {
                "name": "Primary Surveillance Grid",
                "purpose": "Detect movements and anomalies across controlled zones.",
                "latency_ms": 1800,
                "detection_methods": ["Pattern analysis", "Fixed checkpoint scans"],
                "detection_blind_spots": ["Sensor dead angles", "Heavy weather or smoke"],
                "resource_constraints": ["Finite compute budget", "Nightly recalibration windows"],
                "operational_limits": ["Cannot track all targets in real time", "False positives under crowd density"],
                "failure_modes": ["Overload under coordinated decoys", "Delayed alert propagation"],
                "countermeasures": ["Manual review queue", "Tiered alert escalation"],
                "forbidden_capabilities": ["Instant omniscient tracking", "Retroactive perfect reconstruction"],
            }
        ]
        chapter_constraints = []
        for idx, chapter in enumerate(chapter_list or [{"number": 1}], start=1):
            chapter_num = _coerce_positive_int(chapter.get("number"), idx)
            chapter_constraints.append({
                "chapter": chapter_num,
                "must_respect": ["Technology outcomes must follow stated latency and operational limits."],
                "must_not_allow": ["Do not grant instant detection or infinite processing without explicit setup."],
            })
        return {
            "systems": systems,
            "global_constraints": [
                "Every tech action has delay, uncertainty, or resource cost.",
                "Capabilities cannot exceed declared operational limits.",
            ],
            "chapter_constraints": chapter_constraints,
            "continuity_risks": [],
        }

    def normalise(self, data: dict, **ctx) -> dict:
        chapter_list = ctx.get("chapter_list", [])
        fallback = self._build_fallback_impl(chapter_list)
        if not isinstance(data, dict):
            return fallback

        raw_systems = data.get("systems", [])
        if not isinstance(raw_systems, list):
            raw_systems = []
        normalised_systems = []
        seen_names = set()
        for item in raw_systems:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            normalised_systems.append({
                "name": name,
                "purpose": str(item.get("purpose", "")).strip(),
                "latency_ms": _coerce_positive_int(item.get("latency_ms"), 1000),
                "detection_methods": [str(x) for x in item.get("detection_methods", []) if str(x).strip()],
                "detection_blind_spots": [str(x) for x in item.get("detection_blind_spots", []) if str(x).strip()],
                "resource_constraints": [str(x) for x in item.get("resource_constraints", []) if str(x).strip()],
                "operational_limits": [str(x) for x in item.get("operational_limits", []) if str(x).strip()],
                "failure_modes": [str(x) for x in item.get("failure_modes", []) if str(x).strip()],
                "countermeasures": [str(x) for x in item.get("countermeasures", []) if str(x).strip()],
                "forbidden_capabilities": [str(x) for x in item.get("forbidden_capabilities", []) if str(x).strip()],
            })

        global_constraints = data.get("global_constraints", [])
        if not isinstance(global_constraints, list):
            global_constraints = []

        raw_constraints = data.get("chapter_constraints", [])
        if not isinstance(raw_constraints, list):
            raw_constraints = []
        total_chapters = max(1, len(chapter_list))
        normalised_constraints = []
        for idx, item in enumerate(raw_constraints, start=1):
            if not isinstance(item, dict):
                continue
            chapter = _coerce_positive_int(item.get("chapter"), idx)
            chapter = min(chapter, total_chapters)
            normalised_constraints.append({
                "chapter": chapter,
                "must_respect": [str(x) for x in item.get("must_respect", []) if str(x).strip()],
                "must_not_allow": [str(x) for x in item.get("must_not_allow", []) if str(x).strip()],
            })

        continuity_risks = data.get("continuity_risks", [])
        if not isinstance(continuity_risks, list):
            continuity_risks = []

        return {
            "systems": normalised_systems or fallback["systems"],
            "global_constraints": [str(x) for x in global_constraints if str(x).strip()] or fallback["global_constraints"],
            "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
            "continuity_risks": [str(x) for x in continuity_risks if str(x).strip()],
        }

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        if not isinstance(plan, dict):
            return ""

        systems = plan.get("systems", [])
        if not isinstance(systems, list):
            systems = []
        lines = ["Technology Rules Designer output for this chapter:"]
        for system in systems[:6]:
            if not isinstance(system, dict):
                continue
            lines.append(f"- {system.get('name', '?')}: latency={system.get('latency_ms', 0)}ms; purpose={system.get('purpose', '')}")
            for key, label in (
                ("operational_limits", "Operational limits"),
                ("resource_constraints", "Resource constraints"),
                ("detection_blind_spots", "Detection blind spots"),
                ("failure_modes", "Failure modes"),
                ("forbidden_capabilities", "Forbidden capabilities"),
            ):
                values = system.get(key, [])
                if isinstance(values, list) and values:
                    lines.append(f"  - {label}: " + "; ".join(str(x) for x in values[:4]))

        global_constraints = plan.get("global_constraints", [])
        if isinstance(global_constraints, list) and global_constraints:
            lines.append("- Global constraints: " + "; ".join(str(x) for x in global_constraints[:6]))

        chapter_constraints = plan.get("chapter_constraints", [])
        if isinstance(chapter_constraints, list):
            chapter_constraint = next(
                (item for item in chapter_constraints
                 if isinstance(item, dict) and _coerce_positive_int(item.get("chapter"), 0) == chapter_num),
                None,
            )
            if chapter_constraint:
                must_respect = chapter_constraint.get("must_respect", [])
                must_not_allow = chapter_constraint.get("must_not_allow", [])
                if must_respect:
                    lines.append("- Must respect: " + "; ".join(str(x) for x in must_respect[:6]))
                if must_not_allow:
                    lines.append("- Must not allow: " + "; ".join(str(x) for x in must_not_allow[:6]))

        risks = plan.get("continuity_risks", [])
        if isinstance(risks, list) and risks:
            lines.append("- Technology continuity risks: " + "; ".join(str(x) for x in risks[:5]))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Theme Reinforcement Agent
# ---------------------------------------------------------------------------

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
        fallback_themes = [
            {
                "name": "Identity Under Pressure",
                "description": "How characters maintain or lose their sense of self under systemic pressure.",
                "motifs": ["mirrors", "names", "documents"],
                "pillar_moments": ["Inciting incident", "Midpoint crisis", "Final moral choice"],
                "chapter_appearances": [
                    {"chapter": c.get("number", i + 1), "role": "background", "guidance": "Show character making a small compromise."}
                    for i, c in enumerate(chapter_list)
                ],
            },
            {
                "name": "Moral Compromise",
                "description": "The cost of choosing safety over principle.",
                "motifs": ["closed doors", "silence", "small betrayals"],
                "pillar_moments": ["First compromise", "Point of no return", "Reckoning"],
                "chapter_appearances": [
                    {"chapter": c.get("number", i + 1), "role": "background", "guidance": "Show institutional pressure shaping a decision."}
                    for i, c in enumerate(chapter_list)
                ],
            },
        ]
        chapter_constraints = [
            {
                "chapter": c.get("number", i + 1),
                "themes_present": ["Identity Under Pressure"],
                "thematic_guidance": "Reinforce the protagonist's internal conflict quietly.",
            }
            for i, c in enumerate(chapter_list)
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


# ---------------------------------------------------------------------------
# POV & Focal Character Agent
# ---------------------------------------------------------------------------

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

        return render_prompt(
            "pov_focal_character_planner",
            title=title, premise=premise, genre=genre,
            characters_text=characters_text, chapters_text=chapters_text,
            arc_text=arc_text, special_instructions=special_instructions or "",
        )

    def build_fallback(self, **ctx) -> dict:
        character_list = ctx.get("character_list", [])
        chapter_list = ctx.get("chapter_list", [])
        return self._build_fallback_impl(character_list, chapter_list)

    @staticmethod
    def _build_fallback_impl(character_list: list[dict], chapter_list: list[dict]) -> dict:
        safe_characters = [c for c in (character_list or []) if str(c.get("name", "")).strip()]
        if not safe_characters:
            safe_characters = [{"name": "Protagonist", "role": "protagonist"}]

        protagonist_names = []
        for c in safe_characters:
            role = str(c.get("role", "")).lower()
            if any(tag in role for tag in ("protagonist", "lead", "main", "hero")):
                protagonist_names.append(str(c.get("name", "")).strip())
        if not protagonist_names:
            protagonist_names = [str(safe_characters[0].get("name", "Protagonist")).strip()]

        total_chapters = max(1, len(chapter_list or []))
        chapter_pov_plan = []
        for idx, ch in enumerate(chapter_list or [{"number": 1, "title": "Chapter 1"}]):
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
        normalised_plan.sort(key=lambda x: x["chapter"])

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


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_story_architecture_agent = StoryArchitectureAgent()
_master_timeline_agent = MasterTimelineAgent()
_character_fate_registry_agent = CharacterFateRegistryAgent()
_character_arc_plan_agent = CharacterArcPlanAgent()
_antagonist_motivation_agent = AntagonistMotivationAgent()
_technology_rules_agent = TechnologyRulesAgent()
_theme_reinforcement_agent = ThemeReinforcementAgent()
_pov_focal_character_agent = PovFocalCharacterAgent()


# ---------------------------------------------------------------------------
# Backward-compatible module-level wrapper functions
# ---------------------------------------------------------------------------

# --- Story Architecture ---

def plan_story_architecture(**kwargs: object) -> dict:
    return _story_architecture_agent.plan(**kwargs)


def normalise_story_architecture(architecture_data: dict, chapter_list: list[dict], total_chapters: int) -> dict:
    return _story_architecture_agent.normalise(architecture_data, chapter_list=chapter_list, total_chapters=total_chapters)


def get_chapter_architecture_context(story_architecture: dict, chapter_num: int) -> str:
    return _story_architecture_agent.get_chapter_context(story_architecture, chapter_num)


# --- Master Timeline ---

def plan_master_timeline(**kwargs: object) -> dict:
    return _master_timeline_agent.plan(**kwargs)


def normalise_master_timeline(timeline_data: dict, chapter_list: list[dict], character_list: list[dict]) -> dict:
    return _master_timeline_agent.normalise(timeline_data, chapter_list=chapter_list, character_list=character_list)


def get_chapter_timeline_context(master_timeline: dict, chapter_num: int) -> str:
    return _master_timeline_agent.get_chapter_context(master_timeline, chapter_num)


# --- Character Fate Registry ---

def plan_character_fate_registry(**kwargs: object) -> dict:
    return _character_fate_registry_agent.plan(**kwargs)


def normalise_character_fate_registry(registry_data: dict, character_list: list[dict], total_chapters: int) -> dict:
    return _character_fate_registry_agent.normalise(registry_data, character_list=character_list, total_chapters=total_chapters)


def get_chapter_fate_context(character_fate_registry: dict, chapter_num: int) -> str:
    return _character_fate_registry_agent.get_chapter_context(character_fate_registry, chapter_num)


# --- Character Arc Plan ---

def plan_character_arc_plan(**kwargs: object) -> dict:
    return _character_arc_plan_agent.plan(**kwargs)


def normalise_character_arc_plan(arc_data: dict, character_list: list[dict], chapter_list: list[dict]) -> dict:
    return _character_arc_plan_agent.normalise(arc_data, character_list=character_list, chapter_list=chapter_list)


def get_chapter_arc_context(character_arc_plan: dict, chapter_num: int) -> str:
    return _character_arc_plan_agent.get_chapter_context(character_arc_plan, chapter_num)


# --- Antagonist Motivation ---

def plan_antagonist_motivation_plan(**kwargs: object) -> dict:
    return _antagonist_motivation_agent.plan(**kwargs)


def normalise_antagonist_motivation_plan(plan_data: dict, character_list: list[dict], chapter_list: list[dict]) -> dict:
    return _antagonist_motivation_agent.normalise(plan_data, character_list=character_list, chapter_list=chapter_list)


def get_chapter_antagonist_context(antagonist_motivation_plan: dict, chapter_num: int) -> str:
    return _antagonist_motivation_agent.get_chapter_context(antagonist_motivation_plan, chapter_num)


# --- Technology Rules ---

def plan_technology_rules(**kwargs: object) -> dict:
    return _technology_rules_agent.plan(**kwargs)


def normalise_technology_rules(technology_data: dict, chapter_list: list[dict]) -> dict:
    return _technology_rules_agent.normalise(technology_data, chapter_list=chapter_list)


def get_chapter_technology_context(technology_rules: dict, chapter_num: int) -> str:
    return _technology_rules_agent.get_chapter_context(technology_rules, chapter_num)


# --- Theme Reinforcement ---

def plan_theme_reinforcement(**kwargs: object) -> dict:
    return _theme_reinforcement_agent.plan(**kwargs)


def normalise_theme_reinforcement(theme_data: dict, chapter_list: list[dict]) -> dict:
    return _theme_reinforcement_agent.normalise(theme_data, chapter_list=chapter_list)


def get_chapter_theme_context(theme_reinforcement: dict, chapter_num: int) -> str:
    return _theme_reinforcement_agent.get_chapter_context(theme_reinforcement, chapter_num)


# --- POV & Focal Character ---

def plan_pov_focal_character(**kwargs: object) -> dict:
    return _pov_focal_character_agent.plan(**kwargs)


def normalise_pov_focal_character_plan(plan_data: dict, character_list: list[dict], chapter_list: list[dict]) -> dict:
    return _pov_focal_character_agent.normalise(plan_data, character_list=character_list, chapter_list=chapter_list)


def get_chapter_pov_context(pov_plan: dict, chapter_num: int) -> str:
    return _pov_focal_character_agent.get_chapter_context(pov_plan, chapter_num)

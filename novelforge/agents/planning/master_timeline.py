"""Master Timeline planning agent."""

from novelforge.agents.base import BaseAgent
from novelforge.agents.planning._helpers import (
    _coerce_positive_int,
    _safe_chapter_list,
    render_prompt,
)


class MasterTimelineAgent(BaseAgent):
    name = "Master Timeline Builder"
    prompt_action = "Planning Master Timeline"

    def build_prompt(self, **ctx) -> list[dict]:
        """Build the master timeline planning prompt from outline and character context."""
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
        """Build a deterministic fallback timeline from chapter and character lists."""
        chapter_list = ctx.get("chapter_list", [])
        character_list = ctx.get("character_list", [])
        return self._build_fallback_impl(chapter_list, character_list)

    @staticmethod
    def _build_fallback_impl(chapter_list: list[dict], character_list: list[dict]) -> dict:
        """Deterministic fallback timeline when planner output is unavailable."""
        safe_chapters = _safe_chapter_list(chapter_list)
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
        """Validate and merge LLM timeline output with deterministic fallback."""
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
        """Format timeline events and character states as a prompt snippet for a chapter."""
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


# -- Singleton & wrapper functions -------------------------------------------

_master_timeline_agent = MasterTimelineAgent()


def plan_master_timeline(**kwargs: object) -> dict:
    """Delegate to the singleton MasterTimelineAgent instance."""
    return _master_timeline_agent.plan(**kwargs)


def normalise_master_timeline(timeline_data: dict, chapter_list: list[dict], character_list: list[dict]) -> dict:
    """Delegate normalisation to the singleton MasterTimelineAgent."""
    return _master_timeline_agent.normalise(timeline_data, chapter_list=chapter_list, character_list=character_list)


def get_chapter_timeline_context(master_timeline: dict, chapter_num: int) -> str:
    """Delegate context formatting to the singleton MasterTimelineAgent."""
    return _master_timeline_agent.get_chapter_context(master_timeline, chapter_num)

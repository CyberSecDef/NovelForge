"""Technology Rules planning agent."""

from novelforge.agents.base import BaseAgent
from novelforge.agents.planning._helpers import (
    _coerce_positive_int,
    _safe_chapter_list,
    render_prompt,
)


class TechnologyRulesAgent(BaseAgent):
    name = "Technology Rules Designer"
    prompt_action = "Planning Technology Rules"

    def build_prompt(self, **ctx) -> list[dict]:
        """Build the technology rules prompt from outline and genre context."""
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
        """Build a deterministic fallback technology ruleset from the chapter list."""
        chapter_list = ctx.get("chapter_list", [])
        return self._build_fallback_impl(chapter_list)

    @staticmethod
    def _build_fallback_impl(chapter_list: list[dict]) -> dict:
        """Create a deterministic fallback with a single surveillance system and chapter constraints."""
        safe_chapter_list = _safe_chapter_list(chapter_list)
        total_chapters = max(1, len(safe_chapter_list))
        # Minimal default rules — intentionally sparse. The fallback exists
        # only when the LLM call fails entirely; a real novel's LLM output
        # will supply 4-10 concrete, genre-appropriate rules.
        rules = [
            {
                "capability": "Primary detection apparatus",
                "cost": "Finite compute budget; cannot run every check simultaneously.",
                "forbidden": "Instant omniscient tracking of all targets; retroactive perfect reconstruction of past events.",
                "exception": "none",
            }
        ]
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
        for idx, chapter in enumerate(safe_chapter_list, start=1):
            chapter_num = _coerce_positive_int(chapter.get("number"), idx)
            chapter_constraints.append({
                "chapter": chapter_num,
                "must_respect": ["Technology outcomes must follow stated latency and operational limits."],
                "must_not_allow": ["Do not grant instant detection or infinite processing without explicit setup."],
            })
        return {
            "rules": rules,
            "systems": systems,
            "global_constraints": [
                "Every tech action has delay, uncertainty, or resource cost.",
                "Capabilities cannot exceed declared operational limits.",
            ],
            "chapter_constraints": chapter_constraints,
            "continuity_risks": [],
        }

    def normalise(self, data: dict, **ctx) -> dict:
        """Validate and merge LLM technology rules with deterministic fallback."""
        chapter_list = ctx.get("chapter_list", [])
        fallback = self._build_fallback_impl(chapter_list)
        if not isinstance(data, dict):
            return fallback

        # Normalise the authoritative `rules` array. Each rule must have a
        # non-empty `capability`; other fields default to empty string or
        # "none" so the gatekeeper can always render them.
        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            raw_rules = []
        normalised_rules: list[dict] = []
        seen_capabilities: set[str] = set()
        for item in raw_rules:
            if not isinstance(item, dict):
                continue
            capability = str(item.get("capability", "")).strip()
            if not capability or capability.lower() in seen_capabilities:
                continue
            seen_capabilities.add(capability.lower())
            exception = str(item.get("exception", "")).strip() or "none"
            normalised_rules.append({
                "capability": capability,
                "cost": str(item.get("cost", "")).strip(),
                "forbidden": str(item.get("forbidden", "")).strip(),
                "exception": exception,
            })

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
            "rules": normalised_rules or fallback["rules"],
            "systems": normalised_systems or fallback["systems"],
            "global_constraints": [str(x) for x in global_constraints if str(x).strip()] or fallback["global_constraints"],
            "chapter_constraints": normalised_constraints or fallback["chapter_constraints"],
            "continuity_risks": [str(x) for x in continuity_risks if str(x).strip()],
        }

    def get_chapter_context(self, plan: dict, chapter_num: int) -> str:
        """Format world rules, systems, and constraints as a prompt snippet for a chapter.

        Rules are surfaced at the top as the primary, enforceable constraints
        the Continuity Gatekeeper should match chapter plans against. Other
        fields (systems, global/chapter constraints, continuity risks) are
        supplementary detail.
        """
        if not isinstance(plan, dict):
            return ""

        lines = ["Technology Rules Designer output for this chapter:"]

        # Primary: enforceable rules (capability / cost / forbidden / exception).
        rules = plan.get("rules", [])
        if isinstance(rules, list) and rules:
            lines.append("- ENFORCEABLE RULES:")
            for rule in rules[:10]:
                if not isinstance(rule, dict):
                    continue
                capability = str(rule.get("capability", "?")).strip() or "?"
                cost = str(rule.get("cost", "")).strip()
                forbidden = str(rule.get("forbidden", "")).strip()
                exception = str(rule.get("exception", "")).strip() or "none"
                lines.append(f"  - Rule '{capability}':")
                if cost:
                    lines.append(f"    · cost: {cost}")
                if forbidden:
                    lines.append(f"    · forbidden: {forbidden}")
                lines.append(f"    · exception: {exception}")

        systems = plan.get("systems", [])
        if not isinstance(systems, list):
            systems = []
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


# -- Singleton & wrapper functions -------------------------------------------

_technology_rules_agent = TechnologyRulesAgent()


def plan_technology_rules(**kwargs: object) -> dict:
    """Delegate to the singleton TechnologyRulesAgent instance."""
    return _technology_rules_agent.plan(**kwargs)


def normalise_technology_rules(technology_data: dict, chapter_list: list[dict]) -> dict:
    """Delegate normalisation to the singleton TechnologyRulesAgent."""
    return _technology_rules_agent.normalise(technology_data, chapter_list=chapter_list)


def get_chapter_technology_context(technology_rules: dict, chapter_num: int) -> str:
    """Delegate context formatting to the singleton TechnologyRulesAgent."""
    return _technology_rules_agent.get_chapter_context(technology_rules, chapter_num)

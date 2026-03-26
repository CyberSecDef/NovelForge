"""
Centralized chapter position calculations.

Provides a single source of truth for narrative phase assignments,
act boundaries, and structural landmark chapters (inciting incident,
midpoint, climax, resolution).
"""


class ChapterPosition:
    """Calculate narrative position metadata for a chapter within a novel."""

    # 9-phase narrative structure boundaries (as percentage of total chapters)
    _PHASE_BOUNDARIES = [
        (10, "Hook"),
        (15, "Setup"),
        (25, "Inciting Incident"),
        (40, "Rising Action"),
        (50, "Midpoint Shift"),
        (65, "Complications"),
        (75, "Crisis"),
        (90, "Climax"),
        (100, "Resolution"),
    ]

    def __init__(self, chapter_num: int, total_chapters: int) -> None:
        self.chapter_num = chapter_num
        self.total_chapters = max(1, total_chapters)
        self.position_pct = (chapter_num / self.total_chapters * 100) if self.total_chapters > 0 else 50

    # --- Landmark chapters ---

    @staticmethod
    def inciting_chapter(total_chapters: int) -> int:
        return 2 if total_chapters >= 4 else 1

    @staticmethod
    def midpoint_chapter(total_chapters: int) -> int:
        return max(1, min(total_chapters, round((total_chapters + 1) / 2)))

    @staticmethod
    def climax_chapter(total_chapters: int) -> int:
        return max(1, total_chapters - 1) if total_chapters > 1 else 1

    @staticmethod
    def resolution_chapter(total_chapters: int) -> int:
        return total_chapters

    # --- Phase and act classification ---

    def get_phase(self) -> str:
        """Return the narrative phase name for this chapter position."""
        for boundary, phase in self._PHASE_BOUNDARIES:
            if self.position_pct <= boundary:
                return phase
        return "Resolution"

    def get_act(self, four_act: bool = False) -> str:
        """Return the act label for this chapter position."""
        if four_act:
            if self.position_pct <= 25:
                return "Act I"
            elif self.position_pct <= 50:
                return "Act II-A"
            elif self.position_pct <= 75:
                return "Act II-B"
            else:
                return "Act III"
        else:
            if self.position_pct <= 25:
                return "Act I"
            elif self.position_pct <= 75:
                return "Act II"
            else:
                return "Act III"

    def get_structure_phase_hint(self) -> str:
        """Return the broad phase hint used by the structure agent."""
        if self.position_pct <= 25:
            return "Beginning (Hook / Setup / Inciting Incident)"
        elif self.position_pct <= 75:
            return "Middle (Rising Action / Midpoint Shift / Complications)"
        else:
            return "End (Crisis / Climax / Resolution)"

    def get_escalation_target(self) -> str:
        """Return the escalation guidance used by the momentum agent."""
        if self.position_pct <= 25:
            return "establish foundational threat and personal stakes"
        elif self.position_pct <= 50:
            return "deepen the cost of failure and raise the personal price"
        elif self.position_pct <= 75:
            return "force irreversible decisions and close off safe options"
        else:
            return "push stakes to maximum \u2013 survival, identity, or irreversible loss"

    # --- Zone checks ---

    def is_climax_zone(self) -> bool:
        """True if this chapter is in the climax zone (last ~15% of the novel)."""
        return self.position_pct > 85

    def is_opening(self) -> bool:
        """True if this chapter is in the opening zone (first ~15%)."""
        return self.position_pct <= 15

    def is_midpoint(self) -> bool:
        """True if this chapter is near the midpoint (45-55%)."""
        return 45 <= self.position_pct <= 55

    def is_before_midpoint(self) -> bool:
        """True if this chapter is before the midpoint."""
        return self.position_pct < 50

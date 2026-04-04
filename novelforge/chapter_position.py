"""
Centralized chapter position calculations.

Provides a single source of truth for narrative phase assignments,
act boundaries, and structural landmark chapters (inciting incident,
midpoint, climax, resolution).

Canonical structural model
--------------------------
Four landmark chapters anchor the narrative arc for any novel length:

* **Inciting Incident** – ``inciting_chapter(n)``: chapter 2 for novels of
  4+ chapters, chapter 1 otherwise.
* **Midpoint Shift** – ``midpoint_chapter(n)``: the rounded midpoint of the
  chapter range.
* **Climax** – ``climax_chapter(n)``: second-to-last chapter (or chapter 1
  for a single-chapter novel).
* **Resolution** – ``resolution_chapter(n)``: the final chapter.

``get_phase()`` is defined entirely in terms of these landmarks so that
phase classification and landmark helpers are always consistent:

* ``get_phase(inciting_chapter(n), n) == "Inciting Incident"``
* ``get_phase(midpoint_chapter(n), n) == "Midpoint Shift"``  (when distinct
  from the inciting landmark)
* ``get_phase(climax_chapter(n), n) == "Climax"``
* ``get_phase(resolution_chapter(n), n) == "Resolution"``

The nine phases fill the gaps between landmarks as follows::

    Hook → Setup → Inciting Incident → Rising Action →
    Midpoint Shift → Complications → Crisis → Climax → Resolution

For short novels some intermediate phases collapse (e.g. a 4-chapter novel
yields ``[Hook, Inciting Incident, Climax, Resolution]``).
"""


class ChapterPosition:
    """Calculate narrative position metadata for a chapter within a novel.

    Both *chapter_num* and *total_chapters* are clamped to sensible bounds on
    construction so that all derived properties (phase, act, zone checks, …)
    always reflect a structurally valid position:

    * ``total_chapters`` is clamped to at least 1.
    * ``chapter_num`` is clamped to ``[1, total_chapters]`` after
      ``total_chapters`` has been normalised.

    The normalised values are stored as ``chapter_num`` and ``total_chapters``
    so callers can inspect the effective position that was actually used.
    """

    def __init__(self, chapter_num: int, total_chapters: int) -> None:
        self.total_chapters = max(1, total_chapters)
        self.chapter_num = max(1, min(chapter_num, self.total_chapters))
        self.position_pct = self.chapter_num / self.total_chapters * 100

    # --- Landmark chapters ---

    @staticmethod
    def inciting_chapter(total_chapters: int) -> int:
        """Return the inciting-incident chapter number.

        Chapter 2 for novels of 4 or more chapters; chapter 1 otherwise.
        """
        return 2 if total_chapters >= 4 else 1

    @staticmethod
    def midpoint_chapter(total_chapters: int) -> int:
        """Return the midpoint chapter number (rounded centre of the range)."""
        return max(1, min(total_chapters, round((total_chapters + 1) / 2)))

    @staticmethod
    def climax_chapter(total_chapters: int) -> int:
        """Return the climax chapter number (second-to-last, or 1 if only one chapter)."""
        return max(1, total_chapters - 1) if total_chapters > 1 else 1

    @staticmethod
    def resolution_chapter(total_chapters: int) -> int:
        """Return the resolution chapter number (always the final chapter)."""
        return total_chapters

    # --- Phase and act classification ---

    def get_phase(self) -> str:
        """Return the narrative phase name for this chapter position.

        Phases are anchored to the four structural landmark chapters so that
        ``get_phase`` and the landmark helpers are always consistent (see
        module docstring for the full invariant list).

        Priority when landmarks coincide on the same chapter (small novels):
        climax > inciting incident > midpoint shift.
        """
        n = self.total_chapters
        c = self.chapter_num

        if n <= 1:
            return "Resolution"

        inc = ChapterPosition.inciting_chapter(n)
        mid = ChapterPosition.midpoint_chapter(n)
        clx = ChapterPosition.climax_chapter(n)

        # Resolution: the final chapter
        if c >= n:
            return "Resolution"

        # Climax: from the climax landmark through the penultimate chapter
        if c >= clx:
            return "Climax"

        # Inciting Incident takes priority over Midpoint Shift when they share
        # the same chapter (can happen for 4-chapter novels).
        if c == inc:
            return "Inciting Incident"

        # Midpoint Shift: only when distinct from the inciting landmark
        if c == mid:
            return "Midpoint Shift"

        # Between midpoint and climax: Complications → Crisis
        if c > mid:
            gap = clx - mid  # chapters separating midpoint from climax
            if gap > 1:
                # Crisis occupies the last third of the gap (≥1 chapter)
                crisis_start = clx - max(1, gap // 3)
                if c >= crisis_start:
                    return "Crisis"
            return "Complications"

        # Between inciting incident and midpoint
        if c > inc:
            return "Rising Action"

        # Before the inciting incident: Hook (chapter 1) or Setup
        if c == 1:
            return "Hook"
        return "Setup"

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

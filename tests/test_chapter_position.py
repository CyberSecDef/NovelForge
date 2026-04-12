"""
Tests for ChapterPosition – small novels, boundary counts, and
landmark/phase consistency.

Canonical invariants verified for every novel length n:
  - get_phase(inciting_chapter(n),    n) == "Inciting Incident"
  - get_phase(climax_chapter(n),      n) == "Climax"
  - get_phase(resolution_chapter(n),  n) == "Resolution"

And where the midpoint is distinct from the inciting landmark:
  - get_phase(midpoint_chapter(n),    n) == "Midpoint Shift"
"""

import pytest
from novelforge.chapter_position import ChapterPosition


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def phase(chapter_num: int, total: int) -> str:
    return ChapterPosition(chapter_num, total).get_phase()


# ---------------------------------------------------------------------------
# Landmark consistency: landmark chapter → expected phase
# ---------------------------------------------------------------------------

class TestLandmarkPhaseConsistency:
    """Every landmark helper must agree with get_phase() for all tested sizes."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30])
    def test_climax_chapter_yields_climax_phase(self, n):
        clx = ChapterPosition.climax_chapter(n)
        assert phase(clx, n) == "Climax", (
            f"n={n}: climax_chapter={clx}, got '{phase(clx, n)}'"
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30])
    def test_resolution_chapter_yields_resolution_phase(self, n):
        res = ChapterPosition.resolution_chapter(n)
        assert phase(res, n) == "Resolution", (
            f"n={n}: resolution_chapter={res}, got '{phase(res, n)}'"
        )

    @pytest.mark.parametrize("n", [4, 5, 6, 8, 10, 12, 15, 20, 30])
    def test_inciting_chapter_yields_inciting_phase(self, n):
        inc = ChapterPosition.inciting_chapter(n)
        assert phase(inc, n) == "Inciting Incident", (
            f"n={n}: inciting_chapter={inc}, got '{phase(inc, n)}'"
        )

    @pytest.mark.parametrize("n", [5, 6, 8, 10, 12, 15, 20, 30])
    def test_midpoint_chapter_yields_midpoint_phase_when_distinct(self, n):
        inc = ChapterPosition.inciting_chapter(n)
        mid = ChapterPosition.midpoint_chapter(n)
        clx = ChapterPosition.climax_chapter(n)
        if mid != inc and mid != clx:
            assert phase(mid, n) == "Midpoint Shift", (
                f"n={n}: midpoint_chapter={mid}, got '{phase(mid, n)}'"
            )


# ---------------------------------------------------------------------------
# Small-novel phase sequences (deterministic, fully enumerated)
# ---------------------------------------------------------------------------

class TestSmallNovelPhases:
    """Verify the exact phase sequence for novels of 1–6 chapters."""

    def test_one_chapter(self):
        assert phase(1, 1) == "Resolution"

    def test_two_chapters(self):
        # climax=1, resolution=2
        assert phase(1, 2) == "Climax"
        assert phase(2, 2) == "Resolution"

    def test_three_chapters(self):
        # inciting=1 (n<4), midpoint=2, climax=2, resolution=3
        # Chapter 1: inciting_chapter(3)=1 → "Inciting Incident"
        # Chapter 2: >= climax(2) → "Climax"
        # Chapter 3: resolution
        assert phase(1, 3) == "Inciting Incident"
        assert phase(2, 3) == "Climax"
        assert phase(3, 3) == "Resolution"

    def test_four_chapters(self):
        # inciting=2, midpoint=2, climax=3, resolution=4
        # Chapter 1: before inciting → Hook
        # Chapter 2: inciting_chapter(4)=2 → must be "Inciting Incident"
        # Chapter 3: climax_chapter(4)=3 → "Climax"
        # Chapter 4: resolution
        assert phase(1, 4) == "Hook"
        assert phase(2, 4) == "Inciting Incident"
        assert phase(3, 4) == "Climax"
        assert phase(4, 4) == "Resolution"

    def test_five_chapters(self):
        # inciting=2, midpoint=3, climax=4, resolution=5
        assert phase(1, 5) == "Hook"
        assert phase(2, 5) == "Inciting Incident"
        assert phase(3, 5) == "Midpoint Shift"
        assert phase(4, 5) == "Climax"
        assert phase(5, 5) == "Resolution"

    def test_six_chapters(self):
        # inciting=2, midpoint=round((6+1)/2)=round(3.5)=4 (banker's rounding),
        # climax=5, resolution=6
        assert phase(1, 6) == "Hook"
        assert phase(2, 6) == "Inciting Incident"
        # chapters 3: Rising Action (between inciting=2 and midpoint=4)
        assert phase(3, 6) == "Rising Action"
        assert phase(4, 6) == "Midpoint Shift"
        # chapter 5: climax_chapter(6)=5 → "Climax"
        assert phase(5, 6) == "Climax"
        assert phase(6, 6) == "Resolution"


# ---------------------------------------------------------------------------
# Representative larger novels
# ---------------------------------------------------------------------------

class TestLargerNovelPhases:
    """Spot-check phases for 10 and 20-chapter novels."""

    def test_ten_chapter_landmarks(self):
        # inciting=2, midpoint=round((10+1)/2)=round(5.5)=6 (banker's rounding),
        # climax=round(10*0.75)=round(7.5)=8, resolution=10
        assert phase(1, 10) == "Hook"
        assert phase(2, 10) == "Inciting Incident"
        assert phase(6, 10) == "Midpoint Shift"
        assert phase(8, 10) == "Climax"
        assert phase(10, 10) == "Resolution"

    def test_twenty_chapter_landmark_consistency(self):
        # inciting=2, midpoint=round((20+1)/2)=10,
        # climax=round(20*0.75)=15, resolution=20
        assert phase(1, 20) == "Hook"
        assert phase(2, 20) == "Inciting Incident"
        assert phase(10, 20) == "Midpoint Shift"
        assert phase(15, 20) == "Climax"
        assert phase(20, 20) == "Resolution"

    def test_twenty_chapter_intermediate_phases(self):
        # Chapters between inciting(2) and midpoint(10) → Rising Action
        for c in range(3, 10):
            assert phase(c, 20) == "Rising Action", f"chapter {c}/20"
        # Chapters between midpoint(10) and climax(15): Complications or Crisis
        for c in range(11, 15):
            assert phase(c, 20) in {"Complications", "Crisis"}, f"chapter {c}/20"
        # Chapters after climax(15) are Resolution (falling action)
        for c in range(16, 21):
            assert phase(c, 20) == "Resolution", f"chapter {c}/20"


# ---------------------------------------------------------------------------
# Landmark helper return-value sanity
# ---------------------------------------------------------------------------

class TestLandmarkReturnValues:
    """Landmark helpers must return valid chapter numbers (1..n)."""

    @pytest.mark.parametrize("n", range(1, 16))
    def test_all_landmarks_in_range(self, n):
        assert 1 <= ChapterPosition.inciting_chapter(n) <= n
        assert 1 <= ChapterPosition.midpoint_chapter(n) <= n
        assert 1 <= ChapterPosition.climax_chapter(n) <= n
        assert ChapterPosition.resolution_chapter(n) == n

    def test_inciting_threshold(self):
        # Boundary: n < 4 → chapter 1; n >= 4 → chapter 2
        assert ChapterPosition.inciting_chapter(1) == 1
        assert ChapterPosition.inciting_chapter(2) == 1
        assert ChapterPosition.inciting_chapter(3) == 1
        assert ChapterPosition.inciting_chapter(4) == 2
        assert ChapterPosition.inciting_chapter(20) == 2


# ---------------------------------------------------------------------------
# Input clamping / boundary validation
# ---------------------------------------------------------------------------

class TestChapterPositionClamping:
    """chapter_num and total_chapters must be clamped to sensible bounds."""

    # --- total_chapters clamping ---

    def test_total_chapters_zero_clamped_to_one(self):
        cp = ChapterPosition(1, 0)
        assert cp.total_chapters == 1

    def test_total_chapters_negative_clamped_to_one(self):
        cp = ChapterPosition(1, -5)
        assert cp.total_chapters == 1

    # --- chapter_num zero / negative clamped to 1 ---

    def test_chapter_num_zero_clamped_to_one(self):
        cp = ChapterPosition(0, 10)
        assert cp.chapter_num == 1

    def test_chapter_num_negative_clamped_to_one(self):
        cp = ChapterPosition(-3, 10)
        assert cp.chapter_num == 1

    # --- chapter_num over total_chapters clamped down ---

    def test_chapter_num_over_total_clamped(self):
        cp = ChapterPosition(15, 10)
        assert cp.chapter_num == 10

    def test_chapter_num_far_over_total_clamped(self):
        cp = ChapterPosition(999, 5)
        assert cp.chapter_num == 5

    # --- position_pct is always in [0, 100] after clamping ---

    @pytest.mark.parametrize("c,n", [
        (0, 10), (-1, 10), (11, 10), (0, 0), (-5, -3),
    ])
    def test_position_pct_always_in_range(self, c, n):
        cp = ChapterPosition(c, n)
        assert 0 <= cp.position_pct <= 100

    # --- one-chapter novel boundary ---

    def test_one_chapter_novel_chapter_one(self):
        cp = ChapterPosition(1, 1)
        assert cp.chapter_num == 1
        assert cp.total_chapters == 1
        assert cp.position_pct == 100.0
        assert cp.get_phase() == "Resolution"

    def test_one_chapter_novel_over_range_clamped(self):
        cp = ChapterPosition(5, 1)
        assert cp.chapter_num == 1
        assert cp.get_phase() == "Resolution"

    # --- clamped chapter_num drives phase, not raw input ---

    def test_negative_chapter_uses_clamped_phase(self):
        # chapter -1 of 10 is clamped to chapter 1 → Hook
        assert ChapterPosition(-1, 10).get_phase() == "Hook"

    def test_over_range_chapter_uses_clamped_phase(self):
        # chapter 20 of 10 is clamped to 10 → Resolution
        assert ChapterPosition(20, 10).get_phase() == "Resolution"

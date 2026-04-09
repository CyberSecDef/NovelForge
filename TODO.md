# NovelForge TODO — Codebase Review Findings (2026-04-05)

This document captures all issues identified during comprehensive codebase reviews.
Items are grouped by severity and ordered by priority within each group. Each item
includes the problem description, affected files/lines, why it matters, and
recommended fixes.

# NovelForge TODO — UI/UX Redesign: "A Place Where Novels Come to Life" (2026-04-08)

Transform NovelForge from a generic Bootstrap utility into a visually rich, editorially-flavored creative writing environment. All changes use Bootstrap 5.3+ with custom CSS. The app remains a single-page application. Issues are ordered roughly by implementation dependency — foundational items first.

---

## 1. Foundation: Typography & Reading Experience

### 1.1 — Establish a literary type scale ✅ DONE (2026-04-08)

**Files:** `static/css/style.css`, `templates/index.html`

**Completed changes:**
- Added Google Fonts (Lora: 400, 600, 400i) with `preconnect` and `display=swap` to `index.html`.
- Created dual-font CSS custom properties: `--nf-font-ui` (system sans-serif) and `--nf-font-literary` (Lora + Georgia fallback).
- Created `.nf-literary` class applied to: premise textarea, novel title input (Step 2), done-title (Step 4), editor's notes, chapter/character editable cells.
- Created `.nf-title` class for novel titles: `2.441rem` serif, weight 600, tight letter-spacing.
- Applied serif font to `.accordion-body` (chapter previews) and `.navbar-brand` via the new CSS variable.
- Implemented 1.25-ratio modular type scale (`h1`–`h6`) via CSS custom properties.
- All tests pass.

---

### 1.2 — Improve the premise textarea to feel like a writing surface ✅ DONE (2026-04-08)

**Files:** `static/css/style.css`, `templates/index.html`

**Completed changes:**
- Increased premise textarea rows from 4 to 7.
- Created `.nf-writing-surface` class with: generous padding (1.25rem), warm off-white background (`#fffef8`), subtle inset shadow, serif font at 1.05rem/1.75 line-height.
- Italic placeholder in muted parchment tone (`#a09882`).
- On focus: warm amber border glow (`rgba(184,166,126,0.25)`), smooth expand to `min-height: 14rem` via CSS transition.
- Dark mode: warm dark background (`#1e1d1a`), warm text (`#e8e0d4`), matching amber focus glow at reduced opacity.
- All tests pass.

---

### 1.3 — Style the chapter preview accordion as a reading experience ✅

**Files:** `static/css/style.css`

The chapter preview accordion in Step 4 already uses Georgia/serif — enhance it:
- Increase line-height to 1.9 for comfortable reading.
- Add generous horizontal padding (2rem+) to simulate book margins.
- Set a max-width of ~70ch on the text content for optimal reading line length.
- Style the accordion headers with the chapter title in serif and a subtle bottom border instead of the heavy Bootstrap accordion chrome.
- Add a decorative chapter number treatment (e.g. "Chapter One" in small-caps above the title).
- Optional: a subtle drop-cap on the first letter of each chapter's content.

---

## 2. Color & Theme Identity

### 2.1 — Define a NovelForge brand color palette ✅

**Files:** `static/css/style.css`

Replace the default Bootstrap primary/success/warning colors used on card headers with a cohesive brand palette. Suggested direction:
- **Primary accent:** A deep literary tone — ink blue (`#1B2A4A`), burgundy (`#6B2D3E`), or forest green (`#2D4A3E`).
- **Secondary accent:** A warm gold/amber (`#C4960C` or `#B8860B`) for highlights, progress bars, and interactive elements.
- **Background:** Warm off-white (`#FAF9F6`) instead of Bootstrap's cold `#f8f9fa`.
- **Text:** Rich near-black (`#1a1a1a`) instead of Bootstrap's default.
- Define these as CSS custom properties on `:root` so they cascade everywhere:
  ```css
  :root {
    --nf-ink: #1B2A4A;
    --nf-gold: #C4960C;
    --nf-parchment: #FAF9F6;
    --nf-text: #1a1a1a;
    --nf-text-muted: #6B7280;
  }
  ```
- Update dark mode equivalents under `[data-bs-theme="dark"]`.

---

### 2.2 — Unify card header colors with the brand palette ✅

**Files:** `templates/index.html`, `static/css/style.css`

Currently each step uses a different Bootstrap contextual color (`bg-primary`, `bg-success`, `bg-warning`, `bg-dark`). Replace these with:
- A consistent `bg-nf-ink` (the primary brand color) for all card headers, or
- A subtle gradient using the brand palette (e.g. `linear-gradient(135deg, var(--nf-ink), #2a3f6a)`).
- Each step can keep its own *icon* for identity, but the color should be unified.
- Remove the jarring bright-yellow `bg-warning` on the Step 3 card header — replace with the brand treatment plus an animated writing indicator.

---

### 2.3 — Redesign the progress bar with brand colors ✅

**Files:** `static/css/style.css`, `templates/index.html`

- Replace `bg-warning` with a warm gold gradient: `linear-gradient(90deg, var(--nf-gold), #E8B400)`.
- Add a subtle shimmer/pulse animation while writing is in progress (CSS-only, no JS).
- On completion (100%), transition to a solid gold with a brief "glow" keyframe animation.
- Increase height from 24px to 28px with rounded corners (`border-radius: 1rem`).
- Style the percentage text inside the bar with a slight text-shadow for legibility.

---

## 3. Layout & Spacing

### 3.1 — Add breathing room to the Step 1 form ✅

**Files:** `templates/index.html`, `static/css/style.css`

- Increase vertical spacing between form groups (from `mb-3`/`mb-4` to consistent `mb-4`/`mb-5`).
- Add visual grouping: wrap the "Chapters + Word Count" row in a subtle card or fieldset with a light border and label like "Novel Parameters."
- Wrap "Special Events" and "Special Instructions" in a collapsible "Advanced Options" section (Bootstrap collapse) so the primary form feels lighter for first-time users.
- Center the form in a narrower column (`col-lg-8` instead of `col-lg-12`) so inputs don't stretch edge-to-edge.

---

### 3.2 — Add a hero/welcome section above the Step 1 form ✅

**Files:** `templates/index.html`, `static/css/style.css`

When the user first loads the app (no session restored), show a brief hero section above the form:
- A large serif heading: "What story will you tell?" or similar.
- A one-line subtitle in muted text: "Describe your vision and NovelForge will bring it to life."
- Subtle decorative elements: a thin horizontal rule with a small ornamental flourish (CSS `::before`/`::after` with a unicode ornament like ❧ or ✦).
- This section hides when a session is restored or after the outline is generated.

---

### 3.3 — Improve the Step 4 export page layout ✅

**Files:** `templates/index.html`, `static/css/style.css`

The Step 4 panel tries to do too much in a single column. Reorganize:
- **Top:** Novel title (large serif) + subtitle stats line — styled like a title page.
- **Actions row:** Use a Bootstrap grid with two columns: left column for export buttons (Download Manuscript, Editor's Notes), right column for Illustrations.
- **Revision panel:** Give it its own section below the action row, with a subtle border-top divider rather than being inside a card-within-a-card.
- **Preview accordion:** Full width below everything, preceded by a decorative section divider.
- **Writing Statistics / Relationship Map:** Move to a tabbed sub-panel or side-by-side collapsibles so the page isn't endlessly vertical.

---

## 4. Navigation & Step Indicators

### 4.1 — Redesign the navbar with brand identity ✅

**Files:** `templates/index.html`, `static/css/style.css`

- Replace the generic `bg-dark` navbar with the brand ink color (`var(--nf-ink)`).
- Style the "NovelForge" brand text in the serif font at a slightly larger size.
- Replace the `bi-book-half` icon with a more distinctive icon or a simple SVG logo/wordmark.
- Add a subtle bottom border or shadow to lift the navbar off the content.
- Restyle the session/theme buttons: use ghost buttons (no border, just icon + text) with hover effects instead of outlined buttons, to reduce visual clutter.

---

### 4.2 — Replace tab navigation with a step-progress indicator ✅

**Files:** `templates/index.html`, `static/css/style.css`, `static/js/script.js`

The plain Bootstrap nav-tabs look generic. Replace with a horizontal step indicator:
- Four steps displayed as connected nodes: `(1) Imagine → (2) Plan → (3) Write → (4) Publish`
- Use evocative labels instead of "Step 1 - Novel Setup" etc.
- Each step shows as: a numbered circle with a label below. Active step is filled with the brand accent color; completed steps show a checkmark; future steps are outlined/dimmed.
- Connect the circles with a horizontal line that fills with color as steps complete.
- The Step 2 dropdown (Chapter Outline / Character Development) becomes sub-tabs *within* the Step 2 panel rather than a dropdown on the step indicator.
- Keep the existing Bootstrap tab-pane mechanism underneath — this is purely a visual wrapper.
- On mobile, collapse to just icons + the active step's label.

---

### 4.3 — Add subtle step transition animations ✅

**Files:** `static/css/style.css`

- When switching tabs/steps, fade-in the new panel with a slight upward slide (already partially exists — enhance it):
  ```css
  .tab-pane.fade { transition: opacity 0.4s ease, transform 0.4s ease; transform: translateY(12px); }
  .tab-pane.show { transform: translateY(0); }
  ```
- Add a brief highlight pulse to the step indicator node when it becomes active.

---

## 5. Tables → Editorial Cards

### 5.1 — Rethink the chapter outline table (Step 2a) ✅

**Files:** `templates/index.html`, `static/css/style.css`, `static/js/script.js`

The dense table layout makes editing feel like data entry. Replace with a card-list layout:
- Each chapter becomes a card: chapter number as a subtle badge, title as an editable heading (serif), summary as editable body text.
- Cards are stacked vertically with comfortable spacing.
- Action buttons (move up/down, delete) appear on hover or as a kebab menu (three dots) in the top-right corner — not as a persistent column.
- Add drag-and-drop reordering via HTML5 drag API or a small library.
- Keep the table as a fallback option via a view toggle ("Card View / Table View") for users who prefer density.

---

### 5.2 — Rethink the character table (Step 2b) ✅

**Files:** `templates/index.html`, `static/css/style.css`, `static/js/script.js`

Replace the character table with character cards:
- Each character gets a card with their name as a heading, role as a subtitle/badge, and expandable sections for Background and Arc.
- Cards arranged in a responsive grid (2-up on desktop, 1-up on mobile).
- Add a subtle color-coded left border per character (auto-assigned from a palette) that can later be used to highlight their appearances in chapter text.
- The "Add Character" button becomes a dashed-border empty card with a "+" icon (the common "add card" pattern).
- Age displayed as a small detail, not a full table column.

---

### 5.3 — Style the writing statistics table (Step 4) ✅

**Files:** `static/css/style.css`

- Replace the dark table header with a subtle brand-tinted header.
- Add alternating row backgrounds with very subtle tinting.
- Highlight the row with the highest word count and the longest generation time with small badge indicators.
- Add a mini bar-chart sparkline in the "Words" column showing relative chapter length at a glance (pure CSS using `background: linear-gradient()`).

---

## 6. Progress & Writing Experience (Step 3)

### 6.1 — Redesign the chapter progress list ✅

**Files:** `static/css/style.css`, `templates/index.html`

- Replace the flat `list-group` with styled cards or a timeline layout.
- **Timeline layout:** A vertical line on the left, with each chapter as a node. Completed chapters show a filled circle with checkmark; the in-progress chapter shows an animated pulse; upcoming chapters show an empty circle.
- Show the chapter title next to each node (not just "Chapter 1: Title" in a list item).
- The currently-writing chapter should show the current agent pass name (e.g. "Prose Refinement…") as animated subtitle text.
- Completed chapters should be expandable inline to preview their content without navigating away.

---

### 6.2 — Add a live writing atmosphere to Step 3 ✅

**Files:** `static/css/style.css`, `templates/index.html`

While chapters are being written, the page should feel alive:
- Add a subtle CSS animation to the header — a slow quill-writing or typewriter-cursor animation next to "Writing Chapters…"
- Show an auto-updating word counter with count-up animation as chapters complete.
- Display a "Currently writing: *Chapter Title*" callout that updates in real-time.
- Add a faint, repeating CSS background pattern (paper texture or very subtle grid) to the Step 3 panel to differentiate it from the other steps.

---

### 6.3 — Add elapsed time and ETA display ✅

**Files:** `static/css/style.css`, `templates/index.html`

The time estimate element exists but is visually hidden. Enhance it:
- Show elapsed time prominently: "Writing for 2h 14m"
- Show ETA: "Estimated completion: ~1h 30m remaining"
- Display as a small stats bar below the progress bar with clock icons.
- Update smoothly (no flicker on DOM update).

---

## 7. Micro-Interactions & Polish

### 7.1 — Animate the "Generate Outline" button ✅

**Files:** `static/css/style.css`

- On hover: subtle lift effect (`transform: translateY(-2px)`) with increased shadow.
- On click/loading: smooth transition to a "working" state — button text fades to "Conjuring your story…" with the spinner.
- On success: brief green flash or checkmark animation before transitioning to Step 2.

---

### 7.2 — Add hover effects to all interactive cards ✅

**Files:** `static/css/style.css`

- Cards that are clickable or editable should lift slightly on hover (`box-shadow` increase + slight `translateY`).
- Editable cells/areas should show a faint pencil icon or highlight border on hover to signal editability.

---

### 7.3 — Improve toast/alert styling ✅

**Files:** `static/css/style.css`

- Replace the standard Bootstrap alerts in `#global-alert-area` with custom-styled toasts:
  - Rounded corners, subtle drop shadow, left border accent in the alert color.
  - Slide-in animation from the top.
  - Auto-dismiss after 8 seconds with a shrinking progress bar at the bottom.
  - Close button styled as a minimal "×".

---

### 7.4 — Add loading skeleton screens ✅

**Files:** `static/css/style.css`, `static/js/script.js`

When waiting for the outline to generate, show skeleton placeholders instead of a blank panel with a spinner:
- Animated grey bars mimicking the chapter table structure (pulsing shimmer effect).
- Skeleton cards for characters.
- This gives the impression of speed and sets expectations for what's coming.

---

## 8. Decorative & Atmospheric Elements

### 8.1 — Add section dividers with ornamental flourishes ✅

**Files:** `static/css/style.css`

Create a reusable `.divider-ornament` class:
- A thin horizontal line with a small centered ornamental glyph (❦, ✦, ◆, or an SVG flourish).
- Use between major sections (between the form and the button, between stats and preview, etc.).
- Keeps the literary feel without being heavy-handed.
```css
.divider-ornament {
  text-align: center;
  margin: 2rem 0;
  border: 0;
  position: relative;
}
.divider-ornament::before {
  content: "❦";
  display: inline-block;
  padding: 0 1rem;
  background: var(--nf-parchment);
  color: var(--nf-gold);
  font-size: 1.2rem;
  position: relative;
  z-index: 1;
}
.divider-ornament::after {
  content: "";
  position: absolute;
  top: 50%;
  left: 10%;
  right: 10%;
  border-top: 1px solid #d0c8b8;
}
```

---

### 8.2 — Style the "Your Novel is Ready!" completion state ✅

**Files:** `templates/index.html`, `static/css/style.css`

This is the payoff moment — make it feel celebratory:
- Replace the plain green card header with a full-width banner using the brand palette and a subtle gradient.
- Display the novel title in large serif text (2.5rem+) with a subtle text-shadow.
- Add a brief confetti or sparkle CSS animation on first reveal (CSS-only, using `@keyframes` with pseudo-elements).
- Show the stats line (chapters, word count) in an elegant inline format: "25 chapters · ~85,000 words".
- Add a decorative border or vignette around the entire completion card.

---

### 8.3 — Add a subtle page background texture ✅

**Files:** `static/css/style.css`

- Apply a very subtle paper/linen texture as a CSS background pattern on `body`. Can be done with pure CSS gradients (no image needed):
  ```css
  body {
    background-color: var(--nf-parchment);
    background-image: 
      radial-gradient(ellipse at 20% 50%, rgba(200,180,150,0.03) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 50%, rgba(200,180,150,0.03) 0%, transparent 50%);
  }
  ```
- Dark mode: replace with a very subtle dark paper texture.
- Keep it extremely subtle — this should be felt, not seen.

---

## 9. Responsive & Mobile Experience

### 9.1 — Improve mobile layout for Step 1 form ✅

**Files:** `static/css/style.css`

- On mobile, the "Chapters" and "Word Count" side-by-side inputs already stack (Bootstrap grid). Ensure padding and touch targets are generous (44px minimum height on inputs and buttons).
- Make the "Generate Outline" button sticky at the bottom of the viewport on mobile so it's always reachable.

---

### 9.2 — Mobile-friendly chapter and character editing ✅

**Files:** `static/css/style.css`

- If card-based layout is implemented (5.1, 5.2), ensure cards are touch-friendly with large tap targets.
- Editable areas should trigger on tap with a visible focus state.
- Action buttons (delete, reorder) should be accessible via swipe or long-press, not just hover.

---

### 9.3 — Responsive step indicator ✅

**Files:** `static/css/style.css`

- On screens < 768px, collapse the step indicator to show only icons with the active step's label.
- Alternatively, use a compact horizontal scrollable strip.

---

## 10. Dark Mode Refinement

### 10.1 — Warm up the dark mode palette ✅

**Files:** `static/css/style.css`

Current dark mode uses cold Bootstrap greys. Shift to warmer tones:
- Background: `#191715` (warm near-black) instead of `#1a1d21`.
- Card surfaces: `#231f1c` instead of `#212529`.
- Borders: `#3d3630` instead of `#495057`.
- Text: `#e8e0d4` (warm off-white) instead of `#dee2e6`.
- Accent: warm gold (`#D4A833`) that complements the dark background.
- The overall feel should be "reading by lamplight" not "IDE at midnight."

---

### 10.2 — Add a smooth dark/light mode transition ✅

**Files:** `static/css/style.css`

- Add `transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease` to `body`, `.card`, `.navbar`, and major containers.
- The theme toggle should feel seamless, not jarring.

---

## 11. Empty States & Onboarding

### 11.1 — Design meaningful empty states ✅

**Files:** `templates/index.html`, `static/css/style.css`

For panels that start empty (chapter list, character list, session dropdown, illustrations gallery):
- Show an illustrated or iconographic empty state with a short message:
  - Sessions dropdown: "No saved stories yet"
  - Illustrations gallery: "Your illustrations will appear here"
  - Chapter preview: "Chapters will appear as they're written"
- Use a large, muted Bootstrap Icon with text below. Keep it warm and encouraging, not clinical.

---

### 11.2 — Add a first-run tooltip tour ✅

**Files:** `static/js/script.js`, `static/css/style.css`

On first visit (check localStorage), show subtle tooltip popovers pointing to key UI areas:
- "Start here — describe the story you want to write" (pointing to the premise textarea)
- "Choose your genre to set the tone" (pointing to genre dropdown)
- Tooltips dismiss on click and don't reappear after dismissal.
- Use Bootstrap popovers with custom styling (brand colors, serif header text).
- Keep it to 2–3 tips maximum — not an intrusive walkthrough.

---

## 12. Illustration Gallery Enhancement

### 12.1 — Redesign the illustration gallery ✅

**Files:** `static/css/style.css`, `templates/index.html`

- Display illustrations in a masonry or Pinterest-style grid instead of a uniform Bootstrap row.
- Each illustration card should have: the image, a caption (scene description), and the chapter it belongs to.
- On hover/click, open a lightbox-style overlay with the full-size image (use Bootstrap modal with a transparent backdrop and centered image).
- The cover image should be displayed larger and first, with a "Cover" badge.

---

## 13. LLM Log Refinement

### 13.1 — Clean up the Log tab for non-developers ✅

**Files:** `templates/index.html`, `static/css/style.css`

The chat-bubble log is a nice idea but shows raw JSON. Improve:
- Parse and display the actual prompt text and response text, not raw JSON blobs.
- Show a human-readable label for each exchange: "Draft Pass — Chapter 3" or "Character Arc Planning."
- Add a filter/search bar at the top to find specific chapters or agent passes.
- Add a count badge on the Log tab showing total exchanges.
- Consider making this tab hidden by default (accessible via a settings toggle or developer mode), since most users won't care about LLM internals.

---

## 14. Accessibility & Interaction Quality

### 14.1 — Improve focus states for keyboard navigation ✅

**Files:** `static/css/style.css`

- Replace the default browser outline with a visible, brand-colored focus ring on all interactive elements:
  ```css
  :focus-visible {
    outline: 2px solid var(--nf-gold);
    outline-offset: 2px;
  }
  ```
- Ensure all editable cells have clear focus indication.

---

### 14.2 — Add meaningful aria labels and live regions ✅

**Files:** `templates/index.html`

- The step indicator (when implemented) should have `aria-label="Novel creation progress"`.
- When a chapter completes, announce it to screen readers via the existing `aria-live` region.
- Editable cells should have `aria-label="Edit chapter [N] title"` etc.

---

## 15. Font Loading & Performance

### 15.1 — Add Google Fonts with proper loading strategy ✅

**Files:** `templates/index.html`

- Add Google Fonts link for the chosen serif face (e.g., Lora or Merriweather) with `display=swap` and preconnect:
  ```html
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
  ```
- Define fallback stacks: `font-family: 'Lora', Georgia, 'Times New Roman', serif`.
- Keep total added font weight under 100KB (2–3 weights max).

---

## Implementation Notes

- **Approach:** Implement foundational items first (1.x, 2.x, 15.x) since they cascade into everything else. Then layout (3.x, 4.x), then component-level redesigns (5.x–8.x), then polish (7.x, 9.x–14.x).
- **Dark mode:** Every visual change must be implemented for both light and dark modes simultaneously. Use CSS custom properties to minimize duplication.
- **No JS frameworks:** All changes are CSS + vanilla JS + Bootstrap 5 components. No React/Vue/etc.
- **Testing:** Visual changes don't need unit tests, but ensure the existing `pytest` suite still passes after any HTML structure changes (some tests may assert on specific CSS classes or element IDs).
- **Browser support:** Target modern evergreen browsers. CSS features like `@keyframes`, custom properties, `backdrop-filter`, and `:focus-visible` are all safe to use.

---
---

# NovelForge TODO — Chapter Structure Repetition Fixes (2026-04-08)

The anti-repetition pipeline (rhythm classifier, compression check, scene variety auditor,
momentum & distinctiveness, operational distinctiveness) **executes correctly** — all agents
run, all data flows where it should. However, the architecture has 5 structural gaps that
allow the LLM to fall back into its default `event → solution → description` pattern
despite receiving variety directives. These fixes close those gaps.

---

## Fix 1 — Track rhythm classifications across chapters and feed history to the classifier ✅

### Problem
The rhythm classifier recommends a rhythm (e.g. `cat-and-mouse`) for each chapter, but the
chosen rhythm is never recorded. When the classifier runs for chapter N+1, it sees only
*plot summaries* — it cannot say "chapters 3, 5, and 7 all used `revelation-dump-then-reaction`"
because structural classification data doesn't persist between chapters. The classifier is
essentially flying blind, relying on plot summaries to infer structure.

### Fix

**File: `novelforge/routes/generation/chapters.py`**

Add a `rhythm_log` list that accumulates the chosen rhythm per chapter, and pass it to the
rhythm classifier as structured context.

```python
# --- At initialization (near line 194, next to compression_guidance = "") ---
rhythm_log: list[dict] = []

# --- After rhythm classification succeeds (after line 301) ---
chapter_rhythm_shape = str(rhythm_result.get("recommended_shape_for_this_chapter", "")).strip()
chapter_rhythm_reason = str(rhythm_result.get("recommendation_reason", "")).strip()
# NEW: record detected patterns too
chapter_detected_patterns = rhythm_result.get("detected_patterns", [])

# --- After chapter completes, before compression check (near line 355) ---
rhythm_log.append({
    "chapter": chapter_num,
    "recommended": chapter_rhythm_shape,
    "detected_patterns": chapter_detected_patterns,
})

# --- Pass rhythm_log to the classifier for the NEXT chapter ---
rhythm_result = run_chapter_rhythm_classifier(
    chapter_num=chapter_num, chapter_title=chapter_title,
    chapter_summary=chapter_outline_summary,
    previous_summaries=previous_summaries,
    title=title, chapter_architecture_context=chapter_architecture_context,
    rhythm_log=rhythm_log,  # NEW parameter
    degraded_passes=degraded_passes,
)
```

**File: `novelforge/agents/chapter/pipeline.py`** — Update `run_chapter_rhythm_classifier()`:

```python
def run_chapter_rhythm_classifier(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str, title: str,
    chapter_architecture_context: str = "",
    rhythm_log: list[dict] | None = None,  # NEW
    degraded_passes: list[dict] | None = None,
) -> dict:
    try:
        raw = call_llm(
            build_chapter_rhythm_classifier_prompt(
                chapter_num=chapter_num, chapter_title=chapter_title,
                chapter_summary=chapter_summary, previous_summaries=previous_summaries,
                title=title, chapter_architecture_context=chapter_architecture_context,
                rhythm_log=rhythm_log or [],  # NEW
            ),
            action=f"Classifying chapter rhythm for Chapter {chapter_num}",
            json_mode=True,
        )
        ...
```

**File: `novelforge/agents/chapter/prompts.py`** — Update `build_chapter_rhythm_classifier_prompt()`:

```python
def build_chapter_rhythm_classifier_prompt(
    chapter_num: int, chapter_title: str, chapter_summary: str, previous_summaries: str, title: str,
    chapter_architecture_context: str = "",
    rhythm_log: list[dict] | None = None,  # NEW
) -> list[dict[str, str]]:
    formatted_rhythm_log = ""
    if rhythm_log:
        lines = []
        for entry in rhythm_log:
            lines.append(f"Chapter {entry['chapter']}: rhythm = {entry['recommended']}")
        formatted_rhythm_log = "\n".join(lines)
    return render_prompt(
        "chapter_rhythm_classifier", title=title, chapter_num=chapter_num,
        chapter_title=chapter_title, chapter_summary=chapter_summary,
        previous_summaries=previous_summaries or "",
        chapter_architecture_context=chapter_architecture_context or "",
        rhythm_log=formatted_rhythm_log,  # NEW
    )
```

**File: `prompts.yml`** — Update the `chapter_rhythm_classifier` user prompt:

Add this block after the `previous_summaries` section (around line 460):

```yaml
      {%- if rhythm_log.strip() %}

      STRUCTURAL RHYTHM HISTORY (rhythms assigned to previous chapters — do NOT repeat recent entries):
      {{ rhythm_log }}

      You MUST choose a rhythm that does NOT appear in the last 3 entries above.
      {%- endif %}
```

---

## Fix 2 — Add a post-draft rhythm compliance verifier ✅

### Problem
The draft prompt tells the LLM to use a specific rhythm (e.g. `cat-and-mouse`), but no
downstream agent checks whether the draft actually followed the directive. The LLM can
ignore the rhythm instruction and write another `event → solution → description` chapter,
and the 17 subsequent agent passes will never catch it because none of them know what
rhythm was intended.

### Fix

Add a new agent pass — **Rhythm Compliance Verifier** — that runs immediately after the
draft and before prose refinement. It receives the intended rhythm and the draft text,
checks whether the draft actually matches the rhythm, and rewrites structural elements
if it doesn't.

**File: `prompts.yml`** — Add new prompt after `chapter_draft`:

```yaml
  - name: "rhythm_compliance_verifier"
    description: "Verifies a chapter draft actually follows the assigned narrative rhythm and rewrites structural elements if it defaults to a generic pattern."
    stage: "Chapter Generation"
    system: |
      You are a Narrative Rhythm Compliance Verifier. You receive a chapter draft
      and the rhythm it was supposed to follow. Your job is to determine whether
      the draft actually exhibits the assigned rhythm or has defaulted to a
      generic "event → solution → description" pattern.

      The 10 named rhythms and their structural signatures:

      1. cat-and-mouse – Shifting advantage between opposing sides. Multiple
         reversals of who holds power. NOT just "hero chases villain."
      2. mirror-scene – Deliberately echoes an earlier chapter's structure with
         an inverted outcome. Must have a clear parallel-then-divergence.
      3. false-victory-into-disaster – First half reads as success; second half
         collapses into a worse position. The PIVOT is essential.
      4. ensemble-convergence – Multiple character threads running in parallel
         that collide. Must have at least 2 distinct threads before convergence.
      5. lyrical-interlude – Reflective, slower pace. Internal landscape, memory,
         sensory detail. Minimal external plot advancement.
      6. action-crescendo – Escalating physical/operational tempo to a peak.
         Minimal reflection. Momentum carries everything.
      7. revelation-dump-then-reaction – Major information disclosed early.
         Remainder tracks emotional and strategic fallout.
      8. emotional-valley-then-climb – Character hits bottom, then finds resolve.
         Must have a clear nadir before the turn.
      9. ticking-clock – Compressed timeline with deadline pressure throughout.
         The deadline must be present and felt in the prose.
      10. quiet-before-storm – Deceptive calm with mounting dread. Small details
          accumulate tension while characters prepare or wait.

      THE DEFAULT FAILURE MODE: "discover problem → characters discuss/plan →
      execute solution → reflect on outcome." This is the pattern you must
      detect and eliminate. It is NOT any of the 10 rhythms above.

      If the draft matches the assigned rhythm: return it unchanged.
      If the draft has defaulted to a generic pattern: restructure the chapter
      to match the assigned rhythm while preserving all plot events, character
      actions, and continuity. Change the ORDER, PACING, and PROBLEM-RESOLUTION
      APPROACH — not the events themselves.

      Return only the complete chapter text.
    user: |
      Novel: '{{ title }}' – Chapter {{ chapter_num }}

      ASSIGNED RHYTHM: {{ chapter_rhythm_shape }}
      REASON: {{ chapter_rhythm_reason }}

      Does this chapter actually follow the "{{ chapter_rhythm_shape }}" rhythm?

      Check specifically:
      - Does the chapter's macro-structure match the rhythm's signature?
      - Or has it defaulted to "discover problem → plan → execute → reflect"?
      - Does the pacing shape match? (e.g., a "false-victory-into-disaster" MUST
        have a clear pivot point where success collapses)
      - Does the problem-resolution approach match? (e.g., "ticking-clock" must
        feel time-pressured throughout, not just mention a deadline once)

      If the chapter already follows the rhythm, return it unchanged.
      If not, restructure it to match "{{ chapter_rhythm_shape }}" while preserving
      all plot events and continuity.

      ***CRITICAL: Return ONLY the complete chapter text with NO introduction, NO explanation, NO markdown.***

      {{ chapter_text }}
```

**File: `novelforge/agents/chapter/prompts.py`** — Add builder:

```python
def build_rhythm_compliance_verifier_prompt(
    chapter_text: str, chapter_num: int, title: str,
    chapter_rhythm_shape: str, chapter_rhythm_reason: str,
) -> list[dict[str, str]]:
    """Build the rhythm compliance verification prompt."""
    return render_prompt(
        "rhythm_compliance_verifier", title=title, chapter_num=chapter_num,
        chapter_rhythm_shape=chapter_rhythm_shape,
        chapter_rhythm_reason=chapter_rhythm_reason,
        chapter_text=chapter_text,
    )
```

**File: `novelforge/agents/chapter/pipeline.py`** — Add to imports and insert into `_run_all_chapter_agents()`:

Add `chapter_rhythm_shape` and `chapter_rhythm_reason` as new parameters to
`_run_all_chapter_agents()`, and insert the verifier as the FIRST pass (before prose
refinement):

```python
def _run_all_chapter_agents(
    text: str,
    chapter_num: int,
    title: str,
    genre: str,
    total_chapters: int,
    chapter_outline_summary: str,
    characters_text: str,
    previous_summaries: str,
    ctx: ChapterContext | None = None,
    step_callback: Callable[[str], None] | None = None,
    deadline: float = 0,
    degraded_passes: list[dict] | None = None,
    chapter_rhythm_shape: str = "",       # NEW
    chapter_rhythm_reason: str = "",      # NEW
) -> tuple[str, str]:
    ...
    # NEW: First pass — verify rhythm compliance (only if rhythm was assigned)
    if chapter_rhythm_shape:
        _check_deadline()
        if step_callback:
            step_callback(f"Chapter {chapter_num}: verifying rhythm compliance")
        text = _safe(
            lambda t: build_rhythm_compliance_verifier_prompt(
                t, chapter_num, title, chapter_rhythm_shape, chapter_rhythm_reason,
            ),
            text, action=f"Chapter {chapter_num}: rhythm compliance",
        )

    # Existing passes continue from here...
    _check_deadline()
    if step_callback:
        step_callback(f"Chapter {chapter_num}: prose refinement (dialogue + scenes)")
    ...
```

**File: `novelforge/routes/generation/chapters.py`** — Pass rhythm to `_run_all_chapter_agents()`:

```python
text, summary = _run_all_chapter_agents(
    text=text, chapter_num=chapter_num, title=title,
    genre=genre, total_chapters=total_chapters,
    chapter_outline_summary=chapter_outline_summary,
    characters_text=characters_text,
    previous_summaries=previous_summaries,
    ctx=ch_ctx, step_callback=_set_step, deadline=chapter_deadline,
    degraded_passes=degraded_passes,
    chapter_rhythm_shape=chapter_rhythm_shape,       # NEW
    chapter_rhythm_reason=chapter_rhythm_reason,      # NEW
)
```

---

## Fix 3 — Enrich chapter summaries with structural metadata ✅

### Problem
All anti-repetition agents that compare across chapters (`momentum & distinctiveness`,
`operational distinctiveness`, `compression check`) receive `previous_summaries` — but
these are pure *plot summaries*. They describe *what happens* ("Aldric discovers the device
and formulates a plan"), not *how the chapter is structured* ("this chapter used
`revelation-dump-then-reaction` rhythm, opened with a dialogue scene, resolved through
improvisation rather than planning").

An LLM reading plot summaries will struggle to detect structural repetition because
different events can follow identical structures.

### Fix

Modify the chapter summary to include structural metadata. This doesn't require a new
agent — it enriches the existing summarizer prompt.

**File: `prompts.yml`** — Modify the `chapter_summary` prompt (around line 1438):

Replace the current summary prompt:

```yaml
  - name: "chapter_summary"
    description: "Writes a 100-200 word continuity summary of a completed chapter with structural metadata."
    stage: "Chapter Generation"
    system: |
      You are a precise summariser of fiction. You write both a PLOT SUMMARY
      and a STRUCTURAL ANNOTATION for each chapter.
    user: |
      Write a continuity summary for Chapter {{ chapter_num }} in two clearly labelled sections:

      PLOT SUMMARY (100-200 words):
      Summarise the key events, character actions, revelations, and emotional shifts.

      STRUCTURE (3-5 short tags):
      Classify the chapter's structure using these dimensions:
      - RHYTHM: Which of these best describes the chapter's macro-structure?
        cat-and-mouse | mirror-scene | false-victory-into-disaster |
        ensemble-convergence | lyrical-interlude | action-crescendo |
        revelation-dump-then-reaction | emotional-valley-then-climb |
        ticking-clock | quiet-before-storm | generic-linear
      - OPENING: How does the chapter open? (e.g., "in-medias-res action",
        "quiet dialogue", "internal monologue", "scene-setting description",
        "time-skip transition")
      - RESOLUTION: How are problems resolved? (e.g., "planned-execution",
        "improvisation", "failure-and-retreat", "external-intervention",
        "emotional-shift", "revelation", "no-resolution-cliffhanger")
      - EMOTIONAL-ARC: What is the emotional shape? (e.g., "steady-build",
        "valley-then-climb", "high-to-low", "flat-tension", "multiple-peaks",
        "slow-burn")
      - DOMINANT-SCENE-TYPE: What type of scene dominates? (e.g., "dialogue-heavy",
        "action-sequence", "introspection", "ensemble-interaction",
        "investigation", "confrontation")

      Example STRUCTURE section:
      RHYTHM: false-victory-into-disaster
      OPENING: quiet dialogue
      RESOLUTION: failure-and-retreat
      EMOTIONAL-ARC: high-to-low
      DOMINANT-SCENE-TYPE: confrontation

      ***CRITICAL: Return ONLY the summary with both sections. NO introduction, NO chapter number header, NO explanation.***

      {{ chapter_text }}
```

This change is backward-compatible: downstream agents that consume `previous_summaries` will
now see structural tags inline. The Momentum & Distinctiveness agent, Operational
Distinctiveness agent, and Compression Check will naturally use these tags when comparing
chapters, because the tags are present in the text they already receive.

No changes needed in `prompts.py`, `pipeline.py`, or `chapters.py` — the summary builder
signature is unchanged.

---

## Fix 4 — Upgrade the Scene Variety Auditor to cross-chapter awareness ✅

### Problem
The Scene Variety Auditor (`prompts.yml:878-935`) only checks for repetition *within* a
single chapter. It catches "your 3 scenes all open the same way" but cannot detect "this
chapter opens the same way as the last 3 chapters" because it never receives
`previous_summaries` or any cross-chapter context.

### Fix

Feed `previous_summaries` (which now include structural metadata from Fix 3) into the
Scene Variety Auditor so it can check both intra- and inter-chapter repetition.

**File: `novelforge/agents/chapter/prompts.py`** — Update builder signature:

```python
def build_scene_variety_compression_auditor_prompt(
    chapter_text: str, chapter_summary: str, chapter_num: int, title: str,
    previous_summaries: str = "",  # NEW parameter
) -> list[dict[str, str]]:
    """Build the scene variety and compression audit prompt."""
    return render_prompt(
        "scene_variety_compression_auditor", title=title, chapter_num=chapter_num,
        chapter_summary=chapter_summary,
        previous_summaries=previous_summaries or "",  # NEW
        chapter_text=chapter_text,
    )
```

**File: `novelforge/agents/chapter/pipeline.py`** — Pass `previous_summaries` to the auditor:

```python
scene_audit_directives = run_scene_variety_compression_auditor(
    chapter_text=text, chapter_summary=chapter_outline_summary,
    chapter_num=chapter_num, title=title,
    previous_summaries=previous_summaries,  # NEW
    degraded_passes=degraded_passes,
)
```

Also update `run_scene_variety_compression_auditor()` to accept and pass through the new param:

```python
def run_scene_variety_compression_auditor(
    chapter_text: str, chapter_summary: str, chapter_num: int, title: str,
    previous_summaries: str = "",  # NEW
    degraded_passes: list[dict] | None = None,
) -> str:
    try:
        return call_llm(
            build_scene_variety_compression_auditor_prompt(
                chapter_text=chapter_text, chapter_summary=chapter_summary,
                chapter_num=chapter_num, title=title,
                previous_summaries=previous_summaries,  # NEW
            ),
            action=f"Chapter {chapter_num}: scene variety & compression audit",
        )
    ...
```

**File: `prompts.yml`** — Update `scene_variety_compression_auditor` prompt to add cross-chapter checks:

Add after the existing 5 categories in the system prompt (after line 911):

```yaml
      6. CROSS-CHAPTER STRUCTURAL REPETITION – Compare this chapter's macro
         structure against previous chapters. Flag if:
         - This chapter opens the same way as the previous chapter
         - The problem-resolution approach matches recent chapters
           (especially "discover problem → plan → execute")
         - The emotional arc shape repeats the last 2-3 chapters
         - Scene types cluster (e.g., 3 consecutive dialogue-heavy chapters)
         Flag the repetition and suggest a specific structural alternative.
```

Add to the user prompt (after line 921):

```yaml
      {%- if previous_summaries.strip() %}

      Previous chapter summaries (for cross-chapter comparison):
      {{ previous_summaries }}

      In addition to intra-chapter checks, compare this chapter's structure
      against the previous chapters. Flag any cross-chapter repetition in
      openings, resolutions, emotional arcs, or dominant scene types.
      {%- endif %}
```

---

## Fix 5 — Restructure compression guidance as a structured directive block ✅

### Problem
The per-chapter compression check outputs 3-5 bullet points of freeform prose, which the
next chapter's draft receives as one paragraph buried among 20+ other directives (planning
context, gatekeeper brief, character arcs, technology rules, rhythm directive, etc.). The
LLM deprioritizes soft advisory text when it conflicts with its structural instincts.

### Fix

Make the compression guidance structured and position it as a hard constraint, not soft advice.

**File: `prompts.yml`** — Modify the `per_chapter_compression_check` prompt (around line 1450):

Replace the output format instruction:

```yaml
      Return your guidance as a structured directive block in this exact format:

      BANNED OPERATIONS: [list operations from prior chapters that must NOT appear in the next chapter, or "none"]
      BANNED EMOTIONAL BEATS: [list emotional beats already delivered that must NOT repeat, or "none"]
      BANNED OPENINGS: [list chapter opening types used in the last 2-3 chapters, or "none"]
      BANNED RESOLUTIONS: [list problem-resolution approaches used recently, or "none"]
      REQUIRED CONTRAST: [one sentence describing what structural shape the next chapter MUST use to create contrast]

      Each BANNED line should be specific and concise. The REQUIRED CONTRAST line should
      name a concrete alternative, not just "do something different."

      If no significant redundancy risks exist, return: 'No compression concerns for next chapter.'
```

**File: `prompts.yml`** — Modify how compression guidance appears in the `chapter_draft` prompt (around line 721):

Change the injection from soft advisory to hard constraint:

```yaml
      {%- if compression_guidance.strip() %}

      *** HARD CONSTRAINT — STRUCTURAL BANS FROM PREVIOUS CHAPTERS ***
      The following patterns have been used in recent chapters and MUST NOT
      appear in this chapter. Violating these bans will result in the chapter
      being flagged for rewrite.

      {{ compression_guidance }}

      *** END STRUCTURAL BANS ***
      {%- endif %}
```

This reframing — from "Narrative Compression Guidance (avoid these patterns)" to
"HARD CONSTRAINT — STRUCTURAL BANS" — significantly changes how the LLM prioritizes
the directive. Framing constraints as bans with consequences is much more effective than
advisory suggestions.

---

## Fix 6 — Pass rhythm context to downstream post-draft agents ✅

### Problem
Post-draft agents (Momentum & Distinctiveness, Structure, Operational Distinctiveness) receive
the chapter text and `previous_summaries` but have NO knowledge of what rhythm was intended
for this chapter. They reanalyze from scratch. This means they can't enforce rhythm compliance
and may inadvertently *undo* rhythm-appropriate structure during their edits (e.g., a
`lyrical-interlude` chapter might get flagged by Momentum & Distinctiveness for "low stakes"
when low stakes is the point of that rhythm).

### Fix

Pass `chapter_rhythm_shape` to the three agents that could conflict with it, and add a
guard clause to their prompts.

**File: `novelforge/agents/chapter/prompts.py`** — Update the three prompt builders:

```python
def build_narrative_momentum_distinctiveness_prompt(
    chapter_text: str, previous_summaries: str, chapter_summary: str,
    chapter_num: int, title: str, total_chapters: int,
    chapter_rhythm_shape: str = "",  # NEW
) -> list[dict[str, str]]:
    from novelforge.chapter_position import ChapterPosition
    escalation_target = ChapterPosition(chapter_num, total_chapters).get_escalation_target()
    return render_prompt(
        "narrative_momentum_distinctiveness", title=title, chapter_num=chapter_num,
        total_chapters=total_chapters, escalation_target=escalation_target,
        chapter_summary=chapter_summary, previous_summaries=previous_summaries or "",
        chapter_rhythm_shape=chapter_rhythm_shape or "",  # NEW
        chapter_text=chapter_text,
    )


def build_structure_agent_prompt(
    chapter_text: str, chapter_num: int, total_chapters: int, outline_summary: str,
    chapter_architecture_context: str = "",
    chapter_rhythm_shape: str = "",  # NEW
) -> list[dict[str, str]]:
    from novelforge.chapter_position import ChapterPosition
    phase_hint = ChapterPosition(chapter_num, total_chapters).get_structure_phase_hint()
    return render_prompt(
        "structure_agent", chapter_num=chapter_num, total_chapters=total_chapters,
        phase_hint=phase_hint, outline_summary=outline_summary,
        chapter_architecture_context=chapter_architecture_context or "",
        chapter_rhythm_shape=chapter_rhythm_shape or "",  # NEW
        chapter_text=chapter_text,
    )


def build_operational_distinctiveness_prompt(
    chapter_text: str, previous_summaries: str, chapter_summary: str,
    chapter_num: int, title: str,
    chapter_rhythm_shape: str = "",  # NEW
) -> list[dict[str, str]]:
    return render_prompt(
        "operational_distinctiveness", title=title, chapter_num=chapter_num,
        chapter_summary=chapter_summary, previous_summaries=previous_summaries or "",
        chapter_rhythm_shape=chapter_rhythm_shape or "",  # NEW
        chapter_text=chapter_text,
    )
```

**File: `novelforge/agents/chapter/pipeline.py`** — Pass `chapter_rhythm_shape` to each call:

```python
# Momentum & Distinctiveness (around line 293)
text = _safe(
    lambda t: build_narrative_momentum_distinctiveness_prompt(
        t, previous_summaries, chapter_outline_summary, chapter_num, title, total_chapters,
        chapter_rhythm_shape=chapter_rhythm_shape,  # NEW
    ),
    text, action=f"Chapter {chapter_num}: momentum & distinctiveness",
)

# Structure (around line 313)
text = _safe(
    lambda t: build_structure_agent_prompt(
        t, chapter_num, total_chapters, chapter_outline_summary, ctx.architecture,
        chapter_rhythm_shape=chapter_rhythm_shape,  # NEW
    ),
    text, action=f"Chapter {chapter_num}: checking structure",
)

# Operational Distinctiveness (around line 323)
text = _safe(
    lambda t: build_operational_distinctiveness_prompt(
        t, previous_summaries, chapter_outline_summary, chapter_num, title,
        chapter_rhythm_shape=chapter_rhythm_shape,  # NEW
    ),
    text, action=f"Chapter {chapter_num}: verifying operational distinctiveness",
)
```

**File: `prompts.yml`** — Add rhythm-awareness guard to each of the three prompts:

In `narrative_momentum_distinctiveness` system prompt (after line 1088):

```yaml
      RHYTHM AWARENESS: If a rhythm has been assigned to this chapter, respect
      its structural intent. A "lyrical-interlude" is SUPPOSED to have lower
      action stakes. An "emotional-valley-then-climb" is SUPPOSED to start at
      a low point. Do not flag rhythm-appropriate characteristics as problems.
      Only flag genuine redundancy or escalation failures WITHIN the rhythm's
      expected shape.
```

In `narrative_momentum_distinctiveness` user prompt (after line 1090):

```yaml
      {%- if chapter_rhythm_shape.strip() %}
      Assigned rhythm for this chapter: {{ chapter_rhythm_shape }}
      Respect this rhythm's structural intent — do not rewrite rhythm-appropriate pacing
      or tension levels. Focus only on redundancy with prior chapters and escalation within
      the rhythm's expected shape.
      {%- endif %}
```

In `structure_agent` user prompt (around line 955):

```yaml
      {%- if chapter_rhythm_shape.strip() %}
      This chapter has been assigned the "{{ chapter_rhythm_shape }}" narrative rhythm.
      Ensure structural edits preserve this rhythm's shape. Do not restructure the chapter
      into a generic pattern.
      {%- endif %}
```

In `operational_distinctiveness` user prompt (around line 1195):

```yaml
      {%- if chapter_rhythm_shape.strip() %}
      This chapter is using the "{{ chapter_rhythm_shape }}" narrative rhythm. Some rhythms
      (lyrical-interlude, quiet-before-storm, emotional-valley-then-climb) may intentionally
      have fewer or no major operations. Do not force operational variety where the rhythm
      calls for restraint.
      {%- endif %}
```

---

## Implementation Order and Testing

1. **Fix 3 (enriched summaries)** should be implemented first — it's the simplest change
   (prompt-only) and all other fixes benefit from having structural metadata in summaries.

2. **Fix 5 (structured compression guidance)** second — also prompt-only, no code changes,
   and immediately improves how the next chapter's draft interprets anti-repetition directives.

3. **Fix 1 (rhythm history tracking)** third — requires code changes in 4 files but is
   straightforward. Enables the classifier to make truly informed recommendations.

4. **Fix 6 (rhythm context to downstream agents)** fourth — prevents downstream agents from
   undoing rhythm-appropriate structure.

5. **Fix 2 (rhythm compliance verifier)** last — adds a new agent pass (most complex), and
   benefits from all other fixes being in place.

6. **Fix 4 (cross-chapter scene variety)** can be done any time after Fix 3, since it
   benefits from enriched summaries.

### Test impact

- Fixes 3 and 5 are prompt-only changes — no Python code changes, so existing tests pass
  without modification.
- Fixes 1, 2, 4, and 6 add parameters to existing functions. Tests that call these
  functions without the new parameters will still pass because all new parameters have
  defaults. However, the `mock_llm` fixture in `tests/conftest.py` should be checked to
  ensure it patches the new prompt builders if they import `call_llm` directly.
- New unit tests should be added for:
  - `build_rhythm_compliance_verifier_prompt()` — verifies prompt construction.
  - `run_chapter_rhythm_classifier()` with `rhythm_log` — verifies the log is forwarded.
  - `_run_all_chapter_agents()` with `chapter_rhythm_shape` — verifies the compliance
    verifier runs when a rhythm is assigned and is skipped when it's empty.
  - `build_scene_variety_compression_auditor_prompt()` with `previous_summaries` — verifies
    cross-chapter context is included.

### LLM cost impact

- Fix 2 adds one additional LLM call per chapter. For a 25-chapter novel this adds ~25 calls
  (currently ~570 total, so ~4.4% increase).
- All other fixes add zero additional LLM calls — they enrich existing prompts.
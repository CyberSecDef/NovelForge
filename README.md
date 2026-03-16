# NovelForge

**NovelForge** is a Python web application that generates complete, publication-quality fiction novels using a large language model (LLM) API. Built with Flask, Bootstrap 5, and jQuery, it provides a guided, multi-step workflow that takes a story premise from concept to a fully written, exported Markdown manuscript—without requiring any page reloads.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Usage Walkthrough](#usage-walkthrough)
  - [Step 1 – Novel Setup](#step-1--novel-setup)
  - [Step 2 – Review & Edit Outline](#step-2--review--edit-outline)
  - [Step 3 – Chapter Generation](#step-3--chapter-generation)
  - [Step 4 – Export & Revision](#step-4--export--revision)
- [API Routes](#api-routes)
- [LLM Integration & Agents](#llm-integration--agents)
  - [Prompt Architecture](#prompt-architecture)
  - [Planning Agents](#planning-agents)
  - [Per-Chapter Agents](#per-chapter-agents)
  - [Post-Generation Audit Agents](#post-generation-audit-agents)
  - [Anti-LLM Agent](#anti-llm-agent)
  - [Rate Limiting & Retries](#rate-limiting--retries)
- [Novel Architecture Model](#novel-architecture-model)
- [Session Management](#session-management)
- [Security](#security)
- [Testing](#testing)
- [Deployment Notes](#deployment-notes)
- [License](#license)

---

## Features

- **Full Novel Generation** – Produces complete fiction novels (default 80,000–90,000 words) chapter by chapter using a configurable LLM API.
- **Multi-Step Guided Workflow** – Four-step single-page application: input → outline review → chapter writing → export.
- **Editable Outline** – AI-generated title, chapter-by-chapter outline, and character list are all fully editable before writing begins.
- **Seven Genre Options** – Fantasy, Sci-Fi, Mystery, Romance, Horror, Thriller, Historical.
- **Structured Story Architecture** – Outline generation follows a nine-phase narrative model (Hook → Setup → Inciting Incident → Rising Action → Midpoint Shift → Complications → Crisis → Climax → Resolution) with correct structural proportions.
- **Seven Planning Agents** – Before chapter generation, specialized agents create comprehensive constraints: Story Architecture, Master Timeline, Character Fate Registry, Character Arcs, Antagonist Motivations, Technology Rules, and Theme Reinforcement.
- **16+ Per-Chapter Agents** – Each chapter passes through continuity gatekeeper, drafting, dialogue refinement, scene structuring, context analysis, editing, redundancy detection, structure checking, operational distinctiveness, character arc deepening, character thread tracking, synthesis, polishing, anti-LLM pass, quality control, story momentum tracking, and summarization.
- **Per-Chapter Compression Check** – After each chapter, a compression analyzer identifies redundancy patterns and provides guidance to the next chapter to avoid repetition.
- **Anti-LLM Agent** – Dedicated LLM pass that removes robotic language patterns, overused phrases, and LLM hallmarks to produce human-sounding prose.
- **Continuity Tracking** – Each completed chapter generates a 100–200 word summary that is fed to subsequent chapters to maintain consistency. Character state is tracked and updated after each chapter.
- **Eight Post-Generation Audit Agents** – Comprehensive analysis including consistency pass, global continuity audit, narrative compression analysis, character resolution validation, thematic payoff analysis, climax integrity check, loose thread resolution, and reader immersion testing.
- **Chapter Revision** – Users can revise any chapter with custom instructions; the revised chapter runs through the full agent pipeline.
- **Comprehensive Editor's Notes** – Export all diagnostic reports from the 8 post-generation audits to identify chapters needing revision.
- **Session Persistence** – Crash recovery automatically saves progress; interrupted generations can be resumed.
- **Live Progress Bar** – Browser polls the backend every 3 seconds; a Bootstrap progress bar updates in real time as chapters are written.
- **Markdown Export** – Compiled novel (title, chapters as `##` headings, inline summaries, and optional editor's notes) is saved server-side and served as a downloadable `.md` file.
- **AJAX-only UI** – All form submissions and data fetches use jQuery AJAX; the page never reloads.
- **Input Validation** – Both client-side (jQuery) and server-side (Python) validation with Bootstrap feedback messages.
- **XSS Protection** – All user-supplied content is escaped with `markupsafe.escape` before storage and with jQuery's `.text()` before rendering.
- **Flask-Session** – Server-side filesystem sessions keep user data across the multi-step workflow.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, Flask 3.x |
| Sessions | Flask-Session 0.8+ (cachelib filesystem backend) |
| HTTP client | `requests` |
| Frontend framework | Bootstrap 5.3 |
| Frontend icons | Bootstrap Icons 1.11 |
| Client scripting | jQuery 3.7 |
| LLM API | Any OpenAI-compatible chat completions endpoint |
| Templating | Jinja2 (via Flask) |
| Export format | Markdown (`.md`) |

---

## Project Structure

```
NovelForge/
├── app.py                  # Flask application: routes, LLM helpers, prompt builders
├── config.py               # Configuration loaded from environment variables
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variable configuration
├── CLAUDE.md               # Guidance for Claude Code AI assistant
├── templates/
│   └── index.html          # Single-page application HTML (Bootstrap 5)
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   └── js/
│       └── script.js       # jQuery client logic (AJAX, UI state, progress polling)
├── tests/
│   ├── __init__.py
│   └── test_app.py         # pytest test suite
├── flask_session/          # Server-side session files (auto-created at runtime)
├── sessions/               # Session persistence files for crash recovery (auto-created)
├── exports/                # Generated novel Markdown files (auto-created at runtime)
└── logs/                   # LLM request/response logs (auto-created at runtime)
```

---

## Requirements

- Python **3.11** or newer
- An API key for an **OpenAI-compatible** LLM (OpenAI, Azure OpenAI, Ollama, LM Studio, etc.)
- Internet access to reach your chosen LLM endpoint

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/CyberSecDef/NovelForge.git
   cd NovelForge
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Copy and fill in the environment configuration**

   ```bash
   cp .env.example .env
   # Edit .env and set LLM_API_KEY (and optionally LLM_API_URL, LLM_MODEL, SECRET_KEY)
   ```

---

## Configuration

All settings are read from environment variables. Copy `.env.example` to `.env` and set the following:

| Variable | Default | Description |
|---|---|---|
| `LLM_API_URL` | `https://api.openai.com/v1/chat/completions` | LLM API endpoint. Must be OpenAI-compatible (chat completions format). |
| `LLM_API_KEY` | *(empty – required)* | API key sent as `Authorization: Bearer <key>`. |
| `LLM_MODEL` | `gpt-4o` | Model name passed in the request payload. |
| `SECRET_KEY` | `change-me-in-production` | Flask secret key used to sign session cookies. **Must be changed in production.** |
| `SESSION_FILE_DIR` | `./flask_session` | Directory where server-side session files are stored. |
| `EXPORT_DIR` | `./exports` | Directory where generated Markdown files are saved. |
| `FLASK_HOST` | `127.0.0.1` | Host to bind to when running via `python app.py`. |
| `FLASK_PORT` | `5000` | Port to bind to when running via `python app.py`. |

> **Using a local or alternative LLM:** Set `LLM_API_URL` to your endpoint (e.g., `http://localhost:11434/v1/chat/completions` for Ollama) and set `LLM_MODEL` to your model name. JSON mode (`response_format: json_object`) must be supported by the model for structured outputs; if not, the application falls back to best-effort parsing.

Example `.env`:

```dotenv
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=sk-your-api-key-here
LLM_MODEL=gpt-4o
SECRET_KEY=replace-with-a-long-random-string
```

---

## Running the App

```bash
python app.py
```

The application starts on `http://127.0.0.1:5000` by default. Open this URL in your browser.

> **Note:** `python app.py` uses Flask's built-in development server (`debug=False`). For production, use a WSGI server such as **Gunicorn** behind a reverse proxy (e.g., Nginx):
>
> ```bash
> gunicorn -w 4 -b 0.0.0.0:8000 "app:app"
> ```

---

## Usage Walkthrough

### Step 1 – Novel Setup

Fill in the input form on the home page:

| Field | Details |
|---|---|
| **Story Premise** | Required. Describe your story idea. Maximum **2,000 characters** (enforced client- and server-side with a live counter). |
| **Genre** | Required. Select one of: Fantasy, Sci-Fi, Mystery, Romance, Horror, Thriller, Historical. |
| **Number of Chapters** | Required. Minimum **3**. Recommended 15–25 (tooltip shown). |
| **Target Word Count** | Required. Minimum 1,000. Recommended 80,000–90,000 (tooltip shown). |
| **Special Events** | Optional. Comma-separated or bulleted events to incorporate (e.g., *"A dragon attack in chapter 5, A wedding in the final act"*). |
| **Special Instructions** | Optional. Global writing guidance (e.g., *"Avoid clichés, emphasize diversity, dark tone"*). |

Click **Generate Outline**. The form is validated client-side before the AJAX POST to `/generate_outline`. A spinner indicates the LLM is working. The application then:

1. Generates a catchy title.
2. Generates a chapter-by-chapter outline following the nine-phase narrative architecture.
3. Generates 3–7 main characters with name, age, background, role, and arc.
4. Runs all seven planning agents to create comprehensive story constraints.

### Step 2 – Review & Edit Outline

The generated outline is displayed in an editable table. You can:

- Edit the **novel title** in an input field.
- Edit any **chapter title** or **chapter summary** inline (cells are `contenteditable`).
- Edit any **character field** (name, age, role, background, arc) inline.
- Add or delete chapters and characters.

Click **Approve & Write Chapters** when satisfied. Edits are collected by jQuery and POSTed to `/approve_outline`, which saves the final outline to the session and regenerates all planning agents based on your changes. Chapter writing then begins automatically.

### Step 3 – Chapter Generation

A Bootstrap progress bar tracks writing progress. The progress label updates in real time to show the current agent step. The browser polls `/progress/<token>` every 3 seconds.

**Before chapter generation**, seven planning agents create comprehensive constraints:

1. **Story Architecture Planner** – Creates 3-act or 4-act structure with per-chapter phase assignments.
2. **Master Timeline Planner** – Tracks events, constraints, and character states across chapters.
3. **Character Fate Registry Planner** – Monitors character status, injuries, and definitive outcomes.
4. **Character Arc Planner** – Plans character development beats and transformations.
5. **Antagonist Motivation Planner** – Tracks antagonist goals, tactics, and escalation.
6. **Technology Rules Planner** – Defines system limits, costs, and failure modes.
7. **Theme Reinforcement Planner** – Assigns thematic roles and guidance per chapter.

**For each chapter**, the backend runs a **multi-step pipeline**:

1. **Continuity Gatekeeper** – Validates chapter constraints before drafting (hard constraints).
2. **Draft** – Initial prose written with full context and all planning agent guidance.
3. **Dialog Agent** – Refines all dialogue for natural rhythm, distinct character voices, and subtext.
4. **Scene Agent** – Ensures every scene follows the Goal → Obstacle → Outcome → New Problem pattern.
5. **Context Analyzer** – Checks world-building facts, timeline, and technology rules.
6. **Editing Agent** – Fixes plot holes, pacing problems, and character inconsistencies.
7. **Narrative Redundancy Detector** – Eliminates repeated information from previous chapters.
8. **Structure Agent** – Verifies the chapter fulfils its designated role in the story architecture.
9. **Operational Distinctiveness Agent** – Ensures each chapter has unique operations and methods.
10. **Character Agent** – Deepens character arcs and corrects any out-of-character moments.
11. **Character Thread Tracker** – Maintains character arc consistency across chapters.
12. **Synthesizer** – Unifies narrative voice and thematic thread after all specialist passes.
13. **Polish Agent** – Elevates grammar, style, and vivid language.
14. **Anti-LLM Agent** – Dedicated pass to strip robotic patterns and overused LLM words.
15. **Quality Controller** – Checks reader engagement, tension, pacing, and hook strength.
16. **Story Momentum Tracker** – Ensures proper pacing and momentum across the novel.
17. **Summarizer** – Produces a 100–200 word continuity summary for subsequent chapters.

**After each chapter**:
- **Character State Updater** – Records definitive character states for the next chapter.
- **Per-Chapter Compression Check** – Analyzes the chapter against previous chapters and provides guidance to the next chapter about what patterns to avoid repeating.

**After all chapters are written**, eight post-generation audit agents run:

1. **Consistency Pass** – Reviews all summaries for plot holes and unresolved threads.
2. **Global Continuity Auditor** – Checks for contradictions, character state errors, and timeline errors.
3. **Narrative Compression Editor** – Identifies redundant sequences and emotional beat repetitions.
4. **Character Resolution Validator** – Confirms every major character receives closure.
5. **Thematic Payoff Analyzer** – Ensures all themes culminate properly.
6. **Climax Integrity Checker** – Verifies protagonist makes a definitive moral decision.
7. **Loose Thread Resolver** – Identifies unresolved narrative questions.
8. **Reader Immersion Tester** – Evaluates pacing, tension, and engagement.

Completed chapters appear in the list as they finish, each marked with a green check.

### Step 4 – Export & Revision

When generation is complete:

- The novel title and approximate word count are shown.
- Any editor's notes from the audit agents are displayed.
- An expandable **accordion** lets you preview each chapter's content inline.
- **Revise chapters** – Click on any chapter to provide custom revision instructions. The chapter will be re-generated through the full agent pipeline.
- Click **Download as Markdown** to export the full novel. The Markdown file includes:
  - `# Title`
  - `## Chapter N: Title` headings
  - Italicised chapter summaries
  - Full chapter prose
- Click **Download Editor's Notes** to export comprehensive diagnostic reports from all 8 post-generation audits, helping you identify which chapters need revision.

Click **Start Over** to reset the form and begin a new novel.

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page application HTML. |
| `POST` | `/generate_outline` | Phase 1: generates title, chapter outline, characters, and runs all planning agents. |
| `POST` | `/approve_outline` | Saves user-edited outline and regenerates all planning agents. |
| `POST` | `/generate_chapters` | Phase 2: starts background chapter generation thread. Returns `{token}` for progress polling. |
| `GET` | `/progress/<token>` | Returns JSON progress object with status, current chapter, and all audit reports. |
| `POST` | `/revise_chapter` | Revise a specific chapter with custom instructions through the full agent pipeline. |
| `POST` | `/export` | Compiles novel to Markdown and saves to `EXPORT_DIR`. Returns `{download_url}`. |
| `POST` | `/export_editors_notes` | Exports all diagnostic reports to Markdown. Returns `{download_url}`. |
| `GET` | `/download/<filename>` | Serves a generated Markdown file as an attachment. |
| `GET` | `/check_saved_state` | Checks for resumable session from crash recovery. |
| `POST` | `/resume_session` | Resumes an interrupted generation from saved state. |
| `POST` | `/new_session` | Starts a fresh session, archiving current state. |
| `GET` | `/llm_log` | Returns recent LLM log entries for debugging. |

### `/generate_outline` Request Body

```json
{
  "premise": "A young mage discovers a forbidden library…",
  "genre": "Fantasy",
  "chapters": 20,
  "word_count": 85000,
  "special_events": "A dragon attack in chapter 5",
  "special_instructions": "Avoid clichés, emphasize diversity"
}
```

### `/progress/<token>` Response

```json
{
  "status": "running",
  "current": 7,
  "total": 20,
  "step": "Chapter 7: refining dialogue",
  "chapters_done": [
    { "number": 1, "title": "The Awakening", "content": "…", "summary": "…" }
  ],
  "error": null,
  "consistency": {},
  "global_continuity_audit": {},
  "narrative_compression_report": {},
  "character_resolution_report": {},
  "thematic_payoff_report": {},
  "climax_integrity_report": {},
  "loose_thread_report": {},
  "reader_immersion_report": {}
}
```

Status values: `"running"` | `"done"` | `"error"`.

### `/revise_chapter` Request Body

```json
{
  "token": "<progress_token>",
  "chapter_number": 5,
  "instructions": "Add more tension to the confrontation scene and deepen the protagonist's internal conflict."
}
```

---

## LLM Integration & Agents

All LLM calls are made by the `call_llm()` function in `app.py` using the `requests` library. The function:

- Adds `Authorization: Bearer <LLM_API_KEY>` and `Content-Type: application/json` headers.
- Sends the model name, message list, and optional `response_format: {type: "json_object"}` for structured outputs.
- Retries up to **5 times** with exponential back-off (2s, 4s, 6s, 8s, 10s) on HTTP 429 (rate limit) and 5xx errors.
- Uses a 240-second timeout per request.
- Logs all requests and responses to `./logs/llm.log` as JSON.
- Raises `RuntimeError` after exhausting retries, which the route handler converts to an HTTP 502 response.

### Prompt Architecture

Each LLM call uses a two-message structure:

```
system  → defines the agent's role and voice constraints
user    → provides the task with all relevant context
```

JSON mode (`response_format: json_object`) is used wherever structured data is expected (outline, characters, planning agents, audit reports) to guarantee parseable output. A fallback is applied if JSON decoding fails.

### Planning Agents

Before chapter generation begins, seven planning agents create comprehensive constraints that guide all subsequent writing:

| Agent | Purpose | Output |
|---|---|---|
| **Story Architecture Planner** | Creates 3-act (< 16 chapters) or 4-act structure | Per-chapter phase, purpose, escalation targets, operation limits |
| **Master Timeline Planner** | Tracks events and character states | Event ledger, chapter constraints, character state tracking |
| **Character Fate Registry Planner** | Monitors character outcomes | Status tracking, injury records, outcome locks, conflict checks |
| **Character Arc Planner** | Plans character development | Arc beats per chapter, transformation milestones, consistency rules |
| **Antagonist Motivation Planner** | Tracks antagonist goals and tactics | Motivation core, escalation plan, pressure points, consistency rules |
| **Technology Rules Planner** | Defines system constraints | Latency, costs, blind spots, failure modes, forbidden capabilities |
| **Theme Reinforcement Planner** | Assigns thematic guidance | Theme appearances per chapter, thematic arcs, chapter-specific guidance |

All planning agents are regenerated when the user edits the outline in Step 2 to ensure consistency with any changes.

### Per-Chapter Agents

Each chapter passes through a comprehensive pipeline of specialized agents:

| Phase | Agent | Purpose |
|---|---|---|
| Pre-draft | **Continuity Gatekeeper** | Validates hard constraints before writing |
| Step 1 | **Draft Agent** | Writes initial chapter with full context and planning guidance |
| Step 2 | **Dialog Agent** | Refines dialogue for naturalism and distinct voices |
| Step 3 | **Scene Agent** | Enforces Goal → Obstacle → Outcome → New Problem pattern |
| Step 4 | **Context Analyzer** | Fixes world-building and timeline inconsistencies |
| Step 5 | **Editing Agent** | Repairs plot holes, pacing, and character issues |
| Step 6 | **Narrative Redundancy Detector** | Eliminates repeated information |
| Step 7 | **Structure Agent** | Confirms chapter fulfils its architectural role |
| Step 8 | **Operational Distinctiveness Agent** | Ensures unique operations per chapter |
| Step 9 | **Character Agent** | Deepens character arcs |
| Step 10 | **Character Thread Tracker** | Maintains arc consistency |
| Step 11 | **Synthesizer Agent** | Unifies narrative voice and theme |
| Step 12 | **Polish Agent** | Elevates grammar, style, and language |
| Step 13 | **Anti-LLM Agent** | Strips robotic patterns and forbidden words |
| Step 14 | **Quality Controller** | Checks engagement, tension, and pacing |
| Step 15 | **Story Momentum Tracker** | Ensures proper pacing across novel |
| Step 16 | **Summarizer Agent** | Produces continuity summary |
| Post-chapter | **Character State Updater** | Records definitive character states |
| Post-chapter | **Compression Check** | Analyzes redundancy and provides guidance for next chapter |

### Post-Generation Audit Agents

After all chapters are complete, eight audit agents analyze the full manuscript:

| Agent | Purpose | Key Outputs |
|---|---|---|
| **Consistency Pass** | Reviews all summaries for issues | Issues list, overall assessment |
| **Global Continuity Auditor** | Checks for contradictions | Contradictions, character state errors, timeline errors |
| **Narrative Compression Editor** | Identifies redundancy | Redundant sequences, emotional beat repetitions, compression priority |
| **Character Resolution Validator** | Confirms character closure | Unresolved characters, resolution status |
| **Thematic Payoff Analyzer** | Ensures theme culmination | Abandoned themes, weak payoffs, thematic integrity |
| **Climax Integrity Checker** | Verifies protagonist decision | Climax chapter, decision checks, integrity failures |
| **Loose Thread Resolver** | Identifies open questions | Unresolved threads, dangling setups, intentionally open threads |
| **Reader Immersion Tester** | Evaluates reader experience | Engagement score, weak chapters, immersion breaks, recommendations |

All audit reports are included in the Editor's Notes export.

### Anti-LLM Agent

The **Anti-LLM Agent** is a dedicated LLM pass with a single responsibility: making AI-generated text sound genuinely human-written. It:

- Replaces or removes **overused LLM words**: `embark`, `delve`, `realm`, `tapestry`, `testament`, `nuance`, `beacon`, `uncharted`, `multifaceted`, `leverage`, `synergy`, `pivotal`, `groundbreaking`, `commendable`, `meticulous`.
- Eliminates **robotic transition phrases** ("In conclusion", "It is worth noting", "As a result of this").
- Removes **unnecessary hedging** and repetitive sentence openings.
- Introduces **subtle human imperfections**: varied sentence length, occasional fragments for emphasis, colloquial rhythms where fitting.
- Does **not** change plot, characters, or factual content.

### Rate Limiting & Retries

```python
MAX_RETRIES = 5
RETRY_DELAYS = [2, 4, 6, 8, 10]  # seconds (exponential back-off)
TIMEOUT = 240  # seconds per request
```

On HTTP 429 or 5xx responses, the client waits before retrying with exponential back-off. Persistent failures raise `RuntimeError` and return an HTTP 502 with an error message displayed as a Bootstrap alert.

---

## Novel Architecture Model

The outline prompt instructs the LLM to structure the story according to the following nine-phase model:

| Phase | Structural Position | Purpose |
|---|---|---|
| **1. Hook** | Beginning (25%) | Open with movement, tension, or a compelling question. |
| **2. Setup** | Beginning (25%) | Establish protagonist, world, tone, normal state, and central conflict. |
| **3. Inciting Incident** | Beginning (25%) | A disruptive event that sets the story in motion. |
| **4. Rising Action** | Middle (50%) | Escalating obstacles, choices, and consequences. |
| **5. Midpoint Shift** | Middle (50%) | A major revelation or reversal that changes the protagonist's direction. |
| **6. Complications** | Middle (50%) | Worsening problems, rising stakes, ticking clocks. |
| **7. Crisis** | End (25%) | The hardest decision the protagonist must make. |
| **8. Climax** | End (25%) | The decisive confrontation or moment of highest tension. |
| **9. Resolution** | End (25%) | Aftermath and visible character change. |

**Architecture selection:** 3-act model for novels with fewer than 16 chapters; 4-act model for 16+ chapters.

**Scene-level pattern:** Goal → Obstacle → Outcome → New problem.

**Layered consistency targets:** Character arc, theme, cause-and-effect chain, escalation, and payoff.

---

## Session Management

NovelForge uses **Flask-Session** with a `cachelib` filesystem backend to store user workflow state server-side. Cookie-based sessions alone are insufficient for the volume of data (full chapter lists, character tables, planning agent outputs, etc.).

### Session Persistence & Crash Recovery

Progress is automatically saved to `./sessions/<token>_progress.json` after each step. If the browser is closed or the server restarts during generation:

1. On page load, the app checks for saved state via `/check_saved_state`.
2. A modal dialog offers to resume the interrupted generation.
3. Clicking "Resume" calls `/resume_session` to continue from where it left off.
4. Clicking "Start Fresh" calls `/new_session` to archive and clear the state.

### Session Keys

| Key | Set by | Content |
|---|---|---|
| `session_id` | Auto-generated | Unique session identifier |
| `premise` | `/generate_outline` | Sanitised story premise string |
| `genre` | `/generate_outline` | Selected genre string |
| `chapters` | `/generate_outline` | Integer chapter count |
| `word_count` | `/generate_outline` | Integer target word count |
| `special_events` | `/generate_outline` | Optional events string |
| `special_instructions` | `/generate_outline` | Optional instructions string |
| `title` | `/generate_outline`, `/approve_outline` | Novel title string |
| `chapter_list` | `/generate_outline`, `/approve_outline` | List of `{number, title, summary}` dicts |
| `character_list` | `/generate_outline`, `/approve_outline` | List of `{name, age, background, role, arc}` dicts |
| `story_architecture` | `/generate_outline`, `/approve_outline` | Story Architecture Planner output |
| `master_timeline` | `/generate_outline`, `/approve_outline` | Master Timeline Planner output |
| `character_fate_registry` | `/generate_outline`, `/approve_outline` | Character Fate Registry output |
| `character_arc_plan` | `/generate_outline`, `/approve_outline` | Character Arc Planner output |
| `antagonist_motivation_plan` | `/generate_outline`, `/approve_outline` | Antagonist Motivation Planner output |
| `technology_rules` | `/generate_outline`, `/approve_outline` | Technology Rules Planner output |
| `theme_reinforcement` | `/generate_outline`, `/approve_outline` | Theme Reinforcement Planner output |
| `progress_token` | `/generate_chapters` | UUID token for progress polling |

Session files are written to `SESSION_FILE_DIR` (default: `./flask_session`) with file permissions `0o600`.

---

## Security

| Concern | Mitigation |
|---|---|
| **XSS via user input** | All user-supplied strings are escaped with `markupsafe.escape` before being stored in the session and rendered. jQuery's `.text()` method is used exclusively (never `.html()`) when inserting dynamic content into the DOM. |
| **Input validation** | Server-side validation (`validate_outline_input`) rejects empty premises, premises over 2,000 characters, unrecognised genre values, chapter counts below 3, and word counts below 1,000. Mirrored client-side with jQuery. |
| **Directory traversal** | The `/download/<filename>` route strips any path components using `Path(filename).name` before building the export path. |
| **Secret key** | Flask's `SECRET_KEY` must be set to a long random string via environment variable in production. The default value `change-me-in-production` is intentionally insecure and must not be used outside local development. |
| **API key exposure** | The LLM API key is read from an environment variable and never returned to the client. Keys are sanitized in log files. |
| **Session file permissions** | Session files are created with mode `0o600` (owner read/write only). |
| **No `debug=True` in production** | `app.run()` is called with `debug=False`. The interactive debugger is never exposed. |

---

## Testing

The test suite uses **pytest** and Flask's built-in test client. No live LLM calls are made.

```bash
pip install pytest
pytest tests/
```

### Test Coverage

**`TestValidateOutlineInput`** – Unit tests for the `validate_outline_input` helper:

- Valid input passes.
- Empty premise is rejected.
- Premise character limit enforcement.
- Unrecognised genre is rejected.
- All seven allowed genres are accepted.
- Chapter count below 3 is rejected; exactly 3 is accepted.
- Non-numeric chapter count is rejected.
- Word count below 1,000 is rejected; exactly 1,000 is accepted.
- Non-numeric word count is rejected.
- Missing fields (empty dict) are rejected.

**`TestRoutes`** – Integration tests for Flask routes via the test client:

- `GET /` returns HTTP 200 and contains "NovelForge".
- `POST /generate_outline` with empty body returns HTTP 400.
- `POST /generate_outline` with invalid genre returns HTTP 400.
- `POST /approve_outline` with empty title returns HTTP 400.
- `POST /approve_outline` with empty chapter list returns HTTP 400.
- `POST /approve_outline` with valid data returns HTTP 200 `{status: "approved"}`.
- `POST /generate_chapters` without prior session data returns HTTP 400.
- `GET /progress/<unknown-token>` returns HTTP 404.
- `GET /download/<nonexistent>` returns HTTP 404.
- `POST /export` with unknown token returns HTTP 400.
- `GET /download/../../etc/passwd` returns HTTP 404 (directory traversal blocked).

---

## Deployment Notes

- **WSGI server:** Use Gunicorn or uWSGI in production. Do not expose Flask's development server.
- **Reverse proxy:** Place Nginx or Apache in front of Gunicorn for TLS termination and static file serving.
- **`SECRET_KEY`:** Generate a cryptographically random key: `python -c "import secrets; print(secrets.token_hex(32))"`.
- **Export file cleanup:** Files in `EXPORT_DIR` are not automatically purged. Implement a cron job or scheduled task to remove stale exports.
- **Session file cleanup:** Flask-Session's cachelib backend prunes files when the threshold (500 by default) is exceeded. Monitor disk usage in long-running deployments.
- **Threading:** Chapter generation runs in a daemon `threading.Thread`. For multi-process deployments (e.g., multiple Gunicorn workers), replace `_progress_store` with a shared store such as Redis or a SQLite database.
- **Environment variables:** Use a secrets manager or your platform's secret injection (e.g., Docker secrets, Kubernetes Secrets, Heroku Config Vars) rather than committing `.env` to version control.

---

## License

This project is released under the [MIT License](LICENSE).

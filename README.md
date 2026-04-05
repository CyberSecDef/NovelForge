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

## Architecture

### System Overview

```
+------------------+        AJAX / JSON          +---------------------+
|                  | ---------------------------> |                     |
|     Browser      |    POST /generate_outline    |   Flask App         |
|  (Bootstrap 5 +  |    POST /approve_outline     |   (novelforge/)     |
|   jQuery SPA)    |    POST /generate_chapters   |                     |
|                  |    GET  /progress/<token>     |  +--------------+   |     +----------------+
|  - Step 1: Setup |    POST /export              |  | Routes       |   |     |                |
|  - Step 2: Edit  |    POST /revise_chapter      |  | (Blueprints) |-------->| LLM API        |
|  - Step 3: Write | <--------------------------- |  +--------------+   |     | (OpenAI-compat)|
|  - Step 4: Export|        JSON responses        |         |            |     +----------------+
|                  |                              |         v            |
+------------------+                              |  +--------------+   |     +----------------+
                                                  |  | Planning     |   |     |                |
          Progress polling (adaptive backoff)      |  | Agents (8)   |-------->| Image API      |
          15s -> 30s -> 60s cap                   |  +--------------+   |     | (optional)     |
                                                  |         |            |     +----------------+
                                                  |         v            |
                                                  |  +--------------+   |
                                                  |  | Background   |   |
                                                  |  | Thread       |   |
                                                  |  | (generation) |   |
                                                  |  +--------------+   |
                                                  |         |            |
                                                  |         v            |
                                                  |  +--------------+   |     +----------------+
                                                  |  | Session      |   |     | sessions/      |
                                                  |  | Persistence  |-------->|  novels/*.json  |
                                                  |  +--------------+   |     |  flask/*        |
                                                  |                     |     +----------------+
                                                  +---------------------+
```

### Per-Chapter Agent Pipeline

Each chapter passes through this pipeline sequentially. The background generation
thread orchestrates all steps, with a 60-minute per-chapter timeout.

```
                    +-------------------+
                    | Continuity        |
                    | Gatekeeper        |  Pre-draft: validate hard constraints
                    +--------+----------+
                             |
                    +--------v----------+
                    | Chapter Rhythm    |
                    | Classifier        |  Recommend contrasting narrative rhythm
                    +--------+----------+
                             |
                    +--------v----------+
                    | 1. Draft Agent    |  Initial prose with full planning context
                    +--------+----------+
                             |
              +--------------v--------------+
              |  12-Step Refinement Pipeline |
              |                             |
              |  2.  Prose Refinement       |  Dialogue + scene momentum
              |  3.  Scene Variety Audit    |  Detect intra-chapter repetition
              |  4.  Context Analyzer       |  World-building & timeline
              |  5.  Editing Agent          |  Plot holes, pacing, consistency
              |  6.  Momentum & Distinct.   |  Cross-chapter redundancy
              |  7.  Structure Agent        |  Fits story architecture role
              |  8.  Operational Distinct.  |  Unique ops per chapter
              |  9.  Character Agent        |  Deepen arcs & consistency
              |  10. Synthesizer            |  Unify voice and theme
              |  11. Polish Agent           |  Grammar, style, vivid language
              |  12. Anti-LLM Agent         |  Strip robotic patterns
              |                             |
              +--------------+--------------+
                             |
                    +--------v----------+
                    | 13. Quality       |
                    |     Controller    |  Engagement, tension, pacing check
                    +--------+----------+
                             |
                    +--------v----------+
                    | 14. Summarizer    |  100-200 word continuity summary
                    +--------+----------+
                             |
              +--------------v--------------+
              |  Post-Chapter Passes        |
              |                             |
              |  A. Character State Updater |  Record definitive states
              |  B. Compression Check       |  Redundancy guidance for next ch.
              +--------------+--------------+
                             |
                             v
                    [Next Chapter] or [Post-Manuscript Audits]


Post-Manuscript Audits (after all chapters):
  I.    Final Consistency Pass
  II.   Global Continuity Auditor
  III.  Narrative Compression Editor
  IV.   Character Resolution Validator
  V.    Thematic Payoff Analyzer
  VI.   Climax Integrity Checker
  VII.  Loose Thread Resolver
  VIII. Reader Immersion Tester
  IX.   Pacing & Tension Heatmap
```

### Planning Agent Dependency Graph

```
                  User approves outline
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
  +-------+------+ +-----+------+ +------+------+
  | Story Arch.  | | Master     | | Technology  |
  | Planner      | | Timeline   | | Rules       |   Group 1
  +--------------+ +-----+------+ +-------------+   (parallel)
                         |
  +--------------+       |         +-------------+
  | Theme        |       |         |             |
  | Reinforcement|       |         |             |
  +--------------+       |         |             |
          |              |         |             |
          +--------------+---------+             |
                         |                       |
          +--------------+---------------+       |
          |              |               |       |
          v              v               v       |
  +-------+------+ +----+-------+ +-----+-----+ |
  | Char. Fate   | | Char. Arc  | | Antagonist | |  Group 2
  | Registry     | | Planner    | | Motivation | |  (parallel)
  +--------------+ +-----+------+ +-----------+ |
                         |                       |
                         v                       |
                  +------+-------+               |
                  | POV & Focal  |               |  Group 3
                  | Character    | <-------------+  (sequential)
                  +--------------+
```

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
- **XSS Protection** – User-supplied content is stored as plain text in the session and rendered safely using Jinja2 auto-escaping (server-side) and jQuery's `.text()` exclusively (client-side; `.html()` is never used for dynamic content).
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
├── app.py                  # Thin entry point: from novelforge import create_app
├── config.py               # Backward-compat shim (imports from novelforge.config)
├── prompts.yml             # LLM prompt templates (YAML)
├── requirements.txt        # Runtime Python dependencies
├── requirements-dev.txt    # Development-only dependencies (pytest, mypy, type stubs)
├── .env.example            # Template for environment variable configuration
├── CLAUDE.md               # Guidance for Claude Code AI assistant
├── novelforge/             # Main application package
│   ├── __init__.py         # App factory: create_app(), limiter, index route
│   ├── config.py           # Configuration from environment variables
│   ├── progress.py         # Shared progress store, correlation IDs
│   ├── validation.py       # Input validation, ALLOWED_GENRES
│   ├── chapter_position.py # ChapterPosition utility (phase, act, landmarks)
│   ├── llm/
│   │   ├── __init__.py     # Re-exports
│   │   ├── client.py       # call_llm(), circuit breaker, token usage tracking
│   │   ├── prompts.py      # YAML prompt loading and Jinja2 rendering
│   │   └── image.py        # call_image_api()
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py         # BaseAgent abstract class
│   │   ├── planning.py     # 8 planning agent classes (BaseAgent subclasses)
│   │   └── chapter.py      # Chapter pipeline, prompt builders, ChapterContext
│   ├── session/
│   │   ├── __init__.py     # Re-exports
│   │   └── persistence.py  # Save/load/restore/validate session state
│   └── routes/
│       ├── __init__.py     # register_blueprints()
│       ├── outline.py      # /generate_outline, /approve_outline
│       ├── generation.py   # /generate_chapters, /progress, /revise_chapter
│       ├── export.py       # /export, /download, /illustrations
│       └── sessions.py     # /list_sessions, /load_session, /delete_session
├── templates/
│   └── index.html          # Single-page application HTML (Bootstrap 5)
├── static/
│   ├── css/style.css       # Custom styles (light + dark mode)
│   └── js/script.js        # jQuery client (AJAX, progress polling, session mgmt)
├── tests/
│   ├── __init__.py
│   └── test_app.py         # pytest test suite (89 tests)
├── sessions/
│   ├── novels/             # Novel session JSON files (crash recovery)
│   └── flask/              # Flask-Session server-side session files
├── exports/                # Generated Markdown files and illustrations
└── logs/                   # LLM request/response logs (llm.log)
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

   For running the app:

   ```bash
   pip install -r requirements.txt
   ```

   For local development (includes pytest, mypy, and type stubs):

   ```bash
   pip install -r requirements-dev.txt
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
| **LLM Settings** | | |
| `LLM_API_URL` | `https://api.openai.com/v1/chat/completions` | LLM API endpoint (OpenAI-compatible). |
| `LLM_API_KEY` | *(empty -- required)* | API key sent as `Authorization: Bearer <key>`. |
| `LLM_MODEL` | `gpt-4o` | Model name passed in the request payload. |
| `LLM_MAX_RETRIES` | `5` | Maximum retry attempts on transient errors (429, 5xx). |
| `LLM_RETRY_DELAY` | `5` | Base delay in seconds between retries (multiplied by attempt number). |
| `LLM_TIMEOUT` | `240` | Request timeout in seconds per LLM API call. |
| `LLM_CIRCUIT_BREAKER_THRESHOLD` | `3` | Consecutive failures before the circuit breaker trips. |
| **Image Settings** | | |
| `IMAGE_API_URL` | `https://api.openai.com/v1/images/generations` | Image generation API endpoint. |
| `IMAGE_API_KEY` | *(empty -- optional)* | API key for image generation. Required for illustrations. |
| `IMAGE_MODEL` | `gpt-image-1-mini` | Image generation model name. |
| `IMAGE_SIZE` | `1024x1024` | Generated image dimensions. |
| `IMAGE_TIMEOUT` | `120` | Request timeout in seconds per image API call. |
| **Generation Limits** | | |
| `PER_CHAPTER_TIMEOUT` | `3600` | Maximum wall-clock seconds per chapter (60 min default). |
| `MAX_CHAPTERS` | `100` | Upper bound for chapter count input validation. |
| `MAX_WORD_COUNT` | `500000` | Upper bound for word count input validation. |
| **Flask & Storage** | | |
| `SECRET_KEY` | `change-me-in-production` | Flask secret key. **Must be changed in production.** |
| `SESSION_FILE_DIR` | `./sessions/flask` | Directory for Flask-Session server-side files. |
| `EXPORT_DIR` | `./exports` | Directory for generated Markdown files and illustrations. |
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

> **Note:** `python app.py` uses Flask's built-in development server with debug mode **disabled** by default. Set `NOVELFORGE_DEBUG=true` in your `.env` file to enable debug mode for local development. For production, use a WSGI server such as **Gunicorn** behind a reverse proxy (e.g., Nginx):
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

A Bootstrap progress bar tracks writing progress with an estimated time remaining display. The progress label updates in real time to show the current agent step. The browser polls `/progress/<token>` with adaptive backoff (15s → 30s → 60s cap) that resets when progress changes. After 5 consecutive poll failures, a "Connection lost" warning is shown (polling continues in background).

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

- The novel title, chapter count, and approximate word count are shown.
- A collapsible **Writing Statistics** panel shows per-chapter word counts, generation times, LLM call counts, and token usage.
- Any editor's notes from the audit agents are displayed.
- An expandable **accordion** lets you preview each chapter's content inline.
- **Revise chapters** – Select a chapter and provide custom revision instructions. The chapter will be re-generated through the full agent pipeline.
- **Export Manuscript** – Four export variants:
  - **Clean Manuscript** – Title + chapter text only
  - **With Inline Notes** – Includes per-chapter editor annotations from audit data
  - **Publishing-Ready** – Front matter, TOC, page breaks, about-the-author section
  - **Critique Copy** – Includes pacing metrics and tension annotations per chapter
- **Download Editor's Notes** – Exports all diagnostic reports from the 9 post-generation audits.
- **Generate Illustrations** – Creates a cover image and a chapter scene illustration via the image generation API (requires `IMAGE_API_KEY`). Generated images are saved and persist across sessions.

**Session management** (navbar):
- **Sessions dropdown** – Load any previous session by title
- **New Session** – Archives current progress and starts fresh
- **Delete Session** – Permanently removes the current session
- **Dark/Light mode toggle** – Persists via localStorage

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page application HTML with session data. |
| `POST` | `/generate_outline` | Phase 1: generates title, chapter outline, characters, and runs planning agents in parallel. |
| `POST` | `/approve_outline` | Saves user-edited outline, detects character renames, selectively regenerates changed planning agents. |
| `POST` | `/generate_chapters` | Phase 2: starts background chapter generation thread. Returns `{token}` for progress polling. |
| `GET` | `/progress/<token>` | Returns JSON progress object with status, current chapter, and all audit reports. |
| `POST` | `/revise_chapter` | Revise a specific chapter with custom instructions through the full agent pipeline. |
| `POST` | `/export` | Compiles novel to Markdown (4 variants: clean, annotated, publishing, critique). Returns `{download_url}`. |
| `POST` | `/export_editors_notes` | Exports all diagnostic reports to Markdown. Returns `{download_url}`. |
| `POST` | `/generate_illustrations` | Generates cover + scene illustration via image API. Returns illustration metadata. |
| `GET` | `/illustrations/<filename>` | Serves a generated illustration image. |
| `GET` | `/download/<filename>` | Serves a generated export file as an attachment. |
| `GET` | `/list_sessions` | Returns all saved sessions with titles for the session dropdown. |
| `POST` | `/load_session` | Loads a specific session by ID, restoring all state. |
| `POST` | `/delete_session` | Deletes the current session's JSON file and clears session data. |
| `POST` | `/new_session` | Archives LLM log and starts a fresh session. |
| `GET` | `/llm_log` | Returns recent LLM log entries (debug mode only). |
| `POST` | `/clear_log` | Clears the LLM log file (debug mode only). |

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

All LLM calls are made by `call_llm()` in `novelforge/llm/client.py` using the `requests` library. The function:

- Adds `Authorization: Bearer <LLM_API_KEY>` and `Content-Type: application/json` headers.
- Sends the model name, message list, and optional `response_format: {type: "json_object"}` for structured outputs.
- Retries up to `LLM_MAX_RETRIES` times (default 5) with exponential back-off on HTTP 429 and 5xx errors.
- Uses a configurable timeout (`LLM_TIMEOUT`, default 240s) per request.
- Tracks token usage per-call (prompt tokens, completion tokens) for the writing statistics dashboard.
- **Circuit breaker**: After `LLM_CIRCUIT_BREAKER_THRESHOLD` consecutive failures (default 3), all subsequent calls fail immediately until the breaker resets (automatic on next generation start).
- Maps HTTP errors to user-friendly messages (401/403 → "API key rejected", 400 → "prompt rejected", 404 → "endpoint not found").
- Logs all requests and responses to `./logs/llm.log` as JSON.
- Background generation threads include a correlation token (`[token=<uuid>]`) in all log entries for tracing.

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

Planning agents are implemented as `BaseAgent` subclasses with a shared orchestration pattern (build prompt → call LLM → parse JSON → normalise → fallback on error). They run in parallel groups (see [Architecture](#architecture)) and support **selective regeneration**: on re-approval, only agents whose inputs changed are re-run, based on per-agent input hashing. Character renames are automatically propagated to chapter summaries, premise text, and planning agent inputs.

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

All values are configurable via environment variables (see [Configuration](#configuration)):

```bash
LLM_MAX_RETRIES=5                    # retry attempts per call
LLM_RETRY_DELAY=5                    # base delay (seconds), multiplied by attempt number
LLM_TIMEOUT=240                      # per-request timeout (seconds)
LLM_CIRCUIT_BREAKER_THRESHOLD=3      # consecutive failures before tripping
PER_CHAPTER_TIMEOUT=3600             # max wall-clock time per chapter (seconds)
```

On HTTP 429 or 5xx responses, the client waits with exponential back-off (`delay * attempt`). After `LLM_MAX_RETRIES` exhausted, the call raises a user-friendly error. After `LLM_CIRCUIT_BREAKER_THRESHOLD` consecutive failures across any calls, the circuit breaker trips and immediately fails all subsequent calls until the next generation run resets it.

Flask routes are also rate-limited via Flask-Limiter: 60 requests/minute globally, 5/minute for outline generation, 1/10 minutes for chapter generation.

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

Progress is automatically saved to `./sessions/novels/<session_id>.json` after outline approval, generation start, and each completed chapter. Completed chapters are persisted incrementally so no chapter is ever lost.

If the browser is closed or the server restarts during generation:

1. On page load, the **Sessions** dropdown in the navbar lists all saved sessions by title.
2. Clicking a session name loads it and restores all data (Step 1 form, Step 2 outline/characters, Step 4 completed chapters and exports).
3. If chapters were completed before the interruption, they appear on the Step 4 tab with full export capability.
4. The **New Session** button archives the current LLM log and starts fresh.
5. The **Delete Session** button permanently removes the current session file.

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
| `completed_chapters` | Background thread | List of `{number, title, content, summary, word_count, ...}` dicts |
| `illustrations` | `/generate_illustrations` | List of `{type, chapter, image_url, ...}` dicts |
| `_agent_input_hashes` | `/approve_outline` | Per-agent input hashes for selective regeneration |

Session files are written to `SESSION_FILE_DIR` (default: `./sessions/flask`). Novel persistence files are stored in `./sessions/novels/`.

---

## Security

| Concern | Mitigation |
|---|---|
| **XSS via user input** | User-supplied strings are stored as plain text in the session. XSS is prevented at render time: Jinja2 auto-escaping covers all server-rendered HTML, and jQuery's `.text()` method is used exclusively (never `.html()`) when inserting dynamic content into the DOM. |
| **Input validation** | Server-side validation (`validate_outline_input`) rejects empty premises, premises over 2,000 characters, unrecognised genre values, chapter counts below 3, and word counts below 1,000. Mirrored client-side with jQuery. |
| **Directory traversal** | The `/download/<filename>` route strips any path components using `Path(filename).name` before building the export path. |
| **Secret key** | Flask's `SECRET_KEY` must be set to a long random string via environment variable in production. The default value `change-me-in-production` is intentionally insecure and must not be used outside local development. |
| **API key exposure** | The LLM API key is read from an environment variable and never returned to the client. Keys are sanitized in log files. |
| **Session file permissions** | Novel persistence files (under `sessions/novels/`) are written atomically via a temp file; `os.chmod(0o600)` is applied explicitly before the rename so files are always owner-read/write only, regardless of the process umask. |
| **No `debug=True` in production** | Debug mode is controlled by the `NOVELFORGE_DEBUG` environment variable, which defaults to `false`. Set `NOVELFORGE_DEBUG=true` only for local development. The interactive debugger is never exposed unless explicitly enabled. |

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

## Troubleshooting

### Chapter generation stops mid-way

**Symptom:** The progress bar freezes and the status shows an error, or the browser shows "Connection lost."

**Causes & solutions:**

| Cause | What you see | Fix |
|---|---|---|
| **LLM rate limiting (429)** | "LLM API is overloaded" error | Wait a few minutes for the rate limit to clear, then click "Approve & Write Chapters" to restart. Completed chapters are preserved in the session file. |
| **Circuit breaker tripped** | "LLM API is unavailable — 3 consecutive calls failed" | Check your API key and endpoint are correct in `.env`. The circuit breaker resets automatically when you start a new generation. |
| **Per-chapter timeout** | "Chapter N exceeded the 60-minute time limit" | A single chapter took over 60 minutes (all agent steps combined). Increase the limit: `PER_CHAPTER_TIMEOUT=7200` in `.env` (2 hours). |
| **Server crashed / restarted** | Page shows blank after reload | Load the session from the **Sessions** dropdown in the navbar. Completed chapters are saved to `sessions/novels/*.json` after each chapter finishes. |
| **Network interruption** | "Connection lost — generation may still be running" | The generation thread continues server-side. Refresh the page and load the session from the dropdown. |

**To resume from a saved session:**
1. Click the **Sessions** dropdown in the navbar
2. Select the novel title
3. The page will reload with all completed chapters on the Step 4 tab

### LLM API errors and timeouts

**Symptom:** "The AI service is taking too long" or "API key rejected" errors.

**Tunable settings** (set in `.env` or as environment variables):

```bash
# Increase timeout for slow endpoints (default: 240 seconds)
LLM_TIMEOUT=360

# More retries for flaky connections (default: 5)
LLM_MAX_RETRIES=8

# Longer base delay between retries (default: 5 seconds)
LLM_RETRY_DELAY=10

# More tolerance before circuit breaker trips (default: 3)
LLM_CIRCUIT_BREAKER_THRESHOLD=5

# Longer per-chapter timeout (default: 3600 = 60 minutes)
PER_CHAPTER_TIMEOUT=7200
```

**Common API errors:**

| Error message | Cause | Fix |
|---|---|---|
| "API key rejected" | Invalid or expired `LLM_API_KEY` | Check your API key in `.env`. Verify it works with a `curl` test. |
| "API endpoint was not found" | Wrong `LLM_API_URL` | Verify the URL. For OpenAI: `https://api.openai.com/v1/chat/completions`. For Ollama: `http://localhost:11434/v1/chat/completions`. |
| "The AI service rejected the request" | Prompt too long or content policy violation | Try a shorter premise, or use a model with a larger context window. |
| "The AI service is overloaded" | All retry attempts got HTTP 429 | Wait 2-5 minutes and try again. Consider upgrading your API plan. |

### Session file corruption

**Symptom:** Page loads but data is missing or garbled, or you see "Failed to load session" errors.

**Quick fix — clear the corrupted session:**
1. Click **Delete Session** in the navbar (confirms before deleting)
2. Click **New Session** to start fresh

**Manual fix — inspect and repair:**
1. Find your session file in `sessions/novels/` (files are named `<uuid>.json`)
2. Open it in a text editor — it's standard JSON
3. The schema validator will auto-correct minor type mismatches on load (e.g., string where int expected)
4. If the file is truncated or unparseable, delete it and start a new session

**Nuclear option — clear everything:**
```bash
# Stop the server first
rm -rf sessions/novels/*.json
rm -rf sessions/flask/*
# Restart the server
python app.py
```

### Illustrations fail to generate

**Symptom:** "Illustration generation failed" alert after clicking the button.

| Cause | Fix |
|---|---|
| `IMAGE_API_KEY` not set | Add `IMAGE_API_KEY=sk-your-key` to `.env` |
| Rate limited during prompt generation | Wait a few minutes — the illustration LLM call retries up to 3 times with 30/60/90 second delays |
| Image API returned 400 for a prompt | Some art prompts trigger content policy filters. Try again — a different prompt will be generated. |
| Timeout | The 10-minute AJAX timeout may be too short if the API is slow. Check server logs for details. |

### Browser shows stale data after server restart

After restarting the Flask server, the in-memory progress store (`_progress_store`) is empty. However, completed chapters are persisted in the session JSON files.

**Fix:** Click the **Sessions** dropdown and re-select your novel. The session loader rebuilds the progress data from the saved `completed_chapters` array.

### CSRF token expired

**Symptom:** POST requests fail with a 400 error after the page has been open for a very long time.

The CSRF token has a 7-day lifetime. If the page has been open longer than that:
1. Simply refresh the page — a new token is issued automatically via cookie
2. All session data is preserved; no work is lost

### Logs and debugging

- **LLM request/response logs:** `logs/llm.log` — JSON-formatted records of every API call
- **Console output:** All log lines include `[token=<uuid>]` correlation IDs during chapter generation, making it easy to trace activity for a specific run
- **LLM log viewer:** The **Log** tab in the UI shows recent API calls in real-time (debug mode only)
- **Clear logs:** Click the trash icon in the Log tab, or delete `logs/llm.log` manually

---

## Performance Tuning

### Typical Generation Times

Generation time depends on chapter count, target word count, LLM model speed, and API rate limits. Each chapter makes ~16 LLM calls through the agent pipeline, plus 2-3 pre/post-chapter calls.

| Chapters | Approx. LLM Calls | Typical Time (GPT-4o) | Typical Time (GPT-3.5/fast model) |
|----------|-------------------|----------------------|----------------------------------|
| 3 | ~60 | 15-30 min | 5-10 min |
| 10 | ~200 | 1-3 hours | 30-60 min |
| 15 | ~300 | 2-5 hours | 1-2 hours |
| 24 | ~480 | 4-10 hours | 2-4 hours |
| 50 | ~1000 | 10-20 hours | 4-8 hours |

**Planning phase:** 8 agents run in 3 parallel groups (~2-4 minutes total). On re-approval with no changes, skipped entirely via input hashing.

**Post-manuscript audits:** 9 sequential LLM calls (~10-20 minutes).

**Note:** Times vary significantly based on API response speed, rate limiting, and model. Local models (Ollama, LM Studio) avoid rate limits but may be slower per call.

### Memory Requirements

| Component | Memory Usage |
|-----------|-------------|
| Flask application | ~50-80 MB base |
| Per active session | ~1-5 MB (outline, characters, planning data) |
| Per completed chapter (in memory) | ~50-200 KB (content + summary) |
| 24-chapter novel in progress store | ~5-10 MB |
| Session JSON file on disk | ~2-20 MB depending on chapter count |

For a typical single-user local deployment, 512 MB is more than sufficient. The progress store is in-memory only; completed chapters are also persisted to disk.

### Optimization Tips

#### Use a faster or cheaper LLM model

Set `LLM_MODEL` in `.env` to balance quality vs. speed:

```bash
# Fast and cheap — good for testing and iteration
LLM_MODEL=gpt-4o-mini

# Balanced — good quality, reasonable speed
LLM_MODEL=gpt-4o

# Maximum quality — slowest, most expensive
LLM_MODEL=gpt-4-turbo
```

Local models avoid rate limits entirely:
```bash
LLM_API_URL=http://localhost:11434/v1/chat/completions
LLM_MODEL=llama3.1:70b
```

#### Tune retry and timeout settings

For fast local models, reduce timeouts to fail faster on actual errors:
```bash
LLM_TIMEOUT=60          # Default: 240s — reduce for fast local models
LLM_RETRY_DELAY=2       # Default: 5s — reduce for local models (no rate limits)
LLM_MAX_RETRIES=3       # Default: 5 — fewer retries for local models
```

For slow cloud APIs with strict rate limits, increase patience:
```bash
LLM_TIMEOUT=360          # More time per request
LLM_RETRY_DELAY=15       # Longer waits between retries
LLM_MAX_RETRIES=8        # More retry attempts
LLM_CIRCUIT_BREAKER_THRESHOLD=5  # More tolerance before tripping
```

#### Reduce chapter count for faster iteration

When testing a premise or experimenting with style, use fewer chapters:
- **3 chapters** — minimum allowed; generates in ~15 minutes; good for testing the pipeline
- **10 chapters** — a short novella; generates in 1-3 hours
- **15-25 chapters** — standard novel length; generates in 3-10 hours

#### Planning agent caching

When editing the outline in Step 2, only agents whose inputs changed are re-run. To take advantage of this:
- Make small, incremental edits and re-approve rather than changing everything at once
- Editing only chapter summaries skips character-dependent agents
- Editing only characters skips chapter-only agents (story architecture, technology rules, theme reinforcement)

#### Reduce image generation load

Image generation is optional and capped at 2 images (1 cover + 1 scene). If rate limited:
- Wait 2-5 minutes between attempts
- Image generation uses a separate API key (`IMAGE_API_KEY`) — use a different key/plan if your main LLM key is rate limited

#### Monitor with logs

Use correlation IDs in the console output to identify slow chapters:
```
2026-03-25 14:32:01 INFO [novelforge.llm.client] [token=abc123] Chapter 5: drafting
2026-03-25 14:35:47 INFO [novelforge.llm.client] [token=abc123] Chapter 5: prose refinement
```

The gap between log entries shows per-step duration. If one step consistently takes too long, the bottleneck is likely the LLM model's response time for that prompt type.

---

## License

This project is released under the [MIT License](LICENSE).

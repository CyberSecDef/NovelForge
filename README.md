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
  - [Step 4 – Export](#step-4--export)
- [API Routes](#api-routes)
- [LLM Integration & Agents](#llm-integration--agents)
  - [Prompt Architecture](#prompt-architecture)
  - [Specialized Agents](#specialized-agents)
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
- **Specialized Writing Agents** – Twelve dedicated LLM passes per chapter: drafting, dialogue refinement, scene structuring, context analysis, editing, structure checking, character arc deepening, synthesis, polishing, anti-LLM pass, quality control, and chapter summarisation.
- **Anti-LLM Agent** – Dedicated LLM pass (step 10 per chapter) that removes robotic language patterns, overused phrases, and LLM hallmarks to produce human-sounding prose.
- **Continuity Tracking** – Each completed chapter generates a 100–200 word summary that is fed to subsequent chapters to maintain consistency.
- **Final Consistency Pass** – Global review agent checks all chapter summaries for plot holes, unresolved threads, character arc completion, and thematic payoff.
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
└── exports/                # Generated novel Markdown files (auto-created at runtime)
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
| **Story Premise** | Required. Describe your story idea. Maximum **1,200 characters** (enforced client- and server-side with a live counter). |
| **Genre** | Required. Select one of: Fantasy, Sci-Fi, Mystery, Romance, Horror, Thriller, Historical. |
| **Number of Chapters** | Required. Minimum **3**. Recommended 15–25 (tooltip shown). |
| **Target Word Count** | Required. Minimum 1,000. Recommended 80,000–90,000 (tooltip shown). |
| **Special Events** | Optional. Comma-separated or bulleted events to incorporate (e.g., *"A dragon attack in chapter 5, A wedding in the final act"*). |
| **Special Instructions** | Optional. Global writing guidance (e.g., *"Avoid clichés, emphasize diversity, dark tone"*). |

Click **Generate Outline**. The form is validated client-side before the AJAX POST to `/generate_outline`. A spinner indicates the LLM is working. The application then:

1. Generates a catchy title.
2. Generates a chapter-by-chapter outline following the nine-phase narrative architecture.
3. Generates 3–7 main characters with name, age, background, role, and arc.

### Step 2 – Review & Edit Outline

The generated outline is displayed in an editable table. You can:

- Edit the **novel title** in an input field.
- Edit any **chapter title** or **chapter summary** inline (cells are `contenteditable`).
- Edit any **character field** (name, age, role, background, arc) inline.

Click **Approve & Write Chapters** when satisfied. Edits are collected by jQuery and POSTed to `/approve_outline`, which saves the final outline to the session. Chapter writing then begins automatically.

### Step 3 – Chapter Generation

A Bootstrap progress bar tracks writing progress. The progress label updates in real time to show the current agent step. The browser polls `/progress/<token>` every 3 seconds. For each chapter, the backend runs a **twelve-step pipeline**:

1. **Draft** – Initial prose written with full context (premise, genre, title, outline, characters, all previous summaries, special instructions).
2. **Dialog Agent** – Refines all dialogue for natural rhythm, distinct character voices, and subtext.
3. **Scene Agent** – Ensures every scene follows the Goal → Obstacle → Outcome → New Problem pattern.
4. **Context Analyzer** – Checks world-building facts and timeline against previous chapter summaries.
5. **Editing Agent** – Fixes plot holes, pacing problems, and character inconsistencies.
6. **Structure Agent** – Verifies the chapter fulfils its designated role in the nine-phase story architecture.
7. **Character Agent** – Deepens character arcs and corrects any out-of-character moments.
8. **Synthesizer** – Unifies narrative voice and thematic thread after all specialist passes.
9. **Polish Agent** – Elevates grammar, style, and vivid language.
10. **Anti-LLM Agent** – Dedicated pass to strip robotic patterns and overused LLM words.
11. **Quality Controller** – Checks reader engagement, tension, pacing, and hook strength.
12. **Summarizer** – Produces a 100–200 word continuity summary for subsequent chapters.

After all chapters are written, a final **consistency pass** reviews all summaries for plot holes, unresolved threads, arc completion, and thematic payoff.

Completed chapters appear in the list as they finish, each marked with a green check.

### Step 4 – Export

When generation is complete:

- The novel title and approximate word count are shown.
- Any editor's notes from the consistency pass are displayed.
- An expandable **accordion** lets you preview each chapter's content inline.
- Click **Download as Markdown** to export the full novel. The Markdown file includes:
  - `# Title`
  - `## Chapter N: Title` headings
  - Italicised chapter summaries
  - Full chapter prose
  - An optional *Editor's Notes* section from the consistency pass

Click **Start Over** to reset the form and begin a new novel.

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page application HTML. |
| `POST` | `/generate_outline` | Phase 1: generates title, chapter outline, and characters. Requires JSON body (see below). Stores results in session. Returns `{title, chapters, characters}`. |
| `POST` | `/approve_outline` | Saves user-edited outline back to session. Requires JSON body `{title, chapters, characters}`. Returns `{status: "approved"}`. |
| `POST` | `/generate_chapters` | Phase 2: starts background chapter generation thread. Returns `{token}` for progress polling. Requires prior session data from `/approve_outline`. |
| `GET` | `/progress/<token>` | Returns JSON progress object: `{status, current, total, chapters_done, error, consistency}`. |
| `POST` | `/export` | Compiles novel to Markdown and saves to `EXPORT_DIR`. Requires JSON body `{token}`. Returns `{download_url}`. |
| `GET` | `/download/<filename>` | Serves a generated Markdown file as an attachment. Prevents directory traversal. |

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
  "error": null
}
```

Status values: `"running"` | `"done"` | `"error"`.
`step` contains a human-readable label of the current agent step (e.g. `"Chapter 7: polishing"`).

---

## LLM Integration & Agents

All LLM calls are made by the `call_llm()` function in `app.py` using the `requests` library. The function:

- Adds `Authorization: Bearer <LLM_API_KEY>` and `Content-Type: application/json` headers.
- Sends the model name, message list, and optional `response_format: {type: "json_object"}` for structured outputs.
- Retries up to **3 times** with exponential back-off on HTTP 429 (rate limit) and 5xx errors.
- Raises `RuntimeError` after exhausting retries, which the route handler converts to an HTTP 502 response.

### Prompt Architecture

Each LLM call uses a two-message structure:

```
system  → defines the agent's role and voice constraints
user    → provides the task with all relevant context
```

JSON mode (`response_format: json_object`) is used wherever structured data is expected (outline, characters, consistency pass) to guarantee parseable output. A fallback is applied if JSON decoding fails.

### Specialized Agents

Each chapter goes through a **twelve-step pipeline**, with every step using a dedicated LLM agent:

| Step | Agent | Purpose | Invocation |
|---|---|---|---|
| Phase 1 | **Title Agent** | Generates a catchy, original novel title. | Once. |
| Phase 1 | **Story Architect** | Creates a chapter-by-chapter outline following the nine-phase model. | Once. |
| Phase 1 | **Character Agent (outline)** | Develops 3–7 main characters with name, age, background, role, and arc. | Once. |
| Ch. step 1 | **Novelist (Draft Agent)** | Writes the initial chapter draft using full context. | Per chapter. |
| Ch. step 2 | **Dialog Agent** | Refines all dialogue for naturalism, distinct voices, and subtext. | Per chapter. |
| Ch. step 3 | **Scene Agent** | Enforces the Goal → Obstacle → Outcome → New Problem pattern in every scene. | Per chapter. |
| Ch. step 4 | **Context Analyzer** | Fixes world-building and timeline inconsistencies against prior chapter summaries. | Per chapter. |
| Ch. step 5 | **Editing Agent** | Repairs plot holes, pacing problems, and character inconsistencies. | Per chapter. |
| Ch. step 6 | **Structure Agent** | Confirms the chapter fulfils its role in the nine-phase story architecture. | Per chapter. |
| Ch. step 7 | **Character Agent (arc)** | Deepens character arcs and corrects out-of-character moments. | Per chapter. |
| Ch. step 8 | **Synthesizer Agent** | Unifies narrative voice and thematic thread after all specialist passes. | Per chapter. |
| Ch. step 9 | **Polish Agent** | Elevates grammar, style, and vivid language. | Per chapter. |
| Ch. step 10 | **Anti-LLM Agent** | Strips robotic patterns and forbidden overused words (dedicated pass). | Per chapter. |
| Ch. step 11 | **Quality Controller** | Checks reader engagement, tension, pacing, and hook strength. | Per chapter. |
| Ch. step 12 | **Summarizer Agent** | Produces a 100–200 word continuity summary for subsequent chapters. | Per chapter. |
| Post-gen | **Consistency Agent** | Reviews all summaries for plot holes, unresolved threads, arc completion, and thematic payoff. | Once, after all chapters. |

The browser progress bar shows the current agent step label in real time (polled every 3 seconds).

### Anti-LLM Agent

The **Anti-LLM Agent** is a dedicated LLM pass (step 10 per chapter) with a single responsibility: making AI-generated text sound genuinely human-written. It:

- Replaces or removes **overused LLM words**: `embark`, `delve`, `realm`, `tapestry`, `testament`, `nuance`, `beacon`, `uncharted`, `multifaceted`, `leverage`, `synergy`, `pivotal`, `groundbreaking`, `commendable`, `meticulous`.
- Eliminates **robotic transition phrases** ("In conclusion", "It is worth noting", "As a result of this").
- Removes **unnecessary hedging** and repetitive sentence openings.
- Introduces **subtle human imperfections**: varied sentence length, occasional fragments for emphasis, colloquial rhythms where fitting.
- Does **not** change plot, characters, or factual content.

### Rate Limiting & Retries

```python
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (multiplied by attempt number for exponential back-off)
```

On HTTP 429 or 5xx responses, the client waits `RETRY_DELAY × attempt` seconds before retrying. A `requests.exceptions.Timeout` (120-second limit per request) also triggers a retry. Persistent failures raise `RuntimeError` and return an HTTP 502 with an error message displayed as a Bootstrap alert.

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

**Scene-level pattern:** Goal → Obstacle → Outcome → New problem.

**Layered consistency targets:** Character arc, theme, cause-and-effect chain, escalation, and payoff.

---

## Session Management

NovelForge uses **Flask-Session** with a `cachelib` filesystem backend to store user workflow state server-side. Cookie-based sessions alone are insufficient for the volume of data (full chapter lists, character tables, etc.).

Session keys stored across the workflow:

| Key | Set by | Content |
|---|---|---|
| `premise` | `/generate_outline` | Sanitised story premise string |
| `genre` | `/generate_outline` | Selected genre string |
| `chapters` | `/generate_outline` | Integer chapter count |
| `word_count` | `/generate_outline` | Integer target word count |
| `special_events` | `/generate_outline` | Optional events string |
| `special_instructions` | `/generate_outline` | Optional instructions string |
| `title` | `/generate_outline`, `/approve_outline` | Novel title string |
| `chapter_list` | `/generate_outline`, `/approve_outline` | List of `{number, title, summary}` dicts |
| `character_list` | `/generate_outline`, `/approve_outline` | List of `{name, age, background, role, arc}` dicts |
| `progress_token` | `/generate_chapters` | UUID token for progress polling |

Session files are written to the directory specified by `SESSION_FILE_DIR` (default: `./flask_session`) with file permissions `0o600`.

---

## Security

| Concern | Mitigation |
|---|---|
| **XSS via user input** | All user-supplied strings are escaped with `markupsafe.escape` before being stored in the session and rendered. jQuery's `.text()` method is used exclusively (never `.html()`) when inserting dynamic content into the DOM. |
| **Input validation** | Server-side validation (`validate_outline_input`) rejects empty premises, premises over 1,200 characters, unrecognised genre values, chapter counts below 3, and word counts below 1,000. Mirrored client-side with jQuery. |
| **Directory traversal** | The `/download/<filename>` route strips any path components using `Path(filename).name` before building the export path. |
| **Secret key** | Flask's `SECRET_KEY` must be set to a long random string via environment variable in production. The default value `change-me-in-production` is intentionally insecure and must not be used outside local development. |
| **API key exposure** | The LLM API key is read from an environment variable and never returned to the client or logged. |
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
- Premise of exactly 1,200 characters is accepted.
- Premise of 1,201 characters is rejected.
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

# NovelForge TODO

This document tracks planned improvements and enhancements for NovelForge.

---

## Priority Legend

- **[HIGH]** - Critical for security, reliability, or core functionality
- **[MEDIUM]** - Important for user experience or maintainability
- **[LOW]** - Nice to have; polish and optimization

---

## 1. Security

### High Priority

- [ ] **Add CSRF Protection** - Implement Flask-WTF CSRF tokens on all POST routes. Currently, any malicious website could trick users into triggering novel generation, consuming expensive LLM API credits.
  - Install `Flask-WTF`, call `CSRFProtect(app)`, add `{{ csrf_token() }}` to all AJAX headers in `script.js`
  - Location: All POST routes in `app.py`, all AJAX calls in `static/js/script.js`

- [ ] **Implement Rate Limiting** - Add Flask-Limiter with per-IP rate limits (e.g., 5 outline generations per minute, 1 novel generation per 10 minutes) to prevent API abuse and DoS attacks.
  - Install `Flask-Limiter`, apply `@limiter.limit()` decorators to expensive routes (`/generate_outline`, `/generate_chapters`, `/revise_chapter`)
  - Location: `app.py` (route decorators)

- [ ] **Add Upper Bounds to Input Validation** - Enforce maximum values for chapters (cap at 100) and word count (cap at 500,000) to prevent memory exhaustion, runaway thread timeouts, and excessive API spend.
  - Also add length limits to `special_events` and `special_instructions` fields (e.g., 5,000 chars each)
  - Location: `app.py:validate_outline_input()`

### Medium Priority

- [ ] **Add Security Test Suite** - Implement comprehensive XSS, CSRF, injection, and path traversal tests to lock in security guarantees across refactors.
  - Test that `markupsafe.escape()` is applied before LLM calls, that `/download/<filename>` rejects `../` traversal, that CSRF tokens are enforced
  - Location: `tests/test_security.py` (new file)

- [ ] **Sanitize Contenteditable Fields** - Add character limits and server-side validation to inline-editable table cells (chapter titles, character names/backgrounds) to prevent malformed data reaching LLM prompts.
  - Add `maxlength` enforcement via JS `input` event; validate lengths again in `/approve_outline`
  - Location: `static/js/script.js` (contenteditable handlers), `app.py:/approve_outline`

---

## 2. Reliability & Error Handling

### High Priority

- [ ] **Add Configuration Validation at Startup** - Validate all required environment variables (`LLM_API_KEY`, `LLM_API_URL`) before the server begins accepting requests. Print a clear error and exit if required values are missing or obviously wrong.
  - Implement a `validate_config()` function called from `create_app()` (see Code Organization); use environment-based config classes (`DevelopmentConfig`, `ProductionConfig`) so `SECRET_KEY` is required in production but defaults gracefully in development
  - Location: `config.py`, `app.py` (startup)

- [ ] **Replace In-Memory Progress Store** - Migrate `_progress_store` dict to Redis or SQLite. The current design silently loses all progress when running gunicorn with `-w > 1` workers because each worker has its own memory space. Redis is preferred for production; SQLite is acceptable for single-server deployments.
  - Use `flask-caching` with a Redis backend or create a `progress.py` module backed by SQLite with a `threading.Lock`
  - Location: `app.py` (global `_progress_store`, `_progress_lock`, and all callers)

### Medium Priority

- [ ] **Implement Circuit Breaker for LLM API** - After 3 consecutive LLM call failures, abort chapter generation and surface a clear error to the user rather than continuing to send requests to a failing API endpoint.
  - Track consecutive failure count in the progress store; abort with a specific error code the frontend can display with a "Retry" option
  - Location: `app.py:call_llm()`, `app.py:_run_chapter_generation_internal()`

- [ ] **Add Per-Chapter Timeout** - Enforce a maximum wall-clock time (e.g., 30 minutes) for each chapter generation pass through the 16-step agent pipeline to prevent runaway background threads from consuming resources indefinitely.
  - Use `threading.Timer` or track `time.monotonic()` at the start of each chapter and abort if elapsed exceeds limit
  - Location: `app.py:_run_all_chapter_agents()`

- [ ] **Add Session File Cleanup** - Session persistence files in `./sessions/` accumulate indefinitely and are never purged. Add a background cleanup job or a startup sweep to remove sessions older than a configurable TTL (default: 7 days).
  - Scan `./sessions/` on startup and schedule periodic cleanup; expose TTL as a `SESSION_TTL_DAYS` env var in `config.py`
  - Location: `app.py` (startup), `config.py`

- [ ] **Improve Error Logging for Planning Agents** - Log which specific planning agent failed with full context (chapter count, genre, premise snippet) before returning an error response. Currently failures are hard to trace post-mortem.
  - Add structured log entries at each agent call site with agent name, input hash, and error details
  - Location: `app.py` (planning agent call sites, approximately lines 4352–4487)

- [ ] **Add Polling Failure Handling** - After 5 consecutive `/progress/<token>` polling failures in JavaScript, show a visible warning to the user ("Connection lost — generation may still be running in the background") instead of silently continuing.
  - Track consecutive failure count in `pollProgress()`; clear count on success
  - Location: `static/js/script.js:pollProgress()`

- [ ] **Add Exponential Backoff to Progress Polling** - Replace fixed 3-second polling interval with adaptive backoff (e.g., 2s → 4s → 8s → cap at 15s) based on how often the response changes, reducing server load during long generations.
  - Location: `static/js/script.js:pollProgress()`

### Low Priority

- [ ] **Add Schema Validation for Session Persistence** - Use Pydantic or `dataclasses` with field validators to ensure all session keys are correctly typed when saving and restoring session state. Prevents silent data corruption from partial writes or format changes.
  - Location: `app.py` (session persistence functions, approximately lines 77–198)

---

## 3. Architecture & Code Organization

### High Priority

- [ ] **Refactor `app.py` into a Proper Package** - The 5,700-line `app.py` is the single largest maintainability risk. Split it into a `novelforge/` package with the following structure. This is a prerequisite for safely implementing most other items on this list.

  ```
  novelforge/
  ├── __init__.py           # App factory: create_app()
  ├── config.py             # DevelopmentConfig, ProductionConfig, TestingConfig
  ├── routes/
  │   ├── __init__.py       # register_blueprints()
  │   ├── outline.py        # /generate_outline, /approve_outline
  │   ├── generation.py     # /generate_chapters, /progress/<token>, /revise_chapter
  │   └── export.py         # /export, /download, /export_editors_notes
  ├── agents/
  │   ├── __init__.py
  │   ├── base.py           # BaseAgent with shared build/normalize/plan/fallback pattern
  │   ├── planning/         # 7 planning agents (story_arch, timeline, fate_registry, etc.)
  │   └── chapter/          # 16+ per-chapter agents (draft, dialog, scene, polish, etc.)
  ├── llm/
  │   ├── __init__.py
  │   ├── client.py         # call_llm(), retry logic, LLM logging
  │   └── prompts.py        # load_prompt_by_name(), render_prompt()
  ├── session/
  │   ├── __init__.py
  │   └── persistence.py    # save/load/restore/clear session state
  ├── validation.py         # validate_outline_input(), ALLOWED_GENRES, constants
  └── progress.py           # Progress store abstraction (in-memory or Redis/SQLite)
  ```

  - Keep `app.py` as a thin entry point: `from novelforge import create_app; app = create_app()`
  - Location: `app.py` (entire file)

- [ ] **Implement App Factory Pattern** - Replace the current module-level `app = Flask(__name__)` with a `create_app(config=None)` factory function. This is required for proper testing (each test can spin up a fresh app instance with `TestingConfig`), and for supporting multiple deployment environments without code changes.

  ```python
  def create_app(config=None) -> Flask:
      app = Flask(__name__)
      app.config.from_object(config or ProductionConfig)
      validate_config(app)
      register_blueprints(app)
      init_extensions(app)
      return app
  ```

  - Location: `app.py` (new `novelforge/__init__.py`)

### Medium Priority

- [ ] **Create `BaseAgent` Class** - All 7 planning agents share an identical `build_prompt / normalize / build_fallback / plan / get_context` pattern, duplicated ~7 times with minor variations. Extract this into a `BaseAgent` abstract class, reducing boilerplate and making it trivial to add new agents.

  ```python
  class BaseAgent:
      name: str
      def build_prompt(self, **ctx) -> list[dict]: ...
      def normalize(self, data: dict) -> dict: ...
      def build_fallback(self, **ctx) -> dict: ...
      def plan(self, **ctx) -> dict:
          # shared: render_prompt → call_llm → parse_json → normalize → fallback on error
  ```

  - Location: `app.py` (planning agent functions, approximately lines 433–2304); new `novelforge/agents/base.py`

- [ ] **Parallelize Independent Planning Agents** - The 7 planning agents currently run sequentially during `/approve_outline`, taking 2–5 minutes. Several agents are independent of each other and can run concurrently. Use `concurrent.futures.ThreadPoolExecutor` to run independent groups in parallel and cut planning time roughly in half.
  - Group 1 (independent, run in parallel): Story Architecture, Master Timeline, Technology Rules, Theme Reinforcement
  - Group 2 (depend on Group 1 outputs): Character Fate Registry, Character Arc Planner, Antagonist Motivation
  - Location: `app.py:approve_outline()` (approximately lines 4519–4573)

- [ ] **Move Magic Numbers to Config** - Move all hardcoded numeric constants to `config.py` so they can be tuned without touching business logic. Affected values include:
  - `MAX_RETRIES = 5` (LLM retry attempts)
  - `RETRY_DELAY = 5` (base delay seconds)
  - `LLM_TIMEOUT = 240` (request timeout seconds)
  - `POLL_INTERVAL_MS = 3000` (client polling interval)
  - `MAX_CHAPTERS = 100` (upper bound for chapters)
  - `MAX_WORD_COUNT = 500000` (upper bound for word count)
  - Location: `app.py` (approximately lines 210–211, 251), `static/js/script.js`

- [ ] **Add Comprehensive Type Hints** - Add type hints to all functions using Python 3.11+ syntax (`list[dict]`, `str | None`, etc.). Enables IDE autocompletion, catches type errors with mypy, and serves as inline documentation.
  - Prioritize public interfaces: route handlers, `call_llm()`, agent `plan()` methods, session persistence functions
  - Location: `app.py` (throughout)

### Low Priority

- [ ] **Centralize Chapter Split Logic** - The percentage-based chapter position calculation (used to assign narrative phases to chapters) is duplicated in multiple locations. Create a single `ChapterPosition` utility with methods like `get_act()`, `get_phase()`, `is_climax_zone()`.
  - Location: Multiple sites in `app.py`

- [ ] **Add Logging Correlation IDs** - Background chapter generation threads produce log entries that are not linked to the originating request. Add a correlation ID (e.g., the progress token) to all log entries emitted during generation for easier tracing.
  - Use Python's `logging.LoggerAdapter` to inject `token=<progress_token>` into all log records from background threads
  - Location: `app.py` (logging setup, background thread entry points)

---

## 4. Performance & Scalability

### Medium Priority

- [ ] **Implement Selective Agent Regeneration** - When the user edits only one chapter title in Step 2, all 7 planning agents are re-run unnecessarily during `/approve_outline`. Hash the relevant input fields per agent and skip regeneration if inputs are unchanged.
  - Location: `app.py:approve_outline()` (approximately lines 4519–4573)

- [ ] **Add Caching for Planning Agent Outputs** - Extend selective regeneration with a session-level cache: store agent outputs alongside an input hash. On re-approval, compare hashes and reuse cached outputs for unchanged agents.
  - Location: `app.py:approve_outline()`

- [ ] **Replace Progress Polling with Server-Sent Events** - The current 3-second HTTP polling (`/progress/<token>`) generates constant server load for the full duration of novel generation (20–40 minutes). Replace with SSE (`/progress/stream/<token>`) to push updates only when state actually changes, eliminating unnecessary requests.
  - Use Flask's `Response` with `mimetype="text/event-stream"` and a generator that yields from the progress store; update the client to use the `EventSource` API
  - Location: `app.py` (new `/progress/stream/<token>` route), `static/js/script.js:pollProgress()`

### Low Priority

- [ ] **Implement Virtual Scrolling for Chapter Lists** - For novels with 50+ chapters, the DOM accumulates all chapter entries simultaneously, causing UI slowdown. Paginate or virtualize the chapter list to render only visible rows.
  - Location: `static/js/script.js`, `templates/index.html`

- [ ] **Compress Session Files** - Session persistence files in `./sessions/` can grow large for complex novels (many characters, long summaries). Apply gzip compression on write and decompress on read.
  - Location: `app.py:save_session_state()` (approximately line 131)

- [ ] **Return Incremental Progress Updates** - `/progress/<token>` currently returns the full chapter list on every poll. Return only chapters added or updated since the last poll (using a `since` query parameter) to reduce payload size for long novels.
  - Location: `app.py:/progress/<token>` route

---

## 5. Testing

### High Priority

- [ ] **Add Mock LLM Fixture and Integration Tests** - Create a reusable `mock_llm` pytest fixture that intercepts `call_llm()` calls and returns canned responses. Use this to test the full request/response cycle for all routes without live API calls.

  ```python
  @pytest.fixture
  def mock_llm(mocker):
      return mocker.patch("novelforge.llm.client.call_llm", return_value='{"title": "Test Novel"}')
  ```

  - Location: `tests/conftest.py` (new), `tests/test_routes.py` (new)

- [ ] **Add App Factory Tests** - Once `create_app()` is implemented, each test module should create a fresh app instance with `TestingConfig` to ensure test isolation and prevent session/state bleed between test runs.
  - Location: `tests/conftest.py`

### Medium Priority

- [ ] **Add Route Coverage Tests** - Add integration tests for all routes currently missing coverage: `/revise_chapter`, `/llm_log`, `/export_editors_notes`, `/check_saved_state`, `/resume_session`, `/new_session`, `/download/<filename>`.
  - Location: `tests/test_routes.py` (new file)

- [ ] **Add Concurrency and Threading Tests** - Test progress tracking correctness under concurrent access, thread safety of `_progress_store`, and correct behavior when two chapter generation requests arrive simultaneously.
  - Location: `tests/test_concurrency.py` (new file)

- [ ] **Add Boundary Condition Tests** - Test edge cases: exactly 1,000 word count, exactly 3 chapters, premises with Unicode/emoji, special characters in chapter titles, empty character list.
  - Location: `tests/test_validation.py`

- [ ] **Add Session Persistence Tests** - Test the full save → crash-simulate → load → restore cycle for session state, including partial generation state (e.g., 3 of 10 chapters complete).
  - Location: `tests/test_session.py` (new file)

---

## 6. User Experience

### Medium Priority

- [ ] **Add Accessibility Labels** - Add ARIA attributes to step panels (`aria-hidden`, `role="main"`), `for` attributes on all form labels, and keyboard navigation support for the inline-editable chapter/character tables.
  - Location: `templates/index.html`

- [ ] **Implement Mobile Responsive Design** - Replace fixed-width tables in the chapter and character review panels with card-based layouts on small viewports using Bootstrap 5 responsive utilities.
  - Location: `templates/index.html`, `static/css/style.css`

- [ ] **Add User-Friendly Error Messages** - Replace generic HTTP error responses with specific, actionable messages. Examples:
  - LLM timeout → "The AI service is taking too long. Your progress is saved — you can resume when it recovers."
  - LLM auth failure → "API key rejected. Check your LLM_API_KEY setting."
  - Location: `app.py` (all route error handlers), `static/js/script.js` (error display)

### Low Priority

- [ ] **Add Unsaved Changes Warning** - Track dirty state when the user edits the chapter or character tables and show a confirmation modal if they attempt to navigate away without approving.
  - Location: `static/js/script.js`

- [ ] **Add Progress Time Estimation** - Record wall-clock time per completed chapter and display an estimated time remaining ("~12 minutes left") based on the rolling average.
  - Location: `static/js/script.js:pollProgress()`

- [ ] **Implement Clear Log Button** - The "Clear log" button in the LLM log viewer panel exists in the UI but has no functionality. Wire it to clear the displayed entries and optionally POST to a `/clear_log` endpoint.
  - Location: `static/js/script.js`

- [ ] **Disable Export Button During Processing** - Prevent duplicate file generation from rapid clicks on the Export button by disabling it immediately on click and re-enabling only after the download completes or an error occurs.
  - Location: `static/js/script.js` (export button handler)

- [ ] **Add Dark Mode Support** - Implement `@media (prefers-color-scheme: dark)` CSS rules for the Bootstrap theme and custom styles.
  - Location: `static/css/style.css`

- [ ] **Add Print Styles** - Optimize the chapter preview panel for printing with `@media print` CSS rules (hide nav, expand collapsed chapters, set readable font sizes).
  - Location: `static/css/style.css`

---

## 7. Infrastructure & Deployment

### Medium Priority

- [ ] **Add Health Check Endpoint** - Create a `/health` route returning `{"status": "ok", "version": "..."}` for load balancer monitoring and uptime checks. Should return `503` if any critical dependency (e.g., Redis) is unreachable.
  - Location: `app.py` (new route)

- [ ] **Add Environment-Based Config Classes** - Replace the flat `config.py` with a class hierarchy that enforces environment-appropriate settings:
  - `DevelopmentConfig`: debug on, insecure `SECRET_KEY` default allowed, verbose logging
  - `ProductionConfig`: debug off, `SECRET_KEY` required (raises on missing), warnings for default LLM model
  - `TestingConfig`: uses mock LLM URL, in-memory session, no disk I/O
  - Location: `config.py`

- [ ] **Add Database Support for Persistence** - Migrate from file-based sessions and in-memory progress store to SQLite (single-server) or PostgreSQL (multi-server). Store novel metadata, chapter content, and generation progress in normalized tables for better querying and audit trails.
  - Use Flask-SQLAlchemy; implement with Alembic/Flask-Migrate for schema versioning
  - Location: New `novelforge/models.py` and `novelforge/db.py` modules

- [ ] **Add Structured Logging** - Replace ad-hoc `logging.info()` calls with JSON-structured log output (`python-json-logger`) for better aggregation in production log systems (Datadog, CloudWatch, ELK).
  - Include fields: `timestamp`, `level`, `module`, `correlation_id`, `action`, `duration_ms`
  - Location: `app.py` (logging configuration)

### Low Priority

- [ ] **Add Docker Support** - Create a `Dockerfile` and `docker-compose.yml` (with a Redis service for the progress store) to simplify local development setup and production deployment.
  - Location: Repository root (new files)

---

## 8. Documentation

### Low Priority

- [ ] **Add Architecture Diagram** - Create a visual overview of system components: browser → Flask routes → background thread → LLM API → session storage. Include the 16-step per-chapter pipeline as a flowchart.
  - Location: `README.md` or `docs/architecture.md`

- [ ] **Add Troubleshooting Guide** - Document common failure modes and solutions:
  - Chapter generation stops mid-way → how to resume
  - LLM timeouts → how to tune `LLM_TIMEOUT`
  - Session file corruption → how to clear and restart
  - Location: `README.md`

- [ ] **Add Performance Tuning Guide** - Document typical generation times by chapter count, memory requirements, and tips for optimizing (e.g., lighter LLM model, parallel agents).
  - Location: `README.md`

- [ ] **Generate OpenAPI Specification** - Create a `openapi.yml` documenting all endpoints with request/response schemas and example payloads. Use `flask-smorest` or `apispec` to auto-generate from route docstrings.
  - Location: `docs/openapi.yml` or auto-generated at `/api/docs`

---

## 9. Features (Future Enhancements)

### Medium Priority

- [ ] **Add User Accounts** - Implement optional user registration/login (Flask-Login + SQLAlchemy) to persist novels and generation history across browser sessions.

- [ ] **Add Novel Templates** - Allow users to start from pre-defined story archetypes (e.g., "Hero's Journey", "Murder Mystery", "Romance Arc") that pre-populate premise, genre, and chapter structure suggestions.

- [ ] **Add Export Formats** - Support EPUB, PDF, and DOCX export in addition to Markdown. Use `ebooklib` for EPUB, `WeasyPrint` or `reportlab` for PDF, `python-docx` for DOCX.

### Low Priority

- [ ] **Add Collaboration Features** - Allow multiple users to co-edit the same outline in real-time using WebSockets (Flask-SocketIO).

- [ ] **Add Version History** - Track every revision of each chapter and allow rollback to any prior version. Store diffs or full snapshots in the database.

- [ ] **Add Chapter Drag-and-Drop Reordering** - Allow users to reorder chapters in the Step 2 outline table via drag-and-drop (using SortableJS) rather than only up/down arrow buttons.

- [ ] **Add Character Relationship Mapping** - Generate and display a visual graph (using D3.js or Mermaid) showing character relationships as defined by the character agent.

- [ ] **Add Writing Statistics Dashboard** - Show per-chapter word count, generation time, revision count, and LLM token usage in a summary panel after generation completes.

---

## Summary

| Category | High | Medium | Low | Total |
|----------|------|--------|-----|-------|
| Security | 3 | 2 | 0 | 5 |
| Reliability | 2 | 6 | 1 | 9 |
| Architecture & Code Organization | 2 | 4 | 2 | 8 |
| Performance | 0 | 3 | 3 | 6 |
| Testing | 2 | 4 | 0 | 6 |
| User Experience | 0 | 3 | 6 | 9 |
| Infrastructure | 0 | 4 | 1 | 5 |
| Documentation | 0 | 0 | 4 | 4 |
| Future Features | 0 | 3 | 5 | 8 |
| **Total** | **9** | **29** | **22** | **60** |

---

## Recommended Implementation Order

1. **[Security]** CSRF protection + rate limiting — prevent API abuse before anything else
2. **[Security/Reliability]** Input upper bounds + config validation at startup
3. **[Reliability]** Replace in-memory progress store (Redis/SQLite) — required before multi-worker deployment
4. **[Architecture]** App factory pattern + package modularization — unblocks all subsequent refactors
5. **[Architecture]** `BaseAgent` class — eliminates the most duplicated code
6. **[Testing]** Mock LLM fixture + app factory tests — lock in behavior before further refactoring
7. **[Testing]** Route coverage + session persistence tests
8. **[Architecture]** Parallelize independent planning agents — significant UX win (halves approval wait time)
9. **[Performance]** Server-Sent Events (replace polling)
10. **[Reliability]** Session file cleanup + circuit breaker
11. **[UX + Docs]** Polish, accessibility, documentation

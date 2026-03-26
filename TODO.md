# NovelForge TODO

Comprehensive review of the codebase after the major refactoring session. Items are organized by priority and category.

---

## Priority Legend

- **[HIGH]** - Security, data integrity, or reliability risks
- **[MEDIUM]** - Quality, maintainability, or user experience improvements
- **[LOW]** - Polish, optimization, and nice-to-haves

---

## Completed in This Session

The following items were implemented during the current development session and are now complete:

- [X] CSRF token lifetime set to 7 days with cookie-based storage
- [X] Session dropdown replacing resume modal for session management
- [X] Delete session button with confirmation
- [X] Session data population on page load (all steps)
- [X] Completed chapters stored in session JSON (`completed_chapters` array)
- [X] Chapter text line breaks preserved in preview accordion
- [X] Chapter summaries generated for test session data
- [X] LLM log polling interval changed to 15 seconds
- [X] Circuit breaker for LLM API (3 consecutive failures)
- [X] Per-chapter timeout (60 minutes configurable)
- [X] Structured error logging for planning agents
- [X] Polling failure handling (5 consecutive failures shows warning)
- [X] Adaptive polling backoff (15s → 30s → 60s cap)
- [X] `print()` and `console.log()` replaced with proper logging
- [X] Session state schema validation with type coercion
- [X] ARIA attributes and accessibility labels
- [X] User-friendly LLM error messages with specific guidance
- [X] Unsaved changes warning for outline edits
- [X] Progress time estimation display
- [X] Clear log button wired to server-side `/clear_log` endpoint
- [X] Export buttons disabled during processing (all variants)
- [X] Dark mode with toggle button and localStorage persistence
- [X] Writing statistics dashboard (per-chapter word count, time, tokens)
- [X] Refactored `app.py` into `novelforge/` package (18 modules)
- [X] Session files moved to `sessions/novels/`, Flask sessions to `sessions/flask/`
- [X] `BaseAgent` abstract class for planning agents (8 subclasses)
- [X] Magic numbers moved to `config.py` with env var overrides
- [X] Comprehensive type hints across all modules
- [X] Export format functions consolidated with shared `_build_manuscript()` builder
- [X] `ChapterPosition` utility centralizing chapter split logic
- [X] `ChapterContext` dataclass reducing `_run_all_chapter_agents` from 19 to 12 params
- [X] Planning agents parallelized (3 groups via ThreadPoolExecutor)
- [X] Logging correlation IDs for background generation threads
- [X] Selective agent regeneration with per-agent input hashing
- [X] Character rename propagation to chapter summaries and premise
- [X] Architecture diagrams added to README (system overview, pipeline, dependency graph)
- [X] Troubleshooting guide in README
- [X] Performance tuning guide in README
- [X] README updated with current routes, config, and session management
- [X] Mock LLM fixture (`conftest.py`) with smart canned responses
- [X] Integration tests for all routes (22 tests)
- [X] Concurrency and threading tests (10 tests)
- [X] Boundary condition tests (36 tests)
- [X] Session persistence tests (9 tests)
- [X] Illustration path storage in session for reload persistence
- [X] Resilient illustration LLM prompt with extra retries
- [X] Image generation capped at 2 images (1 cover + 1 scene)

---

## 1. Security

### High Priority

- [ ] **Sanitize Session ID in `/load_session`** - The `target_id` from user input is used directly to construct a file path (`Path("./sessions/novels") / f"{target_id}.json"`). While `Path` objects are safer than string concatenation, a crafted `target_id` containing `../` could attempt directory traversal. Validate that `target_id` matches UUID format before constructing the path.
  - Location: `novelforge/routes/sessions.py:47-51`

- [ ] **Atomic Session File Writes** - `save_session_state()` and `_persist_completed_chapters()` use `write_text()` directly. If the process crashes mid-write, the JSON file is corrupted. Use a write-to-temp-then-rename pattern to ensure atomicity.
  - Location: `novelforge/session/persistence.py:184, 257`

- [ ] **Set File Permissions on Session Files** - Session JSON files are written with default permissions. On shared systems, other users could read novel content and session data. Write with mode `0o600` (owner read/write only).
  - Location: `novelforge/session/persistence.py:184, 257`

### Medium Priority

- [ ] **Add Security Test Suite** - Implement comprehensive XSS, CSRF, injection, and path traversal tests to lock in security guarantees across refactors.
  - Test that `markupsafe.escape()` is applied before storage, that `/download/<filename>` rejects `../` traversal, that CSRF tokens are enforced, that session IDs are UUID-validated
  - Location: `tests/test_security.py` (new file)

- [ ] **Sanitize Jinja2 Template Rendering in Prompts** - `render_prompt()` uses `Template(prompt["system"]).render(**context)` which could execute Jinja2 expressions if user-controlled context contains `{{ }}` syntax. While context values are escaped upstream, consider using `SandboxedEnvironment` or `autoescape=True`.
  - Location: `novelforge/llm/prompts.py:60-71`

---

## 2. Reliability & Error Handling

### High Priority

- [ ] **Replace In-Memory Progress Store** - `_progress_store` is a plain dict lost on server restart. Completed chapters are saved to session JSON files, but all progress metadata (status, step, audit reports) is lost. For multi-worker deployments (gunicorn -w > 1), each worker has its own dict. Migrate to Redis or SQLite.
  - Location: `novelforge/progress.py`

### Medium Priority

- [ ] **Add Exponential Backoff Cap to LLM Retries** - Current retry delay is linear (`RETRY_DELAY * attempt` = 5s, 10s, 15s, 20s, 25s). For 5 retries this means up to 75s of waiting. Switch to exponential with a cap: `min(30, RETRY_DELAY * 2^(attempt-1))`.
  - Location: `novelforge/llm/client.py:148`

- [ ] **Validate Image API Response Structure** - `call_image_api()` extracts `data.get("data", [{}])[0]` without validating the response shape. A malformed API response could crash with `IndexError` or `TypeError`.
  - Location: `novelforge/llm/image.py:77-90`

- [ ] **Add Logging for JSON Parse Fallback Attempts** - `parse_llm_json()` silently falls back to heuristic extraction when JSON parsing fails. Log the raw response and which extraction strategy was used to aid debugging.
  - Location: `novelforge/llm/client.py:256-312`

- [ ] **Add Session File Cleanup** - Session persistence files in `./sessions/novels/` accumulate indefinitely. Add a startup sweep to remove sessions older than a configurable TTL (default: 30 days). Expose as `SESSION_TTL_DAYS` env var.
  - Location: `novelforge/__init__.py` (startup), `novelforge/config.py`

- [ ] **Log Rotating File Handler** - `llm.log` uses a plain `FileHandler` with no rotation. On busy servers this file grows indefinitely. Switch to `RotatingFileHandler` with max 10MB and 5 backups.
  - Location: `novelforge/llm/client.py:17`

---

## 3. Code Quality

### Medium Priority

- [ ] **Remove Dead Code: `load_prompt_by_name()`** - This legacy function in `novelforge/llm/prompts.py` is never called anywhere. `render_prompt()` is the canonical function. Remove it.
  - Location: `novelforge/llm/prompts.py:13-39`

- [ ] **Validate Chapter Numbers in `/approve_outline`** - Chapter numbers from the contenteditable table are not validated as sequential 1..N. Non-sequential or duplicate numbers could break agent context lookups. Re-number server-side after receiving.
  - Location: `novelforge/routes/outline.py:196-209`

- [ ] **Update CLAUDE.md** - The CLAUDE.md file still references the old monolithic `app.py` structure, old session paths (`./flask_session`, `./sessions`), and old line numbers. Update to reflect the `novelforge/` package structure.
  - Location: `CLAUDE.md`

- [ ] **Update `.env.example`** - Missing all new tunable constants: `LLM_MAX_RETRIES`, `LLM_RETRY_DELAY`, `LLM_TIMEOUT`, `LLM_CIRCUIT_BREAKER_THRESHOLD`, `PER_CHAPTER_TIMEOUT`, `MAX_CHAPTERS`, `MAX_WORD_COUNT`, `IMAGE_TIMEOUT`, `SESSION_FILE_DIR` (new default path).
  - Location: `.env.example`

- [ ] **Fix Flask-Session Deprecation Warnings** - 94 test warnings about `SESSION_FILE_DIR` being deprecated. Flask-Session wants a `CacheLib` instance passed as `SESSION_CACHELIB` instead of the string config.
  - Location: `novelforge/__init__.py:47-49`

### Low Priority

- [ ] **Add `py.typed` Marker** - Add an empty `py.typed` file to `novelforge/` to enable downstream type checking with mypy.
  - Location: `novelforge/py.typed` (new file)

- [ ] **Prompt Cache Refresh Mechanism** - `_load_prompts()` caches prompts.yml on first load with no invalidation. If prompts are updated while the server runs, changes aren't picked up. Add a dev-mode refresh option or file-modified timestamp check.
  - Location: `novelforge/llm/prompts.py:49-57`

---

## 4. User Experience

### Medium Priority

- [ ] **Dark Mode CSS Completeness** - Dark mode covers LLM chat, editable cells, and accordion body, but custom card backgrounds, form inputs, and table cells may not have sufficient contrast. Test all UI elements in dark mode and add missing `[data-bs-theme="dark"]` rules.
  - Location: `static/css/style.css:144-174`

- [ ] **Mobile Responsive Design** - Replace fixed-width tables in chapter and character review panels with card-based layouts on small viewports using Bootstrap 5 responsive utilities.
  - Location: `templates/index.html`, `static/css/style.css`

- [ ] **Add Print Styles** - Optimize the chapter preview panel for printing with `@media print` CSS rules (hide nav, expand collapsed chapters, set readable font sizes).
  - Location: `static/css/style.css`

### Low Priority

- [ ] **Add Novel Templates** - Allow users to start from pre-defined story archetypes (e.g., "Hero's Journey", "Murder Mystery", "Romance Arc") that pre-populate premise, genre, and chapter structure suggestions.

- [ ] **Add Chapter Drag-and-Drop Reordering** - Allow users to reorder chapters in the Step 2 outline table via drag-and-drop (using SortableJS) rather than only up/down arrow buttons.
  - Location: `static/js/script.js`, `templates/index.html`

- [ ] **Add Export Formats** - Support EPUB, PDF, and DOCX export in addition to Markdown. Use `ebooklib` for EPUB, `WeasyPrint` for PDF, `python-docx` for DOCX.

---

## 5. Infrastructure & Deployment

### Medium Priority

- [ ] **Add Health Check Endpoint** - Create a `/health` route returning `{"status": "ok", "version": "..."}` for load balancer monitoring and uptime checks. Should return `503` if any critical dependency is unreachable.
  - Location: `novelforge/__init__.py` or `novelforge/routes/` (new)

- [ ] **Add Database Support for Persistence** - Migrate from file-based sessions and in-memory progress store to SQLite (single-server) or PostgreSQL (multi-server). Store novel metadata, chapter content, and generation progress in normalized tables.
  - Use Flask-SQLAlchemy; implement with Alembic/Flask-Migrate for schema versioning

- [ ] **Add Docker Support** - Create a `Dockerfile` and `docker-compose.yml` (with optional Redis service for the progress store) to simplify deployment.

### Low Priority

- [ ] **Replace Progress Polling with Server-Sent Events** - The current adaptive polling (`/progress/<token>`) generates HTTP requests even when nothing changes. Replace with SSE (`/progress/stream/<token>`) using Flask's `Response` with `mimetype="text/event-stream"`.

- [ ] **Add Structured Logging** - Replace ad-hoc `logging.info()` calls with JSON-structured output (`python-json-logger`) for better aggregation in production log systems (Datadog, CloudWatch, ELK).

---

## 6. Testing

### Medium Priority

- [ ] **Add Security Test Suite** - Test XSS vectors (contenteditable → server → render), directory traversal in all file-serving routes, session ID validation, CSRF enforcement, and rate limiter behavior.
  - Location: `tests/test_security.py` (new file)

- [ ] **Fix Background Thread Test Warning** - The concurrency test `test_two_generation_requests_get_different_tokens` spawns daemon threads that outlive the test and cause `PytestUnhandledThreadExceptionWarning` when the progress store is cleared. Add proper thread cleanup or use a dedicated progress store per test.
  - Location: `tests/test_concurrency.py`

### Low Priority

- [ ] **Add mypy to CI** - Run `mypy novelforge/` in CI to catch type errors. Current type hints are comprehensive but not verified by a checker.

- [ ] **Measure Test Coverage** - Add `pytest-cov` and generate coverage reports. Target: 85%+ line coverage across `novelforge/` package.

---

## 7. Future Features

### Medium Priority

- [ ] **Add User Accounts** - Implement optional user registration/login (Flask-Login + SQLAlchemy) to persist novels and generation history across browser sessions.

- [ ] **Add POV/Focal Character Planner** - The 8th planning agent exists but is not yet shown in the Step 2 UI. Add a tab or section to display and optionally edit the POV plan before generation.

### Low Priority

- [ ] **Add Collaboration Features** - Allow multiple users to co-edit the same outline in real-time using WebSockets (Flask-SocketIO).

- [ ] **Add Version History** - Track every revision of each chapter and allow rollback to any prior version. Store diffs or full snapshots in the database.

- [ ] **Add Character Relationship Mapping** - Generate and display a visual graph (using D3.js or Mermaid) showing character relationships as defined by the character agent.

- [ ] **Add Writing Statistics Dashboard Enhancements** - Add charts (using Chart.js) for word count per chapter, token usage trends, and generation time distribution. Currently the dashboard is table-only.

---

## Summary

| Category | High | Medium | Low | Total |
|----------|------|--------|-----|-------|
| Security | 3 | 2 | 0 | 5 |
| Reliability | 1 | 5 | 0 | 6 |
| Code Quality | 0 | 4 | 2 | 6 |
| User Experience | 0 | 2 | 3 | 5 |
| Infrastructure | 0 | 2 | 2 | 4 |
| Testing | 0 | 2 | 2 | 4 |
| Future Features | 0 | 2 | 3 | 5 |
| **Total** | **4** | **19** | **12** | **35** |

---

## Strengths of the Current Codebase

- **Well-structured package architecture** - Clean separation of concerns: routes, agents, LLM client, session persistence, and validation each in focused modules
- **Robust error handling** - Circuit breaker, per-chapter timeouts, user-friendly error messages, graceful fallbacks on planning agent failures
- **Comprehensive test suite** - 174 tests covering validation, routes, integration, concurrency, boundaries, and session persistence
- **Session resilience** - Completed chapters persisted incrementally to disk; sessions survive server restarts and can be loaded from dropdown
- **Planning agent architecture** - `BaseAgent` abstract class eliminates duplication; parallel execution with selective regeneration cuts planning time
- **Security foundations** - CSRF protection, XSS escaping, input validation, directory traversal protection, rate limiting
- **Developer experience** - Correlation IDs in logs, type hints throughout, configurable constants via env vars, dark mode, adaptive polling
- **Documentation** - Detailed README with architecture diagrams, troubleshooting guide, performance tuning, and complete API reference

## Key Weaknesses

- **In-memory progress store** is the single largest reliability risk; not suitable for multi-worker deployment
- **No atomic file writes** for session persistence; crash during write corrupts the file
- **Session ID not validated** in load_session route; minor directory traversal risk
- **CLAUDE.md and .env.example are stale** and don't reflect the refactored codebase
- **Dark mode CSS incomplete** for some custom components
- **No log rotation** on llm.log; disk fill risk on long-running servers

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

- [ ] **Add CSRF Protection** - Implement Flask-WTF CSRF tokens on all POST routes. Currently, any malicious website could trick users into triggering novel generation.
  - Location: All routes in `app.py`, all AJAX calls in `script.js`

- [ ] **Implement Rate Limiting** - Add Flask-Limiter with per-IP rate limits (e.g., 1 generation per minute) to prevent API abuse and DoS attacks.
  - Location: `app.py` (new middleware)

- [ ] **Add Upper Bounds to Input Validation** - Enforce maximum values for chapters (e.g., 100) and word count (e.g., 500,000) to prevent memory/timeout issues.
  - Location: `app.py:validate_outline_input()`

### Medium Priority

- [ ] **Add Security Test Suite** - Implement comprehensive XSS, CSRF, injection, and path traversal tests.
  - Location: `tests/test_app.py`

- [ ] **Sanitize Contenteditable Fields** - Add character limits and validation to inline-editable table cells to prevent malformed data.
  - Location: `script.js:279-291`

---

## 2. Reliability & Error Handling

### High Priority

- [ ] **Add Configuration Validation at Startup** - Validate all required environment variables (LLM_API_KEY, LLM_API_URL) before starting the server. Show clear error messages for missing/invalid config.
  - Location: `app.py` (startup), `config.py`

- [ ] **Replace In-Memory Progress Store** - Migrate `_progress_store` dict to Redis or SQLite for multi-process WSGI deployments and crash recovery.
  - Location: `app.py:72-74`

### Medium Priority

- [ ] **Implement Circuit Breaker for LLM API** - If 3 consecutive LLM calls fail, abort the operation instead of continuing to hammer the API.
  - Location: `app.py:call_llm()`, `app.py:_run_chapter_generation_internal()`

- [ ] **Add Per-Chapter Timeout** - Implement a maximum time (e.g., 30 minutes) for each chapter generation to prevent runaway threads.
  - Location: `app.py:_run_all_chapter_agents()`

- [ ] **Improve Error Logging for Planning Agents** - Log which specific planning agent failed with full context before returning error response.
  - Location: `app.py:4352-4487`

- [ ] **Add Polling Failure Handling** - After 5 consecutive polling failures in JavaScript, show a warning to the user instead of silently continuing.
  - Location: `script.js:558-560`

- [ ] **Add Exponential Backoff to Progress Polling** - Implement adaptive polling intervals based on server response time.
  - Location: `script.js:525-561`

### Low Priority

- [ ] **Add Schema Validation for Session Persistence** - Use Pydantic or similar to ensure all session keys are properly serialized/deserialized.
  - Location: `app.py:77-198`

---

## 3. Performance & Scalability

### Medium Priority

- [ ] **Implement Selective Agent Regeneration** - Only regenerate affected planning agents when user edits the outline, not all 7 agents every time.
  - Location: `app.py:4519-4573` (`/approve_outline`)

- [ ] **Add Caching for Planning Agent Outputs** - Hash outline data and skip regeneration if nothing changed.
  - Location: `app.py:approve_outline()`

### Low Priority

- [ ] **Implement Virtual Scrolling for Chapter Lists** - For novels with 50+ chapters, paginate or virtualize the chapter list to prevent UI slowdown.
  - Location: `script.js`, `index.html`

- [ ] **Compress Session Files** - Use gzip compression for session persistence files to reduce disk usage.
  - Location: `app.py:131`

- [ ] **Return Incremental Progress Updates** - Only return new chapters since last poll instead of full data every time.
  - Location: `app.py:5490-5498`

---

## 4. Testing

### High Priority

- [ ] **Add Mock LLM Tests** - Use `unittest.mock` to test LLM integration routes without live API calls.
  - Location: `tests/test_app.py`

### Medium Priority

- [ ] **Add Route Coverage Tests** - Add tests for missing routes: `/revise_chapter`, `/llm_log`, `/export_editors_notes`, `/check_saved_state`, `/resume_session`, `/new_session`.
  - Location: `tests/test_app.py`

- [ ] **Add Concurrency/Threading Tests** - Test progress tracking, thread safety, and concurrent chapter generation.
  - Location: `tests/test_app.py`

- [ ] **Add Boundary Condition Tests** - Test edge cases like exactly 1000 word count, exactly 3 chapters, special characters in premise.
  - Location: `tests/test_app.py`

---

## 5. User Experience

### Medium Priority

- [ ] **Add Accessibility Labels** - Add ARIA attributes to step panels (`aria-hidden`, `role="main"`) and proper `for` attributes on form labels.
  - Location: `index.html`

- [ ] **Implement Mobile Responsive Design** - Replace tables with cards on mobile viewports for better usability.
  - Location: `index.html`, `style.css`

- [ ] **Add User-Friendly Error Messages** - Provide specific, actionable error messages instead of generic ones (e.g., "The AI service is slow. You can wait and retry.").
  - Location: `app.py` (all routes)

### Low Priority

- [ ] **Add Unsaved Changes Warning** - Track dirty state and show confirmation modal when navigating away with unsaved outline edits.
  - Location: `script.js`

- [ ] **Add Progress Time Estimation** - Track average time per chapter and show estimated time remaining.
  - Location: `script.js:564-575`

- [ ] **Add Dark Mode Support** - Implement `@media (prefers-color-scheme: dark)` CSS rules.
  - Location: `style.css`

- [ ] **Add Print Styles** - Optimize chapter preview for printing.
  - Location: `style.css`

- [ ] **Disable Export Button During Processing** - Prevent duplicate exports from rapid clicks.
  - Location: `script.js:643-674`

- [ ] **Implement Clear Log Button** - The "Clear log" button exists but has no functionality.
  - Location: `script.js`

---

## 6. Code Organization & Maintainability

### Medium Priority

- [ ] **Refactor app.py into Modules** - Split into:
  - `app/llm.py` - LLM helpers and call_llm()
  - `app/prompts.py` - All prompt builder functions
  - `app/agents.py` - Planning agent logic
  - `app/routes.py` - Flask route handlers
  - `app/models.py` - Data structures and schemas
  - Location: `app.py` (6000+ lines)

- [ ] **Create PlanningAgent Base Class** - Abstract the common build/normalize/plan/context pattern used by all 7 planning agents.
  - Location: `app.py:433-699`

- [ ] **Use Prompt Templating** - Replace f-string prompts with Jinja2 templates or a structured builder pattern with validation.
  - Location: `app.py:365-422`

### Low Priority

- [ ] **Add Consistent Type Hints** - Add comprehensive type hints to all functions using Python 3.11+ syntax.
  - Location: `app.py` (throughout)

- [ ] **Centralize Chapter Split Logic** - Create a `ChapterSplitter` utility to handle chapter percentage calculations consistently.
  - Location: Multiple locations in `app.py`

- [ ] **Move Magic Numbers to Config** - Move hardcoded values (MAX_RETRIES, RETRY_DELAY, timeout) to `config.py`.
  - Location: `app.py:210-211`, `app.py:251`

---

## 7. Infrastructure & Deployment

### Medium Priority

- [ ] **Add Health Check Endpoint** - Create `/health` route that returns `{"status": "ok"}` for load balancer monitoring.
  - Location: `app.py` (new route)

- [ ] **Add Database Support** - Migrate from file-based sessions to SQLite or PostgreSQL for better scalability and audit trails.
  - Location: New module

- [ ] **Add Database Migrations** - When adding database support, use Alembic or Flask-Migrate for schema management.
  - Location: New configuration

### Low Priority

- [ ] **Add Structured Logging** - Implement JSON structured logging for better log aggregation in production.
  - Location: `app.py` (logging configuration)

---

## 8. Documentation

### Low Priority

- [ ] **Add Architecture Diagram** - Create visual overview of system components (frontend, backend, LLM API, session storage).
  - Location: `README.md`

- [ ] **Add Troubleshooting Guide** - Document common issues and solutions (timeouts, disk space, session problems).
  - Location: `README.md`

- [ ] **Add Performance Tuning Guide** - Document typical generation times, resource requirements, and optimization tips.
  - Location: `README.md`

- [ ] **Generate OpenAPI Specification** - Create detailed API documentation with request/response examples for all endpoints.
  - Location: New file or README section

---

## 9. Features (Future Enhancements)

### Medium Priority

- [ ] **Add User Accounts** - Implement optional user registration/login to save novels across sessions.

- [ ] **Add Novel Templates** - Allow users to start from pre-defined story templates (e.g., "Hero's Journey", "Murder Mystery").

- [ ] **Add Export Formats** - Support additional export formats beyond Markdown (EPUB, PDF, DOCX).

### Low Priority

- [ ] **Add Collaboration Features** - Allow multiple users to edit the same outline.

- [ ] **Add Version History** - Track changes to chapters and allow rollback to previous versions.

- [ ] **Add Chapter Reordering** - Allow drag-and-drop reordering of chapters in the outline.

- [ ] **Add Character Relationship Mapping** - Visual diagram showing character relationships.

- [ ] **Add Writing Statistics Dashboard** - Show word count trends, generation times, and revision history.

---

## Summary

| Category | High | Medium | Low | Total |
|----------|------|--------|-----|-------|
| Security | 3 | 2 | 0 | 5 |
| Reliability | 2 | 5 | 1 | 8 |
| Performance | 0 | 2 | 3 | 5 |
| Testing | 1 | 3 | 0 | 4 |
| User Experience | 0 | 3 | 6 | 9 |
| Code Organization | 0 | 3 | 3 | 6 |
| Infrastructure | 0 | 3 | 1 | 4 |
| Documentation | 0 | 0 | 4 | 4 |
| Future Features | 0 | 3 | 5 | 8 |
| **Total** | **6** | **24** | **23** | **53** |

---

## Next Steps

1. Start with HIGH priority security items (CSRF, rate limiting)
2. Address reliability issues (config validation, progress store)
3. Expand test coverage with mock LLM tests
4. Refactor code organization for maintainability
5. Polish UX and add documentation

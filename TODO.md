# NovelForge TODO

## 3. Architecture & Code Organization

### High Priority


### Medium Priority

- [ ] **Parallelize Independent Planning Agents** - The 7 planning agents currently run sequentially during `/approve_outline`, taking 2–5 minutes. Several agents are independent of each other and can run concurrently. Use `concurrent.futures.ThreadPoolExecutor` to run independent groups in parallel and cut planning time roughly in half.
  - Group 1 (independent, run in parallel): Story Architecture, Master Timeline, Technology Rules, Theme Reinforcement
  - Group 2 (depend on Group 1 outputs): Character Fate Registry, Character Arc Planner, Antagonist Motivation
  - Location: `app.py:approve_outline()` (approximately lines 4519–4573)

- [X] **Add Comprehensive Type Hints** - Add type hints to all functions using Python 3.11+ syntax (`list[dict]`, `str | None`, etc.). Enables IDE autocompletion, catches type errors with mypy, and serves as inline documentation.
  - Prioritize public interfaces: route handlers, `call_llm()`, agent `plan()` methods, session persistence functions
  - Location: `app.py` (throughout)

### Low Priority

- [ ] **Consolidate Export Format Functions** - The 4 export formatters (`_format_clean_manuscript`, `_format_annotated_manuscript`, `_format_publishing_manuscript`, `_format_critique_manuscript`) share ~300 lines of duplicated header/chapter/footer logic. Extract a shared template with format-specific overrides.
  - Location: `app.py` (approximately lines 5180–5391)

- [ ] **Refactor `_run_all_chapter_agents` Signature** - This function takes 18 parameters (9 required + 9 with defaults), creating high cognitive load. Extract a `ChapterContext` dataclass to bundle the related context fields.
  - Location: `app.py:_run_all_chapter_agents()` (approximately lines 4189–4208)

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

### Low Priority

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

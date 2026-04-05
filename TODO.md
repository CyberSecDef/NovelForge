# NovelForge TODO — Codebase Review Findings (2026-04-05)

This document captures all issues identified during comprehensive codebase reviews.
Items are grouped by severity and ordered by priority within each group. Each item
includes the problem description, affected files/lines, why it matters, and
recommended fixes.

---

## Critical / High Priority

### 0. Eliminate Snapshot Duplication in Session JSON — OPEN

**Files:**
- `novelforge/routes/generation/chapters.py` lines 67-102 (snapshot creation, embedded in progress state)
- `novelforge/routes/generation/chapters.py` lines 157-203 (`_run_chapter_generation_internal` reads from `snap`)
- `novelforge/routes/generation/revision.py` lines 78-92 (reads `progress_data.get("snapshot")`)
- `novelforge/routes/export.py` lines 113, 138, 555-603 (reads `progress_data.get("snapshot")`)
- `novelforge/session/persistence.py` lines 471-512 (`persist_completed_chapters` writes `progress_data` including snapshot)
- `novelforge/session/persistence.py` lines 226-280 (`save_session_state` writes top-level session keys)
- `novelforge/session/persistence.py` lines 329-351 (`rebuild_stale_progress` reconstructs progress on load)
- `novelforge/session/persistence.py` lines 354-407 (`restore_session_from_state` restores session + progress)
- `novelforge/progress.py` (ProgressManager stores the snapshot in memory as part of the progress entry)

**Problem:**
The session JSON file stores every piece of novel data **twice**:

1. **Top-level keys** — `title`, `premise`, `genre`, `chapters`, `word_count`,
   `chapter_list`, `character_list`, all 8 planning agent outputs, `voice_seed`,
   `narrative_perspective`, `special_instructions` (~199 KB)
2. **`progress_data.snapshot`** — An identical copy of all the above, embedded
   inside the progress data dict (~199 KB)

This is because two independent persistence paths evolved:
- `save_session_state()` writes the Flask session's top-level keys to the JSON file
- `persist_completed_chapters()` writes the full `progress_manager.get(token)` dict
  (which includes `snapshot`) into `progress_data`

The duplication doubles the file size (currently ~400 KB before any chapters are
written, will grow to ~1-2 MB for a 25-chapter novel). Additionally,
`completed_chapters` at the top level and `progress_data.chapters_done` will also
be duplicated once chapters complete — each chapter is ~20 KB of content, so 25
chapters = ~500 KB duplicated.

Measured from a live session (`bf207166`):
```
Top-level session keys:  ~199 KB (50% of file)
progress_data.snapshot:  ~199 KB (50% of file)  ← IDENTICAL COPY
Total file:              ~398 KB
```

**Why it matters:**
- File size is 2x what it needs to be, growing to multi-MB for long novels
- Two copies of the same data can drift if a bug updates one but not the other
- The `persist_completed_chapters()` path reads the file, merges in the full
  progress_data (with snapshot), and writes it back — moving ~200 KB of redundant
  data through the read-modify-write cycle on every chapter completion
- Confusing mental model: developers must understand that "snapshot" is just a
  frozen copy of session state, not separate data

**Constraints:**
- **All existing functionality must be retained.** The generation background thread,
  revision endpoint, export endpoint, illustration generation, and session
  restore/crash recovery must all continue to work identically.
- **Crash recovery must not lose data.** If the server crashes mid-generation, the
  session JSON must contain everything needed to resume or display completed
  chapters. The file must be saved at every stage change.
- **No data duplication.** Every piece of data should be stored exactly once in the
  JSON file.

**Current data flow (what needs to change):**

1. **Generation start** (`chapters.py:67-102`): Builds a `snapshot` dict by copying
   session keys. Passes it to the background thread AND embeds it in the progress
   manager entry as `initial_state["snapshot"]`.

2. **Background thread** (`chapters.py:157-203`): Reads all novel metadata from the
   `snap` dict (the snapshot). Does not read from the session or JSON file during
   generation.

3. **Per-chapter persist** (`persistence.py:471-512`): `persist_completed_chapters()`
   reads the JSON file, sets `state["completed_chapters"]`, then sets
   `state["progress_data"] = progress_manager.get(token)` — which includes the full
   snapshot. Writes the file back.

4. **Revision** (`revision.py:78-92`): Reads `progress_data.get("snapshot")` from
   the in-memory progress manager to get novel metadata.

5. **Export** (`export.py:113,138,555-603`): Same as revision — reads from
   `progress_data.get("snapshot")`.

6. **Session restore** (`persistence.py:354-407`): Reads top-level keys from the
   JSON file into the Flask session. If `progress_data` exists, loads it into the
   progress manager (which puts the snapshot back in memory).

**Recommended approach (Option A: eliminate snapshot from persistence):**

**Phase 1 — Stop persisting the snapshot:**
- In `persist_completed_chapters()`, strip `snapshot` from the progress data before
  writing:
  ```python
  if progress is not None:
      # Don't persist the snapshot — it duplicates top-level session keys
      progress_copy = {k: v for k, v in progress.items() if k != "snapshot"}
      state["progress_data"] = progress_copy
  ```
- Ensure `save_session_state()` saves all the keys that were in the snapshot as
  top-level keys (it already does).

**Phase 2 — Rebuild snapshot on load:**
- In `restore_session_from_state()`, after restoring the Flask session and creating
  the progress manager entry, inject a snapshot built from the top-level keys:
  ```python
  if token and pd:
      # Rebuild snapshot from top-level session keys (not stored in progress_data)
      pd["snapshot"] = {
          "session_id": state.get("session_id", ""),
          "premise": state.get("premise", ""),
          "genre": state.get("genre", ""),
          "chapters": state.get("chapters", 0),
          "word_count": state.get("word_count", 0),
          "special_instructions": state.get("special_instructions", ""),
          "title": state.get("title", ""),
          "chapter_list": state.get("chapter_list", []),
          "character_list": state.get("character_list", []),
          "story_architecture": state.get("story_architecture", {}),
          "master_timeline": state.get("master_timeline", {}),
          "character_fate_registry": state.get("character_fate_registry", {}),
          "character_arc_plan": state.get("character_arc_plan", {}),
          "antagonist_motivation_plan": state.get("antagonist_motivation_plan", {}),
          "technology_rules": state.get("technology_rules", {}),
          "theme_reinforcement": state.get("theme_reinforcement", {}),
          "pov_focal_character_plan": state.get("pov_focal_character_plan", {}),
          "voice_seed": state.get("voice_seed", {}),
          "narrative_perspective": state.get("narrative_perspective", "third_person"),
      }
      progress_manager.create(token, pd)
  ```
- Do the same in `rebuild_stale_progress()` — inject the snapshot from the state
  dict.

**Phase 3 — Eliminate `completed_chapters` / `chapters_done` duplication:**
- Stop writing `state["completed_chapters"]` as a separate top-level key. Instead,
  read it from `progress_data.chapters_done` on restore.
- OR stop writing `chapters_done` in progress_data and read from top-level
  `completed_chapters` on restore.
- Pick one canonical location and derive the other.

**Phase 4 — Ensure every stage change persists:**
- Audit all places where novel state changes (planning agents, chapter completion,
  revision, audit reports) and confirm the JSON file is written.
- The current `_persist_progress()` throttled writes + forced writes on chapter
  completion handle most cases, but verify that planning agent updates during
  `/approve_outline` also trigger a save.

**Phase 5 — Update all snapshot consumers:**
- `revision.py:78` reads `progress_data.get("snapshot")` — this continues to work
  because the snapshot is rebuilt in memory on load and still set by
  `generate_chapters` at generation start. No change needed.
- `export.py:113,138,555` — same, no change needed.
- `chapters.py:157-203` — receives `snap` as a function argument, no change needed.

**Phase 6 — Backward compatibility:**
- The restore path must handle old JSON files that still contain
  `progress_data.snapshot`. If present, use it; if absent, rebuild from top-level
  keys. This ensures existing session files load correctly.

**Testing:**
- Verify session JSON file size is roughly halved after the change.
- Verify crash recovery: kill the server mid-generation, restart, load session,
  confirm all completed chapters are present.
- Verify revision: after loading a restored session, revise a chapter and confirm
  it uses correct title/genre/characters.
- Verify export: confirm export uses correct title.
- Verify old session files (with embedded snapshot) still load correctly.
- Verify new session files (without embedded snapshot) load correctly.

---

### 1. Zero Test Coverage for 4 New Pipeline Steps — OPEN

**Files:**
- `novelforge/agents/chapter/prompts.py` (builders: `build_voice_dialogue_differentiation_prompt`, `build_human_oddities_prompt`, `build_metaphor_reduction_prompt`, `build_copy_edit_prompt`)
- `novelforge/agents/chapter/pipeline.py` (calls at lines ~250, ~300, ~365, ~393)
- `tests/conftest.py` lines 38-234 (`_canned_llm_response` function)
- `prompts.yml` (entries: `voice_dialogue_differentiation`, `human_oddities`, `metaphor_reduction`, `copy_edit`)

**Problem:**
The voice & dialogue differentiation, human oddities, metaphor reduction, and copy
edit steps are wired into the chapter generation pipeline and execute in production,
but have no dedicated tests. The `_canned_llm_response` mock in `conftest.py` has no
cases matching these prompt actions — they all fall through to the generic response
`"Processed output from the LLM agent."`.

This means:
- No validation that the prompt builders produce correct message structures (system +
  user messages with expected template variables)
- No verification that the pipeline calls them with the right parameters
  (`characters_text`, `perspective_prompt`, `total_chapters`, etc.)
- Integration tests won't catch regressions if `prompts.yml` entries are renamed or
  removed
- The mock returns a generic string, not chapter text, so tests don't verify that
  these steps receive and return chapter content correctly

**Why it matters:**
These are production code paths that run for every chapter (4 extra LLM calls per
chapter). A regression — such as a missing template variable in prompts.yml, a
renamed prompt entry, or a wrong parameter — would cause a generation failure that
no test catches.

**Recommended fix:**
1. Add canned responses to `_canned_llm_response()` in `conftest.py` for each new
   action string:
   ```python
   if "voice & dialogue differentiation" in action.lower():
       return text  # return the chapter text (these are text-in/text-out passes)
   if "human oddities" in action.lower():
       return text
   if "metaphor reduction" in action.lower():
       return text
   if "copy edit" in action.lower():
       return text
   ```

2. Add unit tests for each prompt builder in a new `tests/test_new_pipeline_prompts.py`:
   ```python
   def test_voice_dialogue_differentiation_prompt_structure():
       msgs = build_voice_dialogue_differentiation_prompt(
           chapter_text="Chapter text.", chapter_num=1, title="Novel",
           characters_text="- Alice (age 30): protagonist.", perspective_prompt="",
       )
       assert len(msgs) == 2
       assert msgs[0]["role"] == "system"
       assert "voice" in msgs[0]["content"].lower() or "dialogue" in msgs[0]["content"].lower()
       assert "Chapter text." in msgs[1]["content"]
   ```

3. Add a pipeline integration test verifying all 4 new steps execute without error
   when the full pipeline runs under `mock_llm`.

---

### 2. Assertion-Based Genre Validation Disabled by Python `-O` — OPEN

**Files:**
- `novelforge/names.py` lines 53-56
- `novelforge/voice.py` lines 247-250

**Problem:**
Both files use `assert` statements to validate that every genre in `ALLOWED_GENRES`
has a corresponding entry in their respective mappings:

```python
# names.py
_missing_name_genres = ALLOWED_GENRES - _GENRE_GROUP.keys()
assert not _missing_name_genres, (
    f"Genres missing from _GENRE_GROUP in names.py: {sorted(_missing_name_genres)}. "
    f"Add a style-group mapping for each."
)

# voice.py
_missing_voice_genres = ALLOWED_GENRES - _GENRE_VOICE_WEIGHTS.keys()
assert not _missing_voice_genres, (
    f"Genres missing from _GENRE_VOICE_WEIGHTS in voice.py: ..."
)
```

When Python runs with `-O` (optimize flag), all `assert` statements are compiled out
entirely. This means the genre coverage checks vanish in optimized deployments.

**Why it matters:**
If someone adds a genre to `ALLOWED_GENRES` without adding a name pool mapping or
voice weight mapping, the application silently falls through to default behavior:
- `names.py` returns the `"contemporary"` name pool for unmapped genres (via
  `_GENRE_GROUP.get(genre, "contemporary")`)
- `voice.py` produces an unweighted random voice selection (no genre bias)

The user sees no error — just culturally inappropriate character names and random
prose voice for that genre. The assertions were added specifically to catch this at
startup, but they don't work in all deployment modes.

**Recommended fix:**
Replace assertions with explicit `if` checks that raise `ValueError`:

```python
# names.py
_missing_name_genres = ALLOWED_GENRES - _GENRE_GROUP.keys()
if _missing_name_genres:
    raise ValueError(
        f"Genres missing from _GENRE_GROUP in names.py: {sorted(_missing_name_genres)}. "
        f"Add a style-group mapping for each."
    )

# voice.py
_missing_voice_genres = ALLOWED_GENRES - _GENRE_VOICE_WEIGHTS.keys()
if _missing_voice_genres:
    raise ValueError(
        f"Genres missing from _GENRE_VOICE_WEIGHTS in voice.py: "
        f"{sorted(_missing_voice_genres)}. Add a voice weight mapping for each."
    )
```

This runs unconditionally at import time, regardless of optimization flags.

---

### 3. Brace-Counting JSON Parser in Debug Endpoint — OPEN

**Files:** `novelforge/__init__.py` lines 250-261

**Problem:**
The `/llm_log` debug endpoint reconstructs multi-line JSON objects from the log file
using manual `{`/`}` counting:

```python
for line in content.split('\n'):
    if line.strip().startswith('{') and brace_count == 0:
        if current_obj:
            json_objects.append(current_obj)
        current_obj = line + '\n'
        brace_count = line.count('{') - line.count('}')
    elif brace_count > 0:
        current_obj += line + '\n'
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0:
            json_objects.append(current_obj)
            current_obj = ""
```

This breaks on:
- Strings containing brace characters: `{"key": "value with { brace"}`
- Escaped quotes within JSON strings
- Single-line complete JSON objects that happen to have balanced braces

**Why it matters:**
While this is debug-only (protected by `if not app.debug: abort(404)`), it produces
corrupted or missing log entries when viewing LLM request/response logs. Developers
debugging LLM issues see truncated or garbled entries.

**Recommended fix:**
Replace with a proper JSON-lines reader that treats each complete JSON object as a
separate entry. Since `llm_logger` writes one `json.dumps(..., indent=2)` call per
log entry, the log file contains pretty-printed multi-line JSON. The simplest fix:

Option A: Change the logger to write one JSON object per line (no indent):
```python
llm_logger.info(json.dumps(request_log))  # no indent=2
```
Then the reader becomes `[json.loads(line) for line in content.splitlines() if line.strip()]`.

Option B: Use a streaming JSON parser to read the existing format:
```python
import re
# Split on lines that start with '{' at column 0 (top-level objects)
raw_objects = re.split(r'\n(?=\{)', content)
entries = []
for raw in raw_objects:
    try:
        entries.append(json.loads(raw))
    except json.JSONDecodeError:
        continue
```

---

### 4. Environment Variable Paths Accept Absolute Paths — OPEN

**Files:** `novelforge/config.py` lines 237-246

**Problem:**
Directory path configuration uses `PROJECT_ROOT / os.environ.get(...)`:

```python
SESSION_FILE_DIR = str(PROJECT_ROOT / os.environ.get("SESSION_FILE_DIR", "sessions/flask"))
EXPORT_DIR = str(PROJECT_ROOT / os.environ.get("EXPORT_DIR", "exports"))
NOVELS_DIR = str(PROJECT_ROOT / os.environ.get("NOVELS_DIR", "sessions/novels"))
LOGS_DIR = str(PROJECT_ROOT / os.environ.get("LOGS_DIR", "logs"))
```

Python's `Path.__truediv__` (`/` operator) silently replaces the left operand if the
right operand is an absolute path:

```python
Path("/app") / "/etc/passwd"  # → PosixPath('/etc/passwd')
```

If any of these environment variables is set to an absolute path (misconfiguration,
injection, or CI/CD accident), the application reads/writes files from that arbitrary
location instead of the project directory.

**Why it matters:**
- `NOVELS_DIR` controls where session JSON files (containing full novel content) are
  written with `os.chmod(0o600)` and read back
- `EXPORT_DIR` controls where Markdown exports and illustration images are saved and
  served via `/download/<filename>`
- `LOGS_DIR` controls where LLM request/response logs (containing prompts and API
  key prefixes) are written

A malicious or misconfigured absolute path could cause data to be written to or read
from unintended locations.

**Recommended fix:**
Validate that environment-provided paths are relative before joining with
`PROJECT_ROOT`:

```python
def _resolve_dir(env_var: str, default: str) -> str:
    """Resolve a directory path relative to PROJECT_ROOT.

    Raises ValueError if the environment variable contains an absolute path.
    """
    raw = os.environ.get(env_var, default)
    if os.path.isabs(raw):
        raise ValueError(
            f"{env_var} must be a relative path (got {raw!r}). "
            f"Absolute paths are not allowed for security reasons."
        )
    return str(PROJECT_ROOT / raw)

SESSION_FILE_DIR = _resolve_dir("SESSION_FILE_DIR", "sessions/flask")
EXPORT_DIR = _resolve_dir("EXPORT_DIR", "exports")
NOVELS_DIR = _resolve_dir("NOVELS_DIR", "sessions/novels")
LOGS_DIR = _resolve_dir("LOGS_DIR", "logs")
```

---

## Medium Priority

### 5. Snapshot Not Validated in Revision or Export — OPEN

**Files:**
- `novelforge/routes/generation/revision.py` line 78
- `novelforge/routes/export.py` lines 108-113

**Problem:**
Both the revision and export endpoints read the progress snapshot without validating
it exists or contains required fields:

```python
# revision.py line 78
snap = progress_data.get("snapshot", {})
title = snap.get("title", "Novel")
genre = snap.get("genre", "")

# export.py line 113
title = (progress_data.get("snapshot") or {}).get("title", "Novel")
```

If the snapshot is missing (corrupted progress entry, race condition during
generation startup, or manual deletion):
- **Revision:** proceeds with empty title, genre, characters, and planning data.
  The LLM receives prompts with blank context, producing a revision that ignores
  the story entirely. The revised chapter is saved, overwriting the original.
- **Export:** names the file `Novel.md`. Multiple novels with missing snapshots
  silently overwrite each other.

Neither endpoint logs a warning or returns an error.

**Why it matters:**
A missing snapshot during revision causes silent data degradation — the revised
chapter is saved with no connection to the story's context. The user sees a
successfully revised chapter that doesn't match their novel.

**Recommended fix:**
Validate the snapshot early and return a clear error:

```python
# revision.py
snap = progress_data.get("snapshot")
if not snap or not isinstance(snap, dict):
    logger.error("Progress snapshot missing for token %s", token)
    return jsonify({"error": "Progress data is incomplete. Please regenerate chapters."}), 400

# export.py
snap = progress_data.get("snapshot")
if not snap or not isinstance(snap, dict):
    return jsonify({"error": "Progress data is incomplete (snapshot missing)."}), 400
title = snap.get("title", "Novel")
```

---

### 6. Image Download Timeout Hardcoded — OPEN

**Files:** `novelforge/llm/image.py` line 115

**Problem:**
```python
img_resp = requests.get(image_url, timeout=60, stream=True)
```

The image download timeout is hardcoded to 60 seconds. The configurable
`IMAGE_TIMEOUT` setting (default 120s) is used for the image *generation* API call
(line 58) but not for downloading the generated image.

**Why it matters:**
If a user sets `IMAGE_TIMEOUT=300` because their image API is slow, the generation
call gets 300 seconds but the download step still fails at 60 seconds. Large images
(e.g., 4096x4096) on slow connections could exceed 60 seconds.

**Recommended fix:**
```python
img_resp = requests.get(image_url, timeout=config.IMAGE_TIMEOUT, stream=True)
```

---

### 7. Silent KeyError in Illustration Progress Update — OPEN

**Files:** `novelforge/routes/export.py` lines 571-574

**Problem:**
```python
try:
    progress_manager.update(token, {"illustration_token": illust_token})
except KeyError:
    pass
```

If the progress entry for `token` was deleted (e.g., user deleted the session while
illustrations were generating), the `KeyError` is silently swallowed. The
illustration is generated and saved to disk but never linked to the novel's progress
data.

**Why it matters:**
The illustration exists on disk but the user sees no illustration in the UI because
the progress entry has no `illustration_token`. Disk space is consumed with no way
for the user to access the image.

**Recommended fix:**
```python
try:
    progress_manager.update(token, {"illustration_token": illust_token})
except KeyError:
    logger.warning(
        "Could not link illustration to novel (progress entry %s not found). "
        "The illustration was saved but may not appear in the UI.",
        token,
    )
```

---

### 8. `snapshot()` Method Returns Shallow Copies (Inconsistent with `get()`) — OPEN

**Files:** `novelforge/progress.py` lines 184-190

**Problem:**
```python
def snapshot(self) -> dict[str, ProgressState]:
    """Return shallow copies of every entry, keyed by token."""
    with self._lock:
        return {k: dict(v) for k, v in self._store.items()}
```

The `get()` method was updated to return deep copies (item #20 in previous review),
but `snapshot()` still returns shallow copies. Nested structures (lists, dicts) in
the returned entries are shared references to the internal store.

**Why it matters:**
Tests or diagnostic code using `snapshot()` could inadvertently mutate the internal
store by modifying nested data in the returned entries. This contradicts the safety
guarantee established by making `get()` return deep copies.

**Recommended fix:**
```python
def snapshot(self) -> dict[str, ProgressState]:
    """Return deep copies of every entry, keyed by token."""
    with self._lock:
        return {k: copy.deepcopy(v) for k, v in self._store.items()}
```

---

### 9. Premise Tokenization Strips Non-ASCII Characters — OPEN

**Files:** `novelforge/voice.py` line 268

**Problem:**
```python
premise_tokens = set(re.sub(r"[^a-z\s]", " ", premise.lower()).split())
```

The regex `[^a-z\s]` removes ALL non-ASCII characters before keyword matching.
Accented characters, CJK text, and any Unicode letters are stripped:
- `"café rebellion"` → `"caf rebellion"` (breaks "café" keyword match)
- `"über-dystopian city"` → `"ber dystopian city"` (breaks "über")
- `"résistance movement"` → `"r sistance movement"`

**Why it matters:**
Premise text containing non-English words or accented characters gets incorrect
voice seed selection because keywords are partially destroyed before matching. The
voice seed system was designed to bias selection based on premise content, but the
tokenization step defeats this for any non-ASCII input.

**Recommended fix:**
Use `\w` with Unicode support or `re.findall`:

```python
premise_tokens = set(re.findall(r'\w+', premise.lower()))
```

This preserves accented characters and Unicode word characters while still splitting
on whitespace and punctuation.

---

### 10. f-String Logging in Multiple Locations — OPEN

**Files:**
- `novelforge/routes/sessions.py` lines 68, 70
- `novelforge/__init__.py` line 238

**Problem:**
Several log statements use f-strings instead of parameterized logging:

```python
# sessions.py
logger.info(f"Deleted session file {session_file}")
logger.error(f"Failed to delete session file: {e}")

# __init__.py
logger.warning(f"LLM log file not found at {log_path}")
```

The rest of the codebase consistently uses parameterized logging:
```python
logger.info("Deleted session file %s", session_file)
```

**Why it matters:**
- **Performance:** f-strings are eagerly formatted even if the log level is disabled.
  With parameterized logging, formatting is deferred until the log handler confirms
  the message will be emitted.
- **Log aggregation:** Structured logging tools (Datadog, Splunk, etc.) can group
  parameterized log messages by template pattern. f-strings produce unique strings
  that can't be grouped.
- **Consistency:** The rest of the codebase follows the parameterized pattern.

**Recommended fix:**
Replace f-strings with `%s` parameters in all logger calls:

```python
# sessions.py
logger.info("Deleted session file %s", session_file)
logger.error("Failed to delete session file: %s", e)

# __init__.py
logger.warning("LLM log file not found at %s", log_path)
```

---

## Low Priority / Quality

### 11. Frontend Status Indicator Doesn't Recognise New Pipeline Steps — OPEN

**Files:** `static/js/script.js` lines 203-264 (`inferStatusFromRequestEntry` function)

**Problem:**
The function maps LLM action strings to human-readable status labels for the
progress UI. It recognises ~15 existing agents but has no entries for the 4 new
pipeline steps:
- `"voice & dialogue differentiation"` → no match → shows "Prompting LLM"
- `"human oddities"` → no match → shows "Prompting LLM"
- `"metaphor reduction"` → no match → shows "Prompting LLM"
- `"copy edit"` → no match → shows "Prompting LLM"

**Why it matters:**
Users monitoring chapter generation see generic "Prompting LLM" status during these
steps instead of meaningful feedback like "Differentiating character voices" or
"Reducing metaphor density". This is a UX regression — the progress indicator was
designed to show step-specific labels.

**Recommended fix:**
Add entries to the status mapping in `inferStatusFromRequestEntry`:

```javascript
if (action.includes("voice") && action.includes("dialogue")) return "Differentiating character voices";
if (action.includes("human oddities")) return "Adding human texture";
if (action.includes("metaphor reduction")) return "Reducing metaphor density";
if (action.includes("copy edit")) return "Copy editing";
```

---

### 12. LLM Log Polling Never Stops — OPEN

**Files:** `static/js/script.js` line 1583

**Problem:**
```javascript
_logPollInterval = setInterval(pollLLMLog, 15000);
```

The polling interval is created when the LLM log tab is opened but never cleared.
Even after generation completes, the user navigates away, starts a new session, or
deletes the session, the 15-second polling continues indefinitely.

**Why it matters:**
Each poll makes an AJAX GET to `/llm_log`, which reads and parses the log file.
Over time, many stale polling intervals accumulate (especially if the user loads
multiple sessions), causing unnecessary server load and network traffic.

**Recommended fix:**
Clear the interval when it's no longer needed:

```javascript
// When starting a new session
function clearLogPolling() {
    if (_logPollInterval) {
        clearInterval(_logPollInterval);
        _logPollInterval = null;
    }
}

// Call in: new session handler, delete session handler, session load handler
```

---

### 13. Dynamic Chapter Rows Missing Accessibility Attributes — OPEN

**Files:** `static/js/script.js` lines 632-677

**Problem:**
When users add chapters via "add before"/"add after" buttons, the dynamically
inserted table rows are missing accessibility attributes that the template-rendered
initial rows have:
- `contenteditable` cells lack `aria-label` attributes (e.g., "Chapter title",
  "Chapter summary")
- Action buttons (move up, move down, add before, add after, delete) lack
  `aria-label` attributes
- New rows lack `role="textbox"` on editable cells

The initial rows rendered by Jinja2 in `index.html` have proper ARIA markup, but
the JavaScript that clones/creates new rows doesn't replicate these attributes.

**Why it matters:**
Screen reader users cannot identify the purpose of cells or buttons in dynamically
added chapter rows. They hear "editable text" or "button" without context.

**Recommended fix:**
Update the chapter row insertion code to include the same ARIA attributes as the
template rows:

```javascript
$newTitle.attr({
    "contenteditable": "true",
    "role": "textbox",
    "aria-label": "Chapter title"
});
$newSummary.attr({
    "contenteditable": "true",
    "role": "textbox",
    "aria-label": "Chapter summary"
});
$deleteBtn.attr("aria-label", "Delete chapter");
// etc.
```

---

### 14. Draft Retry Block Mixes Prompt Construction with Retry Logic — OPEN

**Files:** `novelforge/routes/generation/chapters.py` lines 305-342

**Problem:**
The content-rejection retry loop for chapter drafting contains 37 lines inside a
single try block, mixing prompt parameter assembly with the LLM call and retry
logic:

```python
for _draft_attempt in range(3):
    try:
        _draft_instructions = special_instructions
        if _draft_content_note:
            _draft_instructions = f"{special_instructions}\n\n{_draft_content_note}" ...
        text = call_llm(
            build_chapter_draft_prompt(
                premise, genre, title, chapter_num, chapter_title,
                chapter_outline_summary, characters_text,
                previous_summaries, target_per_chapter, _draft_instructions,
                # ... 8 more parameters ...
            ),
            action=f"Chapter {chapter_num}: drafting"
        )
        break
    except ContentRejectionError as _draft_exc:
        if _draft_attempt >= 2:
            raise
        # ... content note setup ...
```

Every other step in the pipeline uses the clean `_safe()` / `_call_with_content_retry`
pattern. The draft step is the only one that rolls its own retry logic because it
needs to modify the *instructions* (not the chapter text) on content rejection.

**Why it matters:**
Hard to maintain. If the retry strategy needs to change (e.g., different backoff,
logging, or notification), this is the one place that doesn't follow the standard
pattern. It's also harder to test the retry behavior in isolation.

**Recommended fix:**
Extract the draft-specific retry into a helper function:

```python
def _draft_with_content_retry(
    build_prompt_fn, *, action: str, special_instructions: str, max_attempts: int = 3,
) -> str:
    """Call the draft LLM with content-rejection retry that modifies instructions."""
    content_note = ""
    for attempt in range(max_attempts):
        try:
            instructions = special_instructions
            if content_note:
                instructions = f"{special_instructions}\n\n{content_note}" if special_instructions else content_note
            return call_llm(build_prompt_fn(instructions), action=action)
        except ContentRejectionError:
            if attempt >= max_attempts - 1:
                raise
            content_note = "CONTENT NOTE: ..."
```

---

### 15. No Progress Token Format Validation — OPEN

**Files:**
- `novelforge/routes/generation/revision.py` line 48
- `novelforge/routes/generation/chapters.py` (progress endpoints)
- `novelforge/routes/export.py` lines 108, 133, 546

**Problem:**
Progress tokens are UUIDs generated by the server (`str(uuid.uuid4())`), but no
endpoint validates the token format before performing a lookup:

```python
# revision.py
token = data.get("token", "")
# ... later ...
progress_data = progress_manager.get(token)
if not progress_data or progress_data.get("status") != "done":
    return jsonify({"error": "Novel generation not complete."}), 400
```

A malformed token (e.g., `"../../../etc/passwd"`, `""`, or a very long string) is
passed directly to `progress_manager.get()`, which does a simple dict lookup and
returns `None`. The error message "Novel generation not complete" is returned for
both invalid tokens and valid tokens that haven't finished.

**Why it matters:**
- **Error clarity:** Users get a misleading error message for malformed tokens
- **Defence in depth:** While the dict lookup is safe, validating the format early
  prevents any future code path from accidentally using the token as a filename or
  path component without sanitisation
- **Log noise:** Invalid tokens generate log entries that look like real failures

**Recommended fix:**
Add a token format validator and use it in all endpoints:

```python
import re
_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

def _is_valid_token(token: str) -> bool:
    return bool(token and _UUID_RE.match(token))

# In each endpoint:
if not _is_valid_token(token):
    return jsonify({"error": "Invalid progress token."}), 400
```

---

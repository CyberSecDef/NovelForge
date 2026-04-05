# NovelForge TODO — Codebase Review Findings (2026-04-05)

This document captures all issues identified during a comprehensive codebase review.
Items are grouped by severity and ordered by priority within each group. Each item
includes the problem description, affected files/lines, why it matters, and
recommended fixes.

---

## Critical / High Priority

### 1. ~~Vocabulary Scanner Uses Substring Matching Instead of Word Boundaries~~ DONE

**Files:** `novelforge/agents/chapter/_helpers.py` lines 222-259

**Problem:**
`scan_vocabulary_overuse()` uses `str.count()` to detect forbidden and overused words,
but `str.count()` matches *substrings*, not whole words. This produces false positives:

- `"audit"` (forbidden) matches `"auditor"`, `"auditing"`, `"auditorium"`
- `"ledger"` (forbidden) matches `"sledgehammer"`
- `"tally"` (forbidden) matches `"metallurgy"`, `"tally"`, `"totally"`
- `"realm"` matches `"realms"` (acceptable) but also any compound like `"realmscape"`
- `"steady"` (soft-limited) matches `"steadily"`, `"unsteady"`, `"steadfast"`

The overused *patterns* list (e.g., `"small mercy"`) is less affected because multi-word
phrases are less likely to appear as substrings, but single-word entries are heavily
impacted.

**Why it matters:**
This function runs after every chapter and triggers a vocabulary fix-up LLM call when
violations are found. False positives waste LLM calls and may degrade prose quality by
replacing words that were used correctly. No tests exist for this function, so the bug
has gone undetected.

**Recommended fix:**
Replace `text_lower.count(word.lower())` with a regex word-boundary match. For
performance, compile a single alternation regex for all words in each tier:

```python
import re

# Build once at module level:
_FORBIDDEN_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in _FORBIDDEN_WORDS) + r')\b',
    re.IGNORECASE,
)

# In scan_vocabulary_overuse():
for match in _FORBIDDEN_RE.finditer(chapter_text):
    word = match.group().lower()
    # count occurrences per word...
```

This also improves performance from O(n * m) substring scans to a single regex pass.

**Tests to add:**
- Test that `"auditor"` does NOT trigger the `"audit"` ban
- Test that `"sledgehammer"` does NOT trigger the `"ledger"` ban
- Test that exact matches DO trigger (e.g., `"delve into"`)
- Test soft-limit threshold behavior (1 occurrence OK, 2+ flagged)
- Test overused patterns match exactly

---

### 2. ~~Missing HTTP Security Headers~~ DONE

**Files:** `novelforge/__init__.py` (add after line 146, in the `@app.after_request` section)

**Problem:**
The application does not set standard security headers on HTTP responses:

- `X-Content-Type-Options: nosniff` — prevents MIME-type sniffing
- `X-Frame-Options: DENY` — prevents clickjacking via iframe embedding
- `Referrer-Policy: strict-origin-when-cross-origin` — limits referrer leakage
- `Content-Security-Policy` — restricts script/style sources
- `Permissions-Policy` — limits browser feature access (camera, mic, geolocation)

**Why it matters:**
Without these headers, the application is vulnerable to clickjacking, MIME confusion
attacks, and has no defense-in-depth against XSS even though the code already escapes
output. These are baseline expectations for any web application.

**Recommended fix:**
Add a security headers handler in `create_app()` alongside the existing
`set_csrf_cookie` after_request handler:

```python
@app.after_request
def set_security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # CSP should allowlist the CDN sources already used in index.html:
    # Bootstrap CSS/JS, Bootstrap Icons, jQuery, Mermaid
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net https://code.jquery.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data:; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "connect-src 'self'"
    )
    return response
```

Note: The CSP `script-src` must include `'unsafe-inline'` if the
`window._savedSessionData` inline script block (index.html line 549) is kept. To
remove the need for `'unsafe-inline'`, move that data injection to a `data-*`
attribute or a separate endpoint.

---

### 3. ~~Mermaid.js Loaded from CDN Without Subresource Integrity (SRI)~~ DONE

**Files:** `templates/index.html` line 554

**Problem:**
The Mermaid.js library is loaded from a CDN with a floating version tag and no
integrity hash:

```html
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
```

In contrast, Bootstrap CSS (line 10), Bootstrap Icons (line 14), jQuery (line 541),
and Bootstrap JS (line 542) all have pinned versions and SRI hashes.

**Why it matters:**
If the jsdelivr CDN is compromised or the `@11` tag is hijacked to point to a
malicious version, arbitrary JavaScript executes in the user's browser with full
access to session data, CSRF tokens, and generated novel content.

**Recommended fix:**
Pin to an exact version and add an SRI hash:

```html
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"
  integrity="sha384-[compute-hash]"
  crossorigin="anonymous"></script>
```

Generate the hash with:
```bash
curl -s https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js | \
  openssl dgst -sha384 -binary | openssl base64 -A
```

---

### 4. ~~No Timeout on Chapter Revision Pipeline~~ DONE

**Files:** `novelforge/routes/generation/revision.py` lines 155-166

**Problem:**
The `revise_chapter()` endpoint calls `_run_all_chapter_agents()` without a `deadline`
parameter:

```python
revised_text, revised_summary = _run_all_chapter_agents(
    text=revised_text, chapter_num=chapter_number,
    title=title, genre=genre, total_chapters=total_chapters,
    chapter_outline_summary=chapter_outline_summary,
    characters_text=characters_text, previous_summaries=previous_summaries,
    ctx=ch_ctx, step_callback=None,
    # <-- No deadline parameter!
)
```

The generation pipeline in `chapters.py` sets `deadline=time.monotonic() + PER_CHAPTER_TIMEOUT`
for every chapter, but the revision path omits this.

**Why it matters:**
A hung LLM call during revision blocks the Flask request thread indefinitely. Unlike
the generation pipeline (which runs in a background thread), revision runs in the
request thread. A single hung revision could exhaust the worker pool.

**Recommended fix:**
Add a deadline parameter:

```python
import time
from novelforge.llm.client import PER_CHAPTER_TIMEOUT

revised_text, revised_summary = _run_all_chapter_agents(
    ...,
    deadline=time.monotonic() + PER_CHAPTER_TIMEOUT,
)
```

Also wrap the entire revision in a try/except for `ChapterTimeoutError` and return
an appropriate HTTP error response (504 Gateway Timeout).

---

### 5. ~~CLAUDE.md Documentation Out of Date~~ DONE

**Files:** `CLAUDE.md` lines 30-46 (Package Structure section)

**Problem:**
The package structure documentation still shows the pre-refactoring flat file layout:

```
├── agents/
│   ├── base.py           # BaseAgent ABC
│   ├── planning.py       # 8 planning agent subclasses + module-level wrappers
│   └── chapter.py        # Chapter pipeline, prompt builders, ChapterContext dataclass
```

The actual structure after refactoring is:

```
├── agents/
│   ├── base.py
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── _helpers.py
│   │   ├── story_architecture.py
│   │   ├── master_timeline.py
│   │   ├── character_fate.py
│   │   ├── character_arc.py
│   │   ├── antagonist_motivation.py
│   │   ├── technology_rules.py
│   │   ├── theme_reinforcement.py
│   │   └── pov_focal.py
│   └── chapter/
│       ├── __init__.py
│       ├── _helpers.py
│       ├── context.py
│       ├── prompts.py
│       └── pipeline.py
```

Similarly, `routes/generation.py` is now `routes/generation/` with `__init__.py`,
`_shared.py`, `chapters.py`, `revision.py`, and `audits.py`. The `services/planning.py`
module is also not shown in the package structure.

**Why it matters:**
CLAUDE.md is loaded into the context of every AI coding session. Outdated structure
documentation causes AI agents to reference nonexistent files or create new files in
the wrong locations.

**Recommended fix:**
Update the Package Structure section to reflect the current layout. Also update:
- The "Per-Chapter Agent Pipeline" section if pipeline steps have changed
- The "Routes" table if any routes moved
- The "Testing" section if test count has changed (currently says "200 tests across 6 files")

---

## Medium Priority

### 6. ~~Shallow Copy of `chapters_done` in Revision Endpoint~~ DONE

**Files:** `novelforge/routes/generation/revision.py` line 65

**Problem:**
```python
chapters_done = list(progress_data.get("chapters_done", []))
```

`list()` creates a shallow copy — a new list containing the same dict references.
When the code later does:

```python
chapters_done[target_idx]["content"] = revised_text   # line 168
chapters_done[target_idx]["summary"] = revised_summary  # line 169
```

...it mutates the dict objects that are still referenced by the progress manager's
internal store. Any concurrent call to `progress_manager.get(token)` will see
partially-updated chapters.

**Why it matters:**
The progress `/progress/<token>/full` endpoint could return a chapter with revised
content but the old summary (or vice versa) if polled during a revision.

**Recommended fix:**
Use `copy.deepcopy()`:

```python
import copy
chapters_done = copy.deepcopy(progress_data.get("chapters_done", []))
```

The same pattern appears in `progress.py:110-113` where `get()` returns a shallow
dict copy. Consider adding a deep copy option to `ProgressManager.get()`.

---

### 7. ~~Session Lock Registry Grows Indefinitely (Memory Leak)~~ DONE

**Files:** `novelforge/session/persistence.py` lines 52-65

**Problem:**
```python
_session_locks: dict[str, threading.Lock] = {}
_session_locks_lock = threading.Lock()

def _get_session_lock(session_id: str) -> threading.Lock:
    if session_id not in _session_locks:
        with _session_locks_lock:
            if session_id not in _session_locks:
                _session_locks[session_id] = threading.Lock()
    return _session_locks[session_id]
```

Locks are created for every session ID but never removed, even after sessions are
deleted via `/delete_session`. In a long-running server with many sessions over time,
this dict grows without bound.

**Why it matters:**
Each `threading.Lock` object is small (~50 bytes), but over months of operation with
thousands of sessions, the accumulated memory is non-trivial. More importantly, it
indicates a resource lifecycle gap — if locks aren't cleaned up, other per-session
resources might not be either.

**Recommended fix:**
Option A: Remove locks from `_session_locks` when a session is deleted in
`delete_session()` route handler.

Option B: Use a bounded LRU cache or `weakref.WeakValueDictionary` so locks are
garbage collected when no thread holds a reference.

Option C: Periodically prune locks for session IDs whose JSON files no longer exist.

---

### 8. ~~Race Condition in Duplicate Generation Guard~~ DONE

**Files:** `novelforge/routes/generation/chapters.py` lines 68-76

**Problem:**
```python
existing_token = session.get("progress_token")
if existing_token:
    existing = progress_manager.get(existing_token)
    if existing and existing.get("status") == "running":
        return jsonify({...}), 409
# <-- Window: another request could pass the same check here
token = str(uuid.uuid4())
progress_manager.create(token, {...})
```

The check and creation are not atomic. Two simultaneous POST requests to
`/generate_chapters` could both pass the duplicate check and start two background
generation threads for the same session.

**Why it matters:**
Two concurrent generation workers writing to the same progress token and session
file would corrupt state, produce duplicate chapters, and waste LLM API calls.

**Recommended fix:**
Use an atomic check-and-set pattern in `progress_manager`:

```python
# In ProgressManager:
def create_if_not_running(self, session_key: str, token: str, initial_state: dict) -> bool:
    """Returns True if created, False if a running entry already exists for session_key."""
    with self._lock:
        for existing in self._store.values():
            if existing.get("snapshot", {}).get("session_id") == session_key \
               and existing.get("status") == "running":
                return False
        self._store[token] = dict(initial_state)
        return True
```

Alternatively, use the Flask session's `progress_token` as a lock key and check
status inside `progress_manager` with the lock held.

---

### 9. ~~Empty First-Person Perspective Name Edge Case~~ DONE

**Files:** `novelforge/routes/outline.py` lines 230-234,
`novelforge/agents/chapter/context.py` lines 40-50

**Problem:**
If `narrative_perspective` is set to `"first_person:"` (colon present but no character
name after it), the extraction logic produces an empty string:

```python
# In outline.py approve_outline():
if cur_persp.startswith("first_person:"):
    cur_pov_name = cur_persp[len("first_person:"):].strip()
    # cur_pov_name is "" — empty string
    if cur_pov_name in rename_map:  # "" not in rename_map, so skipped
        ...
```

The invalid perspective `"first_person:"` persists in the session and is later used
in `build_perspective_prompt()`:

```python
# In context.py:
if narrative_perspective.startswith("first_person:"):
    pov_name = narrative_perspective[len("first_person:"):].strip()
    # pov_name is "" — produces broken prompt:
    # "Write this chapter in FIRST PERSON narration from the perspective of ."
```

**Why it matters:**
The resulting prompt contains grammatically broken instructions like "from the
perspective of ." and "Use I/me/my pronouns for .", which confuses the LLM and
degrades output quality.

**Recommended fix:**
Validate the extracted name is non-empty in both locations:

```python
# In context.py build_perspective_prompt():
if narrative_perspective.startswith("first_person:"):
    pov_name = narrative_perspective[len("first_person:"):].strip()
    if not pov_name:
        # Fall back to third person if name is missing
        return build_perspective_prompt("third_person")
```

```python
# In outline.py approve_outline():
if cur_persp.startswith("first_person:"):
    cur_pov_name = cur_persp[len("first_person:"):].strip()
    if not cur_pov_name:
        working["narrative_perspective"] = "third_person"
    elif cur_pov_name in rename_map:
        working["narrative_perspective"] = f"first_person:{rename_map[cur_pov_name]}"
```

---

### 10. ~~Silent Chapter Outline Lookup Failure in Revision~~ DONE

**Files:** `novelforge/routes/generation/revision.py` lines 92-99

**Problem:**
```python
chapter_outline_summary = ""
for chapter_outline in chapter_list:
    try:
        if int(chapter_outline.get("number", 0)) == chapter_number:
            chapter_outline_summary = chapter_outline.get("summary", "")
            break
    except (TypeError, ValueError):
        continue
```

If no chapter outline matches (e.g., chapter_list is empty, or the chapter was
renumbered), `chapter_outline_summary` remains an empty string. No warning is logged
and no error is returned to the user. The revision proceeds with no outline context,
producing a revision prompt that lacks the chapter's intended summary.

**Why it matters:**
The revision prompt uses the outline summary to ground the LLM's edits. Without it,
the LLM has no guidance on what the chapter should accomplish, leading to revisions
that may drift from the story plan.

**Recommended fix:**
Log a warning when the outline is not found:

```python
if not chapter_outline_summary:
    logger.warning(
        "Chapter %d outline summary not found in snapshot chapter_list (token=%s)",
        chapter_number, token,
    )
```

Optionally, fall back to the chapter's existing summary from `chapters_done`:

```python
if not chapter_outline_summary:
    chapter_outline_summary = target_chapter.get("summary", "")
```

---

### 11. ~~Inconsistent Audit Fallback Structures~~ DONE

**Files:** `novelforge/routes/generation/audits.py` (multiple locations)

**Problem:**
Each post-manuscript audit has a hand-written fallback dict for JSON parse failures.
These fallback dicts are not validated against what `routes/export.py` expects when
rendering editor's notes. Examples:

- Global continuity audit fallback (line 80) includes `"overall_integrity"` but export
  may not check for it
- Character resolution fallback (line 117) includes `"resolution_integrity"` which may
  not be referenced
- Thematic payoff fallback (line 138) includes `"thematic_integrity"` — same concern

If a key expected by export.py is missing from the fallback dict, the export silently
produces incomplete output (`.get()` returns `None`).

**Why it matters:**
When an audit LLM call fails (network error, parse failure), the fallback should
produce output that export.py can safely consume. Currently there's no automated check
that fallback structures match consumer expectations.

**Recommended fix:**
Option A: Define fallback structures as module-level constants (or a shared dict) and
reference them from both `audits.py` and `export.py`.

Option B: Add integration tests that verify export.py can render editor's notes when
all audits return their fallback values.

Option C: Use TypedDict or dataclasses for audit results so the structure is enforced
at the type level.

---

### 12. ~~`_call_single_provider()` Is 186 Lines with Deep Nesting~~ DONE

**Files:** `novelforge/llm/client.py` lines 300-486

**Problem:**
This function handles: request construction, header assembly, logging, HTTP call,
response parsing, retry with exponential backoff, error classification (timeout, auth,
rate limit, content rejection, circuit breaker), and token usage tracking — all in a
single function with 4+ levels of nesting.

The retry loop also lacks jitter, meaning multiple workers retrying simultaneously
after a transient failure will all retry at the same instant (thundering herd).

**Why it matters:**
- Hard to test individual error paths in isolation
- Hard to add new error handling (e.g., new provider error codes) without risking
  regressions
- The thundering herd on retry can worsen rate-limit issues

**Recommended fix:**
Extract into smaller functions:
- `_build_request(provider, messages, json_mode)` -> request dict
- `_send_request(provider, request_dict)` -> response or raise
- `_parse_response(response)` -> content string
- `_classify_error(exc, provider)` -> raise typed exception

Add jitter to retry delay:
```python
import random
delay = config.LLM_RETRY_DELAY * attempt * (0.5 + random.random())
```

---

### 13. ~~Overly Broad Exception Handling in Generation Worker~~ DONE

**Files:** `novelforge/routes/generation/chapters.py` line 503

**Problem:**
```python
except (RuntimeError, requests.exceptions.RequestException,
        json.JSONDecodeError, KeyError, ValueError) as exc:
```

This single except clause catches 5 different exception types and reports them all
to the user via `friendly_llm_error(exc)`. A `KeyError` from a programming bug
(e.g., accessing `snap["nonexistent_key"]`) is indistinguishable from an LLM
communication error.

**Why it matters:**
Programming errors are silently reported as LLM failures, making them hard to
diagnose. Users see "LLM request failed" when the real issue is a missing dict key
in the application code.

**Recommended fix:**
Handle each exception type separately:

```python
except (RuntimeError, requests.exceptions.RequestException) as exc:
    # LLM communication error
    ...
except json.JSONDecodeError as exc:
    # LLM returned unparseable response
    ...
except (KeyError, ValueError) as exc:
    # Application error — log at ERROR level with traceback
    logger.error("Internal error during generation: %s", exc, exc_info=True)
    ...
```

---

## Low Priority / Quality

### 14. ~~Genre Lists Duplicated in Three Places~~ DONE

**Files:**
- `novelforge/validation.py` lines 10-32 (`ALLOWED_GENRES` set)
- `novelforge/names.py` lines 20-48 (`_GENRE_GROUP` dict)
- `novelforge/voice.py` lines 232-254 (`weights` dict in `select_voice_seed()`)

**Problem:**
All three files independently enumerate the supported genres. Adding a new genre
(e.g., "Cli-Fi") requires changes in all three locations. If one is missed, the
genre passes validation but has no name pool or voice seed mapping.

**Why it matters:**
Maintenance burden and risk of inconsistency. A genre that validates but has no
name pool falls back to a generic pool, which may produce culturally inappropriate
names for the genre.

**Recommended fix:**
Define a single authoritative genre list in `config.py` or a new `genres.py` module.
Have `validation.py`, `names.py`, and `voice.py` all import from that source. Include
a startup assertion that all genres in the authoritative list have entries in the name
pool and voice seed mappings.

---

### 15. No Cleanup of Export Files, Progress Files, or Old Sessions — OPEN

**Files:**
- `novelforge/routes/export.py` lines 118-124 (export .md files)
- `novelforge/routes/generation/chapters.py` lines 205-254 (progress JSON files)
- `novelforge/session/persistence.py` (session JSON files)
- `novelforge/routes/export.py` lines 530+ (illustration images)

**Problem:**
All generated files accumulate indefinitely on disk:
- Export `.md` files in `EXPORT_DIR`
- Progress `{token}_progress.json` files in `NOVELS_DIR`
- Session `{uuid}.json` files in `NOVELS_DIR`
- Illustration `.png` files in `EXPORT_DIR/illustrations/`

No cleanup, TTL, or disk space monitoring exists.

**Why it matters:**
Over months of operation, disk usage grows without bound. Progress JSON files are
particularly problematic because they contain full chapter content and are written
frequently during generation.

**Recommended fix:**
Option A: Add a cleanup function called on application startup that removes files
older than a configurable TTL (e.g., 7 days for progress files, 30 days for exports).

Option B: Add a `/admin/cleanup` endpoint (authenticated) that purges old files.

Option C: Delete progress files when generation completes successfully (they're only
needed for crash recovery).

---

### 16. Prompts YAML Cache Not Thread-Safe — OPEN

**Files:** `novelforge/llm/prompts.py` lines 56-68

**Problem:**
```python
_prompts_cache: dict | None = None

def _load_prompts() -> dict:
    global _prompts_cache
    if _prompts_cache is not None:
        return _prompts_cache
    # ... load from disk ...
    _prompts_cache = data
    return data
```

No lock protects `_prompts_cache`. Two threads calling `_load_prompts()` simultaneously
before the cache is populated could both read the YAML file and both assign to the
global.

**Why it matters:**
In practice this is harmless because YAML loading is idempotent and the result is the
same dict. However, it violates the principle of thread-safe initialization and could
become a real issue if the YAML file changes at runtime or if the loading has side
effects.

**Recommended fix:**
Add a lock, matching the double-checked locking pattern used elsewhere in the codebase
(e.g., `_bootstrap_logging` in `__init__.py`):

```python
_prompts_lock = threading.Lock()

def _load_prompts() -> dict:
    global _prompts_cache
    if _prompts_cache is not None:
        return _prompts_cache
    with _prompts_lock:
        if _prompts_cache is not None:
            return _prompts_cache
        # ... load ...
        _prompts_cache = data
    return data
```

---

### 17. CSRF SSL Strict Mode Disabled in All Environments — OPEN

**Files:** `novelforge/__init__.py` line 91

**Problem:**
```python
app.config["WTF_CSRF_SSL_STRICT_MODE"] = False
```

This disables HTTPS enforcement for CSRF validation unconditionally. In production
behind HTTPS, this means a downgrade attack (stripping HTTPS) could bypass CSRF
protection.

**Why it matters:**
Low risk in practice (requires active MITM), but violates defense-in-depth. The
setting should match the deployment environment.

**Recommended fix:**
```python
app.config["WTF_CSRF_SSL_STRICT_MODE"] = not app.debug
```

Or use an environment variable:
```python
app.config["WTF_CSRF_SSL_STRICT_MODE"] = os.environ.get("CSRF_SSL_STRICT", "true").lower() == "true"
```

---

### 18. Dependency Versions Have No Upper Bounds — OPEN

**Files:** `requirements.txt`

**Problem:**
All dependency pins use `>=` with no upper bound:
```
Flask>=3.0.0
flask-limiter>=3.5.0
requests>=2.31.0
PyYAML>=6.0.0
```

**Why it matters:**
A future major version release (Flask 4.0, requests 3.0) could introduce breaking
changes that are automatically pulled in by `pip install`. This creates unpredictable
deployment failures.

**Recommended fix:**
Add upper bounds for major versions:
```
Flask>=3.0.0,<4.0.0
flask-limiter>=3.5.0,<4.0.0
flask-wtf>=1.2.0,<2.0.0
flask-session>=0.8.0,<1.0.0
cachelib>=0.9.0,<1.0.0
requests>=2.31.0,<3.0.0
python-dotenv>=1.0.0,<2.0.0
unidecode>=1.3.0,<2.0.0
PyYAML>=6.0.0,<7.0.0
```

Or use a lock file (`pip freeze > requirements.lock`) for reproducible deployments.

---

### 19. Image Download URL Not Validated — OPEN

**Files:** `novelforge/llm/image.py` line 103

**Problem:**
```python
img_response = requests.get(image_data["url"], timeout=60)
```

The URL returned by the image generation API is fetched without validation of:
- URL scheme (could be `file://`, `ftp://`, or other non-HTTPS schemes)
- Domain (could point to internal services, localhost, or metadata endpoints)
- Response size (could be a multi-GB file that exhausts memory)

**Why it matters:**
Server-Side Request Forgery (SSRF): if the image API returns a malicious URL pointing
to an internal service (e.g., `http://169.254.169.254/latest/meta-data/` on AWS), the
application would fetch and potentially expose internal data.

**Recommended fix:**
```python
from urllib.parse import urlparse

parsed = urlparse(image_data["url"])
if parsed.scheme not in ("https",):
    raise ValueError(f"Refusing to download image from non-HTTPS URL: {parsed.scheme}")
# Optionally: validate domain against an allowlist
```

Also add a `stream=True` + size limit to prevent memory exhaustion:
```python
img_response = requests.get(url, timeout=60, stream=True)
content = img_response.iter_content(chunk_size=8192)
# Read up to 10MB max
```

---

### 20. Progress Manager `get()` Returns Shallow Copy — OPEN

**Files:** `novelforge/progress.py` lines 110-113

**Problem:**
```python
def get(self, token: str) -> ProgressState | None:
    with self._lock:
        data = self._store.get(token)
        return dict(data) if data is not None else None
```

`dict(data)` creates a shallow copy. Nested structures like `chapters_done` (a list
of dicts), `snapshot` (a dict), and `character_state_log` (a list) are shared
references between the returned copy and the internal store.

**Why it matters:**
Any caller that modifies nested data in the returned dict silently corrupts the
progress manager's internal state. This is the root cause of issue #6 (shallow copy
in revision endpoint) and could cause subtle bugs elsewhere.

**Recommended fix:**
Option A: Use `copy.deepcopy(data)` in `get()`. This is safe but adds overhead for
large progress entries (full chapter content).

Option B: Document that the returned dict must not be mutated, and add a
`get_deep_copy()` method for callers that need to modify the result.

Option C: Return a frozen/immutable view (e.g., `types.MappingProxyType`) so
mutations raise at the call site.


### 21. Prompt Updates — OPEN

Each of the following prompts should be added to the per chapter workflows
    
    Each POV character should think in a distinct internal language shaped by their background and expertise. Currently all characters share the same literary-metaphorical register.
        This should step should take place after the first prose refinement step in the chapter processing.
        For this item, first generate a list of each character who has dialog in the chapter.
        Then have the dialog rewritten to match their character, history, point in the novel, etc.  For instance, a 16 year old character should sound like a 16 year old character.  etc.
        The dialog is also too clean, too functional, too information-delivery. The characters shouldn't always have complete thoughts, speak in well-formed sentences.  They should sometimes interrupt each other or themselves.  Make sre the dialog flows in this manner.
        Make sure the prose maintains a 'breathes' -- denser in reflective moments, sparser in action, occasionally raw or clumsy when characters are overwhelmed.

    The novel is relentlessly serious. Every scene is morally weighted. Every observation is significant. Real humans deflect, joke badly, notice irrelevant things, and occasionally do something that doesn't serve the plot.  create a plan to inject these odities throughout the novel.  no more than 2 'oddities' per chapter.

    The manuscript is metaphor heavy.  make sure the text doesn't go overboard with metaphors.  create a plan to remove uneeded ones.
    
    Please do a light copy-edit pass targeting prose repetitions.  Also remove unneeded dashes, em-dashes and hyphens.